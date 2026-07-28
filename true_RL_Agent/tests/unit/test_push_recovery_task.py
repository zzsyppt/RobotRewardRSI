from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_EVALUATION_BINDING,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
    valid_accepted_ledger_record,
)
from tests.fakes.evaluation import (
    TEST_CAPABILITY_PROFILE,
    TEST_PROGRAM_SOURCE_CHECKSUM,
    TEST_SUBJECT_MANIFEST_HASH,
    make_evaluation_binding,
    make_training_binding,
    make_verified_candidate_artifact,
)
from true_rl_agent.agent_core.decision import DecisionOutcome, decide_candidate
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidatePackageRef,
    CandidateSurfaceAssessment,
)
from true_rl_agent.candidates.verifier import (
    CandidateVerifier,
    StaticCandidateVerifier,
    VerificationResult,
    seal_and_verify_candidate,
)
from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.evaluation.acceptance_replay import project_public_history
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.sealed_validation import (
    build_sealed_rollout_evidence,
    build_sealed_validation_gate,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.task_sections import MAX_TASK_PROMPT_SECTIONS_BYTES
from true_rl_agent.runner.base import Runner, RunnerExecutionError, TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.dry_run import DryRunCandidateMetadata
from true_rl_agent.runner.eval_isolation import build_candidate_isolation_gate
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec
from true_rl_agent.tasks.compatibility import check_task_backend_compatibility
from true_rl_agent.tasks.push_recovery import (
    PushRecoveryTaskConfig,
    PushRecoveryTaskPlugin,
    PushTaskMemoryAdapter,
    build_push_public_result,
    build_push_public_result_from_summaries,
    classify_push_failure,
)
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.dry_run_eval import (
    build_push_dry_run_public_result,
    build_push_dry_run_train_summary,
)
from true_rl_agent.tasks.push_recovery.isaac_config import (
    load_runtime_config,
    public_runtime_dynamics,
)
from true_rl_agent.tasks.push_recovery.metrics import extract_push_failure_breakdown


class PassingVerifier(CandidateVerifier):
    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        if Path(candidate.path).is_dir():
            return StaticCandidateVerifier().verify(candidate)
        return VerificationResult(status="pass", reason="test")


def _project_public_records(records):
    raw_records = [
        _bind_record_identity(
            record.ledger_public_record()
            if isinstance(record, ExperimentRecord)
            else record,
            binding=TEST_EVALUATION_BINDING,
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        )
        for record in records
    ]
    return project_public_history(
        raw_records,
        initial_current_best=make_bootstrap_current_best(),
        expected_session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
    ).records


def _bind_record_identity(
    record: dict[str, object],
    *,
    binding,
    session_genesis_hash: str,
) -> dict[str, object]:
    bound = copy.deepcopy(record)
    bound["session_genesis_hash"] = session_genesis_hash
    bound["evaluation_binding"] = binding.to_public_dict()
    public_result = bound.get("public_result")
    if isinstance(public_result, dict) and public_result:
        metrics = public_result.setdefault("metrics", {})
        if isinstance(metrics, dict):
            metrics.setdefault("evaluation_binding", binding.to_public_dict())
        jobs = bound.setdefault("jobs", {})
        if isinstance(jobs, dict):
            eval_job = jobs.setdefault("eval", {})
            if isinstance(eval_job, dict):
                eval_job.setdefault("evaluation_binding", binding.to_public_dict())
    return bound


class PushDryRunRunner(Runner):
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        subject = train_spec.subject
        raw_checkpoint = CheckpointRef(
            uri=f"artifact://{train_spec.job_id}/best.pt",
            checksum=(
                "sha256:"
                + hashlib.sha256(
                    subject.subject_id.encode("utf-8")
                ).hexdigest()
            ),
            role="train_output",
            metadata={
                "local_path": f"outputs/runner/{train_spec.job_id}/checkpoints/best.pt",
                "training_subject": subject.to_public_dict(),
                "subject_manifest_hash": train_spec.subject_manifest_hash,
                "program_source_checksum": train_spec.program_source_checksum,
                "training_binding_hash": train_spec.training_binding.binding_hash,
                "capability_profile_hash": (
                    train_spec.training_binding.capability_profile_hash
                ),
            },
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
            subject=subject,
            subject_manifest_hash=train_spec.subject_manifest_hash,
            program_source_checksum=train_spec.program_source_checksum,
            training_binding_hash=train_spec.training_binding.binding_hash,
            capability_profile_hash=(
                train_spec.training_binding.capability_profile_hash
            ),
            checkpoint=seal_checkpoint(raw_checkpoint, sealed_uri=f"artifact://{train_spec.job_id}/sealed/best.pt"),
            summary={
                "formal_training_executed": False,
                "budget_steps": train_spec.budget_steps,
                "training": {
                    "budget_accounting": {
                        "status": "pass",
                        "budget_steps": train_spec.budget_steps,
                        "planned_env_steps": train_spec.budget_steps,
                    }
                },
                "runtime_activation_summary": {
                    "reward_term_calls": 4,
                    "diagnostic_calls": 1,
                    "nonzero_reward_contribution": True,
                    "reward_output_stats": {"mean": 0.4, "finite_fraction": 1.0},
                    "diagnostic_outputs": {"push_recovery_margin": 0.2},
                    "feature_activation_records": [],
                    "capability_activation_records": ["reward_callable"],
                },
            },
        )

    def run_eval(self, eval_spec: EvalJobSpec):
        result = build_push_public_result(
            success_rate=0.30,
            failure_breakdown={"fall": 0.1, "timeout": 0.05},
            motion_quality={"mean_base_height": 0.78, "mean_tilt_rad": 0.05},
        )
        metrics = dict(result.metrics)
        public_dev_count = max(1, int(eval_spec.public_seed_count))
        public_accept_count = max(1, int(eval_spec.public_accept_seed_count))
        public_dev_stats = {
            "count": public_dev_count,
            "mean": 0.30,
            "std": 0.0,
            "standard_error": 0.0,
            "values": [0.30] * public_dev_count,
        }
        public_accept_stats = {
            "count": public_accept_count,
            "mean": 0.30,
            "std": 0.0,
            "standard_error": 0.0,
            "values": [0.30] * public_accept_count,
        }
        metrics["public_dev"] = {
            "score_stats": public_dev_stats,
            "seed_results": [
                {"run_index": index, "status": "pass", "success_rate": 0.30}
                for index in range(1, public_dev_count + 1)
            ],
        }
        metrics["public_accept"] = {
            "score_stats": public_accept_stats,
            "seed_results": [
                {"run_index": index, "status": "pass", "success_rate": 0.30}
                for index in range(1, public_accept_count + 1)
            ],
        }
        metrics["accept_evidence_gate"] = {
            "status": "pass",
            "observed_count": public_accept_count,
            "required_count": public_accept_count,
        }
        metrics["budget_accounting"] = {"status": "pass"}
        metrics["motion_quality_gate"] = {"status": "pass"}
        metrics["behavior_quality_gate"] = {"status": "pass", "reason": "dry_run_behavior_gate"}
        eval_protocol = eval_spec.evaluation_protocol()
        sealed_evidence = build_sealed_rollout_evidence(
            score_stats={
                "count": int(eval_spec.sealed_validation_seed_count),
                "mean": 0.30,
                "std": 0.0,
                "standard_error": 0.0,
                "eval_protocol_hash": eval_protocol.protocol_hash,
            },
            expected_seed_count=int(eval_spec.sealed_validation_seed_count),
            eval_protocol_hash=eval_protocol.protocol_hash,
            evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            evaluation_subject=eval_spec.evaluation_subject(),
            subject_manifest_hash=eval_spec.subject_manifest_hash,
            program_source_checksum=eval_spec.program_source_checksum,
            training_binding_hash=eval_spec.training_binding_hash,
            capability_profile_hash=eval_spec.capability_profile_hash,
            checkpoint_hash=eval_spec.checkpoint.checksum,
            task_validation_status="pass",
            reason="test_holdout_passed",
        )
        metrics["sealed_validation"] = sealed_evidence
        metrics["sealed_validation_gate"] = build_sealed_validation_gate(
            rollout_evidence=sealed_evidence,
            checkpoint_safety_status="pass",
            checkpoint_hash_match_status="pass",
            candidate_isolation_gate=build_candidate_isolation_gate(),
            expected_eval_protocol_hash=eval_protocol.protocol_hash,
            expected_evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
        )
        acceptance = dict(metrics["acceptance"])
        acceptance["score_source"] = "public_accept.score_stats.mean"
        acceptance["score_stats"] = dict(public_accept_stats)
        acceptance["min_score_count"] = public_accept_count
        acceptance["missing_required_score"] = False
        acceptance["required_pass_gates"] = [
            "accept_evidence_gate",
            "budget_accounting",
            "motion_quality_gate",
            "behavior_quality_gate",
            "sealed_validation_gate",
        ]
        metrics["acceptance"] = acceptance
        return PublicEvalResult(status=result.status, public_score=result.public_score, metrics=metrics)


class CapturingPushDryRunRunner(PushDryRunRunner):
    def __init__(self) -> None:
        self.eval_specs: list[EvalJobSpec] = []

    def run_eval(self, eval_spec: EvalJobSpec):
        self.eval_specs.append(eval_spec)
        return super().run_eval(eval_spec)


class EvalIncompletePushDryRunRunner(PushDryRunRunner):
    def run_eval(self, eval_spec: EvalJobSpec):
        _ = eval_spec
        raise RunnerExecutionError("eval summary file was not created: D:/runner/eval/summary.json")


class UnsafeEvalProjectionPushDryRunRunner(PushDryRunRunner):
    def run_eval(self, eval_spec: EvalJobSpec):
        _ = eval_spec
        return PublicEvalResult(
            status="pass",
            public_score=1.0,
            metrics={
                "success_rate": 1.0,
                "summary_path": "D:/runner/eval/summary.json",
                "acceptance": {
                    "score_source": "probe",
                    "score_stats": {"count": 1, "mean": 1.0},
                    "required_pass_gates": [],
                    "absolute_accept_floor": 0.05,
                },
            },
        )


class PushRecoveryTaskTests(unittest.TestCase):
    def _round_session_fields(
        self,
        *,
        plugin: PushRecoveryTaskPlugin,
        backend: IsaacLabRslRlBackendAdapter,
        initial_best_score: float,
    ) -> dict[str, object]:
        binding = backend.evaluation_binding_for(plugin.task_spec)
        return {
            "current_best_eval_stats": CurrentBestEvalStats.bootstrap(
                initial_best_score=initial_best_score,
                acceptance_protocol=plugin.task_spec.acceptance_protocol,
                evaluation_binding=binding,
                anchor_id="push-test-bootstrap",
            ),
            "session_genesis_hash": "9" * 64,
            "evaluation_binding": binding,
        }

    def _assert_private_seed_ids_absent(
        self,
        payload: object,
        *seed_ids: int,
    ) -> None:
        forbidden_numbers = frozenset(seed_ids)
        forbidden_text = frozenset(str(seed_id) for seed_id in seed_ids)

        def visit(value: object, path: str) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if str(key) in forbidden_text:
                        self.fail(f"private seed ID leaked at {path}.{key}")
                    visit(item, f"{path}.{key}")
                return
            if isinstance(value, list | tuple):
                for index, item in enumerate(value):
                    visit(item, f"{path}[{index}]")
                return
            if isinstance(value, bool) or value is None:
                return
            if isinstance(value, int | float) and value in forbidden_numbers:
                self.fail(f"private seed ID leaked at {path}")
            if isinstance(value, str) and value in forbidden_text:
                self.fail(f"private seed ID leaked at {path}")

        visit(payload, "public_result")

    def test_private_seed_assertion_is_structural_not_substring_based(self) -> None:
        self._assert_private_seed_ids_absent(
            {
                "created_at_utc": "2026-07-27T03:30:09.112001+00:00",
                "opaque_hash": "sha256:abc2001def",
            },
            2001,
        )
        with self.assertRaisesRegex(AssertionError, "private seed ID leaked"):
            self._assert_private_seed_ids_absent(
                {"nested": [{"seed": 2001}]},
                2001,
            )

    def test_dry_run_train_summary_rejects_forged_budget_pass(self) -> None:
        candidate_artifact = make_verified_candidate_artifact("candidate-001")
        train_spec = TrainJobSpec(
            job_id="dry-budget-mismatch",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.candidate("candidate-001"),
            subject_manifest_hash=candidate_artifact.subject_manifest_hash,
            program_source_checksum=candidate_artifact.program_source_checksum,
            policy_initialization=PolicyInitialization.random(),
            training_binding=make_training_binding(
                PolicyInitialization.random(),
                budget_steps=65,
            ),
            budget_steps=65,
            metadata={
                "fixed_train_protocol": {
                    "budget_steps": 65,
                    "train_num_envs": 8,
                    "train_max_iterations": 1,
                    "train_steps_per_iteration": 8,
                }
            },
            candidate_artifact=candidate_artifact,
        )
        with self.assertRaisesRegex(ValueError, "planned env steps mismatch"):
            build_push_dry_run_train_summary(
                train_spec,
                DryRunCandidateMetadata(
                    changed_components=(),
                    public_score=0.0,
                ),
            )

    def test_push_dry_run_rejects_noncanonical_public_score(self) -> None:
        eval_spec = EvalJobSpec(
            job_id="dry-run-eval",
            task_id="unitree_push_recovery",
            subject=ExperimentSubject.candidate("dry-run-candidate"),
            subject_manifest_hash=TEST_SUBJECT_MANIFEST_HASH,
            program_source_checksum=TEST_PROGRAM_SOURCE_CHECKSUM,
            training_binding_hash="7" * 64,
            capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
            checkpoint=CheckpointRef(
                uri="artifact://dry-run/eval.pt",
                checksum="sha256:" + "d" * 64,
                role="eval",
                sealed=True,
            ),
            public_seed_count=1,
            public_accept_seed_count=3,
            sealed_validation_seed_count=1,
            evaluation_binding=make_evaluation_binding(
                public_seed_count=1,
                public_accept_seed_count=3,
                sealed_validation_seed_count=1,
            ),
        )
        for invalid_score in (True, "0.5", float("nan")):
            with self.subTest(public_score=invalid_score):
                metadata = DryRunCandidateMetadata(
                    changed_components=("reward_callable",),
                    public_score=invalid_score,
                )
                with self.assertRaisesRegex(
                    ValueError,
                    "dry-run public_score must be a finite non-boolean number",
                ):
                    build_push_dry_run_public_result(
                        eval_spec,
                        metadata,
                        acceptance_protocol=(
                            PushRecoveryTaskConfig().acceptance_protocol()
                        ),
                    )

    def _backend(self) -> IsaacLabRslRlBackendAdapter:
        provider = PushRecoveryCompositionProvider()
        values = dict(provider.cli_defaults())
        values.update(
            {
                "isaac_python": "/opt/isaac/python.sh",
                "train_entrypoint": "train_push.py",
                "eval_entrypoint": "eval_push.py",
            }
        )
        composition = provider.resolve(
            values=values,
            context=TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=PROJECT_ROOT / "outputs" / "test-push-backend",
            ),
        )
        self.assertIsInstance(
            composition.backend_adapter,
            IsaacLabRslRlBackendAdapter,
        )
        return composition.backend_adapter

    def _write_schedule_only_candidate(
        self,
        root: Path,
        *,
        candidate_id: str = "schedule_only",
    ) -> Path:
        candidate = root / candidate_id
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: {candidate_id}
                entrypoint: training_code.components:build_training_components
                """
            ).format(candidate_id=candidate_id).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            textwrap.dedent(
                """
                from true_rl_agent.candidates.components import ScheduleTerm, TrainingComponents


                def build_training_components():
                    return TrainingComponents(
                        name="schedule_only",
                        schedules=[ScheduleTerm(name="scale", key="reward_weight_scale", value=2.0)],
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def _write_reward_candidate(
        self,
        root: Path,
        *,
        candidate_id: str,
    ) -> Path:
        candidate = root / candidate_id
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                f"""
                schema_version: true_rl_agent.candidate.v1
                candidate_id: {candidate_id}
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            textwrap.dedent(
                """
                from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                def upright_reward(inp):
                    return inp.base_height


                def build_training_components():
                    return TrainingComponents(
                        name="reward_code",
                        reward_terms=[RewardTerm(name="upright", weight=1.0, fn=upright_reward)],
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def test_push_plugin_declares_fixed_invariants_and_no_backend_authority(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        task_spec = plugin.task_spec

        self.assertEqual(task_spec.task_id, "unitree_push_recovery")
        self.assertTrue(task_spec.supports_policy_initialization("random"))
        for key in (
            "success_predicate",
            "eval_seed_visibility",
            "eval_num_envs",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "eval_protocol_version",
            "eval_protocol_hash",
            "budget_steps",
            "train_runtime_profile",
            "eval_runtime_profile",
            "train_runtime_dynamics_hash",
            "eval_runtime_dynamics_hash",
            "checkpoint_selector",
            "accept_threshold",
        ):
            self.assertIn(key, task_spec.invariants.values)
        self.assertIn("baseline_train_distribution", task_spec.fixed_train_protocol)
        self.assertIn("eval_protocol", task_spec.fixed_eval_protocol)
        self.assertEqual(task_spec.invariants.values["budget_steps"], 49_152_000)
        self.assertEqual(task_spec.invariants.values["train_num_envs"], 4096)
        self.assertEqual(task_spec.invariants.values["train_max_iterations"], 500)
        self.assertEqual(task_spec.invariants.values["train_steps_per_iteration"], 24)
        self.assertEqual(task_spec.fixed_train_protocol["budget_steps"], 49_152_000)
        self.assertEqual(task_spec.fixed_train_protocol["train_num_envs"], 4096)
        self.assertEqual(task_spec.fixed_train_protocol["train_max_iterations"], 500)
        self.assertEqual(task_spec.fixed_train_protocol["train_steps_per_iteration"], 24)
        self.assertEqual(task_spec.fixed_eval_protocol["public_seed_count"], 3)
        self.assertEqual(task_spec.fixed_eval_protocol["public_accept_seed_count"], 3)
        self.assertEqual(task_spec.fixed_eval_protocol["sealed_validation_seed_count"], 7)
        self.assertEqual(task_spec.fixed_eval_protocol["eval_num_envs"], 64)
        self.assertEqual(
            task_spec.fixed_eval_protocol["eval_protocol_hash"],
            EvaluationProtocol(
                eval_protocol_version=task_spec.fixed_eval_protocol["eval_protocol_version"],
                public_seed_count=3,
                public_accept_seed_count=3,
                sealed_validation_seed_count=7,
                eval_num_envs=64,
            ).protocol_hash,
        )
        self.assertFalse(hasattr(plugin, "materialize_jobs"))

    def test_push_prompt_adapter_projects_task_specific_sections_without_runner_control(self) -> None:
        plugin = PushRecoveryTaskPlugin()

        sections = plugin.prompt_adapter().build_prompt_sections(
            public_records=[],
            public_context={"previous_rounds": 0, "public_only": True},
        )
        text = json.dumps(sections, sort_keys=True)

        self.assertEqual(sections["task"]["task_id"], "unitree_push_recovery")
        self.assertEqual(sections["task"]["prompt_adapter"], "push_recovery_task_prompt_adapter")
        self.assertIn("push", sections["task"]["objective"].lower())
        self.assertEqual(sections["task"]["fixed_train_protocol_summary"]["train_num_envs"], 4096)
        self.assertEqual(sections["task"]["fixed_train_protocol_summary"]["train_max_iterations"], 500)
        self.assertEqual(sections["task"]["fixed_train_protocol_summary"]["train_steps_per_iteration"], 24)
        runtime_dynamics = sections["task"]["runner_owned_runtime_dynamics"]
        self.assertEqual(runtime_dynamics["train"]["active_profile"], "strong_push")
        self.assertTrue(runtime_dynamics["train"]["standing_command"]["commanded_locomotion_is_zero"])
        self.assertEqual(runtime_dynamics["train"]["environment"]["episode_length_s"], 4.0)
        self.assertEqual(runtime_dynamics["train"]["standing_command"]["lin_vel_x_mps"], [0.0, 0.0])
        self.assertEqual(runtime_dynamics["train"]["standing_command"]["lin_vel_y_mps"], [0.0, 0.0])
        self.assertEqual(runtime_dynamics["train"]["standing_command"]["ang_vel_z_radps"], [0.0, 0.0])
        self.assertEqual(
            runtime_dynamics["train"]["push"]["velocity_range_mps"],
            {"x": [-0.6, 0.6], "y": [-0.6, 0.6]},
        )
        self.assertEqual(runtime_dynamics["train"]["push"]["active_world_axes"], ["x", "y"])
        self.assertTrue(runtime_dynamics["train"]["push"]["xy_ranges_equal"])
        self.assertEqual(runtime_dynamics["train"]["success_thresholds"]["max_final_xy_drift_m"], 0.8)
        self.assertTrue(runtime_dynamics["eval"]["same_as_train"])
        guidance = " ".join(runtime_dynamics["reasoning_guidance"]).lower()
        self.assertIn("no forward locomotion command", guidance)
        self.assertIn("symmetric world x and y", guidance)
        self.assertIn("isotropic xy", guidance)
        self.assertEqual(
            sections["task"]["fixed_eval_protocol_summary"]["seed_visibility"],
            "counts_only_seed_identifiers_omitted",
        )
        self.assertEqual(sections["task"]["fixed_eval_protocol_summary"]["public_accept_seed_count"], 3)
        self.assertEqual(sections["task"]["fixed_eval_protocol_summary"]["sealed_validation_seed_count"], 7)
        self.assertIn("eval_protocol_hash", sections["task"]["fixed_eval_protocol_summary"])
        self.assertEqual(sections["public_eval_contract"]["public_score"]["direction"], "higher_is_better")
        self.assertEqual(sections["public_eval_contract"]["runner_decision_authority"], "runner_owned_read_only")
        self.assertEqual(
            sections["public_eval_contract"]["schema_version"],
            "true_rl_agent.public_eval_contract.push_recovery.v2",
        )
        self.assertEqual(
            sections["public_eval_contract"]["eval_evidence_structure"]["sealed_validation_gate"][
                "prompt_visibility"
            ],
            "gate_summary_only",
        )
        self.assertIn("non_prompt_validation", sections["public_eval_contract"]["eval_evidence_structure"])
        self.assertEqual(
            sections["public_eval_contract"]["non_prompt_validation"]["checkpoint_safety_status"],
            "prompt_visible_status_only",
        )
        self.assertEqual(
            sections["public_eval_contract"]["non_prompt_validation"]["candidate_isolation_status"],
            "prompt_visible_status_only",
        )
        for locked_control in (
            "eval_seed",
            "holdout",
            "eval_profile",
            "checkpoint_policy",
            "accept_threshold",
        ):
            self.assertIn(
                locked_control,
                sections["public_eval_contract"]["candidate_must_not_request_or_modify"],
            )
        self.assertIn("success_rate", sections["public_eval_contract"]["metrics"])
        self.assertIn("accept_evidence_gate", sections["public_eval_contract"]["metrics"])
        self.assertIn("behavior_quality_gate", sections["public_eval_contract"]["metrics"])
        self.assertIn("prompt_feedback", sections["public_eval_contract"]["metrics"])
        reward_contract = sections["reward_inputs_contract"]
        self.assertIn("push_active", reward_contract["fields"])
        self.assertIn("push_vec_public", reward_contract["fields"])
        self.assertIn("push_active", reward_contract["field_order"])
        self.assertIn("push_vec_public", reward_contract["field_order"])
        self.assertEqual(reward_contract["public_push_signal_fields"], ["push_active", "push_vec_public"])
        self.assertNotIn("push_active", reward_contract["reserved_fields"])
        self.assertNotIn("push_vec_public", reward_contract["reserved_fields"])
        self.assertEqual(reward_contract["fields"]["base_height"]["shape"], ["num_envs"])
        self.assertIn("current runner step", reward_contract["fields"]["push_active"]["semantics"].lower())
        self.assertIn("not a guaranteed post-push window", reward_contract["fields"]["push_active"]["candidate_usage_guidance"].lower())
        self.assertIn("not latched", reward_contract["fields"]["push_active"]["known_limitations"].lower())
        self.assertIn("does not reveal future pushes", reward_contract["fields"]["push_vec_public"]["known_limitations"].lower())
        self.assertGreaterEqual(len(sections["known_good_examples"]), 3)
        self.assertTrue(any(example["kind"] == "RewardTerm" for example in sections["known_good_examples"]))
        self.assertTrue(any(example["kind"] == "DiagnosticTerm" for example in sections["known_good_examples"]))
        self.assertIn("candidate_must_not_change", sections["task"])
        self.assertNotIn("trusted_seed", text)
        self.assertNotIn("raw_seed", text)
        self.assertNotIn("hidden_eval", text)
        self.assertNotIn("checkpoint_selector", text)
        self.assertNotIn("stage_list", text)
        self.assertNotIn("D:/", text)
        self.assertNotIn("/data/", text)
        assert_public_only_prompt_payload(sections)

    def test_runtime_dynamics_projection_tracks_task_owned_profile(self) -> None:
        strong = PushRecoveryTaskConfig(train_profile="strong_push").train_runtime_dynamics()
        small = PushRecoveryTaskConfig(train_profile="small_push").train_runtime_dynamics()

        self.assertEqual(strong["push"]["velocity_range_mps"]["x"], [-0.6, 0.6])
        self.assertEqual(small["push"]["velocity_range_mps"]["x"], [-0.25, 0.25])
        self.assertNotEqual(strong["runtime_dynamics_hash"], small["runtime_dynamics_hash"])

    def test_runtime_dynamics_hash_binds_all_consumed_config_controls(self) -> None:
        base_config = load_runtime_config(None)
        baseline = public_runtime_dynamics(base_config)

        self.assertEqual(
            baseline["schema_version"],
            "true_rl_agent.task.push_recovery.public_runtime_dynamics.v2",
        )
        self.assertEqual(baseline["environment"]["terrain_rows"], 2)
        self.assertNotIn("disable_curriculum", baseline["environment"])
        self.assertEqual(
            baseline["standing_command"]["resampling_time_range_s"],
            [10.0, 10.0],
        )
        self.assertIn("evidence", baseline["push"])
        self.assertEqual(baseline["metrics"]["max_per_episode_records"], 24)
        self.assertEqual(baseline["policy_defaults"]["random_action_scale"], 1.0)

        mutations = (
            (("environment", "terrain_rows"), 3),
            (("environment", "disable_curriculum"), False),
            (("standing_command", "resampling_time_range_s"), [9.0, 9.0]),
            (("standing_command", "rel_heading_envs"), 0.25),
            (("standing_command", "heading_command"), True),
            (("push", "evidence", "min_observed_delta_mps"), 0.2),
            (("metrics", "max_per_episode_records"), 12),
            (("policy_defaults", "random_action_scale"), 0.5),
        )
        for path, value in mutations:
            config = copy.deepcopy(base_config)
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value):
                self.assertNotEqual(
                    public_runtime_dynamics(config)["runtime_dynamics_hash"],
                    baseline["runtime_dynamics_hash"],
                )

        invalid_values = (
            (("environment", "terrain_rows"), 2.5),
            (("environment", "disable_curriculum"), "false"),
            (("standing_command", "heading_command"), 1),
            (
                (
                    "push",
                    "evidence",
                    "min_env_fraction_with_single_step_delta",
                ),
                1.2,
            ),
            (("metrics", "max_per_episode_records"), "24"),
            (("policy_defaults", "zero_action_scale"), True),
        )
        for path, value in invalid_values:
            config = copy.deepcopy(base_config)
            target = config
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value):
                with self.assertRaisesRegex(ValueError, r"\.".join(path)):
                    public_runtime_dynamics(config)

    def test_push_metrics_reject_noncanonical_accept_counts(self) -> None:
        for invalid in (True, "3", 3.0, 0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    ValueError,
                    "min_public_accept_count",
                ):
                    build_push_public_result_from_summaries(
                        public_dev_summaries=[],
                        min_public_accept_count=invalid,
                    )

    def test_push_full_prompt_payload_uses_task_owned_reward_inputs_contract(self) -> None:
        payload = PublicContextBuilder(task_prompt_adapter=PushRecoveryTaskPlugin().prompt_adapter()).build_prompt_payload([])

        self.assertEqual(payload["task"]["task_id"], "unitree_push_recovery")
        self.assertNotIn("task_prompt_sections", payload["public_context"])
        self.assertIn("push_active", payload["reward_inputs_contract"]["fields"])
        self.assertIn("push_vec_public", payload["reward_inputs_contract"]["fields"])
        self.assertNotIn("push_active", payload["reward_inputs_contract"]["reserved_fields"])
        self.assertNotIn("push_vec_public", payload["reward_inputs_contract"]["reserved_fields"])
        self.assertNotIn("reward_inputs", payload)
        self.assertNotIn("candidate_component_contract", payload)
        self.assertEqual(payload["public_eval_contract"]["public_score"]["direction"], "higher_is_better")
        self.assertEqual(payload["runtime_feedback_summary"]["status"], "no_prior_runtime_feedback")
        guidance = " ".join(payload["selection_guidance"])
        self.assertIn("public_history.experiment_memory", guidance)
        self.assertIn("name the closest prior candidate", guidance)
        self.assertIn("push_active", guidance)
        assert_public_only_prompt_payload(payload)

    def test_push_task_allows_public_push_signal_names_without_shared_layer_leakage(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "push-round-public-signal-history",
                "task": {"task_id": "unitree_push_recovery"},
                "candidate": {"candidate_id": "public_push_signal_probe", "changed_components": ["reward_callable"]},
                "candidate_surface": {"classification": "code_level"},
                "runtime_activation_summary": {
                    "reward_term_calls": 1,
                    "diagnostic_calls": 1,
                    "nonzero_reward_contribution": False,
                    "reward_output_stats": {"mean": 0.0, "finite_fraction": 1.0},
                    "diagnostic_outputs": {"push_active": {"mean": 1.0}},
                    "capability_activation_records": ["public_push_signal diagnostic was attempted"],
                },
                "runner_error": {
                    "type": "CandidateRuntimeError",
                    "message": "candidate over-weighted public push signal without recovery behavior",
                    "details": {"status": "fail", "code": "push_signal_not_recovery_reward"},
                },
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "public_push_signal_not_sufficient",
                    "public_score": 0.0,
                },
            }
        )

        payload = PublicContextBuilder(
            task_prompt_adapter=PushRecoveryTaskPlugin().prompt_adapter()
        ).build_prompt_payload([record])
        prompt_sections = {
            "public_history": payload["public_history"],
            "public_feedback_summary": payload["public_feedback_summary"],
            "runtime_feedback_summary": payload["runtime_feedback_summary"],
        }

        for section_name, section in prompt_sections.items():
            with self.subTest(section=section_name):
                section_text = json.dumps(section, sort_keys=True)
                self.assertNotIn("trusted_seed", section_text)
                self.assertNotIn("D:/", section_text)
                self.assertNotIn("/data/", section_text)
        self.assertIn("push_active", payload["reward_inputs_contract"]["fields"])
        self.assertIn("push_vec_public", payload["reward_inputs_contract"]["fields"])
        self.assertNotIn("push_active", payload["reward_inputs_contract"]["reserved_fields"])
        self.assertNotIn("push_vec_public", payload["reward_inputs_contract"]["reserved_fields"])
        assert_public_only_prompt_payload(payload)

        direct_sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[record.ledger_public_record()],
            public_context={
                "previous_rounds": 1,
                "public_only": True,
                "task_public_summary": {"push_active": "must_not_leak"},
            },
        )
        for section_name in ("public_feedback_summary", "runtime_feedback_summary"):
            with self.subTest(direct_section=section_name):
                section_text = json.dumps(direct_sections[section_name], sort_keys=True)
                self.assertNotIn("trusted_seed", section_text)
                self.assertNotIn("D:/", section_text)
                self.assertNotIn("/data/", section_text)
        assert_public_only_prompt_payload(direct_sections)

    def test_push_prompt_adapter_summarizes_public_history_and_runtime_feedback(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "push-round-001",
                "task": {"task_id": "unitree_push_recovery"},
                "candidate": {"candidate_id": "upright_reward", "changed_components": ["reward_callable"]},
                "candidate_surface": {"classification": "code_level"},
                "runtime_activation_summary": {
                    "reward_term_calls": 4,
                    "diagnostic_calls": 1,
                    "nonzero_reward_contribution": True,
                    "reward_output_stats": {"mean": 0.4, "finite_fraction": 1.0},
                    "diagnostic_outputs": {"push_recovery_margin": 0.2},
                    "capability_activation_records": ["reward_callable"],
                },
                "runtime_activation_gate": {"status": "pass", "reason": "runner_activation_observed"},
                "mechanism_attribution": {
                    "status": "not_attributed",
                    "reason": "shadow_ablation_not_run",
                },
                "train_result": {
                    "summary": {
                        "candidate_package_loaded": True,
                        "candidate_code_loaded_in_eval": False,
                    }
                },
                "public_result": {
                    "status": "pass",
                    "public_score": 0.30,
                    "metrics": {
                        "success_rate": 0.30,
                        "failure_breakdown": {"fall": 0.20, "push_drift": 0.05},
                        "motion_quality": {"mean_base_height": 0.78, "mean_tilt_rad": 0.05},
                        "budget_accounting": {"status": "pass"},
                        "motion_quality_gate": {"status": "pass"},
                        "acceptance": {"score_source": "single_eval.success_rate"},
                    },
                },
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "score_delta_below_threshold",
                    "public_score": 0.30,
                },
            }
        )

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=_project_public_records([record]),
            public_context={"previous_rounds": 1, "public_only": True},
        )

        feedback = sections["public_feedback_summary"]
        runtime = sections["runtime_feedback_summary"]
        self.assertEqual(feedback["latest_round"]["round_id"], "push-round-001")
        self.assertEqual(
            feedback["latest_round"]["decision_reason"],
            "decision_record_invalid",
        )
        # The synthetic record is intentionally contract-invalid; its raw
        # top-level score must not be reintroduced by the task adapter.
        self.assertIsNone(feedback["latest_public_score"])
        self.assertEqual(feedback["latest_failure_mode"], "fall_dominated")
        self.assertEqual(runtime["latest_runtime_activation"]["reward_term_calls"], 4)
        self.assertTrue(runtime["latest_runtime_activation"]["nonzero_reward_contribution"])
        self.assertEqual(runtime["runtime_activation_gate"]["status"], "pass")
        self.assertEqual(runtime["mechanism_attribution"]["status"], "not_attributed")
        self.assertTrue(runtime["candidate_package_loaded"])
        self.assertFalse(runtime["candidate_code_loaded_in_eval"])
        assert_public_only_prompt_payload(sections)

    def test_push_runtime_feedback_bounds_large_diagnostic_outputs(self) -> None:
        diagnostic_outputs = {
            f"candidate_diag_{index:03d}": {
                "mean": float(index),
                "std": 0.1,
                "min": 0.0,
                "max": float(index) + 0.5,
                "finite_fraction": 1.0,
                "nonzero_fraction": 1.0,
                "extra_candidate_note": "x" * 200,
            }
            for index in range(300)
        }
        record = {
            "round_id": "push-round-large-diagnostics",
            "runtime_activation_summary": {
                "reward_term_calls": 2,
                "diagnostic_calls": 300,
                "nonzero_reward_contribution": True,
                "reward_output_stats": {"mean": 0.2, "finite_fraction": 1.0},
                "component_activation_records": [
                    {
                        "component_kind": "reward_term",
                        "name": f"reward_term_{index:03d}",
                        "observation_count": 500,
                        "nonzero_observation_count": index,
                        "nonzero_observation_fraction": index / 500,
                        "max_abs_weighted_contribution": float(index),
                        "last_weighted_mean": float(index) / 10,
                        "effective_weight": 1.0,
                        "sampling_interval_steps": 24,
                    }
                    for index in range(100)
                ],
                "diagnostic_outputs": diagnostic_outputs,
                "capability_activation_records": ["diagnostic_term"],
            },
            "decision": {"accepted": False, "outcome": "reject", "public_score": 0.1},
        }

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[record],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        outputs = sections["runtime_feedback_summary"]["latest_runtime_activation"]["diagnostic_outputs"]
        self.assertEqual(outputs["schema_version"], "true_rl_agent.runtime_diagnostic_outputs_summary.v1")
        self.assertEqual(outputs["field_count"], 300)
        self.assertGreater(outputs["omitted_field_count"], 0)
        self.assertLessEqual(outputs["shown_field_count"], 24)
        self.assertTrue(all(key.startswith("diagnostic_") for key in outputs["fields"]))
        self.assertIn("field_key_metadata", outputs)
        self.assertNotIn("candidate_diag_000", json.dumps(outputs, sort_keys=True))
        self.assertNotIn("candidate_diag_299", json.dumps(outputs, sort_keys=True))
        reward_terms = sections["runtime_feedback_summary"]["latest_runtime_activation"][
            "reward_term_activation"
        ]
        self.assertEqual(reward_terms["term_count"], 100)
        self.assertGreater(reward_terms["omitted_term_count"], 0)
        self.assertLessEqual(len(json.dumps(reward_terms).encode("utf-8")), reward_terms["byte_budget"])
        self.assertLess(len(json.dumps(sections, sort_keys=True).encode("utf-8")), 32 * 1024)
        assert_public_only_prompt_payload(sections)

    def test_push_runtime_feedback_preserves_bounded_activation_evidence(self) -> None:
        record = {
            "round_id": "push-round-runtime-evidence",
            "runtime_activation_summary": {
                "reward_term_calls": 200,
                "diagnostic_calls": 10,
                "nonzero_reward_contribution": True,
                "reward_output_stats": {"mean": 0.2, "finite_fraction": 1.0},
                "component_activation_records": [
                    {
                        "component_kind": "reward_term",
                        "name": "command_tracking",
                        "observation_count": 100,
                        "nonzero_observation_count": 0,
                        "nonzero_observation_fraction": 0.0,
                        "max_abs_weighted_contribution": 0.0,
                        "last_weighted_mean": 0.0,
                        "effective_weight": 0.3,
                        "sampling_interval_steps": 24,
                    },
                    {
                        "component_kind": "reward_term",
                        "name": "upright_margin",
                        "observation_count": 100,
                        "nonzero_observation_count": 100,
                        "nonzero_observation_fraction": 1.0,
                        "max_abs_weighted_contribution": 1.2,
                        "last_weighted_mean": 0.7,
                        "effective_weight": 1.0,
                        "sampling_interval_steps": 24,
                    },
                ],
                "diagnostic_outputs": {
                    "drift_diagnostic": {
                        "name": "drift_diagnostic",
                        "num_envs": 4096,
                        "fields": {
                            "mean_drift_speed": {
                                "finite": True,
                                "mean": 0.314,
                                "shape": [],
                            }
                        },
                    }
                },
                "capability_activation_records": ["reward_callable", "diagnostic"],
            },
            "decision": {"accepted": False, "outcome": "reject", "public_score": 0.1},
        }

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[record],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        activation = sections["runtime_feedback_summary"]["latest_runtime_activation"]
        reward_terms = activation["reward_term_activation"]
        self.assertEqual(reward_terms["observation_scope"], "bounded_periodic_samples_over_training_steps")
        self.assertEqual(reward_terms["term_count"], 2)
        self.assertEqual(reward_terms["terms"][0]["public_label"], "command_tracking")
        self.assertEqual(reward_terms["terms"][0]["nonzero_observation_fraction"], 0.0)
        self.assertEqual(reward_terms["terms"][1]["public_label"], "upright_margin")
        self.assertEqual(reward_terms["terms"][1]["nonzero_observation_fraction"], 1.0)

        diagnostic = next(iter(activation["diagnostic_outputs"]["fields"].values()))
        self.assertEqual(diagnostic["name"], "drift_diagnostic")
        metric = next(iter(diagnostic["fields"].values()))
        self.assertEqual(metric["public_label"], "mean_drift_speed")
        self.assertEqual(metric["mean"], 0.314)
        self.assertTrue(metric["finite"])
        self.assertEqual(metric["shape"], [])
        self.assertLess(len(json.dumps(sections, sort_keys=True).encode("utf-8")), 32 * 1024)
        assert_public_only_prompt_payload(sections)

    def test_push_runtime_feedback_bounds_long_diagnostic_names_and_nested_keys(self) -> None:
        diagnostic_outputs = {
            f"candidate_diagnostic_{index:03d}_" + ("very_long_name_" * 160): {
                "mean": float(index),
                "nested_candidate_key_" + ("overflow_" * 120): "value",
            }
            for index in range(24)
        }

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[
                {
                    "round_id": "push-round-long-diagnostic-names",
                    "runtime_activation_summary": {
                        "reward_term_calls": 2,
                        "diagnostic_calls": 24,
                        "nonzero_reward_contribution": True,
                        "reward_output_stats": {"mean": 0.2, "finite_fraction": 1.0},
                        "diagnostic_outputs": diagnostic_outputs,
                        "capability_activation_records": ["diagnostic_term"],
                    },
                    "decision": {"accepted": False, "outcome": "reject", "public_score": 0.1},
                }
            ],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        outputs = sections["runtime_feedback_summary"]["latest_runtime_activation"]["diagnostic_outputs"]
        serialized = json.dumps(sections, sort_keys=True)
        self.assertLess(len(serialized.encode("utf-8")), 32 * 1024)
        self.assertEqual(outputs["field_count"], 24)
        self.assertIn("field_key_metadata", outputs)
        self.assertTrue(all(len(key) <= 96 for key in outputs["fields"]))
        self.assertTrue(all(key.startswith("diagnostic_") for key in outputs["fields"]))
        for value in outputs["fields"].values():
            if isinstance(value, dict):
                self.assertTrue(all(len(str(key)) <= 96 for key in value))
                self.assertTrue(
                    all(
                        str(key) in {"mean", "std", "min", "max", "finite_fraction", "nonzero_fraction", "nan_fraction", "inf_fraction", "count", "omitted_field_count"}
                        or str(key).startswith("field_")
                        for key in value
                    )
                )
        self.assertNotIn("very_long_name_very_long_name_very_long_name", serialized)
        self.assertNotIn("overflow_overflow_overflow_overflow", serialized)
        assert_public_only_prompt_payload(sections)

    def test_push_runtime_feedback_hard_bounds_single_wide_diagnostic_field(self) -> None:
        def wide_nested_payload(depth: int) -> dict[str, object] | str:
            if depth <= 0:
                return "leaf_payload_" + ("x" * 320)
            return {
                f"wide_nested_key_{depth}_{index}_" + ("candidate_context_" * 12): wide_nested_payload(depth - 1)
                for index in range(10)
            }

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[
                {
                    "round_id": "push-round-wide-diagnostic-field",
                    "runtime_activation_summary": {
                        "reward_term_calls": 2,
                        "diagnostic_calls": 1,
                        "nonzero_reward_contribution": True,
                        "reward_output_stats": {"mean": 0.2, "finite_fraction": 1.0},
                        "diagnostic_outputs": {
                            "single_wide_diagnostic": wide_nested_payload(3),
                        },
                        "capability_activation_records": ["diagnostic_term"],
                    },
                    "decision": {"accepted": False, "outcome": "reject", "public_score": 0.1},
                }
            ],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        outputs = sections["runtime_feedback_summary"]["latest_runtime_activation"]["diagnostic_outputs"]
        serialized_outputs = json.dumps(outputs, sort_keys=True)
        self.assertLessEqual(len(serialized_outputs.encode("utf-8")), outputs["byte_budget"])
        self.assertEqual(outputs["field_count"], 1)
        self.assertGreaterEqual(outputs["omitted_field_count"], 0)
        self.assertLess(len(json.dumps(sections, sort_keys=True).encode("utf-8")), 32 * 1024)
        self.assertNotIn("leaf_payload_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", serialized_outputs)
        assert_public_only_prompt_payload(sections)

    def test_push_runtime_feedback_does_not_project_raw_candidate_diagnostic_keys(self) -> None:
        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[
                {
                    "round_id": "push-round-candidate-diagnostic-keys",
                    "runtime_activation_summary": {
                        "reward_term_calls": 2,
                        "diagnostic_calls": 2,
                        "nonzero_reward_contribution": True,
                        "reward_output_stats": {"mean": 0.2, "finite_fraction": 1.0},
                        "diagnostic_outputs": {
                            "push_recovery_margin": {
                                "mean": 0.2,
                                "candidate_specific_inner_metric": 0.3,
                            },
                            "custom_candidate_hint": 0.4,
                        },
                        "capability_activation_records": ["diagnostic_term"],
                    },
                    "decision": {"accepted": False, "outcome": "reject", "public_score": 0.1},
                }
            ],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        outputs = sections["runtime_feedback_summary"]["latest_runtime_activation"]["diagnostic_outputs"]
        serialized = json.dumps(sections, sort_keys=True)
        self.assertTrue(all(key.startswith("diagnostic_") for key in outputs["fields"]))
        self.assertIn("field_key_metadata", outputs)
        self.assertNotIn("push_recovery_margin", serialized)
        self.assertNotIn("candidate_specific_inner_metric", serialized)
        self.assertNotIn("custom_candidate_hint", serialized)
        assert_public_only_prompt_payload(sections)

    def test_push_prompt_adapter_summarizes_full_eval_feedback_without_private_detail(self) -> None:
        ledger_record = valid_accepted_ledger_record(
            round_id="push-round-full-feedback",
            candidate_id="full_feedback_candidate",
            public_score=0.42,
        )
        ledger_record["task"] = {"task_id": "unitree_push_recovery"}
        metrics = ledger_record["public_result"]["metrics"]
        metrics["success_rate"] = 0.42
        metrics["public_dev"] = {
            "score_stats": {"count": 2, "mean": 0.31}
        }
        metrics["prompt_feedback"] = {
            "schema_version": "true_rl_agent.public_eval_feedback.push_recovery.v1",
            "dominant_failure": {
                "status": "available",
                "source_split": "public_dev",
                "label": "fall_forward",
                "rate": 0.57,
            },
        }
        metrics["sealed_validation_gate"].update(
            status="fail",
            reason="sealed_rollout_failed",
            checkpoint_report_path="D:/runner/eval/checkpoint_report.json",
        )
        metrics["behavior_quality_gate"] = {
            "status": "fail",
            "reason": "recovery_signal_missing",
        }
        metrics["acceptance"]["required_pass_gates"].append(
            "behavior_quality_gate"
        )
        candidate_stats = dict(metrics["acceptance"]["score_stats"])
        ledger_record["decision"].update(
            accepted=False,
            outcome="reject",
            reason="required_gate_failed",
            comparison={
                "mode": "accepted",
                "score_source": "public_accept.score_stats.mean",
                "candidate_score": 0.42,
                "current_best_score": 0.40,
                "delta": 0.02,
                "threshold": 0.05,
                "delta_met": False,
                "candidate_score_stats": candidate_stats,
                "best_score_stats": {
                    "count": 3,
                    "mean": 0.40,
                    "standard_error": 0.02,
                },
                "current_best": {
                    "accepted_round_id": "push-round-best",
                    "accepted_candidate_id": "best_candidate",
                },
            },
        )
        ledger_record["decision"]["consumed_gates"].append(
            "behavior_quality_gate"
        )
        ledger_record["decision"]["gate_statuses"].update(
            sealed_validation_gate="fail",
            behavior_quality_gate="fail",
        )
        record = ExperimentRecord.from_ledger_record(ledger_record)

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[record.ledger_public_record()],
            public_context={"previous_rounds": 1, "public_only": True},
        )
        feedback = sections["public_feedback_summary"]
        feedback_text = json.dumps(feedback, sort_keys=True)

        self.assertNotIn("comparison", feedback["eval_feedback"])
        self.assertNotIn("latest_dominant_failure", feedback["eval_feedback"])
        self.assertEqual(
            feedback["eval_feedback"]["latest_split_summary"]["public_accept"][
                "score_stats"
            ]["mean"],
            0.42,
        )
        self.assertEqual(
            feedback["eval_feedback"]["sealed_validation"]["latest"]["candidate_isolation_status"],
            "pass",
        )
        self.assertEqual(feedback["eval_feedback"]["failed_gate_trend"]["failed_counts"]["behavior_quality_gate"], 1)
        self.assertEqual(feedback["eval_feedback"]["failed_gate_trend"]["failed_counts"]["sealed_validation_gate"], 1)
        self.assertNotIn("checkpoint_report_path", feedback_text)
        self.assertNotIn("D:/runner", feedback_text)
        self.assertNotIn("trusted_seed", feedback_text)
        assert_public_only_prompt_payload(sections)

    def test_push_prompt_adapter_replaces_bootstrap_examples_with_bounded_real_history(self) -> None:
        ledger_record = valid_accepted_ledger_record(
            round_id="push-round-real-shaped-history",
            candidate_id="real_shaped_candidate",
            public_score=0.744792,
        )
        ledger_record["task"] = {"task_id": "unitree_push_recovery"}
        metrics = ledger_record["public_result"]["metrics"]
        metrics.update(
            success_rate=0.744792,
            failure_breakdown={
                "fall_forward": 0.145833,
                "fall_backward": 0.015625,
                "fall_side": 0.03125,
                "excessive_drift": 0.0625,
            },
            motion_quality={
                "mean_action_l2": 4.281783,
                "mean_action_rate_l2": 1.095947,
                "mean_episode_steps": 200,
                "mean_final_planar_speed_mps": 0.241101,
                "mean_final_xy_drift_m": 0.541359,
                "mean_max_abs_pitch_rad": 0.345028,
                "mean_max_abs_roll_rad": 0.615307,
                "mean_min_base_height_m": 0.674095,
                "mean_return": -1.279707,
                "recovery_window_success_rate": 1.0,
            },
        )
        ledger_record["runtime_activation_summary"] = {
            "reward_term_calls": 36000,
            "diagnostic_calls": 500,
            "nonzero_reward_contribution": True,
            "reward_output_stats": {"finite_fraction": 1.0, "mean": -0.172004},
            "component_activation_records": [
                {
                    "component_kind": "reward_term",
                    "name": name,
                    "observation_count": 500,
                    "nonzero_observation_count": 500,
                    "nonzero_observation_fraction": 1.0,
                    "max_abs_weighted_contribution": 1.9,
                    "last_weighted_mean": weighted_mean,
                    "effective_weight": weight,
                    "sampling_interval_steps": 24,
                }
                for name, weighted_mean, weight in (
                    ("upright_margin", 0.780287, 1.0),
                    ("velocity_penalty", -0.25926, 0.2),
                    ("height_penalty", 0.0, 0.5),
                )
            ],
            "diagnostic_outputs": {
                "push_recovery_metrics": {
                    "name": "push_recovery_metrics",
                    "num_envs": 4096,
                    "fields": {
                        name: {"finite": True, "mean": value, "shape": []}
                        for name, value in (
                            ("mean_action_delta_norm", 11.10906),
                            ("mean_base_height", 0.786801),
                            ("mean_tilt", 0.013029),
                            ("mean_vel_norm", 0.432099),
                        )
                    },
                }
            },
            "capability_activation_records": ["reward_callable", "diagnostic"],
        }
        ledger_record["runtime_activation_gate"] = {
            "status": "pass",
            "reason": "runner_activation_observed",
        }
        ledger_record["mechanism_attribution"] = {
            "schema_version": "true_rl_agent.mechanism_attribution.v1",
            "status": "not_attributed",
            "reason": "shadow_ablation_not_run",
            "causal_claim": False,
            "runtime_activation_gate": {
                "status": "pass",
                "reason": "runner_activation_observed",
            },
            "attribution_method": "knob_only_shadow_candidate",
            "shadow_ablation_status": "static_shadow_generated",
        }
        record = ExperimentRecord.from_ledger_record(ledger_record)
        public_record = record.ledger_public_record()
        task_summary = PushTaskMemoryAdapter().build_public_summary([public_record])

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[public_record],
            public_context={
                "previous_rounds": 1,
                "public_only": True,
                "task_public_summary": task_summary,
            },
        )

        self.assertEqual(sections["known_good_examples"], [])
        self.assertEqual(
            sections["public_feedback_summary"]["latest_public_score"],
            0.744792,
        )
        self.assertEqual(
            sections["runtime_feedback_summary"]["latest_runtime_activation"][
                "reward_term_calls"
            ],
            36000,
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(sections),
            MAX_TASK_PROMPT_SECTIONS_BYTES,
        )
        assert_public_only_prompt_payload(sections)

    def test_push_prompt_feedback_ignores_unbound_raw_scores(self) -> None:
        accepted = ExperimentRecord.from_ledger_record(
            valid_accepted_ledger_record(
                round_id="push-round-accepted",
                candidate_id="accepted_candidate",
                public_score=0.34,
            )
        )
        rejected_high_score = ExperimentRecord.from_ledger_record(
            {
                "round_id": "push-round-parameter-only",
                "task": {"task_id": "unitree_push_recovery"},
                "candidate": {
                    "candidate_id": "parameter_candidate",
                    "changed_components": ["reward_weight_scale"],
                },
                "candidate_surface": {"classification": "parameter_only"},
                "public_result": {
                    "status": "pass",
                    "public_score": 0.99,
                    "metrics": {"success_rate": 0.99},
                },
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "parameter_only_candidate",
                    "public_score": 0.99,
                },
            }
        )
        rejected_high_score = replace(
            rejected_high_score,
            public_record=_bind_record_identity(
                rejected_high_score.ledger_public_record(),
                binding=TEST_EVALUATION_BINDING,
                session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            ),
        )

        payload = PublicContextBuilder(
            task_prompt_adapter=PushRecoveryTaskPlugin().prompt_adapter(),
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        ).build_prompt_payload([accepted, rejected_high_score])
        feedback = payload["public_feedback_summary"]

        self.assertEqual(feedback["best_public_score"], 0.34)
        self.assertEqual(feedback["best_accepted_public_score"], 0.34)
        self.assertEqual(feedback["best_observed_public_score"], 0.34)
        self.assertIsNone(feedback["latest_public_score"])

    def test_push_feedback_does_not_count_invalid_declared_accept(self) -> None:
        valid = valid_accepted_ledger_record(
            round_id="round-valid",
            candidate_id="candidate-valid",
            public_score=0.25,
        )
        invalid = copy.deepcopy(valid)
        invalid["round_id"] = "round-invalid-accept"
        invalid["public_result"]["public_score"] = 0.99
        invalid["decision"].update(
            accepted=True,
            outcome="reject",
            reason="candidate_verifier_failed",
            public_score=0.99,
        )

        payload = PublicContextBuilder(
            task_prompt_adapter=PushRecoveryTaskPlugin().prompt_adapter(),
            acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
            initial_current_best=make_bootstrap_current_best(),
            session_genesis_hash=TEST_SESSION_GENESIS_HASH,
        ).build_prompt_payload([valid, invalid])
        feedback = payload["public_feedback_summary"]

        self.assertEqual(feedback["accepted_count"], 1)
        self.assertEqual(feedback["best_accepted_public_score"], 0.25)
        self.assertEqual(feedback["best_observed_public_score"], 0.25)

    def test_push_prompt_feedback_reads_real_split_failure_breakdown(self) -> None:
        public_result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.25)],
            public_accept_summaries=[self._eval_summary(seed=2001, success_rate=0.25)],
            budget_summary=self._budget_summary(),
        )
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "push-round-real-metrics",
                "task": {"task_id": "unitree_push_recovery"},
                "candidate": {"candidate_id": "real_metrics_candidate", "changed_components": ["reward_callable"]},
                "candidate_surface": {"classification": "code_level"},
                "public_result": public_result.to_public_dict(),
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "fall_forward_dominated",
                    "public_score": public_result.public_score,
                },
            }
        )
        public_record = record.ledger_public_record()

        memory_summary = PushTaskMemoryAdapter().build_public_summary([public_record])
        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[public_record],
            public_context={
                "previous_rounds": 1,
                "public_only": True,
                "task_public_summary": memory_summary,
            },
        )

        self.assertEqual(classify_push_failure(public_result.metrics), "fall_dominated")
        self.assertEqual(memory_summary["latest_failure_mode"], "fall_dominated")
        self.assertEqual(memory_summary["latest_failure_breakdown"]["fall_forward"], 0.75)
        self.assertEqual(memory_summary["latest_motion_quality"]["mean_min_base_height_m"], 0.78)
        self.assertEqual(memory_summary["latest_motion_quality"]["mean_max_abs_roll_rad"], 0.05)
        self.assertEqual(sections["public_feedback_summary"]["latest_failure_mode"], "fall_dominated")
        self.assertEqual(
            sections["public_feedback_summary"]["task_public_summary"]["latest_failure_breakdown"]["fall_forward"],
            0.75,
        )
        self.assertEqual(
            sections["public_feedback_summary"]["task_public_summary"]["latest_motion_quality"][
                "mean_min_base_height_m"
            ],
            0.78,
        )

    def test_push_public_result_exports_bounded_prompt_feedback_without_seed_leakage(self) -> None:
        public_result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.125)],
            public_accept_summaries=[
                self._eval_summary(seed=2001, success_rate=0.625),
                self._eval_summary(seed=2002, success_rate=0.875),
                self._eval_summary(seed=2003, success_rate=0.75),
            ],
            budget_summary=self._budget_summary(),
        )
        prompt_feedback = public_result.metrics["prompt_feedback"]
        public_text = json.dumps(public_result.to_public_dict(), sort_keys=True)

        self.assertEqual(prompt_feedback["schema_version"], "true_rl_agent.public_eval_feedback.push_recovery.v1")
        self.assertEqual(prompt_feedback["dominant_failure"]["status"], "available")
        self.assertEqual(prompt_feedback["dominant_failure"]["source_split"], "public_dev")
        self.assertEqual(prompt_feedback["dominant_failure"]["label"], "fall_forward")
        self.assertEqual(prompt_feedback["dominant_failure"]["rate"], 0.875)
        self._assert_private_seed_ids_absent(
            public_result.to_public_dict(),
            1001,
            2001,
            2002,
            2003,
        )
        self.assertNotIn("D:/", public_text)

    def test_push_public_result_exposes_public_dev_motion_summary_only(self) -> None:
        public_dev_a = self._eval_summary(seed=1001, success_rate=0.25)
        public_dev_b = self._eval_summary(seed=1002, success_rate=0.50)
        public_dev_b["secondary_metrics"]["mean_min_base_height_m"] = 0.68
        public_accept = self._eval_summary(seed=2001, success_rate=0.75)
        public_accept["secondary_metrics"]["mean_min_base_height_m"] = 0.11

        public_result = build_push_public_result_from_summaries(
            public_dev_summaries=[public_dev_a, public_dev_b],
            public_accept_summaries=[public_accept],
            budget_summary=self._budget_summary(),
        )

        motion_quality = public_result.metrics["motion_quality"]
        self.assertEqual(motion_quality["mean_min_base_height_m"], 0.73)
        self.assertEqual(motion_quality["mean_max_abs_roll_rad"], 0.05)
        self.assertLessEqual(len(motion_quality), 10)
        self.assertNotIn("seed", json.dumps(motion_quality, sort_keys=True))

    def test_push_prompt_feedback_does_not_fallback_to_public_accept_failure_labels(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        binding = backend.evaluation_binding_for(plugin.task_spec)
        public_dev = self._eval_summary(seed=1001, success_rate=1.0)
        del public_dev["aggregate"]["failure_breakdown"]
        public_accept = self._eval_summary(seed=2001, success_rate=0.875)
        public_accept["aggregate"]["failure_breakdown"] = {
            "total": 8,
            "counts": {"success": 7, "accept_only_failure": 1},
            "rates": {"success": 0.875, "accept_only_failure": 0.125},
        }

        public_result = build_push_public_result_from_summaries(
            public_dev_summaries=[public_dev],
            public_accept_summaries=[
                public_accept,
                self._eval_summary(seed=2002, success_rate=1.0),
                self._eval_summary(seed=2003, success_rate=1.0),
            ],
            budget_summary=self._budget_summary(),
        )
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "push-round-accept-label",
                "task": {"task_id": "unitree_push_recovery"},
                "candidate": {"candidate_id": "accept_label_candidate", "changed_components": ["reward_callable"]},
                "candidate_surface": {"classification": "code_level"},
                "public_result": public_result.to_public_dict(),
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "required_gate_failed",
                    "public_score": public_result.public_score,
                },
            }
        )
        record = replace(
            record,
            public_record=_bind_record_identity(
                record.ledger_public_record(),
                binding=binding,
                session_genesis_hash="9" * 64,
            ),
        )

        payload = PublicContextBuilder(
            task_memory_adapter=PushTaskMemoryAdapter(),
            task_prompt_adapter=plugin.prompt_adapter(),
            acceptance_protocol=plugin.task_spec.acceptance_protocol,
            initial_current_best=self._round_session_fields(
                plugin=plugin,
                backend=backend,
                initial_best_score=0.0,
            )["current_best_eval_stats"],
            session_genesis_hash="9" * 64,
        ).build_prompt_payload([record])
        payload_text = json.dumps(payload, sort_keys=True)

        self.assertEqual(
            public_result.metrics["prompt_feedback"]["dominant_failure"],
            {"status": "not_available", "reason": "public_dev_failure_breakdown_missing"},
        )
        self.assertNotIn(
            "latest_dominant_failure",
            payload["public_feedback_summary"]["eval_feedback"],
        )
        self.assertEqual(payload["public_feedback_summary"]["latest_failure_mode"], "unknown")
        self.assertEqual(
            payload["public_feedback_summary"]["task_public_summary"]["latest_failure_mode"],
            "unknown",
        )
        self.assertNotIn("accept_only_failure", payload_text)

    def test_push_split_failure_breakdown_weights_missing_labels_as_zero(self) -> None:
        first_seed = self._eval_summary(seed=1001, success_rate=0.0)
        first_seed["aggregate"]["failure_breakdown"] = {
            "total": 10,
            "counts": {"fall_forward": 10},
            "rates": {"fall_forward": 1.0},
        }
        first_seed["aggregate"]["episodes"] = 10
        second_seed = self._eval_summary(seed=1002, success_rate=0.5)
        second_seed["aggregate"]["failure_breakdown"] = {
            "total": 10,
            "counts": {"success": 5, "timeout": 5},
            "rates": {"success": 0.5, "timeout": 0.5},
        }
        second_seed["aggregate"]["episodes"] = 10

        public_result = build_push_public_result_from_summaries(
            public_dev_summaries=[first_seed, second_seed],
            public_accept_summaries=[],
            budget_summary=self._budget_summary(),
        )

        breakdown = extract_push_failure_breakdown(public_result.metrics)
        self.assertEqual(breakdown["fall_forward"], 0.5)
        self.assertEqual(breakdown["timeout"], 0.25)
        self.assertLessEqual(sum(breakdown.values()), 1.0)

    def test_push_runtime_feedback_reads_real_train_candidate_summary_shape(self) -> None:
        record = ExperimentRecord.from_ledger_record(
            {
                "round_id": "push-round-runtime-shape",
                "task": {"task_id": "unitree_push_recovery"},
                "candidate": {"candidate_id": "runtime_shape_candidate", "changed_components": ["reward_callable"]},
                "candidate_surface": {"classification": "code_level"},
                "train_result": {
                    "summary": {
                        "schema_version": "true_rl_agent.push_recovery.train_summary.v1",
                        "candidate": {
                            "schema_version": "true_rl_agent.push_recovery.candidate_runtime.summary.v1",
                            "enabled": True,
                            "components": {"effective_hook_counts": {"reward_terms": 1}},
                        },
                        "runtime_activation_summary": {
                            "reward_term_calls": 8,
                            "diagnostic_calls": 0,
                            "nonzero_reward_contribution": True,
                            "reward_output_stats": {"mean": 0.1, "finite_fraction": 1.0},
                            "diagnostic_outputs": {},
                            "capability_activation_records": ["reward_callable"],
                        },
                    }
                },
                "public_result": build_push_public_result(
                    success_rate=0.20,
                    failure_breakdown={"fall": 0.4},
                    motion_quality={"mean_base_height": 0.78},
                ).to_public_dict(),
                "decision": {
                    "accepted": False,
                    "outcome": "reject",
                    "reason": "runtime_probe",
                    "public_score": 0.20,
                },
            }
        )

        sections = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[record.ledger_public_record()],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        runtime = sections["runtime_feedback_summary"]
        self.assertIs(runtime["candidate_package_loaded"], True)
        self.assertEqual(runtime["latest_runtime_activation"]["reward_term_calls"], 8)

    def test_push_prompt_known_good_examples_verify_as_candidate_packages(self) -> None:
        examples = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[],
            public_context={"previous_rounds": 0, "public_only": True},
        )["known_good_examples"]

        for index, example in enumerate(examples, start=1):
            with self.subTest(kind=example["kind"]):
                with tempfile.TemporaryDirectory() as tmp:
                    candidate = Path(tmp) / f"candidate_{index}"
                    (candidate / "training_code").mkdir(parents=True)
                    (candidate / "candidate_manifest.yaml").write_text(
                        textwrap.dedent(
                            f"""
                            schema_version: true_rl_agent.candidate.v1
                            candidate_id: prompt_example_{index}
                            entrypoint: training_code.components:build_training_components
                            """
                        ).strip()
                        + "\n",
                        encoding="utf-8",
                    )
                    (candidate / "training_code" / "components.py").write_text(
                        example["components_py_snippet"].strip() + "\n",
                        encoding="utf-8",
                    )

                    result = StaticCandidateVerifier().verify(candidate)

                self.assertEqual(result.status, "pass")

    def test_push_public_result_and_failure_classifier_are_public_metrics(self) -> None:
        result = build_push_public_result(
            success_rate=0.42,
            failure_breakdown={"fall": 0.2, "push_drift": 0.1},
            motion_quality={"mean_base_height": 0.79, "mean_tilt_rad": 0.04},
        )

        self.assertEqual(result.public_score, 0.42)
        self.assertEqual(result.metrics["success_rate"], 0.42)
        self.assertIn("failure_breakdown", result.metrics)
        self.assertIn("motion_quality", result.metrics)
        self.assertEqual(classify_push_failure(result.metrics), "fall_dominated")
        self.assertNotIn("trusted_seed_count", result.to_public_dict())

    def test_push_public_result_aggregates_public_splits_without_seed_leakage(self) -> None:
        result = build_push_public_result_from_summaries(
            public_dev_summaries=[
                self._eval_summary(seed=1001, success_rate=0.50),
                self._eval_summary(seed=1002, success_rate=0.75),
            ],
            public_accept_summaries=[
                self._eval_summary(seed=2001, success_rate=0.625),
                self._eval_summary(seed=2002, success_rate=0.875),
                self._eval_summary(seed=2003, success_rate=0.75),
            ],
            budget_summary=self._budget_summary(),
        )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.public_score, 0.75)
        self.assertEqual(result.metrics["public_dev"]["score_stats"]["count"], 2)
        self.assertEqual(result.metrics["public_dev"]["score_stats"]["mean"], 0.625)
        self.assertEqual(result.metrics["public_accept"]["score_stats"]["count"], 3)
        self.assertEqual(result.metrics["public_accept"]["score_stats"]["episode_count"], 24)
        self.assertEqual(result.metrics["public_accept"]["score_stats"]["mean"], 0.75)
        self.assertIsNotNone(result.metrics["public_accept"]["score_stats"]["std"])
        self.assertIsNotNone(result.metrics["public_accept"]["score_stats"]["standard_error"])
        self.assertEqual(result.metrics["accept_evidence_gate"]["status"], "pass")
        self.assertEqual(result.metrics["budget_accounting"]["status"], "pass")
        self.assertEqual(result.metrics["motion_quality_gate"]["status"], "pass")
        self.assertEqual(result.metrics["acceptance"]["score_source"], "public_accept.score_stats.mean")
        self.assertEqual(
            result.metrics["acceptance"]["required_pass_gates"],
            ["accept_evidence_gate", "budget_accounting", "motion_quality_gate", "behavior_quality_gate"],
        )
        self._assert_private_seed_ids_absent(
            result.to_public_dict(),
            1001,
            1002,
            2001,
            2002,
            2003,
        )

    def test_push_metric_ingestion_rejects_coercible_or_nonfinite_success_rates(self) -> None:
        malformed_values = (True, False, "0.75", float("nan"), float("inf"))
        for malformed in malformed_values:
            with self.subTest(malformed=repr(malformed)):
                accept_summaries = [
                    self._eval_summary(seed=2001 + index, success_rate=0.75)
                    for index in range(3)
                ]
                for summary in accept_summaries:
                    summary["aggregate"]["success_rate"] = malformed

                result = build_push_public_result_from_summaries(
                    public_dev_summaries=[
                        self._eval_summary(seed=1001, success_rate=0.50)
                    ],
                    public_accept_summaries=accept_summaries,
                    budget_summary=self._budget_summary(),
                )

                stats = result.metrics["public_accept"]["score_stats"]
                self.assertEqual(result.status, "fail")
                self.assertEqual(stats["count"], 0)
                self.assertIsNone(stats["mean"])
                self.assertEqual(
                    sum(
                        error.get("reason")
                        == "success_rate_missing_nonfinite_or_out_of_range"
                        for error in result.metrics["errors"]
                    ),
                    3,
                )

    def test_legacy_push_result_rejects_coercible_or_nonfinite_success_rate(self) -> None:
        for malformed in (True, "0.42", float("nan"), float("inf")):
            with self.subTest(malformed=repr(malformed)):
                with self.assertRaisesRegex(ValueError, "finite non-boolean number"):
                    build_push_public_result(
                        success_rate=malformed,
                        failure_breakdown={"fall": 0.2},
                        motion_quality={"mean_base_height": 0.79},
                    )

    def test_push_public_result_requires_behavior_quality_gate(self) -> None:
        result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
            public_accept_summaries=[
                self._eval_summary(seed=2001, success_rate=0.625),
                self._eval_summary(seed=2002, success_rate=0.875),
                self._eval_summary(seed=2003, success_rate=0.75),
            ],
            budget_summary=self._budget_summary(),
        )

        self.assertEqual(result.metrics["behavior_quality_gate"]["status"], "pass")
        self.assertEqual(result.metrics["behavior_quality_gate"]["failure_breakdown_status"], "pass")
        self.assertIn("mean_min_base_height_m", result.metrics["behavior_quality_gate"]["checked_metrics"])
        self.assertIn("mean_final_planar_speed_mps", result.metrics["behavior_quality_gate"]["checked_metrics"])
        self.assertIn("recovery_window_success_rate", result.metrics["behavior_quality_gate"]["checked_metrics"])
        self.assertIn("behavior_quality_gate", result.metrics["acceptance"]["required_pass_gates"])

    def test_push_public_result_fails_behavior_gate_for_mixed_malformed_failure_breakdown(self) -> None:
        malformed_payloads = (
            {"total": 8, "counts": {"success": 6, "fall_forward": 1, "timeout": "not_numeric"}},
            {"rates": {"success": 0.75, "fall_forward": 0.125, "timeout": "not_numeric"}},
            {"total": 8},
        )
        for payload in malformed_payloads:
            with self.subTest(payload=payload):
                bad_summary = self._eval_summary(seed=2001, success_rate=0.75)
                bad_summary["aggregate"]["failure_breakdown"] = payload

                result = build_push_public_result_from_summaries(
                    public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
                    public_accept_summaries=[
                        bad_summary,
                        self._eval_summary(seed=2002, success_rate=0.875),
                        self._eval_summary(seed=2003, success_rate=0.75),
                    ],
                    budget_summary=self._budget_summary(),
                )

                gate = result.metrics["behavior_quality_gate"]
                self.assertEqual(gate["status"], "fail")
                self.assertIn(
                    {"split": "public_accept", "run_index": 1, "reason": "failure_breakdown_unparseable"},
                    gate["failures"],
                )

    def test_extract_push_failure_breakdown_rejects_mixed_malformed_public_split(self) -> None:
        metrics = {
            "public_accept": {
                "seed_results": [
                    {
                        "episodes": 8,
                        "failure_breakdown": {
                            "total": 8,
                            "counts": {"success": 6, "fall_forward": 1, "timeout": "not_numeric"},
                        },
                    }
                ]
            }
        }

        self.assertEqual(extract_push_failure_breakdown(metrics), {})

    def test_push_public_result_fails_behavior_gate_for_out_of_range_behavior_metrics(self) -> None:
        cases = (
            ("recovery_window_success_rate", 1.5, "out_of_range"),
            ("mean_final_xy_drift_m", -0.1, "negative_value"),
        )
        for metric, value, reason in cases:
            with self.subTest(metric=metric):
                bad_summary = self._eval_summary(seed=2001, success_rate=0.75)
                bad_summary["secondary_metrics"][metric] = value

                result = build_push_public_result_from_summaries(
                    public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
                    public_accept_summaries=[
                        bad_summary,
                        self._eval_summary(seed=2002, success_rate=0.875),
                        self._eval_summary(seed=2003, success_rate=0.75),
                    ],
                    budget_summary=self._budget_summary(),
                )

                gate = result.metrics["behavior_quality_gate"]
                self.assertEqual(gate["status"], "fail")
                self.assertIn(
                    {"split": "public_accept", "run_index": 1, "metric": metric, "reason": reason},
                    gate["failures"],
                )

    def test_behavior_gate_recovery_signal_status_reports_missing_when_required_signal_absent(self) -> None:
        bad_summary = self._eval_summary(seed=2001, success_rate=0.75)
        del bad_summary["secondary_metrics"]["mean_final_planar_speed_mps"]

        result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
            public_accept_summaries=[
                bad_summary,
                self._eval_summary(seed=2002, success_rate=0.875),
                self._eval_summary(seed=2003, success_rate=0.75),
            ],
            budget_summary=self._budget_summary(),
        )

        gate = result.metrics["behavior_quality_gate"]
        self.assertEqual(gate["status"], "fail")
        self.assertEqual(gate["recovery_signal_status"], "missing")

    def test_push_public_result_fails_behavior_gate_for_missing_behavior_metric(self) -> None:
        bad_summary = self._eval_summary(seed=2001, success_rate=0.75)
        del bad_summary["secondary_metrics"]["mean_min_base_height_m"]

        result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
            public_accept_summaries=[
                bad_summary,
                self._eval_summary(seed=2002, success_rate=0.875),
                self._eval_summary(seed=2003, success_rate=0.75),
            ],
            budget_summary=self._budget_summary(),
        )

        gate = result.metrics["behavior_quality_gate"]
        self.assertEqual(gate["status"], "fail")
        self.assertIn(
            {
                "split": "public_accept",
                "run_index": 1,
                "metric": "mean_min_base_height_m",
                "reason": "missing_or_nonfinite",
            },
            gate["failures"],
        )
        self.assertIn("behavior_quality_gate", result.metrics["acceptance"]["required_pass_gates"])

    def test_push_public_result_fails_behavior_gate_for_malformed_failure_breakdown(self) -> None:
        bad_summary = self._eval_summary(seed=2001, success_rate=0.75)
        bad_summary["aggregate"]["failure_breakdown"] = {
            "total": 8,
            "counts": {"success": 6, "fall_forward": "not_numeric"},
        }

        result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
            public_accept_summaries=[
                bad_summary,
                self._eval_summary(seed=2002, success_rate=0.875),
                self._eval_summary(seed=2003, success_rate=0.75),
            ],
            budget_summary=self._budget_summary(),
        )

        gate = result.metrics["behavior_quality_gate"]
        self.assertEqual(gate["status"], "fail")
        self.assertIn(
            {"split": "public_accept", "run_index": 1, "reason": "failure_breakdown_unparseable"},
            gate["failures"],
        )
        self._assert_private_seed_ids_absent(
            result.to_public_dict(),
            1001,
            2001,
            2002,
            2003,
        )

    def test_push_public_accept_count_below_protocol_fails_accept_evidence_gate(self) -> None:
        result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
            public_accept_summaries=[self._eval_summary(seed=2001, success_rate=1.0)],
            budget_summary=self._budget_summary(),
        )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.public_score, 1.0)
        self.assertEqual(result.metrics["public_accept"]["score_stats"]["count"], 1)
        self.assertEqual(result.metrics["public_accept"]["score_stats"]["episode_count"], 8)
        self.assertEqual(result.metrics["accept_evidence_gate"]["status"], "fail")
        self.assertEqual(result.metrics["accept_evidence_gate"]["reason"], "insufficient_accept_evidence")
        self.assertEqual(result.metrics["accept_evidence_gate"]["observed_count"], 1)
        self.assertEqual(result.metrics["accept_evidence_gate"]["required_count"], 3)
        self.assertIn("accept_evidence_gate", result.metrics["acceptance"]["required_pass_gates"])

    def test_push_public_result_keeps_eval_completion_separate_from_motion_gate(self) -> None:
        result = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50, action_l2=5000.0)],
            public_accept_summaries=[self._eval_summary(seed=2001, success_rate=0.75, action_l2=5000.0)],
            budget_summary=self._budget_summary(),
        )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.metrics["motion_quality_gate"]["status"], "fail")

    def test_push_plugin_projects_budget_gate_without_owning_round_decision(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        raw_result = build_push_public_result(
            success_rate=0.30,
            failure_breakdown={"fall": 0.1},
            motion_quality={"mean_final_xy_drift_m": 0.1},
        )

        projected = plugin.project_public_result(
            public_result=raw_result,
            train_summary={"training": {"budget_accounting": self._budget_summary()["budget_accounting"]}},
        )

        self.assertEqual(projected.metrics["budget_accounting"]["status"], "pass")
        self.assertIn("budget_accounting", projected.metrics["acceptance"]["required_pass_gates"])
        self.assertEqual(projected.metrics["behavior_quality_gate"]["status"], "fail")
        self.assertIn("behavior_quality_gate", projected.metrics["acceptance"]["required_pass_gates"])
        self.assertFalse(hasattr(plugin, "decide"))
        self.assertFalse(hasattr(plugin, "accept"))
        self.assertFalse(hasattr(plugin, "reject"))

    def test_legacy_single_eval_projection_cannot_bypass_behavior_quality_gate(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        raw_result = build_push_public_result(
            success_rate=0.30,
            failure_breakdown={"fall": 0.1},
            motion_quality={"mean_final_xy_drift_m": 0.1},
        )
        metrics = dict(raw_result.metrics)
        metrics["sealed_validation_gate"] = {
            "status": "pass",
            "checkpoint_safety_status": "pass",
            "candidate_isolation_status": "pass",
        }
        acceptance = dict(metrics["acceptance"])
        score_stats = dict(acceptance["score_stats"])
        score_stats["eval_protocol_hash"] = plugin.task_spec.fixed_eval_protocol["eval_protocol_hash"]
        acceptance["score_stats"] = score_stats
        metrics["acceptance"] = acceptance
        raw_result = PublicEvalResult(status=raw_result.status, public_score=raw_result.public_score, metrics=metrics)

        projected = plugin.project_public_result(
            public_result=raw_result,
            train_summary={"training": {"budget_accounting": self._budget_summary()["budget_accounting"]}},
        )
        backend = self._backend()
        decision = decide_candidate(
            surface=CandidateSurfaceAssessment(
                classification="code_level",
                eligible_for_policy_accept=True,
                reason="code_level_candidate",
            ),
            public_result=projected,
            current_best=self._round_session_fields(
                plugin=plugin,
                backend=backend,
                initial_best_score=0.0,
            )["current_best_eval_stats"],
            acceptance_protocol=plugin.task_spec.acceptance_protocol,
        )

        self.assertEqual(projected.metrics["behavior_quality_gate"]["status"], "fail")
        self.assertIn("behavior_quality_gate", projected.metrics["acceptance"]["required_pass_gates"])
        self.assertIs(decision.outcome, DecisionOutcome.REJECT)
        self.assertEqual(decision.reason, "acceptance_protocol_mismatch")
        self.assertIn("behavior_quality_gate", decision.consumed_gates)
        self.assertEqual(decision.gate_statuses["behavior_quality_gate"], "fail")

    def test_push_projection_requires_runner_projected_sealed_validation_gate(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        raw = build_push_public_result_from_summaries(
            public_dev_summaries=[self._eval_summary(seed=1001, success_rate=0.50)],
            public_accept_summaries=[
                self._eval_summary(seed=2001, success_rate=0.60),
                self._eval_summary(seed=2002, success_rate=0.65),
                self._eval_summary(seed=2003, success_rate=0.70),
            ],
            budget_summary=self._budget_summary(),
        )
        raw_with_seal = PublicEvalResult(
            status=raw.status,
            public_score=raw.public_score,
            metrics={
                **dict(raw.metrics),
                "sealed_validation_gate": {
                    "status": "pass",
                    "checkpoint_safety_status": "pass",
                    "candidate_isolation_status": "pass",
                },
            },
        )

        projected = plugin.project_public_result(
            public_result=raw_with_seal,
            train_summary={"training": {"budget_accounting": self._budget_summary()["budget_accounting"]}},
        )

        self.assertIn("sealed_validation_gate", projected.metrics["acceptance"]["required_pass_gates"])

    def test_push_eval_script_emits_recovery_behavior_speed_metrics(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "isaaclab_rsl_rl_eval.py").read_text(encoding="utf-8")

        self.assertIn('"final_planar_speed_mps"', source)
        self.assertIn('"mean_final_planar_speed_mps"', source)
        self.assertIn('"recovery_window_success_rate"', source)

    def test_push_invariants_reject_task_weakening(self) -> None:
        invariants = PushRecoveryTaskPlugin().task_spec.invariants
        weakened = TaskInvariantSpec({**dict(invariants.values), "episode_length": 128})

        with self.assertRaises(ValueError):
            invariants.assert_compatible_with(weakened)

    def test_push_task_materializes_isaac_jobs_with_fixed_protocol_metadata(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = self._write_reward_candidate(
                Path(tmp),
                candidate_id="push-code-candidate",
            )
            candidate = CandidatePackageRef(
                candidate_id="push-code-candidate",
                path=str(candidate_path),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            candidate = seal_and_verify_candidate(
                candidate,
                artifact_root=Path(tmp) / "candidate_artifacts",
                verifier=StaticCandidateVerifier(),
            ).require_verified()

            report = check_task_backend_compatibility(
                task_plugin=plugin,
                backend_adapter=backend,
                policy_initialization="random",
            )
            bundle = backend.materialize_jobs(
                round_id="push-round-001",
                task_spec=plugin.task_spec,
                candidate=candidate,
                policy_initialization=PolicyInitialization.random(),
            )

        self.assertTrue(report.compatible)
        self.assertEqual(bundle.train_spec.task_id, "unitree_push_recovery")
        self.assertEqual(bundle.task_invariants, plugin.task_spec.invariants)
        self.assertEqual(bundle.train_spec.metadata["fixed_train_protocol"]["budget_steps"], 49_152_000)
        self.assertEqual(bundle.train_spec.metadata["fixed_train_protocol"]["train_num_envs"], 4096)
        self.assertEqual(bundle.train_spec.metadata["fixed_train_protocol"]["train_max_iterations"], 500)
        self.assertEqual(
            bundle.eval_spec.metadata["fixed_eval_protocol"]["eval_protocol"],
            "fixed_public_accept_and_sealed_validation",
        )
        self.assertEqual(bundle.eval_spec.to_public_dict()["sealed_validation_seed_count"], 7)
        self.assertIn("eval_protocol_hash", bundle.eval_spec.to_public_dict())
        self.assertNotIn("trusted_seed_count", bundle.eval_spec.to_public_dict())
        self.assertIn(f"--candidate {candidate.path}", bundle.train_spec.metadata["cli_command"])
        self.assertEqual(bundle.train_spec.metadata["cli_argv"][:2], ["/opt/isaac/python.sh", "train_push.py"])
        self.assertIn(candidate.path, bundle.train_spec.metadata["cli_argv"])
        self.assertEqual(bundle.eval_spec.metadata["cli_argv"][:2], ["/opt/isaac/python.sh", "eval_push.py"])
        self.assertIn("artifact://push-round-001/checkpoints/best.pt", bundle.eval_spec.metadata["cli_argv"])
        self.assertIn("--subject-kind", bundle.eval_spec.metadata["cli_argv"])
        self.assertIn("--subject-id", bundle.eval_spec.metadata["cli_argv"])
        self.assertEqual(
            bundle.eval_spec.metadata["cli_argv"][bundle.eval_spec.metadata["cli_argv"].index("--subject-id") + 1],
            "push-code-candidate",
        )
        self.assertNotIn("--candidate-id", bundle.eval_spec.metadata["cli_argv"])
        self.assertNotIn(str(candidate_path), bundle.eval_spec.metadata["cli_argv"])

    def test_push_backend_rejects_missing_sealed_validation_invariant(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        task_spec = replace(
            plugin.task_spec,
            invariants=TaskInvariantSpec(
                {
                    key: value
                    for key, value in plugin.task_spec.invariants.values.items()
                    if key != "sealed_validation_seed_count"
                }
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            candidate = CandidatePackageRef(
                candidate_id="push-code-candidate",
                path=str(
                    self._write_reward_candidate(
                        Path(tmp),
                        candidate_id="push-code-candidate",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            candidate = seal_and_verify_candidate(
                candidate,
                artifact_root=Path(tmp) / "candidate_artifacts",
                verifier=StaticCandidateVerifier(),
            ).require_verified()

            with self.assertRaisesRegex(ValueError, "task_eval_protocol_mismatch:sealed_validation_seed_count"):
                backend.materialize_jobs(
                    round_id="push-round-missing-sealed-count",
                    task_spec=task_spec,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                )

    def test_push_backend_rejects_sealed_validation_count_drift(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        config = self._backend().config
        drifted_binding = make_evaluation_binding(
            eval_protocol_version=config.eval_protocol_version,
            public_seed_count=config.public_seed_count,
            public_accept_seed_count=config.public_accept_seed_count,
            sealed_validation_seed_count=9,
            eval_num_envs=config.eval_num_envs,
            acceptance_protocol_hash=(
                config.evaluation_binding.acceptance_protocol_hash
            ),
            task_protocol_hash=config.evaluation_binding.task_protocol_hash,
            seed_cohort_hash=config.evaluation_binding.seed_cohort_hash,
        )
        backend = IsaacLabRslRlBackendAdapter(
            replace(
                config,
                sealed_validation_seed_count=9,
                evaluation_binding=drifted_binding,
            )
        )
        with tempfile.TemporaryDirectory() as tmp:
            candidate = CandidatePackageRef(
                candidate_id="push-code-candidate",
                path=str(
                    self._write_reward_candidate(
                        Path(tmp),
                        candidate_id="push-code-candidate",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            candidate = seal_and_verify_candidate(
                candidate,
                artifact_root=Path(tmp) / "candidate_artifacts",
                verifier=StaticCandidateVerifier(),
            ).require_verified()

            with self.assertRaisesRegex(ValueError, "task_eval_protocol_mismatch:sealed_validation_seed_count"):
                backend.materialize_jobs(
                    round_id="push-round-sealed-count-drift",
                    task_spec=plugin.task_spec,
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                )

    def test_push_round_rejects_schedule_only_and_accepts_code_with_runtime_evidence(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            schedule_only = CandidatePackageRef(
                candidate_id="schedule-only",
                path=str(
                    self._write_schedule_only_candidate(
                        Path(tmp),
                        candidate_id="schedule-only",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_weight_scale"})),
            )
            code_level = CandidatePackageRef(
                candidate_id="reward-code",
                path=str(
                    self._write_reward_candidate(
                        Path(tmp),
                        candidate_id="reward-code",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )

            schedule_result = run_candidate_round(
                RoundInputs(
                    round_id="push-round-001",
                    task_plugin=plugin,
                    backend_adapter=backend,
                    runner=PushDryRunRunner(),
                    verifier=PassingVerifier(),
                    candidate=schedule_only,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **self._round_session_fields(
                        plugin=plugin,
                        backend=backend,
                        initial_best_score=0.10,
                    ),
                )
            )
            code_result = run_candidate_round(
                RoundInputs(
                    round_id="push-round-002",
                    task_plugin=plugin,
                    backend_adapter=backend,
                    runner=PushDryRunRunner(),
                    verifier=PassingVerifier(),
                    candidate=code_level,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 1, "public_only": True},
                    **self._round_session_fields(
                        plugin=plugin,
                        backend=backend,
                        initial_best_score=0.10,
                    ),
                )
            )
            ledger = load_ledger(ledger_path)

            self.assertIs(schedule_result.decision.outcome, DecisionOutcome.REJECT)
            self.assertEqual(ledger[0]["decision"]["reason"], "parameter_only_candidate")
            self.assertIs(code_result.decision.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
            self.assertEqual(ledger[1]["public_result"]["metrics"]["success_rate"], 0.30)
            self.assertEqual(ledger[1]["train_result"]["checkpoint"]["role"], "eval")
            self.assertTrue(ledger[1]["train_result"]["checkpoint"]["sealed"])
            public_text = json.dumps(
                {
                    "jobs": ledger[1]["jobs"],
                    "train_result": ledger[1]["train_result"],
                    "public_result": ledger[1]["public_result"],
                    "runner_error": ledger[1]["runner_error"],
                },
                sort_keys=True,
            )
            for forbidden in (
                "outputs/runner",
                "local_path",
                "source_latest_checkpoint",
                "checkpoint_output_path",
                "work_dir",
            ):
                self.assertNotIn(forbidden, public_text)
            private_text = json.dumps(ledger[1]["private_artifacts"], sort_keys=True)
            self.assertIn("outputs/runner/push-round-002-train/checkpoints/best.pt", private_text)

    def test_push_round_eval_cli_uses_runner_sealed_checkpoint(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        runner = CapturingPushDryRunRunner()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = CandidatePackageRef(
                candidate_id="reward-code",
                path=str(
                    self._write_reward_candidate(
                        Path(tmp),
                        candidate_id="reward-code",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            run_candidate_round(
                RoundInputs(
                    round_id="push-round-003",
                    task_plugin=plugin,
                    backend_adapter=backend,
                    runner=runner,
                    verifier=PassingVerifier(),
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=Path(tmp) / "ledger.jsonl",
                    public_context={"previous_rounds": 0, "public_only": True},
                    **self._round_session_fields(
                        plugin=plugin,
                        backend=backend,
                        initial_best_score=0.10,
                    ),
                )
            )

        executed_eval_spec = runner.eval_specs[0]
        planned_checkpoint_uri = "artifact://push-round-003/checkpoints/best.pt"

        self.assertTrue(executed_eval_spec.checkpoint.sealed)
        cli_checkpoint_path = executed_eval_spec.checkpoint.metadata["local_path"]

        self.assertIn(cli_checkpoint_path, executed_eval_spec.metadata["cli_argv"])
        self.assertIn(cli_checkpoint_path, executed_eval_spec.metadata["cli_command"])
        argv = executed_eval_spec.metadata["cli_argv"]
        self.assertEqual(argv[argv.index("--checkpoint-sha256") + 1], executed_eval_spec.checkpoint.checksum)
        self.assertIn(f"--checkpoint-sha256 {executed_eval_spec.checkpoint.checksum}", executed_eval_spec.metadata["cli_command"])
        self.assertNotIn(executed_eval_spec.checkpoint.uri, executed_eval_spec.metadata["cli_argv"])
        self.assertNotIn(executed_eval_spec.checkpoint.uri, executed_eval_spec.metadata["cli_command"])
        self.assertNotIn(planned_checkpoint_uri, executed_eval_spec.metadata["cli_argv"])
        self.assertNotIn(planned_checkpoint_uri, executed_eval_spec.metadata["cli_command"])

    def test_train_success_with_missing_eval_records_eval_incomplete_without_accept(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = CandidatePackageRef(
                candidate_id="reward-code",
                path=str(
                    self._write_reward_candidate(
                        Path(tmp),
                        candidate_id="reward-code",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            ledger_path = Path(tmp) / "ledger.jsonl"
            result = run_candidate_round(
                RoundInputs(
                    round_id="push-round-eval-missing",
                    task_plugin=plugin,
                    backend_adapter=backend,
                    runner=EvalIncompletePushDryRunRunner(),
                    verifier=PassingVerifier(),
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **self._round_session_fields(
                        plugin=plugin,
                        backend=backend,
                        initial_best_score=0.10,
                    ),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertIs(result.decision.outcome, DecisionOutcome.REJECT)
        self.assertEqual(result.decision.reason, "eval_incomplete")
        self.assertIsNone(result.public_result)
        self.assertIsNotNone(result.train_result)
        self.assertEqual(ledger[0]["decision"]["reason"], "eval_incomplete")
        self.assertFalse(ledger[0]["decision"]["accepted"])
        self.assertIsNone(ledger[0]["public_result"])
        self.assertEqual(ledger[0]["train_result"]["checkpoint"]["role"], "eval")
        self.assertIn("[path]", ledger[0]["runner_error"]["message"])
        self.assertNotIn("D:/runner", json.dumps(ledger[0]["runner_error"], sort_keys=True))

    def test_unsafe_eval_public_projection_fails_before_decision_and_records_eval_incomplete(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = CandidatePackageRef(
                candidate_id="reward-code",
                path=str(
                    self._write_reward_candidate(
                        Path(tmp),
                        candidate_id="reward-code",
                    )
                ),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            ledger_path = Path(tmp) / "ledger.jsonl"
            result = run_candidate_round(
                RoundInputs(
                    round_id="push-round-unsafe-eval-projection",
                    task_plugin=plugin,
                    backend_adapter=backend,
                    runner=UnsafeEvalProjectionPushDryRunRunner(),
                    verifier=PassingVerifier(),
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=ledger_path,
                    public_context={"previous_rounds": 0, "public_only": True},
                    **self._round_session_fields(
                        plugin=plugin,
                        backend=backend,
                        initial_best_score=0.0,
                    ),
                )
            )
            ledger = load_ledger(ledger_path)

        self.assertIs(result.decision.outcome, DecisionOutcome.REJECT)
        self.assertEqual(result.decision.reason, "eval_incomplete")
        self.assertIsNone(result.public_result)
        self.assertEqual(ledger[0]["decision"]["reason"], "eval_incomplete")
        self.assertFalse(ledger[0]["decision"]["accepted"])
        self.assertIsNone(ledger[0]["public_result"])
        self.assertEqual(ledger[0]["runner_error"]["type"], "ValueError")
        text = json.dumps(ledger[0], sort_keys=True)
        self.assertNotIn("POLICY_IMPROVEMENT_ACCEPT", text)
        self.assertNotIn("D:/runner/eval/summary.json", text)

    def test_round_uses_verifier_derived_change_set_over_candidate_claims(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()

        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_schedule_only_candidate(Path(tmp))
            candidate = CandidatePackageRef(
                candidate_id="schedule_only",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            result = run_candidate_round(
                RoundInputs(
                    round_id="push-round-derived-change-set",
                    task_plugin=plugin,
                    backend_adapter=backend,
                    runner=PushDryRunRunner(),
                    verifier=StaticCandidateVerifier(),
                    candidate=candidate,
                    policy_initialization=PolicyInitialization.random(),
                    ledger_path=Path(tmp) / "ledger.jsonl",
                    public_context={"previous_rounds": 0, "public_only": True},
                    **self._round_session_fields(
                        plugin=plugin,
                        backend=backend,
                        initial_best_score=0.10,
                    ),
                )
            )
            ledger = load_ledger(Path(tmp) / "ledger.jsonl")

        self.assertIs(result.decision.outcome, DecisionOutcome.REJECT)
        self.assertEqual(ledger[0]["candidate_surface"]["classification"], "parameter_only")
        self.assertEqual(ledger[0]["candidate"]["changed_components"], ["reward_weight"])

    def test_isaac_trusted_eval_summary_records_no_candidate_code_loaded(self) -> None:
        source = (PROJECT_ROOT / "scripts" / "isaaclab_rsl_rl_eval.py").read_text(encoding="utf-8")

        self.assertIn('"candidate_code_loaded_in_eval": False', source)
        self.assertIn('"candidate_package_loaded": False', source)
        self.assertIn("--subject-kind", source)
        self.assertIn("--subject-id", source)
        self.assertIn("reject_candidate_path_arguments", source)
        self.assertIn("ExperimentSubject", source)
        self.assertNotIn("--candidate-id", source)
        self.assertIn('"candidate_package_loaded_in_eval": False', source)
        self.assertIn('"sealed_validation_gate":', source)
        self.assertIn('"checkpoint_safety_report": checkpoint_safety_report', source)
        self.assertIn('"checkpoint_safety_status": checkpoint_safety_report["status"]', source)
        self.assertIn('"evidence_manifest": build_eval_evidence_manifest(checkpoint_safety_report)', source)
        self.assertIn("--checkpoint-sha256", source)
        report_index = source.find("build_checkpoint_safety_report(")
        load_index = source.find("runner.load(args_cli.checkpoint)")
        self.assertGreaterEqual(report_index, 0)
        self.assertGreaterEqual(load_index, 0)
        self.assertLess(report_index, load_index)
        self.assertNotIn('parser.add_argument("--candidate", required=True)', source)
        self.assertIn("build_push_public_result_from_summaries", source)
        self.assertIn("--public-accept-seeds", source)

    def _eval_summary(self, *, seed: int, success_rate: float, action_l2: float = 0.4) -> dict:
        success_count = int(round(success_rate * 8))
        return {
            "schema_version": "true_rl_agent.push_recovery.eval_summary.v1",
            "status": "pass",
            "run": {"seed": seed},
            "aggregate": {
                "episodes": 8,
                "success_count": success_count,
                "success_rate": success_rate,
                "failure_breakdown": {
                    "total": 8,
                    "counts": {"success": success_count, "fall_forward": 8 - success_count},
                    "rates": {"success": success_rate, "fall_forward": 1.0 - success_rate},
                },
                "condition_flag_breakdown": {
                    "total": 8,
                    "counts": {"terminated": 8 - success_count},
                    "rates": {"terminated": 1.0 - success_rate},
                },
            },
            "secondary_metrics": {
                "mean_episode_steps": 512.0,
                "mean_return": 1.0,
                "mean_action_l2": action_l2,
                "mean_action_rate_l2": action_l2 * 0.5,
                "mean_min_base_height_m": 0.78,
                "mean_max_abs_pitch_rad": 0.08,
                "mean_max_abs_roll_rad": 0.05,
                "mean_final_xy_drift_m": 0.15,
                "mean_final_planar_speed_mps": 0.12,
                "recovery_window_success_rate": 1.0,
            },
        }

    def _budget_summary(self) -> dict:
        return {
            "status": "pass",
            "budget_accounting": {
                "status": "pass",
                "budget_steps": 49_152_000,
                "planned_env_steps": 49_152_000,
                "config_derived_total_timesteps": 49_152_000,
                "num_envs": 4096,
                "num_steps_per_env": 24,
                "max_iterations": 500,
            },
        }


if __name__ == "__main__":
    unittest.main()
