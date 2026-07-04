from __future__ import annotations

import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.decision import DecisionOutcome
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.round import RoundInputs, run_candidate_round
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter, IsaacLabRslRlBackendConfig
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
from true_rl_agent.candidates.verifier import CandidateVerifier, StaticCandidateVerifier, VerificationResult
from true_rl_agent.memory import ExperimentRecord, PublicContextBuilder
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.runner.checkpoint import seal_checkpoint
from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, TrainJobSpec
from true_rl_agent.tasks.base import TaskInvariantSpec
from true_rl_agent.tasks.compatibility import check_task_backend_compatibility
from true_rl_agent.tasks.push_recovery import (
    PushRecoveryTaskPlugin,
    PushTaskMemoryAdapter,
    build_push_public_result_from_summaries,
    build_push_public_result,
    classify_push_failure,
)
from true_rl_agent.tasks.push_recovery.metrics import extract_push_failure_breakdown


class PassingVerifier(CandidateVerifier):
    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        _ = candidate
        return VerificationResult(status="pass", reason="test")


class PushDryRunRunner(Runner):
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        raw_checkpoint = CheckpointRef(
            uri=f"artifact://{train_spec.job_id}/best.pt",
            checksum=f"sha256:{train_spec.candidate_id}",
            role="train_output",
            metadata={"local_path": f"outputs/runner/{train_spec.job_id}/checkpoints/best.pt"},
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
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
        _ = eval_spec
        return build_push_public_result(
            success_rate=0.30,
            failure_breakdown={"fall": 0.1, "timeout": 0.05},
            motion_quality={"mean_base_height": 0.78, "mean_tilt_rad": 0.05},
        )


class CapturingPushDryRunRunner(PushDryRunRunner):
    def __init__(self) -> None:
        self.eval_specs: list[EvalJobSpec] = []

    def run_eval(self, eval_spec: EvalJobSpec):
        self.eval_specs.append(eval_spec)
        return super().run_eval(eval_spec)


class PushRecoveryTaskTests(unittest.TestCase):
    def _backend(self) -> IsaacLabRslRlBackendAdapter:
        return IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(
                isaac_lab_python="/opt/isaac/python.sh",
                train_entrypoint="train_push.py",
                eval_entrypoint="eval_push.py",
                default_budget_steps=49_152_000,
                train_num_envs=4096,
                train_max_iterations=500,
                train_steps_per_iteration=24,
                public_seed_count=3,
                trusted_seed_count=7,
            )
        )

    def _write_schedule_only_candidate(self, root: Path) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: schedule_only
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
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

    def test_push_plugin_declares_fixed_invariants_and_no_backend_authority(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        task_spec = plugin.task_spec

        self.assertEqual(task_spec.task_id, "unitree_push_recovery")
        self.assertTrue(task_spec.supports_policy_initialization("random"))
        for key in (
            "fall_threshold",
            "episode_length",
            "push_magnitude_xy",
            "push_timing",
            "success_predicate",
            "eval_seed_visibility",
            "eval_num_envs",
            "public_seed_count",
            "public_accept_seed_count",
            "budget_steps",
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
        self.assertEqual(task_spec.fixed_eval_protocol["public_accept_seed_count"], 1)
        self.assertEqual(task_spec.fixed_eval_protocol["eval_num_envs"], 64)
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
        self.assertEqual(sections["task"]["fixed_eval_protocol_summary"]["seed_visibility"], "counts_only_no_raw_seed_ids")
        self.assertEqual(sections["public_eval_contract"]["public_score"]["direction"], "higher_is_better")
        self.assertEqual(sections["public_eval_contract"]["runner_decision_authority"], "runner_owned_read_only")
        self.assertIn("success_rate", sections["public_eval_contract"]["metrics"])
        reward_contract = sections["reward_inputs_contract"]
        self.assertIn("push_active", reward_contract["fields"])
        self.assertIn("push_vec_public", reward_contract["fields"])
        self.assertEqual(reward_contract["fields"]["base_height"]["shape"], ["num_envs"])
        self.assertIn("zero fallback", reward_contract["fields"]["push_active"]["known_limitations"].lower())
        self.assertIn("not guaranteed", reward_contract["fields"]["push_vec_public"]["candidate_usage_guidance"].lower())
        self.assertGreaterEqual(len(sections["known_good_examples"]), 3)
        self.assertTrue(any(example["kind"] == "RewardTerm" for example in sections["known_good_examples"]))
        self.assertTrue(any(example["kind"] == "DiagnosticTerm" for example in sections["known_good_examples"]))
        self.assertIn("candidate_must_not_change", sections["task"])
        self.assertNotIn("trusted_seed", text)
        self.assertNotIn("checkpoint_selector", text)
        self.assertNotIn("stage_list", text)
        self.assertNotIn("D:/", text)
        self.assertNotIn("/data/", text)
        assert_public_only_prompt_payload(sections)

    def test_push_full_prompt_payload_uses_task_owned_reward_inputs_contract(self) -> None:
        payload = PublicContextBuilder(task_prompt_adapter=PushRecoveryTaskPlugin().prompt_adapter()).build_prompt_payload([])

        self.assertEqual(payload["task"]["task_id"], "unitree_push_recovery")
        self.assertNotIn("task_prompt_sections", payload["public_context"])
        self.assertIn("push_active", payload["reward_inputs_contract"]["fields"])
        self.assertIn("push_vec_public", payload["reward_inputs_contract"]["fields"])
        self.assertEqual(payload["reward_inputs"]["status"], "legacy_alias_deprecated")
        self.assertEqual(payload["reward_inputs"]["canonical_section"], "reward_inputs_contract")
        self.assertEqual(payload["reward_inputs"]["fields"], [])
        self.assertEqual(payload["public_eval_contract"]["public_score"]["direction"], "higher_is_better")
        self.assertEqual(payload["runtime_feedback_summary"]["status"], "no_prior_runtime_feedback")
        assert_public_only_prompt_payload(payload)

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
                "mechanism_attribution": {"status": "pass", "reason": "runner_activation_observed"},
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
            public_records=[record.ledger_public_record()],
            public_context={"previous_rounds": 1, "public_only": True},
        )

        feedback = sections["public_feedback_summary"]
        runtime = sections["runtime_feedback_summary"]
        self.assertEqual(feedback["latest_round"]["round_id"], "push-round-001")
        self.assertEqual(feedback["latest_round"]["decision_reason"], "score_delta_below_threshold")
        self.assertEqual(feedback["latest_public_score"], 0.30)
        self.assertEqual(feedback["latest_failure_mode"], "fall_dominated")
        self.assertEqual(runtime["latest_runtime_activation"]["reward_term_calls"], 4)
        self.assertTrue(runtime["latest_runtime_activation"]["nonzero_reward_contribution"])
        self.assertEqual(runtime["mechanism_attribution"]["status"], "pass")
        self.assertTrue(runtime["candidate_package_loaded"])
        self.assertFalse(runtime["candidate_code_loaded_in_eval"])
        assert_public_only_prompt_payload(sections)

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
        self.assertEqual(sections["public_feedback_summary"]["latest_failure_mode"], "fall_dominated")
        self.assertEqual(
            sections["public_feedback_summary"]["task_public_summary"]["latest_failure_breakdown"]["fall_forward"],
            0.75,
        )

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
            ],
            budget_summary=self._budget_summary(),
        )

        self.assertEqual(result.status, "pass")
        self.assertEqual(result.public_score, 0.75)
        self.assertEqual(result.metrics["public_dev"]["success_rate_stats"]["count"], 2)
        self.assertEqual(result.metrics["public_dev"]["success_rate_stats"]["mean"], 0.625)
        self.assertEqual(result.metrics["public_accept"]["success_rate_stats"]["mean"], 0.75)
        self.assertEqual(result.metrics["budget_accounting"]["status"], "pass")
        self.assertEqual(result.metrics["motion_quality_gate"]["status"], "pass")
        self.assertEqual(result.metrics["acceptance"]["score_source"], "public_accept.success_rate_stats.mean")
        self.assertEqual(
            result.metrics["acceptance"]["required_pass_gates"],
            ["budget_accounting", "motion_quality_gate"],
        )
        self.assertNotIn("1001", str(result.to_public_dict()))
        self.assertNotIn("2001", str(result.to_public_dict()))

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
        self.assertFalse(hasattr(plugin, "decide"))
        self.assertFalse(hasattr(plugin, "accept"))
        self.assertFalse(hasattr(plugin, "reject"))

    def test_push_invariants_reject_task_weakening(self) -> None:
        invariants = PushRecoveryTaskPlugin().task_spec.invariants
        weakened = TaskInvariantSpec({**dict(invariants.values), "episode_length": 128})

        with self.assertRaises(ValueError):
            invariants.assert_compatible_with(weakened)

    def test_push_task_materializes_isaac_jobs_with_fixed_protocol_metadata(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        candidate = CandidatePackageRef(
            candidate_id="push-code-candidate",
            path="memory://push-code-candidate",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        )

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
        self.assertEqual(bundle.eval_spec.metadata["fixed_eval_protocol"]["eval_protocol"], "fixed_public_and_trusted")
        self.assertIn("--candidate memory://push-code-candidate", bundle.train_spec.metadata["cli_command"])
        self.assertEqual(bundle.train_spec.metadata["cli_argv"][:2], ["/opt/isaac/python.sh", "train_push.py"])
        self.assertIn("memory://push-code-candidate", bundle.train_spec.metadata["cli_argv"])
        self.assertEqual(bundle.eval_spec.metadata["cli_argv"][:2], ["/opt/isaac/python.sh", "eval_push.py"])
        self.assertIn("artifact://push-round-001/checkpoints/best.pt", bundle.eval_spec.metadata["cli_argv"])

    def test_push_round_rejects_schedule_only_and_accepts_code_with_runtime_evidence(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()

        with tempfile.TemporaryDirectory() as tmp:
            ledger_path = Path(tmp) / "ledger.jsonl"
            schedule_only = CandidatePackageRef(
                candidate_id="schedule-only",
                path="memory://schedule-only",
                change_set=CandidateChangeSet(frozenset({"reward_weight_scale"})),
            )
            code_level = CandidatePackageRef(
                candidate_id="reward-code",
                path="memory://reward-code",
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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
                )
            )
            ledger = load_ledger(ledger_path)

            self.assertIs(schedule_result.decision.outcome, DecisionOutcome.REJECT)
            self.assertEqual(ledger[0]["decision"]["reason"], "parameter_only_candidate")
            self.assertIs(code_result.decision.outcome, DecisionOutcome.POLICY_IMPROVEMENT_ACCEPT)
            self.assertEqual(ledger[1]["public_result"]["metrics"]["success_rate"], 0.30)
            self.assertEqual(ledger[1]["train_result"]["checkpoint"]["role"], "eval")
            self.assertTrue(ledger[1]["train_result"]["checkpoint"]["sealed"])

    def test_push_round_eval_cli_uses_runner_sealed_checkpoint(self) -> None:
        plugin = PushRecoveryTaskPlugin()
        backend = self._backend()
        runner = CapturingPushDryRunRunner()
        candidate = CandidatePackageRef(
            candidate_id="reward-code",
            path="memory://reward-code",
            change_set=CandidateChangeSet(frozenset({"reward_callable"})),
        )

        with tempfile.TemporaryDirectory() as tmp:
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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
                )
            )

        executed_eval_spec = runner.eval_specs[0]
        planned_checkpoint_uri = "artifact://push-round-003/checkpoints/best.pt"

        self.assertTrue(executed_eval_spec.checkpoint.sealed)
        cli_checkpoint_path = executed_eval_spec.checkpoint.metadata["local_path"]

        self.assertIn(cli_checkpoint_path, executed_eval_spec.metadata["cli_argv"])
        self.assertIn(cli_checkpoint_path, executed_eval_spec.metadata["cli_command"])
        self.assertNotIn(executed_eval_spec.checkpoint.uri, executed_eval_spec.metadata["cli_argv"])
        self.assertNotIn(executed_eval_spec.checkpoint.uri, executed_eval_spec.metadata["cli_command"])
        self.assertNotIn(planned_checkpoint_uri, executed_eval_spec.metadata["cli_argv"])
        self.assertNotIn(planned_checkpoint_uri, executed_eval_spec.metadata["cli_command"])

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
                    current_best_score=0.10,
                    policy_accept_delta=0.05,
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
