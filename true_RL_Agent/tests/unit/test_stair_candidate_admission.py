from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import textwrap
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.isaaclab_stair_rsl_rl_train import (  # noqa: E402
    _build_parser as build_train_parser,
)
from scripts.isaaclab_stair_rsl_rl_train import (  # noqa: E402
    _load_and_qualify_program,
)

from true_rl_agent.backends.isaac_lab_rsl_rl import (  # noqa: E402
    IsaacLabRslRlBackendAdapter,
)
from true_rl_agent.candidates.artifact import seal_candidate_package  # noqa: E402
from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import (
    StaticCandidateVerifier,
    StaticCandidateVerifierPolicy,
    seal_and_verify_candidate,
    static_verifier_source_policy_hash,
)
from true_rl_agent.composition import (
    TaskCompositionContext,
    TaskCompositionError,
    resolve_task_composition,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import canonical_json_text
from true_rl_agent.runner.job_argv import validate_train_job_argv
from true_rl_agent.tasks.stair_climbing.candidate_admission import (
    StairCandidateAdmissionPolicy,
    _qualification_failure_evidence,
    _validate_qualification_worker_result,
)
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (
    validate_stair_train_invocation,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    StairRewardArtifactIntegrityError,
    StairRewardProgramError,
    qualify_stair_candidate_reward_program,
)

PROTOCOL_PATH = PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
OFFICIAL_R0_SOURCE_PATH = (
    PROJECT_ROOT
    / "src"
    / "true_rl_agent"
    / "tasks"
    / "stair_climbing"
    / "reference_programs"
    / "official_g1_rough_v1"
    / "training_code"
    / "components.py"
)
TEST_INITIAL_POLICY_STATE_HASH = "1" * 64


def _write_candidate_package(
    root: Path,
    *,
    candidate_id: str,
    source: str,
) -> Path:
    candidate_dir = root / candidate_id
    (candidate_dir / "training_code").mkdir(parents=True)
    (candidate_dir / "candidate_manifest.yaml").write_text(
        "\n".join(
            (
                "schema_version: true_rl_agent.candidate.v1",
                f"candidate_id: {candidate_id}",
                "entrypoint: training_code.components:build_training_components",
                "",
            )
        ),
        encoding="utf-8",
    )
    (candidate_dir / "training_code" / "components.py").write_text(
        source,
        encoding="utf-8",
    )
    (candidate_dir / "rationale.md").write_text("Stair candidate.\n", encoding="utf-8")
    (candidate_dir / "operation_diff.json").write_text("{}\n", encoding="utf-8")
    (candidate_dir / "diagnostics.json").write_text("{}\n", encoding="utf-8")
    return candidate_dir


def _official_source_with_raw_contact_term(*, clip_clause: str) -> str:
    source = OFFICIAL_R0_SOURCE_PATH.read_text(encoding="utf-8")
    source = source.replace(
        "def termination_penalty(inputs):",
        "def base_contact_force_penalty(inputs):\n"
        "    return (inputs.base_contact_force.norm(dim=-1) - 1.0).clamp(min=0.0)\n\n\n"
        "def termination_penalty(inputs):",
        1,
    )
    marker = "        reward_terms=[\n"
    replacement = (
        marker
        + "            RewardTerm(\n"
        + '                name="base_contact_force_penalty",\n'
        + "                weight=-0.01,\n"
        + "                fn=base_contact_force_penalty"
        + clip_clause
        + ",\n"
        + "            ),\n"
    )
    return source.replace(marker, replacement, 1)


class StairCandidateAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = StairClimbingTaskConfig.defaults(PROTOCOL_PATH)
        cls.reference = resolve_official_stair_reference_program(
            cls.config,
            project_root=PROJECT_ROOT,
        ).artifact

    def test_policy_hash_binds_actual_static_verifier_instance(self) -> None:
        short = StairCandidateAdmissionPolicy(
            self.config,
            static_verifier=StaticCandidateVerifier(
                component_load_timeout_s=0.25
            ),
        )
        long = StairCandidateAdmissionPolicy(
            self.config,
            static_verifier=StaticCandidateVerifier(
                component_load_timeout_s=120.0
            ),
        )

        self.assertNotEqual(short.policy_hash, long.policy_hash)

    def test_worker_startup_budget_is_not_candidate_policy(self) -> None:
        short = StairCandidateAdmissionPolicy(
            self.config,
            worker_startup_timeout_s=1.0,
        )
        long = StairCandidateAdmissionPolicy(
            self.config,
            worker_startup_timeout_s=120.0,
        )

        self.assertEqual(short.policy_hash, long.policy_hash)

    def test_candidate_execution_budget_is_part_of_policy(self) -> None:
        short = StairCandidateAdmissionPolicy(
            self.config,
            qualification_timeout_s=1.0,
        )
        long = StairCandidateAdmissionPolicy(
            self.config,
            qualification_timeout_s=8.0,
        )

        self.assertNotEqual(short.policy_hash, long.policy_hash)

    def test_hidden_joint_order_is_rejected_before_and_after_sealing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidate"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "candidate_manifest.yaml").write_text(
                textwrap.dedent(
                    """
                    schema_version: true_rl_agent.candidate.v1
                    candidate_id: invented-leg-order
                    entrypoint: training_code.components:build_training_components
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            source = OFFICIAL_R0_SOURCE_PATH.read_text(encoding="utf-8")
            source = source.replace(
                "return inputs.hip_yaw_roll_joint_pos_error.abs().sum(dim=1)",
                "return inputs.joint_pos_error[:, :12].abs().sum(dim=1)",
                1,
            )
            (candidate_dir / "training_code" / "components.py").write_text(
                source,
                encoding="utf-8",
            )
            (candidate_dir / "rationale.md").write_text("Stair candidate.\n", encoding="utf-8")
            (candidate_dir / "operation_diff.json").write_text("{}\n", encoding="utf-8")
            (candidate_dir / "diagnostics.json").write_text("{}\n", encoding="utf-8")
            candidate = CandidatePackageRef(
                candidate_id="invented-leg-order",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(
                    frozenset({"reward_callable"})
                ),
            )
            admission = StairCandidateAdmissionPolicy(self.config)

            mutable_result = admission.verify(candidate)
            self.assertEqual(mutable_result.status, "fail")
            self.assertEqual(
                mutable_result.reason,
                "stair_candidate_source_semantics_invalid",
            )
            self.assertTrue(
                any(
                    issue.startswith(
                        "opaque_input_axis_selection:joint_pos_error:"
                    )
                    for issue in mutable_result.details["issues"]
                )
            )

            sealed = seal_candidate_package(
                candidate,
                artifact_root=root / "artifacts",
            )
            with self.assertRaisesRegex(
                StairRewardProgramError,
                "opaque_input_axis_selection:joint_pos_error",
            ):
                qualify_stair_candidate_reward_program(
                    artifact=sealed,
                    contract=self.config.protocol_v2.complete_reward_runtime,
                    static_verifier_policy=admission.static_verifier.policy,
                    trusted_reference_artifact=self.reference,
                )

    def test_raw_physical_term_is_repairable_before_sealing_and_rejected_after(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="raw-contact-unbounded",
                source=_official_source_with_raw_contact_term(clip_clause=""),
            )
            candidate = CandidatePackageRef(
                candidate_id="raw-contact-unbounded",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            admission = StairCandidateAdmissionPolicy(self.config)

            mutable = admission.verify(candidate)
            self.assertEqual(mutable.status, "fail")
            self.assertEqual(
                mutable.reason,
                "stair_candidate_source_semantics_invalid",
            )
            self.assertTrue(
                any(
                    issue.startswith(
                        "raw_physical_reward_requires_finite_two_sided_clip:"
                        "RewardTerm:base_contact_force_penalty"
                    )
                    for issue in mutable.details["issues"]
                )
            )

            sealed = seal_candidate_package(
                candidate,
                artifact_root=root / "artifacts",
            )
            with self.assertRaisesRegex(
                StairRewardProgramError,
                "raw_physical_reward_requires_finite_two_sided_clip",
            ):
                qualify_stair_candidate_reward_program(
                    artifact=sealed,
                    contract=self.config.protocol_v2.complete_reward_runtime,
                    static_verifier_policy=admission.static_verifier.policy,
                    trusted_reference_artifact=self.reference,
                )

    def test_raw_physical_term_with_final_clip_passes_mutable_admission(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="raw-contact-bounded",
                source=_official_source_with_raw_contact_term(
                    clip_clause=", clip_min=0.0, clip_max=100.0"
                ),
            )
            candidate = CandidatePackageRef(
                candidate_id="raw-contact-bounded",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            self.assertEqual(
                StairCandidateAdmissionPolicy(self.config).verify(candidate).status,
                "pass",
            )

    def test_qualification_failure_ownership_is_narrow_and_fail_closed(self) -> None:
        candidate = _qualification_failure_evidence(
            StairRewardProgramError("candidate reward is invalid")
        )
        self.assertEqual(candidate["failure_owner"], "candidate")
        self.assertEqual(
            _validate_qualification_worker_result(candidate),
            candidate,
        )

        infrastructure_errors = (
            StairRewardArtifactIntegrityError("artifact drift"),
            OSError("filesystem failed"),
            MemoryError("allocation failed"),
            ValueError("unclassified failure"),
        )
        for error in infrastructure_errors:
            with self.subTest(error_type=type(error).__name__):
                evidence = _qualification_failure_evidence(error)
                self.assertEqual(evidence["failure_owner"], "infrastructure")
                with self.assertRaisesRegex(
                    RuntimeError,
                    "qualification infrastructure failed",
                ):
                    _validate_qualification_worker_result(evidence)

        forged = {**candidate, "failure_owner": "unknown"}
        with self.assertRaisesRegex(RuntimeError, "failure owner is invalid"):
            _validate_qualification_worker_result(forged)

    def test_composition_rejects_execution_authorization_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = StairClimbingCompositionProvider(
                reference_program_artifact=self.reference
            )
            composition = resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=TaskCompositionContext(
                    project_root=PROJECT_ROOT,
                    output_dir=Path(tmp) / "output",
                ),
            )
            drifted_backend = IsaacLabRslRlBackendAdapter(
                replace(
                    composition.backend_adapter.config,
                    candidate_execution_authorization_schema=(
                        "test.execution_authorization.drift.v1"
                    ),
                )
            )

            with self.assertRaisesRegex(
                TaskCompositionError,
                "authorization schema does not match backend",
            ):
                replace(composition, backend_adapter=drifted_backend)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "production Stair qualification requires Torch",
    )
    def test_sealed_qualification_uses_actual_static_source_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = root / "candidate"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "candidate_manifest.yaml").write_text(
                textwrap.dedent(
                    """
                    schema_version: true_rl_agent.candidate.v1
                    candidate_id: custom-source-policy
                    entrypoint: training_code.components:build_training_components
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            source = OFFICIAL_R0_SOURCE_PATH.read_text(encoding="utf-8")
            source = source.replace(
                "def termination_penalty(inputs):\n",
                "def termination_penalty(inputs):\n"
                "    print('custom-source-policy')\n",
                1,
            )
            (candidate_dir / "training_code" / "components.py").write_text(
                source,
                encoding="utf-8",
            )
            candidate = CandidatePackageRef(
                candidate_id="custom-source-policy",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(
                    frozenset({"reward_callable"})
                ),
            )
            default_policy = StaticCandidateVerifierPolicy()
            custom_policy = replace(
                default_policy,
                forbidden_calls=default_policy.forbidden_calls - {"print"},
            )
            admission = StairCandidateAdmissionPolicy(
                self.config,
                static_verifier=StaticCandidateVerifier(policy=custom_policy),
            )

            self.assertNotIn(
                "print",
                admission.authoring_prompt_rules()["blocked_calls"],
            )
            self.assertEqual(admission.verify(candidate).status, "pass")
            verification = seal_and_verify_candidate(
                candidate,
                artifact_root=root / "artifacts",
                verifier=admission,
            )
            artifact = verification.require_verified()
            task_evidence = verification.result.details["task_qualification"]
            qualification = task_evidence["qualification"]
            expected_source_policy_hash = static_verifier_source_policy_hash(
                policy=custom_policy,
            )

            self.assertEqual(
                qualification["static_verifier_policy_hash"],
                expected_source_policy_hash,
            )
            self.assertNotEqual(
                expected_source_policy_hash,
                static_verifier_source_policy_hash(policy=default_policy),
            )

            provider = StairClimbingCompositionProvider(
                reference_program_artifact=self.reference
            )
            composition = resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=TaskCompositionContext(
                    project_root=PROJECT_ROOT,
                    output_dir=root / "output",
                ),
            )
            custom_backend = IsaacLabRslRlBackendAdapter(
                replace(
                    composition.backend_adapter.config,
                    candidate_admission_policy_id=(
                        admission.admission_policy_descriptor().policy_id
                    ),
                    candidate_admission_policy_hash=(
                        admission.admission_policy_descriptor().policy_hash
                    ),
                )
            )
            jobs = custom_backend.materialize_jobs(
                round_id="custom-source-policy",
                task_spec=composition.task_plugin.task_spec,
                candidate=artifact,
                policy_initialization=composition.policy_initialization,
                evaluation_binding=custom_backend.evaluation_binding_for(
                    composition.task_plugin.task_spec
                ),
                expected_initial_policy_state_hash=(
                    TEST_INITIAL_POLICY_STATE_HASH
                ),
            )
            train_argv = jobs.train_spec.metadata["cli_argv"]
            validate_train_job_argv(jobs.train_spec, train_argv)
            train_args = build_train_parser().parse_args(train_argv[2:])
            context = validate_stair_train_invocation(
                train_args,
                project_root=PROJECT_ROOT,
            )
            program = _load_and_qualify_program(context)

            self.assertEqual(
                program.static_verifier_policy_hash,
                expected_source_policy_hash,
            )
            self.assertEqual(
                program.qualification_hash,
                qualification["qualification_hash"],
            )

            authorization_index = train_argv.index(
                "--candidate-execution-authorization-json"
            ) + 1
            envelope = json.loads(train_argv[authorization_index])
            envelope["payload"]["qualification_hash"] = "f" * 64
            stale_hash_argv = list(train_argv[2:])
            stale_hash_argv[
                stale_hash_argv.index(
                    "--candidate-execution-authorization-json"
                )
                + 1
            ] = canonical_json_text(envelope)
            stale_hash_args = build_train_parser().parse_args(stale_hash_argv)
            with self.assertRaisesRegex(ValueError, "authorization hash mismatch"):
                validate_stair_train_invocation(
                    stale_hash_args,
                    project_root=PROJECT_ROOT,
                )

            drifted_policy = replace(
                custom_policy,
                component_load_timeout_s=(
                    custom_policy.component_load_timeout_s + 1.0
                ),
            )
            envelope = json.loads(train_argv[authorization_index])
            envelope["payload"]["static_verifier_policy_snapshot"] = (
                drifted_policy.to_snapshot_payload()
            )
            drifted_json = canonical_json_text(envelope)
            drifted_argv = list(train_argv[2:])
            drifted_argv[
                drifted_argv.index(
                    "--candidate-execution-authorization-json"
                )
                + 1
            ] = drifted_json
            drifted_argv[
                drifted_argv.index(
                    "--candidate-execution-authorization-hash"
                )
                + 1
            ] = canonical_json_hash(envelope)
            drifted_args = build_train_parser().parse_args(drifted_argv)
            with self.assertRaisesRegex(ValueError, "admission identity mismatch"):
                validate_stair_train_invocation(
                    drifted_args,
                    project_root=PROJECT_ROOT,
                )

            tampered_evidence = {
                **task_evidence,
                "qualification": {
                    **qualification,
                    "static_verifier_policy_hash": (
                        static_verifier_source_policy_hash(
                            policy=default_policy,
                        )
                    ),
                },
            }
            with patch(
                "true_rl_agent.tasks.stair_climbing.candidate_admission."
                "_run_qualification_worker",
                return_value=tampered_evidence,
            ), self.assertRaisesRegex(
                RuntimeError,
                "qualification identity mismatch",
            ):
                admission.verify(artifact.sealed_artifact)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "production Stair qualification requires Torch",
    )
    def test_cold_official_r0_passes_admission_and_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = StairClimbingCompositionProvider(
                reference_program_artifact=self.reference
            )
            composition = resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=TaskCompositionContext(
                    project_root=PROJECT_ROOT,
                    output_dir=root / "output",
                ),
            )
            candidate_dir = root / "candidate"
            (candidate_dir / "training_code").mkdir(parents=True)
            (candidate_dir / "candidate_manifest.yaml").write_text(
                textwrap.dedent(
                    """
                    schema_version: true_rl_agent.candidate.v1
                    candidate_id: official-r0-as-candidate
                    entrypoint: training_code.components:build_training_components
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            (candidate_dir / "training_code" / "components.py").write_bytes(
                OFFICIAL_R0_SOURCE_PATH.read_bytes()
            )
            candidate = CandidatePackageRef(
                candidate_id="official-r0-as-candidate",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(
                    frozenset({"reward_callable"})
                ),
            )

            verification = seal_and_verify_candidate(
                candidate,
                artifact_root=root / "artifacts",
                verifier=composition.candidate_admission_policy,
            )
            artifact = verification.require_verified()
            binding = composition.backend_adapter.evaluation_binding_for(
                composition.task_plugin.task_spec
            )
            jobs = composition.backend_adapter.materialize_jobs(
                round_id="official-r0-as-candidate",
                task_spec=composition.task_plugin.task_spec,
                candidate=artifact,
                policy_initialization=composition.policy_initialization,
                evaluation_binding=binding,
                expected_initial_policy_state_hash=(
                    TEST_INITIAL_POLICY_STATE_HASH
                ),
            )

            self.assertEqual(
                verification.result.reason,
                "stair_candidate_admission_passed",
            )
            self.assertEqual(
                artifact.verification_policy_hash,
                composition.candidate_admission_policy
                .admission_policy_descriptor()
                .policy_hash,
            )
            self.assertIn(
                COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
                verification.result.details["capability_ids"],
            )
            self.assertEqual(
                jobs.train_spec.candidate_artifact,
                artifact,
            )
            validate_train_job_argv(
                jobs.train_spec,
                jobs.train_spec.metadata["cli_argv"],
            )


if __name__ == "__main__":
    unittest.main()
