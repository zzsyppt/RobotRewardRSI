# ruff: noqa: E402
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

from tests.fakes.evaluation import make_test_candidate_package  # noqa: E402
from true_rl_agent.backends.isaac_lab_rsl_rl import (  # noqa: E402
    IsaacLabRslRlBackendAdapter,
)
from true_rl_agent.candidates.artifact import (  # noqa: E402
    CandidateExecutionAuthorization,
    seal_candidate_package,
)
from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.candidates.package_contract import CANDIDATE_PACKAGE_FILE_MAX_BYTES
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
from true_rl_agent.json_contract import canonical_json_text
from true_rl_agent.tasks.stair_climbing.candidate_admission import (
    StairCandidateAdmissionPolicy,
    _duplicate_parent_reward_term_issue,
    _qualification_failure_evidence,
    _validate_qualification_worker_result,
    validate_stair_candidate_execution_authorization,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.composition import (
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    StairRewardArtifactIntegrityError,
    StairRewardEnvironmentAxisError,
    StairRewardProgramError,
    assert_stair_reward_program_environment_independent,
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
def _write_candidate_package(
    root: Path,
    *,
    candidate_id: str,
    source: str,
) -> Path:
    candidate = make_test_candidate_package(
        candidate_id,
        provider_root=root / candidate_id,
        components_source=source,
    )
    return Path(candidate.path)


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


def _minimal_reward_source(*, expression: str) -> str:
    return textwrap.dedent(
        f'''\
        from __future__ import annotations

        from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


        def cross_environment_reward(inputs):
            return {expression}


        def build_training_components():
            return TrainingComponents(
                name="cross_environment_components",
                reward_terms=[
                    RewardTerm(
                        name="cross_environment_reward",
                        weight=1.0,
                        fn=cross_environment_reward,
                    ),
                ],
                rationale="Exercise the environment-axis admission contract.",
            )
        '''
    )


def _parent_alias_sources() -> tuple[str, str, str, str]:
    header = textwrap.dedent(
        '''\
        from __future__ import annotations

        from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


        def parent_reward(inputs):
            return inputs.base_height
        '''
    )
    parent = header + textwrap.dedent(
        '''\


        def build_training_components():
            return TrainingComponents(
                name="parent_components",
                reward_terms=[RewardTerm(name="parent_reward", weight=1.0, fn=parent_reward)],
                rationale="Parent reward.",
            )
        '''
    )
    alias = header + textwrap.dedent(
        '''\


        def alias_reward(inputs):
            return inputs.base_height


        def build_training_components():
            return TrainingComponents(
                name="alias_components",
                reward_terms=[
                    RewardTerm(name="parent_reward", weight=1.0, fn=parent_reward),
                    RewardTerm(name="alias_reward", weight=1.0, fn=alias_reward),
                ],
                rationale="Alias the parent reward.",
            )
        '''
    )
    renamed = parent.replace("parent_reward", "renamed_reward")
    novel = header + textwrap.dedent(
        '''\


        def novel_reward(inputs):
            return inputs.roll.square()


        def build_training_components():
            return TrainingComponents(
                name="novel_components",
                reward_terms=[
                    RewardTerm(name="parent_reward", weight=1.0, fn=parent_reward),
                    RewardTerm(name="novel_reward", weight=1.0, fn=novel_reward),
                ],
                rationale="Add a distinct reward.",
            )
        '''
    )
    return parent, alias, renamed, novel


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

    def test_duplicate_parent_reward_term_rejects_alias_only(self) -> None:
        parent, alias, renamed, novel = _parent_alias_sources()

        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = _write_candidate_package(
                Path(tmp),
                candidate_id="duplicate-parent-alias",
                source=alias,
            )
            candidate = CandidatePackageRef(
                candidate_id="duplicate-parent-alias",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            admission = StairCandidateAdmissionPolicy(self.config)
            static_result = admission.verify(candidate)
            self.assertEqual(static_result.status, "pass")
            with patch(
                "true_rl_agent.tasks.stair_climbing.candidate_admission."
                "seal_and_verify_candidate"
            ) as temporary_admission:
                duplicate = admission.verify_authoring_delta(
                    candidate=candidate,
                    verification=static_result,
                    parent_source=parent,
                )

        self.assertEqual(duplicate.status, "fail")
        self.assertEqual(duplicate.reason, "stair_candidate_authoring_delta_invalid")
        self.assertEqual(
            duplicate.details["issues"],
            ["duplicate_parent_reward_term:alias_reward:matches=parent_reward"],
        )
        temporary_admission.assert_not_called()
        self.assertIsNone(
            _duplicate_parent_reward_term_issue(
                candidate_source=renamed,
                parent_source=parent,
            )
        )
        self.assertIsNone(
            _duplicate_parent_reward_term_issue(
                candidate_source=novel,
                parent_source=parent,
            )
        )

    def test_authoring_delta_source_reread_keeps_package_byte_bound(self) -> None:
        parent, alias, _, _ = _parent_alias_sources()

        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = _write_candidate_package(
                Path(tmp),
                candidate_id="oversized-after-static-verify",
                source=alias,
            )
            candidate = CandidatePackageRef(
                candidate_id="oversized-after-static-verify",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            admission = StairCandidateAdmissionPolicy(self.config)
            static_result = admission.verify(candidate)
            self.assertEqual(static_result.status, "pass")
            source_path = candidate_dir / "training_code" / "components.py"
            maximum = CANDIDATE_PACKAGE_FILE_MAX_BYTES[
                "training_code/components.py"
            ]
            source_path.write_bytes(b"#" * (maximum + 1))

            with patch(
                "true_rl_agent.tasks.stair_climbing.candidate_admission."
                "seal_and_verify_candidate"
            ) as temporary_admission:
                result = admission.verify_authoring_delta(
                    candidate=candidate,
                    verification=static_result,
                    parent_source=parent,
                )

        self.assertEqual(result.status, "fail")
        self.assertEqual(
            result.reason,
            "stair_candidate_authoring_source_invalid",
        )
        self.assertEqual(
            result.details["issues"],
            ["authoring_source_identity_invalid:CandidatePackageContractError"],
        )
        temporary_admission.assert_not_called()

    def test_environment_axis_failure_is_repairable_during_authoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="cross-environment-authoring",
                source=_minimal_reward_source(
                    expression=(
                        "inputs.base_height * 0.0 + inputs.base_height[0]"
                    )
                ),
            )
            candidate = CandidatePackageRef(
                candidate_id="cross-environment-authoring",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            admission = StairCandidateAdmissionPolicy(self.config)
            static_result = admission.verify(candidate)
            self.assertEqual(static_result.status, "pass")
            failure = {
                "schema_version": (
                    "true_rl_agent.task.stair_climbing.candidate_qualification.v1"
                ),
                "status": "fail",
                "reason": "stair_candidate_reward_program_invalid",
                "failure_owner": "candidate",
                "error_type": StairRewardEnvironmentAxisError.__name__,
            }

            with patch(
                "true_rl_agent.tasks.stair_climbing.candidate_admission."
                "_run_qualification_worker",
                return_value=failure,
            ):
                result = admission.verify_authoring_delta(
                    candidate=candidate,
                    verification=static_result,
                    parent_source=None,
                )

        self.assertEqual(result.status, "fail")
        self.assertEqual(result.reason, "stair_candidate_environment_axis_invalid")
        self.assertEqual(
            result.details["issues"],
            ["reward_environment_axis_independence_failed"],
        )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "environment-axis qualification requires Torch",
    )
    def test_environment_axis_reverse_permutation_rejects_cross_row_anchor(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="cross-row-anchor",
                source=_minimal_reward_source(
                    expression=(
                        "inputs.base_height * 0.0 + inputs.base_height[0]"
                    )
                ),
            )
            candidate = CandidatePackageRef(
                candidate_id="cross-row-anchor",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            admission = StairCandidateAdmissionPolicy(self.config)
            sealed = seal_candidate_package(
                candidate,
                artifact_root=root / "artifacts",
            )
            program = qualify_stair_candidate_reward_program(
                artifact=sealed,
                contract=self.config.protocol_v2.complete_reward_runtime,
                static_verifier_policy=admission.static_verifier.policy,
                trusted_reference_artifact=self.reference,
            )
            adapter = StairClimbingComponentSmokeAdapter(self.config)

            with self.assertRaisesRegex(
                StairRewardEnvironmentAxisError,
                "reverse_permutation_failed",
            ):
                assert_stair_reward_program_environment_independent(
                    program,
                    adapter.build_reward_inputs,
                )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "environment-axis qualification requires Torch",
    )
    def test_environment_axis_locality_rejects_middle_row_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="cross-row-middle",
                source=_minimal_reward_source(
                    expression="inputs.base_height - inputs.base_height[2]"
                ),
            )
            candidate = CandidatePackageRef(
                candidate_id="cross-row-middle",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            )
            admission = StairCandidateAdmissionPolicy(self.config)
            sealed = seal_candidate_package(
                candidate,
                artifact_root=root / "artifacts",
            )
            program = qualify_stair_candidate_reward_program(
                artifact=sealed,
                contract=self.config.protocol_v2.complete_reward_runtime,
                static_verifier_policy=admission.static_verifier.policy,
                trusted_reference_artifact=self.reference,
            )
            adapter = StairClimbingComponentSmokeAdapter(self.config)

            with self.assertRaisesRegex(
                StairRewardEnvironmentAxisError,
                "single_row_locality_failed",
            ):
                assert_stair_reward_program_environment_independent(
                    program,
                    adapter.build_reward_inputs,
                )

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
            # Exercise the current production composition.  The class-level
            # reference belongs to the explicit legacy-v2 admission fixture and
            # must not be injected into the unified Phase-3R cell.
            provider = StairClimbingCompositionProvider()
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
            source = OFFICIAL_R0_SOURCE_PATH.read_text(encoding="utf-8")
            source = source.replace(
                "def termination_penalty(inputs):\n",
                "def termination_penalty(inputs):\n"
                "    print('custom-source-policy')\n",
                1,
            )
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="custom-source-policy",
                source=source,
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

            descriptor = admission.admission_policy_descriptor()
            authorization = CandidateExecutionAuthorization.from_payload(
                admission_policy_id=descriptor.policy_id,
                admission_policy_hash=descriptor.policy_hash,
                payload=admission.build_execution_authorization_payload(
                    artifact=artifact.sealed_artifact,
                    result=verification.result,
                ),
            )
            verified_authorization = (
                validate_stair_candidate_execution_authorization(
                    authorization,
                    artifact=artifact.sealed_artifact,
                    config=self.config,
                )
            )
            self.assertEqual(
                static_verifier_source_policy_hash(
                    policy=verified_authorization.static_verifier_policy
                ),
                expected_source_policy_hash,
            )
            self.assertEqual(
                verified_authorization.qualification_hash,
                qualification["qualification_hash"],
            )

            envelope = json.loads(authorization.canonical_json)
            envelope["payload"]["qualification_hash"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "authorization hash mismatch"):
                CandidateExecutionAuthorization.from_canonical_json(
                    canonical_json_text(envelope),
                    expected_authorization_hash=authorization.authorization_hash,
                )

            drifted_policy = replace(
                custom_policy,
                component_load_timeout_s=(
                    custom_policy.component_load_timeout_s + 1.0
                ),
            )
            envelope = json.loads(authorization.canonical_json)
            envelope["payload"]["static_verifier_policy_snapshot"] = (
                drifted_policy.to_snapshot_payload()
            )
            drifted_authorization = (
                CandidateExecutionAuthorization.from_canonical_json(
                    canonical_json_text(envelope)
                )
            )
            with self.assertRaisesRegex(ValueError, "admission identity mismatch"):
                validate_stair_candidate_execution_authorization(
                    drifted_authorization,
                    artifact=artifact.sealed_artifact,
                    config=self.config,
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
    def test_legacy_official_r0_admits_but_cannot_enter_unified_cell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_dir = _write_candidate_package(
                root,
                candidate_id="official-r0-as-candidate",
                source=OFFICIAL_R0_SOURCE_PATH.read_text(encoding="utf-8"),
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
                verifier=StairCandidateAdmissionPolicy(self.config),
            )
            artifact = verification.require_verified()

            self.assertEqual(
                verification.result.reason,
                "stair_candidate_admission_passed",
            )
            self.assertEqual(
                artifact.verification_policy_hash,
                StairCandidateAdmissionPolicy(self.config).policy_hash,
            )
            self.assertIn(
                COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
                verification.result.details["capability_ids"],
            )
            provider = StairClimbingCompositionProvider(
                reference_program_artifact=self.reference
            )
            with self.assertRaisesRegex(
                ValueError,
                "reference capability profile mismatch",
            ):
                resolve_task_composition(
                    provider=provider,
                    values=provider.cli_defaults(),
                    context=TaskCompositionContext(
                        project_root=PROJECT_ROOT,
                        output_dir=root / "unified-output",
                    ),
                )


if __name__ == "__main__":
    unittest.main()
