from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.backends.rsl_rl_training_config import (
    RslRlAgentConfiguration,
)
from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    VerifiedCandidateArtifactRef,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import (
    CandidateAdmissionPolicyDescriptor,
    CandidateVerifier,
    VerificationResult,
    seal_and_verify_candidate,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.evaluation.protocol import (
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.training_binding import (
    TrainingBinding,
    initialization_seed_schedule_hash,
)
from true_rl_agent.training_programs import CandidateProgramIdentity

TEST_ACCEPTANCE_PROTOCOL_HASH = "a" * 64
TEST_TASK_PROTOCOL_HASH = "b" * 64
TEST_SEED_COHORT_HASH = "c" * 64
TEST_SUBJECT_MANIFEST_HASH = "d" * 64
TEST_PROGRAM_SOURCE_CHECKSUM = "sha256:" + "e" * 64
TEST_TRAINING_BINDING_HASH = "f" * 64
TEST_CAPABILITY_PROFILE = CandidateCapabilityProfile(
    profile_id="test_candidate_capabilities_v1",
    capability_ids=("reward_term",),
)
TEST_RSL_RL_AGENT_CONFIGURATION = RslRlAgentConfiguration(
    policy={"class_name": "ActorCritic", "hidden_dims": [32, 32]},
    algorithm={"class_name": "PPO", "learning_rate": 0.001},
)
_TEST_CANDIDATE_ROOT = Path(
    tempfile.mkdtemp(prefix="true-rl-agent-verified-candidates-")
)
_TEST_ADMISSION_POLICY_ID = "true_rl_agent.tests.bound_identity_admission.v1"
_TEST_ADMISSION_POLICY_HASH = canonical_json_hash(
    {"schema_version": _TEST_ADMISSION_POLICY_ID}
)


class _BoundIdentityTestVerifier(CandidateVerifier):
    """Fast verifier used only to build non-executing test job specs."""

    def __init__(
        self,
        descriptor: CandidateAdmissionPolicyDescriptor | None = None,
    ) -> None:
        self._descriptor = descriptor or CandidateAdmissionPolicyDescriptor(
            policy_id=_TEST_ADMISSION_POLICY_ID,
            policy_hash=_TEST_ADMISSION_POLICY_HASH,
        )

    def admission_policy_descriptor(self) -> CandidateAdmissionPolicyDescriptor:
        return self._descriptor

    def verify(
        self,
        candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    ) -> VerificationResult:
        if not isinstance(candidate, SealedCandidateArtifactRef):
            return VerificationResult("fail", "test_candidate_not_sealed")
        return VerificationResult(
            "pass",
            "test_bound_identity_verifier_passed",
            {
                "derived_change_set": sorted(
                    candidate.change_set.changed_components
                ),
                "verification_policy_id": self._descriptor.policy_id,
                "verification_policy_hash": self._descriptor.policy_hash,
            },
            program_identity=CandidateProgramIdentity(
                candidate_id=candidate.candidate_id,
                source_checksum=candidate.program_source_checksum,
                source_size_bytes=candidate.program_source_size_bytes,
                manifest_hash=candidate.subject_manifest_hash,
            ),
            candidate_artifact_hash=candidate.artifact_hash,
        )


def make_verified_candidate_artifact(
    candidate_id: str = "test-candidate",
    *,
    change_set: CandidateChangeSet | None = None,
    admission_policy_descriptor: CandidateAdmissionPolicyDescriptor | None = None,
) -> VerifiedCandidateArtifactRef:
    """Build a byte-bound execution capability for tests that do not run code."""

    candidate = make_test_candidate_package(
        candidate_id,
        change_set=change_set,
    )
    verification = seal_and_verify_candidate(
        candidate,
        artifact_root=_TEST_CANDIDATE_ROOT / "artifacts",
        verifier=_BoundIdentityTestVerifier(admission_policy_descriptor),
    )
    return verification.require_verified()


def make_test_candidate_package(
    candidate_id: str = "test-candidate",
    *,
    change_set: CandidateChangeSet | None = None,
    metadata: Mapping[str, Any] | None = None,
    provider_root: Path | None = None,
    components_source: str | None = None,
) -> CandidatePackageRef:
    """Materialize a local provider package for round and authoring tests."""

    package_root = (
        provider_root
        if provider_root is not None
        else _TEST_CANDIDATE_ROOT / "providers" / uuid.uuid4().hex
    )
    training_code = package_root / "training_code"
    training_code.mkdir(parents=True)
    mechanism_depth = 1 + sum(candidate_id.encode("utf-8")) % 16
    reward_expression = " + ".join(
        ["inputs.base_height", *(["inputs.roll"] * mechanism_depth)]
    )
    (package_root / "candidate_manifest.yaml").write_text(
        "schema_version: true_rl_agent.candidate.v1\n"
        f"candidate_id: {json.dumps(candidate_id)}\n"
        "entrypoint: training_code.components:build_training_components\n",
        encoding="utf-8",
    )
    default_source = (
        "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
        "def test_reward(inputs):\n"
        f"    return {reward_expression}\n\n"
        "def build_training_components():\n"
        "    return TrainingComponents(\n"
        f"        name={candidate_id!r},\n"
        "        reward_terms=[RewardTerm(name='test_reward', weight=1.0, fn=test_reward)],\n"
        "    )\n"
    )
    (training_code / "components.py").write_text(
        components_source if components_source is not None else default_source,
        encoding="utf-8",
    )
    (package_root / "rationale.md").write_text(
        "Test candidate package.\n",
        encoding="utf-8",
    )
    (package_root / "operation_diff.json").write_text("{}\n", encoding="utf-8")
    (package_root / "diagnostics.json").write_text("{}\n", encoding="utf-8")
    return CandidatePackageRef(
        candidate_id=candidate_id,
        path=str(package_root),
        change_set=(
            change_set
            if change_set is not None
            else CandidateChangeSet(frozenset({"reward_callable"}))
        ),
        metadata=dict(metadata or {}),
    )


def make_training_binding(
    policy_initialization: PolicyInitialization | None = None,
    *,
    budget_steps: int = 128,
) -> TrainingBinding:
    initialization = policy_initialization or PolicyInitialization.random()
    checkpoint_hash = (
        initialization.baseline.checkpoint.checksum
        if initialization.baseline is not None
        else None
    )
    return TrainingBinding(
        initialization_mode=initialization.kind,
        initialization_checkpoint_hash=checkpoint_hash,
        policy_architecture_hash=str(
            TEST_RSL_RL_AGENT_CONFIGURATION.policy_architecture_hash
        ),
        agent_configuration_hash=str(
            TEST_RSL_RL_AGENT_CONFIGURATION.agent_configuration_hash
        ),
        initialization_seed_schedule_hash=initialization_seed_schedule_hash(42),
        optimizer_training_profile_hash=str(
            TEST_RSL_RL_AGENT_CONFIGURATION.optimizer_training_profile_hash
        ),
        budget_steps=budget_steps,
        private_training_seed_resolver_hash="4" * 64,
        capability_profile_hash=str(TEST_CAPABILITY_PROFILE.profile_hash),
    )


def make_evaluation_binding(
    *,
    eval_protocol_version: str = DEFAULT_EVAL_PROTOCOL_VERSION,
    public_seed_count: int = 3,
    public_accept_seed_count: int = 3,
    sealed_validation_seed_count: int = 7,
    eval_num_envs: int = 64,
    acceptance_protocol_hash: str = TEST_ACCEPTANCE_PROTOCOL_HASH,
    task_protocol_hash: str = TEST_TASK_PROTOCOL_HASH,
    seed_cohort_hash: str = TEST_SEED_COHORT_HASH,
) -> EvaluationBindingV1:
    protocol = EvaluationProtocol(
        eval_protocol_version=eval_protocol_version,
        public_seed_count=public_seed_count,
        public_accept_seed_count=public_accept_seed_count,
        sealed_validation_seed_count=sealed_validation_seed_count,
        eval_num_envs=eval_num_envs,
    )
    return EvaluationBindingV1(
        eval_protocol_hash=protocol.protocol_hash,
        acceptance_protocol_hash=acceptance_protocol_hash,
        task_protocol_hash=task_protocol_hash,
        seed_cohort_hash=seed_cohort_hash,
    )


__all__ = [
    "TEST_ACCEPTANCE_PROTOCOL_HASH",
    "TEST_SEED_COHORT_HASH",
    "TEST_TASK_PROTOCOL_HASH",
    "TEST_SUBJECT_MANIFEST_HASH",
    "TEST_PROGRAM_SOURCE_CHECKSUM",
    "TEST_TRAINING_BINDING_HASH",
    "TEST_CAPABILITY_PROFILE",
    "make_test_candidate_package",
    "make_verified_candidate_artifact",
    "make_evaluation_binding",
    "make_training_binding",
]
