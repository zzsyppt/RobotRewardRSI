from __future__ import annotations

import hashlib
from pathlib import Path

from true_rl_agent.agent_core.session_genesis import (
    VerifiedSessionGenesis,
    prepare_session_genesis,
)
from true_rl_agent.backends.base import BackendAdapter, BackendJobBundle
from true_rl_agent.backends.rsl_rl_training_config import (
    RslRlAgentConfiguration,
)
from true_rl_agent.backends.specs import BackendSpec
from true_rl_agent.candidates.artifact import (
    SealedCandidateArtifactRef,
    VerifiedCandidateArtifactRef,
)
from true_rl_agent.candidates.capability_profile import CandidateCapabilityProfile
from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateSurfacePolicy
from true_rl_agent.candidates.verifier import (
    CandidateAdmissionPolicyDescriptor,
    CandidateVerifier,
    VerificationResult,
    static_verifier_prompt_rules,
)
from true_rl_agent.evaluation.acceptance_protocol import (
    PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
    AcceptanceProtocol,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.protocol import (
    DEFAULT_EVAL_PROTOCOL_VERSION,
    EvaluationProtocol,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.sealed_validation import (
    build_sealed_rollout_evidence,
    build_sealed_validation_gate,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import BaselineManifest
from true_rl_agent.runner.base import Runner, TrainJobResult
from true_rl_agent.runner.checkpoint import CheckpointSafetyReport, seal_checkpoint
from true_rl_agent.runner.eval_isolation import build_candidate_isolation_gate
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.job_attempts import JobAttemptScope
from true_rl_agent.runner.specs import (
    BaselineEvalJobSpec,
    CheckpointOutputPlan,
    CheckpointRef,
    EvalJobSpec,
    EvaluationJobSpec,
    TrainJobSpec,
)
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskPlugin, TaskSpec
from true_rl_agent.training_binding import (
    TrainingBinding,
    initialization_seed_schedule_hash,
)
from true_rl_agent.training_programs import CandidateProgramIdentity

FAKE_ACCEPTANCE_PROTOCOL = AcceptanceProtocol(
    protocol_id="fake_acceptance_v1",
    score_source="public_accept.score_stats.mean",
    min_score_count=3,
    absolute_accept_floor=0.05,
    min_delta=0.05,
    public_score_binding=PUBLIC_SCORE_BINDING_ACCEPTANCE_MEAN,
)
FAKE_SESSION_GENESIS_HASH = "f" * 64
FAKE_CANDIDATE_ADMISSION_POLICY_ID = (
    "true_rl_agent.tests.fake_candidate_admission.v1"
)
FAKE_CANDIDATE_ADMISSION_POLICY_HASH = canonical_json_hash(
    {"schema_version": FAKE_CANDIDATE_ADMISSION_POLICY_ID}
)
FAKE_CAPABILITY_PROFILE = CandidateCapabilityProfile(
    profile_id="fake_candidate_capabilities_v1",
    capability_ids=(
        "diagnostic_term",
        "reward_term",
        "reward_weight_scale_schedule",
    ),
)
FAKE_RSL_RL_AGENT_CONFIGURATION = RslRlAgentConfiguration(
    policy={"class_name": "ActorCritic", "hidden_dims": [32, 32]},
    algorithm={"class_name": "PPO", "learning_rate": 0.001},
)


def fake_candidate_manifest_hash(candidate_id: str) -> str:
    return canonical_json_hash(
        {"candidate_id": candidate_id, "source_layout": "fake_components_v1"}
    )


def fake_program_source_checksum(candidate_id: str) -> str:
    return "sha256:" + hashlib.sha256(
        f"fake-program:{candidate_id}".encode("utf-8")
    ).hexdigest()


def fake_training_binding(
    policy_initialization: PolicyInitialization,
) -> TrainingBinding:
    checkpoint_hash = (
        policy_initialization.baseline.checkpoint.checksum
        if policy_initialization.baseline is not None
        else None
    )
    return TrainingBinding(
        initialization_mode=policy_initialization.kind,
        initialization_checkpoint_hash=checkpoint_hash,
        policy_architecture_hash=str(
            FAKE_RSL_RL_AGENT_CONFIGURATION.policy_architecture_hash
        ),
        agent_configuration_hash=str(
            FAKE_RSL_RL_AGENT_CONFIGURATION.agent_configuration_hash
        ),
        initialization_seed_schedule_hash=initialization_seed_schedule_hash(42),
        optimizer_training_profile_hash=str(
            FAKE_RSL_RL_AGENT_CONFIGURATION.optimizer_training_profile_hash
        ),
        budget_steps=128,
        private_training_seed_resolver_hash=canonical_json_hash(
            {"resolver": "fake_private_v1"}
        ),
        capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
    )


def _fake_evaluation_binding(task_spec: TaskSpec) -> EvaluationBindingV1:
    protocol = EvaluationProtocol(
        eval_protocol_version=DEFAULT_EVAL_PROTOCOL_VERSION,
        public_seed_count=4,
        public_accept_seed_count=3,
        sealed_validation_seed_count=8,
        eval_num_envs=64,
    )
    return EvaluationBindingV1(
        eval_protocol_hash=protocol.protocol_hash,
        acceptance_protocol_hash=FAKE_ACCEPTANCE_PROTOCOL.protocol_hash,
        task_protocol_hash=task_spec.resolved_task_protocol_hash(),
        seed_cohort_hash=canonical_json_hash(
            {"task_id": task_spec.task_id, "seed_cohort": "fake_v1"}
        ),
    )


def fake_evaluation_binding(
    task_plugin: TaskPlugin | None = None,
) -> EvaluationBindingV1:
    plugin = task_plugin or FakeTaskPlugin()
    return _fake_evaluation_binding(plugin.task_spec)


def fake_bootstrap_current_best(
    *,
    initial_best_score: float = 0.10,
    task_plugin: TaskPlugin | None = None,
) -> CurrentBestEvalStats:
    plugin = task_plugin or FakeTaskPlugin()
    return CurrentBestEvalStats.bootstrap(
        initial_best_score=initial_best_score,
        acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
        evaluation_binding=fake_evaluation_binding(plugin),
        anchor_id="fake-bootstrap-anchor",
    )


def fake_round_session_fields(
    *,
    initial_best_score: float = 0.10,
    task_plugin: TaskPlugin | None = None,
) -> dict[str, object]:
    plugin = task_plugin or FakeTaskPlugin()
    return {
        "current_best_eval_stats": fake_bootstrap_current_best(
            initial_best_score=initial_best_score,
            task_plugin=plugin,
        ),
        "session_genesis_hash": FAKE_SESSION_GENESIS_HASH,
        "evaluation_binding": fake_evaluation_binding(plugin),
    }


def prepare_fake_session_genesis(
    *,
    output_dir: Path,
    session_id: str,
    initial_best_score: float = 0.10,
    task_plugin: TaskPlugin | None = None,
    backend_adapter: BackendAdapter | None = None,
    runner: Runner | None = None,
    genesis_identity_snapshot: dict[str, object] | None = None,
) -> VerifiedSessionGenesis:
    plugin = task_plugin or FakeTaskPlugin()
    backend = backend_adapter or FakeBackendAdapter()
    return prepare_session_genesis(
        output_dir=output_dir,
        session_id=session_id,
        task_plugin=plugin,
        backend_adapter=backend,
        runner=runner or FakeRunner(),
        policy_initialization=PolicyInitialization.random(),
        genesis_identity_snapshot=(
            genesis_identity_snapshot or {"test_fixture": "fake_session_v1"}
        ),
        bootstrap_initial_best_score=initial_best_score,
    )
class FakeTaskPlugin(TaskPlugin):
    @property
    def task_spec(self) -> TaskSpec:
        return TaskSpec(
            task_id="fake_push_recovery",
            name="Fake Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            genesis_anchor_policy={
                "random": "bootstrap",
                "warm_start": "policy_baseline",
            },
            candidate_capability_profile=FAKE_CAPABILITY_PROFILE,
            invariants=TaskInvariantSpec(
                {
                    "fall_threshold": 0.30,
                    "episode_length": 512,
                    "push_distribution": "fixed_fake",
                }
            ),
            acceptance_protocol=FAKE_ACCEPTANCE_PROTOCOL,
        )

    def candidate_surface_policy(self) -> CandidateSurfacePolicy:
        return CandidateSurfacePolicy(
            parameter_only_components=frozenset(
                {
                    "reward_weight",
                    "reward_weight_scale",
                    "terrain_range",
                }
            ),
            diagnostic_components=frozenset({"diagnostic"}),
            engineering_components=frozenset(
                {"failure_classifier", "logging", "public_projection"}
            ),
            code_level_components=frozenset(
                {"reward_callable", "control_logic", "recovery_logic"}
            ),
        )


class FakeBackendAdapter(BackendAdapter):
    @property
    def candidate_admission_policy_identity(self) -> tuple[str, str]:
        return (
            FAKE_CANDIDATE_ADMISSION_POLICY_ID,
            FAKE_CANDIDATE_ADMISSION_POLICY_HASH,
        )

    @property
    def backend_spec(self) -> BackendSpec:
        return BackendSpec(
            backend_id="fake_backend",
            supported_tasks=frozenset({"fake_push_recovery"}),
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            supports_baseline_eval=True,
        )

    def evaluation_binding_for(self, task_spec: TaskSpec) -> EvaluationBindingV1:
        return _fake_evaluation_binding(task_spec)

    def materialize_jobs(
        self,
        *,
        round_id: str,
        task_spec: TaskSpec,
        candidate: VerifiedCandidateArtifactRef,
        policy_initialization: PolicyInitialization,
        evaluation_binding: EvaluationBindingV1 | None = None,
        expected_initial_policy_state_hash: str | None = None,
        backend_execution_environment_manifest_hash: str | None = None,
        train_attempt_scope: JobAttemptScope | None = None,
        eval_attempt_scope: JobAttemptScope | None = None,
    ) -> BackendJobBundle:
        _ = backend_execution_environment_manifest_hash
        evaluation_binding = evaluation_binding or self.evaluation_binding_for(
            task_spec
        )
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=evaluation_binding,
        )
        metadata = {
            "fake_public_score": candidate.metadata["fake_public_score"],
            "changed_components": sorted(candidate.change_set.changed_components),
        }
        subject = ExperimentSubject.candidate(candidate.candidate_id)
        subject_manifest_hash = candidate.subject_manifest_hash
        program_source_checksum = candidate.program_source_checksum
        training_binding = fake_training_binding(policy_initialization)
        train_spec = TrainJobSpec(
            job_id=f"{round_id}-train",
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            policy_initialization=policy_initialization,
            training_binding=training_binding,
            budget_steps=128,
            execution_device="cuda:0",
            metadata=metadata,
            candidate_artifact=candidate,
            expected_initial_policy_state_hash=(
                expected_initial_policy_state_hash
            ),
            attempt_scope=train_attempt_scope,
        )
        planned_checkpoint = CheckpointOutputPlan(
            uri=f"artifact://{round_id}/{candidate.candidate_id}/planned.pt",
        )
        eval_spec = EvalJobSpec(
            job_id=f"{round_id}-eval",
            task_id=task_spec.task_id,
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=str(training_binding.binding_hash),
            capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
            checkpoint=None,
            public_seed_count=4,
            evaluation_binding=evaluation_binding,
            execution_device="cuda:0",
            sealed_validation_seed_count=8,
            checkpoint_output_plan=planned_checkpoint,
            metadata=metadata,
            attempt_scope=eval_attempt_scope,
        )
        return BackendJobBundle(train_spec=train_spec, eval_spec=eval_spec)

    def materialize_baseline_eval(
        self,
        *,
        session_id: str,
        task_spec: TaskSpec,
        baseline_manifest: BaselineManifest,
        evaluation_binding: EvaluationBindingV1 | None = None,
    ) -> BaselineEvalJobSpec:
        evaluation_binding = evaluation_binding or self.evaluation_binding_for(
            task_spec
        )
        self.validate_session_evaluation_binding(
            task_spec=task_spec,
            evaluation_binding=evaluation_binding,
        )
        return BaselineEvalJobSpec(
            job_id=f"{session_id}-baseline-eval",
            task_id=task_spec.task_id,
            subject=ExperimentSubject.baseline(baseline_manifest.baseline_id),
            subject_manifest_hash=str(baseline_manifest.manifest_hash),
            capability_profile_hash=str(FAKE_CAPABILITY_PROFILE.profile_hash),
            checkpoint=baseline_manifest.checkpoint,
            public_seed_count=4,
            evaluation_binding=evaluation_binding,
            execution_device="cuda:0",
            sealed_validation_seed_count=8,
            metadata={"fake_public_score": 0.25},
        )


class FakeRunner(Runner):
    def run_train(self, train_spec: TrainJobSpec) -> TrainJobResult:
        changed_components = set(train_spec.metadata.get("changed_components", ()))
        runtime_activation_summary = {
            "reward_term_calls": 3 if "reward_callable" in changed_components else 0,
            "diagnostic_calls": 2 if "diagnostic" in changed_components else 0,
            "nonzero_reward_contribution": "reward_callable" in changed_components,
            "reward_output_stats": (
                {"mean": 0.5, "finite_fraction": 1.0}
                if "reward_callable" in changed_components
                else {}
            ),
            "diagnostic_outputs": (
                {"public_margin": 0.1}
                if "diagnostic" in changed_components
                else {}
            ),
            "feature_activation_records": ["fake_feature_projection"] if "recovery_logic" in changed_components else [],
            "capability_activation_records": sorted(changed_components),
        }
        initial_policy_state_pairing = (
            InitialPolicyStatePairing(
                expected_initial_policy_state_hash=(
                    train_spec.expected_initial_policy_state_hash
                ),
                actual_initial_policy_state_hash=(
                    train_spec.expected_initial_policy_state_hash
                ),
                match=True,
            )
            if train_spec.expected_initial_policy_state_hash is not None
            else None
        )
        checkpoint_metadata = {
            "training_subject": train_spec.subject.to_public_dict(),
            "subject_manifest_hash": train_spec.subject_manifest_hash,
            "program_source_checksum": train_spec.program_source_checksum,
            "training_binding_hash": train_spec.training_binding.binding_hash,
            "capability_profile_hash": (
                train_spec.training_binding.capability_profile_hash
            ),
            "execution_device": train_spec.execution_device,
        }
        if initial_policy_state_pairing is not None:
            checkpoint_metadata["initial_policy_state_pairing"] = (
                initial_policy_state_pairing.to_public_dict()
            )
        raw_checkpoint = CheckpointRef(
            uri=f"artifact://{train_spec.job_id}/best.pt",
            checksum=(
                "sha256:"
                + hashlib.sha256(
                    train_spec.subject.subject_id.encode("utf-8")
                ).hexdigest()
            ),
            role="train_output",
            metadata=checkpoint_metadata,
        )
        checkpoint = seal_checkpoint(
            raw_checkpoint,
            sealed_uri=f"artifact://{train_spec.job_id}/sealed/best.pt",
        )
        return TrainJobResult(
            job_id=train_spec.job_id,
            subject=train_spec.subject,
            subject_manifest_hash=train_spec.subject_manifest_hash,
            program_source_checksum=train_spec.program_source_checksum,
            training_binding_hash=str(train_spec.training_binding.binding_hash),
            capability_profile_hash=(
                train_spec.training_binding.capability_profile_hash
            ),
            execution_device=train_spec.execution_device,
            checkpoint=checkpoint,
            initial_policy_state_pairing=initial_policy_state_pairing,
            summary={
                "formal_training_executed": False,
                "budget_steps": train_spec.budget_steps,
                "runtime_activation_summary": runtime_activation_summary,
            },
        )

    def run_eval(self, eval_spec: EvalJobSpec) -> PublicEvalResult:
        return self._run_evaluation(eval_spec)

    def run_baseline_eval(
        self,
        eval_spec: BaselineEvalJobSpec,
    ) -> PublicEvalResult:
        return self._run_evaluation(eval_spec)

    def _run_evaluation(
        self,
        eval_spec: EvaluationJobSpec,
    ) -> PublicEvalResult:
        score = float(eval_spec.metadata["fake_public_score"])
        public_dev_count = max(1, int(eval_spec.public_seed_count))
        public_accept_count = max(1, int(eval_spec.public_accept_seed_count))
        eval_protocol = eval_spec.evaluation_protocol()
        public_dev_stats = {
            "count": public_dev_count,
            "mean": score,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": eval_protocol.protocol_hash,
            "values": [score] * public_dev_count,
        }
        public_accept_stats = {
            "count": public_accept_count,
            "mean": score,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": eval_protocol.protocol_hash,
            "values": [score] * public_accept_count,
        }
        sealed_stats = {
            "count": int(eval_spec.sealed_validation_seed_count),
            "mean": score,
            "std": 0.0,
            "standard_error": 0.0,
            "eval_protocol_hash": eval_protocol.protocol_hash,
        }
        sealed_evidence = build_sealed_rollout_evidence(
            score_stats=sealed_stats,
            expected_seed_count=int(eval_spec.sealed_validation_seed_count),
            eval_protocol_hash=eval_protocol.protocol_hash,
            evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
            evaluation_subject=eval_spec.evaluation_subject(),
            checkpoint_hash=eval_spec.checkpoint.checksum,
            task_validation_status="pass",
            reason="fake_holdout_passed",
            subject_manifest_hash=eval_spec.evaluation_manifest_hash(),
            program_source_checksum=(
                eval_spec.program_source_checksum
                if isinstance(eval_spec, EvalJobSpec)
                else None
            ),
            training_binding_hash=(
                eval_spec.training_binding_hash
                if isinstance(eval_spec, EvalJobSpec)
                else None
            ),
            capability_profile_hash=eval_spec.capability_profile_hash,
        )
        sealed_gate = build_sealed_validation_gate(
            rollout_evidence=sealed_evidence,
            checkpoint_safety_status="pass",
            checkpoint_hash_match_status="pass",
            candidate_isolation_gate=build_candidate_isolation_gate(),
            expected_eval_protocol_hash=eval_protocol.protocol_hash,
            expected_evaluation_binding_hash=eval_spec.evaluation_binding.binding_hash,
        )
        checkpoint_safety_report = CheckpointSafetyReport(
            status="pass",
            checkpoint_hash=eval_spec.checkpoint.checksum,
            expected_checkpoint_hash=eval_spec.checkpoint.checksum,
            file_exists=True,
            byte_read_status="pass",
            weights_only_load_status="pass",
            checkpoint_hash_match_status="pass",
            loader_error_class=None,
            loader_error_summary=None,
            reason="checkpoint_safe_to_load",
        ).to_public_dict()
        metrics = {
            "success_rate": score,
            "public_seed_count": eval_spec.public_seed_count,
            "public_dev": {
                "score_stats": public_dev_stats,
                "seed_results": [
                    {"run_index": index, "status": "pass", "success_rate": score}
                    for index in range(1, public_dev_count + 1)
                ],
            },
            "public_accept": {
                "score_stats": public_accept_stats,
                "seed_results": [
                    {"run_index": index, "status": "pass", "success_rate": score}
                    for index in range(1, public_accept_count + 1)
                ],
            },
            "accept_evidence_gate": {
                "status": "pass",
                "observed_count": public_accept_count,
                "required_count": public_accept_count,
            },
            "budget_accounting": {"status": "pass"},
            "motion_quality_gate": {"status": "pass"},
            "behavior_quality_gate": {"status": "pass"},
            "sealed_validation": sealed_evidence,
            "sealed_validation_gate": sealed_gate,
            "checkpoint_safety_status": "pass",
            "checkpoint_safety_report": checkpoint_safety_report,
            "evaluation_binding": eval_spec.evaluation_binding.to_public_dict(),
            "evaluation_subject": eval_spec.evaluation_subject().to_public_dict(),
            "subject_manifest_hash": eval_spec.evaluation_manifest_hash(),
            "capability_profile_hash": eval_spec.capability_profile_hash,
            "execution_device": eval_spec.execution_device,
            "acceptance": {
                "score_source": FAKE_ACCEPTANCE_PROTOCOL.score_source,
                "public_score_binding": FAKE_ACCEPTANCE_PROTOCOL.public_score_binding,
                "score_stats": public_accept_stats,
                "min_score_count": FAKE_ACCEPTANCE_PROTOCOL.min_score_count,
                "missing_required_score": False,
                "required_pass_gates": [
                    "accept_evidence_gate",
                    "budget_accounting",
                    "motion_quality_gate",
                    "behavior_quality_gate",
                    "sealed_validation_gate",
                ],
                "min_delta": FAKE_ACCEPTANCE_PROTOCOL.min_delta,
                "absolute_accept_floor": FAKE_ACCEPTANCE_PROTOCOL.absolute_accept_floor,
                "acceptance_protocol_hash": FAKE_ACCEPTANCE_PROTOCOL.protocol_hash,
            },
        }
        if isinstance(eval_spec, EvalJobSpec):
            metrics["program_source_checksum"] = (
                eval_spec.program_source_checksum
            )
            metrics["training_binding_hash"] = eval_spec.training_binding_hash
        return PublicEvalResult(
            status="pass",
            public_score=score,
            metrics=metrics,
        )


class FakeVerifier(CandidateVerifier):
    def admission_policy_descriptor(self) -> CandidateAdmissionPolicyDescriptor:
        return CandidateAdmissionPolicyDescriptor(
            policy_id=FAKE_CANDIDATE_ADMISSION_POLICY_ID,
            policy_hash=FAKE_CANDIDATE_ADMISSION_POLICY_HASH,
        )

    def authoring_prompt_rules(self):
        return static_verifier_prompt_rules()

    def verify(self, candidate: CandidatePackageRef) -> VerificationResult:
        status = str(candidate.metadata.get("verifier_status", "pass"))
        if status == "pass" and isinstance(candidate, SealedCandidateArtifactRef):
            return VerificationResult(
                status=status,
                reason="fake_verifier",
                details={
                    "verification_policy_id": (
                        FAKE_CANDIDATE_ADMISSION_POLICY_ID
                    ),
                    "verification_policy_hash": (
                        FAKE_CANDIDATE_ADMISSION_POLICY_HASH
                    ),
                },
                program_identity=CandidateProgramIdentity(
                    candidate_id=candidate.candidate_id,
                    source_checksum=candidate.program_source_checksum,
                    source_size_bytes=candidate.program_source_size_bytes,
                    manifest_hash=candidate.subject_manifest_hash,
                ),
                candidate_artifact_hash=candidate.artifact_hash,
            )
        return VerificationResult(status=status, reason="fake_verifier")
