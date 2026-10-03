"""Push-recovery-owned application composition."""

from __future__ import annotations
from .trajectory_pipeline import DEFAULT_PROFILE
from .trajectory_card import MAX_PUBLIC_BYTES

from dataclasses import dataclass, fields, replace
from functools import partial
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.backends.isaac_lab_rsl_rl import (
    IsaacLabRslRlBackendAdapter,
    IsaacLabRslRlBackendConfig,
)
from true_rl_agent.candidates.verifier import StaticCandidateVerifier
from true_rl_agent.composition.base import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionProvider,
)
from true_rl_agent.composition.cell_manifest import (
    ResolvedCellManifestV1,
    load_resolved_cell_manifest,
    materialize_resolved_cell_manifest,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.specs import require_execution_device
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.robots.rsl_rl_runtime import rsl_rl_agent_configuration
from true_rl_agent.tasks.push_recovery.candidate_failure import (
    PUSH_CANDIDATE_FAILURE_CONTRACT,
)
from true_rl_agent.tasks.push_recovery.checkpoint_admission import admit_push_evaluation_checkpoint
from true_rl_agent.tasks.push_recovery.config import (
    PushRecoveryTaskConfig, PUSH_ACTIVE_CELLS, PUSH_FORCE_EVAL_BASE_SEED,
    PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID, OBSERVED_REFERENCE_ADMISSION,
)
from true_rl_agent.tasks.push_recovery.force_protocol import ForceEvaluationProtocol
from true_rl_agent.tasks.push_recovery.memory import PushTaskMemoryAdapter
from true_rl_agent.tasks.push_recovery.phased_evaluation import finalize_push_eval_phase_summary
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol
from true_rl_agent.tasks.push_recovery.robot_binding import (
    PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID,
    PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID,
    resolve_push_cell_manifest,
)

PUSH_RECOVERY_TASK_ID = "unitree_push_recovery"
PUSH_RECOVERY_BACKEND_ID = "isaac_lab_rsl_rl"
PUSH_RECOVERY_ROBOT = "unitree_g1"
PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH = "push_recovery_obs_v1"
PUSH_RECOVERY_ACTION_SCHEMA_HASH = "unitree_actions_v1"
PUSH_RECOVERY_POLICY_ARCHITECTURE = "rsl_rl_ppo_mlp"
from true_rl_agent.backends.isaac_lab_runtime import DEFAULT_ISAAC_LAB_PYTHON

DEFAULT_ISAAC_PYTHON = DEFAULT_ISAAC_LAB_PYTHON
DEFAULT_TRAIN_ENTRYPOINT = "scripts/isaaclab_rsl_rl_train.py"
DEFAULT_EVAL_ENTRYPOINT = "scripts/isaaclab_rsl_rl_eval.py"
DEFAULT_EXECUTION_DEVICE = "cuda:0"


@dataclass(frozen=True)
class PushRecoveryCompositionConfig:
    """Canonical validated input for one push-recovery composition snapshot."""

    isaac_python: str
    train_entrypoint: str
    eval_entrypoint: str
    execution_device: str
    budget_steps: int
    public_seed_count: int
    public_accept_seed_count: int
    sealed_validation_seed_count: int
    eval_protocol_version: str
    train_seed: int
    runtime_config: str | None
    train_profile: str
    eval_profile: str
    train_num_envs: int
    train_max_iterations: int
    train_steps_per_iteration: int
    eval_num_envs: int
    runner_timeout_s: int
    reference_program_genesis: bool = False
    cell_variant_id: str = PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID
    eval_base_seed: int = PUSH_FORCE_EVAL_BASE_SEED
    reference_program_id: str | None = None
    push_protocol: PushRecoveryProtocol | None = None
    push_evaluation_protocol: ForceEvaluationProtocol | None = None
    reference_admission_policy: str | None = OBSERVED_REFERENCE_ADMISSION
    behavior_feedback_profile: str = DEFAULT_PROFILE
    trajectory_report_max_bytes: int = MAX_PUBLIC_BYTES

    def __post_init__(self) -> None:
        from .trajectory_pipeline import validate_feedback_profile
        validate_feedback_profile(self.behavior_feedback_profile, self.trajectory_report_max_bytes, self.cell_variant_id)
        for field_name in (
            "isaac_python",
            "train_entrypoint",
            "eval_entrypoint",
            "execution_device",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        require_execution_device(
            self.execution_device,
            "push composition execution_device",
        )
        if self.runtime_config is not None and (
            not isinstance(self.runtime_config, str)
            or not self.runtime_config.strip()
        ):
            raise ValueError("runtime_config must be a non-empty string when provided")
        self.to_task_config()

    @classmethod
    def defaults(cls) -> "PushRecoveryCompositionConfig":
        task_config = PushRecoveryTaskConfig.for_cell(PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID)
        return cls(
            isaac_python=DEFAULT_ISAAC_PYTHON,
            train_entrypoint=DEFAULT_TRAIN_ENTRYPOINT,
            eval_entrypoint=DEFAULT_EVAL_ENTRYPOINT,
            execution_device=DEFAULT_EXECUTION_DEVICE,
            budget_steps=task_config.budget_steps,
            public_seed_count=task_config.public_seed_count,
            public_accept_seed_count=task_config.public_accept_seed_count,
            sealed_validation_seed_count=(
                task_config.sealed_validation_seed_count
            ),
            eval_protocol_version=task_config.eval_protocol_version,
            train_seed=task_config.train_seed,
            eval_base_seed=task_config.eval_base_seed,
            runtime_config=task_config.runtime_config_path,
            train_profile=task_config.train_profile,
            eval_profile=task_config.eval_profile,
            train_num_envs=task_config.train_num_envs,
            train_max_iterations=task_config.train_max_iterations,
            train_steps_per_iteration=task_config.train_steps_per_iteration,
            eval_num_envs=task_config.eval_num_envs,
            runner_timeout_s=task_config.runner_timeout_s,
            reference_program_genesis=task_config.reference_program_genesis,
            cell_variant_id=task_config.cell_variant_id,
            push_protocol=task_config.push_protocol,
            push_evaluation_protocol=task_config.push_evaluation_protocol,
            reference_program_id=task_config.reference_program_id,
            reference_admission_policy=task_config.reference_admission_policy,
        )

    @classmethod
    def from_values(
        cls,
        values: Mapping[str, Any],
    ) -> "PushRecoveryCompositionConfig":
        field_names = tuple(field.name for field in fields(cls))
        missing = [name for name in field_names if name not in values
                   and name not in {"reference_program_genesis", "cell_variant_id",
                                    "eval_base_seed", "reference_program_id", "push_protocol", "push_evaluation_protocol",
                                    "reference_admission_policy", "behavior_feedback_profile", "trajectory_report_max_bytes"}]
        if missing:
            raise ValueError(
                "push composition values missing: " + ", ".join(missing)
            )
        selected = {name: values[name] for name in field_names if name in values}
        selected.setdefault("reference_admission_policy", None)
        if isinstance(selected.get("push_protocol"), Mapping):
            selected["push_protocol"] = PushRecoveryProtocol.from_mapping(selected["push_protocol"])
        if isinstance(selected.get("push_evaluation_protocol"), Mapping):
            selected["push_evaluation_protocol"] = ForceEvaluationProtocol.from_mapping(selected["push_evaluation_protocol"])
        return cls(**selected)

    def to_task_config(self) -> PushRecoveryTaskConfig:
        return PushRecoveryTaskConfig.for_cell(
            self.cell_variant_id,
            reference_program_genesis=self.reference_program_genesis,
            budget_steps=self.budget_steps,
            train_num_envs=self.train_num_envs,
            train_max_iterations=self.train_max_iterations,
            train_steps_per_iteration=self.train_steps_per_iteration,
            eval_num_envs=self.eval_num_envs,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_protocol_version=self.eval_protocol_version,
            train_seed=self.train_seed,
            eval_base_seed=self.eval_base_seed,
            runner_timeout_s=self.runner_timeout_s,
            runtime_config_path=self.runtime_config,
            train_profile=self.train_profile,
            eval_profile=self.eval_profile,
            reference_program_id=self.reference_program_id,
            reference_admission_policy=(self.reference_admission_policy
                if self.cell_variant_id in PUSH_ACTIVE_CELLS else None),
            **({"push_protocol": self.push_protocol} if self.push_protocol is not None else {}),
            **({"push_evaluation_protocol": self.push_evaluation_protocol} if self.push_evaluation_protocol is not None else {}),
        )

    def to_task_defaults(self) -> dict[str, Any]:
        result = {
            field.name: getattr(self, field.name)
            for field in fields(self)
        }
        if self.push_protocol is not None:
            result["push_protocol"] = self.push_protocol.to_dict()
        if self.push_evaluation_protocol is not None:
            result["push_evaluation_protocol"] = self.push_evaluation_protocol.to_dict()
        else:
            result.pop("push_evaluation_protocol")
        if self.reference_admission_policy is None:
            result.pop("reference_admission_policy")
        if self.behavior_feedback_profile == "legacy_process_v1":
            result.pop("behavior_feedback_profile")
            result.pop("trajectory_report_max_bytes")
        return result


class PushRecoveryCompositionProvider(TaskCompositionProvider):
    @property
    def task_id(self) -> str:
        return PUSH_RECOVERY_TASK_ID

    def cli_defaults(self) -> Mapping[str, Any]:
        defaults = PushRecoveryCompositionConfig.defaults().to_task_defaults()
        defaults["cell_variant_id"] = PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID
        # Robot-specific frozen training defaults resolve only after selecting a row.
        defaults["push_protocol"] = defaults["push_evaluation_protocol"] = defaults["reference_program_id"] = None
        # Resolve an omitted seed after selecting the robot row.
        defaults["eval_base_seed"] = None
        return defaults

    def add_cli_arguments(self, parser: Any) -> None:
        from true_rl_agent.tasks.push_recovery.reference_program import PUSH_REFERENCE_IDS
        from .trajectory_pipeline import PROFILES, LEGACY_PROFILE

        parser.add_argument("--behavior-feedback-profile", choices=PROFILES, default=DEFAULT_PROFILE,
                            help="Automatic CPU root cards for physical Push public-dev numeric archives.")
        parser.add_argument("--trajectory-report-max-bytes", type=int, default=MAX_PUBLIC_BYTES)
        parser.add_argument("--eval-base-seed", type=int, default=None)
        parser.add_argument("--reference-program-id", choices=PUSH_REFERENCE_IDS, default=None)
        parser.add_argument("--push-protocol-path", type=Path, default=None,
                            help="Complete frozen TRAINING PushRecoveryProtocol; evaluation uses the force protocol.")
        parser.add_argument(
            "--cell-variant-id",
            choices=(*PUSH_ACTIVE_CELLS, PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID, PUSH_G1_29DOF_REFERENCE_CELL_VARIANT_ID),
            default=PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID,
            help="Active G1/Go2 force evaluation rows; G1-29 legacy is retained for frozen audit replay.",
        )

    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        values = dict(values)
        if context.read_only:
            frozen = load_resolved_cell_manifest(context.output_dir / "resolved_cell_manifest.json")
            # Frozen inputs own their admission policy, including old manifests
            # whose argv predates this field and would otherwise inherit defaults.
            values["reference_admission_policy"] = frozen.robot_task_binding["task_projection"].get(
                "reference_admission_policy")
        if values.get("eval_base_seed") is None:
            values["eval_base_seed"] = (PUSH_FORCE_EVAL_BASE_SEED
                if values.get("cell_variant_id", PUSH_G1_37DOF_FORCE_CELL_VARIANT_ID) in PUSH_ACTIVE_CELLS else 42)
        if values.get("push_protocol_path") is not None:
            path = Path(values["push_protocol_path"])
            if not path.is_absolute():
                path = context.project_root / path
            values = {**values, "push_protocol": load_strict_json_object(path, context="Push protocol")}
        composition_config = PushRecoveryCompositionConfig.from_values(values)
        cell_variant_id = composition_config.cell_variant_id
        composition_config = replace(composition_config, reference_program_genesis=(
            cell_variant_id != PUSH_G1_29DOF_LEGACY_CELL_VARIANT_ID))
        task_config = composition_config.to_task_config()
        composition_config = replace(composition_config,
            push_protocol=task_config.push_protocol,
            push_evaluation_protocol=task_config.push_evaluation_protocol,
            reference_admission_policy=task_config.reference_admission_policy,
            reference_program_id=task_config.reference_program_id)
        task_plugin = PushRecoveryTaskPlugin(task_config)
        resolved_cell_manifest = resolve_push_cell_manifest(
            cell_variant_id=cell_variant_id,
            config=task_config,
            plugin=task_plugin,
        )
        if task_config.push_protocol is not None:
            task_config = replace(task_config, resolved_cell_task_protocol_hash=(
                resolved_cell_manifest.resolved_cell_task_protocol_hash
            ))
            task_plugin = PushRecoveryTaskPlugin(task_config)
        resolved_cell_manifest_path = materialize_resolved_cell_manifest(
            resolved_cell_manifest,
            output_dir=context.output_dir,
            require_existing=context.read_only,
        )
        candidate_admission_policy = StaticCandidateVerifier()
        backend_config = _build_push_backend_config(
            composition_config=composition_config,
            context=context,
            task_config=task_config,
            task_plugin=task_plugin,
            candidate_admission_policy=candidate_admission_policy,
            resolved_cell_manifest=resolved_cell_manifest,
            resolved_cell_manifest_path=resolved_cell_manifest_path,
        )
        reference = None
        if task_config.reference_program_genesis:
            from true_rl_agent.tasks.push_recovery.reference_program import resolve_push_reference_program

            reference = resolve_push_reference_program(
                project_root=context.project_root,
                capability_profile_hash=task_plugin.task_spec.candidate_capability_profile.profile_hash,
                config=task_config,
            )
        return ResolvedTaskComposition(
            task_id=self.task_id,
            task_plugin=task_plugin,
            task_memory_adapter=PushTaskMemoryAdapter(config=task_config),
            backend_adapter=IsaacLabRslRlBackendAdapter(backend_config),
            runner=SubprocessRunner(
                allowed_executables=(composition_config.isaac_python,),
                timeout_s=composition_config.runner_timeout_s,
                cwd=context.project_root,
                eval_checkpoint_admission_hook=admit_push_evaluation_checkpoint if task_config.push_protocol is not None else None,
                eval_phase_summary_finalizer=(partial(finalize_push_eval_phase_summary,
                    behavior_feedback_profile=composition_config.behavior_feedback_profile,
                    trajectory_report_max_bytes=composition_config.trajectory_report_max_bytes,
                    expected_seed_count=task_config.public_seed_count)
                    if task_config.push_protocol is not None else None),
            ),
            candidate_admission_policy=candidate_admission_policy,
            policy_initialization=PolicyInitialization.random(),
            genesis_anchor_kind=task_plugin.task_spec.genesis_anchor_kind_for("random"),
            reference_program_artifact=reference,
            task_defaults={
                **composition_config.to_task_defaults(),
                "cell_variant_id": resolved_cell_manifest.cell_variant_id,
            },
            scientific_parameters={
                "training_agent_configuration_hash": backend_config.training_agent_configuration.agent_configuration_hash,
                **({"behavior_feedback_profile": composition_config.behavior_feedback_profile,
                    "trajectory_report_max_bytes": composition_config.trajectory_report_max_bytes}
                   if composition_config.behavior_feedback_profile != "legacy_process_v1" else {}),
            },
            execution_placement={
                "project_root": str(context.project_root.resolve()),
                "output_dir": str(context.output_dir.resolve()),
                "isaac_python": composition_config.isaac_python,
                "execution_device": composition_config.execution_device,
                "runner_timeout_s": composition_config.runner_timeout_s,
                "train_entrypoint": backend_config.train_entrypoint,
                "eval_entrypoint": backend_config.eval_entrypoint,
            },
            resolved_cell_manifest=resolved_cell_manifest,
        )

    def run_audit(
        self,
        *,
        composition: ResolvedTaskComposition,
        output_dir: Path,
        rounds: int,
        session_id: str | None,
    ) -> dict[str, Any]:
        from true_rl_agent.tasks.push_recovery.loop_audit import (
            run_push_recovery_loop_audit,
        )

        return run_push_recovery_loop_audit(
            composition=composition,
            output_dir=output_dir,
            rounds=rounds,
            session_id=session_id,
        )


def _build_push_backend_config(
    *,
    composition_config: PushRecoveryCompositionConfig,
    context: TaskCompositionContext,
    task_config: PushRecoveryTaskConfig,
    task_plugin: PushRecoveryTaskPlugin,
    candidate_admission_policy: StaticCandidateVerifier,
    resolved_cell_manifest: ResolvedCellManifestV1,
    resolved_cell_manifest_path: Path,
) -> IsaacLabRslRlBackendConfig:
    canonical_config = composition_config.to_task_config()
    if canonical_config.push_protocol is not None:
        canonical_config = replace(canonical_config, resolved_cell_task_protocol_hash=(
            resolved_cell_manifest.resolved_cell_task_protocol_hash
        ))
    if task_config != canonical_config:
        raise ValueError(
            "push task config does not match canonical composition config"
        )
    if task_plugin.config != task_config:
        raise ValueError("push task plugin config does not match composition config")
    train_runtime_dynamics = task_config.train_runtime_dynamics()
    eval_runtime_dynamics = task_config.eval_runtime_dynamics()
    evaluation_protocol = task_config.evaluation_protocol()
    acceptance_protocol = task_config.acceptance_protocol()
    evaluation_binding = EvaluationBindingV1(
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        acceptance_protocol_hash=acceptance_protocol.protocol_hash,
        task_protocol_hash=task_plugin.task_spec.resolved_task_protocol_hash(),
        seed_cohort_hash=task_config.seed_cohort_hash(),
    )
    admission = candidate_admission_policy.admission_policy_descriptor()
    return IsaacLabRslRlBackendConfig(
        isaac_lab_python=composition_config.isaac_python,
        train_entrypoint=composition_config.train_entrypoint,
        eval_entrypoint=composition_config.eval_entrypoint,
        supported_tasks=frozenset({PUSH_RECOVERY_TASK_ID}),
        default_budget_steps=task_config.budget_steps,
        public_seed_count=task_config.public_seed_count,
        public_accept_seed_count=task_config.public_accept_seed_count,
        sealed_validation_seed_count=task_config.sealed_validation_seed_count,
        eval_num_envs=task_config.eval_num_envs,
        eval_protocol_version=task_config.eval_protocol_version,
        evaluation_binding=evaluation_binding,
        robot=(resolved_cell_manifest.robot_profile["robot_family_id"]
               if task_config.push_protocol is not None else PUSH_RECOVERY_ROBOT),
        observation_schema_hash=(resolved_cell_manifest.contract_hashes["observation_content_hash"]
                                 if task_config.push_protocol is not None else PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH),
        action_schema_hash=(resolved_cell_manifest.contract_hashes["action_content_hash"]
                            if task_config.push_protocol is not None else PUSH_RECOVERY_ACTION_SCHEMA_HASH),
        policy_architecture=PUSH_RECOVERY_POLICY_ARCHITECTURE,
        artifact_root=str(context.output_dir / "runner"),
        train_profile=task_config.train_profile,
        eval_profile=task_config.eval_profile,
        train_num_envs=task_config.train_num_envs,
        training_agent_configuration=rsl_rl_agent_configuration(
            resolved_cell_manifest.runtime_adapter["runtime_adapter_id"]
        ),
        train_max_iterations=task_config.train_max_iterations,
        train_steps_per_iteration=task_config.train_steps_per_iteration,
        eval_base_seed=task_config.eval_base_seed,
        execution_device=composition_config.execution_device,
        candidate_admission_policy_id=admission.policy_id,
        candidate_admission_policy_hash=admission.policy_hash,
        candidate_execution_capability_ids=frozenset(),
        candidate_failure_contract=PUSH_CANDIDATE_FAILURE_CONTRACT,
        train_seed=task_config.train_seed,
        runtime_config_path=task_config.runtime_config_path,
        train_runtime_dynamics_hash=train_runtime_dynamics[
            "runtime_dynamics_hash"
        ],
        eval_runtime_dynamics_hash=eval_runtime_dynamics[
            "runtime_dynamics_hash"
        ],
        work_dir=str(context.project_root),
        curriculum_capabilities=PushRecoveryTaskPlugin.curriculum_capabilities,
        resolved_cell_manifest_path=str(resolved_cell_manifest_path),
        resolved_cell_manifest_hash=resolved_cell_manifest.manifest_hash,
        capture_public_dev_numeric=task_config.push_protocol is not None,
        capture_training_numeric=task_config.push_protocol is not None,
    )


__all__ = [
    "PUSH_RECOVERY_TASK_ID",
    "PushRecoveryCompositionConfig",
    "PushRecoveryCompositionProvider",
]
