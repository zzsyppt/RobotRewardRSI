"""Stair-owned application composition provider."""

from __future__ import annotations

import os
from dataclasses import MISSING, dataclass, fields, replace
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from true_rl_agent.backends.isaac_lab_rsl_rl import (
    IsaacLabRslRlBackendAdapter,
    IsaacLabRslRlBackendConfig,
)
from true_rl_agent.backends.isaac_lab_runtime import DEFAULT_ISAAC_LAB_PYTHON
from true_rl_agent.agent_core.generation_finalization import GenerationFinalPlan
from true_rl_agent.candidates.capability_readiness import (
    COMPLETE_REWARD_PROGRAM_CAPABILITY_ID,
)
from true_rl_agent.composition.base import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionProvider,
)
from true_rl_agent.composition.cell_manifest import (
    ResolvedCellManifestV1,
    materialize_resolved_cell_manifest,
)
from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.specs import require_execution_device
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.candidate_admission import (
    StairCandidateAdmissionPolicy,
)
from true_rl_agent.tasks.stair_climbing.candidate_failure import (
    STAIR_CANDIDATE_FAILURE_CONTRACT,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.trajectory_production import VERSIONS as PRODUCTION_VERSIONS, DEFAULT_PUBLIC_BYTES, PROFILE
from true_rl_agent.tasks.stair_climbing.final_confirmation import (
    STAIR_FINAL_CONFIRMATION_FORMAL_SESSION_MODE,
    STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS,
    STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS,
    load_stair_final_confirmation_plan,
    stair_final_confirmation_plan_path,
)
from true_rl_agent.tasks.stair_climbing.checkpoint_ancestry import (
    build_stair_checkpoint_ancestry_admission,
)
from true_rl_agent.tasks.stair_climbing.generation_finalization import read_stair_final_plan
from true_rl_agent.tasks.stair_climbing.memory import StairTaskMemoryAdapter
from true_rl_agent.tasks.stair_climbing.phased_evaluation import finalize_stair_eval_phase_summary
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    DEFAULT_STAIR_CELL_VARIANT_ID,
    active_unified_stair_cell_variant_ids,
    stair_robot_row,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_UNIFIED_EVAL_PROFILE,
    STAIR_UNIFIED_TRAIN_PROFILE,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
)
from true_rl_agent.tasks.stair_climbing.session_start import (
    STAIR_DEFAULT_START_MODE,
    STAIR_REFERENCE_START_MODE,
    STAIR_REWARD_COLD_START_MODE,
    StairSessionStartProfile,
    resolve_stair_session_start_profile,
)
from true_rl_agent.training_programs import VerifiedReferenceProgramArtifact

DEFAULT_ISAAC_PYTHON = DEFAULT_ISAAC_LAB_PYTHON
DEFAULT_STAIR_TRAIN_ENTRYPOINT = "scripts/isaaclab_stair_rsl_rl_train.py"
DEFAULT_STAIR_EVAL_ENTRYPOINT = "scripts/isaaclab_stair_rsl_rl_eval.py"
DEFAULT_STAIR_PROTOCOL = stair_robot_row(DEFAULT_STAIR_CELL_VARIANT_ID).protocol_relative_path
DEFAULT_EXECUTION_DEVICE = "cuda:0"
# Execution placement allowance for the frozen full workload on Sim 6.
DEFAULT_RUNNER_TIMEOUT_S = 36000


@dataclass(frozen=True)
class StairClimbingCompositionConfig:
    isaac_python: str
    train_entrypoint: str
    eval_entrypoint: str
    execution_device: str
    start_mode: str
    budget_steps: int
    public_seed_count: int
    public_accept_seed_count: int
    sealed_validation_seed_count: int
    eval_protocol_version: str
    train_seed: int
    runtime_config: str
    train_profile: str
    eval_profile: str
    train_num_envs: int
    train_max_iterations: int
    train_steps_per_iteration: int
    eval_num_envs: int
    runner_timeout_s: int
    cell_variant_id: str
    behavior_feedback_profile: str = PROFILE
    trajectory_report_max_bytes: int = DEFAULT_PUBLIC_BYTES

    def __post_init__(self) -> None:
        from .trajectory_pipeline import validate_feedback_profile
        validate_feedback_profile(self.behavior_feedback_profile, self.trajectory_report_max_bytes, self.cell_variant_id)
        for field_name in (
            "isaac_python",
            "train_entrypoint",
            "eval_entrypoint",
            "execution_device",
            "start_mode",
            "eval_protocol_version",
            "runtime_config",
            "train_profile",
            "eval_profile",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        require_execution_device(
            self.execution_device,
            "stair composition execution_device",
        )
        for field_name in (
            "budget_steps",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "train_num_envs",
            "train_max_iterations",
            "train_steps_per_iteration",
            "eval_num_envs",
            "runner_timeout_s",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if (
            isinstance(self.train_seed, bool)
            or not isinstance(self.train_seed, int)
            or self.train_seed < 0
        ):
            raise ValueError("train_seed must be a non-negative integer")
        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.train_num_envs,
            num_steps_per_env=self.train_steps_per_iteration,
            max_iterations=self.train_max_iterations,
        )
        if self.train_entrypoint != DEFAULT_STAIR_TRAIN_ENTRYPOINT:
            raise ValueError("stair train_entrypoint is not canonical")
        if self.eval_entrypoint != DEFAULT_STAIR_EVAL_ENTRYPOINT:
            raise ValueError("stair eval_entrypoint is not canonical")
        try:
            row = stair_robot_row(self.cell_variant_id)
        except KeyError as exc:
            raise ValueError("unsupported Stair cell_variant_id") from exc
        if not row.unified:
            raise ValueError("Stair composition requires a unified robot row")
        if self.runtime_config != row.protocol_relative_path:
            raise ValueError("stair runtime_config differs from the selected robot row")
        runtime_path = PurePosixPath(self.runtime_config)
        if runtime_path.is_absolute() or ".." in runtime_path.parts:
            raise ValueError("stair runtime_config must be project-relative")
        if self.start_mode not in {
            STAIR_REFERENCE_START_MODE,
            STAIR_REWARD_COLD_START_MODE,
        }:
            raise ValueError("unsupported Stair start_mode")

    @classmethod
    def defaults(cls, *, project_root: Path) -> "StairClimbingCompositionConfig":
        task = StairClimbingTaskConfig.defaults(
            project_root / DEFAULT_STAIR_PROTOCOL
        )
        return cls(
            isaac_python=DEFAULT_ISAAC_PYTHON,
            train_entrypoint=DEFAULT_STAIR_TRAIN_ENTRYPOINT,
            eval_entrypoint=DEFAULT_STAIR_EVAL_ENTRYPOINT,
            execution_device=DEFAULT_EXECUTION_DEVICE,
            start_mode=STAIR_DEFAULT_START_MODE,
            budget_steps=task.budget_steps,
            public_seed_count=task.public_seed_count,
            public_accept_seed_count=task.public_accept_seed_count,
            sealed_validation_seed_count=task.sealed_validation_seed_count,
            eval_protocol_version=task.eval_protocol_version,
            train_seed=task.train_seed,
            runtime_config=DEFAULT_STAIR_PROTOCOL,
            train_profile=STAIR_UNIFIED_TRAIN_PROFILE,
            eval_profile=STAIR_UNIFIED_EVAL_PROFILE,
            train_num_envs=task.train_num_envs,
            train_max_iterations=task.train_max_iterations,
            train_steps_per_iteration=task.train_steps_per_iteration,
            eval_num_envs=task.eval_num_envs,
            runner_timeout_s=DEFAULT_RUNNER_TIMEOUT_S,
            cell_variant_id=DEFAULT_STAIR_CELL_VARIANT_ID,
        )

    @classmethod
    def from_values(cls, values: Mapping[str, Any]) -> "StairClimbingCompositionConfig":
        names = tuple(item.name for item in fields(cls))
        missing = [item.name for item in fields(cls) if item.name not in values and item.default is MISSING]
        if missing:
            raise ValueError("stair composition values missing: " + ", ".join(missing))
        normalized = {name: values[name] for name in names if name in values}
        if normalized.get("trajectory_report_max_bytes") is None:
            normalized["trajectory_report_max_bytes"] = DEFAULT_PUBLIC_BYTES
        normalized["runtime_config"] = stair_robot_row(
            str(normalized["cell_variant_id"])
        ).protocol_relative_path
        return cls(**normalized)

    def to_task_config(
        self,
        *,
        project_root: Path,
        resolved_cell_task_protocol_hash: str | None = None,
    ) -> StairClimbingTaskConfig:
        # TaskConfig derives historical fields from its own protocol load.
        # The live deadline goes directly to the runner / execution placement.
        return StairClimbingTaskConfig(
            protocol_v2_path=project_root / self.runtime_config,
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
            train_profile=self.train_profile,
            eval_profile=self.eval_profile,
            cell_variant_id=self.cell_variant_id,
            resolved_cell_task_protocol_hash=resolved_cell_task_protocol_hash,
        )

    def bind_selected_cell_contract(
        self,
        *,
        project_root: Path,
    ) -> "StairClimbingCompositionConfig":
        """Replace clean CLI defaults with the selected cell's frozen eval shape."""

        row = stair_robot_row(self.cell_variant_id)
        selected = StairClimbingTaskConfig.defaults(
            project_root / row.protocol_relative_path,
            cell_variant_id=self.cell_variant_id,
        )
        return replace(
            self,
            budget_steps=selected.budget_steps,
            public_seed_count=selected.public_seed_count,
            public_accept_seed_count=selected.public_accept_seed_count,
            sealed_validation_seed_count=selected.sealed_validation_seed_count,
            eval_protocol_version=selected.eval_protocol_version,
            train_seed=selected.train_seed,
            runtime_config=row.protocol_relative_path,
            train_profile=selected.train_profile,
            eval_profile=selected.eval_profile,
            train_num_envs=selected.train_num_envs,
            train_max_iterations=selected.train_max_iterations,
            train_steps_per_iteration=selected.train_steps_per_iteration,
            eval_num_envs=selected.eval_num_envs,
        )

    def to_task_defaults(self) -> dict[str, Any]:
        result = {item.name: getattr(self, item.name) for item in fields(self)}
        return result

    def to_resolved_task_defaults(
        self,
        *,
        start_profile: StairSessionStartProfile,
    ) -> dict[str, Any]:
        defaults = self.to_task_defaults()
        if start_profile.profile_id == STAIR_REFERENCE_START_MODE:
            defaults.pop("start_mode")
            return defaults
        defaults["session_start_profile"] = start_profile.to_public_dict()
        defaults["session_start_profile_hash"] = start_profile.profile_hash
        return defaults


class StairClimbingCompositionProvider(TaskCompositionProvider):
    """The sole Stair provider; reference resolution follows the start profile."""

    def __init__(
        self,
        *,
        reference_program_artifact: VerifiedReferenceProgramArtifact | None = None,
    ) -> None:
        if reference_program_artifact is not None:
            if not isinstance(
                reference_program_artifact,
                VerifiedReferenceProgramArtifact,
            ):
                raise TypeError("stair reference program artifact is invalid")
            if reference_program_artifact.manifest.task_id != STAIR_TASK_ID:
                raise ValueError("stair reference program task_id mismatch")
        self._injected_reference_program_artifact = reference_program_artifact
        self._project_root = Path(__file__).resolve().parents[4]

    @property
    def task_id(self) -> str:
        return STAIR_TASK_ID

    def cli_defaults(self) -> Mapping[str, Any]:
        defaults = StairClimbingCompositionConfig.defaults(
            project_root=self._project_root
        ).to_task_defaults()
        return defaults

    def add_cli_arguments(self, parser: Any) -> None:
        parser.add_argument("--behavior-feedback-profile", choices=(PROFILE,),
                            default=PROFILE,
                            help="Direct CPU behavior summary from the evaluation archive.")
        parser.add_argument("--trajectory-report-max-bytes", type=int, default=None,
                            help="UTF-8 byte capacity; defaults to the current description profile.")
        parser.add_argument(
            "--cell-variant-id",
            choices=active_unified_stair_cell_variant_ids(),
            default=DEFAULT_STAIR_CELL_VARIANT_ID,
        )
        parser.add_argument(
            "--start-mode",
            choices=(STAIR_REFERENCE_START_MODE, STAIR_REWARD_COLD_START_MODE),
            default=STAIR_DEFAULT_START_MODE,
        )
        parser.add_argument(
            "--final-confirmation-plan",
            type=Path,
            default=None,
            help=(
                "Required for a live Stair run. The create-once task-owned plan "
                "must exist under the output directory before Genesis."
            ),
        )

    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        composition_config = StairClimbingCompositionConfig.from_values(
            values
        ).bind_selected_cell_contract(project_root=context.project_root)
        task_config = composition_config.to_task_config(
            project_root=context.project_root
        )
        if task_config.protocol_state != "active" and values.get("mode") in {
            "live",
            STAIR_FINAL_CONFIRMATION_FORMAL_SESSION_MODE,
        }:
            raise ValueError(
                "Stair unified protocol is draft; formal training and evaluation are disabled"
            )
        provisional_plugin = StairClimbingTaskPlugin(task_config)
        resolved_cell_manifest = resolve_stair_cell_manifest(
            cell_variant_id=composition_config.cell_variant_id,
            config=task_config,
            plugin=provisional_plugin,
        )
        if task_config.is_unified_protocol:
            task_config = composition_config.to_task_config(
                project_root=context.project_root,
                resolved_cell_task_protocol_hash=(
                    resolved_cell_manifest.resolved_cell_task_protocol_hash
                ),
            )
        training_binding = task_config.formal_training_binding()
        start_profile = resolve_stair_session_start_profile(
            composition_config.start_mode,
            task_protocol_hash=task_config.task_protocol_hash,
            training_binding_hash=str(training_binding.binding_hash),
        )
        legacy_live_confirmation_commitment: Mapping[str, Any] | None = None
        mode = values.get("mode")
        if mode in {
            "live",
            STAIR_FINAL_CONFIRMATION_FORMAL_SESSION_MODE,
        } and start_profile.genesis_anchor_kind == "reference_program":
            session_id = values.get("session_id")
            if not isinstance(session_id, str) or not session_id:
                raise ValueError(
                    "Stair final confirmation requires an explicit session_id"
                )
            if mode == "live":
                if values.get("rounds") != STAIR_FINAL_CONFIRMATION_REQUIRED_ROUNDS:
                    raise ValueError(
                        "live Stair final confirmation requires exactly 15 rounds"
                    )
            raw_plan_path = values.get("final_confirmation_plan")
            if not isinstance(raw_plan_path, Path):
                raise ValueError(
                    "Stair requires --final-confirmation-plan before Genesis"
                )
            plan_path = Path(
                os.path.abspath(
                    raw_plan_path
                    if raw_plan_path.is_absolute()
                    else context.project_root / raw_plan_path
                )
            )
            expected_plan_path = Path(
                os.path.abspath(
                    stair_final_confirmation_plan_path(context.output_dir)
                )
            )
            if plan_path != expected_plan_path:
                raise ValueError(
                    "Stair final confirmation plan must belong to the live output_dir"
                )
            if mode == "live":
                confirmation_plan = load_stair_final_confirmation_plan(
                    plan_path,
                    expected_session_id=session_id,
                    expected_config=task_config,
                )
            else:
                confirmation_plan = read_stair_final_plan(
                    plan_path,
                    expected_session_id=session_id,
                    expected_config=task_config,
                )
                required_generations = (confirmation_plan.required_closed_generations
                    if isinstance(confirmation_plan, GenerationFinalPlan)
                    else STAIR_FINAL_CONFIRMATION_REQUIRED_GENERATIONS)
                if values.get("max_generations") != required_generations:
                    raise ValueError("formal Stair generation count differs from its final plan")
            # The width-one live path has no generation-session contract, so its
            # terminal plan still needs this direct Genesis commitment.  Formal
            # generation binds the plan through the generation-contract ref and
            # must keep that session-local commitment out of the reusable
            # scientific runtime snapshot.
            if mode == "live":
                legacy_live_confirmation_commitment = (
                    confirmation_plan.genesis_commitment()
                )
        elif values.get("final_confirmation_plan") is not None:
            raise ValueError(
                "Stair final confirmation plan is only valid for reference-program Genesis"
            )
        plugin = StairClimbingTaskPlugin(
            task_config,
            session_start_profile=start_profile,
            resolved_cell_manifest=resolved_cell_manifest,
        )
        final_cell_manifest = resolve_stair_cell_manifest(
            cell_variant_id=composition_config.cell_variant_id,
            config=task_config,
            plugin=plugin,
        )
        if final_cell_manifest.to_dict() != resolved_cell_manifest.to_dict():
            raise ValueError("Stair cell manifest changed after final task resolution")
        resolved_cell_manifest = final_cell_manifest
        resolved_cell_manifest_path = materialize_resolved_cell_manifest(
            resolved_cell_manifest,
            output_dir=context.output_dir,
            require_existing=context.read_only,
        )
        candidate_admission_policy = StairCandidateAdmissionPolicy(
            task_config,
            resolved_cell_manifest=resolved_cell_manifest,
        )
        profile_hash = plugin.task_spec.candidate_capability_profile.profile_hash
        reference_program_artifact = self._resolve_reference_program_artifact(
            start_profile=start_profile,
            task_config=task_config,
            project_root=context.project_root,
            resolved_cell_manifest=resolved_cell_manifest,
        )
        if reference_program_artifact is not None:
            manifest = reference_program_artifact.manifest
            if manifest.capability_profile_hash != profile_hash:
                raise ValueError("stair reference capability profile mismatch")
        if start_profile.genesis_anchor_kind == "bootstrap":
            initial_best_score = values.get("initial_best_score", 0.0)
            if (
                isinstance(initial_best_score, bool)
                or not isinstance(initial_best_score, int | float)
                or float(initial_best_score)
                != float(start_profile.bootstrap_initial_best_score)
            ):
                raise ValueError(
                    "Stair reward cold start requires the profile initial_best_score"
                )
        backend = _build_stair_backend_config(
            composition_config=composition_config,
            context=context,
            task_config=task_config,
            plugin=plugin,
            candidate_admission_policy=candidate_admission_policy,
            resolved_cell_manifest=resolved_cell_manifest,
            resolved_cell_manifest_path=resolved_cell_manifest_path,
        )
        resolved_task_defaults = composition_config.to_resolved_task_defaults(
            start_profile=start_profile
        )
        resolved_task_defaults["cell_variant_id"] = (
            resolved_cell_manifest.cell_variant_id
        )
        if legacy_live_confirmation_commitment is not None:
            resolved_task_defaults["final_confirmation_precommit"] = dict(
                legacy_live_confirmation_commitment
            )
        return ResolvedTaskComposition(
            task_id=self.task_id,
            task_plugin=plugin,
            task_memory_adapter=StairTaskMemoryAdapter(
                config=task_config,
                objective_family_mask_provider=plugin.objective_family_mask,
            ),
            backend_adapter=IsaacLabRslRlBackendAdapter(backend),
            runner=SubprocessRunner(
                allowed_executables=(composition_config.isaac_python,),
                timeout_s=composition_config.runner_timeout_s,
                cwd=context.project_root,
                eval_checkpoint_admission_hook=(
                    build_stair_checkpoint_ancestry_admission(task_config, resolved_cell_manifest)
                    if task_config.is_unified_protocol else None
                ),
                eval_phase_summary_finalizer=(
                    partial(finalize_stair_eval_phase_summary, config=task_config,
                            behavior_feedback_profile=composition_config.behavior_feedback_profile,
                            trajectory_report_max_bytes=composition_config.trajectory_report_max_bytes)
                    if task_config.is_unified_protocol else None
                ),
            ),
            candidate_admission_policy=candidate_admission_policy,
            policy_initialization=PolicyInitialization.random(),
            genesis_anchor_kind=start_profile.genesis_anchor_kind,
            task_defaults=resolved_task_defaults,
            scientific_parameters={
                "training_agent_configuration_hash": backend.training_agent_configuration.agent_configuration_hash,
                "session_start_profile": start_profile.to_public_dict(),
                "behavior_feedback_profile": composition_config.behavior_feedback_profile,
                "trajectory_report_max_bytes": composition_config.trajectory_report_max_bytes,
                "trajectory_production_versions": PRODUCTION_VERSIONS,
                **({"final_confirmation_precommit": dict(legacy_live_confirmation_commitment)}
                   if legacy_live_confirmation_commitment is not None else {}),
            },
            execution_placement={
                "project_root": str(context.project_root.resolve()),
                "output_dir": str(context.output_dir.resolve()),
                "isaac_python": composition_config.isaac_python,
                "execution_device": composition_config.execution_device,
                "runner_timeout_s": composition_config.runner_timeout_s,
                "train_entrypoint": backend.train_entrypoint,
                "eval_entrypoint": backend.eval_entrypoint,
            },
            reference_program_artifact=reference_program_artifact,
            bootstrap_initial_best_score=(
                start_profile.bootstrap_initial_best_score
            ),
            bootstrap_initial_policy_state_hash=(
                start_profile.initial_policy_state_hash
            ),
            resolved_cell_manifest=resolved_cell_manifest,
        )

    def _resolve_reference_program_artifact(
        self,
        *,
        start_profile: StairSessionStartProfile,
        task_config: StairClimbingTaskConfig,
        project_root: Path,
        resolved_cell_manifest: ResolvedCellManifestV1,
    ) -> VerifiedReferenceProgramArtifact | None:
        injected = self._injected_reference_program_artifact
        if not start_profile.requires_reference_program:
            if injected is not None:
                raise ValueError(
                    "Stair reward cold start cannot carry an injected reference"
                )
            return None
        if injected is not None:
            return injected
        from true_rl_agent.tasks.stair_climbing.reference_program import (
            resolve_official_stair_reference_program,
        )

        return resolve_official_stair_reference_program(
            task_config,
            project_root=project_root,
            resolved_cell_manifest=resolved_cell_manifest,
        ).artifact


def _build_stair_backend_config(
    *,
    composition_config: StairClimbingCompositionConfig,
    context: TaskCompositionContext,
    task_config: StairClimbingTaskConfig,
    plugin: StairClimbingTaskPlugin,
    candidate_admission_policy: StairCandidateAdmissionPolicy,
    resolved_cell_manifest: ResolvedCellManifestV1,
    resolved_cell_manifest_path: Path,
) -> IsaacLabRslRlBackendConfig:
    if plugin.config is not task_config:
        raise ValueError("stair plugin config does not match composition config")
    evaluation_protocol = task_config.evaluation_protocol()
    binding = EvaluationBindingV1(
        eval_protocol_hash=evaluation_protocol.protocol_hash,
        acceptance_protocol_hash=task_config.acceptance_protocol.protocol_hash,
        task_protocol_hash=task_config.task_protocol_hash,
        seed_cohort_hash=task_config.seed_resolver().namespace_cohort_hash(
            STAIR_FORMAL_EVAL_NAMESPACE
        ),
    )
    if task_config.is_unified_protocol:
        robot_identity = str(
            resolved_cell_manifest.robot_profile["robot_profile_id"]
        )
        observation_schema_hash = str(
            resolved_cell_manifest.contract_hashes["observation_content_hash"]
        )
        action_schema_hash = str(
            resolved_cell_manifest.contract_hashes["action_content_hash"]
        )
        policy_architecture = "rsl_rl_ppo_mlp"
    else:
        assert task_config.protocol_v1 is not None
        compatibility = task_config.protocol_v1.to_dict()["policy_compatibility"]
        robot_identity = str(compatibility["robot"])
        observation_schema_hash = str(
            compatibility["policy_observation"]["schema_hash"]
        )
        action_schema_hash = str(compatibility["action"]["schema_hash"])
        policy_architecture = str(compatibility["policy_architecture"])
    admission = candidate_admission_policy.admission_policy_descriptor()
    return IsaacLabRslRlBackendConfig(
        capture_public_dev_numeric=task_config.is_unified_protocol,
        isaac_lab_python=composition_config.isaac_python,
        train_entrypoint=composition_config.train_entrypoint,
        eval_entrypoint=composition_config.eval_entrypoint,
        supported_tasks=frozenset({STAIR_TASK_ID}),
        default_budget_steps=task_config.budget_steps,
        public_seed_count=task_config.public_seed_count,
        public_accept_seed_count=task_config.public_accept_seed_count,
        sealed_validation_seed_count=task_config.sealed_validation_seed_count,
        eval_num_envs=task_config.eval_num_envs,
        eval_protocol_version=task_config.eval_protocol_version,
        evaluation_binding=binding,
        robot=robot_identity,
        observation_schema_hash=observation_schema_hash,
        action_schema_hash=action_schema_hash,
        policy_architecture=policy_architecture,
        artifact_root=str(context.output_dir / "runner"),
        train_profile=task_config.train_profile,
        eval_profile=task_config.eval_profile,
        train_num_envs=task_config.train_num_envs,
        training_agent_configuration=task_config.training_agent_configuration(),
        train_max_iterations=task_config.train_max_iterations,
        train_steps_per_iteration=task_config.train_steps_per_iteration,
        eval_base_seed=task_config.seed_resolver().base_seed,
        execution_device=composition_config.execution_device,
        candidate_admission_policy_id=admission.policy_id,
        candidate_admission_policy_hash=admission.policy_hash,
        candidate_execution_capability_ids=frozenset(
            {COMPLETE_REWARD_PROGRAM_CAPABILITY_ID}
        ),
        candidate_execution_authorization_schema=(
            admission.execution_authorization_schema_version
        ),
        candidate_failure_contract=STAIR_CANDIDATE_FAILURE_CONTRACT,
        training_binding_task_protocol_hash=(
            task_config.training_binding_task_protocol_hash
            if task_config.is_unified_protocol
            else None
        ),
        training_predecessor_manifest_hash=(
            task_config.training_predecessor_manifest_hash
            if task_config.is_unified_protocol
            else None
        ),
        train_seed=task_config.train_seed,
        runtime_config_path=composition_config.runtime_config,
        train_runtime_dynamics_hash=(
            task_config.train_runtime_profile().runtime_dynamics_hash
        ),
        eval_runtime_dynamics_hash=(
            task_config.eval_runtime_profile().runtime_dynamics_hash
        ),
        work_dir=str(context.project_root),
        curriculum_capabilities=StairClimbingTaskPlugin.curriculum_capabilities,
        resolved_cell_manifest_path=str(resolved_cell_manifest_path),
        resolved_cell_manifest_hash=resolved_cell_manifest.manifest_hash,
    )


__all__ = [
    "DEFAULT_STAIR_EVAL_ENTRYPOINT",
    "DEFAULT_STAIR_PROTOCOL",
    "DEFAULT_STAIR_TRAIN_ENTRYPOINT",
    "StairClimbingCompositionConfig",
    "StairClimbingCompositionProvider",
]
