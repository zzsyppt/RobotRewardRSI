"""Strict task-owned configuration for production Stair composition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.robots.isaaclab_locomotion import (
    G1_ROUGH_AGENT_CONFIGURATION,
    isaaclab_locomotion_runtime_spec,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    STAIR_EVALUATION_PROFILE_RELATIVE_PATH,
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    STAIR_EVALUATION_SEMANTICS_RELATIVE_PATH,
    StairEvaluationSemantics,
    effective_stair_task_protocol_hash,
)
from true_rl_agent.tasks.stair_climbing.local_paths import resolve_existing_file
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairProtocolV2,
    StairProtocolV2Draft,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_PROFILE,
    STAIR_PILOT_EVAL_PROFILE,
    STAIR_PILOT_TRAIN_PROFILE,
    STAIR_TRAIN_PROFILE,
    STAIR_UNIFIED_EVAL_PROFILE,
    STAIR_UNIFIED_TRAIN_PROFILE,
    StairProductionRuntimeProfile,
    StairRuntimeProfile,
)
from true_rl_agent.tasks.stair_climbing.row_catalog import (
    STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID,
    frozen_stair_predecessor_row,
    stair_robot_row,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_FORMAL_TRAIN_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_PRIVATE_SEED_RESOLVER_VERSION,
    STAIR_TRAIN_SEED_BASE,
    StairEvaluationSeedResolver,
    StairOrderedEvaluationSeedResolver,
    StairPrivateSeedResolver,
    StairPrivateTrainingSeedResolver,
    assert_seed_namespaces_disjoint,
)
from true_rl_agent.tasks.stair_climbing.unified_evaluation import (
    StairUnifiedEvaluationSuite,
)
from true_rl_agent.tasks.stair_climbing.unified_protocol import (
    STAIR_UNIFIED_PROTOCOL_SCHEMA_VERSION,
    StairUnifiedProtocol,
)
from true_rl_agent.tasks.stair_climbing.unified_training import (
    StairUnifiedTrainingProfile,
)
from true_rl_agent.tasks.stair_climbing.vnext_reward_authority import (
    StairVNextRewardAuthority,
)
from true_rl_agent.training_binding import (
    TrainingBinding,
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)

STAIR_TASK_ID = "unitree_stair_climbing"
STAIR_EVAL_BASE_SEED = 170_100
STAIR_PROTOCOL_V2_RELATIVE_PATH = "configs/tasks/stair_protocol_v2.json"
STAIR_ACCEPTANCE_PROTOCOL_RELATIVE_PATH = (
    "configs/tasks/stair_acceptance_protocol_v5.json"
)
EXPECTED_STAIR_ACCEPTANCE_PROTOCOL_HASH = (
    "9fccce3984eed8f04602a07511d7afb79cedf8dc8732fe56b7a18d804ef26382"
)
_STAIR_PILOT_PROTOCOL_FILENAME = "stair_protocol_v2_draft_r2.json"

STAIR_G1_ROUGH_AGENT_CONFIGURATION = G1_ROUGH_AGENT_CONFIGURATION


@dataclass(frozen=True)
class StairClimbingTaskConfig:
    """One validated source for Stair task, backend, and runner settings."""

    protocol_v2_path: Path
    budget_steps: int
    train_num_envs: int
    train_max_iterations: int
    train_steps_per_iteration: int
    eval_num_envs: int
    public_seed_count: int
    public_accept_seed_count: int
    sealed_validation_seed_count: int
    eval_protocol_version: str
    train_seed: int
    runner_timeout_s: int
    train_profile: str
    eval_profile: str
    cell_variant_id: str = STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID
    resolved_cell_task_protocol_hash: str | None = None
    protocol_v1: StairProtocolV1 | None = field(init=False, repr=False, compare=False)
    protocol_v2: StairProtocolV2 | StairProtocolV2Draft | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    evaluation_semantics: StairEvaluationSemantics | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    acceptance_protocol: AcceptanceProtocol = field(
        init=False,
        repr=False,
        compare=False,
    )
    evaluation_profile_config: StairEvaluationProfile | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    ordered_evaluation_suite_config: StairUnifiedEvaluationSuite | None = field(
        init=False,
        repr=False,
        compare=False,
    )
    unified_protocol_config: StairUnifiedProtocol | None = field(
        init=False, repr=False, compare=False
    )
    unified_training_profile_config: StairUnifiedTrainingProfile | None = field(
        init=False, repr=False, compare=False
    )
    vnext_reward_authority_config: StairVNextRewardAuthority | None = field(
        init=False, repr=False, compare=False
    )
    _base_task_protocol_hash: str = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        path = resolve_existing_file(
            self.protocol_v2_path,
            field_name="stair protocol_v2_path",
        )
        if (
            _stair_protocol_schema_version(path)
            == STAIR_UNIFIED_PROTOCOL_SCHEMA_VERSION
        ):
            self._initialize_unified()
            return
        if self.cell_variant_id != STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID:
            raise ValueError("unsupported Stair cell_variant_id")
        protocol_v2 = _load_stair_protocol_v2(path)
        protocol_v1 = StairProtocolV1.load(path.with_name("stair_protocol_v1.json"))
        evaluation_semantics = (
            StairEvaluationSemantics.load(
                path.with_name(Path(STAIR_EVALUATION_SEMANTICS_RELATIVE_PATH).name),
                base_protocol=protocol_v1,
                training_protocol=protocol_v2,
            )
            if isinstance(protocol_v2, StairProtocolV2)
            else None
        )
        legacy_evaluation_profile = (
            StairEvaluationProfile.load(
                path.with_name(Path(STAIR_EVALUATION_PROFILE_RELATIVE_PATH).name),
                base_protocol=protocol_v1,
            )
            if isinstance(protocol_v2, StairProtocolV2)
            else None
        )
        clean_acceptance_protocol = (
            AcceptanceProtocol.from_mapping(
                load_strict_json_object(
                    path.with_name(Path(STAIR_ACCEPTANCE_PROTOCOL_RELATIVE_PATH).name),
                    context="Stair active acceptance protocol",
                )
            )
            if isinstance(protocol_v2, StairProtocolV2)
            else protocol_v1.acceptance_protocol
        )
        if (
            isinstance(protocol_v2, StairProtocolV2)
            and clean_acceptance_protocol.protocol_hash
            != EXPECTED_STAIR_ACCEPTANCE_PROTOCOL_HASH
        ):
            raise ValueError("Stair clean acceptance protocol trust anchor mismatch")
        if isinstance(protocol_v2, StairProtocolV2):
            assert evaluation_semantics is not None
            assert legacy_evaluation_profile is not None
            base_task_protocol_hash = effective_stair_task_protocol_hash(
                training_protocol=protocol_v2,
                evaluation_semantics=evaluation_semantics,
                acceptance_protocol=clean_acceptance_protocol,
                evaluation_profile_hash=legacy_evaluation_profile.profile_hash,
            )
        else:
            base_task_protocol_hash = protocol_v2.protocol_hash
        evaluation_profile = legacy_evaluation_profile
        acceptance_protocol = clean_acceptance_protocol
        if (
            isinstance(protocol_v2, StairProtocolV2)
            and acceptance_protocol.protocol_hash
            != EXPECTED_STAIR_ACCEPTANCE_PROTOCOL_HASH
        ):
            raise ValueError("Stair active acceptance protocol trust anchor mismatch")
        if acceptance_protocol.min_score_count != self.public_accept_seed_count:
            raise ValueError(
                "Stair acceptance sample count must equal public_accept seed count"
            )
        object.__setattr__(self, "protocol_v2_path", path)
        object.__setattr__(self, "protocol_v1", protocol_v1)
        object.__setattr__(self, "protocol_v2", protocol_v2)
        object.__setattr__(self, "evaluation_semantics", evaluation_semantics)
        object.__setattr__(self, "acceptance_protocol", acceptance_protocol)
        object.__setattr__(
            self,
            "evaluation_profile_config",
            evaluation_profile,
        )
        object.__setattr__(
            self,
            "ordered_evaluation_suite_config",
            None,
        )
        object.__setattr__(self, "unified_protocol_config", None)
        object.__setattr__(self, "unified_training_profile_config", None)
        object.__setattr__(self, "vnext_reward_authority_config", None)
        object.__setattr__(
            self,
            "_base_task_protocol_hash",
            base_task_protocol_hash,
        )
        if self.resolved_cell_task_protocol_hash is not None:
            from true_rl_agent.evaluation.binding import require_sha256_digest

            require_sha256_digest(
                self.resolved_cell_task_protocol_hash,
                "Stair resolved cell task protocol hash",
            )
            raise ValueError("legacy Stair config cannot carry resolved cell hash")

        if protocol_v1.task_id != STAIR_TASK_ID:
            raise ValueError("Stair V1 task_id mismatch")
        if protocol_v2.to_public_dict()["task_id"] != STAIR_TASK_ID:
            raise ValueError("Stair V2 task_id mismatch")
        for field_name in (
            "budget_steps",
            "train_num_envs",
            "train_max_iterations",
            "train_steps_per_iteration",
            "eval_num_envs",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "runner_timeout_s",
        ):
            _positive_int(getattr(self, field_name), field_name)
        if (
            isinstance(self.train_seed, bool)
            or not isinstance(self.train_seed, int)
            or self.train_seed < 0
        ):
            raise ValueError("train_seed must be a non-negative integer")
        for field_name in ("eval_protocol_version", "train_profile", "eval_profile"):
            _non_empty_text(getattr(self, field_name), field_name)

        payload_v1 = protocol_v1.to_dict()
        payload_v2 = protocol_v2.to_public_dict()
        training = payload_v1["training"]
        evaluation = payload_v1["evaluation"]
        formal = payload_v2["formal_protocol"]

        expected_eval_protocol_version = (
            evaluation_profile.evaluation_protocol.eval_protocol_version
            if evaluation_profile is not None
            else formal["evaluation_protocol_version"]
        )
        if self.eval_protocol_version != expected_eval_protocol_version:
            raise ValueError(
                "stair eval_protocol_version does not match active protocol"
            )
        if (
            formal["private_seed_resolver_version"]
            != STAIR_PRIVATE_SEED_RESOLVER_VERSION
        ):
            raise ValueError("stair private seed resolver version drifted")
        expected_train_profile, expected_eval_profile = _profile_ids(protocol_v2)
        if self.train_profile != expected_train_profile:
            raise ValueError("stair train_profile is not the frozen profile")
        if self.eval_profile != expected_eval_profile:
            raise ValueError("stair eval_profile is not the frozen profile")

        if isinstance(protocol_v2, StairProtocolV2Draft):
            expected_budget = (
                protocol_v2.pilot_max_iterations
                * int(training["num_envs"])
                * int(training["steps_per_env"])
            )
            if self.budget_steps != expected_budget:
                raise ValueError(
                    "stair budget_steps differs from fixed V2 pilot budget"
                )
            if self.train_max_iterations != protocol_v2.pilot_max_iterations:
                raise ValueError(
                    "stair max_iterations differs from fixed V2 pilot budget"
                )
        elif self.budget_steps != protocol_v2.fixed_budget_steps:
            raise ValueError("stair budget_steps does not match final V2")
        if self.train_num_envs != int(training["num_envs"]):
            raise ValueError("stair train_num_envs does not match base protocol")
        if self.train_steps_per_iteration != int(training["steps_per_env"]):
            raise ValueError(
                "stair train_steps_per_iteration does not match base protocol"
            )
        build_budget_accounting(
            budget_steps=self.budget_steps,
            num_envs=self.train_num_envs,
            num_steps_per_env=self.train_steps_per_iteration,
            max_iterations=self.train_max_iterations,
        )
        if isinstance(protocol_v2, StairProtocolV2):
            production = payload_v2["production_training"]
            production_values = {
                "budget_steps": self.budget_steps,
                "num_envs": self.train_num_envs,
                "steps_per_env": self.train_steps_per_iteration,
                "max_iterations": self.train_max_iterations,
                "runner_timeout_s": self.runner_timeout_s,
                "train_profile": self.train_profile,
                "eval_profile": self.eval_profile,
            }
            for key, actual in production_values.items():
                if production[key] != actual:
                    raise ValueError(
                        f"stair production training {key} differs from final V2"
                    )
            resolver = StairPrivateTrainingSeedResolver(
                base_seed=STAIR_TRAIN_SEED_BASE,
                task_protocol_hash=protocol_v2.seed_anchor_hash,
            )
            expected_seed = resolver.derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
            if self.train_seed != expected_seed:
                raise ValueError("stair train_seed is outside the formal seed domain")
            if production[
                "initialization_seed_schedule_hash"
            ] != initialization_seed_schedule_hash(self.train_seed) or production[
                "private_training_seed_resolver_hash"
            ] != fixed_training_seed_resolver_hash(self.train_seed):
                raise ValueError("stair formal training seed binding drifted")
            agent = self.training_agent_configuration()
            expected_binding = TrainingBinding(
                initialization_mode="random",
                initialization_checkpoint_hash=None,
                policy_architecture_hash=str(agent.policy_architecture_hash),
                agent_configuration_hash=str(agent.agent_configuration_hash),
                initialization_seed_schedule_hash=(
                    initialization_seed_schedule_hash(self.train_seed)
                ),
                optimizer_training_profile_hash=str(
                    agent.optimizer_training_profile_hash
                ),
                budget_steps=self.budget_steps,
                private_training_seed_resolver_hash=(
                    fixed_training_seed_resolver_hash(self.train_seed)
                ),
                capability_profile_hash=protocol_v2.capability_profile.profile_hash,
            )
            binding_checks = {
                "policy_architecture_hash": agent.policy_architecture_hash,
                "optimizer_training_profile_hash": (
                    agent.optimizer_training_profile_hash
                ),
                "agent_configuration_hash": agent.agent_configuration_hash,
                "capability_profile_hash": protocol_v2.capability_profile.profile_hash,
                "formal_training_binding_hash": expected_binding.binding_hash,
            }
            for key, expected in binding_checks.items():
                if production[key] != expected:
                    raise ValueError(f"stair production {key} binding drifted")
            eval_resolver = self.base_seed_resolver()
            if formal["formal_evaluation_seed_cohort_hash"] != (
                eval_resolver.namespace_cohort_hash(STAIR_FORMAL_EVAL_NAMESPACE)
            ) or formal["pilot_evaluation_seed_cohort_hash"] != (
                eval_resolver.namespace_cohort_hash(STAIR_PILOT_EVAL_NAMESPACE)
            ):
                raise ValueError("stair final evaluation seed cohort binding drifted")

        frozen_eval = (
            evaluation_profile.evaluation_protocol
            if evaluation_profile is not None
            else protocol_v1.evaluation_protocol
        )
        expected_counts = {
            "public_seed_count": frozen_eval.public_seed_count,
            "public_accept_seed_count": frozen_eval.public_accept_seed_count,
            "sealed_validation_seed_count": frozen_eval.sealed_validation_seed_count,
            "eval_num_envs": frozen_eval.eval_num_envs,
        }
        for field_name, expected in expected_counts.items():
            if getattr(self, field_name) != expected:
                raise ValueError(f"stair {field_name} does not match base protocol")

        bin_count = len(protocol_v1.eval_bins)
        envs_per_bin = int(evaluation["env_to_bin_mapping"]["envs_per_bin"])
        expected_env_count = (
            len(evaluation_profile.cells)
            * evaluation_profile.episodes_per_seed_per_cell
            if evaluation_profile is not None
            else bin_count * envs_per_bin
        )
        if expected_env_count != self.eval_num_envs:
            raise ValueError("stair eval env count does not cover the active courses")
        for split, split_cfg in evaluation["splits"].items():
            if int(split_cfg["episodes_per_seed_per_bin"]) != envs_per_bin:
                raise ValueError(f"stair split episodes-per-bin mismatch: {split}")
        timeout_signal = self.episode_outcomes()["signals"]["timeout"]
        if evaluation_profile is None:
            timeout_s = float(timeout_signal["threshold"])
            if not math.isclose(
                timeout_s,
                float(evaluation["episode_length_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError(
                    "stair timeout threshold does not match episode horizon"
                )
        elif timeout_signal.get("deadline_authority") != "stair_evaluation_profile":
            raise ValueError(
                "stair active timeout authority differs from evaluation profile"
            )
        assert_seed_namespaces_disjoint(self.seed_resolver())

    def _initialize_unified(self) -> None:
        path = resolve_existing_file(
            self.protocol_v2_path,
            field_name="Stair unified protocol path",
        )
        protocol = StairUnifiedProtocol.load(
            path,
            expected_cell_variant_id=self.cell_variant_id,
        )
        frozen_row = frozen_stair_predecessor_row(self.cell_variant_id)
        if frozen_row is not None:
            frozen_row.require_frozen_protocol_identity(protocol.task_protocol_hash)
        if self.cell_variant_id != protocol.cell_variant_id:
            raise ValueError("Stair unified config/cell identity mismatch")
        training = protocol.training
        evaluation = protocol.evaluation_suite.evaluation_protocol
        expected = {
            "budget_steps": int(training["budget_steps"]),
            "train_num_envs": int(training["num_envs"]),
            "train_max_iterations": int(training["max_iterations"]),
            "train_steps_per_iteration": int(training["steps_per_env"]),
            "eval_num_envs": evaluation.eval_num_envs,
            "public_seed_count": evaluation.public_seed_count,
            "public_accept_seed_count": evaluation.public_accept_seed_count,
            "sealed_validation_seed_count": evaluation.sealed_validation_seed_count,
            "eval_protocol_version": evaluation.eval_protocol_version,
            "runner_timeout_s": int(training["runner_timeout_s"]),
            "train_profile": str(training["train_profile"]),
            "eval_profile": str(training["eval_profile"]),
        }
        for name, value in expected.items():
            if getattr(self, name) != value:
                raise ValueError(f"Stair unified {name} differs from protocol")
        if self.train_profile != STAIR_UNIFIED_TRAIN_PROFILE:
            raise ValueError("Stair unified train profile is not canonical")
        if self.eval_profile != STAIR_UNIFIED_EVAL_PROFILE:
            raise ValueError("Stair unified eval profile is not canonical")
        for name in (
            "budget_steps",
            "train_num_envs",
            "train_max_iterations",
            "train_steps_per_iteration",
            "eval_num_envs",
            "public_seed_count",
            "public_accept_seed_count",
            "sealed_validation_seed_count",
            "runner_timeout_s",
        ):
            _positive_int(getattr(self, name), name)
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
        if (
            protocol.acceptance_protocol.min_score_count
            != self.public_accept_seed_count
        ):
            raise ValueError("Stair unified acceptance cohort size mismatch")
        if self.resolved_cell_task_protocol_hash is not None:
            from true_rl_agent.evaluation.binding import require_sha256_digest

            require_sha256_digest(
                self.resolved_cell_task_protocol_hash,
                "Stair resolved cell task protocol hash",
            )
        object.__setattr__(self, "protocol_v2_path", path)
        object.__setattr__(self, "protocol_v1", None)
        object.__setattr__(self, "protocol_v2", None)
        object.__setattr__(
            self,
            "evaluation_semantics",
            StairEvaluationSemantics.from_task_outcomes(
                task_protocol_hash=protocol.task_protocol_hash,
                task_id=protocol.task_id,
                episode_outcomes=protocol.episode_outcomes,
            ),
        )
        object.__setattr__(self, "acceptance_protocol", protocol.acceptance_protocol)
        object.__setattr__(self, "evaluation_profile_config", None)
        object.__setattr__(
            self, "ordered_evaluation_suite_config", protocol.evaluation_suite
        )
        object.__setattr__(self, "unified_protocol_config", protocol)
        object.__setattr__(
            self, "unified_training_profile_config", protocol.training_profile
        )
        object.__setattr__(
            self, "vnext_reward_authority_config", protocol.reward_authority
        )
        object.__setattr__(
            self, "_base_task_protocol_hash", protocol.task_protocol_hash
        )
        assert_seed_namespaces_disjoint(self.seed_resolver())

    @classmethod
    def defaults(
        cls,
        protocol_v2_path: Path,
        *,
        cell_variant_id: str | None = None,
        resolved_cell_task_protocol_hash: str | None = None,
    ) -> "StairClimbingTaskConfig":
        protocol_v2_path = resolve_existing_file(
            protocol_v2_path,
            field_name="stair protocol_v2_path",
        )
        if (
            _stair_protocol_schema_version(protocol_v2_path)
            == STAIR_UNIFIED_PROTOCOL_SCHEMA_VERSION
        ):
            protocol = StairUnifiedProtocol.load(protocol_v2_path)
            if cell_variant_id is None:
                cell_variant_id = protocol.cell_variant_id
            if cell_variant_id != protocol.cell_variant_id:
                raise ValueError("Stair unified defaults received another cell")
            training = protocol.training
            evaluation_protocol = protocol.evaluation_suite.evaluation_protocol
            train_seed = StairPrivateTrainingSeedResolver(
                base_seed=STAIR_TRAIN_SEED_BASE,
                task_protocol_hash=protocol.training_seed_anchor_hash,
            ).derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
            return cls(
                protocol_v2_path=protocol_v2_path,
                budget_steps=int(training["budget_steps"]),
                train_num_envs=int(training["num_envs"]),
                train_max_iterations=int(training["max_iterations"]),
                train_steps_per_iteration=int(training["steps_per_env"]),
                eval_num_envs=evaluation_protocol.eval_num_envs,
                public_seed_count=evaluation_protocol.public_seed_count,
                public_accept_seed_count=evaluation_protocol.public_accept_seed_count,
                sealed_validation_seed_count=(
                    evaluation_protocol.sealed_validation_seed_count
                ),
                eval_protocol_version=evaluation_protocol.eval_protocol_version,
                train_seed=train_seed,
                runner_timeout_s=int(training["runner_timeout_s"]),
                train_profile=str(training["train_profile"]),
                eval_profile=str(training["eval_profile"]),
                cell_variant_id=cell_variant_id,
                resolved_cell_task_protocol_hash=resolved_cell_task_protocol_hash,
            )
        if cell_variant_id is None:
            cell_variant_id = STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID
        protocol_v2 = _load_stair_protocol_v2(protocol_v2_path)
        protocol_v1 = StairProtocolV1.load(
            protocol_v2_path.with_name("stair_protocol_v1.json")
        )
        v1 = protocol_v1.to_dict()
        v2 = protocol_v2.to_public_dict()
        training = v1["training"]
        evaluation_profile = (
            StairEvaluationProfile.load(
                protocol_v2_path.with_name(
                    Path(STAIR_EVALUATION_PROFILE_RELATIVE_PATH).name
                ),
                base_protocol=protocol_v1,
            )
            if isinstance(protocol_v2, StairProtocolV2)
            else None
        )
        base_eval_protocol = protocol_v1.evaluation_protocol
        evaluation_protocol = (
            evaluation_profile.evaluation_protocol
            if evaluation_profile is not None
            else EvaluationProtocol(
                eval_protocol_version=str(
                    v2["formal_protocol"]["evaluation_protocol_version"]
                ),
                public_seed_count=base_eval_protocol.public_seed_count,
                public_accept_seed_count=base_eval_protocol.public_accept_seed_count,
                sealed_validation_seed_count=(
                    base_eval_protocol.sealed_validation_seed_count
                ),
                eval_num_envs=base_eval_protocol.eval_num_envs,
            )
        )
        if isinstance(protocol_v2, StairProtocolV2Draft):
            max_iterations = protocol_v2.pilot_max_iterations
            budget_steps = (
                max_iterations
                * int(training["num_envs"])
                * int(training["steps_per_env"])
            )
            train_seed = STAIR_TRAIN_SEED_BASE
        else:
            budget_steps = protocol_v2.fixed_budget_steps
            train_seed = StairPrivateTrainingSeedResolver(
                base_seed=STAIR_TRAIN_SEED_BASE,
                task_protocol_hash=protocol_v2.seed_anchor_hash,
            ).derive_seed(namespace=STAIR_FORMAL_TRAIN_NAMESPACE)
        train_profile, eval_profile = _profile_ids(protocol_v2)
        steps_per_iteration = int(training["steps_per_env"])
        num_envs = int(training["num_envs"])
        return cls(
            protocol_v2_path=protocol_v2_path,
            budget_steps=budget_steps,
            train_num_envs=num_envs,
            train_max_iterations=(
                max_iterations
                if isinstance(protocol_v2, StairProtocolV2Draft)
                else budget_steps // (num_envs * steps_per_iteration)
            ),
            train_steps_per_iteration=steps_per_iteration,
            eval_num_envs=evaluation_protocol.eval_num_envs,
            public_seed_count=evaluation_protocol.public_seed_count,
            public_accept_seed_count=evaluation_protocol.public_accept_seed_count,
            sealed_validation_seed_count=(
                evaluation_protocol.sealed_validation_seed_count
            ),
            eval_protocol_version=evaluation_protocol.eval_protocol_version,
            train_seed=train_seed,
            runner_timeout_s=int(training["runner_timeout_s"]),
            train_profile=train_profile,
            eval_profile=eval_profile,
            cell_variant_id=cell_variant_id,
            resolved_cell_task_protocol_hash=resolved_cell_task_protocol_hash,
        )

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_num_envs=self.eval_num_envs,
        )

    @property
    def task_protocol_hash(self) -> str:
        if self.resolved_cell_task_protocol_hash is not None:
            return self.resolved_cell_task_protocol_hash
        return self.base_task_protocol_hash

    @property
    def base_task_protocol_hash(self) -> str:
        if self.unified_protocol_config is not None:
            return self.unified_protocol_config.task_protocol_hash
        if self.evaluation_semantics is None:
            assert self.protocol_v2 is not None
            return self.protocol_v2.protocol_hash
        return self._base_task_protocol_hash

    @property
    def training_base_task_protocol_hash(self) -> str:
        if self.unified_protocol_config is not None:
            lineage = self.unified_protocol_config.training_lineage
            if lineage is not None:
                return str(lineage["source_base_task_protocol_hash"])
        return self.base_task_protocol_hash

    @property
    def training_binding_task_protocol_hash(self) -> str:
        if self.unified_protocol_config is not None:
            lineage = self.unified_protocol_config.training_lineage
            if lineage is not None:
                return str(lineage["source_resolved_cell_task_protocol_hash"])
        return self.task_protocol_hash

    @property
    def training_predecessor_manifest_hash(self) -> str | None:
        if self.unified_protocol_config is not None:
            lineage = self.unified_protocol_config.training_lineage
            if lineage is not None:
                return str(lineage["source_resolved_cell_manifest_hash"])
        return None

    def episode_outcomes(self) -> dict[str, Any]:
        if self.unified_protocol_config is not None:
            return self.unified_protocol_config.to_dict()["episode_outcomes"]
        if self.evaluation_semantics is not None:
            return dict(self.evaluation_semantics.to_dict()["episode_outcomes"])
        assert self.protocol_v1 is not None
        return dict(self.protocol_v1.to_dict()["episode_outcomes"])

    @property
    def evaluation_profile(self) -> StairEvaluationProfile:
        if self.evaluation_profile_config is None:
            raise ValueError("Stair active evaluation profile is unavailable")
        return self.evaluation_profile_config

    @property
    def ordered_evaluation_suite(
        self,
    ) -> StairUnifiedEvaluationSuite:
        if self.ordered_evaluation_suite_config is None:
            raise ValueError("Stair ordered evaluation suite is unavailable")
        return self.ordered_evaluation_suite_config

    @property
    def unified_protocol(self) -> StairUnifiedProtocol:
        if self.unified_protocol_config is None:
            raise ValueError("Stair unified protocol is unavailable")
        return self.unified_protocol_config

    @property
    def unified_training_profile(self) -> StairUnifiedTrainingProfile:
        if self.unified_training_profile_config is None:
            raise ValueError("Stair unified training profile is unavailable")
        return self.unified_training_profile_config

    @property
    def vnext_reward_authority(self) -> StairVNextRewardAuthority:
        if self.vnext_reward_authority_config is None:
            raise ValueError("Stair VNext reward authority is unavailable")
        return self.vnext_reward_authority_config

    @property
    def is_unified_protocol(self) -> bool:
        return self.unified_protocol_config is not None

    @property
    def protocol_state(self) -> str:
        return (
            self.unified_protocol.protocol_state
            if self.is_unified_protocol
            else "archived"
        )

    @property
    def candidate_capability_profile(self):
        if self.is_unified_protocol:
            return self.unified_protocol.candidate_capability_profile
        assert self.protocol_v2 is not None
        return self.protocol_v2.capability_profile

    def base_seed_resolver(self) -> StairPrivateSeedResolver:
        if self.is_unified_protocol:
            raise ValueError("unified Stair uses the ordered seed resolver")
        assert self.protocol_v1 is not None
        assert self.protocol_v2 is not None
        evaluation = self.protocol_v1.to_dict()["evaluation"]
        base_eval = self.protocol_v1.evaluation_protocol
        legacy_eval_protocol = EvaluationProtocol(
            eval_protocol_version=str(
                self.protocol_v2.to_public_dict()["formal_protocol"][
                    "evaluation_protocol_version"
                ]
            ),
            public_seed_count=base_eval.public_seed_count,
            public_accept_seed_count=base_eval.public_accept_seed_count,
            sealed_validation_seed_count=base_eval.sealed_validation_seed_count,
            eval_num_envs=base_eval.eval_num_envs,
        )
        return StairPrivateSeedResolver(
            base_seed=STAIR_EVAL_BASE_SEED,
            task_protocol_hash=_seed_anchor_hash(self.protocol_v2),
            evaluation_protocol=legacy_eval_protocol,
            bin_ids=tuple(str(item["bin_id"]) for item in self.protocol_v1.eval_bins),
            episodes_per_seed_per_bin=int(
                evaluation["env_to_bin_mapping"]["envs_per_bin"]
            ),
        )

    def seed_resolver(
        self,
    ) -> (
        StairPrivateSeedResolver
        | StairEvaluationSeedResolver
        | StairOrderedEvaluationSeedResolver
    ):
        if self.ordered_evaluation_suite_config is not None:
            return StairOrderedEvaluationSeedResolver(
                base_seed=STAIR_EVAL_BASE_SEED,
                task_protocol_hash=self.base_task_protocol_hash,
                evaluation_suite=self.ordered_evaluation_suite_config,
            )
        legacy = self.base_seed_resolver()
        if self.evaluation_profile_config is None:
            return legacy
        return StairEvaluationSeedResolver(
            legacy_resolver=legacy,
            evaluation_profile=self.evaluation_profile_config,
        )

    def training_agent_configuration(self) -> RslRlAgentConfiguration:
        binding = stair_robot_row(self.cell_variant_id).load_binding()
        return isaaclab_locomotion_runtime_spec(
            binding.runtime_adapter_id
        ).agent_configuration

    def training_execution_contract(self) -> dict[str, Any]:
        """Return the active execution contract as ordinary JSON values."""

        if self.is_unified_protocol:
            payload = self.unified_protocol.to_dict()
            return {
                **payload["training_execution"],
                "checkpoint_selector": payload["training"]["checkpoint_selector"],
            }
        assert self.protocol_v2 is not None
        return self.protocol_v2.to_public_dict()["training_execution"]

    def formal_training_binding(self) -> TrainingBinding:
        agent = self.training_agent_configuration()
        return TrainingBinding(
            initialization_mode="random",
            initialization_checkpoint_hash=None,
            policy_architecture_hash=str(agent.policy_architecture_hash),
            agent_configuration_hash=str(agent.agent_configuration_hash),
            initialization_seed_schedule_hash=(
                self.initialization_seed_schedule_hash()
            ),
            optimizer_training_profile_hash=str(agent.optimizer_training_profile_hash),
            budget_steps=self.budget_steps,
            private_training_seed_resolver_hash=(
                self.private_training_seed_resolver_hash()
            ),
            capability_profile_hash=self.candidate_capability_profile.profile_hash,
            task_protocol_hash=(
                self.training_binding_task_protocol_hash
                if self.is_unified_protocol
                else None
            ),
        )

    def train_runtime_profile(
        self,
    ) -> StairRuntimeProfile | StairProductionRuntimeProfile:
        if self.is_unified_protocol:
            return StairProductionRuntimeProfile(
                mode="train",
                profile_id=self.train_profile,
                base_protocol_hash=self.training_base_task_protocol_hash,
                # The resolved cell hash includes this runtime profile through the
                # environment descriptor; binding it back here would be recursive.
                task_protocol_hash=self.training_base_task_protocol_hash,
            )
        if isinstance(self.protocol_v2, StairProtocolV2):
            assert self.protocol_v1 is not None
            return StairProductionRuntimeProfile(
                mode="train",
                profile_id=self.train_profile,
                base_protocol_hash=self.protocol_v1.protocol_hash,
                # Evaluation semantics do not change training dynamics.
                task_protocol_hash=self.protocol_v2.protocol_hash,
            )
        assert self.protocol_v1 is not None
        assert self.protocol_v2 is not None
        return StairRuntimeProfile(
            mode="train",
            profile_id=self.train_profile,
            base_protocol_hash=self.protocol_v1.protocol_hash,
            draft_protocol_hash=self.protocol_v2.protocol_hash,
        )

    def eval_runtime_profile(
        self,
    ) -> StairRuntimeProfile | StairProductionRuntimeProfile:
        if self.is_unified_protocol:
            return StairProductionRuntimeProfile(
                mode="eval",
                profile_id=self.eval_profile,
                base_protocol_hash=self.base_task_protocol_hash,
                task_protocol_hash=self.base_task_protocol_hash,
            )
        if isinstance(self.protocol_v2, StairProtocolV2):
            assert self.protocol_v1 is not None
            return StairProductionRuntimeProfile(
                mode="eval",
                profile_id=self.eval_profile,
                base_protocol_hash=self.protocol_v1.protocol_hash,
                task_protocol_hash=self.base_task_protocol_hash,
            )
        assert self.protocol_v1 is not None
        assert self.protocol_v2 is not None
        return StairRuntimeProfile(
            mode="eval",
            profile_id=self.eval_profile,
            base_protocol_hash=self.protocol_v1.protocol_hash,
            draft_protocol_hash=self.protocol_v2.protocol_hash,
        )

    def initialization_seed_schedule_hash(self) -> str:
        return initialization_seed_schedule_hash(self.train_seed)

    def private_training_seed_resolver_hash(self) -> str:
        return fixed_training_seed_resolver_hash(self.train_seed)


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


def _stair_protocol_schema_version(path: Path) -> str | None:
    payload = load_strict_json_object(path, context="Stair protocol dispatch")
    value = payload.get("schema_version")
    return value if isinstance(value, str) else None


def _load_stair_protocol_v2(path: Path) -> StairProtocolV2 | StairProtocolV2Draft:
    if path.name == _STAIR_PILOT_PROTOCOL_FILENAME:
        return StairProtocolV2Draft.load(path)
    return StairProtocolV2.load(path)


def _profile_ids(
    protocol: StairProtocolV2 | StairProtocolV2Draft,
) -> tuple[str, str]:
    if isinstance(protocol, StairProtocolV2Draft):
        return STAIR_PILOT_TRAIN_PROFILE, STAIR_PILOT_EVAL_PROFILE
    return STAIR_TRAIN_PROFILE, STAIR_EVAL_PROFILE


def _seed_anchor_hash(protocol: StairProtocolV2 | StairProtocolV2Draft) -> str:
    if isinstance(protocol, StairProtocolV2):
        return protocol.seed_anchor_hash
    return protocol.protocol_hash


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = [
    "EXPECTED_STAIR_ACCEPTANCE_PROTOCOL_HASH",
    "STAIR_ACCEPTANCE_PROTOCOL_RELATIVE_PATH",
    "STAIR_G1_37DOF_CLEAN_LEGACY_CELL_VARIANT_ID",
    "STAIR_EVAL_BASE_SEED",
    "STAIR_EVALUATION_SEMANTICS_RELATIVE_PATH",
    "STAIR_G1_ROUGH_AGENT_CONFIGURATION",
    "STAIR_PROTOCOL_V2_RELATIVE_PATH",
    "STAIR_TASK_ID",
    "StairClimbingTaskConfig",
]
