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
    StairProductionRuntimeProfile,
    StairRuntimeProfile,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE,
    STAIR_FORMAL_TRAIN_NAMESPACE,
    STAIR_PILOT_EVAL_NAMESPACE,
    STAIR_PRIVATE_SEED_RESOLVER_VERSION,
    STAIR_TRAIN_SEED_BASE,
    StairEvaluationSeedResolver,
    StairPrivateSeedResolver,
    StairPrivateTrainingSeedResolver,
    assert_seed_namespaces_disjoint,
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

STAIR_G1_ROUGH_AGENT_CONFIGURATION = RslRlAgentConfiguration(
    policy={
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    algorithm={
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.008,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
)


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
    protocol_v1: StairProtocolV1 = field(init=False, repr=False, compare=False)
    protocol_v2: StairProtocolV2 | StairProtocolV2Draft = field(
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

    def __post_init__(self) -> None:
        path = resolve_existing_file(
            self.protocol_v2_path,
            field_name="stair protocol_v2_path",
        )
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
        evaluation_profile = (
            StairEvaluationProfile.load(
                path.with_name(Path(STAIR_EVALUATION_PROFILE_RELATIVE_PATH).name),
                base_protocol=protocol_v1,
            )
            if isinstance(protocol_v2, StairProtocolV2)
            else None
        )
        acceptance_protocol = (
            AcceptanceProtocol.from_mapping(
                load_strict_json_object(
                    path.with_name(
                        Path(STAIR_ACCEPTANCE_PROTOCOL_RELATIVE_PATH).name
                    ),
                    context="Stair active acceptance protocol",
                )
            )
            if isinstance(protocol_v2, StairProtocolV2)
            else protocol_v1.acceptance_protocol
        )
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
            raise ValueError("stair eval_protocol_version does not match active protocol")
        if formal["private_seed_resolver_version"] != STAIR_PRIVATE_SEED_RESOLVER_VERSION:
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
            expected_seed = resolver.derive_seed(
                namespace=STAIR_FORMAL_TRAIN_NAMESPACE
            )
            if self.train_seed != expected_seed:
                raise ValueError("stair train_seed is outside the formal seed domain")
            if (
                production["initialization_seed_schedule_hash"]
                != initialization_seed_schedule_hash(self.train_seed)
                or production["private_training_seed_resolver_hash"]
                != fixed_training_seed_resolver_hash(self.train_seed)
            ):
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
                raise ValueError(
                    f"stair split episodes-per-bin mismatch: {split}"
                )
        timeout_signal = self.episode_outcomes()["signals"]["timeout"]
        if evaluation_profile is None:
            timeout_s = float(timeout_signal["threshold"])
            if not math.isclose(
                timeout_s,
                float(evaluation["episode_length_s"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise ValueError("stair timeout threshold does not match episode horizon")
        elif timeout_signal.get("deadline_authority") != "stair_evaluation_profile":
            raise ValueError("stair active timeout authority differs from evaluation profile")
        assert_seed_namespaces_disjoint(self.seed_resolver())

    @classmethod
    def defaults(
        cls,
        protocol_v2_path: Path,
    ) -> "StairClimbingTaskConfig":
        protocol_v2_path = resolve_existing_file(
            protocol_v2_path,
            field_name="stair protocol_v2_path",
        )
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
        if self.evaluation_semantics is None:
            return self.protocol_v2.protocol_hash
        if not isinstance(self.protocol_v2, StairProtocolV2):
            raise TypeError("production Stair evaluation semantics require final V2")
        return effective_stair_task_protocol_hash(
            training_protocol=self.protocol_v2,
            evaluation_semantics=self.evaluation_semantics,
            acceptance_protocol=self.acceptance_protocol,
            evaluation_profile_hash=self.evaluation_profile.profile_hash,
        )

    def episode_outcomes(self) -> dict[str, Any]:
        if self.evaluation_semantics is not None:
            return dict(self.evaluation_semantics.to_dict()["episode_outcomes"])
        return dict(self.protocol_v1.to_dict()["episode_outcomes"])

    @property
    def evaluation_profile(self) -> StairEvaluationProfile:
        if self.evaluation_profile_config is None:
            raise ValueError("Stair active evaluation profile is unavailable")
        return self.evaluation_profile_config

    def base_seed_resolver(self) -> StairPrivateSeedResolver:
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
    ) -> StairPrivateSeedResolver | StairEvaluationSeedResolver:
        legacy = self.base_seed_resolver()
        if self.evaluation_profile_config is None:
            return legacy
        return StairEvaluationSeedResolver(
            legacy_resolver=legacy,
            evaluation_profile=self.evaluation_profile_config,
        )

    def training_agent_configuration(self) -> RslRlAgentConfiguration:
        return STAIR_G1_ROUGH_AGENT_CONFIGURATION

    def train_runtime_profile(
        self,
    ) -> StairRuntimeProfile | StairProductionRuntimeProfile:
        if isinstance(self.protocol_v2, StairProtocolV2):
            return StairProductionRuntimeProfile(
                mode="train",
                profile_id=self.train_profile,
                base_protocol_hash=self.protocol_v1.protocol_hash,
                # Evaluation semantics do not change training dynamics.
                task_protocol_hash=self.protocol_v2.protocol_hash,
            )
        return StairRuntimeProfile(
            mode="train",
            profile_id=self.train_profile,
            base_protocol_hash=self.protocol_v1.protocol_hash,
            draft_protocol_hash=self.protocol_v2.protocol_hash,
        )

    def eval_runtime_profile(
        self,
    ) -> StairRuntimeProfile | StairProductionRuntimeProfile:
        if isinstance(self.protocol_v2, StairProtocolV2):
            return StairProductionRuntimeProfile(
                mode="eval",
                profile_id=self.eval_profile,
                base_protocol_hash=self.protocol_v1.protocol_hash,
                task_protocol_hash=self.task_protocol_hash,
            )
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
    "STAIR_EVAL_BASE_SEED",
    "STAIR_EVALUATION_SEMANTICS_RELATIVE_PATH",
    "STAIR_G1_ROUGH_AGENT_CONFIGURATION",
    "STAIR_PROTOCOL_V2_RELATIVE_PATH",
    "STAIR_TASK_ID",
    "StairClimbingTaskConfig",
]
