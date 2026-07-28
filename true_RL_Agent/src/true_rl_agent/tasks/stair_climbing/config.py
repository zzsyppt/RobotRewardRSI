"""Strict task-owned configuration for production Stair composition."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.evaluation.protocol import EvaluationProtocol
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import StairProtocolV2Draft
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_PROFILE,
    STAIR_TRAIN_PROFILE,
    StairRuntimeProfile,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_PRIVATE_SEED_RESOLVER_VERSION,
    StairPrivateSeedResolver,
    assert_seed_namespaces_disjoint,
)
from true_rl_agent.training_binding import (
    fixed_training_seed_resolver_hash,
    initialization_seed_schedule_hash,
)

STAIR_TASK_ID = "unitree_stair_climbing"
STAIR_EVAL_BASE_SEED = 170_100
STAIR_PROTOCOL_V2_RELATIVE_PATH = "configs/tasks/stair_protocol_v2_draft.json"

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
    protocol_v2: StairProtocolV2Draft = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.protocol_v2_path, Path):
            raise TypeError("stair protocol_v2_path must be a Path")
        path = self.protocol_v2_path.resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError("stair protocol_v2_path must be a regular file")
        protocol_v2 = StairProtocolV2Draft.load(path)
        protocol_v1 = StairProtocolV1.load(path.with_name("stair_protocol_v1.json"))
        object.__setattr__(self, "protocol_v2_path", path)
        object.__setattr__(self, "protocol_v1", protocol_v1)
        object.__setattr__(self, "protocol_v2", protocol_v2)

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
        pilot = payload_v2["pilot_training"]

        if self.eval_protocol_version != formal["evaluation_protocol_version"]:
            raise ValueError("stair eval_protocol_version does not match V2")
        if formal["private_seed_resolver_version"] != STAIR_PRIVATE_SEED_RESOLVER_VERSION:
            raise ValueError("stair private seed resolver version drifted")
        if self.train_profile != STAIR_TRAIN_PROFILE:
            raise ValueError("stair train_profile is not the frozen profile")
        if self.eval_profile != STAIR_EVAL_PROFILE:
            raise ValueError("stair eval_profile is not the frozen profile")

        if self.budget_steps not in tuple(pilot["allowed_budget_steps"]):
            raise ValueError("stair budget_steps is not an allowed V2 pilot budget")
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

        frozen_eval = protocol_v1.evaluation_protocol
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
        if bin_count * envs_per_bin != self.eval_num_envs:
            raise ValueError("stair eval env count does not cover every bin")
        for split, split_cfg in evaluation["splits"].items():
            if int(split_cfg["episodes_per_seed_per_bin"]) != envs_per_bin:
                raise ValueError(
                    f"stair split episodes-per-bin mismatch: {split}"
                )
        timeout_s = float(
            payload_v1["episode_outcomes"]["signals"]["timeout"]["threshold"]
        )
        if not math.isclose(
            timeout_s,
            float(evaluation["episode_length_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise ValueError("stair timeout threshold does not match episode horizon")
        assert_seed_namespaces_disjoint(self.seed_resolver())

    @classmethod
    def defaults(cls, protocol_v2_path: Path) -> "StairClimbingTaskConfig":
        protocol_v2 = StairProtocolV2Draft.load(protocol_v2_path)
        protocol_v1 = StairProtocolV1.load(
            protocol_v2_path.with_name("stair_protocol_v1.json")
        )
        v1 = protocol_v1.to_dict()
        v2 = protocol_v2.to_public_dict()
        training = v1["training"]
        evaluation_protocol = protocol_v1.evaluation_protocol
        budget_steps = int(v2["pilot_training"]["allowed_budget_steps"][0])
        steps_per_iteration = int(training["steps_per_env"])
        num_envs = int(training["num_envs"])
        return cls(
            protocol_v2_path=protocol_v2_path,
            budget_steps=budget_steps,
            train_num_envs=num_envs,
            train_max_iterations=budget_steps // (num_envs * steps_per_iteration),
            train_steps_per_iteration=steps_per_iteration,
            eval_num_envs=evaluation_protocol.eval_num_envs,
            public_seed_count=evaluation_protocol.public_seed_count,
            public_accept_seed_count=evaluation_protocol.public_accept_seed_count,
            sealed_validation_seed_count=(
                evaluation_protocol.sealed_validation_seed_count
            ),
            eval_protocol_version=str(
                v2["formal_protocol"]["evaluation_protocol_version"]
            ),
            train_seed=42,
            runner_timeout_s=int(training["runner_timeout_s"]),
            train_profile=STAIR_TRAIN_PROFILE,
            eval_profile=STAIR_EVAL_PROFILE,
        )

    def evaluation_protocol(self) -> EvaluationProtocol:
        return EvaluationProtocol(
            eval_protocol_version=self.eval_protocol_version,
            public_seed_count=self.public_seed_count,
            public_accept_seed_count=self.public_accept_seed_count,
            sealed_validation_seed_count=self.sealed_validation_seed_count,
            eval_num_envs=self.eval_num_envs,
        )

    def seed_resolver(self) -> StairPrivateSeedResolver:
        evaluation = self.protocol_v1.to_dict()["evaluation"]
        return StairPrivateSeedResolver(
            base_seed=STAIR_EVAL_BASE_SEED,
            task_protocol_hash=self.protocol_v2.protocol_hash,
            evaluation_protocol=self.evaluation_protocol(),
            bin_ids=tuple(str(item["bin_id"]) for item in self.protocol_v1.eval_bins),
            episodes_per_seed_per_bin=int(
                evaluation["env_to_bin_mapping"]["envs_per_bin"]
            ),
        )

    def training_agent_configuration(self) -> RslRlAgentConfiguration:
        return STAIR_G1_ROUGH_AGENT_CONFIGURATION

    def train_runtime_profile(self) -> StairRuntimeProfile:
        return StairRuntimeProfile(
            mode="train",
            profile_id=self.train_profile,
            base_protocol_hash=self.protocol_v1.protocol_hash,
            draft_protocol_hash=self.protocol_v2.protocol_hash,
        )

    def eval_runtime_profile(self) -> StairRuntimeProfile:
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


def _non_empty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


__all__ = [
    "STAIR_EVAL_BASE_SEED",
    "STAIR_G1_ROUGH_AGENT_CONFIGURATION",
    "STAIR_PROTOCOL_V2_RELATIVE_PATH",
    "STAIR_TASK_ID",
    "StairClimbingTaskConfig",
]
