"""Component schema exposed to code-level candidates.

These classes define the public contract between candidate-authored training
code and the runner-owned adapter.  They intentionally contain no Isaac, env,
metric, seed, or evaluation objects.  Candidate functions receive only
``RewardInputs`` and return tensors/vectors that the runner must guard before
using in training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


TensorLike = Any
ScheduleValue = bool | int | float | list["ScheduleValue"] | tuple["ScheduleValue", ...]
SchedulePayload = dict[str, ScheduleValue]


@dataclass(frozen=True)
class RewardInputs:
    """Public, training-only tensors available to candidate reward/loss code.

    Every field is supplied by the trusted training runner.  The contract must
    not grow hidden evaluation fields, seeds, metrics, env handles, or mutable
    runner objects.
    """

    base_height: TensorLike
    roll: TensorLike
    pitch: TensorLike
    base_pos_xy: TensorLike
    base_lin_vel: TensorLike
    base_ang_vel: TensorLike
    projected_gravity: TensorLike
    commanded_base_vel: TensorLike
    joint_pos: TensorLike
    joint_vel: TensorLike
    joint_pos_error: TensorLike
    last_action: TensorLike
    prev_action: TensorLike
    action_delta: TensorLike
    push_active: TensorLike
    push_vec_public: TensorLike
    progress_fraction: TensorLike

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(cls.__dataclass_fields__)


@dataclass(frozen=True)
class RewardTerm:
    """Candidate-authored reward term.

    ``fn`` must be vectorized and return one finite scalar per environment:
    ``Tensor[num_envs]`` or an equivalent numeric vector.  The runner is
    responsible for shape/finite checks and optional clipping.
    """

    name: str
    fn: Callable[[RewardInputs], TensorLike]
    weight: float = 1.0
    clip_min: float | None = None
    clip_max: float | None = None
    description: str = ""


@dataclass(frozen=True)
class LossTerm:
    """Candidate-authored auxiliary loss term.

    Phase 8E-2 defines and validates the contract only.  A later runner phase
    decides where auxiliary loss terms are inserted into PPO.
    """

    name: str
    fn: Callable[[RewardInputs], TensorLike]
    weight: float = 1.0
    clip_min: float | None = None
    clip_max: float | None = None
    description: str = ""


@dataclass(frozen=True)
class DiagnosticTerm:
    """Candidate-authored public training diagnostic.

    Diagnostics are executable in the isolated training process only.  They may
    summarize public ``RewardInputs`` but must not return raw text, seed
    identifiers, eval state, or runner objects.
    """

    name: str
    fn: Callable[[RewardInputs], dict[str, TensorLike]]
    description: str = ""


@dataclass(frozen=True)
class ScheduleTerm:
    """Candidate-authored curriculum schedule.

    ``fn(progress_fraction)`` is pre-evaluated by the runner on a fixed grid.
    It must return only numeric/bool values for keys whitelisted by both the
    term and the runner.
    """

    name: str
    fn: Callable[[float], SchedulePayload]
    allowed_keys: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True)
class TrainingComponents:
    """Candidate-provided training components.

    Phase 8E exposes callable contracts.  The runner-owned adapter decides
    which component families are active in a given phase.
    """

    name: str = "unnamed_candidate"
    reward_terms: list[Any] = field(default_factory=list)
    loss_terms: list[Any] = field(default_factory=list)
    schedules: list[Any] = field(default_factory=list)
    train_observation_transforms: list[Any] = field(default_factory=list)
    diagnostics: list[Any] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def is_noop(self) -> bool:
        """Return True when the candidate cannot change training behavior."""

        return not (
            self.reward_terms
            or self.loss_terms
            or self.schedules
            or self.train_observation_transforms
            or self.diagnostics
        )

    def to_public_summary(self) -> dict[str, Any]:
        """Return an audit-safe structural summary."""

        return {
            "name": self.name,
            "is_noop": self.is_noop(),
            "counts": {
                "reward_terms": len(self.reward_terms),
                "loss_terms": len(self.loss_terms),
                "schedules": len(self.schedules),
                "train_observation_transforms": len(self.train_observation_transforms),
                "diagnostics": len(self.diagnostics),
            },
            "terms": {
                "reward_terms": [_term_public_summary(item) for item in self.reward_terms],
                "loss_terms": [_term_public_summary(item) for item in self.loss_terms],
                "schedules": [_term_public_summary(item) for item in self.schedules],
                "train_observation_transforms": [_term_public_summary(item) for item in self.train_observation_transforms],
                "diagnostics": [_term_public_summary(item) for item in self.diagnostics],
            },
            "metadata": self.metadata,
            "rationale": self.rationale,
        }


def _term_public_summary(item: Any) -> dict[str, Any]:
    """Summarize a term without exposing function objects or private state."""

    if isinstance(item, RewardTerm | LossTerm):
        return {
            "name": item.name,
            "kind": type(item).__name__,
            "weight": item.weight,
            "clip_min": item.clip_min,
            "clip_max": item.clip_max,
            "description": item.description,
        }
    if isinstance(item, DiagnosticTerm):
        return {
            "name": item.name,
            "kind": "DiagnosticTerm",
            "description": item.description,
        }
    if isinstance(item, ScheduleTerm):
        return {
            "name": item.name,
            "kind": "ScheduleTerm",
            "allowed_keys": list(item.allowed_keys),
            "description": item.description,
        }
    if isinstance(item, dict):
        return {
            "name": item.get("name"),
            "kind": item.get("kind", "dict"),
            "legacy_dict_component": True,
        }
    return {"name": getattr(item, "name", None), "kind": type(item).__name__}
