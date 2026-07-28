"""Task-owned Stair episode state machine and terminal classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1


class StairEpisodeEvidenceError(ValueError):
    """Raised when recorder evidence cannot support a closed classification."""


@dataclass(frozen=True)
class StairEnvTransition:
    """One env's task-derived transition, captured before any auto-reset."""

    env_index: int
    bin_id: str
    episode_step: int
    progress_m: float
    root_clearance_m: float
    abs_pitch_rad: float
    base_contact: bool
    foot_contacts: tuple[bool, bool]
    foot_planar_speeds_mps: tuple[float, float]
    swing_foot_clearances_m: tuple[float, float]
    done: bool
    termination_terms: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.env_index, bool)
            or not isinstance(self.env_index, int)
            or self.env_index < 0
        ):
            raise StairEpisodeEvidenceError("stair env_index must be non-negative")
        if not isinstance(self.bin_id, str) or not self.bin_id:
            raise StairEpisodeEvidenceError("stair bin_id must be non-empty")
        if (
            isinstance(self.episode_step, bool)
            or not isinstance(self.episode_step, int)
            or self.episode_step <= 0
        ):
            raise StairEpisodeEvidenceError("stair episode_step must be positive")
        for field_name in (
            "progress_m",
            "root_clearance_m",
            "abs_pitch_rad",
        ):
            _finite_number(getattr(self, field_name), field_name)
        for field_name in (
            "base_contact",
            "done",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise StairEpisodeEvidenceError(f"stair {field_name} must be boolean")
        _boolean_pair(self.foot_contacts, "foot_contacts")
        _finite_pair(self.foot_planar_speeds_mps, "foot_planar_speeds_mps")
        _finite_pair(self.swing_foot_clearances_m, "swing_foot_clearances_m")
        if (
            not isinstance(self.termination_terms, tuple)
            or len(self.termination_terms) != len(set(self.termination_terms))
            or any(not isinstance(item, str) or not item for item in self.termination_terms)
        ):
            raise StairEpisodeEvidenceError(
                "stair termination_terms must be a unique string tuple"
            )
        if self.done != bool(self.termination_terms):
            raise StairEpisodeEvidenceError(
                "stair done mask does not match active termination terms"
            )


@dataclass(frozen=True)
class StairEpisodeOutcome:
    env_index: int
    bin_id: str
    episode_steps: int
    primary_reason: str
    success: bool
    diagnostic_flags: tuple[str, ...]
    max_progress_m: float
    progress_fraction: float
    active_terminal_terms: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.env_index, bool)
            or not isinstance(self.env_index, int)
            or self.env_index < 0
        ):
            raise StairEpisodeEvidenceError("stair outcome env_index must be non-negative")
        if not isinstance(self.bin_id, str) or not self.bin_id:
            raise StairEpisodeEvidenceError("stair outcome bin_id must be non-empty")
        if (
            isinstance(self.episode_steps, bool)
            or not isinstance(self.episode_steps, int)
            or self.episode_steps <= 0
        ):
            raise StairEpisodeEvidenceError("stair outcome episode_steps must be positive")
        if not isinstance(self.primary_reason, str) or not self.primary_reason:
            raise StairEpisodeEvidenceError("stair outcome primary_reason must be non-empty")
        if not isinstance(self.success, bool):
            raise StairEpisodeEvidenceError("stair outcome success must be boolean")
        _unique_string_tuple(self.diagnostic_flags, "outcome diagnostic_flags")
        _unique_string_tuple(self.active_terminal_terms, "outcome active_terminal_terms")
        max_progress = _finite_number(self.max_progress_m, "outcome max_progress_m")
        progress_fraction = _finite_number(
            self.progress_fraction,
            "outcome progress_fraction",
        )
        if max_progress < 0.0 or progress_fraction < 0.0:
            raise StairEpisodeEvidenceError("stair outcome progress cannot be negative")

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "bin_id": self.bin_id,
            "episode_steps": self.episode_steps,
            "primary_reason": self.primary_reason,
            "success": self.success,
            "diagnostic_flags": list(self.diagnostic_flags),
            "max_progress_m": self.max_progress_m,
            "progress_fraction": self.progress_fraction,
        }


class StairEpisodeTracker:
    """Track one env episode and emit exactly one protocol-owned outcome."""

    def __init__(self, protocol: StairProtocolV1, *, env_index: int, bin_id: str) -> None:
        if not isinstance(protocol, StairProtocolV1):
            raise TypeError("stair tracker protocol must be StairProtocolV1")
        if isinstance(env_index, bool) or not isinstance(env_index, int) or env_index < 0:
            raise StairEpisodeEvidenceError("stair tracker env_index is invalid")
        bin_ids = tuple(str(item["bin_id"]) for item in protocol.eval_bins)
        if bin_id not in bin_ids:
            raise StairEpisodeEvidenceError("stair tracker bin_id is not frozen")
        payload = protocol.to_dict()
        self.protocol = protocol
        self.env_index = env_index
        self.bin_id = bin_id
        self._outcomes = payload["episode_outcomes"]
        goal = payload["terrain"]["goal"]
        self._progress_start_m = float(goal["progress_start_m_from_origin"])
        self._goal_m = float(goal["progress_goal_m_from_origin"])
        if self._goal_m <= self._progress_start_m:
            raise StairEpisodeEvidenceError(
                "stair progress goal must be beyond the progress start"
            )
        self._manager_term_reason = dict(
            payload["phase0_probe"]["terminal_reason_by_manager_term"]
        )
        self._last_step = 0
        self._max_progress_m = self._progress_start_m
        self._top_hold_streak = 0
        self._fall_streak = 0
        self._pitch_streak = 0
        self._slip_streak = 0
        self._clearance_streak = 0
        self._base_contact_latched = False
        self._fall_latched = False
        self._pitch_latched = False
        self._slip_latched = False
        self._clearance_latched = False
        self._complete = False

    def observe(self, transition: StairEnvTransition) -> StairEpisodeOutcome | None:
        if self._complete:
            raise StairEpisodeEvidenceError("stair episode tracker is already complete")
        if transition.env_index != self.env_index or transition.bin_id != self.bin_id:
            raise StairEpisodeEvidenceError("stair transition identity drifted")
        if transition.episode_step != self._last_step + 1:
            raise StairEpisodeEvidenceError("stair episode step sequence drifted")
        self._last_step = transition.episode_step
        self._max_progress_m = max(self._max_progress_m, transition.progress_m)
        self._update_latches(transition)
        if not transition.done:
            return None
        outcome = self._finalize(transition)
        self._complete = True
        return outcome

    def _update_latches(self, transition: StairEnvTransition) -> None:
        signals = self._outcomes["signals"]
        success = self._outcomes["success"]

        self._base_contact_latched |= transition.base_contact
        self._fall_streak = _next_streak(
            self._fall_streak,
            transition.root_clearance_m < float(signals["fall"]["threshold"]),
        )
        self._pitch_streak = _next_streak(
            self._pitch_streak,
            transition.abs_pitch_rad
            > float(signals["base_pitch_instability"]["threshold"]),
        )
        slip = any(
            contact
            and speed > float(signals["slip"]["speed_threshold"])
            for contact, speed in zip(
                transition.foot_contacts,
                transition.foot_planar_speeds_mps,
            )
        )
        clearance = any(
            (not contact)
            and value < float(signals["insufficient_clearance"]["threshold"])
            for contact, value in zip(
                transition.foot_contacts,
                transition.swing_foot_clearances_m,
            )
        )
        self._slip_streak = _next_streak(self._slip_streak, slip)
        self._clearance_streak = _next_streak(self._clearance_streak, clearance)
        required_progress = self._progress_start_m + (
            self._goal_m - self._progress_start_m
        ) * float(success["minimum_progress_fraction"])
        at_goal = transition.progress_m >= required_progress
        self._top_hold_streak = _next_streak(self._top_hold_streak, at_goal)

        self._fall_latched |= self._fall_streak >= int(
            signals["fall"]["consecutive_control_steps"]
        )
        self._pitch_latched |= self._pitch_streak >= int(
            signals["base_pitch_instability"]["consecutive_control_steps"]
        )
        self._slip_latched |= self._slip_streak >= int(
            signals["slip"]["consecutive_control_steps"]
        )
        self._clearance_latched |= self._clearance_streak >= int(
            signals["insufficient_clearance"]["consecutive_control_steps"]
        )

    def _finalize(self, transition: StairEnvTransition) -> StairEpisodeOutcome:
        active_reasons: set[str] = set()
        unknown_terms: list[str] = []
        for term in transition.termination_terms:
            reason = self._manager_term_reason.get(term)
            if reason is None:
                unknown_terms.append(term)
            else:
                active_reasons.add(reason)
        if unknown_terms:
            raise StairEpisodeEvidenceError(
                "stair terminal evidence contains unmapped terms: "
                + ", ".join(sorted(unknown_terms))
            )

        success_cfg = self._outcomes["success"]
        succeeded = (
            self._top_hold_streak >= int(success_cfg["top_hold_control_steps"])
            and (
                not bool(success_cfg["requires_no_base_contact"])
                or not self._base_contact_latched
            )
            and (
                not bool(success_cfg["requires_not_fallen"])
                or not self._fall_latched
            )
        )
        if succeeded:
            primary_reason = "success"
        else:
            reason_flags = {
                "base_contact": self._base_contact_latched
                or "base_contact" in active_reasons,
                "fall": self._fall_latched,
                "base_pitch_instability": self._pitch_latched,
                "slip": self._slip_latched,
                "insufficient_clearance": self._clearance_latched,
                "timeout": "timeout" in active_reasons,
            }
            primary_reason = next(
                (
                    reason
                    for reason in self._outcomes["failure_priority"]
                    if bool(reason_flags.get(reason, False))
                ),
                None,
            )
            if primary_reason is None:
                raise StairEpisodeEvidenceError(
                    "stair done transition has no classifiable primary reason"
                )

        diagnostics = {
            "slip_detected": self._slip_latched,
            "insufficient_clearance_detected": self._clearance_latched,
            "stalled_progress_detected": (
                not succeeded and self._max_progress_m < self._goal_m
            ),
        }
        declared = tuple(self._outcomes["diagnostic_flags"])
        if set(diagnostics) != set(declared):
            raise StairEpisodeEvidenceError(
                "stair diagnostic implementation does not match protocol"
            )
        progress_fraction = max(
            0.0,
            (self._max_progress_m - self._progress_start_m)
            / (self._goal_m - self._progress_start_m),
        )
        return StairEpisodeOutcome(
            env_index=self.env_index,
            bin_id=self.bin_id,
            episode_steps=transition.episode_step,
            primary_reason=primary_reason,
            success=succeeded,
            diagnostic_flags=tuple(
                name for name in declared if diagnostics[name]
            ),
            max_progress_m=round(self._max_progress_m, 8),
            progress_fraction=round(progress_fraction, 8),
            active_terminal_terms=transition.termination_terms,
        )


def _next_streak(current: int, active: bool) -> int:
    return current + 1 if active else 0


def _finite_number(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
    ):
        raise StairEpisodeEvidenceError(f"stair {field_name} must be finite")
    return float(value)


def _boolean_pair(value: Any, field_name: str) -> tuple[bool, bool]:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(not isinstance(item, bool) for item in value)
    ):
        raise StairEpisodeEvidenceError(f"stair {field_name} must be a boolean pair")
    return value


def _finite_pair(value: Any, field_name: str) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise StairEpisodeEvidenceError(f"stair {field_name} must be a pair")
    return (
        _finite_number(value[0], f"{field_name}[0]"),
        _finite_number(value[1], f"{field_name}[1]"),
    )


def _unique_string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, tuple)
        or len(value) != len(set(value))
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise StairEpisodeEvidenceError(
            f"stair {field_name} must be a unique string tuple"
        )
    return value


__all__ = [
    "StairEnvTransition",
    "StairEpisodeEvidenceError",
    "StairEpisodeOutcome",
    "StairEpisodeTracker",
]
