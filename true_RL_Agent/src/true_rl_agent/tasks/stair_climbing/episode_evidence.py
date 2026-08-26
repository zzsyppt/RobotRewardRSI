"""Task-owned Stair episode state machine and terminal classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

from true_rl_agent.tasks.stair_climbing.evaluation_profile import StairEvaluationCell
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
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
    lateral_m: float
    base_height_m: float
    base_clearance_m: float
    abs_roll_rad: float
    abs_pitch_rad: float
    forbidden_contact: bool
    support_contacts: tuple[bool, ...]
    support_progress_m: tuple[float, ...]
    support_surface_levels: tuple[int, ...]
    support_lateral_m: tuple[float, ...]
    support_planar_speeds_mps: tuple[float, ...]
    support_surface_clearances_m: tuple[float, ...]
    support_upward_forces_n: tuple[float, ...]
    done: bool
    termination_terms: tuple[str, ...] = ()
    evaluation_cell_id: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.env_index, bool)
            or not isinstance(self.env_index, int)
            or self.env_index < 0
        ):
            raise StairEpisodeEvidenceError("stair env_index must be non-negative")
        if not isinstance(self.bin_id, str) or not self.bin_id:
            raise StairEpisodeEvidenceError("stair bin_id must be non-empty")
        if self.evaluation_cell_id is not None and (
            not isinstance(self.evaluation_cell_id, str) or not self.evaluation_cell_id
        ):
            raise StairEpisodeEvidenceError(
                "stair evaluation_cell_id must be non-empty when present"
            )
        if (
            isinstance(self.episode_step, bool)
            or not isinstance(self.episode_step, int)
            or self.episode_step <= 0
        ):
            raise StairEpisodeEvidenceError("stair episode_step must be positive")
        for field_name in (
            "progress_m",
            "lateral_m",
            "base_height_m",
            "base_clearance_m",
            "abs_roll_rad",
            "abs_pitch_rad",
        ):
            _finite_number(getattr(self, field_name), field_name)
        for field_name in (
            "forbidden_contact",
            "done",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise StairEpisodeEvidenceError(f"stair {field_name} must be boolean")
        support_count = len(_boolean_tuple(self.support_contacts, "support_contacts"))
        for field_name in (
            "support_progress_m",
            "support_lateral_m",
            "support_planar_speeds_mps",
            "support_surface_clearances_m",
            "support_upward_forces_n",
        ):
            values = _finite_tuple(getattr(self, field_name), field_name)
            if len(values) != support_count:
                raise StairEpisodeEvidenceError(
                    "stair support evidence cardinality mismatch"
                )
        levels = _non_negative_int_tuple(
            self.support_surface_levels,
            "support_surface_levels",
        )
        if len(levels) != support_count:
            raise StairEpisodeEvidenceError(
                "stair support evidence cardinality mismatch"
            )
        if (
            not isinstance(self.termination_terms, tuple)
            or len(self.termination_terms) != len(set(self.termination_terms))
            or any(
                not isinstance(item, str) or not item for item in self.termination_terms
            )
        ):
            raise StairEpisodeEvidenceError(
                "stair termination_terms must be a unique string tuple"
            )
        if self.done != bool(self.termination_terms):
            raise StairEpisodeEvidenceError(
                "stair done mask does not match active termination terms"
            )

    @property
    def support_count(self) -> int:
        return len(self.support_contacts)


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
    evaluation_cell_id: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.env_index, bool)
            or not isinstance(self.env_index, int)
            or self.env_index < 0
        ):
            raise StairEpisodeEvidenceError(
                "stair outcome env_index must be non-negative"
            )
        if not isinstance(self.bin_id, str) or not self.bin_id:
            raise StairEpisodeEvidenceError("stair outcome bin_id must be non-empty")
        if self.evaluation_cell_id is not None and (
            not isinstance(self.evaluation_cell_id, str) or not self.evaluation_cell_id
        ):
            raise StairEpisodeEvidenceError(
                "stair outcome evaluation_cell_id must be non-empty when present"
            )
        if (
            isinstance(self.episode_steps, bool)
            or not isinstance(self.episode_steps, int)
            or self.episode_steps <= 0
        ):
            raise StairEpisodeEvidenceError(
                "stair outcome episode_steps must be positive"
            )
        if not isinstance(self.primary_reason, str) or not self.primary_reason:
            raise StairEpisodeEvidenceError(
                "stair outcome primary_reason must be non-empty"
            )
        if not isinstance(self.success, bool):
            raise StairEpisodeEvidenceError("stair outcome success must be boolean")
        _unique_string_tuple(self.diagnostic_flags, "outcome diagnostic_flags")
        _unique_string_tuple(
            self.active_terminal_terms, "outcome active_terminal_terms"
        )
        diagnostics = set(self.diagnostic_flags)
        arrival_flag = "first_valid_top_arrival_achieved"
        post_arrival_failure_flag = "post_arrival_safety_failure_detected"
        if post_arrival_failure_flag in diagnostics and arrival_flag not in diagnostics:
            raise StairEpisodeEvidenceError(
                "stair post-arrival safety failure requires top arrival evidence"
            )
        if self.success and post_arrival_failure_flag in diagnostics:
            raise StairEpisodeEvidenceError(
                "stair stable-top success conflicts with post-arrival safety failure"
            )
        max_progress = _finite_number(self.max_progress_m, "outcome max_progress_m")
        progress_fraction = _finite_number(
            self.progress_fraction,
            "outcome progress_fraction",
        )
        if max_progress < 0.0 or progress_fraction < 0.0:
            raise StairEpisodeEvidenceError("stair outcome progress cannot be negative")

    def to_public_dict(self) -> dict[str, Any]:
        result = {
            "bin_id": self.bin_id,
            "episode_steps": self.episode_steps,
            "primary_reason": self.primary_reason,
            "success": self.success,
            "diagnostic_flags": list(self.diagnostic_flags),
            "max_progress_m": self.max_progress_m,
            "progress_fraction": self.progress_fraction,
        }
        if self.evaluation_cell_id is not None:
            result["evaluation_cell_id"] = self.evaluation_cell_id
        return result


class StairEpisodeTracker:
    """Track one env episode and emit exactly one protocol-owned outcome."""

    def __init__(
        self,
        protocol: StairProtocolV1,
        semantics: StairEvaluationSemantics,
        *,
        env_index: int,
        bin_id: str,
        evaluation_cell: StairEvaluationCell | None = None,
    ) -> None:
        if not isinstance(protocol, StairProtocolV1):
            raise TypeError("stair tracker protocol must be StairProtocolV1")
        if not isinstance(semantics, StairEvaluationSemantics):
            raise TypeError("stair tracker semantics must be StairEvaluationSemantics")
        if (
            isinstance(env_index, bool)
            or not isinstance(env_index, int)
            or env_index < 0
        ):
            raise StairEpisodeEvidenceError("stair tracker env_index is invalid")
        bin_ids = tuple(str(item["bin_id"]) for item in protocol.eval_bins)
        if bin_id not in bin_ids:
            raise StairEpisodeEvidenceError("stair tracker bin_id is not frozen")
        if evaluation_cell is not None:
            if not isinstance(evaluation_cell, StairEvaluationCell):
                raise TypeError("stair tracker evaluation cell is invalid")
            if evaluation_cell.bin_id != bin_id:
                raise StairEpisodeEvidenceError(
                    "stair tracker evaluation cell/bin identity drifted"
                )
        payload = protocol.to_dict()
        self.protocol = protocol
        self.env_index = env_index
        self.bin_id = bin_id
        self._outcomes = semantics.to_dict()["episode_outcomes"]
        goal = payload["terrain"]["goal"]
        self._progress_start_m = float(goal["progress_start_m_from_origin"])
        self._goal_m = (
            evaluation_cell.progress_goal_m
            if evaluation_cell is not None
            else float(goal["progress_goal_m_from_origin"])
        )
        if self._goal_m <= self._progress_start_m:
            raise StairEpisodeEvidenceError(
                "stair progress goal must be beyond the progress start"
            )
        terrain = payload["terrain"]
        self._step_count = int(terrain["step_count"])
        self._step_height_m = (
            evaluation_cell.height_m
            if evaluation_cell is not None
            else next(
                float(item["height_m"])
                for item in protocol.eval_bins
                if str(item["bin_id"]) == bin_id
            )
        )
        self._evaluation_cell_id = (
            evaluation_cell.cell_id if evaluation_cell is not None else None
        )
        self._episode_horizon_control_steps = (
            evaluation_cell.episode_horizon_control_steps
            if evaluation_cell is not None
            else None
        )
        self._course_min_progress_m = -float(terrain["start_platform_length_m"]) / 2.0
        self._course_max_progress_m = (
            float(terrain["size_m"][0])
            - float(terrain["start_platform_length_m"]) / 2.0
        )
        self._course_half_width_m = float(terrain["size_m"][1]) / 2.0
        manager_term = str(
            self._outcomes["signals"]["forbidden_contact"]["manager_term"]
        )
        self._manager_term_reason = {
            **dict(payload["phase0_probe"]["terminal_reason_by_manager_term"]),
            manager_term: "forbidden_contact",
        }
        self._last_step = 0
        self._max_progress_m = self._progress_start_m
        self._top_platform_stability_streak = 0
        self._fall_streak = 0
        self._orientation_streak = 0
        self._slip_streak = 0
        self._clearance_streak = 0
        self._forbidden_contact_latched = False
        self._fall_latched = False
        self._orientation_diagnostic_latched = False
        self._slip_diagnostic_latched = False
        self._clearance_diagnostic_latched = False
        self._off_course_latched = False
        self._top_arrival_latched = False
        self._top_platform_stability_latched = False
        self._post_arrival_safety_failure_latched = False
        self._complete = False

    def observe(self, transition: StairEnvTransition) -> StairEpisodeOutcome | None:
        if self._complete:
            raise StairEpisodeEvidenceError("stair episode tracker is already complete")
        if (
            transition.env_index != self.env_index
            or transition.bin_id != self.bin_id
            or transition.evaluation_cell_id != self._evaluation_cell_id
        ):
            raise StairEpisodeEvidenceError("stair transition identity drifted")
        terminal_forbidden_contact = any(
            self._manager_term_reason.get(term) == "forbidden_contact"
            for term in transition.termination_terms
        )
        if transition.forbidden_contact != terminal_forbidden_contact:
            raise StairEpisodeEvidenceError(
                "stair forbidden-contact signal disagrees with termination terms"
            )
        if transition.episode_step != self._last_step + 1:
            raise StairEpisodeEvidenceError("stair episode step sequence drifted")
        self._last_step = transition.episode_step
        self._max_progress_m = max(self._max_progress_m, transition.progress_m)
        self._update_state(transition)
        if self._top_platform_stability_latched:
            # Stable success is absorbing: closing on this transition preserves the
            # contract that a successful outcome cannot later gain a post-arrival
            # safety-failure diagnostic.
            outcome = self._finalize(transition)
            self._complete = True
            return outcome
        if not transition.done:
            if (
                self._episode_horizon_control_steps is None
                or transition.episode_step < self._episode_horizon_control_steps
            ):
                return None
            outcome = self._finalize(
                transition,
                additional_active_reasons=("timeout",),
            )
            self._complete = True
            return outcome
        outcome = self._finalize(transition)
        self._complete = True
        return outcome

    def _update_state(self, transition: StairEnvTransition) -> None:
        signals = self._outcomes["signals"]
        success = self._outcomes["success"]
        stability = self._outcomes["top_platform_stability_requirement"]
        arrival_was_latched = self._top_arrival_latched

        self._forbidden_contact_latched |= transition.forbidden_contact
        self._fall_streak = _next_streak(
            self._fall_streak,
            transition.base_clearance_m < float(signals["fall"]["threshold"]),
        )
        self._orientation_streak = _next_streak(
            self._orientation_streak,
            max(transition.abs_roll_rad, transition.abs_pitch_rad)
            > float(signals["base_orientation_instability"]["threshold"]),
        )
        slip = any(
            contact and speed > float(signals["slip"]["speed_threshold"])
            for contact, speed in zip(
                transition.support_contacts,
                transition.support_planar_speeds_mps,
            )
        )
        clearance = any(
            (not contact)
            and value < float(signals["insufficient_clearance"]["threshold"])
            for contact, value in zip(
                transition.support_contacts,
                transition.support_surface_clearances_m,
            )
        )
        self._slip_streak = _next_streak(self._slip_streak, slip)
        self._clearance_streak = _next_streak(self._clearance_streak, clearance)
        off_course = (
            transition.progress_m < self._course_min_progress_m
            or transition.progress_m > self._course_max_progress_m
            or abs(transition.lateral_m) > self._course_half_width_m
        )
        self._off_course_latched |= off_course

        self._fall_latched |= self._fall_streak >= int(
            signals["fall"]["consecutive_control_steps"]
        )
        self._orientation_diagnostic_latched |= self._orientation_streak >= int(
            signals["base_orientation_instability"]["consecutive_control_steps"]
        )
        self._slip_diagnostic_latched |= self._slip_streak >= int(
            signals["slip"]["consecutive_control_steps"]
        )
        self._clearance_diagnostic_latched |= self._clearance_streak >= int(
            signals["insufficient_clearance"]["consecutive_control_steps"]
        )

        required_progress = self._progress_start_m + (
            self._goal_m - self._progress_start_m
        ) * float(success["minimum_progress_fraction"])
        support_clearance_range = success["support_effector_surface_clearance_range_m"]
        minimum_support_clearance = float(support_clearance_range[0])
        maximum_support_clearance = float(support_clearance_range[1])
        minimum_upward_force = float(
            success["minimum_support_effector_upward_force_n"]
        )
        top_support_evidence = tuple(
            contact
            and level == self._step_count
            and self._course_min_progress_m
            <= support_progress
            <= self._course_max_progress_m
            and abs(lateral) <= self._course_half_width_m
            and minimum_support_clearance <= clearance <= maximum_support_clearance
            and upward_force >= minimum_upward_force
            for (
                contact,
                support_progress,
                level,
                lateral,
                clearance,
                upward_force,
            ) in zip(
                transition.support_contacts,
                transition.support_progress_m,
                transition.support_surface_levels,
                transition.support_lateral_m,
                transition.support_surface_clearances_m,
                transition.support_upward_forces_n,
                strict=True,
            )
        )
        top_support_count = sum(top_support_evidence)
        maximum_support_speed = float(
            stability["maximum_support_effector_planar_speed_mps"]
        )
        stable_top_support_count = sum(
            supported and speed <= maximum_support_speed
            for supported, speed in zip(
                top_support_evidence,
                transition.support_planar_speeds_mps,
                strict=True,
            )
        )
        base_clearance_above_top = (
            transition.base_height_m - self._step_count * self._step_height_m
        )
        orientation_stable = max(
            transition.abs_roll_rad,
            transition.abs_pitch_rad,
        ) <= float(signals["base_orientation_instability"]["threshold"])
        valid_top_arrival = (
            transition.progress_m >= required_progress
            and transition.progress_m <= self._course_max_progress_m
            and abs(transition.lateral_m) <= self._course_half_width_m
            and base_clearance_above_top >= float(signals["fall"]["threshold"])
            and orientation_stable
            and top_support_count >= int(success["minimum_support_effectors_on_top"])
        )
        pre_arrival_blocked = (
            self._forbidden_contact_latched
            or self._fall_latched
            or self._off_course_latched
        )
        if valid_top_arrival and not pre_arrival_blocked:
            self._top_arrival_latched = True

        if arrival_was_latched and pre_arrival_blocked:
            self._post_arrival_safety_failure_latched = True

        stable_top_traversal = (
            self._top_arrival_latched
            and valid_top_arrival
            and stable_top_support_count
            >= int(success["minimum_support_effectors_on_top"])
            and not pre_arrival_blocked
        )
        self._top_platform_stability_streak = _next_streak(
            self._top_platform_stability_streak,
            stable_top_traversal,
        )
        self._top_platform_stability_latched |= (
            self._top_platform_stability_streak
            >= int(stability["consecutive_control_steps"])
        )

    def _finalize(
        self,
        transition: StairEnvTransition,
        *,
        additional_active_reasons: tuple[str, ...] = (),
    ) -> StairEpisodeOutcome:
        succeeded = self._top_platform_stability_latched
        active_reasons = set(additional_active_reasons)
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

        if succeeded:
            primary_reason = "success"
        else:
            signals = self._outcomes["signals"]
            reason_flags = {
                "forbidden_contact": self._forbidden_contact_latched
                or "forbidden_contact" in active_reasons,
                "fall": self._fall_latched,
                "off_course": self._off_course_latched,
                "base_orientation_instability": self._orientation_streak
                >= int(
                    signals["base_orientation_instability"]["consecutive_control_steps"]
                ),
                "slip": self._slip_streak
                >= int(signals["slip"]["consecutive_control_steps"]),
                "insufficient_clearance": self._clearance_streak
                >= int(signals["insufficient_clearance"]["consecutive_control_steps"]),
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
            "base_orientation_instability_detected": (
                self._orientation_diagnostic_latched
            ),
            "slip_detected": self._slip_diagnostic_latched,
            "insufficient_clearance_detected": self._clearance_diagnostic_latched,
            "off_course_detected": self._off_course_latched,
            "goal_not_reached_detected": (
                not succeeded and self._max_progress_m < self._goal_m
            ),
            "first_valid_top_arrival_achieved": self._top_arrival_latched,
            "post_arrival_safety_failure_detected": (
                self._post_arrival_safety_failure_latched
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
            diagnostic_flags=tuple(name for name in declared if diagnostics[name]),
            max_progress_m=round(self._max_progress_m, 8),
            progress_fraction=round(progress_fraction, 8),
            active_terminal_terms=transition.termination_terms,
            evaluation_cell_id=self._evaluation_cell_id,
        )


def validate_stair_success_diagnostic_relations(
    outcomes: Sequence[StairEpisodeOutcome],
    *,
    declared_diagnostic_flags: Sequence[str],
) -> None:
    """Validate the active stable-success contract without rewriting legacy outcomes."""

    declared = set(declared_diagnostic_flags)
    arrival_flag = "first_valid_top_arrival_achieved"
    post_arrival_failure_flag = "post_arrival_safety_failure_detected"
    if arrival_flag not in declared:
        return
    if post_arrival_failure_flag not in declared:
        raise StairEpisodeEvidenceError(
            "stair stable-success diagnostics are incomplete"
        )
    for outcome in outcomes:
        diagnostics = set(outcome.diagnostic_flags)
        if outcome.success and arrival_flag not in diagnostics:
            raise StairEpisodeEvidenceError(
                "stair stable-top success requires first valid top arrival evidence"
            )
        if post_arrival_failure_flag in diagnostics and arrival_flag not in diagnostics:
            raise StairEpisodeEvidenceError(
                "stair post-arrival safety failure requires top arrival evidence"
            )
        if outcome.success and post_arrival_failure_flag in diagnostics:
            raise StairEpisodeEvidenceError(
                "stair stable-top success conflicts with post-arrival safety failure"
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


def _boolean_tuple(value: Any, field_name: str) -> tuple[bool, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(not isinstance(item, bool) for item in value)
    ):
        raise StairEpisodeEvidenceError(
            f"stair {field_name} must be a non-empty boolean tuple"
        )
    return value


def _non_negative_int_tuple(value: Any, field_name: str) -> tuple[int, ...]:
    if (
        not isinstance(value, tuple)
        or not value
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise StairEpisodeEvidenceError(
            f"stair {field_name} must be a non-empty non-negative integer tuple"
        )
    return value


def _finite_tuple(value: Any, field_name: str) -> tuple[float, ...]:
    if not isinstance(value, tuple) or not value:
        raise StairEpisodeEvidenceError(f"stair {field_name} must be a non-empty tuple")
    return tuple(
        _finite_number(item, f"{field_name}[{index}]")
        for index, item in enumerate(value)
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
    "validate_stair_success_diagnostic_relations",
]
