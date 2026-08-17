"""Prompt-visible Stair semantics derived only from validated public records."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from true_rl_agent.agent_core.context import (
    sanitize_public_records_with_policy,
    sanitize_public_value_with_policy,
)
from true_rl_agent.agent_core.experiment_memory import (
    compact_component_activation_records,
)
from true_rl_agent.candidates.schema import MAX_CANDIDATE_ID_CHARS
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.prompting.capacity import build_prompt_capacity_receipt
from true_rl_agent.tasks.prompt_adapter import (
    MAX_TASK_PROMPT_SECTIONS_BYTES,
    sanitize_task_prompt_sections,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (
    build_stair_prompt_feedback,
    build_stair_prompt_score_semantics,
)
from true_rl_agent.tasks.stair_climbing.reward_inputs import (
    stair_reward_input_prompt_contract,
)
from true_rl_agent.tasks.stair_climbing.session_start import (
    STAIR_DEFAULT_START_MODE,
    STAIR_REFERENCE_START_MODE,
    STAIR_REWARD_COLD_START_MODE,
    StairSessionStartProfile,
    resolve_stair_session_start_profile,
)

_SCRUBBED_KEY_PARTS = (
    "raw_seed",
    "seed_cohort_nonce",
    "env_to_bin",
    "terrain_env_origins",
    "sealed_validation_evidence",
    "checkpoint_local_path",
    "checkpoint_output_path",
)
_MAX_RUNTIME_FEEDBACK_COMPONENTS = 12
_MAX_RECENT_PUBLIC_FEEDBACK_RECORDS = 3
_MAX_PUBLIC_FEEDBACK_ENTRIES = 5
_TASK_PROMPT_RESERVED_HEADROOM_BYTES = 512
_COMPACT_FEEDBACK_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.prompt_feedback.v9"
)
_LEGACY_COMPACT_FEEDBACK_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.prompt_feedback.v8"
)
_COMPACT_FEEDBACK_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.feedback_entry_projection.v6"
)
_COMPACT_FEEDBACK_SOURCE_SPLIT = "public_dev"
_COMPACT_COUNT_GRID_ROW_SEPARATOR = ";"
_COMPACT_COUNT_GRID_COLUMN_SEPARATOR = ","
_COMPACT_FEEDBACK_KEYS = (
    "stable_success_summary",
    "diagnostic_flag_rates",
    "dominant_failure",
    "generalization_summary",
    "primary_failure_breakdown",
    "progress_summary",
)
_COMPACT_GENERALIZATION_KEYS = (
    "public_dev_objective_score",
    "success_count_grid",
    "zero_count_grids_omitted",
    "primary_failure_count_grids",
)
_DETAILED_FEEDBACK_SELECTION_LABELS = frozenset(
    {
        "reference_current_best",
        "accepted_current_best",
        "latest_evaluated",
    }
)
_HISTORICAL_COMPARISON_DETAIL_OMISSIONS = (
    "historical_cell_grids",
    "historical_completion_and_diagnostic_summaries",
    "historical_progress_summary",
)


@dataclass(frozen=True)
class StairClimbingPromptAdapter:
    config: StairClimbingTaskConfig
    session_start_profile: StairSessionStartProfile | None = None
    task_id: str = STAIR_TASK_ID

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("stair prompt adapter requires StairClimbingTaskConfig")
        profile = self.session_start_profile
        production_training = self.config.protocol_v2.to_public_dict()[
            "production_training"
        ]
        if profile is None:
            profile = resolve_stair_session_start_profile(
                STAIR_DEFAULT_START_MODE,
                task_protocol_hash=self.config.task_protocol_hash,
                training_binding_hash=production_training[
                    "formal_training_binding_hash"
                ],
            )
            object.__setattr__(self, "session_start_profile", profile)
        if not isinstance(profile, StairSessionStartProfile):
            raise TypeError("stair prompt adapter requires a session start profile")
        if (
            profile.task_protocol_hash != self.config.task_protocol_hash
        ):
            raise ValueError("stair prompt start profile protocol mismatch")
        if (
            profile.training_binding_hash
            != production_training["formal_training_binding_hash"]
        ):
            raise ValueError("stair prompt start profile training binding mismatch")

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        policy = self.prompt_public_record_scrub_policy()
        safe_records = sanitize_public_records_with_policy(
            public_records,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        safe_context = sanitize_public_value_with_policy(
            public_context,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        if not isinstance(safe_context, Mapping):
            safe_context = {}
        runtime_feedback = _latest_runtime_feedback(safe_records)
        sections = self._prompt_sections(
            public_feedback=_public_feedback_summary(
                safe_records,
                safe_context,
            ),
            runtime_feedback=runtime_feedback,
        )
        sections = _compact_public_feedback_to_fit_task_sections(
            sections,
            evaluation_profile=self.config.evaluation_profile_config,
        )
        return sanitize_task_prompt_sections(sections)

    def build_prompt_capacity_preflight_sections(self) -> Mapping[str, Any]:
        """Return the task-owned maximum minimum-core shape for startup checks."""

        sections = self._prompt_sections(
            public_feedback=_stair_capacity_preflight_feedback(
                self.config.evaluation_profile_config
            ),
            runtime_feedback={
                "schema_version": (
                    "true_rl_agent.runtime_feedback_summary.stair.preflight.v1"
                ),
                "status": "available",
            },
        )
        minimum = _minimum_core_stair_prompt_sections(
            sections,
            target_bytes=(
                MAX_TASK_PROMPT_SECTIONS_BYTES
                - _TASK_PROMPT_RESERVED_HEADROOM_BYTES
            ),
        )
        return sanitize_task_prompt_sections(minimum)

    def _prompt_sections(
        self,
        *,
        public_feedback: Mapping[str, Any],
        runtime_feedback: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "task": self._task_section(),
            "reward_inputs_contract": stair_reward_input_prompt_contract(
                self.config.protocol_v2.complete_reward_runtime
            ),
            "public_eval_contract": self._public_eval_contract(),
            "public_feedback_summary": dict(public_feedback),
            "runtime_feedback_summary": dict(runtime_feedback),
            "known_good_examples": [],
            "selection_guidance": self._selection_guidance(),
        }

    def nonaccepted_authoring_tiebreak_key(
        self,
        public_record: Mapping[str, Any],
    ) -> tuple[float, ...]:
        _ = public_record
        return ()

    def _selection_guidance(self) -> list[str]:
        profile = self.session_start_profile
        if not isinstance(profile, StairSessionStartProfile):
            raise RuntimeError("stair prompt session start profile is unresolved")
        guidance = [
            "Use only validated public failure, diagnostic, and progress summaries.",
            "When a candidate is rejected below current_best, use current_best as the "
            "default parent; do not carry rejected terms or weights forward unless the "
            "public evidence supports each reuse.",
            "Keep each candidate attributable: change one reward mechanism while "
            "preserving current-best coefficients. Do not combine a new mechanism with "
            "unrelated coefficient retuning.",
            "Treat executable components.py plus verifier and runtime evidence as "
            "authoritative over prose rationale. For any claimed reward range or maximum "
            "weighted contribution, account for every conditioning factor, callable "
            "output clamp, RewardTerm clip, and weight; omit the numeric claim when an "
            "analytic bound is not justified.",
            "When goal_not_reached_detected is frequent, preserve locomotion and "
            "obstacle-negotiation capacity; do not respond only by increasing posture, "
            "contact, smoothness, or termination penalties.",
            "Canonical success is stable_success_summary.stable_top_success_rate; "
            "valid_top_arrival_rate is diagnostic-only, and their gap means top arrival "
            "without stable success.",
            "post_arrival_safety_failure_detected marks failure after top arrival and "
            "before stable success.",
        ]
        if profile.profile_id == STAIR_REFERENCE_START_MODE:
            guidance.append(
                "Match candidate feedback to source_catalog by candidate_id and "
                "program_source_checksum when that source is available; "
                "reference_anchor feedback belongs to the current_best source."
            )
        elif profile.profile_id == STAIR_REWARD_COLD_START_MODE:
            guidance.extend(
                [
                    "This session starts without a reference reward program or hidden "
                    "base reward; the first candidate has no parent reward source.",
                    "Author the complete reward from the declared StairRewardInputsV2 "
                    "contract and validated public evidence only.",
                ]
            )
        else:  # Defensive: the profile validates the closed mode set.
            raise ValueError("unsupported Stair prompt session start profile")
        guidance.extend(
            [
                "When the shared candidate_capabilities section marks "
                "complete_reward_program prompt-visible, return a complete replacement "
                "reward program, not an additive patch.",
                "Use only fields declared by the StairRewardInputsV2 contract.",
                "Do not change training budget, PPO, seeds, terrain bins, termination "
                "rules, or acceptance rules.",
            ]
        )
        return guidance

    def prompt_public_record_scrub_policy(self) -> Mapping[str, Sequence[str]]:
        return {
            "blocked_key_parts": _SCRUBBED_KEY_PARTS,
            "blocked_text_parts": _SCRUBBED_KEY_PARTS,
        }

    def _task_section(self) -> dict[str, Any]:
        base = self.config.protocol_v1.to_dict()
        evaluation = self.config.evaluation_protocol().to_public_dict()
        outcomes = self.config.episode_outcomes()
        profile = self.config.evaluation_profile_config
        if profile is None:
            evaluation_summary = {
                **evaluation,
                "bin_ids": [
                    str(item["bin_id"])
                    for item in self.config.protocol_v1.eval_bins
                ],
                "episodes_per_seed_per_bin": int(
                    base["evaluation"]["env_to_bin_mapping"]["envs_per_bin"]
                ),
                "statistics_unit": (
                    "one_equal_weight_seven_bin_macro_score_per_seed"
                ),
            }
        else:
            evaluation_summary = {
                **evaluation,
                "bin_ids": [
                    str(item["bin_id"])
                    for item in self.config.protocol_v1.eval_bins
                ],
                "anchor_tread_depth_m": profile.anchor_tread_depth_m,
                "generalization_tread_depths_m": list(profile.tread_depths_m),
                "evaluation_cell_count": len(profile.cells),
                "episodes_per_seed_per_cell": (
                    profile.episodes_per_seed_per_cell
                ),
                "episode_horizon_policy": profile.episode_horizon_policy,
                "episode_horizons_s_by_tread_m": (
                    profile.episode_horizons_by_tread_m
                ),
                "statistics_unit": (
                    "one_equal_weight_height_tread_cell_macro_score_per_seed"
                ),
            }
        return {
            "schema_version": "true_rl_agent.task_prompt.stair_climbing.v2",
            "task_id": self.task_id,
            "name": "Unitree G1 Stair Climbing",
            "objective": (
                "From the fixed random initialization, climb the one-way staircase and "
                "complete the protocol-owned stable-top success requirement."
            ),
            "candidate_authoring_status": "governed_by_shared_candidate_capabilities",
            "fixed_train_protocol_summary": {
                "budget_steps": self.config.budget_steps,
                "train_num_envs": self.config.train_num_envs,
                "train_max_iterations": self.config.train_max_iterations,
                "train_steps_per_iteration": self.config.train_steps_per_iteration,
                "policy_initialization": "fixed_random_session_seed",
                "complete_reward_runtime_available": True,
            },
            "fixed_eval_protocol_summary": {
                **evaluation_summary,
                "seed_values_exposed": False,
                "success_criteria": _success_criteria_prompt_contract(outcomes),
            },
            "primary_outcomes": list(
                self.config.episode_outcomes()["primary_reasons"]
            ),
            "diagnostic_flags": list(
                self.config.episode_outcomes()["diagnostic_flags"]
            ),
            "candidate_must_not_change": [
                "checkpoint_policy"
                if item == "checkpoint_selector"
                else str(item)
                for item in base["candidate_boundary"]["forbidden_components"]
            ],
        }

    def _public_eval_contract(self) -> dict[str, Any]:
        acceptance_protocol = self.config.acceptance_protocol
        if self.config.evaluation_profile_config is None:
            return {
                "schema_version": "true_rl_agent.public_eval_contract.stair.v2",
                "prompt_score_semantics": build_stair_prompt_score_semantics(
                    acceptance_protocol
                ),
                "aggregation": (
                    "per_seed_seven_bin_macro_then_cross_seed_statistics"
                ),
                "public_feedback_source": "public_dev_task_recorder_projection",
                "primary_failure_is_mutually_exclusive": True,
                "diagnostic_flags_may_overlap": True,
                "classification_warnings_block_pass": True,
            }
        return {
            "schema_version": "true_rl_agent.public_eval_contract.stair.v4",
            "prompt_score_semantics": build_stair_prompt_score_semantics(
                acceptance_protocol
            ),
            "aggregation": (
                "per_seed_equal_height_tread_cell_macro_then_cross_seed_statistics"
            ),
            "difficulty_monotonicity_role": "diagnostic_only",
            "public_feedback_source": "public_dev_task_recorder_projection",
            "primary_failure_is_mutually_exclusive": True,
            "diagnostic_flags_may_overlap": True,
            "classification_warnings_block_pass": True,
        }


def _public_feedback_summary(
    public_records: Sequence[Mapping[str, Any]],
    public_context: Mapping[str, Any],
) -> dict[str, Any]:
    candidates = _candidate_feedback_entries(public_records)
    selected = _select_candidate_feedback_entries(candidates)
    entries: list[dict[str, Any]] = []
    if not any(entry["accepted"] is True for entry in candidates):
        anchor = _reference_anchor_feedback_entry(public_context)
        if anchor is not None:
            entries.append(anchor)
    entries.extend(selected)
    if entries:
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.public_feedback_summary.v8"
            ),
            "status": "available",
            "feedback_entries": entries,
            "omitted_candidate_feedback_entry_count": max(
                0,
                len(candidates) - len(selected),
            ),
            "byte_budget_compaction_applied": False,
            "limits": {
                "max_recent_candidate_records": (
                    _MAX_RECENT_PUBLIC_FEEDBACK_RECORDS
                ),
                "max_candidate_feedback_entries": _MAX_PUBLIC_FEEDBACK_ENTRIES,
                "max_task_prompt_sections_bytes": (
                    MAX_TASK_PROMPT_SECTIONS_BYTES
                ),
                "reserved_task_prompt_headroom_bytes": (
                    _TASK_PROMPT_RESERVED_HEADROOM_BYTES
                ),
            },
        }
    task_summary = public_context.get("task_public_summary")
    if isinstance(task_summary, Mapping) and task_summary:
        return {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.public_feedback_summary.v8"
            ),
            "status": "memory_only",
            "task_public_summary": dict(task_summary),
        }
    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.public_feedback_summary.v8"
        ),
        "status": "not_available",
        "reason": "no_validated_public_stair_evaluation",
    }


def _stair_capacity_preflight_feedback(
    evaluation_profile: StairEvaluationProfile,
) -> dict[str, Any]:
    """Build the largest bounded shape retained by Stair minimum-core."""

    maximum_number = -1.7976931348623157e308
    stable = {
        "stable_top_success_rate": maximum_number,
        "valid_top_arrival_rate": maximum_number,
        "top_arrival_without_stable_success_rate": maximum_number,
    }
    progress = {
        f"{tread:.2f}": maximum_number
        for tread in evaluation_profile.tread_depths_m
    }

    def entry(
        *,
        fill: str,
        accepted: bool,
        selection_basis: list[str],
    ) -> dict[str, Any]:
        return {
            "source_kind": "candidate",
            "source_public_result_hash": fill * 64,
            "candidate_id": fill * MAX_CANDIDATE_ID_CHARS,
            "program_source_checksum": f"sha256:{fill * 64}",
            "objective_score": maximum_number,
            "guard_score": maximum_number,
            "accepted": accepted,
            "selection_basis": selection_basis,
            "feedback": {
                "schema_version": _COMPACT_FEEDBACK_SCHEMA_VERSION,
                "source_split": _COMPACT_FEEDBACK_SOURCE_SPLIT,
                "stable_success_summary": dict(stable),
                "dominant_failure": {
                    "reason": "goal_not_reached_detected",
                    "rate": maximum_number,
                },
                "generalization_summary": {
                    "public_dev_objective_score": maximum_number,
                },
                "progress_summary": {
                    "mean_max_progress_m_by_tread": dict(progress),
                },
            },
        }

    return {
        "schema_version": (
            "true_rl_agent.task.stair_climbing.public_feedback_summary.v8"
        ),
        "status": "available",
        "feedback_entries": [
            entry(
                fill="a",
                accepted=True,
                selection_basis=["accepted_current_best"],
            ),
            entry(
                fill="b",
                accepted=False,
                selection_basis=["latest_evaluated"],
            ),
        ],
        "omitted_candidate_feedback_entry_count": 28,
        "byte_budget_compaction_applied": True,
    }


def _compact_public_feedback_to_fit_task_sections(
    sections: Mapping[str, Any],
    *,
    evaluation_profile: StairEvaluationProfile | None,
) -> dict[str, Any]:
    """Drop lower-priority detailed history before the shared hard limit."""

    bounded = dict(sections)
    feedback = _mapping(bounded.get("public_feedback_summary"))
    raw_entries = feedback.get("feedback_entries")

    target_bytes = (
        MAX_TASK_PROMPT_SECTIONS_BYTES - _TASK_PROMPT_RESERVED_HEADROOM_BYTES
    )
    if canonical_prompt_json_bytes(bounded) <= target_bytes:
        return bounded

    entries = (
        [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
        if isinstance(raw_entries, list | tuple)
        else []
    )
    omitted = feedback.get("omitted_candidate_feedback_entry_count", 0)
    if isinstance(omitted, bool) or not isinstance(omitted, int) or omitted < 0:
        omitted = 0

    if canonical_prompt_json_bytes(bounded) > target_bytes:
        compacted_entries = [
            _compact_required_feedback_entry(
                entry,
                evaluation_profile=evaluation_profile,
            )
            for entry in entries
        ]
        if compacted_entries != entries:
            entries = compacted_entries
            feedback = {
                **feedback,
                "feedback_entries": entries,
                "byte_budget_compaction_applied": True,
                "feedback_entry_projection": _compact_feedback_projection(
                    entries,
                    evaluation_profile=evaluation_profile,
                ),
            }
            bounded["public_feedback_summary"] = feedback

    if canonical_prompt_json_bytes(bounded) > target_bytes:
        historical_compaction_applied = False
        for index, entry in enumerate(entries):
            if _is_detailed_feedback_entry(entry):
                continue
            entry = entries[index]
            compacted = _compact_historical_comparison_entry(entry)
            if compacted == entry:
                continue
            entries[index] = compacted
            historical_compaction_applied = True
        if historical_compaction_applied:
            projection = _compact_feedback_projection(
                entries,
                evaluation_profile=evaluation_profile,
            )
            projection["historical_comparison_detail_level"] = "summary"
            feedback = {
                **feedback,
                "feedback_entries": entries,
                "byte_budget_compaction_applied": True,
                "feedback_entry_projection": projection,
            }
            bounded["public_feedback_summary"] = feedback

    historical_omitted = 0
    while canonical_prompt_json_bytes(bounded) > target_bytes:
        removable_history = [
            index
            for index, entry in enumerate(entries)
            if not _is_detailed_feedback_entry(entry)
        ]
        if not removable_history:
            break
        removed = entries.pop(removable_history[0])
        if removed.get("source_kind") != "reference_anchor":
            omitted += 1
        historical_omitted += 1
        feedback = {
            **feedback,
            "feedback_entries": entries,
            "omitted_candidate_feedback_entry_count": omitted,
            "historical_comparison_entries_omitted_for_byte_budget": (
                historical_omitted
            ),
            "byte_budget_compaction_applied": True,
            "feedback_entry_projection": _compact_feedback_projection(
                entries,
                evaluation_profile=evaluation_profile,
            ),
        }
        bounded["public_feedback_summary"] = feedback

    if canonical_prompt_json_bytes(bounded) > target_bytes:
        bounded = _minimum_core_stair_prompt_sections(
            bounded,
            target_bytes=target_bytes,
        )
    return bounded


def _minimum_core_stair_prompt_sections(
    sections: Mapping[str, Any],
    *,
    target_bytes: int,
) -> dict[str, Any]:
    """Keep only Stair-owned facts required to author the next candidate."""

    original = dict(sections)
    feedback = _mapping(original.get("public_feedback_summary"))
    raw_entries = feedback.get("feedback_entries")
    entries = (
        [dict(entry) for entry in raw_entries if isinstance(entry, Mapping)]
        if isinstance(raw_entries, list | tuple)
        else []
    )
    selected: list[dict[str, Any]] = []
    current_best = next(
        (
            entry
            for entry in reversed(entries)
            if _entry_has_any_selection(
                entry,
                {"reference_current_best", "accepted_current_best"},
            )
        ),
        None,
    )
    latest = next(
        (
            entry
            for entry in reversed(entries)
            if _entry_has_any_selection(entry, {"latest_evaluated"})
        ),
        None,
    )
    for entry in (current_best, latest):
        if entry is not None and entry not in selected:
            selected.append(entry)
    minimum_entries = [_minimum_core_feedback_entry(entry) for entry in selected]
    raw_omitted_count = feedback.get("omitted_candidate_feedback_entry_count", 0)
    omitted_count = (
        raw_omitted_count
        if isinstance(raw_omitted_count, int)
        and not isinstance(raw_omitted_count, bool)
        and raw_omitted_count >= 0
        else 0
    )
    if minimum_entries:
        minimum_feedback: dict[str, Any] = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.public_feedback_minimum_core.v1"
            ),
            "status": "available",
            "feedback_entries": minimum_entries,
            "omitted_candidate_feedback_entry_count": max(
                omitted_count,
                max(0, len(entries) - len(minimum_entries)),
            ),
            "byte_budget_compaction_applied": True,
            "projection_level": "minimum_core",
        }
    else:
        minimum_feedback = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.public_feedback_minimum_core.v1"
            ),
            "status": "no_closed_outcome",
            "reason": "no_validated_public_stair_evaluation",
            "feedback_entries": [],
            "omitted_candidate_feedback_entry_count": 0,
            "byte_budget_compaction_applied": True,
            "projection_level": "minimum_core",
        }
    minimum = {
        "task": original.get("task"),
        "reward_inputs_contract": original.get("reward_inputs_contract"),
        "public_eval_contract": original.get("public_eval_contract"),
        "public_feedback_summary": minimum_feedback,
        "runtime_feedback_summary": {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.runtime_feedback_minimum_core.v1"
            ),
            "status": "omitted_for_capacity",
            "reason": "minimum_core_projection",
        },
        "known_good_examples": [],
        "selection_guidance": original.get("selection_guidance", []),
    }
    content_without_receipt = dict(minimum)
    receipt = build_prompt_capacity_receipt(
        original=original,
        projected_content=content_without_receipt,
        projection_id="stair_climbing.minimum_core.v1",
        omitted=(
            "historical_feedback_entries",
            "feedback_cell_grids",
            "feedback_diagnostic_detail",
            "runtime_activation_detail",
        ),
    )
    minimum_feedback["capacity_projection_receipt"] = receipt
    if canonical_prompt_json_bytes(minimum) > target_bytes:
        raise ValueError(
            "Stair minimum-core prompt sections exceed reserved task-section capacity"
        )
    return minimum


def _minimum_core_feedback_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: entry[key]
        for key in (
            "source_kind",
            "source_public_result_hash",
            "candidate_id",
            "program_source_checksum",
            "objective_score",
            "guard_score",
            "accepted",
            "selection_basis",
        )
        if key in entry
    }
    feedback = _normalize_supported_prompt_feedback(_mapping(entry.get("feedback")))
    generalization = _mapping(feedback.get("generalization_summary"))
    compact_feedback = {
        "stable_success_summary": _mapping(
            feedback.get("stable_success_summary")
        ),
        "dominant_failure": _mapping(feedback.get("dominant_failure")),
        "generalization_summary": {
            "public_dev_objective_score": generalization.get(
                "public_dev_objective_score"
            )
        },
        "progress_summary": {
            "mean_max_progress_m_by_tread": _mapping(
                _mapping(feedback.get("progress_summary")).get(
                    "mean_max_progress_m_by_tread"
                )
            )
        },
    }
    result["feedback"] = _without_empty_values(compact_feedback)
    return result


def _entry_has_any_selection(
    entry: Mapping[str, Any],
    labels: set[str],
) -> bool:
    basis = entry.get("selection_basis")
    return isinstance(basis, list | tuple) and bool(
        {str(item) for item in basis} & labels
    )


def _without_empty_values(value: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            item = _without_empty_values(item)
        if item not in ({}, [], None):
            result[key] = item
    return result


def _compact_required_feedback_entry(
    entry: Mapping[str, Any],
    *,
    evaluation_profile: StairEvaluationProfile | None,
) -> dict[str, Any]:
    feedback = _normalize_supported_prompt_feedback(
        _mapping(entry.get("feedback"))
    )
    if (
        not feedback
        or feedback.get("source_split") != _COMPACT_FEEDBACK_SOURCE_SPLIT
    ):
        return dict(entry)

    core: dict[str, Any] = {}
    for key in _COMPACT_FEEDBACK_KEYS:
        value = feedback.get(key)
        if value is not None:
            core[key] = dict(value) if isinstance(value, Mapping) else value
    generalization = _mapping(core.get("generalization_summary"))
    core["generalization_summary"] = {
        key: generalization[key]
        for key in _COMPACT_GENERALIZATION_KEYS
        if key in generalization
    }
    failure_grid_strings = _compact_failure_count_grid_strings(
        _mapping(generalization.get("primary_failure_count_grids")),
        evaluation_profile=evaluation_profile,
    )
    core["generalization_summary"].pop(
        "primary_failure_count_grids",
        None,
    )
    success_grid = core["generalization_summary"].pop(
        "success_count_grid",
        None,
    )
    if success_grid is not None:
        core["generalization_summary"]["success_count_grid_string"] = (
            _compact_count_grid_string(
                success_grid,
                evaluation_profile=evaluation_profile,
                context="Stair prompt success grid",
            )
        )
    if failure_grid_strings:
        core["generalization_summary"][
            "primary_failure_count_grid_strings"
        ] = failure_grid_strings
    if core["generalization_summary"].get("zero_count_grids_omitted") == []:
        core["generalization_summary"].pop("zero_count_grids_omitted")
    progress = _mapping(core.get("progress_summary"))
    compact_progress: dict[str, Any] = {}
    progress_by_tread = _mapping(progress.get("mean_max_progress_m_by_tread"))
    if progress_by_tread:
        compact_progress["mean_max_progress_m_by_tread"] = {
            key: round(float(value), 8)
            for key, value in progress_by_tread.items()
        }
    core["progress_summary"] = compact_progress
    for key in ("primary_failure_breakdown", "diagnostic_flag_rates"):
        core[key] = _without_numeric_zero_values(_mapping(core.get(key)))
    if "valid_top_arrival_rate" in _mapping(core.get("stable_success_summary")):
        core["diagnostic_flag_rates"].pop(
            "first_valid_top_arrival_achieved",
            None,
        )
    core = {key: value for key, value in core.items() if value not in ({}, [])}
    return {**dict(entry), "feedback": core}


def _normalize_supported_prompt_feedback(
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize sealed v8 feedback into the current prompt-only vocabulary."""

    schema_version = feedback.get("schema_version")
    if schema_version == _COMPACT_FEEDBACK_SCHEMA_VERSION:
        return dict(feedback)
    if schema_version != _LEGACY_COMPACT_FEEDBACK_SCHEMA_VERSION:
        return {}
    normalized = dict(feedback)
    normalized["schema_version"] = _COMPACT_FEEDBACK_SCHEMA_VERSION
    generalization = dict(_mapping(normalized.get("generalization_summary")))
    if "score" in generalization:
        legacy_score = generalization.pop("score")
        if "public_dev_objective_score" in generalization:
            if generalization["public_dev_objective_score"] != legacy_score:
                return {}
        else:
            generalization["public_dev_objective_score"] = legacy_score
    normalized["generalization_summary"] = generalization
    return normalized


def _compact_historical_comparison_entry(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = _mapping(entry.get("feedback"))
    generalization = _mapping(feedback.get("generalization_summary"))
    progress = _mapping(feedback.get("progress_summary"))
    historical_feedback: dict[str, Any] = {}
    dominant_failure = feedback.get("dominant_failure")
    if dominant_failure is not None:
        historical_feedback["dominant_failure"] = dominant_failure
    if "public_dev_objective_score" in generalization:
        historical_feedback["generalization_summary"] = {
            "public_dev_objective_score": generalization[
                "public_dev_objective_score"
            ]
        }
    primary_failure_breakdown = _mapping(
        feedback.get("primary_failure_breakdown")
    )
    if primary_failure_breakdown:
        historical_feedback["primary_failure_breakdown"] = dict(
            primary_failure_breakdown
        )
    if "mean_max_progress_m" in progress:
        historical_feedback["progress_summary"] = {
            "mean_max_progress_m": progress["mean_max_progress_m"]
        }
    raw_detail_omissions = entry.get("detail_omissions")
    detail_omissions = (
        raw_detail_omissions
        if isinstance(raw_detail_omissions, list | tuple)
        else ()
    )
    return {
        **dict(entry),
        "feedback": historical_feedback,
        "detail_omissions": sorted(
            {
                *(
                    str(item)
                    for item in detail_omissions
                    if isinstance(item, str)
                ),
                *_HISTORICAL_COMPARISON_DETAIL_OMISSIONS,
            }
        ),
    }


def _compact_feedback_projection(
    entries: Sequence[Mapping[str, Any]],
    *,
    evaluation_profile: StairEvaluationProfile | None,
) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "detail_level": "core",
        "schema_version": _COMPACT_FEEDBACK_PROJECTION_SCHEMA_VERSION,
        "source_split": _COMPACT_FEEDBACK_SOURCE_SPLIT,
        "omitted_rates_are_zero": True,
    }
    has_count_grid_strings = any(
        bool(
            {
                "success_count_grid_string",
                "primary_failure_count_grid_strings",
            }
            & set(
                _mapping(
                    _mapping(entry.get("feedback")).get(
                        "generalization_summary"
                    )
                )
            )
        )
        for entry in entries
    )
    if not has_count_grid_strings:
        return projection
    if evaluation_profile is None:
        raise ValueError(
            "Stair prompt cell grids require an active evaluation profile"
        )
    projection["generalization_grid"] = _generalization_grid_axes(
        evaluation_profile
    )
    projection["count_grid_string_encoding"] = {
        "fields": [
            "success_count_grid_string",
            "primary_failure_count_grid_strings.<reason>",
        ],
        "row_order": "generalization_grid.height_bin_ids",
        "column_order": "generalization_grid.tread_depths_m",
        "row_separator": _COMPACT_COUNT_GRID_ROW_SEPARATOR,
        "column_separator": _COMPACT_COUNT_GRID_COLUMN_SEPARATOR,
        "value_encoding": "episode_count",
    }
    return projection


def _compact_failure_count_grid_strings(
    grids: Mapping[str, Any],
    *,
    evaluation_profile: StairEvaluationProfile | None,
) -> dict[str, str]:
    if not grids:
        return {}
    if evaluation_profile is None:
        raise ValueError(
            "Stair prompt failure grids require an active evaluation profile"
        )
    compact: dict[str, str] = {}
    for raw_reason, raw_grid in grids.items():
        reason = str(raw_reason)
        if not reason:
            raise ValueError("Stair prompt failure grid reason is empty")
        compact[reason] = _compact_count_grid_string(
            raw_grid,
            evaluation_profile=evaluation_profile,
            context="Stair prompt failure grid",
        )
    return compact


def _compact_count_grid_string(
    raw_grid: Any,
    *,
    evaluation_profile: StairEvaluationProfile | None,
    context: str,
) -> str:
    if evaluation_profile is None:
        raise ValueError(f"{context} requires an active evaluation profile")
    expected_rows = len(evaluation_profile.anchor_cells)
    expected_columns = len(evaluation_profile.tread_depths_m)
    if not isinstance(raw_grid, list | tuple) or len(raw_grid) != expected_rows:
        raise ValueError(f"{context} row count mismatch")
    rows: list[str] = []
    for raw_row in raw_grid:
        if (
            not isinstance(raw_row, list | tuple)
            or len(raw_row) != expected_columns
        ):
            raise ValueError(f"{context} column count mismatch")
        counts: list[str] = []
        for raw_count in raw_row:
            if (
                isinstance(raw_count, bool)
                or not isinstance(raw_count, int)
                or raw_count < 0
            ):
                raise ValueError(f"{context} count is invalid")
            counts.append(str(raw_count))
        rows.append(_COMPACT_COUNT_GRID_COLUMN_SEPARATOR.join(counts))
    return _COMPACT_COUNT_GRID_ROW_SEPARATOR.join(rows)


def _generalization_grid_axes(
    profile: StairEvaluationProfile,
) -> dict[str, Any]:
    return {
        "layout": "height_rows_x_tread_columns",
        "height_bin_ids": [cell.bin_id for cell in profile.anchor_cells],
        "tread_depths_m": list(profile.tread_depths_m),
        "value_encoding": "episode_count",
        "count_denominator": (
            profile.evaluation_protocol.public_seed_count
            * profile.episodes_per_seed_per_cell
        ),
    }




def _without_numeric_zero_values(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): item
        for key, item in value.items()
        if not (
            not isinstance(item, bool)
            and isinstance(item, int | float)
            and float(item) == 0.0
        )
    }


def _is_required_feedback_entry(entry: Mapping[str, Any]) -> bool:
    basis = entry.get("selection_basis")
    if not isinstance(basis, list | tuple):
        return False
    labels = {str(item) for item in basis}
    return bool(
        labels
        & {
            "reference_current_best",
            "accepted_current_best",
            "best_observed_nonaccepted",
            "latest_evaluated",
        }
    )


def _is_detailed_feedback_entry(entry: Mapping[str, Any]) -> bool:
    basis = entry.get("selection_basis")
    if not isinstance(basis, list | tuple):
        return False
    return bool(
        {str(item) for item in basis} & _DETAILED_FEEDBACK_SELECTION_LABELS
    )


def _candidate_feedback_entries(
    public_records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for index, record in enumerate(public_records):
        result = _mapping(record.get("public_result"))
        if result.get("status") != "pass":
            continue
        feedback = _feedback_from_metrics(_mapping(result.get("metrics")))
        if feedback is None:
            continue
        decision = _mapping(record.get("decision"))
        accepted = decision.get("accepted")
        comparison = _mapping(decision.get("comparison"))
        bound_score = _finite_score(
            decision.get("public_score", result.get("public_score"))
        )
        objective_score = _finite_score(comparison.get("candidate_score"))
        guard_source = comparison.get("guard_score_source")
        has_guard_score = isinstance(guard_source, str) and bool(guard_source)
        if objective_score is None and not has_guard_score:
            objective_score = bound_score
        guard_score = bound_score if has_guard_score else None
        candidate = _mapping(record.get("candidate"))
        train_result = _mapping(record.get("train_result"))
        entries.append(
            {
                "_index": index,
                "round_id": record.get("round_id"),
                "candidate_id": candidate.get("candidate_id"),
                "program_source_checksum": train_result.get(
                    "program_source_checksum"
                ),
                "objective_score": objective_score,
                "guard_score": guard_score,
                "accepted": accepted if isinstance(accepted, bool) else None,
                "feedback": feedback,
            }
        )
    return entries


def _select_candidate_feedback_entries(
    entries: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if not entries:
        return []
    selected_indices = set(
        range(
            max(0, len(entries) - _MAX_RECENT_PUBLIC_FEEDBACK_RECORDS),
            len(entries),
        )
    )
    nonaccepted = [
        (
            float(entry["objective_score"]),
            index,
        )
        for index, entry in enumerate(entries)
        if entry.get("accepted") is False
        and isinstance(entry.get("objective_score"), int | float)
        and not isinstance(entry.get("objective_score"), bool)
    ]
    best_nonaccepted_index = max(nonaccepted)[-1] if nonaccepted else None
    accepted_indices = [
        index
        for index, entry in enumerate(entries)
        if entry.get("accepted") is True
    ]
    current_accepted_index = accepted_indices[-1] if accepted_indices else None
    for special_index in (best_nonaccepted_index, current_accepted_index):
        if special_index is not None:
            selected_indices.add(special_index)
    if len(selected_indices) > _MAX_PUBLIC_FEEDBACK_ENTRIES:
        raise ValueError("stair public feedback selection exceeds its fixed bound")

    selected: list[dict[str, Any]] = []
    latest_index = len(entries) - 1
    for index in sorted(selected_indices):
        entry = entries[index]
        selection_basis: list[str] = []
        if index >= len(entries) - _MAX_RECENT_PUBLIC_FEEDBACK_RECORDS:
            selection_basis.append("recent_evaluated")
        if index == latest_index:
            selection_basis.append("latest_evaluated")
            if entry.get("accepted") is False:
                selection_basis.append("latest_nonaccepted")
        if index == best_nonaccepted_index:
            selection_basis.append("best_observed_nonaccepted")
        if index == current_accepted_index:
            selection_basis.append("accepted_current_best")
        selected.append(
            {
                key: value
                for key, value in entry.items()
                if key != "_index" and value is not None
            }
            | {"selection_basis": selection_basis}
        )
    return selected


def _reference_anchor_feedback_entry(
    public_context: Mapping[str, Any],
) -> dict[str, Any] | None:
    anchor = _mapping(public_context.get("initial_anchor_public_evidence"))
    if anchor.get("anchor_kind") != "reference_program":
        return None
    result = _mapping(anchor.get("public_result"))
    metrics = _mapping(result.get("metrics"))
    feedback = _feedback_from_metrics(metrics)
    result_hash = anchor.get("public_result_hash")
    if feedback is None or not isinstance(result_hash, str):
        return None
    acceptance = _mapping(metrics.get("acceptance"))
    objective_score = _finite_score(
        _mapping(acceptance.get("score_stats")).get("mean")
    )
    has_guard_score = isinstance(
        acceptance.get("guard_score_source"),
        str,
    ) and bool(acceptance.get("guard_score_source"))
    bound_score = _finite_score(result.get("public_score"))
    if objective_score is None and not has_guard_score:
        objective_score = bound_score
    entry = {
        "source_kind": "reference_anchor",
        "source_public_result_hash": result_hash,
        "objective_score": objective_score,
        "guard_score": bound_score if has_guard_score else None,
        "selection_basis": ["reference_current_best"],
        "feedback": feedback,
    }
    return {key: value for key, value in entry.items() if value is not None}


def _feedback_from_metrics(
    metrics: Mapping[str, Any],
) -> dict[str, Any] | None:
    feedback = metrics.get("prompt_feedback")
    if isinstance(feedback, Mapping):
        normalized = _normalize_supported_prompt_feedback(feedback)
        return normalized or None
    public_dev = metrics.get("public_dev")
    generalization = metrics.get("generalization_public_dev")
    if isinstance(public_dev, Mapping):
        return build_stair_prompt_feedback(
            public_dev,
            generalization_public_dev=(
                generalization if isinstance(generalization, Mapping) else None
            ),
        )
    return None


def _latest_runtime_feedback(
    public_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    for record in reversed(public_records):
        decision = _mapping(record.get("decision"))
        gate_statuses = _mapping(decision.get("gate_statuses"))
        train_result = _mapping(record.get("train_result"))
        public_result = _mapping(record.get("public_result"))
        runtime = _mapping(record.get("runtime_activation_summary"))
        capabilities = runtime.get("capability_activation_records")
        raw_components = runtime.get("component_activation_records")
        reward_term_calls = runtime.get("reward_term_calls")
        if (
            gate_statuses.get("runtime_activation_gate") != "pass"
            or not train_result
            or public_result.get("status") != "pass"
            or not isinstance(capabilities, list | tuple)
            or "complete_reward_program" not in capabilities
            or not isinstance(raw_components, list | tuple)
            or not raw_components
            or isinstance(reward_term_calls, bool)
            or not isinstance(reward_term_calls, int)
            or reward_term_calls <= 0
        ):
            continue
        components = compact_component_activation_records(
            raw_components,
            max_components=_MAX_RUNTIME_FEEDBACK_COMPONENTS,
        )
        if not components:
            continue
        candidate = _mapping(record.get("candidate"))
        return {
            "schema_version": "true_rl_agent.runtime_feedback_summary.stair.v1",
            "status": "available",
            "source_round_id": record.get("round_id"),
            "source_candidate_id": candidate.get("candidate_id"),
            "source_decision_reason": decision.get("reason"),
            "latest_runtime_activation": {
                "reward_term_calls": reward_term_calls,
                "nonzero_reward_contribution": runtime.get(
                    "nonzero_reward_contribution"
                ),
                "capability_activation_records": list(capabilities),
                "component_count": len(raw_components),
                "omitted_component_count": max(
                    0,
                    len(raw_components) - len(components),
                ),
                "components": components,
            },
        }
    return {
        "schema_version": "true_rl_agent.runtime_feedback_summary.stair.v1",
        "status": "not_available",
        "reason": "no_validated_runtime_feedback",
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _finite_score(value: Any) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
    ):
        return None
    return float(value)


def _success_criteria_prompt_contract(
    outcomes: Mapping[str, Any],
) -> dict[str, Any]:
    success = _mapping(outcomes.get("success"))
    signals = _mapping(outcomes.get("signals"))
    stability = _mapping(outcomes.get("top_platform_stability_requirement"))
    orientation = _mapping(signals.get("base_orientation_instability"))
    fall = _mapping(signals.get("fall"))
    return {
        **dict(success),
        "maximum_abs_roll_or_pitch_rad": orientation.get("threshold"),
        "minimum_root_clearance_above_top_m": fall.get("threshold"),
        "top_platform_stability_requirement": dict(stability),
    }


__all__ = ["StairClimbingPromptAdapter"]
