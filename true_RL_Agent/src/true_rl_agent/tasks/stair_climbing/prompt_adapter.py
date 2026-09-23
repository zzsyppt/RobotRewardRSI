"""Prompt-visible Stair semantics derived only from validated public records."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from true_rl_agent.agent_core.context import (
    sanitize_public_records_with_policy,
    sanitize_public_value_with_policy,
)
from true_rl_agent.agent_core.experiment_memory import (
    compact_component_activation_records,
)
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS,
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
    latest_nonaccepted_trial,
    project_public_trial_reason,
)
from true_rl_agent.candidates.schema import (
    MAX_CANDIDATE_ID_CHARS,
    require_candidate_id,
)
from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.acceptance_protocol import (
    AcceptanceProtocol,
    validated_score_samples,
)
from true_rl_agent.evaluation.binding import require_sha256_checksum
from true_rl_agent.evaluation.score_stats import validate_score_stats
from true_rl_agent.prompting.capacity import build_prompt_capacity_receipt
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes
from true_rl_agent.tasks.prompt_adapter import (
    MAX_TASK_PROMPT_SECTIONS_BYTES,
    PromptPublicHistoryProjection,
    sanitize_task_prompt_sections,
)
from true_rl_agent.tasks.stair_climbing.component_smoke import (
    StairClimbingComponentSmokeAdapter,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.evaluation_profile import (
    StairEvaluationProfile,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    FIRST_VALID_TOP_ARRIVAL_COMPLETION_MODE,
    STABLE_TOP_TRAVERSAL_COMPLETION_MODE,
)
from true_rl_agent.tasks.stair_climbing.exploration import (
    StairComparisonStats,
    StairExplorationProjector,
)
from true_rl_agent.tasks.stair_climbing.public_projection import (
    build_stair_prompt_feedback,
    build_stair_prompt_score_semantics,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    classify_stair_selector_trial_eligibility,
    stair_selection_contract,
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
from true_rl_agent.tasks.stair_climbing.unified_training import (
    STAIR_TERRAIN_REGIME_C2,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    UNIFIED_STAIR_PROMPT_OBJECTIVE_SCORE_SOURCE,
    build_unified_stair_prompt_feedback,
    unified_stair_prompt_named_guard_score_source,
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
_FIRST_ARRIVAL_FEEDBACK_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.prompt_feedback.v10"
)
_LEGACY_COMPACT_FEEDBACK_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.prompt_feedback.v8"
)
_COMPACT_FEEDBACK_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.feedback_entry_projection.v6"
)
_STRATUM_FEEDBACK_TABLE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.stratum_feedback_table.v3"
)
_FACTOR_FEEDBACK_TABLE_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.factor_feedback_table.v1"
)
_FIRST_ARRIVAL_DERIVED_DIAGNOSTIC_RATES = {
    "first_valid_top_arrival_achieved": "success_rate",
}
_COMPACT_FEEDBACK_SOURCE_SPLIT = "public_dev"
_COMPACT_COUNT_GRID_ROW_SEPARATOR = ";"
_COMPACT_COUNT_GRID_COLUMN_SEPARATOR = ","
_COMPACT_FEEDBACK_KEYS = (
    "climb_completion_summary",
    "stable_success_summary",
    "diagnostic_flag_rates",
    "dominant_failure",
    "generalization_summary",
    "primary_failure_breakdown",
    "progress_summary",
)
_COMPACT_GENERALIZATION_KEYS = (
    "public_dev_objective_score",
    "j_id",
    "j_ood",
    "named_guard_scores",
    "family_scores",
    "qualified_ood_families",
    "objective_family_mask_hash",
    "success_count_grid",
    "zero_count_grids_omitted",
    "primary_failure_count_grids",
    "ordered_suite_hash",
    "split_cohort_hash",
    "success_rate_by_cell",
    "success_count_by_cell",
    "episodes_per_cell",
    "factor_diagnostics",
    "stratum_diagnostics",
)
_UNIFIED_SELECTOR_TRIAL_SUMMARY_KEYS = frozenset(
    {
        "trial_id",
        "lane_id",
        "candidate_id",
        "program_source_checksum",
        "selected",
        "trial_status",
        "objective_score",
        "objective_delta",
        "named_guard_scores",
        "selector_guard_floor_score",
        "selector_guard_regression",
        "selector_guard_max_regression",
        "selector_eligible",
        "selector_status",
        "generation_selector_reason",
        "runtime_effect_evidence",
        "observed_effect",
        "reason_codes",
    }
)


def _unified_suite_prompt_projection(suite: Any) -> dict[str, Any]:
    """Project public suite semantics without copying its task-owned declarations."""

    cells = suite.cells_for_cohort("public_dev")
    if not cells:
        raise ValueError("unified Stair public_dev suite is empty")

    strata: list[str] = []
    cell_counts: dict[str, int] = {}
    claim_kinds: dict[str, str] = {}
    for cell in cells:
        stratum = str(cell.stratum_id)
        claim_kind = str(cell.claim_kind)
        if stratum not in cell_counts:
            strata.append(stratum)
            cell_counts[stratum] = 0
            claim_kinds[stratum] = claim_kind
        elif claim_kinds[stratum] != claim_kind:
            raise ValueError(
                f"unified Stair stratum {stratum!r} mixes claim kinds"
            )
        cell_counts[stratum] += 1

    unique_counts = set(cell_counts.values())
    if len(unique_counts) != 1:
        raise ValueError("unified Stair prompt requires equal cells per stratum")
    factor_names = tuple(str(name) for name in cells[0].factor_levels)
    if not factor_names or any(
        tuple(str(name) for name in cell.factor_levels) != factor_names
        for cell in cells[1:]
    ):
        raise ValueError("unified Stair prompt factor vocabulary drifted")
    return {
        "strata": strata,
        "cells_per_stratum": unique_counts.pop(),
        "claim_kinds_by_stratum": claim_kinds,
        "factor_names": list(factor_names),
    }


def _unified_claim_prompt_semantics(claim_kind: str) -> dict[str, Any]:
    """Explain one frozen suite claim without redefining suite geometry."""

    if claim_kind == "joint_scalar_extrapolation":
        return {
            "protocol_claim_kind": claim_kind,
            "meaning": (
                "simultaneous scalar extrapolation of stair riser height and "
                "stair tread depth"
            ),
            "robot_joint_state_ood": False,
        }
    return {
        "protocol_claim_kind": claim_kind,
        "meaning": claim_kind,
    }


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
_PUBLIC_FEEDBACK_SUMMARY_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.public_feedback_summary.v13"
)
_MAX_LATEST_GENERATION_TRIAL_COMPARISONS = 2
_MAX_GENERATION_HISTORY_ID_CHARS = 128
_MAX_GENERATION_TRIAL_ID_CHARS = _MAX_GENERATION_HISTORY_ID_CHARS + max(
    len(f"-{lane_id}") for lane_id in GENERATION_LANE_IDS
)


@dataclass(frozen=True)
class StairClimbingPromptAdapter:
    config: StairClimbingTaskConfig
    session_start_profile: StairSessionStartProfile | None = None
    resolved_cell_manifest: ResolvedCellManifestV1 | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    task_id: str = STAIR_TASK_ID
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ) = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, StairClimbingTaskConfig):
            raise TypeError("stair prompt adapter requires StairClimbingTaskConfig")
        if self.config.is_unified_protocol and not isinstance(
            self.resolved_cell_manifest,
            ResolvedCellManifestV1,
        ):
            raise ValueError("Stair VNext prompt requires the resolved manifest")
        profile = self.session_start_profile
        training_binding_hash = self.config.formal_training_binding().binding_hash
        if profile is None:
            profile = resolve_stair_session_start_profile(
                STAIR_DEFAULT_START_MODE,
                task_protocol_hash=self.config.task_protocol_hash,
                training_binding_hash=training_binding_hash,
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
            != self.config.formal_training_binding().binding_hash
        ):
            raise ValueError("stair prompt start profile training binding mismatch")

    def build_prompt_sections(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        public_context: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        policy = self.prompt_public_record_scrub_policy()
        prompt_projection = self.project_prompt_public_history(
            public_records=public_records,
            initial_anchor_public_evidence=(
                public_context.get("initial_anchor_public_evidence")
                if isinstance(
                    public_context.get("initial_anchor_public_evidence"),
                    Mapping,
                )
                else None
            ),
        )
        safe_records = sanitize_public_records_with_policy(
            prompt_projection.records,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        latest_generation_trial_comparisons = (
            _latest_generation_trial_comparisons(
                safe_records,
                config=self.config,
                objective_family_mask_provider=(
                    self.objective_family_mask_provider
                ),
            )
        )
        prompt_context = (
            _project_unified_prompt_public_context(
                public_context,
                config=self.config,
                objective_family_mask_provider=(
                    self.objective_family_mask_provider
                ),
            )
            if self.config.is_unified_protocol
            else public_context
        )
        safe_context = sanitize_public_value_with_policy(
            prompt_context,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        if not isinstance(safe_context, Mapping):
            safe_context = {}
        safe_comparisons = sanitize_public_value_with_policy(
            latest_generation_trial_comparisons,
            blocked_key_parts=policy["blocked_key_parts"],
            blocked_text_parts=policy["blocked_text_parts"],
        )
        if not isinstance(safe_comparisons, list):
            safe_comparisons = []
        runtime_feedback = _latest_runtime_feedback(safe_records)
        sections = self._prompt_sections(
            public_feedback=_public_feedback_summary(
                _feedback_trial_records(safe_records, config=self.config,
                                        objective_family_mask_provider=self.objective_family_mask_provider),
                safe_context,
                latest_generation_trial_comparisons=safe_comparisons,
                use_public_dev_objective=self.config.is_unified_protocol,
                latest_nonaccepted_trial_id=runtime_feedback.get("source_round_id"),
            ),
            runtime_feedback=runtime_feedback,
        )
        sections = _compact_public_feedback_to_fit_task_sections(
            sections,
            evaluation_profile=self.config.evaluation_profile_config,
        )
        return sanitize_task_prompt_sections(sections)

    def build_prompt_capacity_preflight_sections(self) -> Mapping[str, Any]:
        """Stress the task-owned minimum-core path before live work starts."""

        sections = self._prompt_sections(
            public_feedback=_stair_capacity_preflight_feedback(
                self.config
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
        smoke_adapter = StairClimbingComponentSmokeAdapter(
            self.config, resolved_cell_manifest=self.resolved_cell_manifest,
        )
        reward_inputs_contract = stair_reward_input_prompt_contract(smoke_adapter.contract)
        reward_inputs_contract["nominal_morphology"] = smoke_adapter.nominal_morphology
        input_type_name = str(reward_inputs_contract["input_type"]).rsplit(".", 1)[-1]
        return {
            "task": self._task_section(),
            "reward_inputs_contract": reward_inputs_contract,
            "public_eval_contract": self._public_eval_contract(),
            "public_feedback_summary": dict(public_feedback),
            "runtime_feedback_summary": dict(runtime_feedback),
            "known_good_examples": [],
            "selection_guidance": self._selection_guidance(
                input_type_name=input_type_name
            ),
        }

    def nonaccepted_authoring_tiebreak_key(
        self,
        public_record: Mapping[str, Any],
    ) -> tuple[float, ...]:
        _ = public_record
        return ()

    def _selection_guidance(self, *, input_type_name: str) -> list[str]:
        profile = self.session_start_profile
        if not isinstance(profile, StairSessionStartProfile):
            raise RuntimeError("stair prompt session start profile is unresolved")
        guidance = [
            "Use only validated public failure/diagnostic/progress summaries.",
            "After rejection, use current_best as the default parent; reuse a rejected "
            "mechanism or weight only with public evidence.",
            "Keep each candidate attributable: change one reward mechanism, preserve "
            "current-best coefficients, and avoid unrelated retuning.",
            "Prefer a recent-untried mechanism only when task score, failure, or "
            "progress evidence supports it. Diagnostic rates, even extreme ones, are "
            "correlates rather than optimization targets or causal proof; judge a "
            "hypothesis by objective and guard movement.",
            "Code plus verifier/runtime evidence are authoritative over prose rationale; "
            "claim a maximum weighted contribution only with conditions, clips, weight, "
            "and an analytic bound.",
        ]
        completion_mode = str(
            self.config.episode_outcomes()["success"]["completion_mode"]
        )
        if completion_mode == FIRST_VALID_TOP_ARRIVAL_COMPLETION_MODE:
            guidance.extend(
                [
                    "Success is the first physically valid top-platform arrival; "
                    "post-arrival stopping position and settling are not evaluated.",
                    "climb_completion_summary.success_rate and "
                    "first_valid_top_arrival_rate describe the same absorbing task "
                    "completion event.",
                ]
            )
        elif completion_mode == STABLE_TOP_TRAVERSAL_COMPLETION_MODE:
            guidance.extend(
                [
                    "Stable success is "
                    "stable_success_summary.stable_top_success_rate; "
                    "valid_top_arrival_rate is diagnostic-only, and their gap means "
                    "arrival without stable success.",
                    "post_arrival_safety_failure_detected marks failure after top "
                    "arrival and before stable success.",
                ]
            )
        else:
            raise ValueError("unsupported Stair completion mode")
        if self.config.is_unified_protocol:
            guidance.append(
                "Optimize J_primary=(2*J_OOD+J_ID)/3 while all three runner-owned "
                "named guards J_OOD, C0-ID, and C1-ID remain within their paired "
                "non-regression tolerances; no score can compensate for a failed guard."
            )
        else:
            guidance.append(
                "Objective materiality and the guard cap are conjunctive; neither can "
                "compensate. When goal_not_reached_detected is frequent, preserve "
                "locomotion/obstacle capacity; penalties alone are insufficient."
            )
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
                    "Author the complete reward from the declared "
                    f"{input_type_name} contract and validated public evidence only.",
                ]
            )
        else:  # Defensive: the profile validates the closed mode set.
            raise ValueError("unsupported Stair prompt session start profile")
        guidance.extend(
            [
                "When the shared candidate_capabilities section marks "
                "complete_reward_program prompt-visible, return a complete replacement "
                "reward program, not an additive patch.",
                f"Use only fields declared by the {input_type_name} contract.",
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

    def project_prompt_public_history(
        self,
        *,
        public_records: Sequence[Mapping[str, Any]],
        initial_anchor_public_evidence: Mapping[str, Any] | None,
    ) -> PromptPublicHistoryProjection:
        if not self.config.is_unified_protocol:
            return PromptPublicHistoryProjection(
                records=tuple(dict(record) for record in public_records),
            )
        projected = tuple(
            _project_unified_prompt_public_record(
                record,
                config=self.config,
                objective_family_mask_provider=(
                    self.objective_family_mask_provider
                ),
            )
            for record in public_records
        )
        anchor_score = None
        if initial_anchor_public_evidence is not None:
            anchor_result = _mapping(
                initial_anchor_public_evidence.get("public_result")
            )
            anchor_feedback = _validated_unified_prompt_feedback(
                anchor_result,
                config=self.config,
                objective_family_mask_provider=(
                    self.objective_family_mask_provider
                ),
            )
            anchor_score = _prompt_feedback_objective_score(anchor_feedback)
            if anchor_score is None:
                raise ValueError(
                    "unified Stair initial anchor lacks public_dev prompt score"
                )
        return PromptPublicHistoryProjection(
            records=projected,
            public_score_source=UNIFIED_STAIR_PROMPT_OBJECTIVE_SCORE_SOURCE,
            initial_anchor_score=anchor_score,
        )

    def _task_section(self) -> dict[str, Any]:
        if self.config.is_unified_protocol:
            return self._unified_task_section()
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
            "name": "Stair Climbing",
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

    def _unified_task_section(self) -> dict[str, Any]:
        protocol = self.config.unified_protocol
        training = protocol.training
        training_profile = self.config.unified_training_profile
        suite = self.config.ordered_evaluation_suite
        suite_projection = _unified_suite_prompt_projection(suite)
        profile_payload = training_profile.to_dict()
        training_exposure = profile_payload["training_exposure"]
        training_phases = {
            int(training_profile.pattern(pattern_id)["training_phase_steps"])
            for pattern_id in training_profile.pattern_ids
        }
        if len(training_phases) != 1:
            raise ValueError("unified Stair prompt requires one training phase")
        c2_training_exposure = (
            "forbidden"
            if STAIR_TERRAIN_REGIME_C2 in training_exposure["forbidden_regimes"]
            and STAIR_TERRAIN_REGIME_C2 not in training_exposure["allowed_regimes"]
            else "allowed"
        )
        authority = protocol.to_dict()["candidate_authority"]
        return {
            "schema_version": "true_rl_agent.task_prompt.stair_climbing.v4",
            "task_id": self.task_id,
            "name": "Stair Climbing",
            "objective": (
                "Improve one reward program for the unified Stair task across runner-owned "
                "C0/C1 training regimes, with OOD performance as the primary emphasis and "
                "in-distribution performance protected by named guards."
            ),
            "candidate_authoring_status": "complete_reward_program_only",
            "fixed_train_protocol_summary": {
                "budget_steps": self.config.budget_steps,
                "train_num_envs": self.config.train_num_envs,
                "train_max_iterations": self.config.train_max_iterations,
                "train_steps_per_iteration": self.config.train_steps_per_iteration,
                "policy_initialization": training["policy_initialization"],
                "terrain_regime_allocation": dict(training_profile.allocation),
                "c0_support": dict(training_profile.terrain["c0"]),
                "c1_support": dict(training_profile.terrain["c1"]),
                "c1_pattern_families": list(training_profile.pattern_ids),
                "c1_training_phase_steps": training_phases.pop(),
                "c2_training_exposure": c2_training_exposure,
                "adaptive_difficulty_authority": "runner_owned",
                "complete_reward_runtime_available": True,
            },
            "fixed_eval_protocol_summary": {
                **self.config.evaluation_protocol().to_public_dict(),
                "ordered_evaluation_suite_hash": suite.suite_hash,
                "strata": suite_projection["strata"],
                "cells_per_stratum": suite_projection["cells_per_stratum"],
                "episodes_per_seed_per_cell": suite.episodes_per_seed_per_cell,
                "seed_values_exposed": False,
                "statistics_unit": "cell_then_stratum_then_paired_seed",
                "success_criteria": _success_criteria_prompt_contract(
                    self.config.episode_outcomes()
                ),
            },
            "primary_outcomes": list(self.config.episode_outcomes()["primary_reasons"]),
            "diagnostic_flags": list(
                self.config.episode_outcomes()["diagnostic_flags"]
            ),
            "candidate_must_not_change": [
                "checkpoint_policy" if item == "checkpoint_selector" else str(item)
                for item in authority["runner_owned_components"]
            ],
        }

    def _public_eval_contract(self) -> dict[str, Any]:
        acceptance_protocol = self.config.acceptance_protocol
        if self.config.is_unified_protocol:
            suite = self.config.ordered_evaluation_suite
            suite_projection = _unified_suite_prompt_projection(suite)
            score_contract = suite.score_contract
            exposure = self.config.unified_training_profile.to_dict()[
                "training_exposure"
            ]
            c2_strata = {
                str(cell.stratum_id)
                for cell in suite.cells_for_cohort("public_dev")
                if str(cell.terrain_regime_id) == STAIR_TERRAIN_REGIME_C2
            }
            c2_is_evaluation_only = (
                len(c2_strata) == 1
                and c2_strata <= set(suite_projection["strata"])
                and STAIR_TERRAIN_REGIME_C2 in exposure["forbidden_regimes"]
                and STAIR_TERRAIN_REGIME_C2 not in exposure["allowed_regimes"]
            )
            if not c2_is_evaluation_only:
                raise ValueError(
                    "unified Stair C2 prompt claim requires zero training exposure"
                )
            selector_contract = stair_selection_contract(self.config)
            return {
                "schema_version": "true_rl_agent.public_eval_contract.stair.unified.v5",
                "prompt_score_semantics": {
                    "score_sources_by_role": {
                        "objective_score": (
                            UNIFIED_STAIR_PROMPT_OBJECTIVE_SCORE_SOURCE
                        ),
                        "named_guards": {
                            guard.name: unified_stair_prompt_named_guard_score_source(
                                guard.name
                            )
                            for guard in acceptance_protocol.named_guards
                        },
                    },
                    "source_split": "public_dev",
                    "acceptance_result_metrics_visible": False,
                },
                "primary_objective": str(score_contract["j_primary"]),
                "objective_structure": {
                    "j_id": str(score_contract["j_id"]),
                    "j_ood": str(score_contract["j_ood"]),
                    "ood_families": {
                        str(name): list(stratum_ids)
                        for name, stratum_ids in score_contract["families"].items()
                    },
                    "family_mask": {
                        "source_split": "public_accept_reference",
                        "mode": suite.qualification_rule["family_mask_mode"],
                        "minimum_reference_score": suite.qualification_rule[
                            "minimum_reference_score"
                        ],
                        "frozen_for_session": True,
                    },
                },
                "named_non_regression_guards": list(
                    score_contract["named_guards"]
                ),
                "public_dev_selector": {
                    "rule": selector_contract["rule"],
                    "objective_score_source": selector_contract[
                        "primary_score_source"
                    ],
                    "guard_seed_score_source": selector_contract[
                        "guard_score_source"
                    ],
                    "guard_anchor_aggregation": (
                        "mean_across_seeds_of_minimum_named_guard_per_seed"
                    ),
                    "guard_rule": selector_contract["guard_rule"],
                    "guard_max_regression": selector_contract[
                        "guard_max_regression"
                    ],
                    "runtime_effect_rule": selector_contract[
                        "runtime_effect_rule"
                    ],
                    "ranking_policy": selector_contract["ranking_policy"],
                    "acceptance_authority": False,
                },
                "aggregation": "episode_to_cell_to_stratum_to_paired_seed",
                "generalization_claims": dict(
                    suite_projection["claim_kinds_by_stratum"]
                ),
                "generalization_claim_semantics": {
                    stratum: _unified_claim_prompt_semantics(claim_kind)
                    for stratum, claim_kind in suite_projection[
                        "claim_kinds_by_stratum"
                    ].items()
                },
                "c2_evidence_scope": (
                    "current evaluation-only surface/contact mechanism shift; "
                    "zero training exposure; height-scan policy salience not claimed"
                ),
                "factor_diagnostic_table": {
                    "schema_version": _FACTOR_FEEDBACK_TABLE_SCHEMA_VERSION,
                    "role": "diagnostic_only_overlapping_group_aggregates",
                    "factor_names": suite_projection["factor_names"],
                    "columns": [
                        "factor_name",
                        "factor_level",
                        "cell_count",
                        "success_rate",
                        "mean_progress_fraction",
                    ],
                    "cell_membership_source": "fixed_ordered_public_dev_suite",
                    "cell_ids_repeated_per_candidate": False,
                    "interpretation": (
                        "groups overlap and are not objective weights; saturated "
                        "groups remain visible as capabilities to preserve"
                    ),
                },
                "stratum_diagnostic_table": {
                    "schema_version": _STRATUM_FEEDBACK_TABLE_SCHEMA_VERSION,
                    "role": "task_owned_failure_and_diagnostic_breakdown",
                    "cell_membership_source": "fixed_ordered_public_dev_suite",
                    "cell_ids_repeated_per_candidate": False,
                    "omitted_rate_columns_are_zero": True,
                },
                "completion_derived_diagnostic_rates": (
                    dict(_FIRST_ARRIVAL_DERIVED_DIAGNOSTIC_RATES)
                    if self.config.episode_outcomes()["success"][
                        "completion_mode"
                    ]
                    == FIRST_VALID_TOP_ARRIVAL_COMPLETION_MODE
                    else {}
                ),
                "public_feedback_source": "public_dev_unified_stratum_projection",
                "primary_failure_is_mutually_exclusive": True,
                "diagnostic_flags_may_overlap": True,
                "diagnostic_interpretation": (
                    "latched or recent-history correlates for hypothesis formation; "
                    "they may coexist with success and do not prove failure causality"
                ),
                "classification_warnings_block_pass": True,
            }
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
    *,
    latest_generation_trial_comparisons: Sequence[Mapping[str, Any]],
    use_public_dev_objective: bool = False,
    latest_nonaccepted_trial_id: str | None = None,
) -> dict[str, Any]:
    candidates = _candidate_feedback_entries(
        public_records,
        use_public_dev_objective=use_public_dev_objective,
    )
    selected = _select_candidate_feedback_entries(
        candidates, latest_nonaccepted_trial_id=latest_nonaccepted_trial_id)
    entries: list[dict[str, Any]] = []
    if not any(entry["accepted"] is True for entry in candidates):
        anchor = _reference_anchor_feedback_entry(
            public_context,
            use_public_dev_objective=use_public_dev_objective,
        )
        if anchor is not None:
            entries.append(anchor)
    entries.extend(selected)
    entries = [_project_feedback_entry_for_prompt(entry) for entry in entries]
    if use_public_dev_objective:
        entries = [entry if _is_detailed_feedback_entry(entry)
                   else _compact_historical_comparison_entry(entry) for entry in entries]
    comparisons = [
        dict(value)
        for value in latest_generation_trial_comparisons[
            :_MAX_LATEST_GENERATION_TRIAL_COMPARISONS
        ]
        if isinstance(value, Mapping)
    ]
    if entries or comparisons:
        summary = {
            "schema_version": _PUBLIC_FEEDBACK_SUMMARY_SCHEMA_VERSION,
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
        if use_public_dev_objective:
            summary.update(_share_unified_feedback_tables(entries))
        if comparisons:
            summary["latest_generation_trial_comparisons"] = comparisons
            if any("selector_guard_regression" in item for item in comparisons):
                summary["latest_generation_trial_comparison_basis"] = (
                    "public_dev_selector_vs_parent"
                )
            elif any("objective_delta" in item for item in comparisons):
                summary["latest_generation_trial_comparison_basis"] = (
                    "public_accept_vs_parent"
                )
            summary["limits"]["max_latest_generation_trial_comparisons"] = (
                _MAX_LATEST_GENERATION_TRIAL_COMPARISONS
            )
        return summary
    task_summary = public_context.get("task_public_summary")
    if isinstance(task_summary, Mapping) and task_summary:
        return {
            "schema_version": _PUBLIC_FEEDBACK_SUMMARY_SCHEMA_VERSION,
            "status": "memory_only",
            "task_public_summary": dict(task_summary),
        }
    return {
        "schema_version": _PUBLIC_FEEDBACK_SUMMARY_SCHEMA_VERSION,
        "status": "not_available",
        "reason": "no_validated_public_stair_evaluation",
    }


def _latest_generation_trial_comparisons(
    public_records: Sequence[Mapping[str, Any]],
    *,
    config: StairClimbingTaskConfig,
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ) = None,
) -> list[dict[str, Any]]:
    """Project the latest generation that has public-complete sibling evidence."""

    for record in reversed(public_records):
        generation = record.get("generation")
        if (
            isinstance(generation, Mapping)
            and generation.get("schema_version")
            == GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION
        ):
            comparisons = _generation_trial_comparisons(
                generation,
                config=config,
                objective_family_mask_provider=objective_family_mask_provider,
            )
            if (
                len(comparisons) == len(GENERATION_LANE_IDS)
                and {item["lane_id"] for item in comparisons}
                == set(GENERATION_LANE_IDS)
            ):
                return comparisons
    return []


def _generation_trial_comparisons(
    generation: Mapping[str, Any],
    *,
    config: StairClimbingTaskConfig,
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ) = None,
) -> list[dict[str, Any]]:
    generation_id = generation.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or not generation_id
        or len(generation_id) > _MAX_GENERATION_HISTORY_ID_CHARS
    ):
        raise ValueError("Stair generation prompt identity is invalid")

    trials = generation.get("trials")
    parent_mapping = generation.get("parent_current_best")
    if not isinstance(trials, list):
        raise ValueError("Stair generation prompt comparison evidence is invalid")
    if config.is_unified_protocol:
        return _generation_public_dev_trial_summaries(
            generation,
            trials=trials,
            config=config,
            objective_family_mask_provider=objective_family_mask_provider,
        )
    if not isinstance(parent_mapping, Mapping):
        raise ValueError("Stair generation prompt comparison parent is invalid")
    parent = _public_parent_comparison(
        parent_mapping,
        config=config,
    )
    selection = _mapping(generation.get("selection"))
    selected_trial_id = selection.get("selected_trial_id")
    projector = StairExplorationProjector(
        config,
        objective_family_mask=(
            objective_family_mask_provider()
            if objective_family_mask_provider is not None
            else None
        ),
    )
    comparisons: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("Stair generation prompt trial evidence is invalid")
        if not isinstance(trial.get("public_accept"), Mapping):
            continue
        assessment = projector.assess_trial(
            generation=generation,
            trial=trial,
            parent=parent,
        )
        summary = _mapping(assessment.public_evidence_summary)
        if summary.get("status") != "public_complete":
            continue
        objective = _mapping(summary.get("objective"))
        guard = _mapping(summary.get("guard"))
        objective_delta = _finite_score(objective.get("delta"))
        objective_threshold = _finite_score(
            objective.get("materiality_threshold")
        )
        guard_delta = _finite_score(guard.get("delta"))
        guard_threshold = _finite_score(guard.get("max_regression"))
        if None in {
            objective_delta,
            objective_threshold,
            guard_delta,
            guard_threshold,
        } or not isinstance(objective.get("materiality_met"), bool) or not isinstance(
            guard.get("guard_met"), bool
        ):
            raise ValueError("Stair generation prompt comparison is incomplete")
        trial_id = trial.get("trial_id")
        lane_id = trial.get("lane_id")
        trial_status = trial.get("trial_status")
        if (
            not isinstance(trial_id, str)
            or not trial_id
            or len(trial_id) > _MAX_GENERATION_TRIAL_ID_CHARS
            or not isinstance(lane_id, str)
            or not lane_id
            or not isinstance(trial_status, str)
            or not trial_status
        ):
            raise ValueError("Stair generation prompt trial identity is invalid")
        candidate = trial.get("candidate")
        if not isinstance(candidate, Mapping):
            raise ValueError("Stair generation prompt candidate identity is absent")
        candidate_id = require_candidate_id(
            candidate.get("candidate_id"),
            "Stair generation prompt candidate_id",
        )
        program_source_checksum = require_sha256_checksum(
            candidate.get("program_source_checksum"),
            "Stair generation prompt program_source_checksum",
        )
        comparisons.append(
            {
                "generation_id": generation_id,
                "trial_id": trial_id,
                "lane_id": lane_id,
                "candidate_id": candidate_id,
                "program_source_checksum": program_source_checksum,
                "trial_status": trial_status,
                "selection_status": (
                    "selected"
                    if trial_id == selected_trial_id
                    else "not_selected"
                ),
                "observed_effect": assessment.observed_effect,
                "objective_delta": objective_delta,
                "objective_materiality_threshold": objective_threshold,
                "objective_materiality_met": objective["materiality_met"],
                "guard_delta": guard_delta,
                "guard_max_regression": guard_threshold,
                "guard_met": guard["guard_met"],
                "reason_codes": list(assessment.reason_codes),
            }
        )
    if len(comparisons) > _MAX_LATEST_GENERATION_TRIAL_COMPARISONS:
        raise ValueError("Stair generation prompt comparison width exceeds two")
    return comparisons


def _generation_public_dev_trial_summaries(
    generation: Mapping[str, Any],
    *,
    trials: Sequence[Any],
    config: StairClimbingTaskConfig,
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ),
) -> list[dict[str, Any]]:
    raw_selection = generation.get("selection")
    if raw_selection is None:
        return []
    if not isinstance(raw_selection, Mapping):
        raise ValueError("Stair public_dev selector evidence must be an object or null")
    selection = raw_selection
    selected_trial_id = selection.get("selected_trial_id")
    selection_reason = selection.get("reason")
    selector_contract = stair_selection_contract(config)
    if selection_reason not in selector_contract["reason_codes"]:
        raise ValueError("Stair public_dev selector reason is invalid")
    raw_guard_regressions = _mapping(
        selection.get("guard_regression_by_trial")
    )
    raw_eligible_trial_ids = selection.get("selection_eligible_trial_ids")
    if not isinstance(raw_eligible_trial_ids, list | tuple):
        raise ValueError("Stair public_dev selector eligibility is missing")
    eligible_trial_ids = {str(trial_id) for trial_id in raw_eligible_trial_ids}
    guard_max_regression = float(selector_contract["guard_max_regression"])
    projector = StairExplorationProjector(
        config,
        objective_family_mask=(
            objective_family_mask_provider()
            if objective_family_mask_provider is not None
            else None
        ),
    )
    summaries: list[dict[str, Any]] = []
    for trial in trials:
        if not isinstance(trial, Mapping):
            raise ValueError("Stair generation prompt trial evidence is invalid")
        if not isinstance(trial.get("public_dev"), Mapping):
            continue
        assessment = projector.assess_trial(
            generation=generation,
            trial=trial,
            parent=None,
        )
        summary = _mapping(assessment.public_evidence_summary)
        if (
            summary.get("status") != "public_complete"
            or summary.get("source_split") != "public_dev"
        ):
            continue
        objective = _mapping(summary.get("objective"))
        objective_score = _finite_score(objective.get("candidate"))
        objective_delta = _finite_score(objective.get("delta"))
        guard_floor = _mapping(summary.get("guard_floor"))
        selector_guard_floor_score = _finite_score(guard_floor.get("candidate"))
        guard_floor_delta = _finite_score(guard_floor.get("delta"))
        raw_guards = _mapping(summary.get("named_guards"))
        named_guard_scores = {
            str(name): score
            for name, item in raw_guards.items()
            for score in [_finite_score(_mapping(item).get("candidate"))]
            if score is not None
        }
        if (
            objective_score is None
            or objective_delta is None
            or selector_guard_floor_score is None
            or guard_floor_delta is None
            or set(named_guard_scores)
            != {guard.name for guard in config.acceptance_protocol.named_guards}
        ):
            raise ValueError("Stair public_dev generation summary is incomplete")
        identity = _generation_prompt_trial_identity(
            trial,
            selected_trial_id=selected_trial_id,
        )
        trial_id = str(identity["trial_id"])
        selector_guard_regression = _finite_score(
            raw_guard_regressions.get(trial_id)
        )
        if selector_guard_regression is None or not math.isclose(
            selector_guard_regression,
            -guard_floor_delta,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Stair public_dev selector guard evidence drifted")
        runtime_effect = _mapping(summary.get("runtime_effect_evidence"))
        if runtime_effect.get("status") not in {"pass", "fail"} or not isinstance(
            runtime_effect.get("reason"), str
        ):
            raise ValueError("Stair public_dev runtime-effect evidence is missing")
        selector_eligible = trial_id in eligible_trial_ids
        eligibility_status = classify_stair_selector_trial_eligibility(
            guard_regression=selector_guard_regression,
            guard_max_regression=guard_max_regression,
            runtime_effect_status=str(runtime_effect["status"]),
        )
        if selector_eligible != (eligibility_status == "eligible"):
            raise ValueError("Stair public_dev selector eligibility evidence drifted")
        if selector_eligible:
            selector_status = (
                "selected"
                if trial_id == selected_trial_id
                else "eligible_not_selected"
                if selected_trial_id is not None
                else "inconclusive"
            )
        else:
            selector_status = eligibility_status
        summaries.append(
            _validated_unified_selector_trial_summary(
                {
                    **identity,
                    "objective_score": objective_score,
                    "objective_delta": objective_delta,
                    "named_guard_scores": named_guard_scores,
                    "selector_guard_floor_score": selector_guard_floor_score,
                    "selector_guard_regression": selector_guard_regression,
                    "selector_guard_max_regression": guard_max_regression,
                    "selector_eligible": selector_eligible,
                    "selector_status": selector_status,
                    "generation_selector_reason": selection_reason,
                    "runtime_effect_evidence": dict(runtime_effect),
                    "observed_effect": assessment.observed_effect,
                    "reason_codes": list(assessment.reason_codes),
                }
            )
        )
    if len(summaries) > _MAX_LATEST_GENERATION_TRIAL_COMPARISONS:
        raise ValueError("Stair generation prompt comparison width exceeds two")
    return summaries


def _generation_prompt_trial_identity(
    trial: Mapping[str, Any],
    *,
    selected_trial_id: Any,
) -> dict[str, Any]:
    trial_id = trial.get("trial_id")
    lane_id = trial.get("lane_id")
    trial_status = trial.get("trial_status")
    if (
        not isinstance(trial_id, str)
        or not trial_id
        or len(trial_id) > _MAX_GENERATION_TRIAL_ID_CHARS
        or not isinstance(lane_id, str)
        or not lane_id
        or not isinstance(trial_status, str)
        or not trial_status
    ):
        raise ValueError("Stair generation prompt trial identity is invalid")
    candidate = trial.get("candidate")
    if not isinstance(candidate, Mapping):
        raise ValueError("Stair generation prompt candidate identity is absent")
    return {
        "trial_id": trial_id,
        "lane_id": lane_id,
        "candidate_id": require_candidate_id(
            candidate.get("candidate_id"),
            "Stair generation prompt candidate_id",
        ),
        "program_source_checksum": require_sha256_checksum(
            candidate.get("program_source_checksum"),
            "Stair generation prompt program_source_checksum",
        ),
        "selected": trial_id == selected_trial_id,
        "trial_status": trial_status,
    }


def _validated_unified_selector_trial_summary(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    if set(value) != _UNIFIED_SELECTOR_TRIAL_SUMMARY_KEYS:
        raise ValueError("Stair unified selector trial summary fields drifted")
    return dict(value)


@dataclass(frozen=True)
class _PublicParentComparison(StairComparisonStats):
    objective_score: float
    objective_samples: tuple[float, ...]
    guard_score: float
    guard_samples: tuple[float, ...]
    named_guard_scores: Mapping[str, float]
    named_guard_samples: Mapping[str, tuple[float, ...]]
    acceptance_protocol_hash: str

    def score_for_comparison(self) -> float:
        return self.objective_score

    def score_samples_for_comparison(
        self,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]:
        return (
            self.objective_samples
            if acceptance_protocol.protocol_hash == self.acceptance_protocol_hash
            else ()
        )

    def guard_score_for_comparison(self) -> float:
        return self.guard_score

    def guard_score_samples_for_comparison(
        self,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]:
        return (
            self.guard_samples
            if acceptance_protocol.protocol_hash == self.acceptance_protocol_hash
            else ()
        )

    def named_guard_score_for_comparison(self, name: str) -> float | None:
        return self.named_guard_scores.get(name)

    def named_guard_score_samples_for_comparison(
        self,
        name: str,
        acceptance_protocol: AcceptanceProtocol,
    ) -> tuple[float, ...]:
        if acceptance_protocol.protocol_hash != self.acceptance_protocol_hash:
            return ()
        return self.named_guard_samples.get(name, ())


def _public_parent_comparison(
    value: Mapping[str, Any],
    *,
    config: StairClimbingTaskConfig,
) -> _PublicParentComparison:
    """Validate only the public score inputs retained after prompt scrubbing."""

    required_fields = {
        "mode",
        "anchor_kind",
        "anchor_id",
        "public_score",
        "public_accept_stats",
        "guard_score_stats",
        "min_score_count",
        "task_protocol_hash",
        "eval_protocol_hash",
        "acceptance_protocol_hash",
    }
    if not required_fields.issubset(value) or "sealed_validation_evidence" in value:
        raise ValueError("Stair public parent comparison fields mismatch")
    for field_name in ("mode", "anchor_kind", "anchor_id"):
        field_value = value.get(field_name)
        if not isinstance(field_value, str) or not field_value:
            raise ValueError("Stair public parent comparison identity is invalid")
    protocol = config.acceptance_protocol
    eval_protocol_hash = config.evaluation_protocol().protocol_hash
    if (
        value.get("task_protocol_hash") != config.task_protocol_hash
        or value.get("eval_protocol_hash") != eval_protocol_hash
        or value.get("acceptance_protocol_hash") != protocol.protocol_hash
    ):
        raise ValueError("Stair public parent comparison protocol mismatch")
    min_score_count = value.get("min_score_count")
    if (
        isinstance(min_score_count, bool)
        or not isinstance(min_score_count, int)
        or min_score_count <= 0
    ):
        raise ValueError("Stair public parent comparison count is invalid")
    objective_stats = value.get("public_accept_stats")
    guard_stats = value.get("guard_score_stats")
    if not isinstance(objective_stats, Mapping) or not isinstance(guard_stats, Mapping):
        raise ValueError("Stair public parent comparison stats are invalid")
    objective_validation = validate_score_stats(
        objective_stats,
        minimum_count=min_score_count,
        require_std=False,
        require_standard_error=min_score_count >= 2,
        require_protocol_hash=True,
    )
    if (
        not objective_validation.valid
        or objective_validation.mean is None
        or objective_validation.eval_protocol_hash != eval_protocol_hash
    ):
        raise ValueError("Stair public parent comparison stats are incomplete")
    objective_samples = validated_score_samples(
        objective_stats.get("score_samples"),
        expected_mean=objective_validation.mean,
        protocol=protocol,
        field_name="Stair public parent objective score_samples",
    )
    named_guard_scores: dict[str, float] = {}
    named_guard_samples: dict[str, tuple[float, ...]] = {}
    if config.is_unified_protocol:
        raw_named = value.get("named_guard_evidence")
        if not isinstance(raw_named, Mapping) or set(raw_named) != {
            guard.name for guard in protocol.named_guards
        }:
            raise ValueError("Stair unified parent named guards are incomplete")
        for guard in protocol.named_guards:
            item = raw_named.get(guard.name)
            if not isinstance(item, Mapping) or item.get("score_source") != guard.score_source:
                raise ValueError("Stair unified parent named guard source mismatch")
            stats = item.get("score_stats")
            if not isinstance(stats, Mapping):
                raise ValueError("Stair unified parent named guard stats are invalid")
            validation = validate_score_stats(
                stats,
                minimum_count=min_score_count,
                require_std=True,
                require_standard_error=min_score_count >= 2,
                require_protocol_hash=True,
            )
            if (
                not validation.valid
                or validation.mean is None
                or validation.eval_protocol_hash != eval_protocol_hash
            ):
                raise ValueError("Stair unified parent named guard stats are incomplete")
            samples = validated_score_samples(
                item.get("score_samples"),
                expected_mean=validation.mean,
                protocol=protocol,
                field_name=f"Stair public parent {guard.name} score_samples",
            )
            named_guard_scores[guard.name] = validation.mean
            named_guard_samples[guard.name] = samples
        limiting_name = min(named_guard_scores, key=named_guard_scores.__getitem__)
        guard_mean = named_guard_scores[limiting_name]
        guard_samples = tuple(
            min(named_guard_samples[name][index] for name in named_guard_samples)
            for index in range(len(named_guard_samples[limiting_name]))
        )
    else:
        guard_validation = validate_score_stats(
            guard_stats,
            minimum_count=min_score_count,
            require_std=True,
            require_standard_error=min_score_count >= 2,
            require_protocol_hash=True,
        )
        if (
            not guard_validation.valid
            or guard_validation.mean is None
            or guard_validation.eval_protocol_hash != eval_protocol_hash
        ):
            raise ValueError("Stair public parent comparison stats are incomplete")
        guard_mean = guard_validation.mean
        guard_samples = validated_score_samples(
            guard_stats.get("score_samples"),
            expected_mean=guard_validation.mean,
            protocol=protocol,
            field_name="Stair public parent guard score_samples",
        )
    return _PublicParentComparison(
        objective_score=objective_validation.mean,
        objective_samples=objective_samples,
        guard_score=guard_mean,
        guard_samples=guard_samples,
        named_guard_scores=named_guard_scores,
        named_guard_samples=named_guard_samples,
        acceptance_protocol_hash=protocol.protocol_hash,
    )


def _stair_capacity_preflight_feedback(
    config: StairClimbingTaskConfig,
) -> dict[str, Any]:
    """Build a conservative bounded shape retained by Stair minimum-core."""

    if config.is_unified_protocol:
        return _stair_unified_capacity_preflight_feedback(config)
    evaluation_profile = config.evaluation_profile

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
        "schema_version": _PUBLIC_FEEDBACK_SUMMARY_SCHEMA_VERSION,
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
        "latest_generation_trial_comparisons": [
            {
                "generation_id": fill * _MAX_GENERATION_HISTORY_ID_CHARS,
                "trial_id": fill * _MAX_GENERATION_TRIAL_ID_CHARS,
                "lane_id": lane_id,
                "candidate_id": fill * MAX_CANDIDATE_ID_CHARS,
                "program_source_checksum": f"sha256:{fill * 64}",
                "trial_status": "qualified_not_selected",
                "selection_status": selection_status,
                "observed_effect": "mixed",
                "objective_delta": maximum_number,
                "objective_materiality_threshold": maximum_number,
                "objective_materiality_met": False,
                "guard_delta": maximum_number,
                "guard_max_regression": maximum_number,
                "guard_met": True,
                "reason_codes": [
                    "public_guard_improved_objective_not_improved"
                ],
            }
            for fill, lane_id, selection_status in (
                ("c", "lane-a", "not_selected"),
                ("d", "lane-b", "selected"),
            )
        ],
        "latest_generation_trial_comparison_basis": "public_accept_vs_parent",
        "omitted_candidate_feedback_entry_count": 28,
        "byte_budget_compaction_applied": True,
    }


def _stair_unified_capacity_preflight_feedback(
    config: StairClimbingTaskConfig,
) -> dict[str, Any]:
    suite = config.ordered_evaluation_suite
    cells = suite.cells_for_cohort("public_dev")
    # Unified scores, rates, and normalized progress are contract-bounded to
    # [0, 1]. Use a long, non-integral legal representation so the stress shape
    # does not collapse to the unusually short JSON encoding of 0.0 or 1.0.
    maximum_number = 0.12345678901234568
    completion_mode = str(
        config.episode_outcomes()["success"]["completion_mode"]
    )
    if completion_mode == FIRST_VALID_TOP_ARRIVAL_COMPLETION_MODE:
        completion_feedback = {
            "schema_version": _FIRST_ARRIVAL_FEEDBACK_SCHEMA_VERSION,
            "climb_completion_summary": {
                "completion_mode": completion_mode,
                "success_rate": maximum_number,
                "first_valid_top_arrival_rate": maximum_number,
            },
        }
    elif completion_mode == STABLE_TOP_TRAVERSAL_COMPLETION_MODE:
        completion_feedback = {
            "schema_version": _COMPACT_FEEDBACK_SCHEMA_VERSION,
            "stable_success_summary": {
                "stable_top_success_rate": maximum_number,
                "valid_top_arrival_rate": None,
                "top_arrival_without_stable_success_rate": None,
            },
        }
    else:
        raise ValueError("unsupported Stair completion mode")
    cell_values = {cell.cell_id: maximum_number for cell in cells}
    stratum_diagnostics = {
        stratum: {
            "cell_count": len(members),
            "cell_ids": [cell.cell_id for cell in members],
            "success_rate": maximum_number,
            "mean_progress_fraction": maximum_number,
            "score": maximum_number,
            "primary_failure_breakdown": {
                reason: maximum_number
                for reason in config.episode_outcomes()["failure_priority"]
            },
            "diagnostic_flag_rates": {
                flag: maximum_number
                for flag in config.episode_outcomes()["diagnostic_flags"]
            },
        }
        for stratum in sorted({cell.stratum_id for cell in cells})
        if (
            members := [cell for cell in cells if cell.stratum_id == stratum]
        )
    }
    factor_diagnostics: dict[str, Any] = {}
    if cells:
        for factor in next(iter(cells)).factor_levels:
            levels: dict[str, Any] = {}
            for level in dict.fromkeys(
                cell.factor_levels[factor] for cell in cells
            ):
                members = [
                    cell for cell in cells if cell.factor_levels[factor] == level
                ]
                levels[str(level)] = {
                    "cell_count": len(members),
                    "cell_ids": [cell.cell_id for cell in members],
                    "success_rate": maximum_number,
                    "mean_progress_fraction": maximum_number,
                }
            factor_diagnostics[str(factor)] = levels

    def entry(*, fill: str, accepted: bool, selection_basis: list[str]) -> dict[str, Any]:
        return _project_feedback_entry_for_prompt({
            "source_kind": "candidate",
            "source_public_result_hash": fill * 64,
            "candidate_id": fill * MAX_CANDIDATE_ID_CHARS,
            "program_source_checksum": f"sha256:{fill * 64}",
            "objective_score": maximum_number,
            "guard_score": maximum_number,
            "accepted": accepted,
            "selection_basis": selection_basis,
            "feedback": {
                **completion_feedback,
                "source_split": _COMPACT_FEEDBACK_SOURCE_SPLIT,
                "dominant_failure": {
                    "reason": "goal_not_reached_detected",
                    "rate": maximum_number,
                },
                "generalization_summary": {
                    "public_dev_objective_score": maximum_number,
                    "j_id": maximum_number,
                    "j_ood": maximum_number,
                    "named_guard_scores": {
                        name: maximum_number
                        for name in ("j_ood", "c0_id", "c1_id")
                    },
                    "family_scores": {
                        str(name): maximum_number
                        for name in suite.score_contract["families"]
                    },
                    "qualified_ood_families": list(
                        suite.score_contract["families"]
                    ),
                    "objective_family_mask_hash": "f" * 64,
                    "ordered_suite_hash": suite.suite_hash,
                    "split_cohort_hash": suite.cohort_hash("public_dev"),
                    "success_count_by_cell": {key: 999999 for key in cell_values},
                    "episodes_per_cell": 999999,
                    "factor_diagnostics": factor_diagnostics,
                    "stratum_diagnostics": stratum_diagnostics,
                },
                "primary_failure_breakdown": {
                    reason: maximum_number
                    for reason in config.episode_outcomes()["failure_priority"]
                },
                "diagnostic_flag_rates": {
                    flag: maximum_number
                    for flag in config.episode_outcomes()["diagnostic_flags"]
                },
                "progress_summary": {
                    "mean_progress_fraction": maximum_number,
                    "mean_progress_fraction_by_cell": dict(cell_values),
                },
            },
        })

    return {
        "schema_version": _PUBLIC_FEEDBACK_SUMMARY_SCHEMA_VERSION,
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
        "latest_generation_trial_comparisons": [
            _validated_unified_selector_trial_summary(
                {
                    "trial_id": fill * _MAX_GENERATION_TRIAL_ID_CHARS,
                    "lane_id": lane_id,
                    "candidate_id": fill * MAX_CANDIDATE_ID_CHARS,
                    "program_source_checksum": f"sha256:{fill * 64}",
                    "trial_status": "qualified_not_selected",
                    "selected": selected,
                    "objective_score": maximum_number,
                    "objective_delta": maximum_number,
                    "named_guard_scores": {
                        guard.name: maximum_number
                        for guard in config.acceptance_protocol.named_guards
                    },
                    "selector_guard_floor_score": maximum_number,
                    "selector_guard_regression": maximum_number,
                    "selector_guard_max_regression": maximum_number,
                    "selector_eligible": selected,
                    "selector_status": (
                        "selected" if selected else "guard_and_runtime_ineligible"
                    ),
                    "generation_selector_reason": "public_dev_guard_filter",
                    "runtime_effect_evidence": {
                        "schema_version": (
                            "true_rl_agent.task.stair_climbing."
                            "candidate_runtime_effect_evidence.v1"
                        ),
                        "status": "pass" if selected else "fail",
                        "reason": (
                            "candidate_reward_operation_runtime_effect_observed"
                            if selected
                            else "candidate_reward_operation_runtime_effect_not_observed"
                        ),
                        "acceptance_authority": False,
                    },
                    "observed_effect": "positive" if selected else "negative",
                    "reason_codes": [
                        "selected_public_dev_objective_improved_without_guard_regression"
                        if selected
                        else "candidate_reward_operation_runtime_effect_not_observed"
                    ],
                }
            )
            for fill, lane_id, selected in (
                ("c", "lane-a", False),
                ("d", "lane-b", True),
            )
        ],
        "latest_generation_trial_comparison_basis": (
            "public_dev_selector_vs_parent"
        ),
        "omitted_candidate_feedback_entry_count": 28,
        "byte_budget_compaction_applied": True,
    }


def _compact_public_feedback_to_fit_task_sections(
    sections: Mapping[str, Any],
    *,
    evaluation_profile: StairEvaluationProfile | None,
) -> dict[str, Any]:
    """Drop lower-priority detailed history before the shared hard limit."""

    original = dict(sections)
    bounded = dict(original)

    target_bytes = (
        MAX_TASK_PROMPT_SECTIONS_BYTES - _TASK_PROMPT_RESERVED_HEADROOM_BYTES
    )
    if canonical_prompt_json_bytes(bounded) <= target_bytes:
        return bounded

    # Static task and ABI prose is repeated every generation. Compact it before
    # touching validated feedback so scarce capacity is spent on measured evidence.
    if isinstance(original.get("public_feedback_summary"), Mapping):
        bounded = _static_core_stair_prompt_sections(original)
        if canonical_prompt_json_bytes(bounded) <= target_bytes:
            return bounded

        # Further feedback compaction changes the visible projection. Remove
        # the static-only receipt now and publish one receipt for the final
        # combined projection below.
        feedback_without_receipt = dict(
            _mapping(bounded.get("public_feedback_summary"))
        )
        feedback_without_receipt.pop("capacity_projection_receipt", None)
        bounded["public_feedback_summary"] = feedback_without_receipt

    feedback = _mapping(bounded.get("public_feedback_summary"))
    raw_entries = feedback.get("feedback_entries")

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

    if canonical_prompt_json_bytes(bounded) <= target_bytes:
        _attach_stair_capacity_receipt(
            bounded,
            original=original,
            projection_id=(
                "stair_climbing.static_contract_and_feedback_core.v1"
            ),
            omitted=(
                "reward_field_descriptions",
                "duplicated_task_identity_detail",
                "feedback_detail_for_capacity",
            ),
        )
        if canonical_prompt_json_bytes(bounded) <= target_bytes:
            return bounded

    # Minimum-core must bind directly to the original unprojected sections;
    # layering receipts or using an already compacted value creates a second
    # source of truth for what was omitted.
    return _minimum_core_stair_prompt_sections(
        original,
        target_bytes=target_bytes,
    )


def _static_core_stair_prompt_sections(
    sections: Mapping[str, Any],
) -> dict[str, Any]:
    """Compact repeated static contracts while preserving all runtime evidence."""

    original = copy.deepcopy(dict(sections))
    projected = copy.deepcopy(original)
    projected["task"] = _minimum_core_task_section(
        _mapping(original.get("task"))
    )
    projected["reward_inputs_contract"] = _minimum_core_reward_inputs_contract(
        _mapping(original.get("reward_inputs_contract"))
    )
    _attach_stair_capacity_receipt(
        projected,
        original=original,
        projection_id="stair_climbing.static_contract_core.v1",
        omitted=(
            "reward_field_descriptions",
            "duplicated_task_identity_detail",
        ),
    )
    return projected


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
    raw_comparisons = feedback.get("latest_generation_trial_comparisons")
    comparisons = (
        [
            dict(value)
            for value in raw_comparisons[
                :_MAX_LATEST_GENERATION_TRIAL_COMPARISONS
            ]
            if isinstance(value, Mapping)
        ]
        if isinstance(raw_comparisons, list | tuple)
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
    minimum_entries = []
    preferred_minimum_entry: dict[str, Any] | None = None
    for entry in selected:
        compact = _minimum_core_feedback_entry(entry)
        if current_best is not None and entry is latest and entry is not current_best:
            # Unified feedback must keep the latest trial's family/stratum
            # deltas: those are the comparative signal for the next reward edit.
            # Legacy feedback has no bounded stratum table and stays a compact
            # scalar comparison.
            compact = _compact_latest_feedback_entry(compact)
        minimum_entries.append(compact)
        if entry is latest:
            preferred_minimum_entry = compact
    if preferred_minimum_entry is None and minimum_entries:
        preferred_minimum_entry = minimum_entries[0]
    raw_omitted_count = feedback.get("omitted_candidate_feedback_entry_count", 0)
    omitted_count = (
        raw_omitted_count
        if isinstance(raw_omitted_count, int)
        and not isinstance(raw_omitted_count, bool)
        and raw_omitted_count >= 0
        else 0
    )
    if minimum_entries or comparisons:
        minimum_feedback: dict[str, Any] = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.public_feedback_minimum_core.v6"
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
                "true_rl_agent.task.stair_climbing.public_feedback_minimum_core.v6"
            ),
            "status": "no_closed_outcome",
            "reason": "no_validated_public_stair_evaluation",
            "feedback_entries": [],
            "omitted_candidate_feedback_entry_count": 0,
            "byte_budget_compaction_applied": True,
            "projection_level": "minimum_core",
        }
    # Shared headers are part of the retained rows' meaning, including repair
    # frames that need minimum-core compaction.
    for key in ("shared_evaluation_identity", "diagnostic_table_definitions"):
        if key in feedback:
            minimum_feedback[key] = copy.deepcopy(feedback[key])
    if comparisons:
        minimum_feedback["latest_generation_trial_comparisons"] = comparisons
        if any("selector_guard_regression" in item for item in comparisons):
            minimum_feedback["latest_generation_trial_comparison_basis"] = (
                "public_dev_selector_vs_parent"
            )
        elif any("objective_delta" in item for item in comparisons):
            minimum_feedback["latest_generation_trial_comparison_basis"] = (
                "public_accept_vs_parent"
            )
    minimum = {
        "task": _minimum_core_task_section(_mapping(original.get("task"))),
        "reward_inputs_contract": _minimum_core_reward_inputs_contract(
            _mapping(original.get("reward_inputs_contract"))
        ),
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
        "selection_guidance": _minimum_core_selection_guidance(
            original.get("selection_guidance")
        ),
    }
    base_omissions = (
        "historical_feedback_entries",
        "feedback_cell_grids",
        "feedback_diagnostic_detail",
        "reward_field_descriptions",
        "duplicated_eval_identity_detail",
        "guidance_prose_detail",
        "runtime_activation_detail",
    )
    _attach_stair_capacity_receipt(
        minimum,
        original=original,
        projection_id="stair_climbing.minimum_core.v6",
        omitted=base_omissions,
    )
    measured_bytes = canonical_prompt_json_bytes(minimum)
    if (
        measured_bytes > target_bytes
        and len(minimum_entries) > 1
        and preferred_minimum_entry is not None
    ):
        removed = [
            entry
            for entry in minimum_entries
            if entry is not preferred_minimum_entry
        ]
        omitted_candidates = sum(
            entry.get("source_kind") != "reference_anchor" for entry in removed
        )
        minimum_feedback["feedback_entries"] = [preferred_minimum_entry]
        minimum_feedback["omitted_candidate_feedback_entry_count"] = (
            int(minimum_feedback["omitted_candidate_feedback_entry_count"])
            + omitted_candidates
        )
        minimum_feedback["feedback_entries_omitted_for_capacity"] = len(removed)
        minimum_feedback["projection_level"] = "minimum_core_single_entry"
        _attach_stair_capacity_receipt(
            minimum,
            original=original,
            projection_id="stair_climbing.minimum_core_single_entry.v2",
            omitted=(*base_omissions, "one_feedback_entry_for_capacity"),
        )
        measured_bytes = canonical_prompt_json_bytes(minimum)
    if measured_bytes > target_bytes:
        raise ValueError(
            "Stair minimum-core prompt sections exceed reserved task-section "
            f"capacity: {measured_bytes} > {target_bytes} bytes"
        )
    return minimum


def _attach_stair_capacity_receipt(
    minimum: dict[str, Any],
    *,
    original: Mapping[str, Any],
    projection_id: str,
    omitted: Sequence[str],
) -> None:
    feedback = minimum.get("public_feedback_summary")
    if not isinstance(feedback, dict):
        raise ValueError("Stair minimum-core feedback is missing")
    feedback.pop("capacity_projection_receipt", None)
    content_without_receipt = copy.deepcopy(minimum)
    feedback["capacity_projection_receipt"] = build_prompt_capacity_receipt(
        original=original,
        projected_content=content_without_receipt,
        projection_id=projection_id,
        omitted=omitted,
    )


def _minimum_core_reward_inputs_contract(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Drop repetitive field prose while preserving the executable reward ABI."""

    result = copy.deepcopy(dict(value))
    fields = result.get("fields")
    if isinstance(fields, list):
        result["fields"] = {
            "schema_version": (
                "true_rl_agent.task.stair_climbing.reward_input_field_table.v1"
            ),
            "row_format": (
                "name|dtype|trailing_shape_csv|unit; empty shape means scalar"
            ),
            "rows": [
                "|".join(
                    (
                        str(item["name"]),
                        str(item["dtype"]),
                        ",".join(
                            str(dimension)
                            for dimension in item["trailing_shape"]
                        ),
                        str(item["unit"]),
                    )
                )
                for item in fields
                if isinstance(item, Mapping)
                and all(
                    key in item
                    for key in ("name", "dtype", "trailing_shape", "unit")
                )
                and isinstance(item["trailing_shape"], list | tuple)
            ],
        }
    result["capacity_projection"] = (
        "field_descriptions_omitted_fields_tabulated"
    )
    return result


def _minimum_core_task_section(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove evaluation identity fields duplicated by public_eval_contract."""

    result = copy.deepcopy(dict(value))
    evaluation = _mapping(result.get("fixed_eval_protocol_summary"))
    if evaluation:
        success = _mapping(evaluation.get("success_criteria"))
        result["fixed_eval_protocol_summary"] = {
            key: evaluation[key]
            for key in (
                "strata",
                "cells_per_stratum",
                "episodes_per_seed_per_cell",
                "public_seed_count",
                "public_accept_seed_count",
                "sealed_validation_seed_count",
                "seed_values_exposed",
                "statistics_unit",
            )
            if key in evaluation
        }
        # These are the task's success rules, not redundant identity metadata.
        result["fixed_eval_protocol_summary"]["success_criteria"] = dict(success)
    return result


def _minimum_core_selection_guidance(value: Any) -> list[str]:
    """Preserve the task-owned, mode-aware guidance as the single source of truth."""

    if not isinstance(value, list | tuple):
        return []
    return [item for item in value if isinstance(item, str)]


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
    raw_feedback = _mapping(entry.get("feedback"))
    feedback = _normalize_supported_prompt_feedback(raw_feedback)
    if (
        not feedback
        and raw_feedback
        and "schema_version" not in raw_feedback
        and "source_split" not in raw_feedback
        and set(raw_feedback).issubset(_COMPACT_FEEDBACK_KEYS)
    ):
        # The required-entry projection intentionally removes schema metadata
        # to save bytes.  A later minimum-core projection in the same capacity
        # pass must recognize that exact task-owned core instead of erasing it.
        feedback = dict(raw_feedback)
    generalization = _mapping(feedback.get("generalization_summary"))
    stratum_diagnostics = _compact_unified_stratum_diagnostics(
        _mapping(generalization.get("stratum_diagnostics"))
    )
    compact_feedback = {
        "climb_completion_summary": _mapping(
            feedback.get("climb_completion_summary")
        ),
        "stable_success_summary": _mapping(
            feedback.get("stable_success_summary")
        ),
        "dominant_failure": _mapping(feedback.get("dominant_failure")),
        "generalization_summary": {
            key: generalization[key]
            for key in (
                "public_dev_objective_score",
                "j_id",
                "j_ood",
                "named_guard_scores",
                "family_scores",
                "qualified_ood_families",
                "objective_family_mask_hash",
                "success_count_by_cell",
                "episodes_per_cell",
            )
            if key in generalization
        },
        "progress_summary": {
            "mean_max_progress_m_by_tread": _mapping(
                _mapping(feedback.get("progress_summary")).get(
                    "mean_max_progress_m_by_tread"
                )
            ),
            "mean_progress_fraction": _mapping(
                feedback.get("progress_summary")
            ).get("mean_progress_fraction"),
            "mean_progress_fraction_by_cell": _mapping(
                _mapping(feedback.get("progress_summary")).get(
                    "mean_progress_fraction_by_cell"
                )
            ),
        },
    }
    if stratum_diagnostics:
        compact_feedback["generalization_summary"][
            "stratum_diagnostics"
        ] = stratum_diagnostics
        compact_feedback["progress_summary"].pop(
            "mean_progress_fraction_by_cell",
            None,
        )
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
    stratum_diagnostics = _compact_unified_stratum_diagnostics(
        _mapping(generalization.get("stratum_diagnostics"))
    )
    if stratum_diagnostics:
        # The stratum projection is the actionable unified-task signal.  Per-cell
        # and factor arrays repeat the same membership at much higher byte cost.
        core["generalization_summary"].pop("success_rate_by_cell", None)
        core["generalization_summary"].pop("factor_diagnostics", None)
        core["generalization_summary"]["stratum_diagnostics"] = (
            stratum_diagnostics
        )
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
    progress_by_cell = _mapping(progress.get("mean_progress_fraction_by_cell"))
    if progress_by_cell and not stratum_diagnostics:
        compact_progress["mean_progress_fraction_by_cell"] = {
            key: round(float(value), 8)
            for key, value in progress_by_cell.items()
        }
    mean_progress_fraction = progress.get("mean_progress_fraction")
    if isinstance(mean_progress_fraction, int | float) and not isinstance(
        mean_progress_fraction,
        bool,
    ):
        compact_progress["mean_progress_fraction"] = round(
            float(mean_progress_fraction),
            8,
        )
    core["progress_summary"] = compact_progress
    for key in ("primary_failure_breakdown", "diagnostic_flag_rates"):
        core[key] = _without_numeric_zero_values(_mapping(core.get(key)))
    completion_summary = _mapping(core.get("climb_completion_summary"))
    stable_summary = _mapping(core.get("stable_success_summary"))
    if (
        "first_valid_top_arrival_rate" in completion_summary
        or "valid_top_arrival_rate" in stable_summary
    ):
        core["diagnostic_flag_rates"].pop(
            "first_valid_top_arrival_achieved",
            None,
        )
    core = {key: value for key, value in core.items() if value not in ({}, [])}
    return {**dict(entry), "feedback": core}


def _project_feedback_entry_for_prompt(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove fixed suite membership repetition without changing measured values."""

    projected = copy.deepcopy(dict(entry))
    feedback = _mapping(projected.get("feedback"))
    if not feedback:
        return projected
    projected_feedback = dict(feedback)
    completion_summary = _mapping(
        projected_feedback.get("climb_completion_summary")
    )
    derived_diagnostic_names = (
        tuple(_FIRST_ARRIVAL_DERIVED_DIAGNOSTIC_RATES)
        if completion_summary.get("completion_mode")
        == FIRST_VALID_TOP_ARRIVAL_COMPLETION_MODE
        else ()
    )
    diagnostic_rates = dict(
        _mapping(projected_feedback.get("diagnostic_flag_rates"))
    )
    for name in derived_diagnostic_names:
        diagnostic_rates.pop(name, None)
    if diagnostic_rates or "diagnostic_flag_rates" in projected_feedback:
        projected_feedback["diagnostic_flag_rates"] = diagnostic_rates
    generalization = dict(
        _mapping(projected_feedback.get("generalization_summary"))
    )
    factor_diagnostics = _mapping(generalization.get("factor_diagnostics"))
    if factor_diagnostics:
        generalization["factor_diagnostics"] = (
            _compact_unified_factor_diagnostics(factor_diagnostics)
        )
    stratum_diagnostics = _mapping(generalization.get("stratum_diagnostics"))
    if stratum_diagnostics:
        projected_strata = copy.deepcopy(dict(stratum_diagnostics))
        for raw_item in projected_strata.values():
            if not isinstance(raw_item, dict):
                continue
            rates = dict(_mapping(raw_item.get("diagnostic_flag_rates")))
            for name in derived_diagnostic_names:
                rates.pop(name, None)
            raw_item["diagnostic_flag_rates"] = rates
        generalization["stratum_diagnostics"] = (
            _compact_unified_stratum_diagnostics(
                projected_strata,
                derived_diagnostic_names=derived_diagnostic_names,
            )
        )
    if generalization:
        projected_feedback["generalization_summary"] = generalization
    projected["feedback"] = projected_feedback
    return projected


def _share_unified_feedback_tables(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Declare common identity and diagnostic columns once per provider frame."""
    summaries = [entry["feedback"]["generalization_summary"] for entry in entries
                 if _mapping(entry.get("feedback")).get("generalization_summary")]
    identity: dict[str, Any] = {}
    for summary in summaries:
        for key in ("objective_family_mask_hash", "ordered_suite_hash", "split_cohort_hash"):
            if key in summary:
                value = summary.pop(key)
                if key in identity and identity[key] != value:
                    raise ValueError("public feedback entries use different evaluation identities")
                identity[key] = value
    definitions = {}
    for name in ("factor_diagnostics", "stratum_diagnostics"):
        tables = [summary[name] for summary in summaries if name in summary]
        if not tables:
            continue
        definition = {key: value for key, value in tables[0].items() if key != "rows"}
        # Only nonzero failure/diagnostic columns are emitted. Align their union
        # before sharing the header, otherwise two rows could label rates wrong.
        for column_key, column in (("primary_failure_rate_columns", "primary_failure_rates"),
                              ("diagnostic_flag_rate_columns", "diagnostic_flag_rates")):
            if column_key not in definition:
                continue
            columns = sorted({value for table in tables for value in table[column_key]})
            position = definition["columns"].index(column)
            for table in tables:
                for row in table["rows"]:
                    values = dict(zip(table[column_key], row[position], strict=True))
                    row[position] = [values.get(key, 0.0) for key in columns]
            definition[column_key] = columns
        definitions[name] = definition
        for table in tables:
            rows, schema = table["rows"], table["schema_version"]
            table.clear()
            table.update(schema_version=schema, rows=rows)
    return {
        "detail_policy": "current best and latest evaluated: detailed; older entries: scores and main failures",
        "shared_evaluation_identity": identity,
        "diagnostic_table_definitions": definitions,
    }


def _compact_unified_factor_diagnostics(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep factor aggregates while omitting repeated fixed-suite cell IDs."""

    if value.get("schema_version") == _FACTOR_FEEDBACK_TABLE_SCHEMA_VERSION:
        return copy.deepcopy(dict(value))

    rows: list[list[Any]] = []
    for raw_factor, raw_levels in value.items():
        if not isinstance(raw_levels, Mapping):
            raise ValueError("Stair factor diagnostics levels must be an object")
        for raw_level, raw_item in raw_levels.items():
            if not isinstance(raw_item, Mapping):
                raise ValueError("Stair factor diagnostic row must be an object")
            cell_count = raw_item.get("cell_count")
            success_rate = _finite_score(raw_item.get("success_rate"))
            mean_progress = _finite_score(
                raw_item.get("mean_progress_fraction")
            )
            if (
                isinstance(cell_count, bool)
                or not isinstance(cell_count, int)
                or cell_count <= 0
                or success_rate is None
                or mean_progress is None
            ):
                raise ValueError("Stair factor diagnostic row is incomplete")
            rows.append(
                [
                    str(raw_factor),
                    str(raw_level),
                    cell_count,
                    success_rate,
                    mean_progress,
                ]
            )
    if not rows:
        return {}
    return {
        "schema_version": _FACTOR_FEEDBACK_TABLE_SCHEMA_VERSION,
        "columns": [
            "factor_name",
            "factor_level",
            "cell_count",
            "success_rate",
            "mean_progress_fraction",
        ],
        "fixed_suite_cell_membership_omitted": True,
        "rows": rows,
    }


def _compact_unified_stratum_diagnostics(
    value: Mapping[str, Any],
    *,
    derived_diagnostic_names: Sequence[str] = (),
) -> dict[str, Any]:
    """Keep comparative per-stratum feedback as one bounded self-describing table."""

    schema_version = _STRATUM_FEEDBACK_TABLE_SCHEMA_VERSION
    if value.get("schema_version") == schema_version:
        result = copy.deepcopy(dict(value))
        if derived_diagnostic_names:
            result["derived_diagnostic_rate_columns_omitted"] = list(
                derived_diagnostic_names
            )
        return result

    normalized: list[
        tuple[
            str,
            dict[str, float | None],
            dict[str, float],
            dict[str, float],
        ]
    ] = []
    failure_names: set[str] = set()
    diagnostic_names: set[str] = set()
    for raw_name, raw_item in sorted(value.items(), key=lambda item: str(item[0])):
        if not isinstance(raw_item, Mapping):
            continue
        numbers: dict[str, float | None] = {}
        for key in ("score", "success_rate", "mean_progress_fraction"):
            raw_number = raw_item.get(key)
            numbers[key] = (
                float(raw_number)
                if isinstance(raw_number, int | float)
                and not isinstance(raw_number, bool)
                else None
            )
        failures = {
            str(name): float(rate)
            for name, rate in _mapping(
                raw_item.get("primary_failure_breakdown")
            ).items()
            if isinstance(rate, int | float)
            and not isinstance(rate, bool)
            and float(rate) != 0.0
        }
        failure_names.update(failures)
        diagnostics = {
            str(name): float(rate)
            for name, rate in _mapping(
                raw_item.get("diagnostic_flag_rates")
            ).items()
            if isinstance(rate, int | float)
            and not isinstance(rate, bool)
            and float(rate) != 0.0
        }
        diagnostic_names.update(diagnostics)
        if any(number is not None for number in numbers.values()) or failures or diagnostics:
            normalized.append(
                (
                    str(raw_name),
                    numbers,
                    failures,
                    diagnostics,
                )
            )
    if not normalized:
        return {}

    ordered_failures = sorted(failure_names)
    ordered_diagnostics = sorted(diagnostic_names)
    result = {
        "schema_version": schema_version,
        "columns": [
            "stratum_id",
            "score",
            "success_rate",
            "mean_progress_fraction",
            "primary_failure_rates",
            "diagnostic_flag_rates",
        ],
        "primary_failure_rate_columns": ordered_failures,
        "diagnostic_flag_rate_columns": ordered_diagnostics,
        "omitted_primary_failure_rates_are_zero": True,
        "omitted_diagnostic_rates_are_zero": True,
        "rows": [
            [
                name,
                numbers["score"],
                numbers["success_rate"],
                numbers["mean_progress_fraction"],
                [
                    failures.get(reason, 0.0)
                    for reason in ordered_failures
                ],
                [diagnostics.get(flag, 0.0) for flag in ordered_diagnostics],
            ]
            for (
                name,
                numbers,
                failures,
                diagnostics,
            ) in normalized
        ],
    }
    if derived_diagnostic_names:
        result["derived_diagnostic_rate_columns_omitted"] = list(
            derived_diagnostic_names
        )
    return result


def _normalize_supported_prompt_feedback(
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize supported historical feedback without rewriting task semantics."""

    schema_version = feedback.get("schema_version")
    if schema_version in {
        _COMPACT_FEEDBACK_SCHEMA_VERSION,
        _FIRST_ARRIVAL_FEEDBACK_SCHEMA_VERSION,
    }:
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
    historical_feedback["generalization_summary"] = {
        key: generalization[key] for key in (
            "public_dev_objective_score", "j_id", "j_ood", "named_guard_scores", "family_scores",
        ) if key in generalization
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
    elif "mean_progress_fraction" in progress:
        historical_feedback["progress_summary"] = {
            "mean_progress_fraction": progress["mean_progress_fraction"]
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


def _compact_latest_feedback_entry(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep bounded unified comparative evidence; compact legacy latest feedback."""

    generalization = _mapping(
        _mapping(entry.get("feedback")).get("generalization_summary")
    )
    stratum_diagnostics = _mapping(generalization.get("stratum_diagnostics"))
    if stratum_diagnostics.get("schema_version") == (
        _STRATUM_FEEDBACK_TABLE_SCHEMA_VERSION
    ):
        return dict(entry)
    return _compact_historical_comparison_entry(entry)


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


def _is_detailed_feedback_entry(entry: Mapping[str, Any]) -> bool:
    basis = entry.get("selection_basis")
    if not isinstance(basis, list | tuple):
        return False
    return bool(
        {str(item) for item in basis} & _DETAILED_FEEDBACK_SELECTION_LABELS
    )


def _project_unified_prompt_public_record(
    record: Mapping[str, Any],
    *,
    config: StairClimbingTaskConfig,
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ),
) -> dict[str, Any]:
    """Build the Stair provider view without copying acceptance measurements."""

    projected = copy.deepcopy(dict(record))
    raw_result = _mapping(projected.get("public_result"))
    feedback = _validated_unified_prompt_feedback(
        raw_result,
        config=config,
        objective_family_mask_provider=objective_family_mask_provider,
    )
    score = _prompt_feedback_objective_score(feedback)
    if raw_result.get("status") == "pass" and score is None:
        raise ValueError(
            "unified Stair evaluated record lacks public_dev prompt score"
        )
    if raw_result:
        projected["public_result"] = _project_unified_prompt_public_result(
            raw_result,
            feedback=feedback,
            score=score,
        )

    raw_generation = _mapping(projected.get("generation"))
    no_formal_generation_outcome = None
    if raw_generation and raw_generation.get("formal") is None:
        generation_outcome = raw_generation.get("generation_outcome")
        if isinstance(generation_outcome, str) and generation_outcome.strip():
            no_formal_generation_outcome = generation_outcome

    raw_decision = _mapping(projected.get("decision"))
    if raw_decision:
        projected["decision"] = _project_unified_prompt_decision(
            raw_decision,
            score=score,
            no_formal_generation_outcome=no_formal_generation_outcome,
        )

    if raw_generation:
        projected["generation"] = _project_unified_prompt_generation(
            raw_generation, config=config,
            objective_family_mask_provider=objective_family_mask_provider,
        )
    return projected


def _project_unified_prompt_public_context(
    public_context: Mapping[str, Any],
    *,
    config: StairClimbingTaskConfig,
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ),
) -> dict[str, Any]:
    projected = copy.deepcopy(dict(public_context))
    anchor = _mapping(projected.get("initial_anchor_public_evidence"))
    if not anchor:
        return projected
    result = _mapping(anchor.get("public_result"))
    feedback = _validated_unified_prompt_feedback(
        result,
        config=config,
        objective_family_mask_provider=objective_family_mask_provider,
    )
    score = _prompt_feedback_objective_score(feedback)
    if result.get("status") == "pass" and score is None:
        raise ValueError(
            "unified Stair initial anchor lacks public_dev prompt score"
        )
    projected["initial_anchor_public_evidence"] = {
        key: copy.deepcopy(anchor[key])
        for key in ("anchor_kind", "public_result_hash")
        if key in anchor
    }
    if result:
        projected["initial_anchor_public_evidence"]["public_result"] = (
            _project_unified_prompt_public_result(
                result,
                feedback=feedback,
                score=score,
            )
        )
    return projected


def _project_unified_prompt_public_result(
    result: Mapping[str, Any],
    *,
    feedback: Mapping[str, Any],
    score: float | None,
) -> dict[str, Any]:
    metrics = _mapping(result.get("metrics"))
    visible_metrics: dict[str, Any] = {}
    if feedback:
        visible_metrics["prompt_feedback"] = copy.deepcopy(dict(feedback))
    candidate_loaded = metrics.get("candidate_code_loaded_in_eval")
    if isinstance(candidate_loaded, bool):
        visible_metrics["candidate_code_loaded_in_eval"] = candidate_loaded
    projected: dict[str, Any] = {
        "status": result.get("status"),
        "metrics": visible_metrics,
    }
    if score is not None:
        projected["public_score"] = score
    return projected


def _project_unified_prompt_decision(
    decision: Mapping[str, Any],
    *,
    score: float | None,
    no_formal_generation_outcome: str | None = None,
) -> dict[str, Any]:
    accepted = decision.get("accepted")
    if accepted is True:
        reason = "runner_accepted_candidate"
    elif accepted is False:
        reason = (
            no_formal_generation_outcome
            or "runner_did_not_accept_candidate"
        )
    else:
        reason = "runner_decision_unavailable"
    projected: dict[str, Any] = {
        "outcome": decision.get("outcome"),
        "accepted": accepted,
        "reason": reason,
        "surface_classification": decision.get("surface_classification"),
    }
    gate_statuses = _mapping(decision.get("gate_statuses"))
    runtime_gate = gate_statuses.get("runtime_activation_gate")
    if isinstance(runtime_gate, str):
        projected["gate_statuses"] = {
            "runtime_activation_gate": runtime_gate,
        }
    if score is not None:
        projected.update(
            {
                "public_score": score,
                "comparison": {
                    "mode": "prompt_visible_public_dev_absolute_v1",
                    "score_source": UNIFIED_STAIR_PROMPT_OBJECTIVE_SCORE_SOURCE,
                    "candidate_score": score,
                },
            }
        )
    return projected


def _project_unified_prompt_generation(
    generation: Mapping[str, Any],
    *, config: StairClimbingTaskConfig,
    objective_family_mask_provider: Callable[[], Mapping[str, Any] | None] | None,
) -> dict[str, Any]:
    projected = {
        key: copy.deepcopy(generation[key])
        for key in (
            "schema_version",
            "generation_id",
            "generation_index",
            "generation_outcome",
            "selection",
        )
        if key in generation
    }
    if "formal" in generation:
        projected["formal"] = _project_unified_prompt_formal(
            generation.get("formal")
        )
    projector = StairExplorationProjector(
        config,
        objective_family_mask=(
            objective_family_mask_provider() if objective_family_mask_provider else None
        ),
    )
    reference = projector.public_dev_parent_reference(generation)
    if reference is not None:
        projected["public_dev_parent_reference"] = reference
    trials = generation.get("trials")
    if isinstance(trials, list | tuple):
        projected["trials"] = [
            _project_unified_prompt_trial(trial)
            for trial in trials
            if isinstance(trial, Mapping)
        ]
        for raw, trial in zip(trials, projected["trials"], strict=True):
            screen = _mapping(raw.get("selection_screen"))
            outcome: dict[str, Any] = {}
            assessment = projector.assess_trial(generation=generation, trial=raw, parent=None)
            summary = assessment.public_evidence_summary
            for field, source in (
                ("objective_delta", "objective"),
                ("guard_delta", "guard_floor"),
            ):
                delta = _finite_score(_mapping(summary.get(source)).get("delta"))
                if delta is not None:
                    outcome[field] = round(delta, 6)
            effect = _mapping(screen.get("runtime_effect_evidence")).get("status")
            if effect is not None:
                outcome["runtime_effect"] = effect
            trial["prompt_outcome"] = outcome
    return projected


def _project_unified_prompt_trial(trial: Mapping[str, Any]) -> dict[str, Any]:
    projected = {
        key: copy.deepcopy(trial[key])
        for key in (
            "schema_version",
            "lane_id",
            "trial_id",
            "trial_status",
            "authoring_status",
            "runtime_activation_summary",
            "candidate",
            "candidate_surface",
            "mechanism_operation_delta",
            "public_dev",
            "lane_intent",
        )
        if key in trial
    }
    screen = _mapping(trial.get("selection_screen"))
    reason = project_public_trial_reason(trial.get("status_reason"))
    if reason is not None:
        projected["status_reason"] = reason
    if isinstance(trial.get("train"), Mapping):
        projected["train"] = {"completed": True}
    runtime_effect = screen.get("runtime_effect_evidence")
    if isinstance(runtime_effect, Mapping):
        # Full screen vectors stay in the internal public history. The provider
        # only needs the task-owned runtime-effect result used by the selector.
        projected["selection_screen"] = {
            "runtime_effect_evidence": copy.deepcopy(dict(runtime_effect))
        }
    return projected


def _project_unified_prompt_formal(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    formal = _mapping(value)
    return {
        key: copy.deepcopy(formal[key])
        for key in (
            "schema_version",
            "selected_lane_id",
            "selected_trial_id",
            "decision_outcome",
            "accepted",
        )
        if key in formal
    }


def _validated_unified_prompt_feedback(
    public_result: Mapping[str, Any],
    *,
    config: StairClimbingTaskConfig,
    objective_family_mask_provider: (
        Callable[[], Mapping[str, Any] | None] | None
    ),
) -> Mapping[str, Any]:
    metrics = _mapping(public_result.get("metrics"))
    stored = _mapping(metrics.get("prompt_feedback"))
    public_dev = metrics.get("public_dev")
    if not isinstance(public_dev, Mapping):
        return stored
    mask = (
        objective_family_mask_provider()
        if objective_family_mask_provider is not None
        else None
    )
    if mask is None:
        raise ValueError("unified Stair prompt feedback lacks Genesis family mask")
    expected = build_unified_stair_prompt_feedback(
        suite=config.ordered_evaluation_suite,
        semantics=config.evaluation_semantics,
        summary=public_dev,
        objective_family_mask=mask,
    )
    # Prompt feedback is a derived view, not a second scientific authority.
    # Old Genesis/ledger snapshots can carry earlier projections of this data.
    return expected


def _prompt_feedback_objective_score(
    feedback: Mapping[str, Any],
) -> float | None:
    generalization = _mapping(feedback.get("generalization_summary"))
    return _finite_score(generalization.get("public_dev_objective_score"))


def _feedback_trial_records(
    public_records: Sequence[Mapping[str, Any]], *, config: StairClimbingTaskConfig,
    objective_family_mask_provider: Callable[[], Mapping[str, Any] | None] | None,
) -> list[Mapping[str, Any]]:
    """Use each lane's public-dev and source identity, including no-selection."""

    if not config.is_unified_protocol:
        return list(public_records)
    records: list[Mapping[str, Any]] = []
    for record in public_records:
        generation = _mapping(record.get("generation"))
        if not generation:
            records.append(record)
            continue
        for trial in generation.get("trials", ()):
            summary = _mapping(_mapping(trial.get("public_dev")).get("public_summary"))
            if not summary:
                continue
            feedback = build_unified_stair_prompt_feedback(
                suite=config.ordered_evaluation_suite, semantics=config.evaluation_semantics,
                summary=summary, objective_family_mask=objective_family_mask_provider()
                if objective_family_mask_provider else None,
            )
            candidate = _mapping(trial.get("candidate"))
            records.append({
                "round_id": trial["trial_id"], "candidate": candidate,
                "train_result": {"program_source_checksum": candidate.get("program_source_checksum")},
                "decision": {"accepted": trial["trial_status"] == "accepted"},
                "public_result": {"status": "pass", "metrics": {"prompt_feedback": feedback}},
            })
    return records


def _candidate_feedback_entries(
    public_records: Sequence[Mapping[str, Any]],
    *,
    use_public_dev_objective: bool = False,
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
        objective_score = (
            _prompt_feedback_objective_score(feedback)
            if use_public_dev_objective
            else None
        )
        guard_source = comparison.get("guard_score_source")
        has_guard_score = isinstance(guard_source, str) and bool(guard_source)
        if objective_score is None:
            objective_score = _finite_score(comparison.get("candidate_score"))
            if objective_score is None and not has_guard_score:
                objective_score = bound_score
            guard_score = bound_score if has_guard_score else None
        else:
            guard_score = None
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
    *, latest_nonaccepted_trial_id: str | None = None,
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
    latest_nonaccepted_index = next((index for index, entry in enumerate(entries)
        if entry.get("round_id") == latest_nonaccepted_trial_id), None)
    for special_index in (best_nonaccepted_index, current_accepted_index, latest_nonaccepted_index):
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
        if index == latest_nonaccepted_index:
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
    *,
    use_public_dev_objective: bool = False,
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
    objective_score = (
        _prompt_feedback_objective_score(feedback)
        if use_public_dev_objective
        else None
    )
    has_guard_score = isinstance(
        acceptance.get("guard_score_source"),
        str,
    ) and bool(acceptance.get("guard_score_source"))
    bound_score = _finite_score(result.get("public_score"))
    if objective_score is None:
        objective_score = _finite_score(
            _mapping(acceptance.get("score_stats")).get("mean")
        )
        if objective_score is None and not has_guard_score:
            objective_score = bound_score
        guard_score = bound_score if has_guard_score else None
    else:
        guard_score = None
    entry = {
        "source_kind": "reference_anchor",
        "source_public_result_hash": result_hash,
        "objective_score": objective_score,
        "guard_score": guard_score,
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
    selected = latest_nonaccepted_trial(public_records, include_failed_training_sampling=True)
    if selected is None:
        return {"schema_version": "true_rl_agent.runtime_feedback_summary.stair.v2",
                "status": "not_available", "reason": "no_latest_nonaccepted_trial"}
    trial = selected[1]
    candidate = _mapping(trial.get("candidate"))
    runtime = _mapping(trial.get("runtime_activation_summary"))
    measured = RuntimeActivationEvidence.from_mapping(runtime).measurement_summary()
    components = compact_component_activation_records(runtime.get("component_activation_records"),
                                                     max_components=_MAX_RUNTIME_FEEDBACK_COMPONENTS)
    return {
        "schema_version": "true_rl_agent.runtime_feedback_summary.stair.v2",
        "status": measured["status"],
        "reason": "same_trial_training_receipt" if measured["status"] == "available" else "selected_trial_has_no_runtime_observations",
        "source_round_id": trial.get("trial_id", trial.get("round_id")),
        "source_candidate_id": candidate.get("candidate_id"),
        "program_source_checksum": candidate.get("program_source_checksum"),
        "source_trial_status": trial.get("trial_status"),
        "latest_runtime_activation": {**measured,
                                      "capability_activation_records": runtime.get("capability_activation_records", []),
                                      "component_count": len(runtime.get("component_activation_records", [])),
                                      "omitted_component_count": max(0, len(runtime.get("component_activation_records", [])) - len(components)),
                                      "components": components},
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
    legacy_fall = _mapping(signals.get("fall"))
    result = {
        **dict(success),
        "maximum_abs_roll_or_pitch_rad": orientation.get("threshold"),
    }
    if success.get("requires_kinematic_base_clearance_above_top") is True:
        result["minimum_kinematic_base_clearance_above_top_m"] = success.get(
            "minimum_kinematic_base_clearance_above_top_m",
            legacy_fall.get("threshold"),
        )
    if stability:
        result["top_platform_stability_requirement"] = dict(stability)
    return result


__all__ = ["StairClimbingPromptAdapter"]
