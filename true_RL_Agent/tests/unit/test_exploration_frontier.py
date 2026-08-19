from __future__ import annotations

import hashlib
import unittest
from dataclasses import dataclass
from typing import Any, Mapping

from true_rl_agent.agent_core.exploration import (
    EXPLORATION_LANE_INTENT_SCHEMA_VERSION,
    EXPLORATION_PROMPT_MAX_BYTES,
    GenerationExplorationPlan,
    RevisitFrontier,
    TaskExplorationAssessment,
    build_exploration_index,
    build_exploration_prompt_projection,
    build_generation_exploration_plan,
    build_revisit_frontier,
    freeze_exploration_lane_intents,
)
from true_rl_agent.agent_core.experiment_memory import (
    EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES,
    build_experiment_memory,
    recompact_experiment_memory,
)
from true_rl_agent.agent_core.exploration_projection import (
    compact_exploration_prompt_projection,
)
from true_rl_agent.agent_core.generation_contract import (
    MAX_GENERATION_LANE_INTENT_BYTES,
    require_generation_lane_intent,
)
from true_rl_agent.agent_core.generation_record import (
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
)
from true_rl_agent.candidates.authoring_policy import (
    single_component_verifier_equivalence_key,
    verifier_equivalence_exclusion_set_hash,
)
from true_rl_agent.evaluation.acceptance_replay import ProjectedPublicHistory
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes
from tests.fakes.fake_components import (
    FAKE_ACCEPTANCE_PROTOCOL,
    FAKE_SESSION_GENESIS_HASH,
    fake_bootstrap_current_best,
)


@dataclass(frozen=True)
class _Projector:
    task_id: str = "fake_push_recovery"

    def contract(self) -> Mapping[str, Any]:
        return {
            "schema_version": "tests.exploration_projector.v1",
            "task_id": self.task_id,
            "acceptance_protocol_hash": FAKE_ACCEPTANCE_PROTOCOL.protocol_hash,
            "acceptance_authority": False,
        }

    def assess_trial(
        self,
        *,
        generation: Mapping[str, Any],
        trial: Mapping[str, Any],
        parent: CurrentBestEvalStats,
    ) -> TaskExplorationAssessment:
        del generation, parent
        candidate_id = trial.get("candidate", {}).get("candidate_id", "")
        if "negative" in candidate_id:
            effect, value = "negative", "low"
        elif "mixed" in candidate_id:
            effect, value = "mixed", "medium"
        else:
            effect, value = "strong_positive", "high"
        rank_delta = (
            0.30
            if "strong" in candidate_id
            else 0.01
            if "weak" in candidate_id
            else 0.10
        )
        return TaskExplorationAssessment(
            observed_effect=effect,
            value_label=value,
            public_evidence_summary={
                "candidate_objective": 0.6,
                "parent_objective": 0.5,
                "delta": 0.1,
            },
            reason_codes=(f"test_{effect}",),
            revisit_rank_key=(round(rank_delta * 1_000_000),),
        )


_PROJECTOR = _Projector()
_PROJECTOR_CONTRACT_HASH = canonical_json_hash(_PROJECTOR.contract())


@dataclass(frozen=True)
class _LargeSummaryProjector(_Projector):
    def contract(self) -> Mapping[str, Any]:
        return {
            **super().contract(),
            "schema_version": "tests.large_exploration_projector.v1",
        }

    def assess_trial(
        self,
        *,
        generation: Mapping[str, Any],
        trial: Mapping[str, Any],
        parent: CurrentBestEvalStats,
    ) -> TaskExplorationAssessment:
        del generation, trial, parent
        return TaskExplorationAssessment(
            observed_effect="strong_positive",
            value_label="high",
            public_evidence_summary={"bounded_task_summary": "s" * 3500},
            reason_codes=("test_strong_positive",),
            revisit_rank_key=(100_000,),
        )


_LARGE_SUMMARY_PROJECTOR = _LargeSummaryProjector()
_LARGE_SUMMARY_PROJECTOR_CONTRACT_HASH = canonical_json_hash(
    _LARGE_SUMMARY_PROJECTOR.contract()
)


class ExplorationFrontierTests(unittest.TestCase):
    def test_verifier_equivalence_ignores_name_source_and_parent_parameters(self) -> None:
        first = _delta("equivalent-base")
        renamed = _delta("equivalent-renamed")
        renamed["after_component"]["mechanism_id"] = "RewardTerm:renamed_term"
        renamed["after_component"]["parameter_normalized_structure_hash"] = (
            first["after_component"]["parameter_normalized_structure_hash"]
        )
        self.assertEqual(
            single_component_verifier_equivalence_key(first),
            single_component_verifier_equivalence_key(renamed),
        )

        axis_changed = _delta("axis-changed")
        axis_changed["after_component"]["mechanism_id"] = "RewardTerm:renamed_term"
        self.assertNotEqual(
            single_component_verifier_equivalence_key(first),
            single_component_verifier_equivalence_key(axis_changed),
        )

    def test_good_loser_is_revisited_and_accepted_is_incorporated(self) -> None:
        parent = fake_bootstrap_current_best()
        generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-good-loser", "qualified_not_selected", seed="1"),
            trial_b=_trial("trial-accepted", "accepted", seed="2"),
            selected_trial_id="trial-accepted",
        )
        history = _history(parent, [generation])
        first = build_exploration_index(
            history,
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        second = build_exploration_index(
            history,
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        self.assertEqual(first.index_hash, second.index_hash)
        with self.assertRaisesRegex(ValueError, "not generation-session bound"):
            build_exploration_index(
                history,
                task_projector=_PROJECTOR,
                expected_projector_contract_hash="f" * 64,
            )
        self.assertEqual(len(first.cards), 2)
        loser, accepted = first.cards
        self.assertEqual(loser.revisit_disposition, "priority")
        self.assertEqual(accepted.revisit_disposition, "incorporated")
        self.assertEqual(accepted.evidence_tier, "formal")

        frontier = build_revisit_frontier(first)
        self.assertEqual(frontier.queued_card_hashes, (loser.card_hash,))
        lane_a, lane_b = freeze_exploration_lane_intents(first, frontier)
        self.assertEqual(lane_a["schema_version"], EXPLORATION_LANE_INTENT_SCHEMA_VERSION)
        self.assertEqual(lane_a["mode"], "revisit")
        self.assertEqual(lane_a["inspired_by_trial_id"], loser.trial_id)
        self.assertEqual(lane_b["mode"], "novel")
        projection = build_exploration_prompt_projection(first, frontier)
        self.assertLessEqual(
            canonical_prompt_json_bytes(projection),
            EXPLORATION_PROMPT_MAX_BYTES,
        )
        self.assertFalse(projection["acceptance_authority"])
        plan = build_generation_exploration_plan(
            history,
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        self.assertEqual(plan.index.index_hash, first.index_hash)
        self.assertEqual(plan.blocked_verifier_equivalence_keys, ())
        self.assertEqual(plan.lane_intents, (lane_a, lane_b))
        self.assertEqual(plan.prompt_projection, projection)
        impossible_deferred = {**projection, "deferred_frontier_card_count": 1}
        with self.assertRaisesRegex(ValueError, "omission accounting"):
            compact_exploration_prompt_projection(impossible_deferred)
        impossible_backlog = {
            **projection,
            "frontier_backlog_count": 0,
            "omitted_frontier_card_count": 0,
        }
        with self.assertRaisesRegex(ValueError, "omission accounting"):
            compact_exploration_prompt_projection(impossible_backlog)

    def test_legal_large_cards_are_omitted_from_prompt_without_losing_evidence(
        self,
    ) -> None:
        parent = fake_bootstrap_current_best()
        for label, fields in (
            (
                "more-than-prompt-field-contract",
                [f"field_{index:03d}_" + ("x" * 44) for index in range(80)],
            ),
            (
                "minimal-card-still-over-envelope",
                [f"field_{index:03d}_" + ("y" * 168) for index in range(32)],
            ),
            (
                "field-longer-than-prompt-contract",
                ["field_" + ("q" * 600)],
            ),
        ):
            with self.subTest(label=label):
                trial_a = _trial(
                    f"trial-{label}-loser",
                    "qualified_not_selected",
                    seed=label,
                )
                trial_a["mechanism_operation_delta"]["after_component"][
                    "input_fields"
                ] = fields
                generation = _generation(
                    parent,
                    generation_index=1,
                    trial_a=trial_a,
                    trial_b=_trial(
                        f"trial-{label}-failed",
                        "candidate_failed",
                        seed=f"{label}-failed",
                    ),
                    selected_trial_id=None,
                )
                index = build_exploration_index(
                    _history(parent, [generation]),
                    task_projector=_PROJECTOR,
                    expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
                )
                frontier = build_revisit_frontier(index)
                source = next(
                    card for card in index.cards if card.trial_id == trial_a["trial_id"]
                )
                self.assertEqual(source.revisit_disposition, "priority")
                projection = build_exploration_prompt_projection(index, frontier)
                self.assertEqual(projection["priority_or_eligible_cards"], [])
                self.assertEqual(projection["omitted_frontier_card_count"], 1)
                self.assertEqual(
                    compact_exploration_prompt_projection(projection),
                    projection,
                )
                self.assertLessEqual(
                    canonical_prompt_json_bytes(projection),
                    EXPLORATION_PROMPT_MAX_BYTES,
                )

    def test_large_evidence_card_uses_bounded_lane_preview(self) -> None:
        parent = fake_bootstrap_current_best()
        trial_a = _trial(
            "trial-large-summary-loser",
            "qualified_not_selected",
            seed="large-summary",
        )
        input_fields = [
            f"field_{index:03d}_" + ("z" * 44)
            for index in range(80)
        ]
        trial_a["mechanism_operation_delta"]["after_component"][
            "input_fields"
        ] = input_fields
        plan = build_generation_exploration_plan(
            _history(
                parent,
                [
                    _generation(
                        parent,
                        generation_index=1,
                        trial_a=trial_a,
                        trial_b=_trial(
                            "trial-large-summary-failed",
                            "candidate_failed",
                            seed="large-summary-failed",
                        ),
                        selected_trial_id=None,
                    )
                ],
            ),
            task_projector=_LARGE_SUMMARY_PROJECTOR,
            expected_projector_contract_hash=(
                _LARGE_SUMMARY_PROJECTOR_CONTRACT_HASH
            ),
        )
        lane = plan.lane_intents[0]
        hypothesis = lane["hypothesis"]
        self.assertEqual(lane["mode"], "revisit")
        self.assertEqual(
            hypothesis["verifier_equivalence_key"],
            plan.index.cards[0].verifier_equivalence_key,
        )
        self.assertEqual(hypothesis["public_input_field_count"], len(input_fields))
        self.assertGreater(hypothesis["omitted_public_input_field_count"], 0)
        self.assertTrue(hypothesis["public_evidence_summary_omitted"])
        self.assertIsNone(hypothesis["public_evidence_summary"])
        self.assertEqual(require_generation_lane_intent(lane), lane)
        self.assertLessEqual(
            len(canonical_json_bytes(lane)),
            MAX_GENERATION_LANE_INTENT_BYTES,
        )

    def test_both_good_no_selection_are_retained_and_backlog_is_explicit(self) -> None:
        parent = fake_bootstrap_current_best()
        generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-good-a", "qualified_not_selected", seed="3"),
            trial_b=_trial("trial-good-b", "qualified_not_selected", seed="4"),
            selected_trial_id=None,
        )
        index = build_exploration_index(
            _history(parent, [generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)
        self.assertEqual(frontier.backlog_count, 2)
        self.assertEqual(
            {card.selection_status for card in index.cards},
            {"inconclusive"},
        )
        lane_a, lane_b = freeze_exploration_lane_intents(index, frontier)
        self.assertEqual([lane_a["mode"], lane_b["mode"]], ["revisit", "novel"])

    def test_frontier_and_lane_intents_must_be_canonical_index_projections(self) -> None:
        parent = fake_bootstrap_current_best()
        history = _history(
            parent,
            [
                _generation(
                    parent,
                    generation_index=1,
                    trial_a=_trial(
                        "trial-canonical-loser",
                        "qualified_not_selected",
                        seed="canonical-loser",
                    ),
                    trial_b=_trial(
                        "trial-canonical-failure",
                        "candidate_failed",
                        seed="canonical-failure",
                    ),
                    selected_trial_id=None,
                )
            ],
        )
        plan = build_generation_exploration_plan(
            history,
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        forged = RevisitFrontier(
            index_hash=plan.index.index_hash,
            current_head_hash=plan.index.current_head_hash,
            queued_card_hashes=(),
            deferred_cards=(),
            ineligible_cards=(),
            backlog_count=0,
        )
        with self.assertRaisesRegex(ValueError, "not the canonical"):
            freeze_exploration_lane_intents(plan.index, forged)
        with self.assertRaisesRegex(ValueError, "not the canonical"):
            build_exploration_prompt_projection(plan.index, forged)
        with self.assertRaisesRegex(ValueError, "not the canonical"):
            GenerationExplorationPlan(
                index=plan.index,
                frontier=forged,
                blocked_verifier_equivalence_keys=(
                    plan.blocked_verifier_equivalence_keys
                ),
                lane_intents=plan.lane_intents,
                prompt_projection=plan.prompt_projection,
            )

        forged_intents = (
            {
                **plan.lane_intents[0],
                "mode": "novel",
                "inspired_by_trial_id": None,
                "inspired_by_card_hash": None,
                "hypothesis": None,
            },
            plan.lane_intents[1],
        )
        with self.assertRaisesRegex(ValueError, "lane intents mismatch"):
            GenerationExplorationPlan(
                index=plan.index,
                frontier=plan.frontier,
                blocked_verifier_equivalence_keys=(
                    plan.blocked_verifier_equivalence_keys
                ),
                lane_intents=forged_intents,
                prompt_projection=plan.prompt_projection,
            )

    def test_loser_is_ineligible_after_equivalent_mechanism_is_incorporated(self) -> None:
        parent = fake_bootstrap_current_best()
        source_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-source", "qualified_not_selected", seed="shared"),
            trial_b=_trial("trial-other-head", "accepted", seed="other"),
            selected_trial_id="trial-other-head",
        )
        incorporated_generation = _generation(
            parent,
            generation_index=2,
            trial_a=_trial("trial-equivalent", "accepted", seed="shared"),
            trial_b=_trial("trial-other-loser", "qualified_not_selected", seed="new"),
            selected_trial_id="trial-equivalent",
        )
        index = build_exploration_index(
            _history(parent, [source_generation, incorporated_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        source = next(card for card in index.cards if card.trial_id == "trial-source")
        frontier = build_revisit_frontier(index)

        self.assertNotIn(source.card_hash, frontier.queued_card_hashes)
        self.assertIn(
            {"card_hash": source.card_hash, "reason": "incorporated_equivalence"},
            frontier.ineligible_cards,
        )

    def test_negative_suppresses_its_cross_parent_verifier_equivalence(self) -> None:
        parent = fake_bootstrap_current_best()
        generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-negative", "qualified_not_selected", seed="5"),
            trial_b=_trial("trial-good", "accepted", seed="6"),
            selected_trial_id="trial-good",
        )
        index = build_exploration_index(
            _history(parent, [generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        negative = index.cards[0]
        self.assertEqual(negative.revisit_disposition, "suppress_equivalent")
        plan = build_generation_exploration_plan(
            _history(parent, [generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        self.assertEqual(
            plan.blocked_verifier_equivalence_keys,
            (negative.verifier_equivalence_key,),
        )
        self.assertEqual(
            plan.lane_intents[0]["verifier_equivalence_exclusion_set_hash"],
            verifier_equivalence_exclusion_set_hash(
                plan.blocked_verifier_equivalence_keys
            ),
        )
        self.assertEqual(plan.frontier.queued_card_hashes, ())

    def test_candidate_failure_cannot_become_negative_mechanism_evidence(self) -> None:
        parent = fake_bootstrap_current_best()
        generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-negative", "candidate_failed", seed="d"),
            trial_b=_trial("trial-head", "accepted", seed="e"),
            selected_trial_id="trial-head",
        )
        index = build_exploration_index(
            _history(parent, [generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        failed = next(card for card in index.cards if card.trial_id == "trial-negative")
        self.assertEqual(failed.observed_effect, "unknown")
        self.assertEqual(failed.value_label, "unknown")
        self.assertEqual(failed.revisit_disposition, "not_evaluable")
        self.assertNotIn(
            failed.verifier_equivalence_key,
            build_generation_exploration_plan(
                _history(parent, [generation]),
                task_projector=_PROJECTOR,
                expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
            ).blocked_verifier_equivalence_keys,
        )

    def test_full_negative_exclusion_set_is_not_truncated_with_prompt_frontier(self) -> None:
        parent = fake_bootstrap_current_best()
        generations = [
            _generation(
                parent,
                generation_index=index,
                trial_a=_trial(
                    f"trial-negative-{index}-loser",
                    "qualified_not_selected",
                    seed=f"negative-{index}",
                ),
                trial_b=_trial(
                    f"trial-head-{index}",
                    "accepted",
                    seed=f"head-{index}",
                ),
                selected_trial_id=f"trial-head-{index}",
            )
            for index in range(1, 7)
        ]
        history = _history(parent, generations)
        plan = build_generation_exploration_plan(
            history,
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )

        self.assertEqual(len(plan.blocked_verifier_equivalence_keys), 6)
        self.assertEqual(
            plan.prompt_projection["blocked_verifier_equivalence_count"],
            6,
        )
        self.assertEqual(
            len(plan.prompt_projection["verifier_equivalent_avoidance_cards"]),
            2,
        )
        self.assertEqual(
            plan.prompt_projection[
                "omitted_verifier_equivalent_avoidance_count"
            ],
            4,
        )

    def test_frontier_window_is_bounded_and_reports_full_backlog(self) -> None:
        parent = fake_bootstrap_current_best()
        generations = [
            _generation(
                parent,
                generation_index=index,
                trial_a=_trial(
                    f"trial-good-{index}-loser",
                    "qualified_not_selected",
                    seed=seed,
                ),
                trial_b=_trial(
                    f"trial-paired-failure-{index}",
                    "candidate_failed",
                    seed=head_seed,
                ),
                selected_trial_id=None,
            )
            for index, (seed, head_seed) in enumerate(
                zip("123456", "789abc", strict=True),
                start=1,
            )
        ]
        index = build_exploration_index(
            _history(parent, generations),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)
        self.assertEqual(len(frontier.queued_card_hashes), 4)
        self.assertEqual(frontier.backlog_count, 6)
        self.assertEqual(
            [item["queue_position"] for item in frontier.deferred_cards],
            [5, 6],
        )
        self.assertTrue(
            all(
                item["reason"] == "frontier_capacity"
                for item in frontier.deferred_cards
            )
        )
        projection = build_exploration_prompt_projection(index, frontier)
        self.assertEqual(projection["frontier_backlog_count"], 6)
        self.assertEqual(projection["omitted_frontier_card_count"], 4)

    def test_priority_order_uses_task_rank_then_recent_generation(self) -> None:
        parent = fake_bootstrap_current_best()
        generations = [
            _generation(
                parent,
                generation_index=1,
                trial_a=_trial(
                    "trial-weak-loser",
                    "qualified_not_selected",
                    seed="rank-weak",
                ),
                trial_b=_trial(
                    "trial-paired-failure-1",
                    "candidate_failed",
                    seed="rank-failure-1",
                ),
                selected_trial_id=None,
            ),
            _generation(
                parent,
                generation_index=2,
                trial_a=_trial(
                    "trial-strong-loser",
                    "qualified_not_selected",
                    seed="rank-strong",
                ),
                trial_b=_trial(
                    "trial-paired-failure-2",
                    "candidate_failed",
                    seed="rank-failure-2",
                ),
                selected_trial_id=None,
            ),
        ]
        index = build_exploration_index(
            _history(parent, generations),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        by_hash = {card.card_hash: card for card in index.cards}
        frontier = build_revisit_frontier(index)

        self.assertEqual(
            by_hash[frontier.queued_card_hashes[0]].trial_id,
            "trial-strong-loser",
        )

    def test_failed_revisit_rotates_then_exhausts_after_two_services(self) -> None:
        parent = fake_bootstrap_current_best()
        source_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial(
                "trial-strong-source",
                "qualified_not_selected",
                seed="service-strong",
            ),
            trial_b=_trial(
                "trial-weak-b",
                "qualified_not_selected",
                seed="service-weak",
            ),
            selected_trial_id=None,
        )
        first = build_exploration_index(
            _history(parent, [source_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        source = next(
            card for card in first.cards if card.trial_id == "trial-strong-source"
        )
        intent = freeze_exploration_lane_intents(
            first,
            build_revisit_frontier(first),
        )[0]
        failed_revisit = _generation(
            parent,
            generation_index=2,
            trial_a=_trial(
                "trial-first-revisit",
                "verified_not_run",
                seed="service-failed-1",
                lane_intent=intent,
            ),
            trial_b=_trial(
                "trial-paired-failure-1",
                "candidate_failed",
                seed="service-paired-1",
            ),
            selected_trial_id=None,
        )
        after_one = build_exploration_index(
            _history(parent, [source_generation, failed_revisit]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        by_hash = {card.card_hash: card for card in after_one.cards}
        frontier = build_revisit_frontier(after_one)
        self.assertEqual(
            by_hash[frontier.queued_card_hashes[0]].trial_id,
            "trial-weak-b",
        )

        failed_before_candidate = _trial(
            "trial-second-revisit",
            "candidate_failed",
            seed="service-failed-2",
            lane_intent=intent,
        )
        failed_before_candidate["candidate"] = None
        failed_before_candidate["mechanism_operation_delta"] = None
        second_failed_revisit = _generation(
            parent,
            generation_index=3,
            trial_a=failed_before_candidate,
            trial_b=_trial(
                "trial-paired-failure-2",
                "candidate_failed",
                seed="service-paired-2",
            ),
            selected_trial_id=None,
        )
        after_two = build_exploration_index(
            _history(
                parent,
                [source_generation, failed_revisit, second_failed_revisit],
            ),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        exhausted = build_revisit_frontier(after_two)
        self.assertIn(
            {
                "card_hash": source.card_hash,
                "reason": "revisit_attempt_limit_reached",
            },
            exhausted.ineligible_cards,
        )

    def test_public_revisit_consumes_all_equivalent_source_cards(self) -> None:
        parent = fake_bootstrap_current_best()
        source_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial(
                "trial-equivalent-source-a",
                "qualified_not_selected",
                seed="same-equivalence",
            ),
            trial_b=_trial(
                "trial-equivalent-source-b",
                "qualified_not_selected",
                seed="same-equivalence",
            ),
            selected_trial_id=None,
        )
        initial = build_exploration_index(
            _history(parent, [source_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        sources = tuple(initial.cards)
        self.assertEqual(
            len({card.verifier_equivalence_key for card in sources}),
            1,
        )
        intent = freeze_exploration_lane_intents(
            initial,
            build_revisit_frontier(initial),
        )[0]
        public_revisit = _generation(
            parent,
            generation_index=2,
            trial_a=_trial(
                "trial-public-revisit",
                "qualified_not_selected",
                seed="public-revisit-result",
                lane_intent=intent,
            ),
            trial_b=_trial(
                "trial-public-revisit-pair-failure",
                "candidate_failed",
                seed="public-revisit-failure",
            ),
            selected_trial_id=None,
        )
        index = build_exploration_index(
            _history(parent, [source_generation, public_revisit]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)
        for source in sources:
            self.assertNotIn(source.card_hash, frontier.queued_card_hashes)
            self.assertIn(
                {
                    "card_hash": source.card_hash,
                    "reason": "already_revisited_with_public_evidence",
                },
                frontier.ineligible_cards,
            )

    def test_two_failed_services_exhaust_an_equivalence_class(self) -> None:
        parent = fake_bootstrap_current_best()
        source_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial(
                "trial-equivalent-failure-source-a",
                "qualified_not_selected",
                seed="failed-equivalence",
            ),
            trial_b=_trial(
                "trial-equivalent-failure-source-b",
                "qualified_not_selected",
                seed="failed-equivalence",
            ),
            selected_trial_id=None,
        )
        generations = [source_generation]
        initial = build_exploration_index(
            _history(parent, generations),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        sources = tuple(initial.cards)
        for generation_index in (2, 3):
            index = build_exploration_index(
                _history(parent, generations),
                task_projector=_PROJECTOR,
                expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
            )
            intent = freeze_exploration_lane_intents(
                index,
                build_revisit_frontier(index),
            )[0]
            failed = _trial(
                f"trial-class-failure-{generation_index}",
                "candidate_failed",
                seed=f"class-failure-{generation_index}",
                lane_intent=intent,
            )
            failed["candidate"] = None
            failed["mechanism_operation_delta"] = None
            generations.append(
                _generation(
                    parent,
                    generation_index=generation_index,
                    trial_a=failed,
                    trial_b=_trial(
                        f"trial-class-pair-failure-{generation_index}",
                        "candidate_failed",
                        seed=f"class-pair-failure-{generation_index}",
                    ),
                    selected_trial_id=None,
                )
            )
        exhausted = build_revisit_frontier(
            build_exploration_index(
                _history(parent, generations),
                task_projector=_PROJECTOR,
                expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
            )
        )
        for source in sources:
            self.assertNotIn(source.card_hash, exhausted.queued_card_hashes)
            self.assertIn(
                {
                    "card_hash": source.card_hash,
                    "reason": "revisit_attempt_limit_reached",
                },
                exhausted.ineligible_cards,
            )

    def test_revisit_expires_after_two_head_transitions(self) -> None:
        parent = fake_bootstrap_current_best()
        first_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial(
                "trial-source",
                "qualified_not_selected",
                seed="stale-source",
            ),
            trial_b=_trial("trial-head-1", "accepted", seed="stale-head-1"),
            selected_trial_id="trial-head-1",
        )
        first = build_exploration_index(
            _history(parent, [first_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        source = next(card for card in first.cards if card.trial_id == "trial-source")
        self.assertIn(source.card_hash, build_revisit_frontier(first).queued_card_hashes)

        second_generation = _generation(
            parent,
            generation_index=2,
            trial_a=_trial("trial-head-2", "accepted", seed="stale-head-2"),
            trial_b=_trial(
                "trial-paired-failure",
                "candidate_failed",
                seed="stale-failure",
            ),
            selected_trial_id="trial-head-2",
        )
        index = build_exploration_index(
            _history(parent, [first_generation, second_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)
        self.assertIn(
            {
                "card_hash": source.card_hash,
                "reason": "stale_after_head_transitions",
            },
            frontier.ineligible_cards,
        )

    def test_repeated_negative_input_signature_is_advisory_only(self) -> None:
        parent = fake_bootstrap_current_best()
        generations = [
            _generation(
                parent,
                generation_index=generation_index,
                trial_a=_trial(
                    f"trial-negative-{generation_index}",
                    "qualified_not_selected",
                    seed=f"advisory-negative-{generation_index}",
                ),
                trial_b=_trial(
                    f"trial-paired-failure-{generation_index}",
                    "candidate_failed",
                    seed=f"advisory-failure-{generation_index}",
                ),
                selected_trial_id=None,
            )
            for generation_index in range(1, 4)
        ]
        plan = build_generation_exploration_plan(
            _history(parent, generations),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        advisories = plan.prompt_projection[
            "repeated_negative_input_signature_advisories"
        ]
        self.assertEqual(len(advisories), 1)
        self.assertEqual(advisories[0]["negative_trial_count"], 3)
        self.assertEqual(
            advisories[0]["distinct_verifier_equivalence_count"],
            3,
        )
        self.assertFalse(advisories[0]["blocking"])
        self.assertEqual(len(plan.blocked_verifier_equivalence_keys), 3)
        self.assertEqual(
            plan.prompt_projection["blocked_verifier_equivalence_count"],
            3,
        )

    def test_medium_card_is_visible_but_does_not_consume_revisit_lane(self) -> None:
        parent = fake_bootstrap_current_best()
        generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial(
                "trial-mixed-a",
                "qualified_not_selected",
                seed="medium-a",
            ),
            trial_b=_trial(
                "trial-mixed-b",
                "qualified_not_selected",
                seed="medium-b",
            ),
            selected_trial_id=None,
        )
        index = build_exploration_index(
            _history(parent, [generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)
        lane_a, lane_b = freeze_exploration_lane_intents(index, frontier)
        self.assertEqual(frontier.backlog_count, 2)
        self.assertEqual((lane_a["mode"], lane_b["mode"]), ("novel", "novel"))

    def test_fifteen_generation_exploration_shares_the_fourteen_kib_memory_cap(
        self,
    ) -> None:
        parent = fake_bootstrap_current_best()
        generations = [
            _generation(
                parent,
                generation_index=index,
                trial_a=_trial(
                    f"trial-generation-{index:02d}-a",
                    "qualified_not_selected",
                    seed=f"generation-{index:02d}-a",
                ),
                trial_b=_trial(
                    f"trial-generation-{index:02d}-b",
                    "qualified_not_selected",
                    seed=f"generation-{index:02d}-b",
                ),
                selected_trial_id=None,
            )
            for index in range(1, 16)
        ]
        index = build_exploration_index(
            _history(parent, generations),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)
        self.assertEqual(len(index.cards), 30)
        self.assertEqual(len(frontier.queued_card_hashes), 4)
        self.assertEqual(frontier.backlog_count, 30)
        projection = build_exploration_prompt_projection(index, frontier)
        records = [
            {
                "candidate": {"candidate_id": f"candidate-{item:02d}"},
                "decision": {
                    "accepted": item == 1,
                    "reason": f"reason-{item:02d}-" + ("x" * 490),
                    "surface_classification": "code_level",
                    "gate_statuses": {
                        f"gate-{gate}-" + ("y" * 120): "fail"
                        for gate in range(8)
                    },
                },
            }
            for item in range(1, 21)
        ]
        memory = build_experiment_memory(
            records,
            exploration_projection=projection,
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(memory),
            EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES,
        )
        candidate_ids = {
            item["candidate_id"] for item in memory["experiments"]
        }
        self.assertIn("candidate-01", candidate_ids)
        self.assertIn("candidate-20", candidate_ids)
        self.assertIn("exploration", memory)
        self.assertGreaterEqual(
            len(memory["exploration"]["priority_or_eligible_cards"]),
            1,
        )

    def test_explicit_revisit_creates_new_card_without_rewriting_old_card(self) -> None:
        parent = fake_bootstrap_current_best()
        first_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-source", "qualified_not_selected", seed="7"),
            trial_b=_trial("trial-head", "accepted", seed="8"),
            selected_trial_id="trial-head",
        )
        first_index = build_exploration_index(
            _history(parent, [first_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        source = first_index.cards[0]
        revisit_intent = {
            "schema_version": EXPLORATION_LANE_INTENT_SCHEMA_VERSION,
            "lane_id": "lane-a",
            "mode": "revisit",
            "current_head_hash": first_index.current_head_hash,
            "inspired_by_trial_id": source.trial_id,
            "inspired_by_card_hash": source.card_hash,
            "hypothesis": {"public_input_fields": ["height"]},
            "verifier_equivalence_exclusion_set_hash": (
                verifier_equivalence_exclusion_set_hash(())
            ),
            "acceptance_authority": False,
        }
        second_generation = _generation(
            parent,
            generation_index=2,
            trial_a=_trial(
                "trial-revisit",
                "accepted",
                seed="9",
                lane_intent=revisit_intent,
            ),
            trial_b=_trial("trial-other", "qualified_not_selected", seed="a"),
            selected_trial_id="trial-revisit",
        )
        index = build_exploration_index(
            _history(parent, [first_generation, second_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        self.assertEqual(index.cards[0].card_hash, source.card_hash)
        revisit = next(card for card in index.cards if card.trial_id == "trial-revisit")
        self.assertEqual(revisit.lineage_status, "incorporated")
        self.assertEqual(revisit.inspired_by_card_hash, source.card_hash)
        frontier = build_revisit_frontier(index)
        self.assertNotIn(source.card_hash, frontier.queued_card_hashes)
        self.assertIn(
            {
                "card_hash": source.card_hash,
                "reason": "already_revisited_with_public_evidence",
            },
            frontier.ineligible_cards,
        )

    def test_verified_not_run_revisit_does_not_consume_source_card(self) -> None:
        parent = fake_bootstrap_current_best()
        first_generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-source", "qualified_not_selected", seed="v1"),
            trial_b=_trial("trial-head", "accepted", seed="v2"),
            selected_trial_id="trial-head",
        )
        first_index = build_exploration_index(
            _history(parent, [first_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        source = first_index.cards[0]
        revisit_intent = freeze_exploration_lane_intents(
            first_index,
            build_revisit_frontier(first_index),
        )[0]
        second_generation = _generation(
            parent,
            generation_index=2,
            trial_a=_trial(
                "trial-revisit-unrun",
                "verified_not_run",
                seed="v3",
                lane_intent=revisit_intent,
            ),
            trial_b=_trial("trial-paired-failure", "candidate_failed", seed="v4"),
            selected_trial_id=None,
        )
        index = build_exploration_index(
            _history(parent, [first_generation, second_generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        frontier = build_revisit_frontier(index)

        self.assertIn(source.card_hash, frontier.queued_card_hashes)

    def test_exploration_shares_existing_memory_budget_without_changing_width1(self) -> None:
        parent = fake_bootstrap_current_best()
        generation = _generation(
            parent,
            generation_index=1,
            trial_a=_trial("trial-good-a", "qualified_not_selected", seed="b"),
            trial_b=_trial("trial-negative", "qualified_not_selected", seed="c"),
            selected_trial_id="trial-good-a",
        )
        index = build_exploration_index(
            _history(parent, [generation]),
            task_projector=_PROJECTOR,
            expected_projector_contract_hash=_PROJECTOR_CONTRACT_HASH,
        )
        projection = build_exploration_prompt_projection(
            index,
            build_revisit_frontier(index),
        )
        memory = build_experiment_memory(
            [],
            exploration_projection=projection,
        )
        self.assertEqual(
            memory["schema_version"],
            "true_rl_agent.experiment_memory.v4",
        )
        self.assertLessEqual(
            canonical_prompt_json_bytes(memory),
            EXPERIMENT_MEMORY_MAX_SERIALIZED_BYTES,
        )
        self.assertEqual(recompact_experiment_memory(memory), memory)
        width1 = build_experiment_memory([])
        self.assertEqual(
            width1["schema_version"],
            "true_rl_agent.experiment_memory.v3",
        )
        self.assertNotIn("exploration", width1)


def _history(
    parent: CurrentBestEvalStats,
    generations: list[Mapping[str, Any]],
) -> ProjectedPublicHistory:
    records = tuple(
        {
            "record_kind": "generation",
            "generation": generation,
            "decision": {"accepted": generation["generation_outcome"] == "accepted"},
        }
        for generation in generations
    )
    return ProjectedPublicHistory(
        records=records,
        initial_current_best=parent,
        session_genesis_hash=FAKE_SESSION_GENESIS_HASH,
        acceptance_protocol_hash=FAKE_ACCEPTANCE_PROTOCOL.protocol_hash,
        current_best=parent,
        public_score_source=FAKE_ACCEPTANCE_PROTOCOL.score_source,
    )


def _generation(
    parent: CurrentBestEvalStats,
    *,
    generation_index: int,
    trial_a: Mapping[str, Any],
    trial_b: Mapping[str, Any],
    selected_trial_id: str | None,
) -> dict[str, Any]:
    parent_payload = parent.to_public_dict()
    return {
        "schema_version": GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
        "generation_id": f"generation-{generation_index:03d}",
        "generation_index": generation_index,
        "generation_outcome": "accepted" if selected_trial_id else "no_selection",
        "parent_current_best_hash": canonical_json_hash(parent_payload),
        "parent_current_best": parent_payload,
        "resulting_current_best_hash": canonical_json_hash(parent_payload),
        "trials": [dict(trial_a), dict(trial_b)],
        "selection": {
            "selected_trial_id": selected_trial_id,
            "acceptance_authority": False,
        },
        "guard_reference": None,
        "formal": {"status": "complete"} if selected_trial_id else None,
    }


def _trial(
    trial_id: str,
    status: str,
    *,
    seed: str,
    lane_intent: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    lane_id = "lane-a" if trial_id.endswith(("a", "source", "loser", "negative", "revisit")) else "lane-b"
    formal = status in {"accepted", "formal_rejected"}
    trained = status in {
        "screened_only",
        "qualified_not_selected",
        "formal_rejected",
        "accepted",
    }
    public_dev = trained
    public_accept = status in {
        "qualified_not_selected",
        "formal_rejected",
        "accepted",
    }
    return {
        "lane_id": lane_id,
        "trial_id": trial_id,
        "trial_status": status,
        "candidate": {
            "candidate_id": trial_id.replace("trial-", "candidate-"),
            "program_source_checksum": "sha256:" + digest,
        },
        "mechanism_operation_delta": _delta(seed),
        "trial_contract_hash": digest,
        "authoring": {
            "prompt_payload_checksum": "sha256:" + digest,
            "authoring_evidence_checksum": "sha256:" + digest,
        },
        "train": {"train_result_hash": digest} if trained else None,
        "public_dev": (
            {"phase_result_hash": digest, "public_summary": {}}
            if public_dev
            else None
        ),
        "public_accept": (
            {"phase_result_hash": digest, "public_summary": {}}
            if public_accept
            else None
        ),
        "formal_record_hash": digest if formal else None,
        "lane_intent": lane_intent or {
            "schema_version": "test.generation_lane_intent.v1",
            "lane_id": lane_id,
            "intent": "novel",
        },
    }


def _delta(seed: str) -> dict[str, Any]:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return {
        "schema_version": "true_rl_agent.candidate_mechanism_operation_delta.v2",
        "operation": "add",
        "before_component": None,
        "after_component": {
            "mechanism_id": f"RewardTerm:term_{seed}",
            "structure_hash": digest,
            "parameter_normalized_structure_hash": digest,
            "input_fields": ["height"],
        },
        "anchor_source_checksum": "sha256:" + "f" * 64,
        "candidate_source_checksum": "sha256:" + digest,
    }


if __name__ == "__main__":
    unittest.main()
