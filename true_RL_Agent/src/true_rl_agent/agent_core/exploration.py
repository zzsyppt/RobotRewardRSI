"""Ledger-derived exploration evidence and bounded revisit scheduling.

The index is a deterministic, disposable projection.  It has no acceptance or
ledger authority and deliberately contains no model-writable persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from true_rl_agent.agent_core.exploration_projection import (
    EXPLORATION_PROMPT_MAX_BYTES as EXPLORATION_PROMPT_MAX_BYTES,
    EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION,
    MAX_EXPLORATION_PROMPT_INPUT_FIELDS,
    MAX_EXPLORATION_PROMPT_TEXT_CHARS,
    fit_exploration_prompt_projection,
    minimal_exploration_prompt_card,
)
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
    require_generation_lane_intent,
)
from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.candidates.authoring_policy import (
    mechanism_operation_effect,
    require_mechanism_operation_delta,
    single_component_verifier_equivalence_key,
    verifier_equivalence_exclusion_set_hash,
)
from true_rl_agent.evaluation.acceptance_replay import ProjectedPublicHistory
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes


EXPLORATION_EVIDENCE_CARD_SCHEMA_VERSION = (
    "true_rl_agent.exploration_evidence_card.v2"
)
EXPLORATION_INDEX_SCHEMA_VERSION = "true_rl_agent.exploration_index.v2"
REVISIT_FRONTIER_SCHEMA_VERSION = "true_rl_agent.revisit_frontier.v2"
EXPLORATION_LANE_INTENT_SCHEMA_VERSION = (
    "true_rl_agent.generation_lane_intent.exploration.v1"
)
EXPLORATION_MAX_FRONTIER_CARDS = 4
EXPLORATION_MAX_REVISIT_ATTEMPTS = 2
EXPLORATION_MAX_REVISIT_HEAD_TRANSITIONS = 1
EXPLORATION_MAX_INPUT_SIGNATURE_ADVISORIES = 2
_MAX_LANE_INPUT_FIELD_PREVIEW = 8
# Full evidence cards live only in the disposable ledger-derived index.  They
# are not prompt sections; the bounded prompt projection may omit any card that
# cannot be represented within its smaller, independently enforced contract.
_MAX_EVIDENCE_CARD_BYTES = 12 * 1024
_EVIDENCE_TIERS = frozenset(
    {"verified", "trained", "public_dev", "public_complete", "formal"}
)
_OBSERVED_EFFECTS = frozenset(
    {"strong_positive", "positive", "mixed", "neutral", "negative", "unknown"}
)
_VALUE_LABELS = frozenset({"high", "medium", "low", "unknown"})
_FAILURE_SCOPES = frozenset(
    {
        "none",
        "authoring_or_implementation",
        "paired_authoring",
        "scientific_public",
        "formal_private",
        "infrastructure",
    }
)
_SELECTION_STATUSES = frozenset(
    {
        "not_evaluated",
        "selected",
        "qualified_not_selected",
        "inconclusive",
    }
)
_LINEAGE_STATUSES = frozenset({"independent", "inspired", "incorporated"})
_REVISIT_DISPOSITIONS = frozenset(
    {
        "priority",
        "eligible",
        "cooldown",
        "suppress_equivalent",
        "incorporated",
        "not_evaluable",
    }
)


@dataclass(frozen=True)
class TaskExplorationAssessment:
    observed_effect: str
    value_label: str
    public_evidence_summary: Mapping[str, Any]
    reason_codes: tuple[str, ...]
    revisit_rank_key: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.observed_effect not in _OBSERVED_EFFECTS:
            raise ValueError("exploration observed effect is invalid")
        if self.value_label not in _VALUE_LABELS:
            raise ValueError("exploration value label is invalid")
        summary = _public_mapping(self.public_evidence_summary)
        if canonical_prompt_json_bytes(summary) > 4096:
            raise ValueError("task exploration evidence summary exceeds its byte limit")
        reasons = _bounded_text_tuple(self.reason_codes, "exploration reason code", 8)
        rank_key = _revisit_rank_key(self.revisit_rank_key)
        object.__setattr__(self, "public_evidence_summary", summary)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "revisit_rank_key", rank_key)


class TaskExplorationProjector(Protocol):
    task_id: str

    def contract(self) -> Mapping[str, Any]: ...

    def assess_trial(
        self,
        *,
        generation: Mapping[str, Any],
        trial: Mapping[str, Any],
        parent: CurrentBestEvalStats,
    ) -> TaskExplorationAssessment: ...


@dataclass(frozen=True)
class ExplorationEvidenceCard:
    session_genesis_hash: str
    task_id: str
    generation_id: str
    generation_index: int
    trial_id: str
    lane_id: str
    parent_current_best_hash: str
    candidate_id: str | None
    candidate_source_checksum: str | None
    mechanism_operation_delta: Mapping[str, Any] | None
    verifier_equivalence_key: str | None
    exact_implementation_key: str | None
    trial_status: str
    evidence_tier: str
    observed_effect: str
    value_label: str
    novelty: str
    failure_scope: str
    selection_status: str
    lineage_status: str
    revisit_disposition: str
    inspired_by_trial_id: str | None
    inspired_by_card_hash: str | None
    public_evidence_summary: Mapping[str, Any]
    reason_codes: tuple[str, ...]
    revisit_rank_key: tuple[int, ...]
    evidence_refs: Mapping[str, str]
    context_hash: str

    def __post_init__(self) -> None:
        require_sha256_digest(self.session_genesis_hash, "exploration session Genesis")
        for field_name in (
            "task_id",
            "generation_id",
            "trial_id",
            "lane_id",
            "trial_status",
            "novelty",
        ):
            _bounded_text(getattr(self, field_name), f"exploration {field_name}")
        if (
            isinstance(self.generation_index, bool)
            or not isinstance(self.generation_index, int)
            or self.generation_index <= 0
        ):
            raise ValueError("exploration generation_index must be positive")
        require_sha256_digest(
            self.parent_current_best_hash,
            "exploration parent current-best hash",
        )
        require_sha256_digest(self.context_hash, "exploration context hash")
        if self.candidate_source_checksum is not None and not (
            isinstance(self.candidate_source_checksum, str)
            and self.candidate_source_checksum.startswith("sha256:")
            and len(self.candidate_source_checksum) == 71
        ):
            raise ValueError("exploration candidate source checksum is invalid")
        delta = (
            require_mechanism_operation_delta(self.mechanism_operation_delta)
            if self.mechanism_operation_delta is not None
            else None
        )
        if (delta is None) != (self.verifier_equivalence_key is None):
            raise ValueError("exploration mechanism delta/key must be paired")
        if self.verifier_equivalence_key is not None:
            require_sha256_digest(
                self.verifier_equivalence_key,
                "exploration verifier equivalence key",
            )
        if self.exact_implementation_key is not None:
            require_sha256_digest(
                self.exact_implementation_key,
                "exploration exact implementation key",
            )
            if delta is None or self.candidate_source_checksum is None:
                raise ValueError(
                    "exploration exact implementation key lacks source/delta binding"
                )
        if self.evidence_tier not in _EVIDENCE_TIERS:
            raise ValueError("exploration evidence tier is invalid")
        if self.observed_effect not in _OBSERVED_EFFECTS:
            raise ValueError("exploration observed effect is invalid")
        if self.value_label not in _VALUE_LABELS:
            raise ValueError("exploration value label is invalid")
        if self.failure_scope not in _FAILURE_SCOPES:
            raise ValueError("exploration failure scope is invalid")
        if self.selection_status not in _SELECTION_STATUSES:
            raise ValueError("exploration selection status is invalid")
        if self.lineage_status not in _LINEAGE_STATUSES:
            raise ValueError("exploration lineage status is invalid")
        if self.revisit_disposition not in _REVISIT_DISPOSITIONS:
            raise ValueError("exploration revisit disposition is invalid")
        if self.lineage_status == "inspired":
            if self.inspired_by_trial_id is None or self.inspired_by_card_hash is None:
                raise ValueError("inspired exploration card lacks lineage binding")
            require_sha256_digest(
                self.inspired_by_card_hash,
                "exploration inspired-by card hash",
            )
        elif self.lineage_status == "incorporated":
            if (self.inspired_by_trial_id is None) != (
                self.inspired_by_card_hash is None
            ):
                raise ValueError("incorporated inspired-by binding must be paired")
            if self.inspired_by_card_hash is not None:
                require_sha256_digest(
                    self.inspired_by_card_hash,
                    "exploration incorporated inspired-by card hash",
                )
        elif self.inspired_by_trial_id is not None or self.inspired_by_card_hash is not None:
            raise ValueError("non-inspired exploration card carries inspired-by binding")
        summary = _public_mapping(self.public_evidence_summary)
        refs = _hash_mapping(self.evidence_refs, "exploration evidence ref")
        reasons = _bounded_text_tuple(self.reason_codes, "exploration reason code", 8)
        rank_key = _revisit_rank_key(self.revisit_rank_key)
        object.__setattr__(self, "mechanism_operation_delta", delta)
        object.__setattr__(self, "public_evidence_summary", summary)
        object.__setattr__(self, "evidence_refs", refs)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "revisit_rank_key", rank_key)
        if canonical_prompt_json_bytes(self.to_dict()) > _MAX_EVIDENCE_CARD_BYTES:
            raise ValueError("exploration evidence card exceeds its byte limit")

    @property
    def card_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPLORATION_EVIDENCE_CARD_SCHEMA_VERSION,
            "session_genesis_hash": self.session_genesis_hash,
            "task_id": self.task_id,
            "generation_id": self.generation_id,
            "generation_index": self.generation_index,
            "trial_id": self.trial_id,
            "lane_id": self.lane_id,
            "parent_current_best_hash": self.parent_current_best_hash,
            "candidate_id": self.candidate_id,
            "candidate_source_checksum": self.candidate_source_checksum,
            "mechanism_operation_delta": (
                dict(self.mechanism_operation_delta)
                if self.mechanism_operation_delta is not None
                else None
            ),
            "verifier_equivalence_key": self.verifier_equivalence_key,
            "exact_implementation_key": self.exact_implementation_key,
            "trial_status": self.trial_status,
            "evidence_tier": self.evidence_tier,
            "observed_effect": self.observed_effect,
            "value_label": self.value_label,
            "novelty": self.novelty,
            "failure_scope": self.failure_scope,
            "selection_status": self.selection_status,
            "lineage_status": self.lineage_status,
            "revisit_disposition": self.revisit_disposition,
            "inspired_by_trial_id": self.inspired_by_trial_id,
            "inspired_by_card_hash": self.inspired_by_card_hash,
            "public_evidence_summary": dict(self.public_evidence_summary),
            "reason_codes": list(self.reason_codes),
            "revisit_rank_key": list(self.revisit_rank_key),
            "evidence_refs": dict(self.evidence_refs),
            "context_hash": self.context_hash,
            "acceptance_authority": False,
        }


@dataclass(frozen=True)
class ExplorationIndex:
    task_id: str
    session_genesis_hash: str
    source_projection_hash: str
    current_head_hash: str
    projector_contract_hash: str
    cards: tuple[ExplorationEvidenceCard, ...]
    head_transition_generations: tuple[int, ...]
    revisit_service_facts: tuple[Mapping[str, Any], ...]

    @property
    def index_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPLORATION_INDEX_SCHEMA_VERSION,
            "task_id": self.task_id,
            "session_genesis_hash": self.session_genesis_hash,
            "source_projection_hash": self.source_projection_hash,
            "current_head_hash": self.current_head_hash,
            "projector_contract_hash": self.projector_contract_hash,
            "cards": [card.to_dict() for card in self.cards],
            "head_transition_generations": list(
                self.head_transition_generations
            ),
            "revisit_service_facts": [
                dict(item) for item in self.revisit_service_facts
            ],
            "acceptance_authority": False,
        }


@dataclass(frozen=True)
class RevisitFrontier:
    index_hash: str
    current_head_hash: str
    queued_card_hashes: tuple[str, ...]
    deferred_cards: tuple[Mapping[str, Any], ...]
    ineligible_cards: tuple[Mapping[str, str], ...]
    backlog_count: int

    def __post_init__(self) -> None:
        require_sha256_digest(self.index_hash, "revisit frontier index hash")
        require_sha256_digest(self.current_head_hash, "revisit frontier current head")
        for value in self.queued_card_hashes:
            require_sha256_digest(value, "revisit frontier hash")
        if (
            len(self.queued_card_hashes) > EXPLORATION_MAX_FRONTIER_CARDS
            or self.backlog_count
            != len(self.queued_card_hashes) + len(self.deferred_cards)
        ):
            raise ValueError("revisit frontier backlog count mismatch")
        deferred: list[dict[str, Any]] = []
        deferred_hashes: set[str] = set()
        expected_position = len(self.queued_card_hashes) + 1
        for item in self.deferred_cards:
            mapping = _exact_mapping(
                item,
                {"card_hash", "queue_position", "reason"},
                "revisit frontier deferred card",
            )
            require_sha256_digest(
                mapping["card_hash"],
                "revisit frontier deferred card hash",
            )
            if (
                mapping["reason"] != "frontier_capacity"
                or mapping["queue_position"] != expected_position
                or mapping["card_hash"] in deferred_hashes
                or mapping["card_hash"] in self.queued_card_hashes
            ):
                raise ValueError("revisit frontier deferred card is invalid")
            deferred_hashes.add(mapping["card_hash"])
            deferred.append(mapping)
            expected_position += 1
        ineligible: list[dict[str, str]] = []
        seen_hashes: set[str] = set()
        for item in self.ineligible_cards:
            mapping = _exact_mapping(
                item,
                {"card_hash", "reason"},
                "revisit frontier ineligible card",
            )
            require_sha256_digest(
                mapping["card_hash"],
                "revisit frontier ineligible card hash",
            )
            if mapping["reason"] not in {
                "already_revisited_with_public_evidence",
                "revisit_attempt_limit_reached",
                "stale_after_head_transitions",
                "incorporated_equivalence",
                "duplicate_eligible_equivalence",
            }:
                raise ValueError("revisit frontier ineligible reason is invalid")
            if (
                mapping["card_hash"] in seen_hashes
                or mapping["card_hash"] in deferred_hashes
                or mapping["card_hash"] in self.queued_card_hashes
            ):
                raise ValueError("revisit frontier ineligible card is duplicated")
            seen_hashes.add(mapping["card_hash"])
            ineligible.append(mapping)
        object.__setattr__(self, "deferred_cards", tuple(deferred))
        object.__setattr__(self, "ineligible_cards", tuple(ineligible))

    @property
    def frontier_hash(self) -> str:
        return canonical_json_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": REVISIT_FRONTIER_SCHEMA_VERSION,
            "index_hash": self.index_hash,
            "current_head_hash": self.current_head_hash,
            "queued_card_hashes": list(self.queued_card_hashes),
            "deferred_cards": [dict(item) for item in self.deferred_cards],
            "ineligible_cards": [dict(item) for item in self.ineligible_cards],
            "backlog_count": self.backlog_count,
            "acceptance_authority": False,
        }


@dataclass(frozen=True)
class GenerationExplorationPlan:
    """One disposable scheduling view derived from a validated ledger replay."""

    index: ExplorationIndex
    frontier: RevisitFrontier
    blocked_verifier_equivalence_keys: tuple[str, ...]
    lane_intents: tuple[Mapping[str, Any], Mapping[str, Any]]
    prompt_projection: Mapping[str, Any]

    @property
    def evaluated_verifier_equivalence_keys(self) -> tuple[str, ...]:
        # The replayed index is scoped to one Genesis (protocol, seed and backend).
        # Untrained candidates retain signatures but cannot block scientific work.
        return tuple(sorted({
            single_component_verifier_equivalence_key(card.mechanism_operation_delta)
            for card in self.index.cards
            if card.parent_current_best_hash == self.index.current_head_hash
            and card.evidence_tier in {"public_dev", "public_complete", "formal"}
            and card.mechanism_operation_delta is not None
        }))

    def __post_init__(self) -> None:
        expected_frontier = _require_canonical_revisit_frontier(
            self.index,
            self.frontier,
        )
        expected_blocked = _blocked_verifier_equivalence_keys(self.index)
        if self.blocked_verifier_equivalence_keys != expected_blocked:
            raise ValueError("generation exploration plan exclusion set mismatch")
        if not isinstance(self.lane_intents, tuple) or len(self.lane_intents) != 2:
            raise ValueError("generation exploration plan requires two lane intents")
        intents = tuple(
            validate_exploration_lane_intent(value)
            for value in self.lane_intents
        )
        if tuple(value["lane_id"] for value in intents) != ("lane-a", "lane-b"):
            raise ValueError("generation exploration plan lane order mismatch")
        expected_intents = freeze_exploration_lane_intents(
            self.index,
            expected_frontier,
        )
        if intents != expected_intents:
            raise ValueError("generation exploration plan lane intents mismatch")
        projection = build_exploration_prompt_projection(
            self.index,
            expected_frontier,
        )
        if dict(self.prompt_projection) != projection:
            raise ValueError("generation exploration plan prompt projection mismatch")
        object.__setattr__(self, "lane_intents", intents)
        object.__setattr__(self, "prompt_projection", projection)

    @property
    def plan_hash(self) -> str:
        return canonical_json_hash(
            {
                "schema_version": "true_rl_agent.generation_exploration_plan.v2",
                "index_hash": self.index.index_hash,
                "frontier_hash": self.frontier.frontier_hash,
                "blocked_verifier_equivalence_keys_hash": (
                    verifier_equivalence_exclusion_set_hash(
                        self.blocked_verifier_equivalence_keys
                    )
                ),
                "lane_intent_hashes": [
                    canonical_json_hash(value) for value in self.lane_intents
                ],
                "prompt_projection_hash": canonical_json_hash(
                    self.prompt_projection
                ),
                "acceptance_authority": False,
            }
        )


def build_exploration_index(
    history: ProjectedPublicHistory,
    *,
    task_projector: TaskExplorationProjector,
    expected_projector_contract_hash: str,
) -> ExplorationIndex:
    """Project cards only after the authoritative replay has succeeded."""

    if not isinstance(history, ProjectedPublicHistory):
        raise TypeError("exploration index requires ProjectedPublicHistory")
    contract = _public_mapping(task_projector.contract())
    if contract.get("task_id") != task_projector.task_id:
        raise ValueError("task exploration projector contract identity mismatch")
    expected_hash = require_sha256_digest(
        expected_projector_contract_hash,
        "expected exploration projector contract",
    )
    if canonical_json_hash(contract) != expected_hash:
        raise ValueError("exploration projector is not generation-session bound")
    session_hash = require_sha256_digest(
        history.session_genesis_hash,
        "exploration history session Genesis",
    )
    if history.current_best is None:
        raise ValueError("exploration index requires a replay current head")
    generation_projections: list[Mapping[str, Any]] = []
    cards: list[ExplorationEvidenceCard] = []
    head_transition_generations: list[int] = []
    revisit_service_facts: list[dict[str, Any]] = []
    prior_cards_by_hash: dict[str, ExplorationEvidenceCard] = {}
    for record in history.records:
        generation = record.get("generation")
        if not isinstance(generation, Mapping):
            continue
        if (
            generation.get("schema_version")
            != GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION
        ):
            raise ValueError("exploration generation projection schema mismatch")
        generation_projections.append(generation)
        generation_index = int(generation["generation_index"])
        if generation.get("generation_outcome") == "accepted":
            head_transition_generations.append(generation_index)
        parent = CurrentBestEvalStats.from_mapping(
            generation.get("parent_current_best")
        )
        if canonical_json_hash(parent.to_public_dict()) != generation.get(
            "parent_current_best_hash"
        ):
            raise ValueError("exploration generation parent hash mismatch")
        trials = generation.get("trials")
        if not isinstance(trials, list) or len(trials) != 2:
            raise ValueError("exploration generation requires two trial projections")
        for trial in trials:
            service_fact = _revisit_service_fact(
                generation_index=generation_index,
                trial=trial,
                prior_cards_by_hash=prior_cards_by_hash,
            )
            if service_fact is not None:
                revisit_service_facts.append(service_fact)
            card = _build_card(
                session_genesis_hash=session_hash,
                generation=generation,
                trial=trial,
                parent=parent,
                task_projector=task_projector,
                prior_cards_by_hash=prior_cards_by_hash,
            )
            if card is not None:
                cards.append(card)
                prior_cards_by_hash[card.card_hash] = card
    return ExplorationIndex(
        task_id=task_projector.task_id,
        session_genesis_hash=session_hash,
        source_projection_hash=canonical_json_hash(generation_projections),
        current_head_hash=canonical_json_hash(history.current_best.to_public_dict()),
        projector_contract_hash=canonical_json_hash(contract),
        cards=tuple(cards),
        head_transition_generations=tuple(head_transition_generations),
        revisit_service_facts=tuple(revisit_service_facts),
    )


def _revisit_service_fact(
    *,
    generation_index: int,
    trial: Any,
    prior_cards_by_hash: Mapping[str, ExplorationEvidenceCard],
) -> dict[str, Any] | None:
    """Record scheduling service even when authoring produced no scientific card."""

    if not isinstance(trial, Mapping):
        raise ValueError("exploration trial projection is invalid")
    intent = trial.get("lane_intent")
    if not (
        isinstance(intent, Mapping)
        and intent.get("schema_version") == EXPLORATION_LANE_INTENT_SCHEMA_VERSION
    ):
        return None
    canonical = validate_exploration_lane_intent(intent)
    if canonical["mode"] != "revisit":
        return None
    source_hash = canonical["inspired_by_card_hash"]
    source = prior_cards_by_hash.get(source_hash)
    if (
        source is None
        or source.trial_id != canonical["inspired_by_trial_id"]
        or source.generation_index >= generation_index
    ):
        raise ValueError("exploration revisit service source is not in prior history")
    status = _bounded_text(trial.get("trial_status"), "revisit service trial status")
    return {
        "source_card_hash": source_hash,
        "source_trial_id": source.trial_id,
        "generation_index": generation_index,
        "trial_id": _bounded_text(trial.get("trial_id"), "revisit service trial_id"),
        "trial_status": status,
        "public_evidence_obtained": trial.get("public_dev") is not None,
    }


def _head_transition_distance(
    index: ExplorationIndex,
    card: ExplorationEvidenceCard,
) -> int:
    """Count accepted-head advances since the card's frozen parent context."""

    return sum(
        generation_index >= card.generation_index
        for generation_index in index.head_transition_generations
    )


def build_revisit_frontier(index: ExplorationIndex) -> RevisitFrontier:
    if not isinstance(index, ExplorationIndex):
        raise TypeError("revisit frontier requires an ExplorationIndex")
    cards_by_hash = {card.card_hash: card for card in index.cards}
    service_attempt_counts: dict[str, int] = {}
    consumed_equivalence_keys: set[str] = set()
    for fact in index.revisit_service_facts:
        source_hash = str(fact["source_card_hash"])
        source = cards_by_hash.get(source_hash)
        if source is None or source.verifier_equivalence_key is None:
            raise ValueError("revisit service source has no verifier equivalence")
        equivalence_key = source.verifier_equivalence_key
        service_attempt_counts[equivalence_key] = (
            service_attempt_counts.get(equivalence_key, 0) + 1
        )
        if fact["public_evidence_obtained"] is True:
            consumed_equivalence_keys.add(equivalence_key)
    incorporated_keys = {
        card.verifier_equivalence_key
        for card in index.cards
        if card.lineage_status == "incorporated"
        and card.verifier_equivalence_key is not None
    }
    candidates = [
        card
        for card in index.cards
        if card.revisit_disposition in {"priority", "eligible"}
        and card.candidate_source_checksum is not None
        and card.verifier_equivalence_key is not None
    ]
    candidates.sort(
        key=lambda card: (
            0 if card.revisit_disposition == "priority" else 1,
            service_attempt_counts.get(card.verifier_equivalence_key, 0),
            _head_transition_distance(index, card),
            0 if card.revisit_rank_key else 1,
            tuple(-value for value in card.revisit_rank_key),
            -card.generation_index,
            card.trial_id,
        )
    )
    queued: list[ExplorationEvidenceCard] = []
    ineligible: list[dict[str, str]] = []
    queued_equivalence_keys: set[str] = set()
    for card in candidates:
        if card.verifier_equivalence_key in consumed_equivalence_keys:
            reason = "already_revisited_with_public_evidence"
        elif card.verifier_equivalence_key in incorporated_keys:
            reason = "incorporated_equivalence"
        elif (
            service_attempt_counts.get(card.verifier_equivalence_key, 0)
            >= EXPLORATION_MAX_REVISIT_ATTEMPTS
        ):
            reason = "revisit_attempt_limit_reached"
        elif (
            _head_transition_distance(index, card)
            > EXPLORATION_MAX_REVISIT_HEAD_TRANSITIONS
        ):
            reason = "stale_after_head_transitions"
        elif card.verifier_equivalence_key in queued_equivalence_keys:
            reason = "duplicate_eligible_equivalence"
        else:
            queued.append(card)
            queued_equivalence_keys.add(card.verifier_equivalence_key)
            continue
        ineligible.append({"card_hash": card.card_hash, "reason": reason})
    queued_count = len(queued)
    deferred = tuple(
        {
            "card_hash": card.card_hash,
            "queue_position": position,
            "reason": "frontier_capacity",
        }
        for position, card in enumerate(
            queued[EXPLORATION_MAX_FRONTIER_CARDS:],
            start=EXPLORATION_MAX_FRONTIER_CARDS + 1,
        )
    )
    queued = queued[:EXPLORATION_MAX_FRONTIER_CARDS]
    return RevisitFrontier(
        index_hash=index.index_hash,
        current_head_hash=index.current_head_hash,
        queued_card_hashes=tuple(card.card_hash for card in queued),
        deferred_cards=deferred,
        ineligible_cards=tuple(ineligible),
        backlog_count=queued_count,
    )


def _require_canonical_revisit_frontier(
    index: ExplorationIndex,
    frontier: RevisitFrontier,
) -> RevisitFrontier:
    if not isinstance(index, ExplorationIndex) or not isinstance(
        frontier,
        RevisitFrontier,
    ):
        raise TypeError("canonical revisit frontier requires typed inputs")
    expected = build_revisit_frontier(index)
    if frontier.to_dict() != expected.to_dict():
        raise ValueError("revisit frontier is not the canonical index projection")
    return expected


def _blocked_verifier_equivalence_keys(
    index: ExplorationIndex,
) -> tuple[str, ...]:
    """Return the complete scientific-negative exclusion set for this session.

    The prompt projection stays bounded, while the trusted authoring gate sees
    every verifier-equivalent negative card.  Infrastructure and unevaluated
    failures never enter this set because their disposition is not
    ``suppress_equivalent``.
    """

    return tuple(
        sorted(
            {
                card.verifier_equivalence_key
                for card in index.cards
                if card.revisit_disposition == "suppress_equivalent"
                and card.verifier_equivalence_key is not None
            }
        )
    )


def freeze_exploration_lane_intents(
    index: ExplorationIndex,
    frontier: RevisitFrontier,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Freeze at most one revisit and always preserve one novel lane."""

    frontier = _require_canonical_revisit_frontier(index, frontier)
    by_hash = {card.card_hash: card for card in index.cards}
    revisit = next(
        (
            by_hash[value]
            for value in frontier.queued_card_hashes
            if value in by_hash
            and by_hash[value].revisit_disposition == "priority"
        ),
        None,
    )
    lane_a = _lane_intent(
        lane_id="lane-a",
        current_head_hash=index.current_head_hash,
        revisit=revisit,
        exclusion_set_hash=verifier_equivalence_exclusion_set_hash(
            _blocked_verifier_equivalence_keys(index)
        ),
    )
    lane_b = _lane_intent(
        lane_id="lane-b",
        current_head_hash=index.current_head_hash,
        revisit=None,
        exclusion_set_hash=verifier_equivalence_exclusion_set_hash(
            _blocked_verifier_equivalence_keys(index)
        ),
    )
    return lane_a, lane_b


def build_generation_exploration_plan(
    history: ProjectedPublicHistory,
    *,
    task_projector: TaskExplorationProjector,
    expected_projector_contract_hash: str,
) -> GenerationExplorationPlan:
    index = build_exploration_index(
        history,
        task_projector=task_projector,
        expected_projector_contract_hash=expected_projector_contract_hash,
    )
    frontier = build_revisit_frontier(index)
    lane_intents = freeze_exploration_lane_intents(index, frontier)
    projection = build_exploration_prompt_projection(index, frontier)
    return GenerationExplorationPlan(
        index=index,
        frontier=frontier,
        blocked_verifier_equivalence_keys=(
            _blocked_verifier_equivalence_keys(index)
        ),
        lane_intents=lane_intents,
        prompt_projection=projection,
    )


def validate_exploration_lane_intent(value: Any) -> dict[str, Any]:
    mapping = _exact_mapping(
        value,
        {
            "schema_version",
            "lane_id",
            "mode",
            "current_head_hash",
            "inspired_by_trial_id",
            "inspired_by_card_hash",
            "hypothesis",
            "verifier_equivalence_exclusion_set_hash",
            "acceptance_authority",
        },
        "exploration lane intent",
    )
    if (
        mapping["schema_version"] != EXPLORATION_LANE_INTENT_SCHEMA_VERSION
        or mapping["lane_id"] not in {"lane-a", "lane-b"}
        or mapping["mode"] not in {"novel", "revisit"}
        or mapping["acceptance_authority"] is not False
    ):
        raise ValueError("exploration lane intent contract mismatch")
    require_sha256_digest(mapping["current_head_hash"], "lane intent current head")
    if mapping["mode"] == "revisit":
        _bounded_text(mapping["inspired_by_trial_id"], "inspired_by_trial_id")
        require_sha256_digest(mapping["inspired_by_card_hash"], "inspired-by card hash")
        if not isinstance(mapping["hypothesis"], Mapping):
            raise ValueError("revisit lane intent hypothesis is required")
    elif any(
        mapping[key] is not None
        for key in ("inspired_by_trial_id", "inspired_by_card_hash", "hypothesis")
    ):
        raise ValueError("novel lane intent cannot carry revisit evidence")
    require_sha256_digest(
        mapping["verifier_equivalence_exclusion_set_hash"],
        "lane intent verifier equivalence exclusion set hash",
    )
    return require_generation_lane_intent(_public_mapping(mapping))


def build_exploration_prompt_projection(
    index: ExplorationIndex,
    frontier: RevisitFrontier,
) -> dict[str, Any]:
    frontier = _require_canonical_revisit_frontier(index, frontier)
    by_hash = {card.card_hash: card for card in index.cards}
    positive = []
    for value in frontier.queued_card_hashes[:2]:
        card = by_hash.get(value)
        prompt_card = _prompt_card(card) if card is not None else None
        if prompt_card is not None:
            positive.append(prompt_card)
    negative: list[dict[str, Any]] = []
    seen_negative_keys: set[str] = set()
    for card in reversed(index.cards):
        key = card.verifier_equivalence_key
        if (
            card.revisit_disposition != "suppress_equivalent"
            or key is None
            or key in seen_negative_keys
        ):
            continue
        prompt_card = _prompt_card(card)
        if prompt_card is None:
            continue
        negative.append(minimal_exploration_prompt_card(prompt_card))
        seen_negative_keys.add(key)
        if len(negative) == 2:
            break
    advisories = _negative_input_signature_advisories(index)
    return fit_exploration_prompt_projection(
        _prompt_payload(
            index,
            frontier,
            positive,
            negative,
            advisories,
        )
    )


def _negative_input_signature_advisories(
    index: ExplorationIndex,
) -> list[dict[str, Any]]:
    """Summarize repeated negative input regions without declaring equivalence."""

    groups: dict[tuple[str, tuple[str, ...]], list[ExplorationEvidenceCard]] = {}
    for card in index.cards:
        if (
            card.revisit_disposition != "suppress_equivalent"
            or card.parent_current_best_hash != index.current_head_hash
            or card.mechanism_operation_delta is None
            or card.verifier_equivalence_key is None
        ):
            continue
        component = card.mechanism_operation_delta.get("after_component") or (
            card.mechanism_operation_delta.get("before_component")
        )
        if not isinstance(component, Mapping):
            continue
        mechanism_id = component.get("mechanism_id")
        fields = component.get("input_fields")
        if not isinstance(mechanism_id, str) or not isinstance(fields, list):
            continue
        if (
            not fields
            or len(fields) > MAX_EXPLORATION_PROMPT_INPUT_FIELDS
            or any(
                not isinstance(field, str)
                or not field.strip()
                or len(field) > MAX_EXPLORATION_PROMPT_TEXT_CHARS
                for field in fields
            )
        ):
            # The complete card remains in the disposable index. This optional
            # non-blocking advisory is omitted when it cannot satisfy the
            # smaller prompt-facing contract.
            continue
        signature = (mechanism_id.split(":", 1)[0], tuple(sorted(str(v) for v in fields)))
        groups.setdefault(signature, []).append(card)

    advisories: list[dict[str, Any]] = []
    for (component_kind, fields), cards in groups.items():
        distinct_keys = {card.verifier_equivalence_key for card in cards}
        if len(cards) < 2 or len(distinct_keys) < 2:
            continue
        advisories.append(
            {
                "component_kind": component_kind,
                "public_input_fields": list(fields),
                "negative_trial_count": len(cards),
                "distinct_verifier_equivalence_count": len(distinct_keys),
                "latest_generation_index": max(
                    card.generation_index for card in cards
                ),
                "blocking": False,
                "reason": "repeated_negative_input_signature_variants",
                "instruction": (
                    "This public input combination has produced multiple distinct "
                    "negative mechanisms against the current head. Prefer another "
                    "input region unless the new structural hypothesis is materially "
                    "different; this advisory is not an equivalence or acceptance gate."
                ),
            }
        )
    advisories.sort(
        key=lambda value: (
            -int(value["negative_trial_count"]),
            -int(value["latest_generation_index"]),
            str(value["component_kind"]),
            tuple(value["public_input_fields"]),
        )
    )
    return advisories[:EXPLORATION_MAX_INPUT_SIGNATURE_ADVISORIES]


def _build_card(
    *,
    session_genesis_hash: str,
    generation: Mapping[str, Any],
    trial: Any,
    parent: CurrentBestEvalStats,
    task_projector: TaskExplorationProjector,
    prior_cards_by_hash: Mapping[str, ExplorationEvidenceCard],
) -> ExplorationEvidenceCard | None:
    if not isinstance(trial, Mapping):
        raise ValueError("exploration trial projection is invalid")
    status = trial.get("trial_status")
    if status == "not_started":
        return None
    candidate = trial.get("candidate")
    candidate_map = candidate if isinstance(candidate, Mapping) else {}
    delta_raw = trial.get("mechanism_operation_delta")
    delta = require_mechanism_operation_delta(delta_raw) if isinstance(delta_raw, Mapping) else None
    if not candidate_map and delta is None:
        # A provider/static failure without a verified implementation is an
        # operational fact, not a mechanism evidence card.
        return None
    if status in {"candidate_failed", "verified_not_run"}:
        assessment = TaskExplorationAssessment(
            observed_effect="unknown",
            value_label="unknown",
            public_evidence_summary={
                "status": "not_scientifically_evaluable",
                "trial_status": status,
            },
            reason_codes=("scientific_public_evidence_unavailable",),
        )
    else:
        assessment = task_projector.assess_trial(
            generation=generation,
            trial=trial,
            parent=parent,
        )
    intent = trial.get("lane_intent")
    inspired_by_trial_id: str | None = None
    inspired_by_card_hash: str | None = None
    novelty = "exact_new"
    lineage_status = "independent"
    if isinstance(intent, Mapping) and intent.get("schema_version") == EXPLORATION_LANE_INTENT_SCHEMA_VERSION:
        canonical_intent = validate_exploration_lane_intent(intent)
        if canonical_intent["mode"] == "revisit":
            inspired_by_trial_id = canonical_intent["inspired_by_trial_id"]
            inspired_by_card_hash = canonical_intent["inspired_by_card_hash"]
            source_card = prior_cards_by_hash.get(inspired_by_card_hash)
            if source_card is None or source_card.trial_id != inspired_by_trial_id:
                raise ValueError("exploration inspired-by lineage is not in prior history")
            novelty = "explicit_revisit"
            lineage_status = "inspired"
    if status == "accepted":
        lineage_status = "incorporated"
    selection = generation.get("selection")
    selected_id = selection.get("selected_trial_id") if isinstance(selection, Mapping) else None
    if selected_id == trial.get("trial_id"):
        selection_status = "selected"
    elif status == "qualified_not_selected" and selected_id is None:
        selection_status = "inconclusive"
    elif status == "qualified_not_selected":
        selection_status = "qualified_not_selected"
    elif status == "screened_only":
        selection_status = "inconclusive"
    else:
        selection_status = "not_evaluated"
    tier = _evidence_tier(trial)
    failure_scope = _failure_scope(status, assessment.observed_effect)
    disposition = _revisit_disposition(
        status=status,
        value_label=assessment.value_label,
        observed_effect=assessment.observed_effect,
        selection_status=selection_status,
    )
    equivalence_key = (
        single_component_verifier_equivalence_key(delta)
        if delta is not None
        else None
    )
    source_checksum = (
        str(candidate_map.get("program_source_checksum"))
        if candidate_map.get("program_source_checksum") is not None
        else delta.get("candidate_source_checksum") if delta is not None else None
    )
    exact_key = (
        _exact_implementation_key(
            task_id=task_projector.task_id,
            parent_current_best_hash=str(generation["parent_current_best_hash"]),
            candidate_source_checksum=source_checksum,
            delta=delta,
        )
        if source_checksum is not None and delta is not None
        else None
    )
    if (
        lineage_status == "independent"
        and exact_key is not None
        and any(card.exact_implementation_key == exact_key for card in prior_cards_by_hash.values())
    ):
        novelty = "exact_repeat"
    refs = _trial_evidence_refs(trial)
    context_hash = canonical_json_hash(
        {
            "task_id": task_projector.task_id,
            "parent_current_best_hash": generation["parent_current_best_hash"],
            "projector_contract_hash": canonical_json_hash(
                _public_mapping(task_projector.contract())
            ),
            "trial_contract_hash": trial.get("trial_contract_hash"),
            "phase_evidence_refs": refs,
        }
    )
    return ExplorationEvidenceCard(
        session_genesis_hash=session_genesis_hash,
        task_id=task_projector.task_id,
        generation_id=str(generation["generation_id"]),
        generation_index=int(generation["generation_index"]),
        trial_id=str(trial["trial_id"]),
        lane_id=str(trial["lane_id"]),
        parent_current_best_hash=str(generation["parent_current_best_hash"]),
        candidate_id=(str(candidate_map["candidate_id"]) if candidate_map.get("candidate_id") else None),
        candidate_source_checksum=source_checksum,
        mechanism_operation_delta=delta,
        verifier_equivalence_key=equivalence_key,
        exact_implementation_key=exact_key,
        trial_status=str(status),
        evidence_tier=tier,
        observed_effect=assessment.observed_effect,
        value_label=assessment.value_label,
        novelty=novelty,
        failure_scope=failure_scope,
        selection_status=selection_status,
        lineage_status=lineage_status,
        revisit_disposition=disposition,
        inspired_by_trial_id=inspired_by_trial_id,
        inspired_by_card_hash=inspired_by_card_hash,
        public_evidence_summary=assessment.public_evidence_summary,
        reason_codes=assessment.reason_codes,
        revisit_rank_key=assessment.revisit_rank_key,
        evidence_refs=refs,
        context_hash=context_hash,
    )


def _evidence_tier(trial: Mapping[str, Any]) -> str:
    if trial.get("formal_record_hash") is not None:
        return "formal"
    if trial.get("public_accept") is not None:
        return "public_complete"
    if trial.get("public_dev") is not None:
        return "public_dev"
    if trial.get("train") is not None:
        return "trained"
    return "verified"


def _failure_scope(status: Any, observed_effect: str) -> str:
    if status == "candidate_failed":
        return "authoring_or_implementation"
    if status == "verified_not_run":
        return "paired_authoring"
    if status == "formal_rejected":
        return "formal_private"
    if observed_effect == "negative":
        return "scientific_public"
    return "none"


def _revisit_disposition(
    *,
    status: Any,
    value_label: str,
    observed_effect: str,
    selection_status: str,
) -> str:
    if status == "accepted":
        return "incorporated"
    if status == "formal_rejected":
        return "cooldown"
    if status in {"candidate_failed", "verified_not_run"}:
        return "not_evaluable"
    if observed_effect == "negative":
        return "suppress_equivalent"
    if selection_status in {"qualified_not_selected", "inconclusive"}:
        if value_label == "high":
            return "priority"
        if value_label == "medium":
            return "eligible"
    return "cooldown" if value_label in {"low", "unknown"} else "eligible"


def _exact_implementation_key(
    *,
    task_id: str,
    parent_current_best_hash: str,
    candidate_source_checksum: str,
    delta: Mapping[str, Any],
) -> str:
    return canonical_json_hash(
        {
            "schema_version": "true_rl_agent.exact_mechanism_implementation.v1",
            "task_id": task_id,
            "parent_current_best_hash": parent_current_best_hash,
            "candidate_source_checksum": candidate_source_checksum,
            "mechanism_operation_delta": dict(delta),
        }
    )


def _trial_evidence_refs(trial: Mapping[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    authoring = trial.get("authoring")
    if isinstance(authoring, Mapping):
        for source, target in (
            ("prompt_payload_checksum", "prompt_payload_checksum"),
            ("authoring_evidence_checksum", "authoring_evidence_checksum"),
        ):
            value = authoring.get(source)
            if isinstance(value, str):
                refs[target] = _checksum_as_digest(value)
    for phase_name in ("public_dev", "public_accept"):
        phase = trial.get(phase_name)
        if isinstance(phase, Mapping):
            value = phase.get("phase_result_hash")
            if isinstance(value, str):
                refs[f"{phase_name}_result_hash"] = require_sha256_digest(
                    value,
                    f"exploration {phase_name} result hash",
                )
    value = trial.get("formal_record_hash")
    if isinstance(value, str):
        refs["formal_record_hash"] = require_sha256_digest(
            value,
            "exploration formal record hash",
        )
    return refs


def _checksum_as_digest(value: str) -> str:
    if value.startswith("sha256:") and len(value) == 71:
        return require_sha256_digest(value[7:], "exploration checksum digest")
    return require_sha256_digest(value, "exploration evidence digest")


def _lane_intent(
    *,
    lane_id: str,
    current_head_hash: str,
    revisit: ExplorationEvidenceCard | None,
    exclusion_set_hash: str,
) -> dict[str, Any]:
    hypotheses = (
        _revisit_lane_hypothesis_candidates(revisit)
        if revisit is not None
        else (None,)
    )
    for hypothesis in hypotheses:
        intent = {
            "schema_version": EXPLORATION_LANE_INTENT_SCHEMA_VERSION,
            "lane_id": lane_id,
            "mode": "revisit" if revisit is not None else "novel",
            "current_head_hash": current_head_hash,
            "inspired_by_trial_id": revisit.trial_id if revisit is not None else None,
            "inspired_by_card_hash": revisit.card_hash if revisit is not None else None,
            "hypothesis": hypothesis,
            "verifier_equivalence_exclusion_set_hash": exclusion_set_hash,
            "acceptance_authority": False,
        }
        try:
            return validate_exploration_lane_intent(intent)
        except ValueError as exc:
            if str(exc) != "generation lane intent exceeds its byte limit":
                raise
    raise ValueError("minimum revisit lane hypothesis exceeds generation limit")


def _revisit_lane_hypothesis_candidates(
    card: ExplorationEvidenceCard,
) -> tuple[dict[str, Any], ...]:
    """Build a bounded directive without copying the full evidence card.

    The task-owned evidence summary remains intact in the disposable index.
    The lane receives it only when the generation-owned 2 KiB envelope can
    carry it; otherwise it receives a hash plus an explicit omission marker.
    """

    if card.mechanism_operation_delta is None:
        raise ValueError("revisit lane source card has no mechanism delta")
    component = card.mechanism_operation_delta.get("after_component") or (
        card.mechanism_operation_delta.get("before_component")
    )
    if not isinstance(component, Mapping):
        raise ValueError("revisit lane source component is unavailable")
    all_input_fields = list(component["input_fields"])
    summary = dict(card.public_evidence_summary)
    summary_hash = canonical_json_hash(summary)
    instruction = (
        "Re-author this public hypothesis as one mechanism change against the "
        "current head; do not copy the old source or checkpoint."
    )

    # Prefer the useful public summary, then deterministically shed only the
    # lane preview. The complete evidence card is never altered or discarded.
    candidates: list[dict[str, Any]] = []
    seen_shapes: set[tuple[bool, tuple[str, ...]]] = set()
    for include_summary, field_limit in (
        (True, _MAX_LANE_INPUT_FIELD_PREVIEW),
        (False, _MAX_LANE_INPUT_FIELD_PREVIEW),
        (False, 4),
        (False, 2),
        (False, 0),
    ):
        shown_fields = all_input_fields[:field_limit]
        shape = (include_summary, tuple(shown_fields))
        if shape in seen_shapes:
            continue
        seen_shapes.add(shape)
        candidates.append({
            "verifier_equivalence_key": card.verifier_equivalence_key,
            "operation": card.mechanism_operation_delta["operation"],
            "operation_effect": mechanism_operation_effect(
                card.mechanism_operation_delta
            ),
            "component_kind": str(component["mechanism_id"]).split(":", 1)[0],
            "public_input_fields": shown_fields,
            "public_input_field_count": len(all_input_fields),
            "omitted_public_input_field_count": (
                len(all_input_fields) - len(shown_fields)
            ),
            "observed_effect": card.observed_effect,
            "value_label": card.value_label,
            "public_evidence_summary": summary if include_summary else None,
            "public_evidence_summary_hash": summary_hash,
            "public_evidence_summary_omitted": not include_summary,
            "instruction": instruction,
        })
    return tuple(candidates)


def _prompt_card(
    card: ExplorationEvidenceCard,
) -> dict[str, Any] | None:
    component = (
        card.mechanism_operation_delta.get("after_component")
        or card.mechanism_operation_delta.get("before_component")
        if card.mechanism_operation_delta is not None
        else None
    )
    input_fields = (
        list(component["input_fields"])
        if isinstance(component, Mapping)
        else []
    )
    if (
        len(input_fields) > MAX_EXPLORATION_PROMPT_INPUT_FIELDS
        or any(
            not isinstance(field, str)
            or not field.strip()
            or len(field) > MAX_EXPLORATION_PROMPT_TEXT_CHARS
            for field in input_fields
        )
    ):
        return None
    return _public_mapping(
        {
            "trial_id": card.trial_id,
            "card_hash": card.card_hash,
            "verifier_equivalence_key": card.verifier_equivalence_key,
            "operation": (
                card.mechanism_operation_delta.get("operation")
                if card.mechanism_operation_delta is not None
                else None
            ),
            "operation_effect": (
                mechanism_operation_effect(card.mechanism_operation_delta)
                if card.mechanism_operation_delta is not None
                else None
            ),
            "component_kind": (
                str(component["mechanism_id"]).split(":", 1)[0]
                if isinstance(component, Mapping)
                else None
            ),
            "public_input_fields": input_fields,
            "evidence_tier": card.evidence_tier,
            "observed_effect": card.observed_effect,
            "value_label": card.value_label,
            "selection_status": card.selection_status,
            "revisit_disposition": card.revisit_disposition,
            "public_evidence_summary": dict(card.public_evidence_summary),
            "reason_codes": list(card.reason_codes),
        }
    )


def _prompt_payload(
    index: ExplorationIndex,
    frontier: RevisitFrontier,
    positive: Sequence[Mapping[str, Any]],
    negative: Sequence[Mapping[str, Any]],
    input_signature_advisories: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    blocked_count = len(_blocked_verifier_equivalence_keys(index))
    return _public_mapping(
        {
            "schema_version": EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION,
            "task_id": index.task_id,
            "source_index_hash": index.index_hash,
            "current_head_hash": index.current_head_hash,
            "semantics": (
                "public exploration hypotheses only; they do not accept candidates, "
                "prove causality, or authorize checkpoint/source reuse; only priority "
                "cards may consume the automatic revisit lane"
            ),
            "priority_or_eligible_cards": list(positive),
            "verifier_equivalent_avoidance_cards": list(negative),
            "repeated_negative_input_signature_advisories": list(
                input_signature_advisories
            ),
            "blocked_verifier_equivalence_count": blocked_count,
            "omitted_verifier_equivalent_avoidance_count": max(
                0,
                blocked_count - len(negative),
            ),
            "frontier_backlog_count": frontier.backlog_count,
            "omitted_frontier_card_count": frontier.backlog_count - len(positive),
            "deferred_frontier_card_count": len(frontier.deferred_cards),
            "ineligible_frontier_card_count": len(frontier.ineligible_cards),
            "acceptance_authority": False,
        }
    )


def _public_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("public exploration value must be an object")
    clean = sanitize_prompt_public_value(dict(value))
    if not isinstance(clean, Mapping):
        raise ValueError("public exploration value was not preserved")
    return dict(clean)


def _hash_mapping(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label}s must be an object")
    result: dict[str, str] = {}
    for key, raw in value.items():
        _bounded_text(key, f"{label} name")
        result[str(key)] = require_sha256_digest(raw, label)
    return result


def _bounded_text_tuple(value: Any, label: str, maximum: int) -> tuple[str, ...]:
    if not isinstance(value, tuple | list) or len(value) > maximum:
        raise ValueError(f"{label}s are invalid")
    result = tuple(_bounded_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label}s contain duplicates")
    return result


def _revisit_rank_key(value: Any) -> tuple[int, ...]:
    """Validate an opaque, task-owned key used only for revisit ordering."""

    if not isinstance(value, tuple | list) or len(value) > 4:
        raise ValueError("exploration revisit rank key is invalid")
    result: list[int] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or not -1_000_000_000 <= item <= 1_000_000_000
        ):
            raise ValueError("exploration revisit rank key item is invalid")
        result.append(item)
    return tuple(result)


def _bounded_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError(f"{label} must be a non-empty bounded string")
    return value


def _exact_mapping(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise ValueError(f"{label} fields mismatch")
    return dict(value)


__all__ = [
    "EXPLORATION_EVIDENCE_CARD_SCHEMA_VERSION",
    "EXPLORATION_INDEX_SCHEMA_VERSION",
    "EXPLORATION_LANE_INTENT_SCHEMA_VERSION",
    "EXPLORATION_MAX_FRONTIER_CARDS",
    "EXPLORATION_PROMPT_MAX_BYTES",
    "EXPLORATION_PROMPT_PROJECTION_SCHEMA_VERSION",
    "REVISIT_FRONTIER_SCHEMA_VERSION",
    "ExplorationEvidenceCard",
    "ExplorationIndex",
    "GenerationExplorationPlan",
    "RevisitFrontier",
    "TaskExplorationAssessment",
    "TaskExplorationProjector",
    "build_exploration_index",
    "build_exploration_prompt_projection",
    "build_generation_exploration_plan",
    "build_revisit_frontier",
    "freeze_exploration_lane_intents",
    "validate_exploration_lane_intent",
]
