"""Canonical fixed-width generation records and exactly-once ledger commit."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_LANE_IDS,
    GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
    GENERATION_WIDTH,
    GenerationLaneAssignment,
    GenerationManifest,
    require_generation_lane_intent,
    require_generation_trial_status,
)
from true_rl_agent.agent_core.generation_receipts import (
    GenerationReceiptJournal,
    GenerationReceiptRef,
    load_generation_manifest_from_receipt,
    load_generation_selection_receipt,
)
from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.agent_core.ledger import (
    PROVIDER_AUTHORING_MODE,
    append_ledger_record_once,
    dispatch_ledger_record_schema,
    load_ledger,
)
from true_rl_agent.agent_core.trial_receipts import (
    LoadedTrialEvalPhaseReceipt,
    LoadedTrialTrainReceipt,
    TrialReceiptRef,
    load_trial_eval_phase_receipt,
    load_trial_execution_receipt,
    load_trial_train_receipt,
)
from true_rl_agent.candidates.sibling_exclusion import (
    build_single_component_semantic_exclusion_projection,
    validate_single_component_semantic_exclusion_projection,
)
from true_rl_agent.evaluation.acceptance_protocol import AcceptanceProtocol
from true_rl_agent.evaluation.acceptance_replay import replay_acceptance_ledger
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.prompt_artifacts import (
    AuthoringEvidenceRef,
    PromptPayloadRef,
    authoring_outcome_from_evidence,
    verify_authoring_evidence,
    verify_prompt_payload,
)


GENERATION_TRIAL_RECORD_SCHEMA_VERSION = "true_rl_agent.generation_trial.v1"
GENERATION_PHASE_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.generation_phase_projection.v1"
)
GENERATION_FORMAL_PROJECTION_SCHEMA_VERSION = (
    "true_rl_agent.generation_formal_projection.v1"
)
GENERATION_OUTCOMES = frozenset(
    {
        "paired_authoring_incomplete",
        "candidate_execution_incomplete",
        "no_selection",
        "formal_rejected",
        "accepted",
    }
)

_GENERATION_FIELDS = {
    "schema_version",
    "record_kind",
    "authoring_mode",
    "generation_id",
    "generation_index",
    "session_genesis_hash",
    "ledger_epoch_hash",
    "generation_session_contract_hash",
    "manifest",
    "manifest_receipt_checksum",
    "parent_current_best",
    "parent_current_best_hash",
    "trials",
    "selection",
    "selection_receipt_checksum",
    "formal",
    "generation_outcome",
    "resulting_current_best",
    "resulting_current_best_hash",
    "private_artifacts",
}
_TRIAL_FIELDS = {
    "schema_version",
    "lane_id",
    "trial_id",
    "trial_status",
    "status_reason",
    "authoring",
    "candidate",
    "verifier",
    "candidate_surface",
    "mechanism_operation_delta",
    "trial_contract_hash",
    "train",
    "public_dev",
    "public_accept",
    "execution_record_hash",
    "execution_receipt_checksum",
    "formal_record_hash",
}
_AUTHORING_FIELDS = {
    "base_public_context_hash",
    "lane_intent_hash",
    "prompt_payload_checksum",
    "authoring_evidence_checksum",
    "sibling_exclusion_hash",
}
_TRAIN_FIELDS = {
    "job_spec_hash",
    "train_result_hash",
    "train_receipt_checksum",
    "checkpoint_hash",
    "execution_device",
    "resource_binding_hash",
}
_PHASE_FIELDS = {
    "schema_version",
    "phase",
    "job_spec_hash",
    "evaluation_plan_hash",
    "phase_result_hash",
    "phase_receipt_checksum",
    "checkpoint_hash",
    "execution_device",
    "resource_binding_hash",
    "task_evidence_hash",
    "public_summary",
}


def _load_bound_generation_manifest(
    record: Mapping[str, Any],
) -> tuple[GenerationManifest, GenerationReceiptRef]:
    checksum = require_sha256_checksum(
        record.get("manifest_receipt_checksum"),
        "generation manifest receipt checksum",
    )
    manifest_ref = _find_private_generation_receipt(
        record.get("private_artifacts"),
        artifact_kind="generation_manifest_receipt",
        receipt_kind="manifest",
        checksum=checksum,
    )
    manifest = load_generation_manifest_from_receipt(manifest_ref)
    if record.get("manifest") != manifest.to_public_dict():
        raise ValueError("generation manifest public projection mismatch")
    return manifest, manifest_ref


def load_bound_generation_manifest(
    record: Mapping[str, Any],
) -> GenerationManifest:
    """Load the private manifest exactly bound by a public generation record."""

    if not isinstance(record, Mapping):
        raise ValueError("generation ledger record must be an object")
    manifest, _ref = _load_bound_generation_manifest(record)
    return manifest


def validate_generation_ledger_record(
    record: Mapping[str, Any],
    *,
    acceptance_protocol: AcceptanceProtocol | None = None,
    expected_parent: CurrentBestEvalStats | None = None,
    expected_generation_session_contract_hash: str | None = None,
) -> CurrentBestEvalStats:
    """Validate one generation and return its recomputed resulting head."""

    if not isinstance(record, Mapping) or set(record) != _GENERATION_FIELDS:
        raise ValueError("generation ledger record fields mismatch")
    if (
        record.get("schema_version") != GENERATION_LEDGER_RECORD_SCHEMA_VERSION
        or record.get("record_kind") != "generation"
        or record.get("authoring_mode") != PROVIDER_AUTHORING_MODE
    ):
        raise ValueError("generation ledger record contract mismatch")
    generation_id = _required_text(record.get("generation_id"), "generation_id")
    generation_index = _positive_int(
        record.get("generation_index"), "generation_index"
    )
    session_hash = require_sha256_digest(
        record.get("session_genesis_hash"), "generation session_genesis_hash"
    )
    ledger_epoch_hash = require_sha256_digest(
        record.get("ledger_epoch_hash"), "generation ledger_epoch_hash"
    )
    session_contract_hash = require_sha256_digest(
        record.get("generation_session_contract_hash"),
        "generation session contract hash",
    )
    if (
        expected_generation_session_contract_hash is not None
        and session_contract_hash
        != require_sha256_digest(
            expected_generation_session_contract_hash,
            "expected generation session contract hash",
        )
    ):
        raise ValueError("generation session contract changed within the ledger")
    manifest, manifest_ref = _load_bound_generation_manifest(record)
    if (
        manifest.generation_id != generation_id
        or manifest.generation_index != generation_index
        or manifest.session_genesis_hash != session_hash
        or manifest.ledger_epoch_hash != ledger_epoch_hash
        or manifest.generation_session_contract_hash != session_contract_hash
    ):
        raise ValueError("generation record/manifest identity mismatch")
    journal = GenerationReceiptJournal(
        Path(manifest_ref.path).resolve(strict=False).parent.parent,
        manifest=manifest,
    )

    parent = CurrentBestEvalStats.from_mapping(record.get("parent_current_best"))
    parent_hash = canonical_json_hash(parent.to_public_dict())
    if (
        record.get("parent_current_best_hash") != parent_hash
        or manifest.current_best_evidence_hash != parent_hash
        or manifest.parent_program_source_checksum
        != parent.program_source_checksum
        or manifest.parent_checkpoint_hash != parent.checkpoint_hash
    ):
        raise ValueError("generation frozen parent binding mismatch")
    if expected_parent is not None and parent != expected_parent:
        raise ValueError("generation parent is not the replay current-best")

    raw_trials = record.get("trials")
    if not isinstance(raw_trials, list) or len(raw_trials) != GENERATION_WIDTH:
        raise ValueError("generation requires exactly two trials")
    trials = [_validate_trial(item) for item in raw_trials]
    if tuple(item["lane_id"] for item in trials) != GENERATION_LANE_IDS:
        raise ValueError("generation trials must be ordered lane-a/lane-b")
    if len({item["trial_id"] for item in trials}) != GENERATION_WIDTH:
        raise ValueError("generation trial IDs must be unique")
    assignments = {item.lane_id: item for item in manifest.lanes}
    for trial in trials:
        authoring = trial["authoring"]
        if trial["trial_status"] == "not_started":
            if authoring is not None or trial["candidate"] is not None:
                raise ValueError("not-started generation trial contains candidate evidence")
        else:
            if not isinstance(authoring, Mapping):
                raise ValueError("started generation trial lacks authoring binding")
            assignment = assignments[trial["lane_id"]]
            if (
                authoring["base_public_context_hash"]
                != manifest.base_public_context_hash
                or authoring["lane_intent_hash"] != assignment.lane_intent_hash
            ):
                raise ValueError("generation trial authoring/manifest binding mismatch")
            if trial["lane_id"] == GENERATION_LANE_IDS[0]:
                if authoring["sibling_exclusion_hash"] is not None:
                    raise ValueError("generation lane A cannot carry sibling exclusion")
            elif authoring["sibling_exclusion_hash"] is None:
                raise ValueError("generation lane B must bind sibling exclusion")
            _load_bound_authoring_evidence(
                trial,
                private_artifacts=record.get("private_artifacts"),
            )
        assignment = assignments[trial["lane_id"]]
        for projection_name in ("train", "public_dev", "public_accept"):
            projection = trial.get(projection_name)
            if projection is not None and (
                projection.get("execution_device") != assignment.execution_device
                or projection.get("resource_binding_hash")
                != assignment.resource_binding_hash
            ):
                raise ValueError(
                    "generation trial resource projection differs from manifest"
                )
        if trial["execution_record_hash"] is not None:
            _load_bound_trial_execution(
                trial,
                private_artifacts=record.get("private_artifacts"),
                session_genesis_hash=session_hash,
                ledger_epoch_hash=ledger_epoch_hash,
            )
        _load_bound_train_projection(
            trial,
            assignment=assignment,
            private_artifacts=record.get("private_artifacts"),
        )
        for phase in (
            EvaluationPhase.PUBLIC_DEV,
            EvaluationPhase.PUBLIC_ACCEPT,
        ):
            if trial[phase.value] is not None:
                _load_bound_phase_projection(
                    trial,
                    assignment=assignment,
                    phase=phase,
                    private_artifacts=record.get("private_artifacts"),
                )
    if trials[1]["trial_status"] != "not_started":
        delta = trials[0].get("mechanism_operation_delta")
        if not isinstance(delta, Mapping):
            raise ValueError("generation lane B has no authoritative lane A exclusion")
        expected_exclusion = build_single_component_semantic_exclusion_projection(
            delta
        )
        if trials[1]["authoring"]["sibling_exclusion_hash"] != expected_exclusion[
            "exclusion_hash"
        ]:
            raise ValueError("generation sibling exclusion does not match lane A delta")

    selection = record.get("selection")
    selection_receipt_checksum = record.get("selection_receipt_checksum")
    if selection is None:
        if selection_receipt_checksum is not None:
            raise ValueError("generation has a selection receipt without selection")
        if journal.load_selection() is not None:
            raise ValueError("generation omits an existing bound selection receipt")
        selected_trial_id = None
    else:
        if not isinstance(selection, Mapping):
            raise ValueError("generation selection must be an object or null")
        assert_public_projection_safe(selection)
        require_sha256_checksum(
            selection_receipt_checksum,
            "generation selection receipt checksum",
        )
        selection_ref = _find_private_generation_receipt(
            record.get("private_artifacts"),
            artifact_kind="generation_selection_receipt",
            receipt_kind="selection",
            checksum=selection_receipt_checksum,
        )
        loaded_selection = load_generation_selection_receipt(
            selection_ref,
            manifest=manifest,
        )
        if dict(loaded_selection.selection) != dict(selection):
            raise ValueError("generation selection differs from sealed receipt")
        _validate_selection_receipt_bindings(
            loaded_selection,
            trials=trials,
            manifest=manifest,
            private_artifacts=record.get("private_artifacts"),
        )
        if selection.get("acceptance_authority") is not False:
            raise ValueError("generation selection has acceptance authority")
        selected_trial_id = selection.get("selected_trial_id")
        if selected_trial_id is not None and selected_trial_id not in {
            item["trial_id"] for item in trials
        }:
            raise ValueError("generation selection names an unknown trial")

    formal = record.get("formal")
    formal_round: Mapping[str, Any] | None = None
    private_artifacts = record.get("private_artifacts")
    sealed_phase_artifacts = [
        item
        for item in private_artifacts
        if isinstance(item, Mapping)
        and isinstance(item.get("kind"), str)
        and item["kind"].endswith("_sealed_validation_phase_receipt")
    ] if isinstance(private_artifacts, list) else []
    if formal is None:
        if any(item["formal_record_hash"] is not None for item in trials):
            raise ValueError("generation trial names formal evidence without formal result")
        if sealed_phase_artifacts:
            raise ValueError("generation has sealed evidence without a formal result")
    else:
        if len(sealed_phase_artifacts) != 1:
            raise ValueError("generation formal result requires one sealed phase receipt")
        formal_round = _load_and_validate_formal(
            formal,
            manifest=manifest,
            private_artifacts=private_artifacts,
            selected_trial_id=selected_trial_id,
            trials=trials,
            session_genesis_hash=session_hash,
            ledger_epoch_hash=ledger_epoch_hash,
        )

    outcome = record.get("generation_outcome")
    if outcome not in GENERATION_OUTCOMES:
        raise ValueError("generation outcome is invalid")
    _validate_status_relationships(
        trials=trials,
        selection_present=selection is not None,
        selected_trial_id=selected_trial_id,
        formal_round=formal_round,
        outcome=str(outcome),
    )

    result_head = parent
    if formal_round is not None:
        if acceptance_protocol is None:
            acceptance_protocol = AcceptanceProtocol.from_mapping(
                formal_round.get("acceptance_protocol")
            )
        replay = replay_acceptance_ledger(
            [formal_round],
            initial_current_best=parent,
            expected_session_genesis_hash=session_hash,
            acceptance_protocol=acceptance_protocol,
            expected_ledger_epoch_hash=ledger_epoch_hash,
            expected_authoring_mode=PROVIDER_AUTHORING_MODE,
        )
        replay.raise_for_continuation()
        result_head = replay.current_best
    declared_result = CurrentBestEvalStats.from_mapping(
        record.get("resulting_current_best")
    )
    result_hash = canonical_json_hash(result_head.to_public_dict())
    if (
        declared_result != result_head
        or record.get("resulting_current_best_hash") != result_hash
    ):
        raise ValueError("generation resulting current-best projection mismatch")
    _validate_private_artifacts(record.get("private_artifacts"))
    assert_public_projection_safe(
        {
            "manifest": manifest.to_public_dict(),
            "parent_current_best": parent.to_public_dict(),
            "trials": trials,
            "selection": selection,
            "formal": formal,
            "generation_outcome": outcome,
            "resulting_current_best": result_head.to_public_dict(),
        }
    )
    return result_head


def project_generation_formal_round_record(
    record: Mapping[str, Any],
    *,
    acceptance_protocol: AcceptanceProtocol | None = None,
    expected_parent: CurrentBestEvalStats | None = None,
) -> Mapping[str, Any] | None:
    """Recover the sole formal round-v7 record from its bound P2 receipt."""

    validate_generation_ledger_record(
        record,
        acceptance_protocol=acceptance_protocol,
        expected_parent=expected_parent,
    )
    formal = record.get("formal")
    if formal is None:
        return None
    trials = [_validate_trial(item) for item in record["trials"]]
    return _load_and_validate_formal(
        formal,
        manifest=load_bound_generation_manifest(record),
        private_artifacts=record.get("private_artifacts"),
        selected_trial_id=record["selection"]["selected_trial_id"],
        trials=trials,
        session_genesis_hash=record["session_genesis_hash"],
        ledger_epoch_hash=record["ledger_epoch_hash"],
    )


def project_generation_public_history_record(
    record: Mapping[str, Any],
    *,
    public_decision: Mapping[str, Any],
    acceptance_protocol: AcceptanceProtocol,
    expected_parent: CurrentBestEvalStats,
) -> dict[str, Any]:
    """Build the one deterministic round-compatible view used by consumers.

    The generation ledger remains authoritative.  This view exists so the
    established source catalog and memory pipeline can consume a promoted
    sibling without inventing a second persisted winner record.
    """

    result_head = validate_generation_ledger_record(
        record,
        acceptance_protocol=acceptance_protocol,
        expected_parent=expected_parent,
    )
    formal = project_generation_formal_round_record(
        record,
        acceptance_protocol=acceptance_protocol,
        expected_parent=expected_parent,
    )
    public_trials: list[dict[str, Any]] = []
    for raw_trial in record["trials"]:
        trial = _public_copy(raw_trial)
        if trial.get("trial_status") != "not_started":
            _evidence, lane_intent = _load_bound_authoring_evidence(
                raw_trial,
                private_artifacts=record.get("private_artifacts"),
            )
            trial["lane_intent"] = lane_intent
        else:
            trial["lane_intent"] = None
        public_trials.append(trial)
    guard_reference = None
    if record.get("selection") is not None:
        selection_ref = _find_private_generation_receipt(
            record.get("private_artifacts"),
            artifact_kind="generation_selection_receipt",
            receipt_kind="selection",
            checksum=record.get("selection_receipt_checksum"),
        )
        loaded_selection = load_generation_selection_receipt(
            selection_ref,
            manifest=load_bound_generation_manifest(record),
        )
        guard_reference = _public_copy(loaded_selection.guard_reference)
    generation_summary = {
        "schema_version": GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
        "generation_id": record["generation_id"],
        "generation_index": record["generation_index"],
        "generation_outcome": record["generation_outcome"],
        "parent_current_best_hash": record["parent_current_best_hash"],
        "parent_current_best": _public_copy(record["parent_current_best"]),
        "resulting_current_best_hash": record["resulting_current_best_hash"],
        "trials": public_trials,
        "selection": _public_copy(record["selection"]),
        "guard_reference": guard_reference,
        "formal": _public_copy(record["formal"]),
    }
    if formal is None:
        projected: dict[str, Any] = {
            "schema_version": GENERATION_LEDGER_RECORD_SCHEMA_VERSION,
            "record_kind": "generation",
            "round_id": record["generation_id"],
            "generation_id": record["generation_id"],
            "generation_index": record["generation_index"],
            "session_genesis_hash": record["session_genesis_hash"],
            "ledger_epoch_hash": record["ledger_epoch_hash"],
            "authoring_mode": record["authoring_mode"],
        }
    else:
        projected = _public_copy(formal)
        projected.pop("private_artifacts", None)
        projected.pop("feedback_safe", None)
        candidate = projected.get("candidate")
        if isinstance(candidate, dict):
            candidate.pop("path", None)
            candidate.pop("candidate_path", None)
        projected["record_kind"] = "generation"
        projected["generation_id"] = record["generation_id"]
        projected["generation_index"] = record["generation_index"]
    projected["decision"] = dict(public_decision)
    projected["generation"] = generation_summary
    if canonical_json_hash(result_head.to_public_dict()) != record[
        "resulting_current_best_hash"
    ]:
        raise ValueError("generation public history head projection mismatch")
    projected = sanitize_prompt_public_value(projected)
    assert_public_projection_safe(projected)
    return projected


def commit_generation_record(
    *,
    ledger_path: Path,
    record: Mapping[str, Any],
    initial_current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
) -> bool:
    """Validate a snapshot, then CAS-append exactly one generation.

    Receipt-backed replay deliberately happens before the ledger write lock is
    acquired.  The lock-time callback only proves that the ledger snapshot has
    not changed, so slow filesystem validation cannot block another writer.
    """

    record_dict = dict(record)
    snapshot = tuple(load_ledger(ledger_path))
    _validate_generation_commit_snapshot(
        snapshot,
        incoming=record_dict,
        initial_current_best=initial_current_best,
        acceptance_protocol=acceptance_protocol,
    )

    def _pre_append(
        existing: Sequence[Mapping[str, Any]],
        incoming: Mapping[str, Any],
    ) -> None:
        del incoming
        if tuple(existing) != snapshot:
            raise ValueError("generation ledger changed after commit prevalidation")

    return append_ledger_record_once(
        ledger_path,
        record_dict,
        pre_append_validator=_pre_append,
    )


def _validate_generation_commit_snapshot(
    existing: Sequence[Mapping[str, Any]],
    *,
    incoming: Mapping[str, Any],
    initial_current_best: CurrentBestEvalStats,
    acceptance_protocol: AcceptanceProtocol,
) -> None:
    """Replay one immutable snapshot and validate a new slot outside the lock."""

    current = initial_current_best
    seen_ids: set[str] = set()
    session_contract_hash = require_sha256_digest(
        incoming.get("generation_session_contract_hash"),
        "incoming generation session contract hash",
    )
    for expected_index, item in enumerate(existing, start=1):
        descriptor = dispatch_ledger_record_schema(item)
        if descriptor.record_kind != "generation":
            raise ValueError("generation ledger cannot contain round records")
        if item.get("generation_index") != expected_index:
            raise ValueError("generation ledger index sequence mismatch")
        generation_id = _required_text(item.get("generation_id"), "generation_id")
        if generation_id in seen_ids:
            raise ValueError("generation ledger contains duplicate generation IDs")
        seen_ids.add(generation_id)
        current = validate_generation_ledger_record(
            item,
            acceptance_protocol=acceptance_protocol,
            expected_parent=current,
            expected_generation_session_contract_hash=session_contract_hash,
        )

    incoming_id = _required_text(incoming.get("generation_id"), "generation_id")
    if incoming_id in seen_ids:
        # append_ledger_record_once owns the exact-idempotence/conflict decision.
        # The existing snapshot has still been fully replayed above.
        return
    if incoming.get("generation_index") != len(existing) + 1:
        raise ValueError("incoming generation index is not the next ledger slot")
    validate_generation_ledger_record(
        incoming,
        acceptance_protocol=acceptance_protocol,
        expected_parent=current,
        expected_generation_session_contract_hash=session_contract_hash,
    )


def load_bound_generation_trial_phase_receipt(
    record: Mapping[str, Any],
    *,
    trial_id: str,
    phase: EvaluationPhase,
) -> LoadedTrialEvalPhaseReceipt:
    """Reload one trial phase from the immutable receipt named by a generation.

    Callers that use the result for lineage must first validate the full
    generation against its replay parent. This helper deliberately owns no
    task-specific projection semantics.
    """

    if not isinstance(record, Mapping) or set(record) != _GENERATION_FIELDS:
        raise ValueError("generation ledger record fields mismatch")
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("generation trial_id is required")
    if not isinstance(phase, EvaluationPhase):
        raise TypeError("generation phase is invalid")
    raw_trials = record.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("generation trials are invalid")
    matches = [
        _validate_trial(item)
        for item in raw_trials
        if isinstance(item, Mapping) and item.get("trial_id") == trial_id
    ]
    if len(matches) != 1:
        raise ValueError("generation phase trial is not uniquely identified")
    manifest = load_bound_generation_manifest(record)
    assignments = {item.lane_id: item for item in manifest.lanes}
    assignment = assignments[matches[0]["lane_id"]]
    return _load_bound_phase_projection(
        matches[0],
        assignment=assignment,
        phase=phase,
        private_artifacts=record.get("private_artifacts"),
    )


def load_bound_generation_trial_train_receipt(
    record: Mapping[str, Any],
    *,
    trial_id: str,
) -> LoadedTrialTrainReceipt:
    """Reload one trial train result exactly bound by a generation record.

    The helper owns only the generic generation/private-receipt boundary. A
    caller that gives the result task-specific meaning must first replay and
    validate the generation chain that selected the trial.
    """

    if not isinstance(record, Mapping) or set(record) != _GENERATION_FIELDS:
        raise ValueError("generation ledger record fields mismatch")
    if not isinstance(trial_id, str) or not trial_id:
        raise ValueError("generation trial_id is required")
    raw_trials = record.get("trials")
    if not isinstance(raw_trials, list):
        raise ValueError("generation trials are invalid")
    matches = [
        _validate_trial(item)
        for item in raw_trials
        if isinstance(item, Mapping) and item.get("trial_id") == trial_id
    ]
    if len(matches) != 1:
        raise ValueError("generation train trial is not uniquely identified")
    manifest = load_bound_generation_manifest(record)
    assignments = {item.lane_id: item for item in manifest.lanes}
    loaded = _load_bound_train_projection(
        matches[0],
        assignment=assignments[matches[0]["lane_id"]],
        private_artifacts=record.get("private_artifacts"),
    )
    if loaded is None:
        raise ValueError("generation train projection is missing")
    return loaded


def _validate_trial(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TRIAL_FIELDS:
        raise ValueError("generation trial fields mismatch")
    trial = dict(value)
    if trial.get("schema_version") != GENERATION_TRIAL_RECORD_SCHEMA_VERSION:
        raise ValueError("generation trial schema mismatch")
    if trial.get("lane_id") not in GENERATION_LANE_IDS:
        raise ValueError("generation trial lane is invalid")
    _required_text(trial.get("trial_id"), "trial_id")
    require_generation_trial_status(trial.get("trial_status"))
    _required_text(trial.get("status_reason"), "trial status_reason")
    authoring = trial.get("authoring")
    if authoring is not None:
        if not isinstance(authoring, Mapping) or set(authoring) != _AUTHORING_FIELDS:
            raise ValueError("generation trial authoring fields mismatch")
        for field_name in ("base_public_context_hash", "lane_intent_hash"):
            require_sha256_digest(
                authoring.get(field_name), f"generation authoring {field_name}"
            )
        for field_name in ("prompt_payload_checksum", "authoring_evidence_checksum"):
            require_sha256_checksum(
                authoring.get(field_name), f"generation authoring {field_name}"
            )
        if authoring.get("sibling_exclusion_hash") is not None:
            require_sha256_digest(
                authoring.get("sibling_exclusion_hash"),
                "generation authoring sibling_exclusion_hash",
            )
    for field_name in ("candidate", "verifier", "candidate_surface"):
        value = trial.get(field_name)
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"generation trial {field_name} must be an object or null")
    candidate = trial.get("candidate")
    if isinstance(candidate, Mapping) and any(
        key in candidate for key in ("path", "candidate_path")
    ):
        raise ValueError("generation nested candidate contains a private path")
    delta = trial.get("mechanism_operation_delta")
    if delta is not None and not isinstance(delta, Mapping):
        raise ValueError("generation mechanism delta must be an object or null")
    contract_hash = trial.get("trial_contract_hash")
    if contract_hash is not None:
        require_sha256_digest(contract_hash, "generation trial contract hash")
    train = trial.get("train")
    if train is not None:
        if not isinstance(train, Mapping) or set(train) != _TRAIN_FIELDS:
            raise ValueError("generation train projection fields mismatch")
        for hash_field in ("job_spec_hash", "train_result_hash"):
            require_sha256_digest(train.get(hash_field), f"generation train {hash_field}")
        require_sha256_checksum(
            train.get("train_receipt_checksum"),
            "generation train receipt checksum",
        )
        require_sha256_checksum(
            train.get("checkpoint_hash"), "generation train checkpoint hash"
        )
        _required_text(train.get("execution_device"), "generation train execution_device")
        if train.get("resource_binding_hash") is not None:
            require_sha256_digest(
                train.get("resource_binding_hash"),
                "generation train resource_binding_hash",
            )
    for phase_name in ("public_dev", "public_accept"):
        phase = trial.get(phase_name)
        if phase is not None:
            _validate_phase_projection(phase, expected_phase=phase_name)
    execution_hash = trial.get("execution_record_hash")
    execution_checksum = trial.get("execution_receipt_checksum")
    if (execution_hash is None) != (execution_checksum is None):
        raise ValueError("generation trial execution hash/checksum must be paired")
    if execution_hash is not None:
        require_sha256_digest(execution_hash, "generation execution record hash")
        require_sha256_checksum(
            execution_checksum, "generation execution receipt checksum"
        )
    formal_hash = trial.get("formal_record_hash")
    if formal_hash is not None:
        require_sha256_digest(formal_hash, "generation formal record hash")
        if formal_hash != execution_hash:
            raise ValueError("generation formal record must be the trial execution record")
    assert_public_projection_safe(trial)
    return trial


def _validate_phase_projection(value: Any, *, expected_phase: str) -> None:
    if not isinstance(value, Mapping) or set(value) != _PHASE_FIELDS:
        raise ValueError("generation phase projection fields mismatch")
    if (
        value.get("schema_version") != GENERATION_PHASE_PROJECTION_SCHEMA_VERSION
        or value.get("phase") != expected_phase
    ):
        raise ValueError("generation phase projection contract mismatch")
    for field_name in (
        "job_spec_hash",
        "evaluation_plan_hash",
        "phase_result_hash",
        "task_evidence_hash",
    ):
        require_sha256_digest(value.get(field_name), f"generation phase {field_name}")
    for field_name in ("phase_receipt_checksum", "checkpoint_hash"):
        require_sha256_checksum(value.get(field_name), f"generation phase {field_name}")
    _required_text(value.get("execution_device"), "generation phase execution_device")
    if value.get("resource_binding_hash") is not None:
        require_sha256_digest(
            value.get("resource_binding_hash"),
            "generation phase resource_binding_hash",
        )
    if not isinstance(value.get("public_summary"), Mapping):
        raise ValueError("generation public phase summary must be an object")


def _load_and_validate_formal(
    value: Any,
    *,
    manifest: GenerationManifest,
    private_artifacts: Any,
    selected_trial_id: Any,
    trials: Sequence[Mapping[str, Any]],
    session_genesis_hash: str,
    ledger_epoch_hash: str,
) -> Mapping[str, Any]:
    expected_fields = {
        "schema_version",
        "selected_lane_id",
        "selected_trial_id",
        "trial_execution_receipt_checksum",
        "round_record_hash",
        "sealed_phase_receipt_checksum",
        "decision_outcome",
        "accepted",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise ValueError("generation formal projection fields mismatch")
    if value.get("schema_version") != GENERATION_FORMAL_PROJECTION_SCHEMA_VERSION:
        raise ValueError("generation formal projection schema mismatch")
    selected_lane_id = value.get("selected_lane_id")
    if (
        selected_lane_id not in GENERATION_LANE_IDS
        or value.get("selected_trial_id") != selected_trial_id
    ):
        raise ValueError("generation formal selection binding mismatch")
    selected_trial = next(
        (item for item in trials if item["lane_id"] == selected_lane_id), None
    )
    if selected_trial is None or selected_trial["trial_id"] != selected_trial_id:
        raise ValueError("generation formal trial binding mismatch")
    selected_assignment = next(
        item for item in manifest.lanes if item.lane_id == selected_lane_id
    )
    expected_gpu_process_binding = (
        selected_assignment.gpu_process_binding.to_dict()
        if selected_assignment.gpu_process_binding is not None
        else None
    )
    checksum = require_sha256_checksum(
        value.get("trial_execution_receipt_checksum"),
        "generation formal execution receipt checksum",
    )
    require_sha256_digest(value.get("round_record_hash"), "generation round record hash")
    sealed_checksum = require_sha256_checksum(
        value.get("sealed_phase_receipt_checksum"),
        "generation sealed phase receipt checksum",
    )
    ref = _find_private_execution_receipt(
        private_artifacts,
        lane_id=selected_lane_id,
        checksum=checksum,
    )
    sealed_ref = _find_private_trial_receipt(
        private_artifacts,
        artifact_kind=f"{selected_lane_id}_sealed_validation_phase_receipt",
        receipt_kind="eval_phase",
        checksum=sealed_checksum,
    )
    loaded_sealed = load_trial_eval_phase_receipt(sealed_ref)
    contract, loaded = load_trial_execution_receipt(ref)
    round_record = dict(loaded.record)
    if (
        canonical_json_hash(round_record) != value.get("round_record_hash")
        or selected_trial.get("formal_record_hash") != value.get("round_record_hash")
        or selected_trial.get("execution_record_hash") != value.get("round_record_hash")
        or selected_trial.get("execution_receipt_checksum") != checksum
        or contract.get("round_id") != selected_trial_id
        or contract.get("session_genesis_hash") != session_genesis_hash
        or contract.get("ledger_epoch_hash") != ledger_epoch_hash
        or round_record.get("session_genesis_hash") != session_genesis_hash
        or round_record.get("ledger_epoch_hash") != ledger_epoch_hash
        or loaded_sealed.result.phase is not EvaluationPhase.SEALED_VALIDATION
        or loaded_sealed.result.execution_device
        != selected_trial["train"]["execution_device"]
        or loaded_sealed.result.resource_binding_hash
        != selected_trial["train"]["resource_binding_hash"]
        or loaded_sealed.result.gpu_process_binding
        != expected_gpu_process_binding
        or canonical_json_hash(loaded_sealed.contract)
        != selected_trial.get("trial_contract_hash")
    ):
        raise ValueError("generation formal round receipt binding mismatch")
    decision = round_record.get("decision")
    if not isinstance(decision, Mapping):
        raise ValueError("generation formal round has no decision")
    if (
        value.get("decision_outcome") != decision.get("outcome")
        or value.get("accepted") is not decision.get("accepted")
    ):
        raise ValueError("generation formal decision projection mismatch")
    return round_record


def _load_bound_trial_execution(
    trial: Mapping[str, Any],
    *,
    private_artifacts: Any,
    session_genesis_hash: str,
    ledger_epoch_hash: str,
) -> Mapping[str, Any]:
    checksum = require_sha256_checksum(
        trial.get("execution_receipt_checksum"),
        "generation trial execution receipt checksum",
    )
    ref = _find_private_execution_receipt(
        private_artifacts,
        lane_id=str(trial["lane_id"]),
        checksum=checksum,
    )
    contract, loaded = load_trial_execution_receipt(ref)
    record = dict(loaded.record)
    if (
        canonical_json_hash(record) != trial.get("execution_record_hash")
        or contract.get("round_id") != trial.get("trial_id")
        or contract.get("session_genesis_hash") != session_genesis_hash
        or contract.get("ledger_epoch_hash") != ledger_epoch_hash
    ):
        raise ValueError("generation trial execution receipt binding mismatch")
    return record


def _load_bound_train_projection(
    trial: Mapping[str, Any],
    *,
    assignment: GenerationLaneAssignment,
    private_artifacts: Any,
) -> LoadedTrialTrainReceipt | None:
    projection = trial.get("train")
    if projection is None:
        return None
    ref = _find_private_trial_receipt(
        private_artifacts,
        artifact_kind=f"{trial['lane_id']}_train_receipt",
        receipt_kind="train",
        checksum=projection.get("train_receipt_checksum"),
    )
    loaded = load_trial_train_receipt(ref)
    result = loaded.result
    if (
        canonical_json_hash(loaded.contract) != trial.get("trial_contract_hash")
        or loaded.contract.get("mechanism_operation_delta")
        != trial.get("mechanism_operation_delta")
        or loaded.job_spec_hash != projection.get("job_spec_hash")
        or canonical_json_hash(result.to_public_dict())
        != projection.get("train_result_hash")
        or result.checkpoint.checksum != projection.get("checkpoint_hash")
        or result.execution_device != projection.get("execution_device")
        or result.gpu_process_binding != assignment.gpu_process_binding
        or (
            result.gpu_process_binding is not None
            and result.gpu_process_binding.resource_binding_hash
            != projection.get("resource_binding_hash")
        )
    ):
        raise ValueError("generation train projection/receipt binding mismatch")
    return loaded


def _load_bound_phase_projection(
    trial: Mapping[str, Any],
    *,
    assignment: GenerationLaneAssignment,
    phase: EvaluationPhase,
    private_artifacts: Any,
) -> LoadedTrialEvalPhaseReceipt:
    projection = trial.get(phase.value)
    if not isinstance(projection, Mapping):
        raise ValueError("generation phase projection is missing")
    ref = _find_private_trial_receipt(
        private_artifacts,
        artifact_kind=f"{trial['lane_id']}_{phase.value}_phase_receipt",
        receipt_kind="eval_phase",
        checksum=projection.get("phase_receipt_checksum"),
    )
    loaded = load_trial_eval_phase_receipt(ref)
    result = loaded.result
    expected_gpu_process_binding = (
        assignment.gpu_process_binding.to_dict()
        if assignment.gpu_process_binding is not None
        else None
    )
    if (
        canonical_json_hash(loaded.contract) != trial.get("trial_contract_hash")
        or loaded.job_spec_hash != projection.get("job_spec_hash")
        or result.phase is not phase
        or canonical_json_hash(result.to_private_dict())
        != projection.get("phase_result_hash")
        or result.evaluation_plan_hash != projection.get("evaluation_plan_hash")
        or result.checkpoint_hash != projection.get("checkpoint_hash")
        or result.execution_device != projection.get("execution_device")
        or result.resource_binding_hash != projection.get("resource_binding_hash")
        or result.gpu_process_binding != expected_gpu_process_binding
        or result.task_evidence_hash != projection.get("task_evidence_hash")
    ):
        raise ValueError("generation phase projection/receipt binding mismatch")
    return loaded


def _find_private_execution_receipt(
    private_artifacts: Any,
    *,
    lane_id: str,
    checksum: str,
) -> TrialReceiptRef:
    if not isinstance(private_artifacts, list):
        raise ValueError("generation private_artifacts must be a list")
    expected_kind = f"{lane_id}_trial_execution_receipt"
    matches = [
        item
        for item in private_artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == expected_kind
        and item.get("checksum") == checksum
    ]
    if len(matches) != 1:
        raise ValueError("generation formal execution receipt is not uniquely bound")
    item = matches[0]
    return TrialReceiptRef(
        kind="execution",
        path=item.get("uri"),
        checksum=checksum,
        size_bytes=item.get("size_bytes"),
    )


def _find_private_trial_receipt(
    private_artifacts: Any,
    *,
    artifact_kind: str,
    receipt_kind: str,
    checksum: Any,
) -> TrialReceiptRef:
    checksum = require_sha256_checksum(checksum, f"{artifact_kind} checksum")
    if not isinstance(private_artifacts, list):
        raise ValueError("generation private_artifacts must be a list")
    matches = [
        item
        for item in private_artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == artifact_kind
        and item.get("checksum") == checksum
    ]
    if len(matches) != 1:
        raise ValueError(f"{artifact_kind} is not uniquely bound")
    item = matches[0]
    return TrialReceiptRef(
        kind=receipt_kind,
        path=item.get("uri"),
        checksum=checksum,
        size_bytes=item.get("size_bytes"),
    )


def _load_bound_authoring_evidence(
    trial: Mapping[str, Any],
    *,
    private_artifacts: Any,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Verify both authoring artifacts named by one public trial projection."""

    authoring = trial.get("authoring")
    if not isinstance(authoring, Mapping):
        raise ValueError("generation trial has no authoring projection")
    prompt_item = _find_private_artifact(
        private_artifacts,
        kind=f"{trial['lane_id']}_prompt_payload",
        checksum=authoring.get("prompt_payload_checksum"),
    )
    evidence_item = _find_private_artifact(
        private_artifacts,
        kind=f"{trial['lane_id']}_authoring_evidence",
        checksum=authoring.get("authoring_evidence_checksum"),
    )
    prompt_ref = PromptPayloadRef(
        path=prompt_item["uri"],
        checksum=prompt_item["checksum"],
        size_bytes=prompt_item["size_bytes"],
    )
    evidence_ref = AuthoringEvidenceRef(
        path=evidence_item["uri"],
        checksum=evidence_item["checksum"],
        size_bytes=evidence_item["size_bytes"],
    )
    prompt_payload = verify_prompt_payload(prompt_ref)
    verifier_rules = _provider_verifier_rules(prompt_payload)
    lane_intent = require_generation_lane_intent(
        verifier_rules.get("generation_lane_intent")
    )
    if canonical_json_hash(lane_intent) != authoring.get("lane_intent_hash"):
        raise ValueError("generation authoring prompt lane intent mismatch")
    raw_exclusion = verifier_rules.get("single_component_semantic_exclusion")
    if trial["lane_id"] == GENERATION_LANE_IDS[0]:
        if raw_exclusion is not None or authoring.get("sibling_exclusion_hash") is not None:
            raise ValueError("generation lane A authoring prompt contains exclusion")
    else:
        exclusion = validate_single_component_semantic_exclusion_projection(
            raw_exclusion
        )
        if exclusion["exclusion_hash"] != authoring.get("sibling_exclusion_hash"):
            raise ValueError("generation lane B authoring prompt exclusion mismatch")
    evidence = verify_authoring_evidence(
        evidence_ref,
        expected_round_id=str(trial["trial_id"]),
        expected_selected_prompt=prompt_ref,
    )
    outcome = authoring_outcome_from_evidence(evidence)
    candidate_verified = outcome.get("stop_reason") == "candidate_verified"
    expected_verified = (
        trial.get("trial_status") != "candidate_failed"
        or trial.get("trial_contract_hash") is not None
    )
    if candidate_verified != expected_verified:
        raise ValueError("generation trial status/authoring outcome mismatch")
    return evidence, lane_intent


def _provider_verifier_rules(prompt_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    required = prompt_payload.get("required_candidate_package")
    component_contract = (
        required.get("component_contract") if isinstance(required, Mapping) else None
    )
    rules = (
        component_contract.get("static_verifier_rules")
        if isinstance(component_contract, Mapping)
        else None
    )
    if not isinstance(rules, Mapping):
        raise ValueError("generation authoring prompt has no verifier rules")
    return rules


def _find_private_artifact(
    private_artifacts: Any,
    *,
    kind: str,
    checksum: Any,
) -> Mapping[str, Any]:
    checksum = require_sha256_checksum(checksum, f"{kind} checksum")
    if not isinstance(private_artifacts, list):
        raise ValueError("generation private_artifacts must be a list")
    matches = [
        item
        for item in private_artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == kind
        and item.get("checksum") == checksum
    ]
    if len(matches) != 1:
        raise ValueError(f"{kind} is not uniquely bound")
    item = matches[0]
    if (
        not isinstance(item.get("uri"), str)
        or not item.get("uri")
        or isinstance(item.get("size_bytes"), bool)
        or not isinstance(item.get("size_bytes"), int)
        or item.get("size_bytes") <= 0
    ):
        raise ValueError(f"{kind} ref is invalid")
    return item


def _find_private_generation_receipt(
    private_artifacts: Any,
    *,
    artifact_kind: str,
    receipt_kind: str,
    checksum: Any,
) -> GenerationReceiptRef:
    checksum = require_sha256_checksum(
        checksum,
        f"{artifact_kind} checksum",
    )
    if not isinstance(private_artifacts, list):
        raise ValueError("generation private_artifacts must be a list")
    matches = [
        item
        for item in private_artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == artifact_kind
        and item.get("checksum") == checksum
    ]
    if len(matches) != 1:
        raise ValueError(f"{artifact_kind} is not uniquely bound")
    item = matches[0]
    return GenerationReceiptRef(
        kind=receipt_kind,
        path=item.get("uri"),
        checksum=checksum,
        size_bytes=item.get("size_bytes"),
    )


def _find_guard_phase_receipt(
    private_artifacts: Any,
) -> TrialReceiptRef:
    if not isinstance(private_artifacts, list):
        raise ValueError("generation private_artifacts must be a list")
    matches = [
        item
        for item in private_artifacts
        if isinstance(item, Mapping)
        and item.get("kind") == "generation_guard_reference_phase_receipt"
    ]
    if len(matches) != 1:
        raise ValueError("generation guard phase receipt is not uniquely bound")
    item = matches[0]
    return TrialReceiptRef(
        kind="eval_phase",
        path=item.get("uri"),
        checksum=item.get("checksum"),
        size_bytes=item.get("size_bytes"),
    )


def _validate_selection_receipt_bindings(
    loaded: Any,
    *,
    trials: Sequence[Mapping[str, Any]],
    manifest: GenerationManifest,
    private_artifacts: Any,
) -> None:
    guard_ref = _find_guard_phase_receipt(private_artifacts)
    if (
        canonical_json_hash(guard_ref.to_dict())
        != manifest.guard_reference_phase_receipt_hash
        or loaded.guard_reference_phase_receipt_checksum != guard_ref.checksum
        or canonical_json_hash(loaded.guard_reference)
        != manifest.guard_reference_evidence_hash
    ):
        raise ValueError("generation selection guard receipt binding mismatch")
    trials_by_id = {str(item["trial_id"]): item for item in trials}
    if len(loaded.screens) != GENERATION_WIDTH:
        raise ValueError("generation selection receipt requires two screens")
    for screen in loaded.screens:
        trial_id = screen.get("trial_id")
        trial = trials_by_id.get(str(trial_id))
        if trial is None or trial_id != trial["trial_id"]:
            raise ValueError("generation selection screen names an unknown trial")
        public_dev = trial.get("public_dev")
        train = trial.get("train")
        if not isinstance(public_dev, Mapping):
            raise ValueError("generation selection screen has no public-dev receipt")
        for field_name in (
            "phase_result_hash",
            "evaluation_plan_hash",
            "checkpoint_hash",
            "execution_device",
            "resource_binding_hash",
        ):
            if screen.get(field_name) != public_dev.get(field_name):
                raise ValueError(
                    f"generation selection screen {field_name} binding mismatch"
                )
        if "train_receipt_checksum" in screen and (
            not isinstance(train, Mapping)
            or screen.get("train_receipt_checksum")
            != train.get("train_receipt_checksum")
        ):
            raise ValueError(
                "generation selection screen train receipt binding mismatch"
            )


def _validate_status_relationships(
    *,
    trials: Sequence[Mapping[str, Any]],
    selection_present: bool,
    selected_trial_id: str | None,
    formal_round: Mapping[str, Any] | None,
    outcome: str,
) -> None:
    statuses = [item["trial_status"] for item in trials]
    train_count = sum(item["train"] is not None for item in trials)
    dev_count = sum(item["public_dev"] is not None for item in trials)
    accept_count = sum(item["public_accept"] is not None for item in trials)
    if train_count > 2 or accept_count > 2:
        raise ValueError("generation execution count exceeds fixed width")
    if statuses == ["candidate_failed", "not_started"]:
        expected_outcome = "paired_authoring_incomplete"
    elif statuses in (
        ["verified_not_run", "candidate_failed"],
        ["candidate_failed", "verified_not_run"],
    ):
        expected_outcome = "paired_authoring_incomplete"
    elif formal_round is None:
        expected_outcome = (
            "candidate_execution_incomplete"
            if any(status == "candidate_failed" for status in statuses)
            else "no_selection"
        )
    else:
        if not selection_present or selected_trial_id is None:
            raise ValueError("generation formal result requires a selected trial")
        decision = formal_round.get("decision")
        accepted = isinstance(decision, Mapping) and decision.get("accepted") is True
        expected_outcome = "accepted" if accepted else "formal_rejected"
        selected = next(
            item for item in trials if item["trial_id"] == selected_trial_id
        )
        expected_status = "accepted" if accepted else "formal_rejected"
        if selected["trial_status"] != expected_status:
            raise ValueError("generation selected trial status mismatch")
        if any(
            selected[field_name] is None
            for field_name in ("train", "public_dev", "public_accept")
        ):
            raise ValueError("generation selected trial evidence is incomplete")
        for item in trials:
            if item is not selected and item["trial_status"] not in {
                "qualified_not_selected",
                "candidate_failed",
            }:
                raise ValueError("generation nonselected evaluated trial status mismatch")
    if outcome != expected_outcome:
        raise ValueError("generation outcome/status relationship mismatch")
    if expected_outcome == "paired_authoring_incomplete" and (
        selection_present
        or formal_round is not None
        or train_count != 0
        or dev_count != 0
        or accept_count != 0
    ):
        raise ValueError("incomplete sibling authoring contains execution evidence")
    if expected_outcome == "no_selection" and (
        not selection_present
        or selected_trial_id is not None
        or train_count != GENERATION_WIDTH
        or dev_count != GENERATION_WIDTH
        or accept_count != GENERATION_WIDTH
        or statuses
        != ["qualified_not_selected", "qualified_not_selected"]
    ):
        raise ValueError("generation no-selection evidence is incomplete")
    if formal_round is None and selected_trial_id is not None and outcome != "candidate_execution_incomplete":
        raise ValueError("generation selected a trial without formal evidence")


def _validate_private_artifacts(value: Any) -> None:
    if not isinstance(value, list):
        raise ValueError("generation private_artifacts must be a list")
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("generation private artifact ref must be an object")
        if not isinstance(item.get("kind"), str) or not isinstance(
            item.get("uri"), str
        ):
            raise ValueError("generation private artifact identity is invalid")
        if item.get("checksum") is not None:
            require_sha256_checksum(
                item.get("checksum"), "generation private artifact checksum"
            )


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"generation {field_name} is required")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"generation {field_name} must be positive")
    return value


def _public_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _public_copy(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_public_copy(item) for item in value]
    return value


__all__ = [
    "GENERATION_FORMAL_PROJECTION_SCHEMA_VERSION",
    "GENERATION_OUTCOMES",
    "GENERATION_PHASE_PROJECTION_SCHEMA_VERSION",
    "GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION",
    "GENERATION_TRIAL_RECORD_SCHEMA_VERSION",
    "commit_generation_record",
    "load_bound_generation_manifest",
    "load_bound_generation_trial_phase_receipt",
    "load_bound_generation_trial_train_receipt",
    "project_generation_formal_round_record",
    "project_generation_public_history_record",
    "validate_generation_ledger_record",
]
