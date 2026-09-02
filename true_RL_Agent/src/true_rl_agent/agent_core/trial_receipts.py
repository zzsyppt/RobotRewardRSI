"""Private create-once receipts for ledger-independent trial execution."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.decision import DecisionOutcome, DecisionRecord
from true_rl_agent.candidates.authoring_policy import (
    require_mechanism_operation_delta,
)
from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION,
    EVALUATION_PHASE_RECEIPT_REF_SCHEMA_VERSION,
    MAX_EVALUATION_PHASE_RECEIPT_BYTES,
    EvaluationPhaseResult,
)
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.initial_policy import InitialPolicyStatePairing
from true_rl_agent.runner.specs import (
    CheckpointRef,
    EvalJobSpec,
    EvalPhaseJobSpec,
    GpuProcessBinding,
    TrainJobSpec,
    require_eval_phase_result_matches_spec,
)

TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION = (
    "true_rl_agent.trial_execution_contract.v1"
)
TRIAL_TRAIN_RECEIPT_SCHEMA_VERSION = "true_rl_agent.trial_train_receipt.v1"
TRIAL_EVAL_RECEIPT_SCHEMA_VERSION = "true_rl_agent.trial_eval_receipt.v1"
TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION = EVALUATION_PHASE_RECEIPT_SCHEMA_VERSION
TRIAL_EXECUTION_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.trial_execution_receipt.v1"
)
TRIAL_RECEIPT_REF_SCHEMA_VERSION = EVALUATION_PHASE_RECEIPT_REF_SCHEMA_VERSION

MAX_TRIAL_CONTRACT_BYTES = 2 * 1024 * 1024
MAX_TRIAL_PHASE_RECEIPT_BYTES = MAX_EVALUATION_PHASE_RECEIPT_BYTES
MAX_TRIAL_EXECUTION_RECEIPT_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class TrialReceiptRef:
    kind: str
    path: str
    checksum: str
    size_bytes: int
    schema_version: str = TRIAL_RECEIPT_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TRIAL_RECEIPT_REF_SCHEMA_VERSION:
            raise ValueError("trial receipt ref schema mismatch")
        if self.kind not in {"train", "eval", "eval_phase", "execution"}:
            raise ValueError("trial receipt ref kind is invalid")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("trial receipt ref path is required")
        require_sha256_checksum(self.checksum, "trial receipt checksum")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > MAX_TRIAL_EXECUTION_RECEIPT_BYTES
        ):
            raise ValueError("trial receipt ref size is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class LoadedTrialExecution:
    record: Mapping[str, Any]
    decision: DecisionRecord
    train_result: TrainJobResult | None
    public_result: PublicEvalResult | None
    round_stage: str
    mechanism_operation_delta: Mapping[str, Any] | None
    receipt_ref: TrialReceiptRef


@dataclass(frozen=True)
class LoadedTrialTrainReceipt:
    """A train receipt loaded through its bound trial contract."""

    contract: Mapping[str, Any]
    job_spec_hash: str
    result: TrainJobResult
    receipt_ref: TrialReceiptRef


@dataclass(frozen=True)
class LoadedTrialEvalPhaseReceipt:
    """An eval-phase receipt loaded through its bound trial contract."""

    contract: Mapping[str, Any]
    job_spec_hash: str
    result: EvaluationPhaseResult
    receipt_ref: TrialReceiptRef


class TrialReceiptJournal:
    """One content-addressed operational journal for a prepared trial."""

    def __init__(self, root: Path, *, contract: Mapping[str, Any]) -> None:
        if not isinstance(root, Path):
            raise TypeError("trial receipt root must be a Path")
        if not isinstance(contract, Mapping):
            raise TypeError("trial execution contract must be an object")
        self.contract = _validate_trial_contract(contract)
        self.contract_hash = canonical_json_hash(self.contract)
        self.root = root.resolve(strict=False) / self.contract_hash

    def bind_contract(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "contract": dict(self.contract),
        }
        self._publish_or_validate(
            self.root / "contract.json",
            payload,
            max_bytes=MAX_TRIAL_CONTRACT_BYTES,
            context="trial execution contract",
        )

    def load_train(self, train_spec: TrainJobSpec) -> tuple[TrainJobResult, TrialReceiptRef] | None:
        path = self.root / "train.json"
        if not path.exists():
            return None
        payload, ref = self._load(
            path,
            kind="train",
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial train receipt",
        )
        _require_receipt_identity(
            payload,
            schema_version=TRIAL_TRAIN_RECEIPT_SCHEMA_VERSION,
            contract_hash=self.contract_hash,
            expected_fields={
                "schema_version",
                "contract_hash",
                "job_spec_hash",
                "train_result",
                "train_result_hash",
            },
        )
        expected_spec_hash = canonical_json_hash(train_spec.to_public_dict())
        if payload.get("job_spec_hash") != expected_spec_hash:
            raise ValueError("trial train receipt job spec mismatch")
        result_payload = payload.get("train_result")
        if canonical_json_hash(result_payload) != payload.get("train_result_hash"):
            raise ValueError("trial train receipt result hash mismatch")
        result = _train_result_from_private_mapping(result_payload)
        _require_train_result_matches_spec(result, train_spec)
        return result, ref

    def publish_train(
        self,
        train_spec: TrainJobSpec,
        result: TrainJobResult,
    ) -> TrialReceiptRef:
        _require_train_result_matches_spec(result, train_spec)
        result_payload = _train_result_to_private_mapping(result)
        payload = {
            "schema_version": TRIAL_TRAIN_RECEIPT_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "job_spec_hash": canonical_json_hash(train_spec.to_public_dict()),
            "train_result": result_payload,
            "train_result_hash": canonical_json_hash(result_payload),
        }
        return self._publish_or_validate(
            self.root / "train.json",
            payload,
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial train receipt",
            kind="train",
        )

    def load_eval(self, eval_spec: EvalJobSpec) -> tuple[PublicEvalResult, TrialReceiptRef] | None:
        path = self.root / "eval.json"
        if not path.exists():
            return None
        payload, ref = self._load(
            path,
            kind="eval",
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial eval receipt",
        )
        _require_receipt_identity(
            payload,
            schema_version=TRIAL_EVAL_RECEIPT_SCHEMA_VERSION,
            contract_hash=self.contract_hash,
            expected_fields={
                "schema_version",
                "contract_hash",
                "job_spec_hash",
                "public_result",
                "public_result_hash",
            },
        )
        if payload.get("job_spec_hash") != canonical_json_hash(eval_spec.to_public_dict()):
            raise ValueError("trial eval receipt job spec mismatch")
        result_payload = payload.get("public_result")
        if canonical_json_hash(result_payload) != payload.get("public_result_hash"):
            raise ValueError("trial eval receipt result hash mismatch")
        return _public_result_from_mapping(result_payload), ref

    def publish_eval(
        self,
        eval_spec: EvalJobSpec,
        result: PublicEvalResult,
    ) -> TrialReceiptRef:
        result_payload = result.to_public_dict()
        payload = {
            "schema_version": TRIAL_EVAL_RECEIPT_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "job_spec_hash": canonical_json_hash(eval_spec.to_public_dict()),
            "public_result": result_payload,
            "public_result_hash": canonical_json_hash(result_payload),
        }
        return self._publish_or_validate(
            self.root / "eval.json",
            payload,
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial eval receipt",
            kind="eval",
        )

    def load_eval_phase(
        self,
        eval_spec: EvalPhaseJobSpec,
    ) -> tuple[EvaluationPhaseResult, TrialReceiptRef] | None:
        path = self.root / "eval_phases" / f"{eval_spec.phase.value}.json"
        if not path.exists():
            return None
        payload, ref = self._load(
            path,
            kind="eval_phase",
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial eval phase receipt",
        )
        _require_receipt_identity(
            payload,
            schema_version=TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION,
            contract_hash=self.contract_hash,
            expected_fields={
                "schema_version",
                "contract_hash",
                "job_spec_hash",
                "phase_result",
                "phase_result_hash",
            },
        )
        if payload.get("job_spec_hash") != canonical_json_hash(
            eval_spec.to_public_dict()
        ):
            raise ValueError("trial eval phase receipt job spec mismatch")
        result_payload = payload.get("phase_result")
        if canonical_json_hash(result_payload) != payload.get("phase_result_hash"):
            raise ValueError("trial eval phase receipt result hash mismatch")
        result = EvaluationPhaseResult.from_private_mapping(result_payload)
        require_eval_phase_result_matches_spec(result, eval_spec)
        return result, ref

    def publish_eval_phase(
        self,
        eval_spec: EvalPhaseJobSpec,
        result: EvaluationPhaseResult,
    ) -> TrialReceiptRef:
        require_eval_phase_result_matches_spec(result, eval_spec)
        result_payload = result.to_private_dict()
        payload = {
            "schema_version": TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "job_spec_hash": canonical_json_hash(eval_spec.to_public_dict()),
            "phase_result": result_payload,
            "phase_result_hash": canonical_json_hash(result_payload),
        }
        path = self.root / "eval_phases" / f"{eval_spec.phase.value}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        ref = self._publish_or_validate(
            path,
            payload,
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial eval phase receipt",
            kind="eval_phase",
        )
        assert isinstance(ref, TrialReceiptRef)
        return ref

    def load_execution(self) -> LoadedTrialExecution | None:
        path = self.root / "execution.json"
        if not path.exists():
            return None
        payload, ref = self._load(
            path,
            kind="execution",
            max_bytes=MAX_TRIAL_EXECUTION_RECEIPT_BYTES,
            context="trial execution receipt",
        )
        _require_receipt_identity(
            payload,
            schema_version=TRIAL_EXECUTION_RECEIPT_SCHEMA_VERSION,
            contract_hash=self.contract_hash,
            expected_fields={
                "schema_version",
                "contract_hash",
                "train_receipt_checksum",
                "eval_receipt_checksum",
                "mechanism_operation_delta",
                "record",
                "record_hash",
                "round_stage",
            },
        )
        record = payload.get("record")
        if not isinstance(record, Mapping):
            raise ValueError("trial execution receipt record is invalid")
        if canonical_json_hash(record) != payload.get("record_hash"):
            raise ValueError("trial execution receipt record hash mismatch")
        _require_trial_contract_record_binding(self.contract, record)
        jobs = record.get("jobs")
        if jobs is not None and not isinstance(jobs, Mapping):
            raise ValueError("trial execution record jobs are invalid")
        train_result = self._load_bound_train(
            payload.get("train_receipt_checksum"),
            expected_job_spec=(jobs.get("train") if jobs is not None else None),
        )
        self._require_bound_eval(
            payload.get("eval_receipt_checksum"),
            expected_job_spec=(jobs.get("eval") if jobs is not None else None),
        )
        if (
            train_result.to_public_dict() if train_result is not None else None
        ) != record.get("train_result"):
            raise ValueError(
                "trial execution train result differs from its bound receipt"
            )
        mechanism_delta = payload.get("mechanism_operation_delta")
        if mechanism_delta is not None:
            mechanism_delta = require_mechanism_operation_delta(mechanism_delta)
        if mechanism_delta != self.contract.get("mechanism_operation_delta"):
            raise ValueError(
                "trial execution mechanism delta differs from its contract"
            )
        round_stage = payload.get("round_stage")
        if not isinstance(round_stage, str) or not round_stage:
            raise ValueError("trial execution receipt round stage is invalid")
        return LoadedTrialExecution(
            record=dict(record),
            decision=_decision_from_mapping(record.get("decision")),
            train_result=train_result,
            public_result=(
                _public_result_from_mapping(record.get("public_result"))
                if record.get("public_result") is not None
                else None
            ),
            round_stage=round_stage,
            mechanism_operation_delta=mechanism_delta,
            receipt_ref=ref,
        )

    def publish_execution(
        self,
        *,
        record: Mapping[str, Any],
        round_stage: str,
        train_receipt_ref: TrialReceiptRef | None,
        eval_receipt_ref: TrialReceiptRef | None,
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> TrialReceiptRef:
        if not isinstance(record, Mapping):
            raise TypeError("trial execution record must be an object")
        _require_trial_contract_record_binding(self.contract, record)
        if not isinstance(round_stage, str) or not round_stage:
            raise ValueError("trial execution round stage is required")
        if mechanism_operation_delta is not None:
            mechanism_operation_delta = require_mechanism_operation_delta(
                mechanism_operation_delta
            )
        _require_optional_ref(train_receipt_ref, "train")
        _require_optional_ref(eval_receipt_ref, "eval")
        payload = {
            "schema_version": TRIAL_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "contract_hash": self.contract_hash,
            "train_receipt_checksum": (
                train_receipt_ref.checksum if train_receipt_ref is not None else None
            ),
            "eval_receipt_checksum": (
                eval_receipt_ref.checksum if eval_receipt_ref is not None else None
            ),
            "mechanism_operation_delta": (
                dict(mechanism_operation_delta)
                if mechanism_operation_delta is not None
                else None
            ),
            "record": dict(record),
            "record_hash": canonical_json_hash(record),
            "round_stage": round_stage,
        }
        return self._publish_or_validate(
            self.root / "execution.json",
            payload,
            max_bytes=MAX_TRIAL_EXECUTION_RECEIPT_BYTES,
            context="trial execution receipt",
            kind="execution",
        )

    def _load_bound_train(
        self,
        expected_checksum: Any,
        *,
        expected_job_spec: Any,
    ) -> TrainJobResult | None:
        path = self.root / "train.json"
        if expected_checksum is None:
            if path.exists():
                raise ValueError("trial execution omits an existing train receipt")
            return None
        require_sha256_checksum(expected_checksum, "bound train receipt checksum")
        payload, ref = self._load(
            path,
            kind="train",
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial train receipt",
        )
        _require_receipt_identity(
            payload,
            schema_version=TRIAL_TRAIN_RECEIPT_SCHEMA_VERSION,
            contract_hash=self.contract_hash,
            expected_fields={
                "schema_version",
                "contract_hash",
                "job_spec_hash",
                "train_result",
                "train_result_hash",
            },
        )
        if ref.checksum != expected_checksum:
            raise ValueError("trial execution train receipt checksum mismatch")
        if canonical_json_hash(expected_job_spec) != payload.get("job_spec_hash"):
            raise ValueError("trial execution train job spec mismatch")
        result = _train_result_from_private_mapping(payload.get("train_result"))
        if canonical_json_hash(payload.get("train_result")) != payload.get(
            "train_result_hash"
        ):
            raise ValueError("trial train receipt result hash mismatch")
        return result

    def _require_bound_eval(
        self,
        expected_checksum: Any,
        *,
        expected_job_spec: Any,
    ) -> None:
        path = self.root / "eval.json"
        if expected_checksum is None:
            if path.exists():
                raise ValueError("trial execution omits an existing eval receipt")
            return
        require_sha256_checksum(expected_checksum, "bound eval receipt checksum")
        payload, ref = self._load(
            path,
            kind="eval",
            max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
            context="trial eval receipt",
        )
        _require_receipt_identity(
            payload,
            schema_version=TRIAL_EVAL_RECEIPT_SCHEMA_VERSION,
            contract_hash=self.contract_hash,
            expected_fields={
                "schema_version",
                "contract_hash",
                "job_spec_hash",
                "public_result",
                "public_result_hash",
            },
        )
        if ref.checksum != expected_checksum:
            raise ValueError("trial execution eval receipt checksum mismatch")
        if canonical_json_hash(expected_job_spec) != payload.get("job_spec_hash"):
            raise ValueError("trial execution eval job spec mismatch")
        if canonical_json_hash(payload.get("public_result")) != payload.get(
            "public_result_hash"
        ):
            raise ValueError("trial eval receipt result hash mismatch")

    def _publish_or_validate(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        max_bytes: int,
        context: str,
        kind: str | None = None,
    ) -> TrialReceiptRef | None:
        encoded = canonical_json_bytes(dict(payload), trailing_newline=True)
        if len(encoded) > max_bytes:
            raise ValueError(f"{context} exceeds its typed byte limit")
        try:
            create_once_bytes(path, encoded)
        except FileExistsError:
            pass
        if kind is None:
            loaded = _load_canonical(path, max_bytes=max_bytes, context=context)
            if loaded != dict(payload):
                raise ValueError(f"{context} differs from the sealed artifact")
            return None
        loaded, ref = self._load(
            path,
            kind=kind,
            max_bytes=max_bytes,
            context=context,
        )
        if loaded != dict(payload):
            raise ValueError(f"{context} differs from the sealed artifact")
        return ref

    def _load(
        self,
        path: Path,
        *,
        kind: str,
        max_bytes: int,
        context: str,
    ) -> tuple[dict[str, Any], TrialReceiptRef]:
        payload = read_bounded_regular_file(
            path,
            max_bytes=max_bytes,
            label=context,
        )
        parsed = strict_json_object_loads(payload, context=context)
        if payload != canonical_json_bytes(parsed, trailing_newline=True):
            raise ValueError(f"{context} is not canonical JSON")
        return parsed, TrialReceiptRef(
            kind=kind,
            path=str(path.resolve(strict=False)),
            checksum="sha256:" + hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
        )


def _validate_trial_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    contract = dict(value)
    if contract.get("schema_version") != TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("trial execution contract schema mismatch")
    for field_name in ("session_genesis_hash", "ledger_epoch_hash"):
        require_sha256_digest(contract.get(field_name), f"trial contract {field_name}")
    round_id = contract.get("round_id")
    if not isinstance(round_id, str) or not round_id:
        raise ValueError("trial execution contract round_id is required")
    return contract


def _require_trial_contract_record_binding(
    contract: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Bind the receipt contract to the exact scientific record it encloses."""

    shared_fields = (
        "authoring_mode",
        "round_id",
        "parent_program_source_checksum",
        "session_genesis_hash",
        "ledger_epoch_hash",
        "evaluation_binding",
        "public_context",
        "task",
        "acceptance_protocol",
        "backend",
        "candidate",
        "candidate_surface",
        "curriculum",
    )
    for field_name in shared_fields:
        if contract.get(field_name) != record.get(field_name):
            raise ValueError(
                "trial execution contract does not match record field: "
                + field_name
            )
    if contract.get("verification") != record.get("verifier"):
        raise ValueError(
            "trial execution contract does not match record field: verifier"
        )


def load_trial_execution_receipt(
    ref: TrialReceiptRef,
) -> tuple[dict[str, Any], LoadedTrialExecution]:
    """Load a final receipt through its create-once contract envelope."""

    if not isinstance(ref, TrialReceiptRef) or ref.kind != "execution":
        raise TypeError("final trial execution receipt ref is invalid")
    execution_path = Path(ref.path).resolve(strict=False)
    if execution_path.name != "execution.json":
        raise ValueError("final trial receipt path is invalid")
    contract_path = execution_path.parent / "contract.json"
    envelope = _load_canonical(
        contract_path,
        max_bytes=MAX_TRIAL_CONTRACT_BYTES,
        context="trial execution contract",
    )
    if set(envelope) != {"schema_version", "contract_hash", "contract"}:
        raise ValueError("trial execution contract envelope fields mismatch")
    if envelope.get("schema_version") != TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("trial execution contract envelope schema mismatch")
    contract = envelope.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("trial execution contract envelope is invalid")
    validated_contract = _validate_trial_contract(contract)
    contract_hash = canonical_json_hash(validated_contract)
    if envelope.get("contract_hash") != contract_hash:
        raise ValueError("trial execution contract envelope hash mismatch")
    journal = TrialReceiptJournal(
        execution_path.parent.parent,
        contract=validated_contract,
    )
    if journal.root != execution_path.parent:
        raise ValueError("trial execution receipt is outside its contract directory")
    loaded = journal.load_execution()
    if loaded is None or loaded.receipt_ref != ref:
        raise ValueError("final trial execution receipt ref mismatch")
    return validated_contract, loaded


def load_trial_train_receipt(
    ref: TrialReceiptRef,
) -> LoadedTrialTrainReceipt:
    """Load a train receipt without reconstructing its runner job object."""

    journal, contract, path = _journal_for_standalone_receipt(
        ref,
        expected_kind="train",
        expected_relative_path=Path("train.json"),
    )
    payload, loaded_ref = journal._load(
        path,
        kind="train",
        max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
        context="trial train receipt",
    )
    _require_receipt_identity(
        payload,
        schema_version=TRIAL_TRAIN_RECEIPT_SCHEMA_VERSION,
        contract_hash=journal.contract_hash,
        expected_fields={
            "schema_version",
            "contract_hash",
            "job_spec_hash",
            "train_result",
            "train_result_hash",
        },
    )
    job_spec_hash = require_sha256_digest(
        payload.get("job_spec_hash"),
        "trial train receipt job spec hash",
    )
    result_payload = payload.get("train_result")
    if canonical_json_hash(result_payload) != payload.get("train_result_hash"):
        raise ValueError("trial train receipt result hash mismatch")
    if loaded_ref != ref:
        raise ValueError("trial train receipt ref mismatch")
    return LoadedTrialTrainReceipt(
        contract=contract,
        job_spec_hash=job_spec_hash,
        result=_train_result_from_private_mapping(result_payload),
        receipt_ref=loaded_ref,
    )


def load_trial_eval_phase_receipt(
    ref: TrialReceiptRef,
) -> LoadedTrialEvalPhaseReceipt:
    """Load one phase receipt without duplicating task projection semantics."""

    if not isinstance(ref, TrialReceiptRef) or ref.kind != "eval_phase":
        raise TypeError("trial eval phase receipt ref is invalid")
    path = Path(ref.path).resolve(strict=False)
    if path.parent.name != "eval_phases" or path.suffix != ".json":
        raise ValueError("trial eval phase receipt path is invalid")
    journal, contract, path = _journal_for_standalone_receipt(
        ref,
        expected_kind="eval_phase",
        expected_relative_path=Path("eval_phases") / path.name,
    )
    payload, loaded_ref = journal._load(
        path,
        kind="eval_phase",
        max_bytes=MAX_TRIAL_PHASE_RECEIPT_BYTES,
        context="trial eval phase receipt",
    )
    _require_receipt_identity(
        payload,
        schema_version=TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION,
        contract_hash=journal.contract_hash,
        expected_fields={
            "schema_version",
            "contract_hash",
            "job_spec_hash",
            "phase_result",
            "phase_result_hash",
        },
    )
    job_spec_hash = require_sha256_digest(
        payload.get("job_spec_hash"),
        "trial eval phase receipt job spec hash",
    )
    result_payload = payload.get("phase_result")
    if canonical_json_hash(result_payload) != payload.get("phase_result_hash"):
        raise ValueError("trial eval phase receipt result hash mismatch")
    result = EvaluationPhaseResult.from_private_mapping(result_payload)
    if path.name != f"{result.phase.value}.json":
        raise ValueError("trial eval phase receipt filename/phase mismatch")
    if loaded_ref != ref:
        raise ValueError("trial eval phase receipt ref mismatch")
    return LoadedTrialEvalPhaseReceipt(
        contract=contract,
        job_spec_hash=job_spec_hash,
        result=result,
        receipt_ref=loaded_ref,
    )


def _journal_for_standalone_receipt(
    ref: TrialReceiptRef,
    *,
    expected_kind: str,
    expected_relative_path: Path,
) -> tuple[TrialReceiptJournal, dict[str, Any], Path]:
    if not isinstance(ref, TrialReceiptRef) or ref.kind != expected_kind:
        raise TypeError(f"trial {expected_kind} receipt ref is invalid")
    path = Path(ref.path).resolve(strict=False)
    contract_root = (
        path.parent.parent if expected_kind == "eval_phase" else path.parent
    )
    expected_path = (contract_root / expected_relative_path).resolve(strict=False)
    if path != expected_path:
        raise ValueError(f"trial {expected_kind} receipt path mismatch")
    envelope = _load_canonical(
        contract_root / "contract.json",
        max_bytes=MAX_TRIAL_CONTRACT_BYTES,
        context="trial execution contract",
    )
    if set(envelope) != {"schema_version", "contract_hash", "contract"}:
        raise ValueError("trial execution contract envelope fields mismatch")
    if envelope.get("schema_version") != TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION:
        raise ValueError("trial execution contract envelope schema mismatch")
    raw_contract = envelope.get("contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("trial execution contract envelope is invalid")
    contract = _validate_trial_contract(raw_contract)
    contract_hash = canonical_json_hash(contract)
    if envelope.get("contract_hash") != contract_hash:
        raise ValueError("trial execution contract envelope hash mismatch")
    journal = TrialReceiptJournal(contract_root.parent, contract=contract)
    if journal.root != contract_root:
        raise ValueError(f"trial {expected_kind} receipt is outside its contract")
    return journal, contract, path


def _require_receipt_identity(
    payload: Mapping[str, Any],
    *,
    schema_version: str,
    contract_hash: str,
    expected_fields: set[str],
) -> None:
    if set(payload) != expected_fields:
        raise ValueError("trial receipt fields mismatch")
    if (
        payload.get("schema_version") != schema_version
        or payload.get("contract_hash") != contract_hash
    ):
        raise ValueError("trial receipt identity mismatch")


def _train_result_to_private_mapping(result: TrainJobResult) -> dict[str, Any]:
    payload = {
        "job_id": result.job_id,
        "subject": result.subject.to_public_dict(),
        "subject_manifest_hash": result.subject_manifest_hash,
        "program_source_checksum": result.program_source_checksum,
        "training_binding_hash": result.training_binding_hash,
        "capability_profile_hash": result.capability_profile_hash,
        "execution_device": result.execution_device,
        "checkpoint": result.checkpoint.to_public_dict(include_metadata=True),
        "initial_policy_state_pairing": (
            result.initial_policy_state_pairing.to_public_dict()
            if result.initial_policy_state_pairing is not None
            else None
        ),
        "summary": dict(result.summary),
    }
    if result.gpu_process_binding is not None:
        payload["gpu_process_binding"] = result.gpu_process_binding.to_dict()
    return payload


def _train_result_from_private_mapping(value: Any) -> TrainJobResult:
    expected_fields = {
        "job_id",
        "subject",
        "subject_manifest_hash",
        "program_source_checksum",
        "training_binding_hash",
        "capability_profile_hash",
        "execution_device",
        "checkpoint",
        "initial_policy_state_pairing",
        "summary",
    }
    if not isinstance(value, Mapping) or set(value) not in (
        expected_fields,
        expected_fields | {"gpu_process_binding"},
    ):
        raise ValueError("trial train result fields mismatch")
    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "uri",
        "checksum",
        "role",
        "sealed",
        "metadata",
    }:
        raise ValueError("trial train checkpoint fields mismatch")
    pairing = value.get("initial_policy_state_pairing")
    return TrainJobResult(
        job_id=value.get("job_id"),
        subject=ExperimentSubject.from_mapping(value.get("subject")),
        subject_manifest_hash=value.get("subject_manifest_hash"),
        program_source_checksum=value.get("program_source_checksum"),
        training_binding_hash=value.get("training_binding_hash"),
        capability_profile_hash=value.get("capability_profile_hash"),
        execution_device=value.get("execution_device"),
        checkpoint=CheckpointRef(
            uri=checkpoint.get("uri"),
            checksum=checkpoint.get("checksum"),
            role=checkpoint.get("role"),
            sealed=checkpoint.get("sealed"),
            metadata=checkpoint.get("metadata"),
        ),
        initial_policy_state_pairing=(
            InitialPolicyStatePairing.from_mapping(pairing)
            if pairing is not None
            else None
        ),
        gpu_process_binding=(
            GpuProcessBinding.from_mapping(value.get("gpu_process_binding"))
            if "gpu_process_binding" in value
            else None
        ),
        summary=value.get("summary"),
    )


def _require_train_result_matches_spec(
    result: TrainJobResult,
    spec: TrainJobSpec,
) -> None:
    expected = {
        "job_id": spec.job_id,
        "subject": spec.subject,
        "subject_manifest_hash": spec.subject_manifest_hash,
        "program_source_checksum": spec.program_source_checksum,
        "training_binding_hash": spec.training_binding.binding_hash,
        "capability_profile_hash": (
            spec.training_binding.capability_profile_hash
        ),
        "execution_device": spec.execution_device,
        "gpu_process_binding": spec.gpu_process_binding,
    }
    actual = {
        "job_id": result.job_id,
        "subject": result.subject,
        "subject_manifest_hash": result.subject_manifest_hash,
        "program_source_checksum": result.program_source_checksum,
        "training_binding_hash": result.training_binding_hash,
        "capability_profile_hash": result.capability_profile_hash,
        "execution_device": result.execution_device,
        "gpu_process_binding": result.gpu_process_binding,
    }
    if actual != expected:
        raise ValueError("trial train result does not match its job spec")


def _public_result_from_mapping(value: Any) -> PublicEvalResult:
    if not isinstance(value, Mapping) or set(value) != {
        "status",
        "public_score",
        "metrics",
    }:
        raise ValueError("trial public result fields mismatch")
    return PublicEvalResult(
        status=value.get("status"),
        public_score=value.get("public_score"),
        metrics=value.get("metrics"),
    )


def _decision_from_mapping(value: Any) -> DecisionRecord:
    if not isinstance(value, Mapping) or set(value) != {
        "outcome",
        "accepted",
        "reason",
        "surface_classification",
        "public_score",
        "comparison",
        "consumed_gates",
        "gate_statuses",
    }:
        raise ValueError("trial decision fields mismatch")
    return DecisionRecord(
        outcome=DecisionOutcome(value.get("outcome")),
        accepted=value.get("accepted"),
        reason=value.get("reason"),
        surface_classification=value.get("surface_classification"),
        public_score=value.get("public_score"),
        comparison=value.get("comparison"),
        consumed_gates=tuple(value.get("consumed_gates")),
        gate_statuses=value.get("gate_statuses"),
    )


def _require_optional_ref(ref: TrialReceiptRef | None, kind: str) -> None:
    if ref is not None and (not isinstance(ref, TrialReceiptRef) or ref.kind != kind):
        raise TypeError(f"trial {kind} receipt ref is invalid")


def _load_canonical(path: Path, *, max_bytes: int, context: str) -> dict[str, Any]:
    payload = read_bounded_regular_file(path, max_bytes=max_bytes, label=context)
    parsed = strict_json_object_loads(payload, context=context)
    if payload != canonical_json_bytes(parsed, trailing_newline=True):
        raise ValueError(f"{context} is not canonical JSON")
    return parsed


__all__ = [
    "LoadedTrialEvalPhaseReceipt",
    "LoadedTrialExecution",
    "LoadedTrialTrainReceipt",
    "TRIAL_EVAL_PHASE_RECEIPT_SCHEMA_VERSION",
    "TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION",
    "TrialReceiptJournal",
    "TrialReceiptRef",
    "load_trial_eval_phase_receipt",
    "load_trial_execution_receipt",
    "load_trial_train_receipt",
]
