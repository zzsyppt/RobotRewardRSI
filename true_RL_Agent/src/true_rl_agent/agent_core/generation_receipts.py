"""Create-once orchestration receipts for one fixed-width generation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.generation_contract import GenerationManifest
from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_checksum
from true_rl_agent.fs_contract import create_once_bytes, read_bounded_regular_file
from true_rl_agent.json_contract import canonical_json_bytes, strict_json_object_loads


GENERATION_MANIFEST_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.generation_manifest_receipt.v1"
)
GENERATION_SELECTION_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.generation_selection_receipt.v1"
)
GENERATION_EXECUTION_RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.generation_execution_receipt.v1"
)
GENERATION_RECEIPT_REF_SCHEMA_VERSION = "true_rl_agent.generation_receipt_ref.v1"
MAX_GENERATION_RECEIPT_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class GenerationReceiptRef:
    kind: str
    path: str
    checksum: str
    size_bytes: int
    schema_version: str = GENERATION_RECEIPT_REF_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_RECEIPT_REF_SCHEMA_VERSION:
            raise ValueError("generation receipt ref schema mismatch")
        if self.kind not in {"manifest", "selection", "execution"}:
            raise ValueError("generation receipt ref kind is invalid")
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("generation receipt ref path is required")
        require_sha256_checksum(self.checksum, "generation receipt checksum")
        if (
            isinstance(self.size_bytes, bool)
            or not isinstance(self.size_bytes, int)
            or self.size_bytes <= 0
            or self.size_bytes > MAX_GENERATION_RECEIPT_BYTES
        ):
            raise ValueError("generation receipt ref size is invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "path": self.path,
            "checksum": self.checksum,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class LoadedGenerationSelection:
    selection: Mapping[str, Any]
    screens: tuple[Mapping[str, Any], Mapping[str, Any]]
    guard_reference: Mapping[str, Any]
    guard_reference_phase_receipt_checksum: str
    receipt_ref: GenerationReceiptRef


class GenerationReceiptJournal:
    """Single owner of manifest, selection and final generation receipts."""

    def __init__(self, root: Path, *, manifest: GenerationManifest) -> None:
        if not isinstance(root, Path):
            raise TypeError("generation receipt root must be a Path")
        if not isinstance(manifest, GenerationManifest):
            raise TypeError("generation manifest is invalid")
        self.manifest = manifest
        self.root = root.resolve(strict=False) / manifest.manifest_hash

    def bind_manifest(self) -> GenerationReceiptRef:
        payload = {
            "schema_version": GENERATION_MANIFEST_RECEIPT_SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.manifest_hash,
        }
        return self._publish_or_verify(
            self.root / "manifest.json",
            payload,
            kind="manifest",
            context="generation manifest receipt",
        )

    def publish_selection(
        self,
        *,
        selection: Mapping[str, Any],
        screens: Sequence[Mapping[str, Any]],
        guard_reference: Mapping[str, Any],
        guard_reference_phase_receipt_checksum: str,
    ) -> GenerationReceiptRef:
        if not isinstance(selection, Mapping):
            raise TypeError("generation selection must be an object")
        if len(screens) != 2 or any(not isinstance(item, Mapping) for item in screens):
            raise ValueError("generation selection requires exactly two screens")
        if not isinstance(guard_reference, Mapping):
            raise TypeError("generation guard reference must be an object")
        require_sha256_checksum(
            guard_reference_phase_receipt_checksum,
            "generation guard phase receipt checksum",
        )
        selection_copy = dict(selection)
        screen_copies = [dict(item) for item in screens]
        guard_copy = dict(guard_reference)
        if selection_copy.get("acceptance_authority") is not False:
            raise ValueError("generation selection cannot carry acceptance authority")
        screen_hashes = [canonical_json_hash(item) for item in screen_copies]
        if selection_copy.get("screen_evidence_hashes") != screen_hashes:
            raise ValueError("generation selection screen order/hash mismatch")
        guard_hash = canonical_json_hash(guard_copy)
        if selection_copy.get("guard_reference_evidence_hash") != guard_hash:
            raise ValueError("generation selection guard reference mismatch")
        payload = {
            "schema_version": GENERATION_SELECTION_RECEIPT_SCHEMA_VERSION,
            "manifest_hash": self.manifest.manifest_hash,
            "selection": selection_copy,
            "selection_hash": canonical_json_hash(selection_copy),
            "screens": screen_copies,
            "screen_evidence_hashes": screen_hashes,
            "guard_reference": guard_copy,
            "guard_reference_evidence_hash": guard_hash,
            "guard_reference_phase_receipt_checksum": (
                guard_reference_phase_receipt_checksum
            ),
            "acceptance_authority": False,
        }
        return self._publish_or_verify(
            self.root / "selection.json",
            payload,
            kind="selection",
            context="generation selection receipt",
        )

    def load_selection(self) -> LoadedGenerationSelection | None:
        path = self.root / "selection.json"
        if not path.exists():
            return None
        payload, ref = self._load(
            path,
            kind="selection",
            context="generation selection receipt",
        )
        expected_fields = {
            "schema_version",
            "manifest_hash",
            "selection",
            "selection_hash",
            "screens",
            "screen_evidence_hashes",
            "guard_reference",
            "guard_reference_evidence_hash",
            "guard_reference_phase_receipt_checksum",
            "acceptance_authority",
        }
        if set(payload) != expected_fields or (
            payload.get("schema_version")
            != GENERATION_SELECTION_RECEIPT_SCHEMA_VERSION
            or payload.get("manifest_hash") != self.manifest.manifest_hash
            or payload.get("acceptance_authority") is not False
        ):
            raise ValueError("generation selection receipt contract mismatch")
        selection = payload.get("selection")
        screens = payload.get("screens")
        guard = payload.get("guard_reference")
        if (
            not isinstance(selection, Mapping)
            or not isinstance(screens, list)
            or len(screens) != 2
            or any(not isinstance(item, Mapping) for item in screens)
            or not isinstance(guard, Mapping)
        ):
            raise ValueError("generation selection receipt payload is invalid")
        if canonical_json_hash(selection) != payload.get("selection_hash"):
            raise ValueError("generation selection receipt selection hash mismatch")
        screen_hashes = [canonical_json_hash(item) for item in screens]
        if screen_hashes != payload.get("screen_evidence_hashes"):
            raise ValueError("generation selection receipt screen hash mismatch")
        if canonical_json_hash(guard) != payload.get(
            "guard_reference_evidence_hash"
        ):
            raise ValueError("generation selection receipt guard hash mismatch")
        require_sha256_checksum(
            payload.get("guard_reference_phase_receipt_checksum"),
            "generation guard phase receipt checksum",
        )
        return LoadedGenerationSelection(
            selection=dict(selection),
            screens=(dict(screens[0]), dict(screens[1])),
            guard_reference=dict(guard),
            guard_reference_phase_receipt_checksum=payload[
                "guard_reference_phase_receipt_checksum"
            ],
            receipt_ref=ref,
        )

    def publish_execution(self, record: Mapping[str, Any]) -> GenerationReceiptRef:
        if not isinstance(record, Mapping):
            raise TypeError("generation record must be an object")
        loaded_selection = self.load_selection()
        payload = {
            "schema_version": GENERATION_EXECUTION_RECEIPT_SCHEMA_VERSION,
            "manifest_hash": self.manifest.manifest_hash,
            "selection_receipt_checksum": (
                loaded_selection.receipt_ref.checksum
                if loaded_selection is not None
                else None
            ),
            "record": dict(record),
            "record_hash": canonical_json_hash(record),
        }
        return self._publish_or_verify(
            self.root / "execution.json",
            payload,
            kind="execution",
            context="generation execution receipt",
        )

    def load_execution(self) -> tuple[dict[str, Any], GenerationReceiptRef] | None:
        path = self.root / "execution.json"
        if not path.exists():
            return None
        payload, ref = self._load(
            path,
            kind="execution",
            context="generation execution receipt",
        )
        if set(payload) != {
            "schema_version",
            "manifest_hash",
            "selection_receipt_checksum",
            "record",
            "record_hash",
        } or (
            payload.get("schema_version")
            != GENERATION_EXECUTION_RECEIPT_SCHEMA_VERSION
            or payload.get("manifest_hash") != self.manifest.manifest_hash
        ):
            raise ValueError("generation execution receipt contract mismatch")
        record = payload.get("record")
        if not isinstance(record, Mapping) or canonical_json_hash(record) != payload.get(
            "record_hash"
        ):
            raise ValueError("generation execution receipt record mismatch")
        loaded_selection = self.load_selection()
        expected_selection_checksum = (
            loaded_selection.receipt_ref.checksum
            if loaded_selection is not None
            else None
        )
        if payload.get("selection_receipt_checksum") != expected_selection_checksum:
            raise ValueError("generation execution selection binding mismatch")
        return dict(record), ref

    def _publish_or_verify(
        self,
        path: Path,
        payload: Mapping[str, Any],
        *,
        kind: str,
        context: str,
    ) -> GenerationReceiptRef:
        encoded = canonical_json_bytes(dict(payload), trailing_newline=True)
        if len(encoded) > MAX_GENERATION_RECEIPT_BYTES:
            raise ValueError(f"{context} exceeds its byte limit")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            create_once_bytes(path, encoded)
        except FileExistsError:
            existing = read_bounded_regular_file(
                path,
                max_bytes=MAX_GENERATION_RECEIPT_BYTES,
                label=context,
            )
            if existing != encoded:
                raise ValueError(f"existing {context} differs from requested payload")
        return GenerationReceiptRef(
            kind=kind,
            path=str(path.resolve(strict=False)),
            checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )

    def _load(
        self,
        path: Path,
        *,
        kind: str,
        context: str,
    ) -> tuple[dict[str, Any], GenerationReceiptRef]:
        encoded = read_bounded_regular_file(
            path,
            max_bytes=MAX_GENERATION_RECEIPT_BYTES,
            label=context,
        )
        payload = strict_json_object_loads(encoded, context=context)
        if encoded != canonical_json_bytes(payload, trailing_newline=True):
            raise ValueError(f"{context} is not canonical JSON")
        return payload, GenerationReceiptRef(
            kind=kind,
            path=str(path.resolve(strict=False)),
            checksum="sha256:" + hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
        )


def load_generation_manifest_receipt(
    ref: GenerationReceiptRef,
    *,
    manifest: GenerationManifest,
) -> GenerationReceiptRef:
    """Load the manifest through its canonical journal location and exact ref."""

    journal = _journal_for_ref(ref, manifest=manifest, expected_kind="manifest")
    loaded_ref = journal.bind_manifest()
    if loaded_ref != ref:
        raise ValueError("generation manifest receipt ref mismatch")
    return loaded_ref


def load_generation_selection_receipt(
    ref: GenerationReceiptRef,
    *,
    manifest: GenerationManifest,
) -> LoadedGenerationSelection:
    """Load one immutable selector receipt without trusting ledger projections."""

    journal = _journal_for_ref(ref, manifest=manifest, expected_kind="selection")
    loaded = journal.load_selection()
    if loaded is None or loaded.receipt_ref != ref:
        raise ValueError("generation selection receipt ref mismatch")
    return loaded


def _journal_for_ref(
    ref: GenerationReceiptRef,
    *,
    manifest: GenerationManifest,
    expected_kind: str,
) -> GenerationReceiptJournal:
    if not isinstance(ref, GenerationReceiptRef) or ref.kind != expected_kind:
        raise TypeError(f"generation {expected_kind} receipt ref is invalid")
    path = Path(ref.path).resolve(strict=False)
    if path.parent.name != manifest.manifest_hash:
        raise ValueError(f"generation {expected_kind} receipt is outside manifest root")
    journal = GenerationReceiptJournal(path.parent.parent, manifest=manifest)
    expected_path = journal.root / f"{expected_kind}.json"
    if path != expected_path.resolve(strict=False):
        raise ValueError(f"generation {expected_kind} receipt path mismatch")
    return journal


__all__ = [
    "GENERATION_EXECUTION_RECEIPT_SCHEMA_VERSION",
    "GENERATION_MANIFEST_RECEIPT_SCHEMA_VERSION",
    "GENERATION_RECEIPT_REF_SCHEMA_VERSION",
    "GENERATION_SELECTION_RECEIPT_SCHEMA_VERSION",
    "GenerationReceiptJournal",
    "GenerationReceiptRef",
    "LoadedGenerationSelection",
    "load_generation_manifest_receipt",
    "load_generation_selection_receipt",
]
