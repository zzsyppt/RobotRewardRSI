"""Bounded numeric chunks, published once by a final content-bound manifest.

Tasks own array meanings and coverage. This module only owns file integrity and
storage limits; it neither interprets episodes nor grants public visibility.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np

NUMERIC_EVIDENCE_SCHEMA = "true_rl_agent.numeric_evidence.v1"
NUMERIC_EVIDENCE_REF_SCHEMA = "true_rl_agent.numeric_evidence.ref.v1"
DEFAULT_MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_CHUNK_RAW_BYTES = 64 * 1024**2


class NumericEvidenceError(RuntimeError):
    """A numeric archive is incomplete, inconsistent, or exceeds its budget."""


def _json_bytes(value: Any) -> bytes:
    # Task contracts may deliberately bind declaration order (for example term
    # or pattern order). Hash the stored bytes without rewriting that order.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class NumericEvidenceWriter:
    """Write bounded chunks to a fresh directory; complete publishes the ref."""

    def __init__(self, root: Path, identity: Mapping[str, Any], max_bytes: int = DEFAULT_MAX_ARCHIVE_BYTES) -> None:
        self.root = Path(root).resolve()
        self.identity = json.loads(_json_bytes(dict(identity)))
        if max_bytes <= 0:
            raise ValueError("numeric evidence byte budget must be positive")
        self.max_bytes = max_bytes
        self.root.mkdir(parents=True, exist_ok=False)
        self._chunks: list[dict[str, Any]] = []
        self._stored_bytes = 0
        self._complete = False

    def write_chunk(self, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> None:
        if self._complete:
            raise NumericEvidenceError("numeric evidence already published")
        values = {name: np.asarray(value) for name, value in arrays.items()}
        if not values or any(not name or value.dtype.kind not in "biuf" for name, value in values.items()):
            raise NumericEvidenceError("numeric evidence requires named numeric arrays")
        raw_bytes = sum(value.nbytes for value in values.values())
        if raw_bytes > MAX_CHUNK_RAW_BYTES:
            raise NumericEvidenceError("numeric evidence chunk exceeds raw byte budget")
        chunk_metadata = json.loads(_json_bytes(dict(metadata)))
        name = f"chunk_{len(self._chunks):06d}.npz"
        temporary = self.root / (name + ".partial")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **values)
        size = temporary.stat().st_size
        if self._stored_bytes + size > self.max_bytes:
            temporary.unlink()
            raise NumericEvidenceError("numeric evidence exceeds archive byte budget")
        destination = self.root / name
        temporary.replace(destination)
        self._chunks.append({
            "file": name, "sha256": _sha256(destination), "stored_bytes": size,
            "raw_bytes": raw_bytes, "metadata": chunk_metadata,
            "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in values.items()},
        })
        self._stored_bytes += size

    def complete(self, metadata: Mapping[str, Any]) -> dict[str, Any]:
        if self._complete:
            raise NumericEvidenceError("numeric evidence already published")
        if not self._chunks:
            raise NumericEvidenceError("numeric evidence has no chunks")
        manifest = {
            "schema_version": NUMERIC_EVIDENCE_SCHEMA, "complete": True,
            "identity": self.identity, "metadata": dict(metadata), "chunks": self._chunks,
            "stored_chunk_bytes": self._stored_bytes,
            "raw_bytes": sum(chunk["raw_bytes"] for chunk in self._chunks),
        }
        content = _json_bytes(manifest)
        if self._stored_bytes + len(content) > self.max_bytes:
            raise NumericEvidenceError("numeric evidence manifest exceeds archive byte budget")
        temporary = self.root / "manifest.json.partial"
        temporary.write_bytes(content)
        path = self.root / "manifest.json"
        temporary.replace(path)
        self._complete = True
        return {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(path), "manifest_sha256": hashlib.sha256(content).hexdigest()}


def validate_numeric_evidence_reference(reference: Mapping[str, Any]) -> None:
    """Validate a sealed locator without requiring the attachment to be online."""
    if reference.get("schema_version") != NUMERIC_EVIDENCE_REF_SCHEMA:
        raise NumericEvidenceError("numeric evidence reference schema mismatch")
    checksum = reference.get("manifest_sha256")
    if (set(reference) != {"schema_version", "manifest_path", "manifest_sha256"}
            or not isinstance(reference.get("manifest_path"), str) or not reference["manifest_path"]
            or not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)):
        raise NumericEvidenceError("numeric evidence reference fields mismatch")


def load_numeric_evidence_manifest(reference: Mapping[str, Any]) -> dict[str, Any]:
    validate_numeric_evidence_reference(reference)
    path = Path(reference["manifest_path"])
    try:
        content = path.read_bytes()
    except OSError as exc:
        raise NumericEvidenceError("numeric evidence manifest is not available") from exc
    if hashlib.sha256(content).hexdigest() != reference.get("manifest_sha256"):
        raise NumericEvidenceError("numeric evidence manifest hash mismatch")
    manifest = json.loads(content)
    if manifest.get("schema_version") != NUMERIC_EVIDENCE_SCHEMA or manifest.get("complete") is not True or not manifest.get("chunks"):
        raise NumericEvidenceError("numeric evidence is not complete")
    return manifest


def iter_numeric_evidence(reference: Mapping[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, np.ndarray]]]:
    """Read one verified chunk at a time, without retaining the whole archive."""
    manifest = load_numeric_evidence_manifest(reference)
    root = Path(reference["manifest_path"]).parent
    for index, chunk in enumerate(manifest["chunks"]):
        if chunk["file"] != f"chunk_{index:06d}.npz":
            raise NumericEvidenceError("numeric evidence chunk order mismatch")
        path = root / chunk["file"]
        if not path.is_file() or path.stat().st_size != chunk["stored_bytes"] or _sha256(path) != chunk["sha256"]:
            raise NumericEvidenceError("numeric evidence chunk hash or size mismatch")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(chunk["arrays"]):
                raise NumericEvidenceError("numeric evidence array fields mismatch")
            arrays = {name: archive[name] for name in archive.files}
        for name, array in arrays.items():
            expected = chunk["arrays"][name]
            if list(array.shape) != expected["shape"] or array.dtype.str != expected["dtype"]:
                raise NumericEvidenceError("numeric evidence array shape or dtype mismatch")
        yield chunk["metadata"], arrays
