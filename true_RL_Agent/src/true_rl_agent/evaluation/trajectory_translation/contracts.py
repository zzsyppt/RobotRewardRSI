"""Small source-bound contracts shared by measurement and temporal operators."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

EXTRACTOR_VERSION = "trajectory.numeric.v1"
CATALOG_VERSION = "trajectory.p0.measurements.v2"


def stable_id(kind: str, *parts: Any) -> str:
    encoded = json.dumps(parts, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{kind}:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class NumericEpisode:
    """One actor in one episode, with explicit source-array row locators.

    Missing rows may be represented by step gaps; missing channel components by
    NaN. A reset belongs to a different episode and is rejected here.
    """

    source: dict[str, Any]
    run_index: int
    env_index: int
    steps: np.ndarray
    dt_s: float
    arrays: dict[str, np.ndarray]
    entities: dict[str, tuple[str, ...]]
    source_chunks: tuple[dict[str, Any], ...]
    chunk_indices: np.ndarray
    chunk_rows: np.ndarray
    task: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source["acceptance_split"] != "public_dev":
            raise ValueError("trajectory evidence accepts public_dev only")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ValueError("control dt must be positive and finite")
        self.steps = np.asarray(self.steps)
        if (self.steps.ndim != 1 or not len(self.steps) or self.steps.dtype.kind not in "iu"
                or np.any(self.steps < 1) or np.any(self.steps[1:] <= self.steps[:-1])):
            raise ValueError("episode steps must increase; resets require a new episode")
        for name, values in self.arrays.items():
            if len(values) != len(self.steps):
                raise ValueError(f"{name}: channel length differs from episode clock")
        for values in (self.chunk_indices, self.chunk_rows):
            if (np.shape(values) != self.steps.shape or np.asarray(values).dtype.kind not in "iu"):
                raise ValueError("source row locator length mismatch")
        if (np.any(self.chunk_indices < 0) or np.any(self.chunk_indices >= len(self.source_chunks))
                or np.any(self.chunk_rows < 0)):
            raise ValueError("invalid source row locator")
        for names in self.entities.values():
            if len(names) != len(set(names)):
                raise ValueError("duplicate entity binding")

    @property
    def episode_id(self) -> str:
        # Display labels and checkpoint identity are not archive identity.
        return stable_id("episode", self.source["manifest_sha256"], self.source["trial_id"],
                         self.run_index, self.env_index)

    @property
    def times(self) -> np.ndarray:
        return self.steps * self.dt_s

    @property
    def adjacent(self) -> np.ndarray:
        return np.diff(self.steps) == 1

    def reference(self, channels: tuple[str, ...], start: int = 0,
                  stop: int | None = None) -> dict[str, Any]:
        stop = len(self.steps) if stop is None else stop
        if not 0 <= start < stop <= len(self.steps):
            raise ValueError("reference interval outside episode")
        spans = []
        first = start
        for end in range(start + 1, stop + 1):
            if (end == stop or self.chunk_indices[end] != self.chunk_indices[first]
                    or self.chunk_rows[end] != self.chunk_rows[end - 1] + 1):
                chunk = self.source_chunks[int(self.chunk_indices[first])]
                spans.append({"file": chunk["file"], "sha256": chunk["sha256"],
                              "rows_half_open": [int(self.chunk_rows[first]), int(self.chunk_rows[end - 1]) + 1],
                              "env_index": self.env_index})
                first = end
        return {"manifest_sha256": self.source["manifest_sha256"], "run_index": self.run_index,
                "channels": list(channels), "steps_half_open": [int(self.steps[start]), int(self.steps[stop - 1]) + 1],
                "chunks": spans}

    def record(self, key: str, record_type: str, entities: tuple[str, ...],
               channels: tuple[str, ...], value: dict[str, Any], *, unit: str,
               frame: str, config: dict[str, Any], start: int = 0,
               stop: int | None = None) -> dict[str, Any]:
        ref = self.reference(channels, start, stop)
        return {"record_id": stable_id("evidence", self.episode_id, EXTRACTOR_VERSION,
                                       config, key, record_type, entities, channels, unit, frame,
                                       ref["steps_half_open"], value),
                "record_type": record_type, "episode_id": self.episode_id, "key": key,
                "entity_ids": list(entities), "unit": unit, "frame": frame,
                "extractor_version": EXTRACTOR_VERSION, "catalog_version": CATALOG_VERSION,
                "measurement_config": config, "evidence_ref": ref, "value": value}


@dataclass
class PredicateTrack:
    episode_id: str
    key: str
    entity_ids: tuple[str, ...]
    channels: tuple[str, ...]
    values: np.ndarray  # -1 unknown, 0 false, 1 true
    config: dict[str, Any]

    def __post_init__(self) -> None:
        self.values = np.asarray(self.values)
        if self.values.ndim != 1 or not np.isin(self.values, [-1, 0, 1]).all():
            raise ValueError("predicate must be a scalar three-valued track")
        self.values = self.values.astype(np.int8)


def predicate(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.where(valid, np.asarray(values, dtype=bool).astype(np.int8), -1)
