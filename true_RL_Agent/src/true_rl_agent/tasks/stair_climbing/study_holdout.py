"""Study-specific Stair final geometry and seeds, committed before search."""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_digest
from true_rl_agent.json_contract import canonical_json_bytes, load_strict_json_object
from true_rl_agent.tasks.stair_climbing.unified_evaluation import (
    StairUnifiedEvaluationSuite, resolve_stair_suite_cells, stair_geometry_variant,
)

LEGACY_FINAL_HOLDOUT_POLICY = {
    "schema_version": "true_rl_agent.task.stair_climbing.final_holdout_policy.v1",
    "source_cohort": "sealed_validation",
    "height_and_tread_shift_over_support_length": [0.025, 0.045],
    "generator": "sha256_study_nonce_per_base_geometry_v2",
    "seed_namespace": "study_scoped_final_v1",
    "replication_must_be_declared": True,
}
FINAL_HOLDOUT_POLICY = {
    **LEGACY_FINAL_HOLDOUT_POLICY,
    "schema_version": "true_rl_agent.task.stair_climbing.final_holdout_policy.v2",
    "generator": "sha256_study_nonce_per_base_geometry_v3",
    "roughness_scaling": "preserve_amplitude_over_minimum_riser",
}


class _HoldoutGeometryCollision(ValueError):
    """A nonce repeats source geometry or another cell in its own sealed cohort."""


@dataclass(frozen=True)
class StairStudyHoldout:
    study_id: str
    source_suite_hash: str
    nonce: str
    replication_of: str | None = None

    def __post_init__(self):
        if not isinstance(self.study_id, str) or not self.study_id.strip():
            raise ValueError("Stair final holdout requires a study identity")
        require_sha256_digest(self.source_suite_hash, "Stair final source suite hash")
        require_sha256_digest(self.nonce, "Stair final nonce")
        if self.replication_of is not None:
            require_sha256_digest(self.replication_of, "Stair replication source commitment")

    def to_private_dict(self) -> dict[str, Any]:
        return {"schema_version": "true_rl_agent.task.stair_climbing.study_holdout.v1",
                "study_id": self.study_id, "source_suite_hash": self.source_suite_hash,
                "nonce": self.nonce, "replication_of": self.replication_of}

    @property
    def commitment(self) -> str:
        return canonical_json_hash(self.to_private_dict())

    def geometry(self, template, source: Mapping[str, Any], *, length_m: float,
                 training_profile, preserve_surface_ratio: bool = True) -> dict[str, Any]:
        # C2 and its flat control use the same macro-geometry key.
        key = canonical_json_hash(source.get("base_geometry", source))
        digest = hashlib.sha256((self.nonce + ":" + key).encode()).digest()
        low, high = FINAL_HOLDOUT_POLICY["height_and_tread_shift_over_support_length"]
        height = (low + (high - low) * digest[0] / 255) * (1 if digest[2] & 1 else -1)
        tread = (low + (high - low) * digest[1] / 255) * (1 if digest[3] & 1 else -1)
        return stair_geometry_variant(source, stratum_id=template["stratum_id"],
            training_profile=training_profile, support_length_m=length_m,
            height_fraction=height, tread_fraction=tread, surface_phase_shift=1 + digest[4] % 7,
            preserve_surface_ratio=preserve_surface_ratio)

    def bind_suite(self, source, *, training_profile):
        if source.suite_hash != self.source_suite_hash:
            raise ValueError("Stair study holdout belongs to another source suite")
        payload = source.to_dict()
        policy = payload.get("final_holdout_policy")
        if policy not in (FINAL_HOLDOUT_POLICY, LEGACY_FINAL_HOLDOUT_POLICY):
            raise ValueError("Stair source suite has no frozen study holdout policy")
        length = payload["scale_context"]["support_length_m"]
        templates = {item["template_id"]: item for item in payload["cell_templates"]}
        payload["cohort_geometry_overrides"]["sealed_validation"] = {
            template_id: self.geometry(templates[template_id], geometry, length_m=length,
                                       training_profile=training_profile,
                                       preserve_surface_ratio=policy == FINAL_HOLDOUT_POLICY)
            for template_id, geometry in payload["cohort_geometry_overrides"]["sealed_validation"].items()
        }
        payload["study_instance"] = {
            "source_suite_hash": self.source_suite_hash,
            "instance_key_hash": hashlib.sha256(self.nonce.encode()).hexdigest(),
        }
        sealed = [cell.geometry_digest for cell in
                  resolve_stair_suite_cells(payload, training_profile=training_profile)
                  if cell.cohort == "sealed_validation"]
        if len(set(sealed)) != len(sealed):
            raise _HoldoutGeometryCollision("Stair final sealed cohort contains duplicate geometries")
        existing = {cell.geometry_digest for cell in source.cells}
        if existing & set(sealed):
            raise _HoldoutGeometryCollision("Stair final geometry overlaps a previously available course")
        return StairUnifiedEvaluationSuite.freeze(payload, training_profile=training_profile)

    @classmethod
    def create(cls, *, path: Path, study_id: str, source_suite: StairUnifiedEvaluationSuite,
               training_profile,
               replication: "StairStudyHoldout | None" = None) -> "StairStudyHoldout":
        if replication is not None and replication.source_suite_hash != source_suite.suite_hash:
            raise ValueError("Stair replication must use the identical source suite")
        # The frozen byte-derived offsets can coincide with another existing
        # course. Enforce the existing geometric exclusion before commitment;
        # no policy outcome or difficulty measurement participates in the draw.
        for _ in range(64):
            result = cls(study_id, source_suite.suite_hash,
                         replication.nonce if replication else secrets.token_hex(32),
                         replication.commitment if replication else None)
            try:
                result.bind_suite(source_suite, training_profile=training_profile)
            except _HoldoutGeometryCollision:
                if replication is not None:
                    raise
            else:
                break
        else:
            raise ValueError("Stair final holdout exhausted 64 geometry draws before commitment")
        path.parent.mkdir(parents=True, exist_ok=True)
        # The existing session owns resume; creation never silently redraws a cohort.
        with path.open("xb") as stream:
            stream.write(canonical_json_bytes(result.to_private_dict(), trailing_newline=True))
        return result

    @classmethod
    def load(cls, path: Path, *, expected_commitment: str | None = None) -> "StairStudyHoldout":
        payload = load_strict_json_object(path, require_canonical=True, trailing_newline=True)
        if set(payload) != {"schema_version", "study_id", "source_suite_hash", "nonce", "replication_of"} or payload["schema_version"] != "true_rl_agent.task.stair_climbing.study_holdout.v1":
            raise ValueError("Stair study holdout fields mismatch")
        result = cls(**{key: payload[key] for key in ("study_id", "source_suite_hash", "nonce", "replication_of")})
        if expected_commitment is not None and result.commitment != expected_commitment:
            raise ValueError("Stair final holdout commitment mismatch")
        return result

