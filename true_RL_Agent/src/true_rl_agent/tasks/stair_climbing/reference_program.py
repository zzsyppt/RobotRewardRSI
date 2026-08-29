"""Canonical official R0 resolver and Stair-owned qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1

from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.local_paths import (
    resolve_existing_directory,
    resolve_existing_file,
)
from true_rl_agent.tasks.stair_climbing.reward_program import (
    QualifiedStairRewardProgram,
    qualify_stair_reference_reward_program,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    stair_runtime_projection_from_manifest,
)
from true_rl_agent.tasks.stair_climbing.vnext_reward_contract import (
    build_stair_vnext_reward_runtime_contract,
)
from true_rl_agent.training_programs import (
    VerifiedReferenceProgramArtifact,
    resolve_reference_program_artifact,
)


@dataclass(frozen=True)
class StairOfficialReferenceProgram:
    artifact: VerifiedReferenceProgramArtifact
    reward_program: QualifiedStairRewardProgram
    source_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, VerifiedReferenceProgramArtifact):
            raise TypeError("Stair official reference artifact is invalid")
        if not isinstance(self.reward_program, QualifiedStairRewardProgram):
            raise TypeError("Stair official reward program is invalid")
        if self.artifact.manifest.manifest_hash != (
            self.reward_program.subject_manifest_hash
        ):
            raise ValueError("Stair official reference identity chain is broken")
        if self.artifact.manifest.source_checksum != (
            self.reward_program.program_source_checksum
        ):
            raise ValueError("Stair official reference source chain is broken")


def resolve_official_stair_reference_program(
    config: StairClimbingTaskConfig,
    *,
    project_root: Path | None = None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
) -> StairOfficialReferenceProgram:
    """Resolve actual repository bytes, then qualify the exact same payload."""

    artifact, source_path = resolve_official_stair_reference_artifact(
        config,
        project_root=project_root,
        resolved_cell_manifest=resolved_cell_manifest,
    )
    if config.is_unified_protocol:
        if resolved_cell_manifest is None:
            raise ValueError("unified Stair reference requires the resolved manifest")
        contract = build_stair_vnext_reward_runtime_contract(
            projection=stair_runtime_projection_from_manifest(
                resolved_cell_manifest
            ),
            authority=config.vnext_reward_authority,
        )
    else:
        assert config.protocol_v2 is not None
        contract = config.protocol_v2.complete_reward_runtime
    reward_program = qualify_stair_reference_reward_program(
        artifact=artifact,
        contract=contract,
    )
    return StairOfficialReferenceProgram(
        artifact=artifact,
        reward_program=reward_program,
        source_path=source_path,
    )


def resolve_official_stair_reference_artifact(
    config: StairClimbingTaskConfig,
    *,
    project_root: Path | None = None,
    resolved_cell_manifest: ResolvedCellManifestV1 | None = None,
) -> tuple[VerifiedReferenceProgramArtifact, Path]:
    """Resolve the one trusted source artifact without qualifying candidate code."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("official Stair reference requires StairClimbingTaskConfig")
    root = resolve_existing_directory(
        project_root or Path(__file__).resolve().parents[4],
        field_name="official Stair project root",
    )
    if config.is_unified_protocol:
        reference = dict(config.unified_protocol.reference_program)
        equivalence = config.vnext_reward_authority.runtime_authority_payload()[
            "reference_equivalence"
        ]
    else:
        assert config.protocol_v2 is not None
        protocol_v2 = config.protocol_v2.to_public_dict()
        reference = protocol_v2["reference_program"]
        equivalence = protocol_v2["complete_reward_runtime"][
            "reference_equivalence"
        ]
    repository_source_path = str(reference["repository_source_path"])
    reference_id = str(reference["reference_id"])
    authorship_origin = str(reference["required_authorship_origin"])
    source_path = resolve_existing_file(
        root / repository_source_path,
        field_name="official Stair reference source",
    )
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("official Stair reference source escapes project root") from exc
    artifact = resolve_reference_program_artifact(
        source_path,
        reference_id=reference_id,
        task_id=STAIR_TASK_ID,
        authorship_origin=authorship_origin,
        upstream_project=str(equivalence["upstream_project"]),
        upstream_revision=str(equivalence["upstream_revision"]),
        upstream_source_path=str(equivalence["upstream_source_path"]),
        capability_profile_hash=(
            config.candidate_capability_profile.profile_hash
        ),
    )
    return artifact, source_path


__all__ = [
    "StairOfficialReferenceProgram",
    "resolve_official_stair_reference_artifact",
    "resolve_official_stair_reference_program",
]
