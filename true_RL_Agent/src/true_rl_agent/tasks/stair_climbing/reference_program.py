"""Canonical official R0 resolver and Stair-owned qualification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
) -> StairOfficialReferenceProgram:
    """Resolve actual repository bytes, then qualify the exact same payload."""

    artifact, source_path = resolve_official_stair_reference_artifact(
        config,
        project_root=project_root,
    )
    reward_program = qualify_stair_reference_reward_program(
        artifact=artifact,
        contract=config.protocol_v2.complete_reward_runtime,
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
) -> tuple[VerifiedReferenceProgramArtifact, Path]:
    """Resolve the one trusted source artifact without qualifying candidate code."""

    if not isinstance(config, StairClimbingTaskConfig):
        raise TypeError("official Stair reference requires StairClimbingTaskConfig")
    root = resolve_existing_directory(
        project_root or Path(__file__).resolve().parents[4],
        field_name="official Stair project root",
    )
    protocol_v2 = config.protocol_v2.to_public_dict()
    reference = protocol_v2["reference_program"]
    equivalence = protocol_v2["complete_reward_runtime"]["reference_equivalence"]
    source_path = resolve_existing_file(
        root / str(reference["repository_source_path"]),
        field_name="official Stair reference source",
    )
    try:
        source_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("official Stair reference source escapes project root") from exc
    artifact = resolve_reference_program_artifact(
        source_path,
        reference_id=str(reference["reference_id"]),
        task_id=STAIR_TASK_ID,
        authorship_origin=str(reference["required_authorship_origin"]),
        upstream_project=str(equivalence["upstream_project"]),
        upstream_revision=str(equivalence["upstream_revision"]),
        upstream_source_path=str(equivalence["upstream_source_path"]),
        capability_profile_hash=(
            config.protocol_v2.capability_profile.profile_hash
        ),
    )
    return artifact, source_path


__all__ = [
    "StairOfficialReferenceProgram",
    "resolve_official_stair_reference_artifact",
    "resolve_official_stair_reference_program",
]
