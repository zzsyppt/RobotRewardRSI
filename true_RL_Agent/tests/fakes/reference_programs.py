from __future__ import annotations

from pathlib import Path

from true_rl_agent.training_programs import (
    VerifiedReferenceProgramArtifact,
    resolve_reference_program_artifact,
)

_REFERENCE_SOURCE = """from true_rl_agent.candidates.components import TrainingComponents


def build_training_components() -> TrainingComponents:
    return TrainingComponents(name=\"phase-b-verified-reference-fixture\")
"""


def make_verified_stair_reference(
    root: Path,
    *,
    task_id: str,
    capability_profile_hash: str,
    authorship_origin: str = "official_upstream",
) -> VerifiedReferenceProgramArtifact:
    source_path = root / "training_code" / "components.py"
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(_REFERENCE_SOURCE, encoding="utf-8")
    return resolve_reference_program_artifact(
        source_path,
        reference_id="stair-r0-phase-b-verified-fixture",
        task_id=task_id,
        authorship_origin=authorship_origin,
        upstream_project="IsaacLab",
        upstream_revision="phase-b-verified-fixture",
        upstream_source_path=(
            "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/"
            "velocity/config/g1/rough_env_cfg.py"
        ),
        capability_profile_hash=capability_profile_hash,
    )


__all__ = ["make_verified_stair_reference"]
