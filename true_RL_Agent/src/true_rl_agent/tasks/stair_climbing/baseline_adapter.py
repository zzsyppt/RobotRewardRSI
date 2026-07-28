"""Task-owned bridge from frozen Stair Phase 0 identity to runtime baseline."""

from __future__ import annotations

from pathlib import Path

from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec,
    BaselineManifest,
    BaselineSourceBinding,
    CheckpointProvenance,
)
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.stair_climbing.phase0_contract import (
    StairPhase0Bundle,
    verify_checkpoint_file,
)


def build_stair_runtime_baseline_manifest(
    *,
    bundle: StairPhase0Bundle,
    checkpoint_path: str | Path,
) -> BaselineManifest:
    """Derive one runtime manifest from the admitted, descriptor-bound source."""

    if not isinstance(bundle, StairPhase0Bundle):
        raise TypeError("bundle must be StairPhase0Bundle")
    report = bundle.load_report(require_admitted=True)
    verification = verify_checkpoint_file(checkpoint_path, bundle.baseline)
    source_manifest = bundle.baseline.to_dict()
    source = source_manifest["source"]
    compatibility = source_manifest["compatibility"]
    artifact_ref = str(source["artifact_ref"])
    checkpoint = CheckpointRef(
        uri=artifact_ref,
        checksum=f"sha256:{verification.sha256}",
        role="warm_start",
        sealed=True,
        metadata={
            "local_path": str(verification.path),
            "size_bytes": verification.size_bytes,
            "sealed_by": "stair_phase0_admission",
        },
    )
    return BaselineManifest(
        baseline_id=bundle.baseline.baseline_id,
        task_id=str(source_manifest["target_task_id"]),
        checkpoint=checkpoint,
        provenance=CheckpointProvenance(
            source_run_id=artifact_ref,
            source_task_id=str(source["source_task_id"]),
            created_by=str(source["provider"]),
        ),
        compatibility=BaselineCompatibilitySpec(
            backend_id=str(compatibility["backend_id"]),
            robot=str(compatibility["robot"]),
            observation_schema_hash=str(
                compatibility["policy_observation_schema_hash"]
            ),
            action_schema_hash=str(compatibility["action_schema_hash"]),
            policy_architecture=str(compatibility["policy_architecture"]),
        ),
        source_binding=BaselineSourceBinding(
            source_schema_version=str(source_manifest["schema_version"]),
            source_manifest_hash=bundle.baseline.manifest_hash,
            trust_anchor_hash=bundle.descriptor.descriptor_hash,
            admission_evidence_hash=report.report_hash,
        ),
    )


__all__ = ["build_stair_runtime_baseline_manifest"]
