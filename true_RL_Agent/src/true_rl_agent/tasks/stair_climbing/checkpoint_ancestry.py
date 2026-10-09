"""Stair-owned admission of checkpoints from evaluation-only predecessors."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.runner.argv_contract import unique_option_value
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.row_catalog import stair_robot_row


@dataclass(frozen=True)
class StairCheckpointAncestryAdmission:
    """Compiled manifests, ordered current-to-origin, from a verified lineage.

    Construct through ``build_stair_checkpoint_ancestry_admission``. This is an
    in-process task policy, not an artifact-supplied list of admissible hashes.
    """

    manifests: tuple[ResolvedCellManifestV1, ...]
    training_binding_hash: str

    def __call__(
        self,
        checkpoint: CheckpointRef,
        current_manifest: ResolvedCellManifestV1,
        argv: tuple[str, ...],
    ) -> None:
        current = self.manifests[0]
        if current_manifest.manifest_hash != current.manifest_hash:
            raise ValueError("Stair checkpoint admission targets another current manifest")
        if checkpoint.role != "eval":
            raise ValueError("Stair successor ancestry is evaluation-only")
        source_hash = checkpoint.metadata.get("resolved_cell_manifest_hash")
        source = next(
            (manifest for manifest in self.manifests if manifest.manifest_hash == source_hash),
            None,
        )
        if source is None:
            raise ValueError("checkpoint resolved cell manifest is not a verified Stair ancestor")
        if unique_option_value(argv, "--training-binding-hash") != self.training_binding_hash:
            raise ValueError("Stair ancestor evaluation training binding mismatch")
        # Ordinary current-cell jobs retain their existing provenance contract.
        # Crossing a protocol version additionally needs the source train receipt's
        # binding; it must never be filled from the target evaluation config.
        source_training_binding = checkpoint.metadata.get("training_binding_hash")
        if source != current or source_training_binding is not None:
            if source_training_binding != self.training_binding_hash:
                raise ValueError("Stair ancestor checkpoint training binding is missing or mismatched")
        source_identity = checkpoint.metadata.get("resolved_cell_identity")
        if source_identity is not None and source_identity != source.summary_identity():
            raise ValueError("Stair ancestor checkpoint source cell identity mismatch")


def build_stair_checkpoint_ancestry_admission(
    config: StairClimbingTaskConfig,
    current_manifest: ResolvedCellManifestV1,
) -> StairCheckpointAncestryAdmission:
    """Recompile each declared predecessor and verify every edge, not just hashes.

    ``StairUnifiedProtocol.load`` already checks cycle freedom, the declared base
    protocol hash, physical robot binding, and all training/reward/authoring
    projections. Here the declared resolved identities are checked against real
    manifest compilation and the full resolved formal TrainingBinding.
    """

    if not config.is_unified_protocol:
        raise ValueError("Stair checkpoint ancestry requires a unified protocol")
    manifests: list[ResolvedCellManifestV1] = []
    training_binding = None
    expected_manifest = current_manifest
    while True:
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=config.cell_variant_id,
            config=config,
            plugin=StairClimbingTaskPlugin(config),
        )
        if manifest.manifest_hash != expected_manifest.manifest_hash:
            raise ValueError("Stair checkpoint ancestry current manifest mismatch")
        bound_config = replace(
            config,
            resolved_cell_task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
        )
        binding = bound_config.formal_training_binding()
        if training_binding is None:
            training_binding = binding
        elif binding.to_private_dict() != training_binding.to_private_dict():
            raise ValueError("Stair evaluation-only ancestor changed formal training binding")
        manifests.append(manifest)
        lineage = config.unified_protocol.training_lineage
        if lineage is None:
            break
        source_row = stair_robot_row(str(lineage["source_cell_variant_id"]))
        source_config = StairClimbingTaskConfig.defaults(
            config.protocol_v2_path.with_name(Path(source_row.protocol_relative_path).name),
            cell_variant_id=source_row.cell_variant_id,
        )
        source_manifest = resolve_stair_cell_manifest(
            cell_variant_id=source_config.cell_variant_id,
            config=source_config,
            plugin=StairClimbingTaskPlugin(source_config),
        )
        if (
            source_manifest.manifest_hash != lineage["source_resolved_cell_manifest_hash"]
            or source_manifest.resolved_cell_task_protocol_hash
            != lineage["source_resolved_cell_task_protocol_hash"]
        ):
            raise ValueError("Stair training predecessor resolved manifest identity mismatch")
        for name in ("robot_profile", "runtime_adapter", "resolved_roles", "provided_capabilities"):
            if getattr(manifest, name) != getattr(source_manifest, name):
                raise ValueError(f"Stair evaluation-only ancestor changed {name}")
        for name in (
            "policy_io_schema_hash", "observation_content_hash", "action_content_hash",
            "sensor_schema_hash", "reward_input_schema_hash",
        ):
            if manifest.contract_hashes[name] != source_manifest.contract_hashes[name]:
                raise ValueError(f"Stair evaluation-only ancestor changed {name}")
        config, expected_manifest = source_config, source_manifest
    assert training_binding is not None
    return StairCheckpointAncestryAdmission(
        manifests=tuple(manifests),
        training_binding_hash=str(training_binding.binding_hash),
    )
