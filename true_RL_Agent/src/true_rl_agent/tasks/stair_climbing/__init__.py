"""Task-owned Stair contracts and staged production support.

Importing this package remains inert: it does not register a task or import Isaac Lab.
"""

from true_rl_agent.tasks.stair_climbing.baseline_adapter import (
    build_stair_runtime_baseline_manifest,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    aggregate_stair_seed_scores,
    assert_primary_outcome_counts,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import (
    DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF,
    StairBaselineManifestV1,
    StairExtensionDescriptorV1,
    StairPhase0Bundle,
    StairPhase0ReportV1,
    StairProtocolV1,
    verify_checkpoint_file,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import StairProtocolV2Draft

__all__ = [
    "DEFAULT_STAIR_EXTENSION_DESCRIPTOR_REF",
    "StairBaselineManifestV1",
    "StairExtensionDescriptorV1",
    "StairPhase0Bundle",
    "StairPhase0ReportV1",
    "StairProtocolV1",
    "StairProtocolV2Draft",
    "aggregate_stair_seed_scores",
    "assert_primary_outcome_counts",
    "build_stair_runtime_baseline_manifest",
    "verify_checkpoint_file",
]
