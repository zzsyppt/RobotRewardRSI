"""Projection-derived complete-reward contract for non-legacy Stair cells."""

from __future__ import annotations

import copy

from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    StairCompleteRewardRuntimeContract,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection,
)
from true_rl_agent.tasks.stair_climbing.vnext_reward_authority import (
    StairVNextRewardAuthority,
)

STAIR_COMPLETE_REWARD_RUNTIME_VNEXT_SCHEMA_VERSION = (
    "true_rl_agent.task.stair_climbing.complete_reward_runtime.vnext.v1"
)


def build_stair_vnext_reward_runtime_contract(
    *,
    projection: StairRuntimeProjection,
    authority: StairVNextRewardAuthority | None = None,
    base_protocol: StairProtocolV1 | None = None,
    legacy_contract: StairCompleteRewardRuntimeContract | None = None,
) -> StairCompleteRewardRuntimeContract:
    """Preserve runner/output authority while replacing morphology-bound inputs."""

    if not isinstance(projection, StairRuntimeProjection):
        raise TypeError("Stair VNext reward contract requires a runtime projection")
    if projection.legacy_reward_adapter_id is not None:
        raise ValueError("legacy Stair projection cannot mint a VNext reward contract")
    if authority is not None:
        if not isinstance(authority, StairVNextRewardAuthority):
            raise TypeError("Stair VNext reward authority is invalid")
        if base_protocol is not None or legacy_contract is not None:
            raise ValueError("Stair VNext reward authority cannot be mixed with legacy")
        runtime_authority = authority.runtime_authority_payload()
        common_input_fields = authority.common_input_field_names
    else:
        if not isinstance(base_protocol, StairProtocolV1):
            raise TypeError("legacy Stair VNext contract requires StairProtocolV1")
        if not isinstance(legacy_contract, StairCompleteRewardRuntimeContract):
            raise TypeError("legacy Stair VNext contract requires its runtime authority")
        legacy = legacy_contract.to_public_dict()
        runtime_authority = {
            name: copy.deepcopy(legacy[name])
            for name in (
                "output_contract",
                "dispatcher",
                "reference_equivalence",
                "performance_gate",
            )
        }
        common_input_fields = None
    payload = {
        "schema_version": STAIR_COMPLETE_REWARD_RUNTIME_VNEXT_SCHEMA_VERSION,
        "input_schema": projection.reward_input_schema_vnext(),
        **runtime_authority,
    }
    contract = StairCompleteRewardRuntimeContract.from_mapping(
        payload,
        base=base_protocol,
        common_input_fields=common_input_fields,
    )
    if contract.input_schema_hash != projection.reward_input_schema_hash_vnext:
        raise ValueError("Stair VNext contract/projection input schema mismatch")
    return contract


__all__ = [
    "STAIR_COMPLETE_REWARD_RUNTIME_VNEXT_SCHEMA_VERSION",
    "build_stair_vnext_reward_runtime_contract",
]
