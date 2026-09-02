"""Backward-compatible exports for prompt-facing candidate projection."""

from __future__ import annotations

from true_rl_agent.prompting.payload import (
    PROMPT_PAYLOAD_SECTION_ORDER,
    assert_public_only_prompt_payload,
    build_prompt_payload,
    candidate_allowed_controls,
    candidate_authoring_capabilities,
    candidate_component_contract,
    control_effectiveness_contract,
    required_candidate_package,
    reward_inputs_projection,
    serialize_prompt_payload,
)

__all__ = [
    "PROMPT_PAYLOAD_SECTION_ORDER",
    "assert_public_only_prompt_payload",
    "build_prompt_payload",
    "candidate_authoring_capabilities",
    "candidate_allowed_controls",
    "candidate_component_contract",
    "control_effectiveness_contract",
    "required_candidate_package",
    "reward_inputs_projection",
    "serialize_prompt_payload",
]
