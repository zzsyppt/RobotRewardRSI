"""Shared task-owned prompt section sanitizer."""

from __future__ import annotations

from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.serialization import canonical_prompt_json_bytes

TASK_PROMPT_SECTION_KEYS = frozenset(
    {
        "task",
        "reward_inputs_contract",
        "public_eval_contract",
        "public_feedback_summary",
        "runtime_feedback_summary",
        "known_good_examples",
        "selection_guidance",
        "task_capability_metadata",
    }
)
DISALLOWED_TASK_PROMPT_KEY_PARTS = ("curriculum", "stage_list")
MAX_TASK_PROMPT_SECTIONS_BYTES = 32 * 1024


def sanitize_task_prompt_sections(sections: Mapping[str, Any]) -> dict[str, Any]:
    """Sanitize and validate task-owned prompt sections before prompt insertion."""

    if not isinstance(sections, Mapping):
        raise TypeError("task prompt sections must be a mapping")
    clean = sanitize_prompt_public_context(dict(sections))
    unknown_sections = sorted(set(clean) - TASK_PROMPT_SECTION_KEYS)
    if unknown_sections:
        raise ValueError(f"unknown task prompt section keys: {unknown_sections}")
    _reject_disallowed_task_prompt_keys(clean, path="$")
    assert_public_only_prompt_payload(clean)
    size = canonical_prompt_json_bytes(clean)
    if size > MAX_TASK_PROMPT_SECTIONS_BYTES:
        raise ValueError(
            "task prompt sections exceed "
            f"{MAX_TASK_PROMPT_SECTIONS_BYTES} bytes after sanitization: {size}"
        )
    return clean


def _reject_disallowed_task_prompt_keys(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(part in lowered for part in DISALLOWED_TASK_PROMPT_KEY_PARTS):
                raise ValueError(f"task prompt sections cannot expose curriculum/stage interfaces at {path}.{key}")
            _reject_disallowed_task_prompt_keys(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_disallowed_task_prompt_keys(item, path=f"{path}[{index}]")
