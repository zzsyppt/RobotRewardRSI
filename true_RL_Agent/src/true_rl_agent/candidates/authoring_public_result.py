"""Canonical public-safe serialization for authoring checks."""

from __future__ import annotations

import json
from typing import Any, Mapping

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload


def public_authoring_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached, public-safe JSON object for authoring evidence."""

    clean = sanitize_prompt_public_context(dict(value))
    serialized = json.loads(json.dumps(clean, ensure_ascii=False))
    assert_public_only_prompt_payload(serialized)
    return serialized
