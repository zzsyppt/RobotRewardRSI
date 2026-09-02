"""Shared Phase 8E constants and public/prompt safety guards."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCE_PACK_MAX_TEXT_BYTES = 64 * 1024
SOURCE_PACK_FORBIDDEN_TOKENS = (
    "iid_hidden",
    "hidden_eval",
    "success_metric",
    "checkpoint_selector",
    "eval_config",
    "OPENAI_API_KEY",
    "DEEPSEEK_API_KEY",
    "api_key",
    "secret",
)
PROMPT_FACING_OMITTED_VALUE = "[omitted: non-prompt-facing training surface]"
PROMPT_FACING_INACTIVE_TOKENS = (
    "not_instrumented",
    "accepted_precomputed_not_applied",
    "accepted_but_ineffective",
    "ineffective_schedule_keys_not_primary_mechanism",
    "LossTerm",
    "loss_terms",
    "train_observation_transforms",
    "push_magnitude_scale",
    "push_start_fraction",
    "push_interval_s",
    "domain_randomization_scale",
    "domain randomization",
)


class Phase8EError(RuntimeError):
    """Raised when Phase 8E orchestration cannot continue safely."""


def provider_source_read_turn_budget(config: Mapping[str, Any]) -> int:
    """Return the runtime source-read turn budget exposed to provider prompts."""

    raw = config.get("provider_inner_loop")
    if not isinstance(raw, Mapping) or not raw.get("enabled"):
        return 0
    try:
        budget = int(raw.get("max_source_read_turns", 2))
    except (TypeError, ValueError):
        return 0
    return max(0, budget)


def provider_source_read_enabled(config: Mapping[str, Any]) -> bool:
    return provider_source_read_turn_budget(config) > 0


def contains_inactive_prompt_token(text: str) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in PROMPT_FACING_INACTIVE_TOKENS)


def assert_prompt_facing_payload(value: Any, *, context: str) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = text.lower()
    hits = [token for token in PROMPT_FACING_INACTIVE_TOKENS if token.lower() in lowered]
    if hits:
        raise Phase8EError(f"prompt-facing payload contains inactive capability tokens in {context}: {hits}")


def scrub_prompt_facing_value(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if contains_inactive_prompt_token(key_text):
                continue
            out[key] = scrub_prompt_facing_value(item)
        return out
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and contains_inactive_prompt_token(item):
                continue
            out.append(scrub_prompt_facing_value(item))
        return out
    if isinstance(value, tuple):
        out = []
        for item in value:
            if isinstance(item, str) and contains_inactive_prompt_token(item):
                continue
            out.append(scrub_prompt_facing_value(item))
        return tuple(out)
    if isinstance(value, str) and contains_inactive_prompt_token(value):
        return PROMPT_FACING_OMITTED_VALUE
    return value


def assert_public_only_payload(value: Any, *, config: dict[str, Any], context: str) -> None:
    safety = config.get("public_safety", {})
    forbidden_text = [str(item) for item in safety.get("forbidden_string_tokens", [])]
    forbidden_numbers = {int(item) for item in safety.get("forbidden_numeric_tokens", [])}
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    lowered = text.lower()
    hits = [token for token in forbidden_text if token.lower() in lowered]
    number_re = _forbidden_number_regex(forbidden_numbers)
    number_hits = sorted(set(number_re.findall(text)))
    if hits or number_hits:
        raise Phase8EError(f"public-only payload contains forbidden tokens in {context}: text={hits}, numbers={number_hits}")


def _forbidden_number_regex(numbers: set[int]) -> re.Pattern[str]:
    if not numbers:
        return re.compile(r"(?!x)x")
    joined = "|".join(re.escape(str(item)) for item in sorted(numbers, key=lambda value: len(str(value)), reverse=True))
    # Do not match protected seed tokens when they are embedded in an identifier or
    # fingerprint ("...a1001b...").  Eval seeds always surface as standalone
    # numbers in public artifacts, never inside alphanumeric hashes.
    return re.compile(rf"(?<![\w.])(?:{joined})(?![\w.])")
