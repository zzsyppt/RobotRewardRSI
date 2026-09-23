"""Dependency-light defaults for provider request limits."""

from collections.abc import Mapping


def validate_provider_model_context(value: Mapping) -> dict:
    """Freeze declared model capacity; token counts come from the serving provider."""
    if not isinstance(value, Mapping) or set(value) != {
        "context_window_tokens", "max_output_tokens", "tokenizer_count_source", "specification_source"
    }:
        raise ValueError("provider model context fields mismatch")
    for key in ("context_window_tokens", "max_output_tokens"):
        if isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] <= 0:
            raise ValueError(f"provider {key} must be a positive integer")
    if (value["tokenizer_count_source"] != "provider_usage_prompt_tokens"
            or not isinstance(value["specification_source"], str) or not value["specification_source"]):
        raise ValueError("provider capacity must state its specification and token-count authority")
    return dict(value)

# This is a practical ceiling, not a tokenizer or JSON-capacity proof.  It was
# validated against production-shaped Stair prompts.  V4-Pro completed at 16K,
# but V4-Flash with max reasoning exhausted 16K before emitting final JSON in
# three of five bounded rechecks.  32K is the smallest conservative step up;
# package preflight and the response-byte envelope remain hard authorities.
DEFAULT_PROVIDER_MAX_OUTPUT_TOKENS = 32_768
DEFAULT_PROVIDER_MAX_RESPONSE_BYTES = 512 * 1024
DEFAULT_PROVIDER_REASONING_MODE = "provider_default"
DEFAULT_PROVIDER_MODEL = "deepseek-v4-flash"
DEFAULT_PROVIDER_REASONING_EFFORT = "max"
PROVIDER_TRANSPORT_POLICY_ID = (
    "true_rl_agent.openai_compatible_transport.https_or_loopback_http_no_redirect.v1"
)
PROVIDER_ADAPTIVE_REASONING_POLICY_ID = (
    "true_rl_agent.openai_compatible_provider.adaptive_reasoning.v2"
)
PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND = (
    "empty_final_content_with_reasoning"
)
PROVIDER_REASONING_MODES = (
    "adaptive",
    "disabled",
    "enabled",
    "provider_default",
)
PROVIDER_REASONING_EFFORTS = ("high", "max")


__all__ = [
    "DEFAULT_PROVIDER_MAX_OUTPUT_TOKENS",
    "DEFAULT_PROVIDER_MAX_RESPONSE_BYTES",
    "DEFAULT_PROVIDER_MODEL",
    "DEFAULT_PROVIDER_REASONING_EFFORT",
    "DEFAULT_PROVIDER_REASONING_MODE",
    "PROVIDER_REASONING_EFFORTS",
    "PROVIDER_ADAPTIVE_REASONING_POLICY_ID",
    "PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND",
    "PROVIDER_TRANSPORT_POLICY_ID",
    "PROVIDER_REASONING_MODES",
]
