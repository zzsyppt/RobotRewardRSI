"""Dependency-light defaults for provider request limits."""

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
    "PROVIDER_TRANSPORT_POLICY_ID",
    "PROVIDER_REASONING_MODES",
]
