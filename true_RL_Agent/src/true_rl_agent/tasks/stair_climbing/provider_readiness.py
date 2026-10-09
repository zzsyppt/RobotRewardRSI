"""Task-owned DeepSeek account readiness gate for formal Stair generation."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from true_rl_agent.agent_core.openai_compatible_provider import (
    UrllibJsonTransport,
    load_env_file,
)
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError

STAIR_PROVIDER_READINESS_SCHEMA_VERSION = (
    "true_rl_agent.stair.provider_readiness.v1"
)
STAIR_PROVIDER_BALANCE_ENDPOINT = "/user/balance"
STAIR_PROVIDER_BALANCE_ORIGIN = "https://api.deepseek.com"
STAIR_PROVIDER_READINESS_MAX_RESPONSE_BYTES = 16 * 1024
STAIR_PROVIDER_READINESS_TIMEOUT_S = 30
_BALANCE_CURRENCIES = frozenset({"CNY", "USD"})
_BALANCE_AMOUNT_FIELDS = (
    "total_balance",
    "granted_balance",
    "topped_up_balance",
)


@dataclass(frozen=True)
class StairProviderReadinessError(RuntimeError):
    failure_kind: str
    reason: str
    http_status: int | None = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.reason)


def require_stair_provider_readiness(
    *,
    base_url: str,
    api_key_env: str,
    env_file: Path,
    timeout_s: int = STAIR_PROVIDER_READINESS_TIMEOUT_S,
) -> dict[str, Any]:
    """Require an authenticated, non-generative DeepSeek balance response."""

    endpoint = _balance_endpoint(base_url)
    api_key = _provider_credential(env_file=env_file, api_key_env=api_key_env)
    try:
        payload = UrllibJsonTransport(
            max_response_bytes=STAIR_PROVIDER_READINESS_MAX_RESPONSE_BYTES
        ).get_json(
            url=endpoint,
            api_key=api_key,
            timeout_s=timeout_s,
        )
    except ProviderCandidateError as exc:
        status_value = exc.details.get("http_status")
        status = int(status_value) if isinstance(status_value, int) else None
        transport_failure = exc.details.get("failure_kind")
        failure_kind, reason = _readiness_transport_failure(
            failure_kind=transport_failure,
            http_status=status,
        )
        raise StairProviderReadinessError(
            failure_kind=failure_kind,
            reason=reason,
            http_status=status,
        ) from exc

    is_available = payload.get("is_available")
    balance_infos = payload.get("balance_infos")
    if not isinstance(is_available, bool) or not isinstance(balance_infos, list):
        raise StairProviderReadinessError(
            failure_kind="invalid_response",
            reason="provider readiness response did not match the balance schema",
        )
    currencies = _balance_currencies(balance_infos)
    if not is_available:
        raise StairProviderReadinessError(
            failure_kind="insufficient_balance",
            reason="provider readiness reports insufficient balance",
        )
    return {
        "schema_version": STAIR_PROVIDER_READINESS_SCHEMA_VERSION,
        "checked_at_utc": (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        ),
        "endpoint": endpoint,
        "request_projection": {
            "method": "GET",
            "provider_balance_request_sent": True,
            "generation_request_sent": False,
        },
        "checks": {
            "authenticated_balance_response_valid": True,
            "provider_balance_available": True,
        },
        "account_projection": {
            "is_available": True,
            "currencies": currencies,
        },
        "privacy": {
            "credential_persisted": False,
            "balance_amounts_persisted": False,
        },
        "side_effects": {
            "provider_attempt_charged_to_scientific_budget": False,
        },
    }


def validate_stair_provider_credential(
    *,
    env_file: Path,
    api_key_env: str,
) -> None:
    """Validate the same fixed secret boundary without network I/O."""

    _provider_credential(env_file=env_file, api_key_env=api_key_env)


def _provider_credential(*, env_file: Path, api_key_env: str) -> str:
    if not env_file.is_file():
        raise FileNotFoundError("provider env file does not exist")
    value = load_env_file(env_file).get(api_key_env) or os.environ.get(api_key_env)
    if not isinstance(value, str) or not value:
        raise ValueError("provider credential is missing from its fixed boundary")
    try:
        value.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise ValueError("provider credential is not HTTP-header safe") from exc
    if any(
        character in "\r\n"
        or ord(character) < 0x20
        or 0x7F <= ord(character) <= 0x9F
        for character in value
    ):
        raise ValueError("provider credential is not HTTP-header safe")
    return value


def _balance_endpoint(base_url: str) -> str:
    if base_url.rstrip("/") != STAIR_PROVIDER_BALANCE_ORIGIN:
        raise ValueError("Stair provider readiness requires the frozen DeepSeek origin")
    return STAIR_PROVIDER_BALANCE_ORIGIN + STAIR_PROVIDER_BALANCE_ENDPOINT


def _balance_currencies(balance_infos: list[Any]) -> list[str]:
    currencies: set[str] = set()
    for value in balance_infos:
        if not isinstance(value, Mapping):
            raise StairProviderReadinessError(
                failure_kind="invalid_response",
                reason="provider readiness response did not match the balance schema",
            )
        currency = value.get("currency")
        if currency not in _BALANCE_CURRENCIES or any(
            not isinstance(value.get(field_name), str)
            or not value.get(field_name)
            for field_name in _BALANCE_AMOUNT_FIELDS
        ):
            raise StairProviderReadinessError(
                failure_kind="invalid_response",
                reason="provider readiness response did not match the balance schema",
            )
        currencies.add(currency)
    return sorted(currencies)


def _readiness_transport_failure(
    *,
    failure_kind: Any,
    http_status: int | None,
) -> tuple[str, str]:
    if http_status == 401:
        return "authentication_failed", "provider readiness authentication failed"
    if http_status == 402:
        return "insufficient_balance", "provider readiness reports insufficient balance"
    if isinstance(failure_kind, str) and failure_kind:
        normalized = failure_kind
    else:
        normalized = "transport_error"
    if normalized == "redirect_rejected":
        reason = "provider readiness redirect was rejected"
    elif normalized == "tls_certificate_error":
        reason = "provider readiness TLS certificate verification failed"
    elif normalized == "timeout":
        reason = "provider readiness request timed out"
    elif normalized == "network_error":
        reason = "provider readiness network request failed"
    elif http_status is not None:
        reason = f"provider readiness returned HTTP {http_status}"
    else:
        reason = "provider readiness response was unavailable"
    return normalized, reason


__all__ = [
    "STAIR_PROVIDER_READINESS_SCHEMA_VERSION",
    "StairProviderReadinessError",
    "require_stair_provider_readiness",
    "validate_stair_provider_credential",
]
