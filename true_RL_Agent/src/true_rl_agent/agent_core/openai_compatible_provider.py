"""OpenAI-compatible LLM candidate provider."""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from true_rl_agent.agent_core.authoring_payload import AuthoringPayloadMaterializer, authoring_system_prompt
from true_rl_agent.agent_core.authoring_response import (
    AuthoringProviderResponse,
    SourceRequestResponse,
    build_source_request_denied_error,
)
from true_rl_agent.agent_core.provider_errors import (
    ProviderCandidateError,
    normalize_provider_receipt,
    normalize_provider_usage,
)
from true_rl_agent.agent_core.provider_limits import (
    DEFAULT_PROVIDER_MAX_OUTPUT_TOKENS,
    DEFAULT_PROVIDER_MAX_RESPONSE_BYTES,
    DEFAULT_PROVIDER_REASONING_MODE,
    PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND,
    PROVIDER_REASONING_EFFORTS,
    PROVIDER_REASONING_MODES,
    PROVIDER_TRANSPORT_POLICY_ID,
)
from true_rl_agent.agent_core.public_safety import (
    redact_secret_key_markers,
)
from true_rl_agent.candidates.schema import (
    CandidatePackageRef,
)
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.prompting import (
    assert_public_only_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.prompting.source_catalog import (
    has_inline_current_authoring_failed_source,
)


class JsonTransport(Protocol):
    def post_json(self, *, url: str, payload: dict[str, Any], api_key: str, timeout_s: int) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class OpenAICompatibleProviderConfig:
    model: str
    output_dir: Path
    base_url: str = "https://api.openai.com/v1"
    endpoint: str = "/chat/completions"
    api_key_env: str = "OPENAI_API_KEY"
    api_key: str | None = None
    env_file: Path | None = None
    max_tokens: int = DEFAULT_PROVIDER_MAX_OUTPUT_TOKENS
    timeout_s: int = 180
    temperature: float = 0.2
    max_response_bytes: int = DEFAULT_PROVIDER_MAX_RESPONSE_BYTES
    reasoning_mode: str = DEFAULT_PROVIDER_REASONING_MODE
    reasoning_effort: str | None = None
    empty_final_reasoning_retry: bool = False
    model_context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("provider max_tokens must be a positive integer")
        if self.model_context is not None:
            from true_rl_agent.agent_core.provider_limits import validate_provider_model_context
            context = validate_provider_model_context(self.model_context)
            if self.max_tokens > min(context["max_output_tokens"], context["context_window_tokens"]):
                raise ValueError("requested output exceeds the declared provider model capacity")
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or self.max_response_bytes <= 0
        ):
            raise ValueError("provider max_response_bytes must be a positive integer")
        if self.reasoning_mode not in PROVIDER_REASONING_MODES:
            raise ValueError("provider reasoning_mode is unsupported")
        if (
            self.reasoning_effort is not None
            and self.reasoning_effort not in PROVIDER_REASONING_EFFORTS
        ):
            raise ValueError("provider reasoning_effort is unsupported")
        if not isinstance(self.empty_final_reasoning_retry, bool):
            raise ValueError(
                "provider empty_final_reasoning_retry must be boolean"
            )
        if (
            self.empty_final_reasoning_retry
            and self.reasoning_mode != "adaptive"
        ):
            raise ValueError(
                "provider empty_final_reasoning_retry requires adaptive reasoning"
            )
        _join_url(self.base_url, self.endpoint)


class _RejectProviderRedirects(urllib.request.HTTPRedirectHandler):
    """Keep bearer credentials on the operator-selected provider origin."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


@dataclass(frozen=True)
class UrllibJsonTransport:
    max_response_bytes: int = DEFAULT_PROVIDER_MAX_RESPONSE_BYTES
    transport_policy_id: str = field(
        default=PROVIDER_TRANSPORT_POLICY_ID,
        init=False,
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_response_bytes, bool)
            or not isinstance(self.max_response_bytes, int)
            or self.max_response_bytes <= 0
        ):
            raise ValueError("provider response byte limit must be positive")

    def _failure(
        self,
        reason: str,
        *,
        failure_kind: str,
        retryable: bool,
        http_status: int | None = None,
    ) -> ProviderCandidateError:
        details: dict[str, Any] = {
            "failure_kind": failure_kind,
            "retryable": retryable,
            "transport_policy_id": self.transport_policy_id,
            "response_max_bytes": self.max_response_bytes,
        }
        if http_status is not None:
            details["http_status"] = http_status
        return ProviderCandidateError(
            "runtime_error",
            reason,
            details=details,
            provider_receipt={
                "schema_version": "true_rl_agent.provider_response_receipt.v2",
                "transport_policy_id": self.transport_policy_id,
                "response_max_bytes": self.max_response_bytes,
            },
        )

    def post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        api_key: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        return self._request_json(
            method="POST",
            url=url,
            payload=payload,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def get_json(
        self,
        *,
        url: str,
        api_key: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        """Issue one bounded authenticated GET without changing JsonTransport."""

        return self._request_json(
            method="GET",
            url=url,
            payload=None,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    def _request_json(
        self,
        *,
        method: str,
        url: str,
        payload: dict[str, Any] | None,
        api_key: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        try:
            url = _validated_provider_url(url)
        except (TypeError, ValueError) as exc:
            raise self._failure(
                "provider URL violates the configured transport policy",
                failure_kind="url_policy_error",
                retryable=False,
            ) from exc
        if not _is_safe_authorization_credential(api_key):
            raise self._failure(
                "provider credential cannot be encoded as a safe HTTP header",
                failure_kind="credential_format_error",
                retryable=False,
            )
        if method not in {"GET", "POST"}:
            raise ValueError("provider JSON request method is invalid")
        if method == "POST" and payload is None:
            raise ValueError("provider POST requires a JSON payload")
        if method == "GET" and payload is not None:
            raise ValueError("provider GET cannot carry a JSON payload")
        body = (
            json.dumps(payload, ensure_ascii=False).encode("utf-8")
            if payload is not None
            else None
        )
        headers = {"Authorization": f"Bearer {api_key}"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        else:
            headers["Accept"] = "application/json"
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=method,
        )
        try:
            opener = urllib.request.build_opener(_RejectProviderRedirects())
            with opener.open(request, timeout=timeout_s) as response:
                raw_bytes = response.read(self.max_response_bytes + 1)
                if len(raw_bytes) > self.max_response_bytes:
                    raise self._failure(
                        "provider response exceeded the bounded JSON envelope",
                        failure_kind="response_too_large",
                        retryable=False,
                    )
                raw = raw_bytes.decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                exc.close()
            except AttributeError:
                pass
            status = int(exc.code)
            redirected = 300 <= status < 400
            raise self._failure(
                (
                    "provider redirect was rejected"
                    if redirected
                    else "provider request returned an HTTP error"
                ),
                failure_kind=(
                    "redirect_rejected" if redirected else "http_error"
                ),
                http_status=status,
                retryable=(
                    False
                    if redirected
                    else status in {408, 429} or status >= 500
                ),
            ) from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise self._failure(
                    "provider TLS certificate verification failed",
                    failure_kind="tls_certificate_error",
                    retryable=False,
                ) from exc
            timed_out = isinstance(exc.reason, (TimeoutError, socket.timeout))
            raise self._failure(
                "provider request timed out" if timed_out else "provider network request failed",
                failure_kind="timeout" if timed_out else "network_error",
                retryable=True,
            ) from exc
        except (TimeoutError, socket.timeout) as exc:
            raise self._failure(
                "provider request timed out",
                failure_kind="timeout",
                retryable=True,
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise self._failure(
                "provider network request failed",
                failure_kind="network_error",
                retryable=True,
            ) from exc
        except UnicodeDecodeError as exc:
            raise self._failure(
                "provider response was not valid UTF-8",
                failure_kind="invalid_utf8",
                retryable=False,
            ) from exc
        try:
            return strict_json_object_loads(raw, context="provider response")
        except ValueError as exc:
            raise self._failure(
                "provider response envelope was not a valid JSON object",
                failure_kind="invalid_json",
                retryable=False,
            ) from exc


class OpenAICompatibleCandidateProvider:
    """Generate controlled candidate packages from an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        config: OpenAICompatibleProviderConfig,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport(
            max_response_bytes=config.max_response_bytes
        )
        self._materializer = AuthoringPayloadMaterializer(config.output_dir, backend="openai_compatible_chat")

    def next_candidate(self, *, prompt_payload: dict[str, Any]) -> CandidatePackageRef:
        response = self.next_authoring_response(prompt_payload=prompt_payload)
        if isinstance(response, SourceRequestResponse):
            raise build_source_request_denied_error(response)
        return response.candidate

    def next_authoring_response(self, *, prompt_payload: dict[str, Any]) -> AuthoringProviderResponse:
        self._materializer.begin_attempt()
        assert_public_only_prompt_payload(prompt_payload)
        request_payload = self._chat_completion_request(prompt_payload)
        response = self.transport.post_json(
            url=_join_url(self.config.base_url, self.config.endpoint),
            payload=request_payload,
            api_key=self._api_key(),
            timeout_s=self.config.timeout_s,
        )
        provider_usage = normalize_provider_usage(response.get("usage"))
        finish_reason = _provider_finish_reason(response)
        provider_receipt = _provider_response_receipt(
            response,
            request_payload=request_payload,
            finish_reason=finish_reason,
            transport_policy_id=_transport_policy_id(self.transport),
            response_max_bytes=_transport_response_max_bytes(self.transport),
            model_context=self.config.model_context,
        )
        if (
            _receipt_proves_empty_final_content_with_reasoning(
                provider_receipt
            )
        ):
            raise ProviderCandidateError(
                "reasoning_output_exhausted" if finish_reason == "length" else "json_parse_error",
                "reasoning exhausted the output budget before final JSON" if finish_reason == "length"
                else "provider final content channel was empty",
                details={
                    "failure_kind": (
                        PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND
                    ),
                    **(
                        {"finish_reason": finish_reason}
                        if finish_reason is not None
                        else {}
                    ),
                },
                provider_usage=provider_usage,
                provider_receipt=provider_receipt,
            )
        try:
            response_text = _response_text(response)
        except ValueError as exc:
            truncated = finish_reason == "length"
            raise ProviderCandidateError(
                "response_truncated" if truncated else "schema_error",
                (
                    "provider response reached its output limit before candidate JSON completed"
                    if truncated
                    else "provider response did not contain candidate text output"
                ),
                details=(
                    {"finish_reason": finish_reason}
                    if finish_reason is not None
                    else None
                ),
                provider_usage=provider_usage,
                provider_receipt=provider_receipt,
            ) from exc
        return self._materializer.parse(response_text, prompt_payload=prompt_payload,
            provider_usage=provider_usage, provider_receipt=provider_receipt,
            finish_reason=finish_reason)

    def _chat_completion_request(self, prompt_payload: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": authoring_system_prompt(),
                },
                {
                    "role": "user",
                    "content": serialize_prompt_payload(prompt_payload),
                },
            ],
            "response_format": {"type": "json_object"},
            "stream": False,
            "max_tokens": self.config.max_tokens,
        }
        thinking_mode = _resolved_thinking_mode(
            self.config.reasoning_mode,
            prompt_payload,
            empty_final_reasoning_retry=(
                self.config.empty_final_reasoning_retry
            ),
        )
        if thinking_mode != "provider_default":
            request["thinking"] = {"type": thinking_mode}
        if thinking_mode == "enabled":
            if self.config.reasoning_effort is not None:
                request["reasoning_effort"] = self.config.reasoning_effort
            # DeepSeek ignores sampling controls in thinking mode.  Omitting the
            # inert field keeps the immutable request receipt unambiguous.
        else:
            request["temperature"] = self.config.temperature
        return request

    def validate_credential(self) -> None:
        """Check local credential availability before expensive session work."""
        if not _is_safe_authorization_credential(self._api_key()):
            raise RuntimeError("provider API key is not a valid authorization credential")

    def _api_key(self) -> str:
        if self.config.api_key:
            return self.config.api_key
        values: dict[str, str] = {}
        if self.config.env_file is not None:
            values.update(load_env_file(self.config.env_file))
        value = values.get(self.config.api_key_env) or os.environ.get(self.config.api_key_env)
        if not value:
            raise RuntimeError(f"missing provider API key env var: {self.config.api_key_env}")
        return value

def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        values[key] = value.strip().strip("\"'")
    return values


def _join_url(base_url: str, endpoint: str) -> str:
    """Compose an opaque transport URL without imposing urllib policy.

    ``JsonTransport`` is a public extension seam and may use a non-HTTP URI.
    Scheme, TLS, credential, and redirect enforcement therefore belongs only
    to the built-in :class:`UrllibJsonTransport` immediately before I/O.
    """

    if not isinstance(base_url, str) or not base_url:
        raise ValueError("provider base_url must be non-empty")
    if not isinstance(endpoint, str) or not endpoint:
        raise ValueError("provider endpoint must be non-empty")
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


def _validated_provider_url(
    url: str,
    *,
    require_secure_external: bool = True,
) -> str:
    try:
        parsed = urllib.parse.urlsplit(url)
        port = parsed.port
    except (TypeError, ValueError) as exc:
        raise ValueError("provider URL is invalid") from exc
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError("provider URL scheme must be http or https")
    if (
        not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ValueError(
            "provider URL must contain a host and no credentials or fragment"
        )
    if port is not None and not (1 <= port <= 65535):
        raise ValueError("provider URL port is invalid")
    if (
        require_secure_external
        and scheme == "http"
        and not _is_loopback_provider_host(parsed.hostname)
    ):
        raise ValueError("non-loopback provider URLs must use HTTPS")
    if not parsed.path.startswith("/"):
        raise ValueError("provider URL path must be absolute")
    return urllib.parse.urlunsplit(
        (scheme, parsed.netloc, parsed.path, parsed.query, "")
    )


def _is_loopback_provider_host(hostname: str) -> bool:
    normalized = hostname.rstrip(".").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_safe_authorization_credential(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        value.encode("latin-1")
    except UnicodeEncodeError:
        return False
    return all(
        character not in "\r\n"
        and ord(character) >= 0x20
        and not 0x7F <= ord(character) <= 0x9F
        for character in value
    )


def _response_text(response: Mapping[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str):
        return output_text
    choices = response.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, Mapping):
            message = first.get("message")
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return str(message["content"])
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, Mapping) or item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, Mapping) and isinstance(part.get("text"), str):
                        return str(part["text"])
    raise ValueError("provider response did not contain text output")


def _provider_finish_reason(response: Mapping[str, Any]) -> str | None:
    """Normalize only bounded, non-content completion status metadata."""

    choices = response.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], Mapping):
        value = choices[0].get("finish_reason")
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {
                "content_filter",
                "function_call",
                "insufficient_system_resource",
                "length",
                "stop",
                "tool_calls",
            }:
                return normalized
            return "other"
    if response.get("status") == "incomplete":
        details = response.get("incomplete_details")
        if isinstance(details, Mapping) and details.get("reason") in {
            "max_output_tokens",
            "max_tokens",
        }:
            return "length"
    return None


def _resolved_thinking_mode(
    configured_mode: str,
    prompt_payload: Mapping[str, Any],
    *,
    empty_final_reasoning_retry: bool = False,
) -> str:
    """Resolve reasoning without interpreting any task-owned semantics."""

    if configured_mode != "adaptive":
        return configured_mode
    repair = prompt_payload.get("provider_or_runtime_repair_feedback")
    if (
        empty_final_reasoning_retry
        and (_is_empty_final_reasoning_retry_feedback(repair)
             or (isinstance(repair, Mapping) and (
                 repair.get("failure_category") == "reasoning_output_exhausted"
                 or repair.get("reasoning_output_exhausted_in_lane") is True)))
    ):
        return "disabled"
    if (
        isinstance(repair, Mapping)
        and repair.get("failure_category") == "response_truncated"
        and has_inline_current_authoring_failed_source(prompt_payload)
    ):
        return "disabled"
    return "enabled"


def _receipt_proves_empty_final_content_with_reasoning(
    receipt: Mapping[str, Any],
) -> bool:
    """Use only content-free receipt metadata to identify the routing failure."""

    empty_sha256 = "sha256:" + hashlib.sha256(b"").hexdigest()
    return (
        (receipt.get("response_text_present") is False or (
            receipt.get("response_text_utf8_bytes") == 0
            and receipt.get("response_text_sha256") == empty_sha256))
        and receipt.get("reasoning_content_present") is True
        and isinstance(receipt.get("reasoning_content_utf8_bytes"), int)
        and not isinstance(receipt.get("reasoning_content_utf8_bytes"), bool)
        and receipt["reasoning_content_utf8_bytes"] > 0
        and isinstance(receipt.get("reasoning_content_sha256"), str)
    )


def _is_empty_final_reasoning_retry_feedback(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("failure_category") not in {"json_parse_error", "reasoning_output_exhausted"}:
        return False
    failure = value.get("failure")
    if (
        not isinstance(failure, Mapping)
        or failure.get("category") != value.get("failure_category")
    ):
        return False
    details = failure.get("details")
    return (
        isinstance(details, Mapping)
        and details.get("failure_kind")
        == PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND
    )


def _provider_response_receipt(
    response: Mapping[str, Any],
    *,
    request_payload: Mapping[str, Any],
    finish_reason: str | None,
    transport_policy_id: str | None,
    response_max_bytes: int | None,
    model_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build content-free, bounded metadata for immutable attempt evidence."""

    try:
        response_text = _response_text(response)
    except ValueError:
        response_text = None
    reasoning_text = _provider_reasoning_text(response)
    thinking = request_payload.get("thinking")
    requested_thinking_mode = (
        thinking.get("type")
        if isinstance(thinking, Mapping)
        and thinking.get("type") in {"enabled", "disabled"}
        else "provider_default"
    )
    transport_bound = (
        isinstance(transport_policy_id, str)
        and bool(transport_policy_id)
        and isinstance(response_max_bytes, int)
        and not isinstance(response_max_bytes, bool)
        and response_max_bytes > 0
    )
    receipt: dict[str, Any] = {
        "schema_version": (
            "true_rl_agent.provider_response_receipt.v4" if model_context is not None and transport_bound
            else "true_rl_agent.provider_response_receipt.v3"
            if transport_bound
            else "true_rl_agent.provider_response_receipt.v1"
        ),
        "requested_model": request_payload.get("model"),
        "response_model": response.get("model"),
        "system_fingerprint": response.get("system_fingerprint"),
        "requested_output_units": request_payload.get("max_tokens"),
        "requested_thinking_mode": requested_thinking_mode,
        "requested_reasoning_effort": request_payload.get(
            "reasoning_effort", "provider_default"
        ),
        "finish_reason": finish_reason,
        **_text_receipt("response_text", response_text),
        **_text_receipt("reasoning_content", reasoning_text),
    }
    if transport_bound:
        receipt["transport_policy_id"] = transport_policy_id
        receipt["response_max_bytes"] = response_max_bytes
        if model_context is not None:
            usage = normalize_provider_usage(response.get("usage"))
            input_tokens = usage.get("request_units")
            receipt["model_context"] = dict(model_context)
            receipt["input_tokens"] = input_tokens
            receipt["joint_capacity_status"] = (
                "unknown_without_provider_usage" if input_tokens is None else "within_declared_window"
                if input_tokens + int(request_payload["max_tokens"]) <= model_context["context_window_tokens"]
                else "exceeds_declared_window")
    return normalize_provider_receipt(receipt)


def _transport_policy_id(transport: JsonTransport) -> str | None:
    value = getattr(transport, "transport_policy_id", None)
    return value if isinstance(value, str) and value else None


def _transport_response_max_bytes(transport: JsonTransport) -> int | None:
    value = getattr(transport, "max_response_bytes", None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _provider_reasoning_text(response: Mapping[str, Any]) -> str | None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, Mapping):
        return None
    message = first.get("message")
    if not isinstance(message, Mapping):
        return None
    value = message.get("reasoning_content")
    return value if isinstance(value, str) else None


def _text_receipt(prefix: str, value: str | None) -> dict[str, Any]:
    if value is None:
        return {
            f"{prefix}_present": False,
            f"{prefix}_utf8_bytes": 0,
            f"{prefix}_sha256": None,
        }
    payload = value.encode("utf-8")
    return {
        f"{prefix}_present": True,
        f"{prefix}_utf8_bytes": len(payload),
        f"{prefix}_sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
    }


def _safe_error(text: str) -> str:
    compact = " ".join(text.split())
    return redact_secret_key_markers(compact)[:1200]
