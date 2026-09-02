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

from true_rl_agent.agent_core.authoring_response import (
    AuthoringProviderResponse,
    CandidateDraftResponse,
    SourceRequestResponse,
    build_source_request_denied_error,
)
from true_rl_agent.agent_core.provider_contract import (
    provider_authoring_response_mode,
    provider_json_output_contract,
    validate_declared_changed_components,
    validate_provider_json_output_keys,
    validate_source_request_json_output_keys,
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
    contains_forbidden_public_value_marker,
    redact_secret_key_markers,
)
from true_rl_agent.candidates.component_source_semantics import (
    extract_training_components_rationale,
)
from true_rl_agent.candidates.loader import (
    CANDIDATE_SCHEMA_VERSION,
    CONTROLLED_ENTRYPOINT,
)
from true_rl_agent.candidates.package_contract import (
    CANDIDATE_PACKAGE_FILE_MAX_BYTES,
)
from true_rl_agent.candidates.schema import (
    CandidateChangeSet,
    CandidatePackageRef,
    require_candidate_id,
)
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.prompting import (
    assert_public_only_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.prompting.source_catalog import (
    has_inline_current_authoring_failed_source,
    prompt_source_catalog_logical_ids,
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

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or self.max_tokens <= 0
        ):
            raise ValueError("provider max_tokens must be a positive integer")
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
        self._counter = _existing_candidate_counter(config.output_dir)

    def next_candidate(self, *, prompt_payload: dict[str, Any]) -> CandidatePackageRef:
        response = self.next_authoring_response(prompt_payload=prompt_payload)
        if isinstance(response, SourceRequestResponse):
            raise build_source_request_denied_error(response)
        return response.candidate

    def next_authoring_response(self, *, prompt_payload: dict[str, Any]) -> AuthoringProviderResponse:
        self._counter += 1
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
        if (
            self.config.empty_final_reasoning_retry
            and finish_reason != "length"
            and _receipt_proves_empty_final_content_with_reasoning(
                provider_receipt
            )
        ):
            raise ProviderCandidateError(
                "json_parse_error",
                "provider final content channel was empty",
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
            proposal = _json_object_from_text(response_text, "candidate proposal")
        except ValueError as exc:
            truncated = finish_reason == "length"
            raise ProviderCandidateError(
                "response_truncated" if truncated else "json_parse_error",
                (
                    "provider response reached its output limit before candidate JSON completed"
                    if truncated
                    else "candidate proposal was not valid JSON object"
                ),
                details=(
                    {"finish_reason": finish_reason}
                    if finish_reason is not None
                    else None
                ),
                provider_usage=provider_usage,
                provider_receipt=provider_receipt,
            ) from exc
        try:
            mode = provider_authoring_response_mode(proposal)
            if mode == "source_request":
                return self._parse_source_request(
                    proposal,
                    prompt_payload=prompt_payload,
                    provider_usage=provider_usage,
                    provider_receipt=provider_receipt,
                )
            return CandidateDraftResponse(
                candidate=self._materialize_candidate(proposal),
                provider_usage=provider_usage,
                provider_receipt=provider_receipt,
            )
        except ProviderCandidateError as exc:
            if (
                (not provider_usage or exc.provider_usage)
                and (not provider_receipt or exc.provider_receipt)
            ):
                raise
            raise ProviderCandidateError(
                exc.category,
                exc.reason,
                details=exc.details,
                provider_usage=exc.provider_usage or provider_usage,
                provider_receipt=exc.provider_receipt or provider_receipt,
            ) from exc
        except ValueError as exc:
            raise ProviderCandidateError(
                "schema_error",
                str(exc),
                provider_usage=provider_usage,
                provider_receipt=provider_receipt,
            ) from exc
        except OSError as exc:
            raise ProviderCandidateError(
                "candidate_materialize_error",
                "candidate package could not be materialized",
                details={"error_type": type(exc).__name__},
                provider_usage=provider_usage,
                provider_receipt=provider_receipt,
            ) from exc

    def _chat_completion_request(self, prompt_payload: Mapping[str, Any]) -> dict[str, Any]:
        request = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": _system_prompt(),
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

    def _materialize_candidate(self, proposal: Mapping[str, Any]) -> CandidatePackageRef:
        validate_provider_json_output_keys(dict(proposal))
        candidate_id = require_candidate_id(proposal.get("candidate_id"))
        components_py = _require_text(proposal.get("components_py"), "components_py")
        components_bytes = _bounded_candidate_file_bytes(
            "training_code/components.py",
            components_py,
        )
        _reject_secret_markers(components_py)
        rationale = extract_training_components_rationale(components_py)
        if rationale is None:
            raise ValueError(
                "components_py must set TrainingComponents.rationale to a "
                "non-empty statically resolvable string literal"
            )
        rationale_bytes = _bounded_candidate_file_bytes(
            "rationale.md",
            rationale,
        )
        changed_components = _changed_components(proposal.get("declared_changed_components"))

        file_bytes = {
            "candidate_manifest.yaml": _bounded_candidate_file_bytes(
                "candidate_manifest.yaml",
                "\n".join(
                    [
                        f"schema_version: {CANDIDATE_SCHEMA_VERSION}",
                        f"candidate_id: {candidate_id}",
                        f"entrypoint: {CONTROLLED_ENTRYPOINT}",
                        "",
                    ]
                ),
            ),
            "training_code/components.py": components_bytes,
            "rationale.md": rationale_bytes,
            "operation_diff.json": _bounded_candidate_file_bytes(
                "operation_diff.json",
                json.dumps(
                    {
                        "schema_version": "true_rl_agent.provider.operation_diff.v1",
                        "candidate_id": candidate_id,
                        "declared_changed_components": sorted(changed_components),
                        "derived_from": ["declared_changed_components"],
                        "materialized_by": "openai_compatible_candidate_materializer",
                        "provider_backend": "openai_compatible_chat",
                        "verifier_derived_change_set_is_authoritative": True,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            ),
            "diagnostics.json": _bounded_candidate_file_bytes(
                "diagnostics.json",
                json.dumps(
                    {
                        "schema_version": "true_rl_agent.provider.diagnostics.v1",
                        "candidate_id": candidate_id,
                        "materialized_by": "openai_compatible_candidate_materializer",
                        "provider_backend": "openai_compatible_chat",
                        "status": "provider_materialized",
                        "diagnostic_items": [],
                        "notes": [
                            "Provider JSON v2 does not accept LLM-authored diagnostics.",
                            "Task runtime diagnostics must be declared through DiagnosticTerm in components.py.",
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
            ),
        }

        candidate_dir = self.config.output_dir / f"{self._counter:03d}_{candidate_id}"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        for relative_path, content in file_bytes.items():
            target = candidate_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return CandidatePackageRef(
            candidate_id=candidate_id,
            path=str(candidate_dir),
            change_set=CandidateChangeSet(frozenset(changed_components)),
            metadata={"provider": "openai_compatible_chat", "declared_change_set_provisional": True},
        )

    def _parse_source_request(
        self,
        proposal: Mapping[str, Any],
        *,
        prompt_payload: Mapping[str, Any],
        provider_usage: Mapping[str, int],
        provider_receipt: Mapping[str, Any],
    ) -> SourceRequestResponse:
        validate_source_request_json_output_keys(dict(proposal))
        logical_ids = proposal.get("logical_ids")
        if not isinstance(logical_ids, list) or not all(isinstance(item, str) for item in logical_ids):
            raise ValueError("source_request logical_ids must be a JSON array of strings")
        rationale = _require_text(proposal.get("rationale", ""), "rationale")
        _reject_secret_markers(rationale)
        return SourceRequestResponse(
            logical_ids=tuple(logical_ids),
            rationale=rationale,
            source_catalog_logical_ids=tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload))),
            provider_usage=provider_usage,
            provider_receipt=provider_receipt,
        )


def _bounded_candidate_file_bytes(relative_path: str, content: str) -> bytes:
    maximum = CANDIDATE_PACKAGE_FILE_MAX_BYTES[relative_path]
    encoded = content.encode("utf-8")
    if len(encoded) > maximum:
        raise ValueError(
            "candidate package file exceeds its byte limit: "
            f"{relative_path} ({len(encoded)} > {maximum})"
        )
    return encoded


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


def _existing_candidate_counter(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0
    if not output_dir.is_dir():
        raise NotADirectoryError(
            f"provider candidate output is not a directory: {output_dir}"
        )
    indices = []
    for child in output_dir.iterdir():
        prefix, separator, _ = child.name.partition("_")
        if child.is_dir() and separator and prefix.isdecimal():
            indices.append(int(prefix))
    return max(indices, default=0)


def _system_prompt() -> str:
    contract = provider_json_output_contract()
    draft_required = ", ".join(contract["one_of"]["candidate_draft"]["required_keys"])
    source_request_required = ", ".join(contract["one_of"]["source_request"]["required_keys"])
    return (
        "You are an OpenAI-compatible candidate-package provider. "
        "The user payload is the only source of task-specific context, public evidence, component "
        "contract details, and output-field constraints. Return exactly one valid JSON object and "
        "no prose outside JSON. The JSON object must match exactly one response shape from "
        "provider_json_output.one_of in the user payload. The default response mode is "
        "candidate_draft; candidate_draft may omit response_mode and must include: "
        f"{draft_required}. A source_request must set response_mode='source_request' and include: "
        f"{source_request_required}. Do not mix candidate_draft and source_request fields. "
        "Use prompt-visible task/eval summaries only as read-only evidence; do not treat them as "
        "candidate-controllable targets. Do not read non-prompt files, environment variables, raw "
        "seeds, checkpoints, runner state, eval state, private budgets, private success metrics, "
        "or private task invariants. Only use candidate-writable surfaces listed in the user payload. "
        "Do not propose changes to runner code, eval protocol, training budget, seed policy, "
        "checkpoint policy, PPO, task invariants, curriculum, or acceptance rules. Follow the "
        "candidate package and component contracts in the user payload. Do not include secrets or "
        "hidden/private data."
    )


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
        and _is_empty_final_reasoning_retry_feedback(repair)
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
        receipt.get("response_text_present") is True
        and receipt.get("response_text_utf8_bytes") == 0
        and receipt.get("response_text_sha256") == empty_sha256
        and receipt.get("reasoning_content_present") is True
        and isinstance(receipt.get("reasoning_content_utf8_bytes"), int)
        and not isinstance(receipt.get("reasoning_content_utf8_bytes"), bool)
        and receipt["reasoning_content_utf8_bytes"] > 0
        and isinstance(receipt.get("reasoning_content_sha256"), str)
    )


def _is_empty_final_reasoning_retry_feedback(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("failure_category") != "json_parse_error":
        return False
    failure = value.get("failure")
    if (
        not isinstance(failure, Mapping)
        or failure.get("category") != "json_parse_error"
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
            "true_rl_agent.provider_response_receipt.v3"
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


def _json_object_from_text(text: str, name: str) -> dict[str, Any]:
    try:
        return strict_json_object_loads(text, context=name)
    except ValueError as strict_error:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise ValueError(f"{name} was not valid JSON: {_safe_error(text)}") from strict_error
        try:
            return strict_json_object_loads(extracted, context=name)
        except ValueError as extracted_error:
            raise ValueError(f"{name} contained an invalid JSON object: {_safe_error(text)}") from extracted_error


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _changed_components(value: Any) -> set[str]:
    return set(validate_declared_changed_components(value))


def _reject_secret_markers(text: str) -> None:
    if contains_forbidden_public_value_marker(text):
        raise ValueError("candidate text contains a secret-looking marker")


def _safe_error(text: str) -> str:
    compact = " ".join(text.split())
    return redact_secret_key_markers(compact)[:1200]
