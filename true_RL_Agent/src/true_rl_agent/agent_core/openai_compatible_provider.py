"""OpenAI-compatible LLM candidate provider."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

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
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
from true_rl_agent.candidates.loader import (
    CANDIDATE_SCHEMA_VERSION,
    CONTROLLED_ENTRYPOINT,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.prompting import (
    assert_public_only_prompt_payload,
    serialize_prompt_payload,
)
from true_rl_agent.prompting.source_catalog import prompt_source_catalog_logical_ids

_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")


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
    max_tokens: int = 8192
    timeout_s: int = 180
    temperature: float = 0.2


class UrllibJsonTransport:
    def post_json(self, *, url: str, payload: dict[str, Any], api_key: str, timeout_s: int) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"provider request failed with HTTP {exc.code}: {_safe_error(raw_error)}") from exc
        return strict_json_object_loads(raw, context="provider response")


class OpenAICompatibleCandidateProvider:
    """Generate controlled candidate packages from an OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        config: OpenAICompatibleProviderConfig,
        *,
        transport: JsonTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport or UrllibJsonTransport()
        self._counter = 0

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
        try:
            response_text = _response_text(response)
        except ValueError as exc:
            raise ProviderCandidateError(
                "schema_error",
                "provider response did not contain candidate text output",
            ) from exc
        try:
            proposal = _json_object_from_text(response_text, "candidate proposal")
        except ValueError as exc:
            raise ProviderCandidateError(
                "json_parse_error",
                "candidate proposal was not valid JSON object",
            ) from exc
        try:
            mode = provider_authoring_response_mode(proposal)
            if mode == "source_request":
                return self._parse_source_request(proposal, prompt_payload=prompt_payload)
            return CandidateDraftResponse(candidate=self._materialize_candidate(proposal))
        except ProviderCandidateError:
            raise
        except ValueError as exc:
            raise ProviderCandidateError("schema_error", str(exc)) from exc
        except OSError as exc:
            raise ProviderCandidateError(
                "candidate_materialize_error",
                "candidate package could not be materialized",
                details={"error_type": type(exc).__name__},
            ) from exc

    def _chat_completion_request(self, prompt_payload: Mapping[str, Any]) -> dict[str, Any]:
        return {
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
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

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
        candidate_id = _require_candidate_id(proposal.get("candidate_id"))
        components_py = _require_text(proposal.get("components_py"), "components_py")
        rationale = _require_text(proposal.get("rationale", ""), "rationale")
        _reject_secret_markers(components_py)
        _reject_secret_markers(rationale)
        changed_components = _changed_components(proposal.get("declared_changed_components"))

        candidate_dir = self.config.output_dir / f"{self._counter:03d}_{candidate_id}"
        training_dir = candidate_dir / "training_code"
        training_dir.mkdir(parents=True, exist_ok=False)
        (candidate_dir / "candidate_manifest.yaml").write_text(
            "\n".join(
                [
                    f"schema_version: {CANDIDATE_SCHEMA_VERSION}",
                    f"candidate_id: {candidate_id}",
                    f"entrypoint: {CONTROLLED_ENTRYPOINT}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (training_dir / "components.py").write_text(components_py.rstrip() + "\n", encoding="utf-8")
        (candidate_dir / "rationale.md").write_text(rationale.rstrip() + "\n", encoding="utf-8")
        (candidate_dir / "operation_diff.json").write_text(
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
            encoding="utf-8",
        )
        (candidate_dir / "diagnostics.json").write_text(
            json.dumps(
                {
                    "schema_version": "true_rl_agent.provider.diagnostics.v1",
                    "candidate_id": candidate_id,
                    "materialized_by": "openai_compatible_candidate_materializer",
                    "provider_backend": "openai_compatible_chat",
                    "status": "provider_materialized",
                    "diagnostic_items": [],
                    "notes": [
                        "Provider JSON v1 does not accept LLM-authored diagnostics.",
                        "Task runtime diagnostics must be declared through DiagnosticTerm in components.py.",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
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
        )


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
    return base_url.rstrip("/") + "/" + endpoint.lstrip("/")


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


def _require_candidate_id(value: Any) -> str:
    if not isinstance(value, str) or not _CANDIDATE_ID_RE.fullmatch(value):
        raise ValueError("candidate_id must contain only letters, digits, dot, underscore, and dash")
    return value


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _changed_components(value: Any) -> set[str]:
    return set(validate_declared_changed_components(value))


def _reject_secret_markers(text: str) -> None:
    if "sk-" in text:
        raise ValueError("candidate text contains a secret-looking marker")


def _safe_error(text: str) -> str:
    compact = " ".join(text.split())
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", compact)[:1200]
