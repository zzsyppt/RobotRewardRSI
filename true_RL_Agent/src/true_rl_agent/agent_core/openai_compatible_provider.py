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

from true_rl_agent.agent_core.provider_contract import (
    MATERIALIZER_DECLARED_COMPONENTS_ALLOWED_VALUES,
    provider_json_output_contract,
    validate_provider_json_output_keys,
)
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload, serialize_prompt_payload
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef


_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_ALLOWED_DECLARED_COMPONENTS = frozenset(MATERIALIZER_DECLARED_COMPONENTS_ALLOWED_VALUES)


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
        return _json_object(json.loads(raw), "provider response")


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
            return self._materialize_candidate(proposal)
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
                    "schema_version: true_rl_agent.candidate.v1",
                    f"candidate_id: {candidate_id}",
                    "entrypoint: training_code.components:build_training_components",
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
                    "provider": "openai_compatible_chat",
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
                    "provider": "openai_compatible_chat",
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
    required_keys = ", ".join(contract["required_keys"])
    return (
        "You are an OpenAI-compatible candidate-package provider. "
        "The user payload is the only source of task-specific context, public evidence, component "
        "contract details, and output-field constraints. Return exactly one valid JSON object and "
        "no prose outside JSON. The JSON object must contain only the keys required by the user "
        f"payload provider_json_output contract, currently: {required_keys}. "
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


def _json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _json_object_from_text(text: str, name: str) -> dict[str, Any]:
    try:
        return _json_object(json.loads(text), name)
    except json.JSONDecodeError as strict_error:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise ValueError(f"{name} was not valid JSON: {_safe_error(text)}") from strict_error
        try:
            return _json_object(json.loads(extracted), name)
        except json.JSONDecodeError as extracted_error:
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
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError("declared_changed_components must be a JSON array of strings")
    changed = set(value)
    unsupported = sorted(changed - _ALLOWED_DECLARED_COMPONENTS)
    if unsupported:
        joined = ", ".join(unsupported)
        raise ValueError(f"declared_changed_components contains unsupported values: {joined}")
    return changed


def _reject_secret_markers(text: str) -> None:
    if "sk-" in text:
        raise ValueError("candidate text contains a secret-looking marker")


def _safe_error(text: str) -> str:
    compact = " ".join(text.split())
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-REDACTED", compact)[:1200]
