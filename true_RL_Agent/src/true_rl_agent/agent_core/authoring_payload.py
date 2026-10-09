"""Shared authoring JSON parser and candidate package materializer."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Mapping
from true_rl_agent.agent_core.authoring_response import AuthoringProviderResponse, CandidateDraftResponse, SourceRequestResponse
from true_rl_agent.agent_core.provider_contract import (provider_authoring_response_mode, provider_json_output_contract,
    validate_declared_changed_components, validate_provider_json_output_keys, validate_source_request_json_output_keys)
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
from true_rl_agent.agent_core.public_safety import contains_forbidden_public_value_marker, redact_secret_key_markers
from true_rl_agent.candidates.component_source_semantics import extract_training_components_rationale
from true_rl_agent.candidates.loader import CANDIDATE_SCHEMA_VERSION, CONTROLLED_ENTRYPOINT
from true_rl_agent.candidates.package_contract import CANDIDATE_PACKAGE_FILE_MAX_BYTES
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef, require_candidate_id
from true_rl_agent.json_contract import strict_json_object_loads
from true_rl_agent.prompting.source_catalog import prompt_source_catalog_logical_ids


class AuthoringPayloadMaterializer:
    def __init__(self, output_dir: Path, *, backend: str):
        self.output_dir = output_dir
        self.backend = backend
        self._counter = _existing_candidate_counter(output_dir)

    def begin_attempt(self, *, sequence: int | None = None):
        self._counter = self._counter + 1 if sequence is None else sequence

    def parse(self, response_text: str, *, prompt_payload: Mapping[str, Any],
              provider_usage: Mapping[str, int], provider_receipt: Mapping[str, Any],
              finish_reason: str | None = None) -> AuthoringProviderResponse:
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
                        "materialized_by": ("openai_compatible_candidate_materializer" if self.backend == "openai_compatible_chat" else "authoring_payload_materializer"),
                        "provider_backend": self.backend,
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
                        "materialized_by": ("openai_compatible_candidate_materializer" if self.backend == "openai_compatible_chat" else "authoring_payload_materializer"),
                        "provider_backend": self.backend,
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

        candidate_dir = self.output_dir / f"{self._counter:03d}_{candidate_id}"
        candidate_dir.mkdir(parents=True, exist_ok=False)
        for relative_path, content in file_bytes.items():
            target = candidate_dir / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
        return CandidatePackageRef(
            candidate_id=candidate_id,
            path=str(candidate_dir),
            change_set=CandidateChangeSet(frozenset(changed_components)),
            metadata={"provider": self.backend, "declared_change_set_provisional": True},
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

def authoring_system_prompt(*, backend="openai_compatible_chat") -> str:
    contract = provider_json_output_contract()
    draft_required = ", ".join(contract["one_of"]["candidate_draft"]["required_keys"])
    source_request_required = ", ".join(contract["one_of"]["source_request"]["required_keys"])
    return (
        ("You are an OpenAI-compatible candidate-package provider. " if backend == "openai_compatible_chat"
         else "You are an external candidate-package author. ") +
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

def _json_object_from_text(text: str, name: str) -> dict[str, Any]:
    safe_text = redact_secret_key_markers(" ".join(text.split()))[:1200]
    try:
        return strict_json_object_loads(text, context=name)
    except ValueError as strict_error:
        extracted = _extract_first_json_object(text)
        if extracted is None:
            raise ValueError(f"{name} was not valid JSON: {safe_text}") from strict_error
        try:
            return strict_json_object_loads(extracted, context=name)
        except ValueError as extracted_error:
            raise ValueError(f"{name} contained an invalid JSON object: {safe_text}") from extracted_error


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
