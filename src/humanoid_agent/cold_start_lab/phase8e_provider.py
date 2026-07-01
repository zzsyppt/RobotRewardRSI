"""Phase 8E provider output ingestion for code-level candidate packages."""

from __future__ import annotations

import json
import hashlib
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from humanoid_agent.code_sandbox.isolated_loader import load_mapping
from humanoid_agent.code_sandbox.loader import CandidateLoadError
from humanoid_agent.code_sandbox.verifier import verify_candidate_package
from humanoid_agent.cold_start_lab.phase8e import (
    Phase8EError,
    SOURCE_PACK_FORBIDDEN_TOKENS,
    SOURCE_PACK_MAX_TEXT_BYTES,
    assert_prompt_facing_payload,
    assert_public_only_payload,
    build_phase8e_prompt_package,
    load_ledger,
    project_candidate_package,
    project_verifier_summary,
    resolve_current_recipe,
)
from humanoid_agent.cold_start_lab.phase8e_common import provider_source_read_turn_budget, scrub_prompt_facing_value
from humanoid_agent.cold_start_lab.curriculum_contract import (
    CurriculumContractError,
    validate_v1_curriculum_candidate_artifacts,
)
from humanoid_agent.cold_start_lab.phase8e_projection import CandidateOptionalJsonError
from humanoid_agent.phase5_llm_agent.providers import ProviderError, ProviderResult, call_provider


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PHASE8E_PROVIDER_SUMMARY_SCHEMA = "humanoid_agent.phase8e.provider_candidate.summary.v1"
REPAIR_FEEDBACK_LOCAL_PATH = "[omitted: server-local path]"
ABSOLUTE_POSIX_PATH_RE = re.compile(r"(?<![:\w.-])/(?:[^\s\"'=,:;{}\[\]<>|]+/)*[^\s\"'=,:;{}\[\]<>|]+")
ABSOLUTE_WINDOWS_DRIVE_PATH_RE = re.compile(
    r"(?<![\w.-])[A-Za-z]:\\(?:[^\s\"'=,:;{}\[\]<>|\\]+\\)*[^\s\"'=,:;{}\[\]<>|\\]+"
)
ABSOLUTE_WINDOWS_UNC_PATH_RE = re.compile(
    r"(?<![\w.-])\\\\[^\s\"'=,:;{}\[\]<>|\\]+\\(?:[^\s\"'=,:;{}\[\]<>|\\]+\\)*[^\s\"'=,:;{}\[\]<>|\\]+"
)
REPAIR_FEEDBACK_DROP_KEYS = {
    "candidate_dir",
    "config_path",
    "entrypoint_path",
    "manifest_path",
    "run_dir",
}
DEFAULT_ALLOWED_TOP_LEVEL = {
    "candidate_manifest.yaml",
    "training_code",
    "rationale.md",
    "operation_diff.json",
    "diagnostics.json",
}


class Phase8EProviderError(RuntimeError):
    """Raised when a provider response cannot become a safe candidate package."""


class Phase8EInnerLoopFailure(Phase8EProviderError):
    """Raised when the provider inner loop rejects all candidates before training."""

    def __init__(self, message: str, *, inner_loop: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.inner_loop = dict(inner_loop)


@dataclass(frozen=True)
class CandidateFile:
    path: str
    content: str


def resolve_source_requests(
    *,
    source_requests: Sequence[Mapping[str, Any]],
    source_pack: Mapping[str, Any],
    source_store: Mapping[str, Any] | None = None,
    config: Mapping[str, Any],
    artifact_path: Path,
    max_requests: int,
    sensitive_values: Sequence[str] = (),
) -> dict[str, Any]:
    """Resolve untrusted LLM source requests against an audited logical-id pack."""

    if max_requests < 0:
        raise Phase8EProviderError("source request max_requests must be non-negative")
    if len(source_requests) > max_requests:
        raise Phase8EProviderError("source request count exceeds configured budget")
    raw_items = source_pack.get("items")
    if not isinstance(raw_items, Mapping):
        raise Phase8EProviderError("source pack items must be a mapping")
    store_items = source_store.get("items") if isinstance(source_store, Mapping) else None
    if source_store is not None and not isinstance(store_items, Mapping):
        raise Phase8EProviderError("source store items must be a mapping")

    payloads: list[dict[str, Any]] = []
    public_requests: list[dict[str, str]] = []
    for index, request in enumerate(source_requests):
        if not isinstance(request, Mapping):
            raise Phase8EProviderError(f"source request {index} must be an object")
        logical_id = request.get("id")
        if not isinstance(logical_id, str) or not logical_id:
            raise Phase8EProviderError(f"source request {index} id must be a non-empty string")
        _validate_source_request_id(logical_id)
        if logical_id not in raw_items:
            raise Phase8EProviderError(f"source request id is not allowlisted: {logical_id}")
        item = raw_items[logical_id]
        if not isinstance(item, Mapping):
            raise Phase8EProviderError(f"source pack item must be an object: {logical_id}")
        store_item = store_items.get(logical_id) if isinstance(store_items, Mapping) else None
        if store_item is not None and not isinstance(store_item, Mapping):
            raise Phase8EProviderError(f"source store item must be an object: {logical_id}")
        content_item = store_item if isinstance(store_item, Mapping) else item
        content = content_item.get("content")
        if not isinstance(content, str):
            if item.get("content_available") is True and not isinstance(store_item, Mapping):
                raise Phase8EProviderError(
                    f"manifest-only source pack item requires backing source_store content: {logical_id}"
                )
            raise Phase8EProviderError(f"source pack item is missing text content: {logical_id}")
        encoded = content.encode("utf-8")
        if len(encoded) > SOURCE_PACK_MAX_TEXT_BYTES:
            raise Phase8EProviderError("source request content exceeds size limit")
        expected_sha = item.get("content_sha256")
        actual_sha = hashlib.sha256(encoded).hexdigest()
        if isinstance(expected_sha, str) and expected_sha and expected_sha != actual_sha:
            raise Phase8EProviderError("source request content hash mismatch")
        _audit_source_request_content(logical_id, content, config=config)
        reason = request.get("reason")
        public_request = {"id": logical_id}
        if reason is not None and not isinstance(reason, str):
            raise Phase8EProviderError(f"source request {index} reason must be a string")
        if isinstance(reason, str) and reason:
            _reject_sensitive_values(reason, sensitive_values)
            _audit_source_request_reason(reason, config=config)
            public_request["reason"] = reason
        public_requests.append(public_request)
        payloads.append(
            {
                "id": logical_id,
                "reason": public_request.get("reason"),
                "kind": content_item.get("kind", item.get("kind")),
                "format": content_item.get("format", item.get("format")),
                "source": content_item.get("source", item.get("source")),
                "content": content,
                "bytes": len(encoded),
                "content_sha256": actual_sha,
                "public_safe": True,
            }
        )

    summary = {
        "schema_version": "humanoid_agent.phase8e.source_request_payloads.v1",
        "status": "pass",
        "requested_count": len(source_requests),
        "returned_count": len(payloads),
        "source_requests": public_requests,
        "payloads": payloads,
        "policy": {
            "ids_are_closed_allowlist": True,
            "request_ids_are_untrusted": True,
            "provider_filesystem_access": False,
        },
    }
    assert_public_only_payload(summary, config=dict(config), context="phase8e_source_request_payloads")
    _write_json(artifact_path, summary)
    return summary


def run_phase8e_provider_candidate(
    *,
    config_path: Path,
    run_dir: Path,
    provider_name: str | None = None,
    replay_response: Path | None = None,
    allow_api_spend: bool = False,
    verify: bool = True,
    require_server_enforcement: bool = False,
    lab_dir: Path | None = None,
    current_recipe_dir: Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Generate and stage one Phase 8E candidate package from a provider.

    This is intentionally a provider/ingestion step only.  It prepares a public
    prompt, asks a provider for a file package, writes a staging candidate dir,
    and optionally runs Phase 8D.  It does not launch training or eval.
    """

    config_path = config_path.resolve()
    config = load_mapping(config_path)
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    effective_lab_dir = _effective_lab_dir(config, run_dir, lab_dir)
    effective_lab_dir.mkdir(parents=True, exist_ok=True)

    ledger = load_ledger(_ledger_path(config, effective_lab_dir))
    current_recipe = resolve_current_recipe(config, ledger, current_recipe_dir=current_recipe_dir)
    prompt_package = build_phase8e_prompt_package(config=config, ledger=ledger, current_recipe_dir=current_recipe)
    _write_prompt_files(config, run_dir, prompt_package)

    provider = provider_name or str(_provider_config(config).get("default", "dry_run"))
    inner_loop_summary: dict[str, Any] | None = None
    try:
        if _provider_inner_loop_enabled(config):
            loop_result = _run_provider_inner_loop(
                provider=provider,
                prompt_package=prompt_package,
                config=config,
                run_dir=run_dir,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
                verify=verify,
                require_server_enforcement=require_server_enforcement,
                python_executable=python_executable,
            )
            provider_result = loop_result["provider_result"]
            candidate_dir = loop_result["candidate_dir"]
            stage_summary = loop_result["stage_summary"]
            verifier_summary = loop_result["verifier_summary"]
            verifier_status = loop_result["verifier_status"]
            inner_loop_summary = loop_result["inner_loop"]
        else:
            provider_result = _call_phase8e_provider(
                provider_name=provider,
                messages=prompt_package["messages"],
                config=config,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
            )
            candidate_dir = run_dir / "candidate_package"
            stage_summary = stage_candidate_package_from_response(
                response_content=provider_result.content,
                candidate_dir=candidate_dir,
                config=config,
                sensitive_values=_sensitive_values(config, provider),
            )
            verifier_summary = None
            verifier_status = "skipped"
            if verify:
                verifier_run_dir = run_dir / "phase8d_verifier"
                verifier_summary = verify_candidate_package(
                    candidate_dir=candidate_dir,
                    config_path=_resolve_project_path(config["verifier"]["config_path"]),
                    run_dir=verifier_run_dir,
                    require_server_enforcement=require_server_enforcement,
                    python_executable=python_executable,
                )
                _write_json(verifier_run_dir / "phase8d_verifier_summary.json", verifier_summary)
                verifier_status = str(verifier_summary.get("status", "unknown"))
        status = "pass" if verifier_status == "pass" else "staged_unverified" if verifier_status == "skipped" else "fail"
        errors: list[dict[str, str]] = []
    except (Phase8EProviderError, Phase8EError, ProviderError, ValueError, json.JSONDecodeError) as exc:
        provider_result = None
        candidate_dir = run_dir / "candidate_package"
        stage_summary = None
        verifier_summary = None
        verifier_status = "skipped"
        if isinstance(exc, Phase8EInnerLoopFailure):
            inner_loop_summary = exc.inner_loop
            status = "reject"
        else:
            status = "fail"
        errors = [{"type": type(exc).__name__, "message": _safe_error_message(str(exc))}]

    summary = {
        "schema_version": PHASE8E_PROVIDER_SUMMARY_SCHEMA,
        "phase": "8E-6-3-provider-to-candidate-package",
        "status": status,
        "created_at_utc": _now(),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "prompt": {
            "messages_file": str(run_dir / str(config["paths"].get("prompt_messages_filename", "phase8e_prompt_messages.json"))),
            "payload_file": str(run_dir / str(config["paths"].get("prompt_payload_filename", "phase8e_prompt_payload.json"))),
            "text_file": str(run_dir / str(config["paths"].get("prompt_text_filename", "phase8e_prompt.txt"))),
            "public_only": True,
        },
        "provider": _provider_public_summary(provider_result, provider_name=provider),
        "candidate_dir": str(candidate_dir),
        "candidate": _safe_project_candidate_package(candidate_dir),
        "stage": stage_summary,
        "inner_loop": inner_loop_summary or {"enabled": False},
        "verifier_status": verifier_status,
        "verifier": verifier_summary if verifier_summary is not None else {"status": "skipped"},
        "verifier_public_projection": project_verifier_summary(verifier_summary),
        "training_ready": status == "pass" and verifier_status == "pass",
        "errors": errors,
        "next_step": "Run linked full-budget train/eval/decision orchestration after a candidate has passed Phase 8D on the target server.",
    }
    assert_public_only_payload(summary, config=config, context="phase8e_provider_candidate_summary")
    _write_json(run_dir / "phase8e_provider_candidate_summary.json", summary)
    return summary


def stage_candidate_package_from_response(
    *,
    response_content: str,
    candidate_dir: Path,
    config: Mapping[str, Any],
    sensitive_values: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Parse provider JSON and write one candidate package into an empty dir."""

    _reject_sensitive_values(response_content, sensitive_values or ())
    response = _parse_response_json(response_content)
    files = _candidate_files(response)
    for item in files:
        _validate_candidate_path(item.path, config)
    if not any(item.path == "candidate_manifest.yaml" for item in files):
        raise Phase8EProviderError("provider candidate package is missing candidate_manifest.yaml")
    if not any(item.path.startswith("training_code/") and item.path.endswith(".py") for item in files):
        raise Phase8EProviderError("provider candidate package is missing training_code/*.py")
    for item in files:
        _validate_candidate_content(item, config=config, sensitive_values=sensitive_values or ())

    candidate_dir = candidate_dir.resolve()
    if candidate_dir.exists() and any(candidate_dir.iterdir()):
        raise Phase8EProviderError(f"candidate staging dir must be empty: {candidate_dir}")

    written: list[dict[str, Any]] = []
    candidate_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{candidate_dir.name}.tmp_", dir=str(candidate_dir.parent))).resolve()
    try:
        for item in files:
            target = (temp_dir / Path(*PurePosixPath(item.path).parts)).resolve()
            if temp_dir not in target.parents and target != temp_dir:
                raise Phase8EProviderError(f"unsafe candidate file path: {item.path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(item.content, encoding="utf-8")
            written.append({"path": item.path, "bytes": len(item.content.encode("utf-8"))})
        temp_dir.rename(candidate_dir)
    except Exception:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    try:
        candidate_projection = project_candidate_package(candidate_dir)
        validate_v1_curriculum_candidate_artifacts(candidate_projection)
    except CandidateLoadError as exc:
        raise Phase8EProviderError(
            f"provider candidate_manifest.yaml failed validation: {_safe_error_message(str(exc))}"
        ) from exc
    except CandidateOptionalJsonError as exc:
        raise Phase8EProviderError(f"provider candidate optional artifact failed validation: {exc}") from exc
    except CurriculumContractError as exc:
        raise Phase8EProviderError(f"provider curriculum_design failed validation: {exc}") from exc
    return {
        "status": "pass",
        "candidate_dir": str(candidate_dir),
        "files_written": written,
        "candidate": candidate_projection,
        "public_only_checked": True,
        "sensitive_values_checked": True,
    }


def _safe_project_candidate_package(candidate_dir: Path) -> dict[str, Any] | None:
    if not candidate_dir.is_dir() or not (candidate_dir / "candidate_manifest.yaml").is_file():
        return None
    try:
        return project_candidate_package(candidate_dir)
    except Exception:
        return None


def _validate_candidate_content(item: CandidateFile, *, config: Mapping[str, Any], sensitive_values: Sequence[str]) -> None:
    try:
        assert_public_only_payload(item.content, config=dict(config), context=f"provider_candidate_file:{item.path}")
    except Phase8EError as exc:
        raise Phase8EProviderError(f"provider candidate file failed public-only check: {item.path}") from exc
    _reject_sensitive_values(item.content, sensitive_values)


def _call_phase8e_provider(
    *,
    provider_name: str,
    messages: list[dict[str, str]],
    config: Mapping[str, Any],
    replay_response: Path | None,
    allow_api_spend: bool,
    call_index: int = 0,
) -> ProviderResult:
    provider_cfg = _provider_config(config)
    provider_type = str(provider_cfg.get(provider_name, {}).get("type", provider_name))
    if provider_type == "dry_run":
        return _phase8e_dry_run_response()
    if provider_type == "replay_sequence":
        if replay_response is None:
            raise ProviderError("replay_sequence provider requires --replay-response")
        return _phase8e_replay_sequence_response(Path(replay_response), provider_name=provider_name, call_index=call_index)
    return call_provider(
        provider_name=provider_name,
        messages=messages,
        phase5_config={"provider": provider_cfg},
        replay_response=replay_response,
        allow_api_spend=allow_api_spend,
    )


def _run_provider_inner_loop(
    *,
    provider: str,
    prompt_package: Mapping[str, Any],
    config: Mapping[str, Any],
    run_dir: Path,
    replay_response: Path | None,
    allow_api_spend: bool,
    verify: bool,
    require_server_enforcement: bool,
    python_executable: str | None,
) -> dict[str, Any]:
    inner_cfg = _provider_inner_loop_config(config)
    max_attempts = int(inner_cfg.get("max_attempts", 3))
    max_source_read_turns = int(inner_cfg.get("max_source_read_turns", 2))
    json_repair_attempts = int(inner_cfg.get("json_repair_attempts", 1))
    root = run_dir
    attempts_root = root / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    base_messages = list(prompt_package["messages"])
    source_pack = prompt_package["payload"]["source_pack"]
    source_store = prompt_package.get("source_store")
    sensitive_values = _sensitive_values(config, provider)
    call_index = 0
    attempts: list[dict[str, Any]] = []
    repair_messages: list[dict[str, str]] | None = None

    for attempt in range(1, max_attempts + 1):
        attempt_dir = attempts_root / f"attempt_{attempt:03d}"
        attempt_dir.mkdir(parents=True, exist_ok=False)
        messages = list(repair_messages) if repair_messages is not None else list(base_messages)
        _write_json(attempt_dir / "prompt_messages.json", messages)
        source_read_turns = 0
        provider_result: ProviderResult | None = None
        candidate_response: str | None = None

        for _source_turn in range(max_source_read_turns):
            turn_result = _call_phase8e_provider(
                provider_name=provider,
                messages=messages,
                config=config,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
                call_index=call_index,
            )
            call_index += 1
            turn_result, response, call_index = _parse_response_json_with_optional_repair(
                provider_result=turn_result,
                provider_name=provider,
                messages=messages,
                config=config,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
                call_index=call_index,
                attempt_dir=attempt_dir,
                max_repair_attempts=json_repair_attempts,
                sensitive_values=sensitive_values,
            )
            if _is_source_request_response(response):
                source_read_turns += 1
                source_requests = response["source_requests"]
                source_summary = resolve_source_requests(
                    source_requests=source_requests,
                    source_pack=source_pack,
                    source_store=source_store if isinstance(source_store, Mapping) else None,
                    config=config,
                    artifact_path=attempt_dir / "source_payloads.json",
                    max_requests=max_source_read_turns,
                    sensitive_values=sensitive_values,
                )
                safe_source_request_response = json.dumps(
                    {"source_requests": source_summary.get("source_requests", [])},
                    ensure_ascii=False,
                    indent=2,
                )
                _write_json(
                    attempt_dir / "source_requests.json",
                    {"source_requests": source_summary.get("source_requests", [])},
                )
                messages = _messages_with_source_payloads(
                    messages,
                    safe_source_request_response,
                    source_summary,
                    remaining_source_read_turns=max_source_read_turns - source_read_turns,
                )
                _write_json(attempt_dir / "prompt_messages.json", messages)
                continue
            provider_result = turn_result
            candidate_response = turn_result.content
            break

        if candidate_response is None:
            provider_result = _call_phase8e_provider(
                provider_name=provider,
                messages=messages,
                config=config,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
                call_index=call_index,
            )
            call_index += 1
            provider_result, response, call_index = _parse_response_json_with_optional_repair(
                provider_result=provider_result,
                provider_name=provider,
                messages=messages,
                config=config,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
                call_index=call_index,
                attempt_dir=attempt_dir,
                max_repair_attempts=json_repair_attempts,
                sensitive_values=sensitive_values,
            )
            if _is_source_request_response(response):
                raise Phase8EProviderError("source request budget exhausted before candidate package")
            candidate_response = provider_result.content

        raw_response_path = attempt_dir / "raw_response.txt"
        if raw_response_path.exists():
            (attempt_dir / "candidate_response.txt").write_text(candidate_response, encoding="utf-8")
        else:
            raw_response_path.write_text(candidate_response, encoding="utf-8")
        staged_candidate = attempt_dir / "staged_candidate"
        verifier_summary = None
        verifier_status = "skipped"
        stage_error: Phase8EProviderError | None = None
        try:
            stage_summary = stage_candidate_package_from_response(
                response_content=candidate_response,
                candidate_dir=staged_candidate,
                config=config,
                sensitive_values=sensitive_values,
            )
            if verify:
                verifier_run_dir = attempt_dir / "phase8d_verifier"
                verifier_summary = verify_candidate_package(
                    candidate_dir=staged_candidate,
                    config_path=_resolve_project_path(config["verifier"]["config_path"]),
                    run_dir=verifier_run_dir,
                    require_server_enforcement=require_server_enforcement,
                    python_executable=python_executable,
                )
                _write_json(verifier_run_dir / "phase8d_verifier_summary.json", verifier_summary)
                verifier_status = str(verifier_summary.get("status", "unknown"))
        except Phase8EProviderError as exc:
            stage_error = exc
            verifier_status = "stage_failed"
            stage_summary = _stage_failure_summary(exc, staged_candidate)
            _write_json(attempt_dir / "stage_failure_summary.json", stage_summary)

        selected = stage_error is None and (verifier_status == "pass" or verifier_status == "skipped")
        attempt_record = {
            "attempt": attempt,
            "status": "selected" if selected else "failed_pretraining_gate",
            "source_read_turns": source_read_turns,
            "staged_candidate_dir": str(staged_candidate),
            "stage_status": stage_summary.get("status"),
            "verifier_status": verifier_status,
        }
        if stage_error is not None:
            attempt_record["error_type"] = type(stage_error).__name__
            attempt_record["error_message"] = _safe_error_message(str(stage_error))
        attempts.append(attempt_record)
        if selected:
            candidate_dir = run_dir / "candidate_package"
            if candidate_dir.exists():
                shutil.rmtree(candidate_dir)
            shutil.copytree(staged_candidate, candidate_dir, symlinks=False)
            selected_attempt = {
                "attempt": attempt,
                "candidate_dir": str(candidate_dir),
                "staged_candidate_dir": str(staged_candidate),
                "source_read_turns": source_read_turns,
                "verifier_status": verifier_status,
            }
            _write_json(root / "selected_attempt.json", selected_attempt)
            return {
                "provider_result": provider_result,
                "candidate_dir": candidate_dir,
                "stage_summary": stage_summary,
                "verifier_summary": verifier_summary,
                "verifier_status": verifier_status,
                "inner_loop": {
                    "enabled": True,
                    "status": "selected",
                    "attempts": attempts,
                    "selected_attempt": attempt,
                    "provider_call_count": call_index,
                    "parent_imported_candidate_code": False,
                },
            }
        if attempt < max_attempts:
            repair_feedback = _build_repair_feedback(
                attempt=attempt,
                prompt_package=prompt_package,
                staged_candidate=staged_candidate,
                stage_summary=stage_summary,
                verifier_summary=verifier_summary,
                verifier_status=verifier_status,
                config=config,
                sensitive_values=sensitive_values,
            )
            repair_feedback_path = attempt_dir / "repair_feedback.json"
            _write_json(repair_feedback_path, repair_feedback)
            attempt_record["repair_feedback_file"] = str(repair_feedback_path)
            repair_messages = _messages_with_repair_feedback(base_messages, repair_feedback)

    inner_loop = {
        "enabled": True,
        "status": "rejected",
        "reason": "candidate_failed_pretraining_inner_loop",
        "attempts": attempts,
        "selected_attempt": None,
        "provider_call_count": call_index,
        "parent_imported_candidate_code": False,
    }
    _write_json(root / "selected_attempt.json", {"attempt": None, "reason": "candidate_failed_pretraining_inner_loop", "attempts": attempts})
    raise Phase8EInnerLoopFailure("candidate_failed_pretraining_inner_loop", inner_loop=inner_loop)


def _phase8e_dry_run_response() -> ProviderResult:
    content = json.dumps(
        {
            "candidate_package": {
                "files": {
                    "candidate_manifest.yaml": json.dumps(
                        {
                            "schema_version": "humanoid_agent.cold_start.candidate.v1",
                            "candidate_id": "phase8e_provider_dry_run_noop",
                            "description": "Deterministic no-op package for Phase 8E provider ingestion.",
                            "author": "llm",
                            "llm_authored": True,
                            "entrypoint": "training_code.components:build_training_components",
                            "phase": "8E-6-3",
                            "expected_behavior": "no training behavior changes",
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    "training_code/components.py": (
                        "\"\"\"Deterministic no-op package for Phase 8E provider ingestion.\"\"\"\n\n"
                        "from __future__ import annotations\n\n"
                        "from humanoid_agent.code_sandbox.components import TrainingComponents\n\n\n"
                        "def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:\n"
                        "    _ = (task_spec, base_env_cfg, base_agent_cfg)\n"
                        "    return TrainingComponents(\n"
                        "        name=\"phase8e_provider_dry_run_noop\",\n"
                        "        rationale=\"No-op package for provider ingestion plumbing.\",\n"
                        "    )\n"
                    ),
                    "rationale.md": "No-op package used only to validate provider ingestion plumbing.\n",
                    "operation_diff.json": json.dumps(
                        {
                            "schema_version": "humanoid_agent.phase8e.operation_diff.v1",
                            "operations": [],
                            "summary": "Two-stage V1 curriculum design for dry-run provider ingestion.",
                            "curriculum_design": {
                                "stage_list": [
                                    {
                                        "name": "balance_warmup",
                                        "progress_range": [0.0, 0.35],
                                        "capabilities": ["external_push_curriculum", "train_only_dr_scale"],
                                        "external_push_curriculum": {"relative_difficulty": "reduced"},
                                        "train_only_dr_scale": 0.25,
                                        "reward_weight_scale": 1.0,
                                        "rationale": "Start with reduced train-only perturbation and robustness variation.",
                                    },
                                    {
                                        "name": "target_consolidation",
                                        "progress_range": [0.35, 1.0],
                                        "capabilities": ["external_push_curriculum", "train_only_dr_scale"],
                                        "external_push_curriculum": {"relative_difficulty": "target"},
                                        "train_only_dr_scale": 1.0,
                                        "reward_weight_scale": 1.0,
                                        "rationale": "Return to the runner-owned target distribution for consolidation.",
                                    },
                                ],
                                "final_consolidation": {
                                    "fraction": 0.65,
                                    "distribution": "runner_owned_target_distribution",
                                    "includes_target_push": True,
                                    "includes_target_robustness_variation": True,
                                },
                                "anti_tuning_rationale": "The design changes train-only curriculum structure, not just scalar reward weights.",
                                "improvement_attribution_intent": "Attribute improvement to the staged curriculum plus reward terms.",
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    "diagnostics.json": json.dumps(
                        {
                            "schema_version": "humanoid_agent.phase8e.diagnostics.v1",
                            "items": [],
                            "curriculum_hypothesis": {
                                "summary": (
                                    "A warmup-to-target curriculum should reduce early falls while preserving target "
                                    "performance."
                                ),
                                "final_consolidation": {
                                    "fraction": 0.65,
                                    "distribution": "runner_owned_target_distribution",
                                    "includes_target_push": True,
                                    "includes_target_robustness_variation": True,
                                },
                                "anti_tuning_rationale": (
                                    "The proposal changes stage structure and target exposure, not only scalar weights."
                                ),
                                "improvement_attribution_intent": (
                                    "Attribute improvement to the staged curriculum plus reward terms."
                                ),
                            },
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                }
            }
        },
        ensure_ascii=False,
    )
    return ProviderResult(
        provider="dry_run",
        model="deterministic-local-phase8e-dry-run",
        content=content,
        raw_response={"dry_run": True},
        request_summary={"provider": "dry_run", "network": False, "phase": "8E-6-3"},
    )


def _phase8e_replay_sequence_response(path: Path, *, provider_name: str, call_index: int) -> ProviderResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    responses = data.get("responses") if isinstance(data, Mapping) else None
    if not isinstance(responses, list):
        raise ProviderError("replay_sequence file must contain a responses list")
    if call_index < 0 or call_index >= len(responses):
        raise ProviderError(f"replay_sequence response index out of range: {call_index}")
    item = responses[call_index]
    if isinstance(item, str):
        content = item
        raw_item: Any = {"content": item}
    elif isinstance(item, Mapping):
        content = str(item.get("content")) if isinstance(item.get("content"), str) else json.dumps(item, ensure_ascii=False)
        raw_item = dict(item)
    else:
        raise ProviderError(f"replay_sequence responses[{call_index}] must be a JSON object or string")
    return ProviderResult(
        provider=provider_name,
        model="replayed-sequence",
        content=content,
        raw_response={"replay_sequence": True, "call_index": call_index, "response": raw_item},
        request_summary={"provider": "replay_sequence", "path": str(path), "call_index": call_index, "network": False},
    )


def _parse_response_json_with_optional_repair(
    *,
    provider_result: ProviderResult,
    provider_name: str,
    messages: list[dict[str, str]],
    config: Mapping[str, Any],
    replay_response: Path | None,
    allow_api_spend: bool,
    call_index: int,
    attempt_dir: Path,
    max_repair_attempts: int,
    sensitive_values: Sequence[str],
) -> tuple[ProviderResult, dict[str, Any], int]:
    _reject_sensitive_values(provider_result.content, sensitive_values)
    try:
        return provider_result, _parse_response_json(provider_result.content), call_index
    except json.JSONDecodeError as first_exc:
        if max_repair_attempts <= 0:
            raise
        assert_public_only_payload(provider_result.content, config=dict(config), context="phase8e_json_repair_raw_response")
        raw_response_path = attempt_dir / "raw_response.txt"
        if not raw_response_path.exists():
            raw_response_path.write_text(provider_result.content, encoding="utf-8")
        summary: dict[str, Any] = {
            "schema_version": "humanoid_agent.phase8e.json_repair_summary.v1",
            "status": "fail",
            "raw_response_file": str(raw_response_path),
            "raw_response_bytes": len(provider_result.content.encode("utf-8")),
            "raw_response_sha256": hashlib.sha256(provider_result.content.encode("utf-8")).hexdigest(),
            "initial_error": {"type": type(first_exc).__name__, "message": _safe_error_message(str(first_exc))},
            "max_repair_attempts": max_repair_attempts,
            "repair_attempts": [],
            "semantic_rewrite_allowed": False,
        }
        repair_messages = _messages_with_json_repair_request(messages, provider_result.content, first_exc)
        last_exc: json.JSONDecodeError = first_exc
        for repair_attempt in range(1, max_repair_attempts + 1):
            repaired_result = _call_phase8e_provider(
                provider_name=provider_name,
                messages=repair_messages,
                config=config,
                replay_response=replay_response,
                allow_api_spend=allow_api_spend,
                call_index=call_index,
            )
            call_index += 1
            _reject_sensitive_values(repaired_result.content, sensitive_values)
            repaired_path = attempt_dir / f"json_repaired_response_{repair_attempt:03d}.txt"
            repaired_path.write_text(repaired_result.content, encoding="utf-8")
            try:
                parsed = _parse_response_json(repaired_result.content)
            except json.JSONDecodeError as exc:
                last_exc = exc
                summary["repair_attempts"].append(
                    {
                        "attempt": repair_attempt,
                        "status": "failed_parse",
                        "response_file": str(repaired_path),
                        "error": {"type": type(exc).__name__, "message": _safe_error_message(str(exc))},
                    }
                )
                repair_messages = _messages_with_json_repair_request(repair_messages, repaired_result.content, exc)
                continue
            summary["status"] = "pass"
            summary["repair_attempts"].append(
                {
                    "attempt": repair_attempt,
                    "status": "pass",
                    "response_file": str(repaired_path),
                    "response_bytes": len(repaired_result.content.encode("utf-8")),
                    "response_sha256": hashlib.sha256(repaired_result.content.encode("utf-8")).hexdigest(),
                }
            )
            _write_json(attempt_dir / "json_repair_summary.json", summary)
            return repaired_result, parsed, call_index
        _write_json(attempt_dir / "json_repair_summary.json", summary)
        raise Phase8EProviderError("provider response JSON repair failed") from last_exc


def _parse_response_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            raise Phase8EProviderError("provider response contains an unterminated fenced JSON block")
        text = match.group(1).strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise Phase8EProviderError("provider response JSON must be an object")
    return data


def _is_source_request_response(response: Mapping[str, Any]) -> bool:
    return isinstance(response.get("source_requests"), list) and "candidate_package" not in response and "files" not in response


def _messages_with_source_payloads(
    messages: list[dict[str, str]],
    source_request_response: str,
    source_summary: Mapping[str, Any],
    *,
    remaining_source_read_turns: int,
) -> list[dict[str, str]]:
    remaining = max(0, int(remaining_source_read_turns))
    if remaining > 0:
        instruction = (
            "Use only these public source payloads plus the original prompt contract. "
            f"{remaining} source-read turn(s) remain; return another source_requests object only if more audited source is necessary. "
            "Otherwise return the required candidate_package JSON."
        )
    else:
        instruction = (
            "Use only these public source payloads plus the original prompt contract. "
            "The source-read budget is exhausted; now return the required candidate_package JSON."
        )
    payload = {
        "instruction": instruction,
        "source_read_turn_budget": {
            "remaining_turns": remaining,
            "provider_filesystem_access": False,
        },
        "source_payloads": source_summary.get("payloads", []),
    }
    return [
        *messages,
        {"role": "assistant", "content": source_request_response},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
    ]


def _messages_with_json_repair_request(
    messages: list[dict[str, str]],
    raw_response: str,
    error: json.JSONDecodeError,
) -> list[dict[str, str]]:
    payload = {
        "instruction": (
            "Return only valid JSON. Preserve the candidate package semantics and file contents exactly except for "
            "the minimal escaping, quoting, brace, comma, or fence fixes required to make the response parse."
        ),
        "json_repair_request": {
            "role": "syntax_only_repair",
            "semantic_rewrite_allowed": False,
            "raw_response": raw_response,
            "parse_error": {"type": type(error).__name__, "message": _safe_error_message(str(error))},
        },
    }
    return [*messages, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}]


def _build_repair_feedback(
    *,
    attempt: int,
    prompt_package: Mapping[str, Any],
    staged_candidate: Path,
    stage_summary: Mapping[str, Any],
    verifier_summary: Mapping[str, Any] | None,
    verifier_status: str,
    config: Mapping[str, Any],
    sensitive_values: Sequence[str],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "humanoid_agent.phase8e.repair_feedback.v1",
        "status": "repair_required",
        "failed_attempt": attempt,
        "gate": "provider_candidate_staging"
        if verifier_status == "stage_failed"
        else "phase8d_pretraining_static_or_dynamic_fast",
        "verifier_status": verifier_status,
        "issues": [
            *_stage_issues_for_repair(stage_summary),
            *_verifier_issues_for_repair(verifier_summary),
        ],
        "stage": _stage_summary_for_repair(stage_summary),
        "candidate_files": _prompt_facing_repair_value(
            _read_candidate_files_for_repair(
                staged_candidate,
                config=config,
                sensitive_values=sensitive_values,
            )
        ),
        "runner_validation_is_authoritative": True,
        "candidate_self_certification_is_not_accepted": True,
    }
    prompt_payload = prompt_package.get("payload")
    if isinstance(prompt_payload, Mapping):
        checklist = prompt_payload.get("candidate_self_audit_checklist")
        if isinstance(checklist, Mapping):
            payload["candidate_self_audit_checklist"] = checklist
    if verifier_summary is not None:
        payload["verifier_public_projection"] = project_verifier_summary(verifier_summary)
    payload = _prompt_facing_repair_value(payload)
    assert_public_only_payload(payload, config=dict(config), context="phase8e_repair_feedback")
    assert_prompt_facing_payload(payload, context="phase8e_repair_feedback")
    return payload


def _stage_failure_summary(error: Exception, staged_candidate: Path) -> dict[str, Any]:
    return {
        "status": "fail",
        "candidate_dir": str(staged_candidate),
        "error": {
            "type": type(error).__name__,
            "message": _safe_error_message(str(error)),
        },
        "candidate": _safe_project_candidate_package(staged_candidate),
        "public_only_checked": True,
        "sensitive_values_checked": True,
    }


def _stage_summary_for_repair(stage_summary: Mapping[str, Any]) -> dict[str, Any]:
    candidate = stage_summary.get("candidate") if isinstance(stage_summary.get("candidate"), Mapping) else {}
    return {
        "status": stage_summary.get("status"),
        "error": stage_summary.get("error") if isinstance(stage_summary.get("error"), Mapping) else None,
        "files_written": stage_summary.get("files_written"),
        "candidate": {
            "manifest": candidate.get("manifest") if isinstance(candidate, Mapping) else None,
            "fingerprint": _prompt_facing_fingerprint(candidate.get("fingerprint")) if isinstance(candidate, Mapping) else None,
            "operation_diff": candidate.get("operation_diff") if isinstance(candidate, Mapping) else None,
            "diagnostics": candidate.get("diagnostics") if isinstance(candidate, Mapping) else None,
        },
        "public_only_checked": stage_summary.get("public_only_checked"),
        "sensitive_values_checked": stage_summary.get("sensitive_values_checked"),
    }


def _stage_issues_for_repair(stage_summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    if stage_summary.get("status") != "fail":
        return []
    error = stage_summary.get("error")
    if not isinstance(error, Mapping):
        return [{"code": "candidate_staging_failed"}]
    issue = {
        "code": "candidate_staging_failed",
        "error_type": error.get("type"),
        "message": error.get("message"),
    }
    message = str(error.get("message", ""))
    if "candidate_manifest.yaml" in message:
        issue["path"] = "candidate_manifest.yaml"
    return _prompt_facing_repair_value([_project_repair_issue(issue)])


def _prompt_facing_fingerprint(fingerprint: Any) -> dict[str, Any] | None:
    if not isinstance(fingerprint, Mapping):
        return None
    return {
        str(key): value
        for key, value in fingerprint.items()
        if str(key) not in {"path", "manifest_path", "entrypoint_path"}
    }


def _prompt_facing_repair_value(value: Any) -> Any:
    return _scrub_repair_local_paths(scrub_prompt_facing_value(value))


def _scrub_repair_local_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text in REPAIR_FEEDBACK_DROP_KEYS:
                continue
            out[key] = _scrub_repair_local_paths(item)
        return out
    if isinstance(value, list):
        return [_scrub_repair_local_paths(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_scrub_repair_local_paths(item) for item in value)
    if isinstance(value, str):
        return _scrub_local_paths(value)
    return value


def _scrub_local_paths(text: str) -> str:
    text = ABSOLUTE_WINDOWS_UNC_PATH_RE.sub(REPAIR_FEEDBACK_LOCAL_PATH, text)
    text = ABSOLUTE_WINDOWS_DRIVE_PATH_RE.sub(REPAIR_FEEDBACK_LOCAL_PATH, text)
    return ABSOLUTE_POSIX_PATH_RE.sub(REPAIR_FEEDBACK_LOCAL_PATH, text)


def _project_repair_issue(issue: Mapping[str, Any], *, parent_code: str | None = None) -> dict[str, Any]:
    allowed_keys = (
        "code",
        "component_name",
        "error_type",
        "import_name",
        "kind",
        "line",
        "message",
        "name",
        "path",
        "reason",
    )
    projected = {
        key: issue.get(key)
        for key in allowed_keys
        if key in issue and _repair_issue_value_is_scalar(issue.get(key))
    }
    if parent_code is not None:
        projected["parent_code"] = parent_code
    return {key: value for key, value in projected.items() if value is not None}


def _repair_issue_value_is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _nested_repair_issues(issue: Mapping[str, Any]) -> list[dict[str, Any]]:
    code = issue.get("code")
    detail = issue.get("detail")
    if not isinstance(detail, Mapping):
        return []
    nested = detail.get("issues")
    if not isinstance(nested, list):
        return []
    parent_code = str(code) if code is not None else None
    return [
        _project_repair_issue(item, parent_code=parent_code)
        for item in nested
        if isinstance(item, Mapping)
    ]


def _read_candidate_files_for_repair(
    candidate_dir: Path,
    *,
    config: Mapping[str, Any],
    sensitive_values: Sequence[str],
) -> dict[str, str]:
    candidate_root = candidate_dir.resolve()
    if not candidate_root.exists():
        return {}
    files: dict[str, str] = {}
    for path in sorted(candidate_root.rglob("*")):
        if path.is_symlink():
            raise Phase8EProviderError(f"repair feedback refuses symlinked candidate file: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if candidate_root not in resolved.parents:
            raise Phase8EProviderError(f"repair feedback candidate file escapes staging dir: {path}")
        rel_path = resolved.relative_to(candidate_root).as_posix()
        _validate_candidate_path(rel_path, config)
        content = resolved.read_text(encoding="utf-8")
        if len(content.encode("utf-8")) > SOURCE_PACK_MAX_TEXT_BYTES:
            raise Phase8EProviderError(f"repair feedback candidate file exceeds size limit: {rel_path}")
        _validate_candidate_content(CandidateFile(path=rel_path, content=content), config=config, sensitive_values=sensitive_values)
        files[rel_path] = content
    return files


def _verifier_issues_for_repair(verifier_summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if verifier_summary is None:
        return []
    issues = verifier_summary.get("issues")
    if not isinstance(issues, list):
        return []
    projected: list[dict[str, Any]] = []
    for issue in issues:
        if not isinstance(issue, Mapping):
            continue
        top_level = _project_repair_issue(issue)
        if top_level:
            projected.append(top_level)
        projected.extend(item for item in _nested_repair_issues(issue) if item)
    return _prompt_facing_repair_value(projected)


def _messages_with_repair_feedback(
    base_messages: list[dict[str, str]],
    repair_feedback: Mapping[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "instruction": (
            "Repair the previous candidate package using the exact runner feedback above. "
            "Return only an allowed source_requests object or the required candidate_package JSON. "
            "Do not claim acceptance yourself; the runner will rerun the static and isolated dynamic fast gates."
        ),
        "repair_feedback": repair_feedback,
    }
    return [*base_messages, {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)}]


def _candidate_files(response: Mapping[str, Any]) -> list[CandidateFile]:
    package = response.get("candidate_package", response)
    if not isinstance(package, Mapping):
        raise Phase8EProviderError("provider response candidate_package must be an object")
    raw_files = package.get("files")
    if isinstance(raw_files, Mapping):
        items = [CandidateFile(path=str(path), content=_require_content(content, str(path))) for path, content in raw_files.items()]
    elif isinstance(raw_files, list):
        items = []
        for index, item in enumerate(raw_files):
            if not isinstance(item, Mapping):
                raise Phase8EProviderError(f"candidate_package.files[{index}] must be an object")
            path = item.get("path")
            if not isinstance(path, str) or not path:
                raise Phase8EProviderError(f"candidate_package.files[{index}].path must be a non-empty string")
            items.append(CandidateFile(path=path, content=_require_content(item.get("content"), path)))
    elif _is_top_level_candidate_file_map(response):
        items = [CandidateFile(path=str(path), content=_require_content(content, str(path))) for path, content in response.items()]
    else:
        raise Phase8EProviderError("provider response must contain candidate_package.files as an object or list")
    if not items:
        raise Phase8EProviderError("provider candidate package contains no files")
    return items


def _require_content(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise Phase8EProviderError(f"candidate file content must be a string: {path}")
    return value


def _is_top_level_candidate_file_map(response: Mapping[str, Any]) -> bool:
    if "candidate_package" in response or "files" in response:
        return False
    if not response:
        return False
    if not all(isinstance(path, str) and isinstance(content, str) for path, content in response.items()):
        return False
    paths = set(response.keys())
    return "candidate_manifest.yaml" in paths and any(path.startswith("training_code/") and path.endswith(".py") for path in paths)


def _validate_candidate_path(path: str, config: Mapping[str, Any]) -> None:
    normalized = path.replace("\\", "/")
    if normalized != path:
        raise Phase8EProviderError(f"unsafe candidate file path: {path}")
    posix = PurePosixPath(path)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise Phase8EProviderError(f"unsafe candidate file path: {path}")
    allowed = _allowed_top_level_entries(config)
    if posix.parts[0] not in allowed:
        raise Phase8EProviderError(f"candidate file top-level entry is not allowed: {path}")


def _validate_source_request_id(logical_id: str) -> None:
    normalized = logical_id.replace("\\", "/")
    if normalized != logical_id:
        raise Phase8EProviderError(f"unsafe source request id: {logical_id}")
    posix = PurePosixPath(logical_id)
    if posix.is_absolute() or not posix.parts or any(part in {"", ".", ".."} for part in posix.parts):
        raise Phase8EProviderError(f"unsafe source request id: {logical_id}")


def _audit_source_request_content(logical_id: str, content: str, *, config: Mapping[str, Any]) -> None:
    try:
        assert_public_only_payload(content, config=dict(config), context=f"source_request:{logical_id}")
        assert_prompt_facing_payload(content, context=f"source_request:{logical_id}")
    except Phase8EError as exc:
        raise Phase8EProviderError("source request content failed audit") from exc
    lowered = content.lower()
    if any(token.lower() in lowered for token in SOURCE_PACK_FORBIDDEN_TOKENS):
        raise Phase8EProviderError("source request content failed audit")


def _audit_source_request_reason(reason: str, *, config: Mapping[str, Any]) -> None:
    try:
        assert_public_only_payload({"reason": reason}, config=dict(config), context="phase8e_source_request_reason")
        assert_prompt_facing_payload({"reason": reason}, context="phase8e_source_request_reason")
    except Phase8EError as exc:
        raise Phase8EProviderError("source request reason failed audit") from exc


def _allowed_top_level_entries(config: Mapping[str, Any]) -> set[str]:
    static = config.get("static") if isinstance(config.get("static"), Mapping) else {}
    entries = static.get("allowed_top_level_entries", []) if isinstance(static, Mapping) else []
    if isinstance(entries, list) and entries:
        return {str(item) for item in entries}
    return set(DEFAULT_ALLOWED_TOP_LEVEL)


def _provider_config(config: Mapping[str, Any]) -> dict[str, Any]:
    provider = config.get("provider")
    if isinstance(provider, Mapping):
        return dict(provider)
    return {
        "default": "dry_run",
        "dry_run": {"type": "dry_run", "model": "deterministic-local-phase8e-dry-run"},
        "replay": {"type": "replay", "model": "replayed-response"},
        "glm": {
            "type": "openai_compatible",
            "api_key_env": "GLM_API_KEY",
            "base_url_env": "GLM_BASE_URL",
            "model_env": "GLM_MODEL",
            "endpoint": "/chat/completions",
            "temperature": 0.2,
            "max_tokens": 8192,
            "response_format": {"type": "json_object"},
        },
    }


def _provider_inner_loop_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("provider_inner_loop")
    if isinstance(raw, Mapping):
        return {
            "enabled": bool(raw.get("enabled", False)),
            "max_attempts": int(raw.get("max_attempts", 3)),
            "max_source_read_turns": provider_source_read_turn_budget(config),
            "json_repair_attempts": int(raw.get("json_repair_attempts", 1)),
            "static_gate_required": bool(raw.get("static_gate_required", True)),
            "isolated_dynamic_fast_validation_required": bool(raw.get("isolated_dynamic_fast_validation_required", True)),
        }
    return {
        "enabled": False,
        "max_attempts": 1,
        "max_source_read_turns": 0,
        "json_repair_attempts": 0,
        "static_gate_required": True,
        "isolated_dynamic_fast_validation_required": True,
    }


def _provider_inner_loop_enabled(config: Mapping[str, Any]) -> bool:
    return bool(_provider_inner_loop_config(config).get("enabled"))


def _provider_public_summary(provider_result: ProviderResult | None, *, provider_name: str) -> dict[str, Any]:
    if provider_result is None:
        return {"provider": provider_name, "status": "not_available"}
    request_summary = dict(provider_result.request_summary)
    redacted_request = {
        key: value
        for key, value in request_summary.items()
        if key not in {"api_key_env", "api_key_present"}
    }
    if "api_key_present" in request_summary:
        redacted_request["credential_present"] = bool(request_summary["api_key_present"])
    return {
        "provider": provider_result.provider,
        "model": provider_result.model,
        "request_summary": redacted_request,
        "response_content_saved": False,
        "raw_response_saved": False,
    }


def _sensitive_values(config: Mapping[str, Any], provider_name: str) -> list[str]:
    provider_cfg = _provider_config(config).get(provider_name, {})
    names: set[str] = set()
    if isinstance(provider_cfg, Mapping):
        for key, value in provider_cfg.items():
            if (
                key.endswith("_env")
                and isinstance(value, str)
                and any(token in value.upper() for token in ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL"))
            ):
                names.add(value)
        api_env = provider_cfg.get("api_key_env")
        if isinstance(api_env, str):
            names.add(api_env)
    values = []
    for name in names:
        value = os.environ.get(name)
        if value and len(value) >= 8:
            values.append(value)
    return values


def _reject_sensitive_values(text: str, values: Iterable[str]) -> None:
    for value in values:
        if value and value in text:
            raise Phase8EProviderError("provider response contains a sensitive value")


def _safe_error_message(message: str) -> str:
    safe = re.sub(r"sk-[A-Za-z0-9_.-]+", "<redacted>", message)
    for token in sorted(SOURCE_PACK_FORBIDDEN_TOKENS, key=len, reverse=True):
        if token:
            safe = re.sub(re.escape(token), "<redacted>", safe, flags=re.IGNORECASE)
    return safe


def _write_prompt_files(config: Mapping[str, Any], run_dir: Path, prompt_package: Mapping[str, Any]) -> None:
    paths = config["paths"]
    _write_json(run_dir / str(paths.get("prompt_messages_filename", "phase8e_prompt_messages.json")), prompt_package["messages"])
    (run_dir / str(paths.get("prompt_text_filename", "phase8e_prompt.txt"))).write_text(str(prompt_package["prompt_text"]), encoding="utf-8")
    _write_prompt_payload_json(
        run_dir / str(paths.get("prompt_payload_filename", "phase8e_prompt_payload.json")),
        prompt_package["payload"],
    )


def _effective_lab_dir(config: Mapping[str, Any], run_dir: Path, lab_dir: Path | None) -> Path:
    if lab_dir is not None:
        return lab_dir.resolve()
    configured = config.get("paths", {}).get("lab_dir") if isinstance(config.get("paths"), Mapping) else None
    if isinstance(configured, str) and configured:
        return _resolve_project_path(configured)
    return run_dir


def _ledger_path(config: Mapping[str, Any], lab_dir: Path) -> Path:
    ledger = str(config.get("paths", {}).get("ledger_filename", "phase8e_ledger.jsonl"))
    path = Path(ledger)
    return path if path.is_absolute() else lab_dir / path


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_prompt_payload_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
