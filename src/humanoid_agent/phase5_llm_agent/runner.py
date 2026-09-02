"""Run one Phase 5 LLM-agent proposal iteration."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from humanoid_agent.phase3_verifier.verifier import (
    _load_structured_file,  # noqa: PLC2701 - reuse local YAML fallback.
    load_verifier_config,
    verify_static_documents,
)
from humanoid_agent.phase4_rule_baseline.rules import dump_yaml_like, render_unified_diff, stable_json
from humanoid_agent.phase5_llm_agent.patching import PatchContractError, apply_llm_patch
from humanoid_agent.phase5_llm_agent.prompting import build_phase5_messages, messages_to_text
from humanoid_agent.phase5_llm_agent.providers import ProviderError, call_provider


def run_phase5_iteration(
    *,
    repo_root: Path,
    summary_file: Path,
    phase1_config_file: Path,
    phase2_config_file: Path,
    phase5_config_file: Path,
    verifier_config_file: Path,
    output_dir: Path,
    provider_name: str | None = None,
    replay_response: Path | None = None,
    allow_api_spend: bool = False,
    skip_verify: bool = False,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    phase1_path = _resolve(repo_root, phase1_config_file)
    phase2_path = _resolve(repo_root, phase2_config_file)
    phase5_path = _resolve(repo_root, phase5_config_file)
    verifier_path = _resolve(repo_root, verifier_config_file)
    summary = _load_json(summary_file)
    phase1 = _load_structured_file(phase1_path)
    phase2 = _load_structured_file(phase2_path)
    phase5 = _load_structured_file(phase5_path)
    verifier_settings = load_verifier_config(verifier_path)
    reward_bounds = _load_reward_bounds(repo_root, phase5)
    # The Phase 6 loop may feed a current-best config from a run directory
    # rather than directly from configs/.  The verifier must still protect the
    # canonical Phase 1/2 document contracts; otherwise moving the file would
    # accidentally bypass protected_keys checks.
    phase1_key = str(phase5["patch_contract"]["documents"]["phase1"])
    phase2_key = str(phase5["patch_contract"]["documents"]["phase2"])
    verifier_key = _repo_key(verifier_config_file)

    provider = provider_name or phase5.get("provider", {}).get("default", "dry_run")
    messages = build_phase5_messages(
        summary=summary,
        phase1_config=phase1,
        phase2_config=phase2,
        phase5_config=phase5,
    )
    (output_dir / "prompt.txt").write_text(messages_to_text(messages), encoding="utf-8")
    (output_dir / "prompt_messages.json").write_text(stable_json(messages), encoding="utf-8")

    try:
        provider_result = call_provider(
            provider_name=provider,
            messages=messages,
            phase5_config=phase5,
            replay_response=replay_response,
            allow_api_spend=allow_api_spend,
        )
        response = _parse_llm_json(provider_result.content)
        candidate_phase1, candidate_phase2, applied = apply_llm_patch(
            phase1_config=phase1,
            phase2_config=phase2,
            response=response,
            phase5_config=phase5,
            reward_bounds=reward_bounds,
        )
        status = "pass"
        errors: list[dict[str, Any]] = []
    except (ProviderError, PatchContractError, ValueError, json.JSONDecodeError) as exc:
        provider_result = None
        response = {}
        candidate_phase1 = phase1
        candidate_phase2 = phase2
        applied = []
        status = "fail"
        errors = [{"type": type(exc).__name__, "message": str(exc)}]

    if provider_result is not None:
        (output_dir / "provider_request_summary.json").write_text(
            stable_json(provider_result.request_summary),
            encoding="utf-8",
        )
        (output_dir / "raw_response.json").write_text(stable_json(provider_result.raw_response), encoding="utf-8")
        (output_dir / "parsed_response.json").write_text(stable_json(response), encoding="utf-8")

    candidate_phase1_text = dump_yaml_like(candidate_phase1)
    candidate_phase2_text = dump_yaml_like(candidate_phase2)
    phase1_out = output_dir / "candidate_phase1_push_recovery.yaml"
    phase2_out = output_dir / "candidate_phase2_baseline.yaml"
    phase1_out.write_text(candidate_phase1_text, encoding="utf-8")
    phase2_out.write_text(candidate_phase2_text, encoding="utf-8")

    diff_text = (
        render_unified_diff(
            phase1_path.read_text(encoding="utf-8"),
            candidate_phase1_text,
            fromfile=str(phase1_config_file),
            tofile=_display_path(phase1_out, repo_root),
        )
        + render_unified_diff(
            phase2_path.read_text(encoding="utf-8"),
            candidate_phase2_text,
            fromfile=str(phase2_config_file),
            tofile=_display_path(phase2_out, repo_root),
        )
    )
    (output_dir / "candidate.diff").write_text(diff_text, encoding="utf-8")

    verifier_summary = None
    verifier_status = "skipped"
    if status == "pass" and not skip_verify:
        verifier_summary = verify_static_documents(
            base_docs={
                phase1_key: phase1,
                phase2_key: phase2,
                verifier_key: verifier_settings,
            },
            current_docs={
                phase1_key: candidate_phase1,
                phase2_key: candidate_phase2,
                verifier_key: verifier_settings,
            },
            changed_files=[phase1_key, phase2_key],
            settings=verifier_settings,
            repo_root=repo_root,
            check_imports=False,
        )
        verifier_status = verifier_summary["status"]
        (output_dir / "verifier_summary.json").write_text(stable_json(verifier_summary), encoding="utf-8")
        if verifier_status != "pass":
            status = "fail"

    analysis = _build_analysis(
        response=response,
        provider_result=provider_result,
        source_summary=summary,
        applied=applied,
        verifier_status=verifier_status,
        errors=errors,
    )
    (output_dir / "analysis.json").write_text(stable_json(analysis), encoding="utf-8")
    operation_diff = _build_operation_diff(applied=applied, verifier_status=verifier_status)
    (output_dir / "operation_diff.json").write_text(stable_json(operation_diff), encoding="utf-8")
    (output_dir / "operation_diff.md").write_text(_render_operation_diff_markdown(operation_diff), encoding="utf-8")

    run_summary = {
        "schema_version": "humanoid_agent.phase5.llm_agent.run.v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "output_dir": str(output_dir),
        "analysis_file": str(output_dir / "analysis.json"),
        "candidate_phase1_config": str(phase1_out),
        "candidate_phase2_config": str(phase2_out),
        "candidate_diff": str(output_dir / "candidate.diff"),
        "operation_diff": str(output_dir / "operation_diff.json"),
        "operation_diff_markdown": str(output_dir / "operation_diff.md"),
        "verifier_status": verifier_status,
        "verifier_summary_file": str(output_dir / "verifier_summary.json") if verifier_summary else None,
        "errors": errors,
    }
    (output_dir / "run_summary.json").write_text(stable_json(run_summary), encoding="utf-8")
    return run_summary


def _parse_llm_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("LLM response JSON must be an object")
    if not isinstance(data.get("analysis"), dict):
        raise ValueError("LLM response must contain analysis object")
    return data


def _build_analysis(
    *,
    response: dict[str, Any],
    provider_result: Any,
    source_summary: dict[str, Any],
    applied: list[dict[str, Any]],
    verifier_status: str,
    errors: list[dict[str, Any]],
) -> dict[str, Any]:
    provider_meta = None
    if provider_result is not None:
        provider_meta = {
            "provider": provider_result.provider,
            "model": provider_result.model,
            "request_summary": provider_result.request_summary,
        }
    return {
        "schema_version": "humanoid_agent.phase5.llm_agent.analysis.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": provider_meta,
        "source_summary": {
            "schema_version": source_summary.get("schema_version"),
            "status": source_summary.get("status"),
            "run_dir": source_summary.get("run_dir"),
            "public_success_mean": source_summary.get("public_eval", {}).get("success_rate_stats", {}).get("mean"),
            "public_success_values": source_summary.get("public_eval", {}).get("success_rate_stats", {}).get("values"),
        },
        "llm_analysis": response.get("analysis", {}),
        "applied_changes": applied,
        "verifier_status": verifier_status,
        "errors": errors,
        "security_notes": [
            "API keys are read only from environment variables and are not written to artifacts.",
            "LLM output is a JSON patch contract; local code applies it and Phase 3 verifies it.",
            "IID-hidden results are not included in the proposal prompt.",
        ],
    }


def _build_operation_diff(*, applied: list[dict[str, Any]], verifier_status: str) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase5.operation_diff.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Semantic operation-only candidate diff for Agent feedback; excludes YAML serialization noise.",
        "verifier_status": verifier_status,
        "operations": applied,
    }


def _render_operation_diff_markdown(operation_diff: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 operation-only diff",
        "",
        "This file records semantic config operations only. It intentionally excludes full-file YAML reserialization noise.",
        "",
        f"Verifier status: `{operation_diff.get('verifier_status')}`",
        "",
        "| # | document | path | old | new | reason |",
        "| ---: | --- | --- | --- | --- | --- |",
    ]
    for index, item in enumerate(operation_diff.get("operations", []), start=1):
        lines.append(
            "| {index} | `{document}` | `{path}` | `{old}` | `{new}` | {reason} |".format(
                index=index,
                document=item.get("document"),
                path=item.get("path"),
                old=json.dumps(item.get("old"), ensure_ascii=False, sort_keys=True),
                new=json.dumps(item.get("new"), ensure_ascii=False, sort_keys=True),
                reason=str(item.get("reason", "")).replace("|", "\\|"),
            )
        )
    return "\n".join(lines) + "\n"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def _load_reward_bounds(repo_root: Path, phase5_config: dict[str, Any]) -> dict[str, Any]:
    source = phase5_config.get("patch_contract", {}).get("reward_weight_bounds_source")
    if not source:
        return {}
    data = _load_structured_file(_resolve(repo_root, Path(source)))
    return data.get("reward_weight_bounds", {}) if isinstance(data, dict) else {}


def _resolve(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _display_path(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def _repo_key(path: Path) -> str:
    return path.as_posix()
