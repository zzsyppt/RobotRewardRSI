#!/usr/bin/env python3
"""Self-test for Phase 8E-6-3 provider-to-candidate-package ingestion."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import humanoid_agent.cold_start_lab.phase8e_provider as phase8e_provider_module  # noqa: E402
from humanoid_agent.code_sandbox.loader import candidate_manifest_prompt_contract  # noqa: E402
from humanoid_agent.cold_start_lab.phase8e_provider import (  # noqa: E402
    Phase8EProviderError,
    _build_repair_feedback,
    _messages_with_json_repair_request,
    _messages_with_repair_feedback,
    _messages_with_source_payloads,
    _sensitive_values,
    resolve_source_requests,
    run_phase8e_provider_candidate,
    stage_candidate_package_from_response,
)
from humanoid_agent.cold_start_lab.phase8e import SOURCE_PACK_MAX_TEXT_BYTES, assert_public_only_payload  # noqa: E402
from humanoid_agent.code_sandbox.isolated_loader import load_mapping  # noqa: E402


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _assert_content_instruction_before_payload(content: str, payload_marker: str, context: str) -> None:
    instruction_pos = content.find('"instruction"')
    payload_pos = content.find(payload_marker)
    _assert(
        instruction_pos >= 0 and payload_pos >= 0 and instruction_pos < payload_pos,
        f"{context} prompt must put instruction before payload marker {payload_marker}: {content[:400]}",
    )


def _assert_instruction_before_payload(prompt_messages_path: Path, payload_marker: str, context: str) -> None:
    messages = json.loads(prompt_messages_path.read_text(encoding="utf-8"))
    _assert(isinstance(messages, list) and messages, f"{context} prompt messages must be a non-empty list")
    content = messages[-1]["content"]
    _assert_content_instruction_before_payload(content, payload_marker, context)


def _valid_curriculum_operation_diff() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.operation_diff.v1",
        "operations": [],
        "summary": "Two-stage V1 curriculum design carried in operation_diff.",
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
    }


def _valid_curriculum_diagnostics() -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.diagnostics.v1",
        "items": [],
        "curriculum_hypothesis": {
            "summary": "A warmup-to-target curriculum should reduce early falls while preserving target performance.",
            "final_consolidation": {
                "fraction": 0.65,
                "distribution": "runner_owned_target_distribution",
                "includes_target_push": True,
                "includes_target_robustness_variation": True,
            },
            "anti_tuning_rationale": "The proposal changes stage structure and target exposure, not only scalar weights.",
            "improvement_attribution_intent": "Attribute improvement to the staged curriculum plus reward terms.",
        },
    }


def _package_response(
    *,
    candidate_id: str = "phase8e_llm_noop",
    operation_diff: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> str:
    manifest = {
        "schema_version": "humanoid_agent.cold_start.candidate.v1",
        "candidate_id": candidate_id,
        "description": "LLM-authored no-op package for Phase 8E-6-3 ingestion self-test.",
        "author": "llm",
        "llm_authored": True,
        "entrypoint": "training_code.components:build_training_components",
        "phase": "8E-6-3",
        "expected_behavior": "no training behavior changes",
    }
    components = '''"""No-op Phase 8E candidate generated by provider ingestion self-test."""

from __future__ import annotations

from humanoid_agent.code_sandbox.components import TrainingComponents


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(
        name="phase8e_llm_noop",
        rationale="No-op provider ingestion regression candidate.",
    )
'''
    return json.dumps(
        {
            "candidate_package": {
                "files": {
                    "candidate_manifest.yaml": json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    "training_code/components.py": components,
                    "rationale.md": "No-op package used to verify provider ingestion only.\n",
                    "operation_diff.json": json.dumps(
                        operation_diff if operation_diff is not None else _valid_curriculum_operation_diff(),
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    "diagnostics.json": json.dumps(
                        diagnostics if diagnostics is not None else _valid_curriculum_diagnostics(),
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


def _package_response_with_forbidden_import(*, candidate_id: str) -> dict[str, Any]:
    response = json.loads(_package_response(candidate_id=candidate_id))
    files = response["candidate_package"]["files"]
    files["training_code/components.py"] = files["training_code/components.py"].replace(
        "from humanoid_agent.code_sandbox.components import TrainingComponents",
        "import subprocess\nfrom humanoid_agent.code_sandbox.components import TrainingComponents",
    )
    return response


def _package_response_missing_manifest_schema(*, candidate_id: str) -> dict[str, Any]:
    response = json.loads(_package_response(candidate_id=candidate_id))
    manifest = json.loads(response["candidate_package"]["files"]["candidate_manifest.yaml"])
    manifest.pop("schema_version")
    response["candidate_package"]["files"]["candidate_manifest.yaml"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    return response


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    _assert(isinstance(data, dict), f"expected JSON object: {path}")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _local_provider_config(root: Path) -> Path:
    config = load_mapping(PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml")
    verifier = load_mapping(PROJECT_ROOT / "configs" / "cold_start" / "phase8d_verifier.yaml")
    verifier["dynamic"]["require_phase8c_isolation"] = False
    verifier_path = root / "phase8d_verifier_local.json"
    _write_json(verifier_path, verifier)
    config["verifier"]["config_path"] = str(verifier_path)
    config["provider"] = dict(config.get("provider", {}))
    config["provider"]["replay_sequence"] = {"type": "replay_sequence", "model": "replayed-sequence"}
    config["provider_inner_loop"] = {
        "enabled": True,
        "max_attempts": 2,
        "max_source_read_turns": 1,
        "json_repair_attempts": 0,
        "static_gate_required": True,
        "isolated_dynamic_fast_validation_required": False,
    }
    config["paths"]["ledger_filename"] = "phase8e_ledger.jsonl"
    path = root / "phase8e_provider_inner_loop.json"
    _write_json(path, config)
    return path


def test_provider_safe_error_message_redacts_public_forbidden_tokens() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-safe-error-") as tmp:
        root = Path(tmp)
        config = load_mapping(_local_provider_config(root))
        message = phase8e_provider_module._safe_error_message(
            "missing DEEPSEEK_API_KEY; api_key_present=False; secret env sk-test-secret"
        )
        summary = {"errors": [{"type": "ProviderError", "message": message}]}
        assert_public_only_payload(summary, config=config, context="provider_error_selftest")
        text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        _assert("DEEPSEEK_API_KEY" not in text, "safe provider error must not expose API key env names")
        _assert("api_key" not in text.lower(), "safe provider error must not expose provider credential field names")
        _assert("secret" not in text.lower(), "safe provider error must not expose secret markers")


def _source_pack_item(logical_id: str, content: str) -> dict[str, Any]:
    encoded = content.encode("utf-8")
    return {
        "id": logical_id,
        "kind": "selftest",
        "format": "json" if logical_id.endswith(".json") else "text",
        "source": "selftest",
        "content": content,
        "bytes": len(encoded),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "public_safe": True,
    }


def _source_pack(items: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "humanoid_agent.phase8e.source_pack.v1",
        "policy": {"ids_are_closed_allowlist": True},
        "items": items,
    }


def test_source_requests_resolve_only_allowlisted_logical_ids_and_write_artifact() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-ok-") as tmp:
        root = Path(tmp)
        logical_id = "contract/reward_inputs_contract.json"
        source_pack = _source_pack({logical_id: _source_pack_item(logical_id, '{"field_order": ["base_height"]}\n')})
        artifact = root / "source_payloads.json"
        summary = resolve_source_requests(
            source_requests=[{"id": logical_id, "reason": "confirm shape"}],
            source_pack=source_pack,
            config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            artifact_path=artifact,
            max_requests=2,
        )
        _assert(summary["status"] == "pass", f"source request should pass: {summary}")
        _assert(summary["requested_count"] == 1 and summary["returned_count"] == 1, f"unexpected source request counts: {summary}")
        _assert(summary["payloads"][0]["id"] == logical_id, f"wrong payload id: {summary}")
        _assert(summary["payloads"][0]["content"].startswith('{"field_order"'), "source content should be returned verbatim")
        artifact_data = _load_json(artifact)
        _assert(artifact_data == summary, "source payload artifact must match returned summary")


def test_source_requests_resolve_manifest_item_from_backing_source_store() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-store-") as tmp:
        root = Path(tmp)
        logical_id = "examples/known_good_reward_terms.py"
        content = "from humanoid_agent.code_sandbox.components import RewardTerm\n"
        encoded = content.encode("utf-8")
        source_pack = _source_pack(
            {
                logical_id: {
                    "id": logical_id,
                    "kind": "selftest",
                    "format": "text",
                    "source": "selftest",
                    "bytes": len(encoded),
                    "content_sha256": hashlib.sha256(encoded).hexdigest(),
                    "public_safe": True,
                    "content_available": True,
                }
            }
        )
        source_store = _source_pack({logical_id: _source_pack_item(logical_id, content)})
        summary = resolve_source_requests(
            source_requests=[{"id": logical_id, "reason": "need example"}],
            source_pack=source_pack,
            source_store=source_store,
            config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            artifact_path=root / "source_payloads.json",
            max_requests=2,
        )
        _assert(summary["status"] == "pass", f"manifest-backed source request should pass: {summary}")
        _assert(summary["payloads"][0]["content"] == content, "source request must return backing store content")


def test_source_requests_reject_manifest_only_item_without_backing_source_store() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-store-missing-") as tmp:
        root = Path(tmp)
        logical_id = "examples/known_good_reward_terms.py"
        content = "from humanoid_agent.code_sandbox.components import RewardTerm\n"
        encoded = content.encode("utf-8")
        source_pack = _source_pack(
            {
                logical_id: {
                    "id": logical_id,
                    "kind": "selftest",
                    "format": "text",
                    "source": "selftest",
                    "bytes": len(encoded),
                    "content_sha256": hashlib.sha256(encoded).hexdigest(),
                    "public_safe": True,
                    "content_available": True,
                }
            }
        )
        try:
            resolve_source_requests(
                source_requests=[{"id": logical_id, "reason": "need example"}],
                source_pack=source_pack,
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                artifact_path=root / "source_payloads.json",
                max_requests=2,
            )
        except Phase8EProviderError as exc:
            _assert(
                "backing source_store" in str(exc),
                f"manifest-only source pack item should require explicit backing store content: {exc}",
            )
        else:
            raise AssertionError("manifest-only source pack item was accepted without backing source_store")


def test_source_requests_reject_untrusted_ids_before_lookup() -> None:
    source_pack = _source_pack(
        {
            "contract/reward_inputs_contract.json": _source_pack_item(
                "contract/reward_inputs_contract.json", '{"field_order": []}\n'
            )
        }
    )
    bad_ids = [
        "../contract/reward_inputs_contract.json",
        "/contract/reward_inputs_contract.json",
        "contract/../reward_inputs_contract.json",
        "src/humanoid_agent/cold_start_lab/training_adapter.py",
    ]
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-badid-") as tmp:
        root = Path(tmp)
        for index, logical_id in enumerate(bad_ids, start=1):
            try:
                resolve_source_requests(
                    source_requests=[{"id": logical_id, "reason": "bad"}],
                    source_pack=source_pack,
                    config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                    artifact_path=root / f"source_payloads_{index}.json",
                    max_requests=2,
                )
            except Phase8EProviderError as exc:
                _assert(
                    "source request" in str(exc) and ("not allowlisted" in str(exc) or "unsafe" in str(exc)),
                    f"bad source id should be rejected explicitly: {logical_id}: {exc}",
                )
            else:
                raise AssertionError(f"bad source id was accepted: {logical_id}")


def test_source_requests_reaudit_content_before_returning() -> None:
    logical_id = "public_failures/recent_training_errors.json"
    source_pack = _source_pack({logical_id: _source_pack_item(logical_id, '{"note": "success_metric implementation"}\n')})
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-audit-") as tmp:
        try:
            resolve_source_requests(
                source_requests=[{"id": logical_id, "reason": "bad content"}],
                source_pack=source_pack,
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                artifact_path=Path(tmp) / "source_payloads.json",
                max_requests=2,
            )
        except Phase8EProviderError as exc:
            _assert("source request content failed audit" in str(exc), f"content audit rejection should be explicit: {exc}")
        else:
            raise AssertionError("source request returned unaudited forbidden content")


def test_source_requests_enforce_size_limit_before_returning() -> None:
    logical_id = "examples/known_good_reward_terms.py"
    source_pack = _source_pack({logical_id: _source_pack_item(logical_id, "x" * (SOURCE_PACK_MAX_TEXT_BYTES + 1))})
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-size-") as tmp:
        try:
            resolve_source_requests(
                source_requests=[{"id": logical_id, "reason": "too large"}],
                source_pack=source_pack,
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                artifact_path=Path(tmp) / "source_payloads.json",
                max_requests=2,
            )
        except Phase8EProviderError as exc:
            _assert("source request content exceeds size limit" in str(exc), f"size rejection should be explicit: {exc}")
        else:
            raise AssertionError("oversized source request content was returned")


def test_stage_candidate_package_rejects_path_escape() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-path-") as tmp:
        response = json.dumps({"files": {"../escape.py": "print('bad')\n"}}, ensure_ascii=False)
        try:
            stage_candidate_package_from_response(
                response_content=response,
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            _assert("unsafe candidate file path" in str(exc), "path escape should be rejected with explicit reason")
        else:
            raise AssertionError("path escape was accepted")


def test_stage_candidate_package_rejects_missing_manifest() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-manifest-") as tmp:
        response = json.dumps({"files": {"training_code/components.py": "x = 1\n"}}, ensure_ascii=False)
        try:
            stage_candidate_package_from_response(
                response_content=response,
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            _assert("candidate_manifest.yaml" in str(exc), "missing manifest should be explicit")
        else:
            raise AssertionError("missing manifest was accepted")


def test_stage_candidate_package_rejects_invalid_manifest_as_provider_error() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-invalid-manifest-") as tmp:
        response = _package_response_missing_manifest_schema(candidate_id="phase8e_manifest_missing_schema")
        try:
            stage_candidate_package_from_response(
                response_content=json.dumps(response, ensure_ascii=False),
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            message = str(exc)
            _assert(
                "candidate_manifest.yaml" in message and "schema_version" in message,
                f"invalid manifest should become a repairable provider error: {message}",
            )
        else:
            raise AssertionError("invalid manifest was accepted")


def test_stage_candidate_package_rejects_bad_optional_json_as_provider_error() -> None:
    for filename in ("operation_diff.json", "diagnostics.json"):
        with tempfile.TemporaryDirectory(prefix=f"phase8e-provider-bad-{filename.split('.')[0]}-") as tmp:
            response = json.loads(_package_response(candidate_id=f"phase8e_bad_{filename.split('.')[0]}"))
            response["candidate_package"]["files"][filename] = "{not valid json"
            try:
                stage_candidate_package_from_response(
                    response_content=json.dumps(response, ensure_ascii=False),
                    candidate_dir=Path(tmp) / "candidate",
                    config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                )
            except Phase8EProviderError as exc:
                message = str(exc)
                _assert(
                    filename in message and "JSON" in message,
                    f"bad optional JSON should identify the repairable provider file: {message}",
                )
            else:
                raise AssertionError(f"bad {filename} was accepted")


def test_stage_candidate_package_rejects_missing_curriculum_design() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-missing-curriculum-") as tmp:
        operation_diff = {
            "schema_version": "humanoid_agent.phase8e.operation_diff.v1",
            "operations": [],
            "summary": "No curriculum design.",
        }
        try:
            stage_candidate_package_from_response(
                response_content=_package_response(
                    candidate_id="phase8e_missing_curriculum_design",
                    operation_diff=operation_diff,
                ),
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            _assert("curriculum_design" in str(exc), f"missing curriculum_design should be explicit: {exc}")
        else:
            raise AssertionError("missing curriculum_design was accepted")


def test_stage_candidate_package_rejects_single_stage_curriculum_design() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-single-stage-curriculum-") as tmp:
        operation_diff = _valid_curriculum_operation_diff()
        operation_diff["curriculum_design"]["stage_list"] = operation_diff["curriculum_design"]["stage_list"][:1]
        try:
            stage_candidate_package_from_response(
                response_content=_package_response(
                    candidate_id="phase8e_single_stage_curriculum",
                    operation_diff=operation_diff,
                ),
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            _assert("stage_list" in str(exc), f"single-stage curriculum should be explicit: {exc}")
        else:
            raise AssertionError("single-stage curriculum_design was accepted")


def test_stage_candidate_package_rejects_reward_only_curriculum_attribution() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-reward-only-curriculum-") as tmp:
        operation_diff = _valid_curriculum_operation_diff()
        operation_diff["curriculum_design"][
            "improvement_attribution_intent"
        ] = "Any improvement is attributed to reward terms, not to any curriculum change."
        try:
            stage_candidate_package_from_response(
                response_content=_package_response(
                    candidate_id="phase8e_reward_only_curriculum_attribution",
                    operation_diff=operation_diff,
                ),
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            _assert("curriculum attribution" in str(exc), f"reward-only attribution should be explicit: {exc}")
        else:
            raise AssertionError("reward-only curriculum attribution was accepted")


def _assert_stage_capabilities_rejected(operation_diff: dict[str, Any], *, candidate_id: str) -> None:
    with tempfile.TemporaryDirectory(prefix=f"phase8e-provider-{candidate_id}-") as tmp:
        try:
            stage_candidate_package_from_response(
                response_content=_package_response(
                    candidate_id=candidate_id,
                    operation_diff=operation_diff,
                ),
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
            )
        except Phase8EProviderError as exc:
            message = str(exc)
            _assert(
                "capabilities" in message and ("unregistered" in message or "list[str]" in message),
                f"invalid capabilities should be explicit: {message}",
            )
        else:
            raise AssertionError(f"{candidate_id} invalid curriculum capabilities were accepted")


def test_stage_candidate_package_rejects_unregistered_capability_mixed_with_valid_id() -> None:
    operation_diff = _valid_curriculum_operation_diff()
    operation_diff["curriculum_design"]["stage_list"][0]["capabilities"] = [
        "external_push_curriculum",
        "invented_eval_seed_control",
    ]
    _assert_stage_capabilities_rejected(
        operation_diff,
        candidate_id="phase8e_invented_plus_valid_capability",
    )


def test_stage_candidate_package_rejects_capability_ids_that_only_appear_in_rationale() -> None:
    operation_diff = _valid_curriculum_operation_diff()
    for stage in operation_diff["curriculum_design"]["stage_list"]:
        stage.pop("external_push_curriculum", None)
        stage.pop("train_only_dr_scale", None)
        stage.pop("reward_weight_scale", None)
        stage["capabilities"] = ["invented_eval_seed_control"]
        stage["rationale"] = "Mentions external_push_curriculum in prose, but not as a capability list value."
    _assert_stage_capabilities_rejected(
        operation_diff,
        candidate_id="phase8e_valid_capability_only_in_rationale",
    )


def test_stage_candidate_package_rejects_unknown_only_capability_list() -> None:
    operation_diff = _valid_curriculum_operation_diff()
    for stage in operation_diff["curriculum_design"]["stage_list"]:
        stage.pop("external_push_curriculum", None)
        stage.pop("train_only_dr_scale", None)
        stage.pop("reward_weight_scale", None)
        stage["capabilities"] = ["invented_eval_seed_control"]
        stage["rationale"] = "No approved capability list value is present."
    _assert_stage_capabilities_rejected(
        operation_diff,
        candidate_id="phase8e_unknown_only_capability",
    )


def test_stage_candidate_package_rejects_non_string_capability_values() -> None:
    operation_diff = _valid_curriculum_operation_diff()
    operation_diff["curriculum_design"]["stage_list"][0]["capabilities"] = ["external_push_curriculum", 123]
    _assert_stage_capabilities_rejected(
        operation_diff,
        candidate_id="phase8e_non_string_capability",
    )


def test_stage_candidate_package_accepts_final_consolidation_aliases() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-final-alias-") as tmp:
        operation_diff = _valid_curriculum_operation_diff()
        operation_diff["curriculum_design"]["final_consolidation"] = {
            "min_fraction": 0.2,
            "distribution": "runner_owned_target_distribution",
            "target_push_included": True,
            "target_robustness_included": True,
        }
        summary = stage_candidate_package_from_response(
            response_content=_package_response(
                candidate_id="phase8e_final_consolidation_aliases",
                operation_diff=operation_diff,
            ),
            candidate_dir=Path(tmp) / "candidate",
            config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
        )
        _assert(summary["status"] == "pass", f"final consolidation aliases should be accepted: {summary}")


def test_stage_candidate_package_does_not_wrap_internal_projection_errors() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-internal-projection-error-") as tmp:
        old_project = phase8e_provider_module.project_candidate_package

        def broken_project_candidate_package(candidate_dir: Path) -> dict[str, Any]:
            _ = candidate_dir
            raise RuntimeError("synthetic internal projection bug")

        phase8e_provider_module.project_candidate_package = broken_project_candidate_package
        try:
            try:
                stage_candidate_package_from_response(
                    response_content=_package_response(candidate_id="phase8e_internal_projection_error"),
                    candidate_dir=Path(tmp) / "candidate",
                    config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                )
            except RuntimeError as exc:
                _assert(
                    "synthetic internal projection bug" in str(exc),
                    f"internal projection errors must bubble unchanged: {exc}",
                )
            except Phase8EProviderError as exc:
                raise AssertionError(
                    f"internal projection bug was incorrectly turned into provider repair feedback: {exc}"
                ) from exc
            else:
                raise AssertionError("internal projection error did not raise")
        finally:
            phase8e_provider_module.project_candidate_package = old_project


def test_stage_candidate_package_rejects_sensitive_value() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-secret-") as tmp:
        leaked = "sk-phase8e-provider-selftest-secret"
        response = _package_response().replace("Two-stage V1 curriculum design carried in operation_diff.", leaked)
        try:
            stage_candidate_package_from_response(
                response_content=response,
                candidate_dir=Path(tmp) / "candidate",
                config={"public_safety": {"forbidden_string_tokens": [], "forbidden_numeric_tokens": []}},
                sensitive_values=[leaked],
            )
        except Phase8EProviderError as exc:
            _assert(leaked not in str(exc), "error message must not echo leaked secret")
            _assert("sensitive value" in str(exc), "secret rejection should be explicit")
        else:
            raise AssertionError("provider response with sensitive value was accepted")


def test_stage_candidate_package_rejects_public_only_token_without_echo() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-public-only-") as tmp:
        candidate_dir = Path(tmp) / "candidate"
        response = _package_response().replace("Two-stage V1 curriculum design carried in operation_diff.", "iid_hidden")
        try:
            stage_candidate_package_from_response(
                response_content=response,
                candidate_dir=candidate_dir,
                config={
                    "public_safety": {
                        "forbidden_string_tokens": ["iid_hidden"],
                        "forbidden_numeric_tokens": [],
                    }
                },
            )
        except Phase8EProviderError as exc:
            _assert("public-only" in str(exc), "public-only rejection should be explicit")
            _assert("iid_hidden" not in str(exc), "public-only rejection must not echo forbidden marker")
            _assert(
                not candidate_dir.exists() or not any(candidate_dir.rglob("*")),
                "failed package staging must not leave partial candidate files",
            )
        else:
            raise AssertionError("provider response with forbidden public-only token was accepted")


def test_stage_candidate_package_accepts_top_level_file_map() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-file-map-") as tmp:
        root = Path(tmp)
        wrapped = json.loads(_package_response(candidate_id="phase8e_top_level_file_map"))
        response = json.dumps(wrapped["candidate_package"]["files"], ensure_ascii=False)
        summary = stage_candidate_package_from_response(
            response_content=response,
            candidate_dir=root / "candidate",
            config=load_mapping(PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml"),
        )
        _assert(summary["status"] == "pass", f"top-level file map should stage: {summary}")
        manifest = _load_json(root / "candidate" / "candidate_manifest.yaml")
        _assert(manifest["candidate_id"] == "phase8e_top_level_file_map", "top-level file map manifest mismatch")


def test_sensitive_scan_ignores_noncredential_provider_env_values() -> None:
    import os

    old_key = os.environ.get("PHASE8E_SELFTEST_KEY")
    old_model = os.environ.get("PHASE8E_SELFTEST_MODEL")
    try:
        os.environ["PHASE8E_SELFTEST_KEY"] = "sk-phase8e-provider-selftest-key"
        os.environ["PHASE8E_SELFTEST_MODEL"] = "glm-selftest-model"
        values = _sensitive_values(
            {
                "provider": {
                    "glm": {
                        "type": "openai_compatible",
                        "api_key_env": "PHASE8E_SELFTEST_KEY",
                        "model_env": "PHASE8E_SELFTEST_MODEL",
                    }
                }
            },
            "glm",
        )
        _assert("sk-phase8e-provider-selftest-key" in values, "credential env value should be scanned")
        _assert("glm-selftest-model" not in values, "model env value should not be treated as a secret")
    finally:
        if old_key is None:
            os.environ.pop("PHASE8E_SELFTEST_KEY", None)
        else:
            os.environ["PHASE8E_SELFTEST_KEY"] = old_key
        if old_model is None:
            os.environ.pop("PHASE8E_SELFTEST_MODEL", None)
        else:
            os.environ["PHASE8E_SELFTEST_MODEL"] = old_model


def test_provider_followup_prompts_use_explicit_instruction_first_order() -> None:
    base_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "{}"},
    ]
    source_messages = _messages_with_source_payloads(
        base_messages,
        '{"source_requests": [{"id": "contract/reward_inputs_contract.json"}]}',
        {"payloads": [{"id": "contract/reward_inputs_contract.json", "content": "{}"}]},
        remaining_source_read_turns=1,
    )
    _assert_content_instruction_before_payload(
        source_messages[-1]["content"],
        '"source_payloads"',
        "source-request follow-up",
    )
    exhausted_source_messages = _messages_with_source_payloads(
        base_messages,
        '{"source_requests": [{"id": "contract/reward_inputs_contract.json"}]}',
        {"payloads": [{"id": "contract/reward_inputs_contract.json", "content": "{}"}]},
        remaining_source_read_turns=0,
    )
    _assert(
        "source-read budget is exhausted" in exhausted_source_messages[-1]["content"]
        and "candidate_package JSON" in exhausted_source_messages[-1]["content"],
        f"exhausted source-read follow-up must require candidate package: {exhausted_source_messages[-1]['content']}",
    )

    try:
        json.loads("{")
    except json.JSONDecodeError as exc:
        json_repair_messages = _messages_with_json_repair_request(base_messages, "{", exc)
    else:
        raise AssertionError("malformed JSON probe should fail")
    _assert_content_instruction_before_payload(
        json_repair_messages[-1]["content"],
        '"json_repair_request"',
        "JSON repair follow-up",
    )

    repair_messages = _messages_with_repair_feedback(
        base_messages,
        {"schema_version": "humanoid_agent.phase8e.repair_feedback.v1", "status": "repair_required"},
    )
    _assert_content_instruction_before_payload(
        repair_messages[-1]["content"],
        '"repair_feedback"',
        "candidate repair follow-up",
    )


def test_provider_dry_run_writes_candidate_and_static_verifier_passes() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-dry-run-") as tmp:
        root = Path(tmp)
        summary = run_phase8e_provider_candidate(
            config_path=_local_provider_config(root),
            run_dir=root / "round",
            provider_name="dry_run",
            verify=True,
            require_server_enforcement=False,
        )
        _assert(summary["status"] in {"pass", "fail"}, "summary status must be explicit")
        _assert(summary["candidate"]["manifest"]["llm_authored"] is True, "dry-run candidate must be marked LLM-authored")
        _assert(Path(summary["candidate_dir"]).is_dir(), "candidate dir should exist")
        _assert(summary["provider"]["request_summary"]["network"] is False, "dry-run provider must not use network")
        _assert(summary["verifier"]["checks"]["static"]["status"] == "pass", "generated dry-run package must pass static 8D checks")
        _assert(
            Path(summary["prompt"]["payload_file"]).is_file(),
            f"provider summary must expose the prompt payload artifact for lineage hashing: {summary['prompt']}",
        )
        prompt_payload = _load_json(root / "round" / "phase8e_prompt_payload.json")
        _assert(
            next(iter(prompt_payload)) == "task",
            f"provider prompt payload artifact must preserve task-first order: {list(prompt_payload)[:5]}",
        )
        prompt_top_30 = "\n".join((root / "round" / "phase8e_prompt.txt").read_text(encoding="utf-8").splitlines()[:30])
        _assert(
            "Output exactly one JSON object with candidate_package.files only" in prompt_top_30,
            f"provider prompt top section must state delivery format early: {prompt_top_30}",
        )
        _assert(
            "source_requests only" in prompt_top_30,
            f"provider prompt top section must explicitly allow source_requests before candidate authoring: {prompt_top_30}",
        )
        manifest_contract = prompt_payload["required_candidate_package"].get("candidate_manifest_contract", {})
        _assert(
            manifest_contract
            == candidate_manifest_prompt_contract(
                required_entrypoint="training_code.components:build_training_components",
                phase="8E",
            ),
            f"prompt manifest contract must be derived from the loader contract: {manifest_contract}",
        )
        _assert(
            manifest_contract.get("schema_version") == "humanoid_agent.cold_start.candidate.v1",
            f"prompt must expose the required manifest schema_version explicitly: {manifest_contract}",
        )
        manifest_required_fields = manifest_contract.get("required_fields_in_order", [])
        _assert(
            manifest_required_fields and manifest_required_fields[0] == "schema_version",
            f"manifest schema_version must be first in the required manifest fields: {manifest_contract}",
        )
        manifest_template = manifest_contract.get("template", "")
        _assert(
            "schema_version: humanoid_agent.cold_start.candidate.v1" in manifest_template
            and "entrypoint: training_code.components:build_training_components" in manifest_template,
            f"prompt must include a complete manifest template: {manifest_contract}",
        )
        manifest_file_contract = prompt_payload["required_candidate_package"]["provider_json_output"]["format"][
            "candidate_package"
        ]["files"]["candidate_manifest.yaml"]
        _assert(
            "schema_version: humanoid_agent.cold_start.candidate.v1" in manifest_file_contract,
            f"provider JSON output contract must repeat the concrete manifest skeleton: {manifest_file_contract}",
        )
        text = json.dumps(summary, ensure_ascii=False, sort_keys=True)
        _assert("DEEPSEEK_API_KEY" not in text and "OPENAI_API_KEY" not in text, "summary must not mention old provider secret names")


def test_provider_replay_json_package_is_provider_swappable() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-replay-") as tmp:
        root = Path(tmp)
        replay = root / "replay.json"
        replay.write_text(_package_response(candidate_id="phase8e_replay_noop"), encoding="utf-8")
        summary = run_phase8e_provider_candidate(
            config_path=PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml",
            run_dir=root / "round",
            provider_name="replay",
            replay_response=replay,
            verify=False,
        )
        manifest = _load_json(Path(summary["candidate_dir"]) / "candidate_manifest.yaml")
        _assert(manifest["candidate_id"] == "phase8e_replay_noop", "replay provider should land the supplied package")
        _assert(summary["status"] == "staged_unverified", "skip-verify must not report pass")
        _assert(summary["verifier_status"] == "skipped", "verify=False should skip verifier only for provider ingestion tests")


def test_provider_inner_loop_handles_source_request_before_candidate() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-loop-") as tmp:
        root = Path(tmp)
        replay = root / "replay_sequence.json"
        _write_json(
            replay,
            {
                "responses": [
                    {
                        "source_requests": [
                            {
                                "id": "contract/reward_inputs_contract.json",
                                "reason": "confirm available RewardInputs fields",
                            }
                        ]
                    },
                    json.loads(_package_response(candidate_id="phase8e_source_loop_noop")),
                ]
            },
        )
        summary = run_phase8e_provider_candidate(
            config_path=_local_provider_config(root),
            run_dir=root / "round",
            provider_name="replay_sequence",
            replay_response=replay,
            verify=False,
        )
        _assert(summary["status"] == "staged_unverified", f"inner loop replay should stage candidate: {summary}")
        _assert(summary["candidate"]["manifest"]["candidate_id"] == "phase8e_source_loop_noop", "selected attempt candidate id mismatch")
        inner = summary.get("inner_loop", {})
        _assert(inner.get("enabled") is True and inner.get("selected_attempt") == 1, f"inner loop summary missing selected attempt: {inner}")
        attempt_dir = root / "round" / "attempts" / "attempt_001"
        _assert((attempt_dir / "prompt_messages.json").is_file(), "attempt prompt artifact missing")
        _assert_instruction_before_payload(
            attempt_dir / "prompt_messages.json",
            '"source_payloads"',
            "source-request follow-up",
        )
        _assert((attempt_dir / "source_requests.json").is_file(), "source request artifact missing")
        source_payloads = _load_json(attempt_dir / "source_payloads.json")
        _assert(source_payloads["returned_count"] == 1, f"source payload artifact mismatch: {source_payloads}")
        _assert((attempt_dir / "raw_response.txt").is_file(), "attempt raw candidate response artifact missing")
        _assert((attempt_dir / "staged_candidate" / "candidate_manifest.yaml").is_file(), "attempt staged candidate missing")
        selected = _load_json(root / "round" / "selected_attempt.json")
        _assert(selected["attempt"] == 1 and selected["candidate_dir"] == summary["candidate_dir"], f"selected attempt artifact mismatch: {selected}")


def test_provider_inner_loop_rejects_sensitive_source_request_reason_before_artifacts() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-source-key-") as tmp:
        root = Path(tmp)
        secret_env = "PHASE8E_PROVIDER_SELFTEST_API_KEY"
        secret_value = "sk-phase8eAuditValue12345"
        old_secret = os.environ.get(secret_env)
        os.environ[secret_env] = secret_value
        try:
            config_path = _local_provider_config(root)
            config = _load_json(config_path)
            config["provider"]["replay_sequence"]["api_key_env"] = secret_env
            _write_json(config_path, config)
            replay = root / "replay_source_key_sequence.json"
            _write_json(
                replay,
                {
                    "responses": [
                        {
                            "source_requests": [
                                {
                                    "id": "contract/reward_inputs_contract.json",
                                    "reason": f"need contract {secret_value}",
                                }
                            ]
                        },
                        json.loads(_package_response(candidate_id="phase8e_source_key_should_not_stage")),
                    ]
                },
            )
            summary = run_phase8e_provider_candidate(
                config_path=config_path,
                run_dir=root / "round",
                provider_name="replay_sequence",
                replay_response=replay,
                verify=False,
            )
        finally:
            if old_secret is None:
                os.environ.pop(secret_env, None)
            else:
                os.environ[secret_env] = old_secret
        _assert(summary["status"] == "fail", f"sensitive source request must fail before staging: {summary}")
        text_hits = []
        for path in (root / "round").rglob("*"):
            if path.is_file() and secret_value in path.read_text(encoding="utf-8", errors="ignore"):
                text_hits.append(str(path.relative_to(root / "round")))
        _assert(not text_hits, f"sensitive source request value leaked into artifacts: {text_hits}")


def test_provider_inner_loop_repairs_invalid_json_before_staging() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-json-repair-") as tmp:
        root = Path(tmp)
        malformed = _package_response(candidate_id="phase8e_json_bad")[:-1]
        replay = root / "replay_json_repair_sequence.json"
        _write_json(
            replay,
            {
                "responses": [
                    malformed,
                    json.loads(_package_response(candidate_id="phase8e_json_repaired")),
                ]
            },
        )
        config_path = _local_provider_config(root)
        config = _load_json(config_path)
        config["provider_inner_loop"]["json_repair_attempts"] = 1
        _write_json(config_path, config)
        summary = run_phase8e_provider_candidate(
            config_path=config_path,
            run_dir=root / "round",
            provider_name="replay_sequence",
            replay_response=replay,
            verify=True,
        )
        _assert(summary["status"] == "pass", f"JSON repair loop should select repaired candidate: {summary}")
        inner = summary["inner_loop"]
        _assert(inner["selected_attempt"] == 1 and inner["provider_call_count"] == 2, f"JSON repair should stay in attempt 1: {inner}")
        attempt_dir = root / "round" / "attempts" / "attempt_001"
        json_repair = _load_json(attempt_dir / "json_repair_summary.json")
        _assert(json_repair["status"] == "pass", f"JSON repair artifact should pass: {json_repair}")
        _assert((attempt_dir / "raw_response.txt").read_text(encoding="utf-8") == malformed, "raw response should preserve malformed text")
        _assert((attempt_dir / "json_repaired_response_001.txt").is_file(), "repaired JSON response artifact missing")
        _assert(summary["candidate"]["manifest"]["candidate_id"] == "phase8e_json_repaired", "selected candidate should use repaired JSON")


def test_provider_inner_loop_repairs_static_failure_before_selecting() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-repair-loop-") as tmp:
        root = Path(tmp)
        bad = _package_response_with_forbidden_import(candidate_id="phase8e_static_bad")
        replay = root / "replay_repair_sequence.json"
        _write_json(
            replay,
            {
                "responses": [
                    bad,
                    json.loads(_package_response(candidate_id="phase8e_static_repaired")),
                ]
            },
        )
        summary = run_phase8e_provider_candidate(
            config_path=_local_provider_config(root),
            run_dir=root / "round",
            provider_name="replay_sequence",
            replay_response=replay,
            verify=True,
        )
        _assert(summary["status"] == "pass", f"repair loop should select repaired candidate: {summary}")
        _assert(summary["inner_loop"]["selected_attempt"] == 2, f"second attempt should be selected: {summary['inner_loop']}")
        repair_feedback = _load_json(root / "round" / "attempts" / "attempt_001" / "repair_feedback.json")
        issue_codes = {issue.get("code") for issue in repair_feedback.get("issues", [])}
        _assert("forbidden_import" in issue_codes, f"repair feedback must include exact static issue: {repair_feedback}")
        _assert(
            "training_code/components.py" in repair_feedback.get("candidate_files", {}),
            f"repair feedback must include previous candidate source files: {repair_feedback}",
        )
        _assert(summary["candidate"]["manifest"]["candidate_id"] == "phase8e_static_repaired", "selected candidate should be repaired response")


def test_provider_inner_loop_repairs_invalid_manifest_before_selecting() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-manifest-repair-loop-") as tmp:
        root = Path(tmp)
        bad = _package_response_missing_manifest_schema(candidate_id="phase8e_manifest_bad")
        replay = root / "replay_manifest_repair_sequence.json"
        _write_json(
            replay,
            {
                "responses": [
                    bad,
                    json.loads(_package_response(candidate_id="phase8e_manifest_repaired")),
                ]
            },
        )
        summary = run_phase8e_provider_candidate(
            config_path=_local_provider_config(root),
            run_dir=root / "round",
            provider_name="replay_sequence",
            replay_response=replay,
            verify=True,
        )
        _assert(summary["status"] == "pass", f"manifest repair loop should select repaired candidate: {summary}")
        _assert(summary["inner_loop"]["selected_attempt"] == 2, f"second attempt should be selected: {summary['inner_loop']}")
        repair_feedback = _load_json(root / "round" / "attempts" / "attempt_001" / "repair_feedback.json")
        repair_text = json.dumps(repair_feedback, ensure_ascii=False, sort_keys=True)
        _assert("candidate_manifest.yaml" in repair_text and "schema_version" in repair_text, f"manifest repair feedback should identify the missing schema_version: {repair_feedback}")
        _assert(summary["candidate"]["manifest"]["candidate_id"] == "phase8e_manifest_repaired", "selected candidate should be repaired response")


def test_provider_inner_loop_repairs_bad_optional_json_before_selecting() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-optional-json-repair-loop-") as tmp:
        root = Path(tmp)
        bad = json.loads(_package_response(candidate_id="phase8e_optional_json_bad"))
        bad["candidate_package"]["files"]["operation_diff.json"] = "{not valid json"
        replay = root / "replay_optional_json_repair_sequence.json"
        _write_json(
            replay,
            {
                "responses": [
                    bad,
                    json.loads(_package_response(candidate_id="phase8e_optional_json_repaired")),
                ]
            },
        )
        summary = run_phase8e_provider_candidate(
            config_path=_local_provider_config(root),
            run_dir=root / "round",
            provider_name="replay_sequence",
            replay_response=replay,
            verify=True,
        )
        _assert(summary["status"] == "pass", f"optional JSON repair loop should select repaired candidate: {summary}")
        _assert(summary["inner_loop"]["selected_attempt"] == 2, f"second attempt should be selected: {summary['inner_loop']}")
        repair_feedback = _load_json(root / "round" / "attempts" / "attempt_001" / "repair_feedback.json")
        repair_text = json.dumps(repair_feedback, ensure_ascii=False, sort_keys=True)
        _assert("operation_diff.json" in repair_text and "JSON" in repair_text, f"repair feedback must identify bad optional JSON: {repair_feedback}")
        _assert(
            summary["candidate"]["manifest"]["candidate_id"] == "phase8e_optional_json_repaired",
            "selected candidate should be repaired response",
        )


def test_repair_feedback_is_prompt_facing_projection() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-repair-feedback-projection-") as tmp:
        root = Path(tmp)
        config = load_mapping(PROJECT_ROOT / "configs" / "cold_start" / "phase8e_loop.yaml")
        staged_candidate = root / "staged_candidate"
        stage_summary = stage_candidate_package_from_response(
            response_content=_package_response(candidate_id="phase8e_repair_projection"),
            candidate_dir=staged_candidate,
            config=config,
        )
        verifier_summary = {
            "status": "fail",
            "fingerprint": {
                "manifest_sha256": "a" * 64,
                "entrypoint_sha256": "b" * 64,
                "manifest_path": str(staged_candidate / "candidate_manifest.yaml"),
                "entrypoint_path": str(staged_candidate / "training_code" / "components.py"),
            },
            "issues": [
                {
                    "code": "isolated_dynamic_fast_validation_failed",
                    "detail": {
                        "component_counts": {
                            "reward_terms": 1,
                            "loss_terms": 0,
                            "train_observation_transforms": 0,
                        },
                        "issues": [
                            {
                                "code": "diagnostic_term_invalid",
                                "component_name": "standing_diagnostics",
                                "kind": "diagnostic",
                                "error_type": "TypeError",
                                "path": r"D:\humanoid_agent\configs\cold_start\phase8d_verifier.yaml",
                                "message": (
                                    "'float' object is not iterable at "
                                    r"C:\Users\lover\project_dir\candidate\training_code\components.py:12 "
                                    r"via \\humanoid-server\share\candidate\training_code\components.py"
                                ),
                            }
                        ],
                        "schedules": [
                            {
                                "name": "constant_reward_scale",
                                "table": {
                                    "runner_allowed_keys": [
                                        "push_magnitude_scale",
                                        "push_start_fraction",
                                        "push_interval_s",
                                        "domain_randomization_scale",
                                        "reward_weight_scale",
                                    ]
                                },
                            }
                        ],
                        "synthetic_inputs": {
                            "field_order": ["base_height", "push_active", "push_vec_public", "progress_fraction"]
                        },
                    },
                }
            ],
        }
        feedback = _build_repair_feedback(
            attempt=1,
            prompt_package={"payload": {"candidate_self_audit_checklist": {"items": []}}},
            staged_candidate=staged_candidate,
            stage_summary=stage_summary,
            verifier_summary=verifier_summary,
            verifier_status="fail",
            config=config,
            sensitive_values=[],
        )
        messages = _messages_with_repair_feedback([{"role": "system", "content": "s"}], feedback)
        text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        for forbidden in (
            "push_active",
            "push_vec_public",
            "loss_terms",
            "train_observation_transforms",
            "push_magnitude_scale",
            "push_start_fraction",
            "push_interval_s",
            "domain_randomization_scale",
            "manifest_path",
            "entrypoint_path",
            str(staged_candidate),
            "/workspace/phase8e/run/candidate/training_code/components.py",
            r"D:\humanoid_agent\configs\cold_start\phase8d_verifier.yaml",
            r"C:\Users\lover\project_dir\candidate\training_code\components.py",
            r"\\humanoid-server\share\candidate\training_code\components.py",
        ):
            _assert(forbidden not in text, f"repair feedback leaked non-prompt-facing detail: {forbidden}")
        _assert("[omitted: server-local path]" in text, f"repair feedback should preserve a local path placeholder: {text}")
        _assert("diagnostic_term_invalid" in text, f"repair feedback must retain actionable verifier issue code: {text}")
        _assert("standing_diagnostics" in text, f"repair feedback must retain failing component name: {text}")


def test_provider_inner_loop_rejects_when_pretraining_attempts_exhausted() -> None:
    with tempfile.TemporaryDirectory(prefix="phase8e-provider-reject-loop-") as tmp:
        root = Path(tmp)
        replay = root / "replay_reject_sequence.json"
        _write_json(
            replay,
            {
                "responses": [
                    _package_response_with_forbidden_import(candidate_id="phase8e_static_bad_1"),
                    _package_response_with_forbidden_import(candidate_id="phase8e_static_bad_2"),
                ]
            },
        )
        summary = run_phase8e_provider_candidate(
            config_path=_local_provider_config(root),
            run_dir=root / "round",
            provider_name="replay_sequence",
            replay_response=replay,
            verify=True,
        )
        _assert(summary["status"] == "reject", f"exhausted pretraining loop should reject: {summary}")
        inner = summary["inner_loop"]
        _assert(inner["enabled"] is True and inner["status"] == "rejected", f"inner loop rejection should be explicit: {inner}")
        _assert(inner["selected_attempt"] is None and len(inner["attempts"]) == 2, f"inner loop should exhaust two attempts: {inner}")
        selected = _load_json(root / "round" / "selected_attempt.json")
        _assert(selected["reason"] == "candidate_failed_pretraining_inner_loop", f"selected artifact should record rejection reason: {selected}")
        _assert(not (root / "round" / "candidate_package" / "candidate_manifest.yaml").exists(), "rejected loop must not publish a trainable candidate")


def main() -> int:
    test_source_requests_resolve_only_allowlisted_logical_ids_and_write_artifact()
    test_source_requests_resolve_manifest_item_from_backing_source_store()
    test_source_requests_reject_manifest_only_item_without_backing_source_store()
    test_source_requests_reject_untrusted_ids_before_lookup()
    test_source_requests_reaudit_content_before_returning()
    test_source_requests_enforce_size_limit_before_returning()
    test_stage_candidate_package_rejects_path_escape()
    test_stage_candidate_package_rejects_missing_manifest()
    test_stage_candidate_package_rejects_invalid_manifest_as_provider_error()
    test_stage_candidate_package_rejects_bad_optional_json_as_provider_error()
    test_stage_candidate_package_rejects_missing_curriculum_design()
    test_stage_candidate_package_rejects_single_stage_curriculum_design()
    test_stage_candidate_package_rejects_reward_only_curriculum_attribution()
    test_stage_candidate_package_rejects_unregistered_capability_mixed_with_valid_id()
    test_stage_candidate_package_rejects_capability_ids_that_only_appear_in_rationale()
    test_stage_candidate_package_rejects_unknown_only_capability_list()
    test_stage_candidate_package_rejects_non_string_capability_values()
    test_stage_candidate_package_accepts_final_consolidation_aliases()
    test_stage_candidate_package_does_not_wrap_internal_projection_errors()
    test_stage_candidate_package_rejects_sensitive_value()
    test_stage_candidate_package_rejects_public_only_token_without_echo()
    test_provider_safe_error_message_redacts_public_forbidden_tokens()
    test_stage_candidate_package_accepts_top_level_file_map()
    test_sensitive_scan_ignores_noncredential_provider_env_values()
    test_provider_followup_prompts_use_explicit_instruction_first_order()
    test_provider_dry_run_writes_candidate_and_static_verifier_passes()
    test_provider_replay_json_package_is_provider_swappable()
    test_provider_inner_loop_handles_source_request_before_candidate()
    test_provider_inner_loop_rejects_sensitive_source_request_reason_before_artifacts()
    test_provider_inner_loop_repairs_invalid_json_before_staging()
    test_provider_inner_loop_repairs_static_failure_before_selecting()
    test_provider_inner_loop_repairs_invalid_manifest_before_selecting()
    test_provider_inner_loop_repairs_bad_optional_json_before_selecting()
    test_repair_feedback_is_prompt_facing_projection()
    test_provider_inner_loop_rejects_when_pretraining_attempts_exhausted()
    print(json.dumps({"status": "pass", "test": "phase8e_provider_ingest_selftest"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
