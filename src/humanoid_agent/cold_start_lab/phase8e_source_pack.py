"""Prompt-facing source-pack construction for Phase 8E."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from humanoid_agent.cold_start_lab.phase8e_common import (
    SOURCE_PACK_FORBIDDEN_TOKENS,
    SOURCE_PACK_MAX_TEXT_BYTES,
    Phase8EError,
    assert_prompt_facing_payload,
    assert_public_only_payload,
    contains_inactive_prompt_token,
    scrub_prompt_facing_value,
)
from humanoid_agent.cold_start_lab.phase8e_projection import (
    latest_accepted_record,
    project_ledger_record,
    prompt_history_public_result,
)


ALLOWED_TOP_LEVEL_CANDIDATE_FILES = (
    "candidate_manifest.yaml",
    "training_code/",
    "rationale.md",
    "operation_diff.json",
    "diagnostics.json",
)

KNOWN_GOOD_REWARD_TERMS_SOURCE = '''"""Minimal Phase 8E component example using public RewardInputs only."""

from __future__ import annotations

from humanoid_agent.code_sandbox.components import RewardInputs, RewardTerm, ScheduleTerm, TrainingComponents


def _rows(values):
    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "tolist"):
        return values.tolist()
    return list(values)


def _first_two_rows(values):
    try:
        return _rows(values[:, :2])
    except Exception:  # noqa: BLE001 - list-backed selftests do not support tuple slicing.
        return [list(row)[:2] for row in _rows(values)]


def upright_action_smoothness(inp: RewardInputs):
    heights = _rows(inp.base_height)
    rolls = _rows(inp.roll)
    pitches = _rows(inp.pitch)
    action_delta = _rows(inp.action_delta)
    out = []
    for index, height in enumerate(heights):
        delta_row = action_delta[index] if index < len(action_delta) else []
        smooth = sum(float(value) * float(value) for value in delta_row) / max(1, len(delta_row))
        upright_penalty = float(rolls[index]) * float(rolls[index]) + float(pitches[index]) * float(pitches[index])
        out.append(float(height) - 0.5 * upright_penalty - 0.01 * smooth)
    return out


def xy_velocity_command_tracking(inp: RewardInputs):
    base_xy = _first_two_rows(inp.base_lin_vel)
    command_xy = _first_two_rows(inp.commanded_base_vel)
    out = []
    for base, command in zip(base_xy, command_xy, strict=True):
        dx = float(base[0]) - float(command[0])
        dy = float(base[1]) - float(command[1])
        out.append(-(dx * dx + dy * dy))
    return out


def reward_scale_schedule(progress_fraction: float):
    return {"reward_weight_scale": 1.0 if progress_fraction < 0.5 else 1.2}


def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents:
    _ = (task_spec, base_env_cfg, base_agent_cfg)
    return TrainingComponents(
        name="minimal_public_reward_example",
        reward_terms=[
            RewardTerm(
                name="upright_action_smoothness",
                fn=upright_action_smoothness,
                weight=1.0,
                clip_min=-2.0,
                clip_max=2.0,
                description="upright height reward with small action_delta smoothness penalty",
            ),
            RewardTerm(
                name="xy_velocity_command_tracking",
                fn=xy_velocity_command_tracking,
                weight=0.25,
                clip_min=-2.0,
                clip_max=0.0,
                description="correctly compares base_lin_vel[:, :2] with commanded_base_vel[:, :2]",
            )
        ],
        schedules=[
            ScheduleTerm(
                name="reward_weight_scale_after_balance",
                fn=reward_scale_schedule,
                allowed_keys=("reward_weight_scale",),
            )
        ],
        rationale="Uses only public RewardInputs and the currently effective reward_weight_scale schedule key.",
    )
'''


def known_good_reward_examples_metadata(*, source_read_enabled: bool) -> dict[str, Any]:
    notes = [
        "Examples are runnable component code, not a reward menu.",
        "They demonstrate shape-safe RewardTerm patterns over public RewardInputs.",
    ]
    if source_read_enabled:
        notes.append(
            "If you need the full runnable example body, request examples/known_good_reward_terms.py via source_requests."
        )
    else:
        notes.append(
            "Source-read is disabled in this author call. Use required_candidate_package.component_contract.minimal_component_template as the inline fallback skeleton."
        )
    return {
        "schema_version": "humanoid_agent.phase8e.known_good_reward_examples.v1",
        "source_item_id": "examples/known_good_reward_terms.py",
        "examples": [
            {
                "id": "upright_action_smoothness",
                "source_item_id": "examples/known_good_reward_terms.py",
                "reward_term_name": "upright_action_smoothness",
                "covers_fields": ["base_height", "roll", "pitch", "action_delta"],
                "return_shape": ["N"],
                "validation": "covered_by_phase8e_selftest_evaluate_reward_terms",
            },
            {
                "id": "xy_velocity_command_tracking",
                "source_item_id": "examples/known_good_reward_terms.py",
                "reward_term_name": "xy_velocity_command_tracking",
                "covers_fields": ["base_lin_vel[:, :2]", "commanded_base_vel[:, :2]"],
                "return_shape": ["N"],
                "validation": "covered_by_phase8e_selftest_evaluate_reward_terms",
            },
        ],
        "notes": notes,
    }


def build_source_access_policy(
    source_pack: dict[str, Any],
    *,
    enabled: bool,
    max_source_read_turns: int = 0,
) -> dict[str, Any]:
    items = source_pack.get("items", {})
    allowed_ids = list(items) if isinstance(items, dict) else []
    turn_budget = max(0, int(max_source_read_turns)) if enabled else 0
    return {
        "schema_version": "humanoid_agent.phase8e.source_access_policy.v1",
        "enabled": enabled,
        "scope": "training_side_public_source_pack_only",
        "max_source_read_turns": turn_budget,
        "ids_are_closed_allowlist": True,
        "request_ids_are_untrusted": True,
        "allowed_logical_ids": allowed_ids,
        "max_bytes_per_item": SOURCE_PACK_MAX_TEXT_BYTES,
        "source_resolution": (
            "orchestrator_returns_audited_source_pack_items_by_logical_id"
            if enabled
            else "disabled_in_this_author_call"
        ),
        "provider_filesystem_access": False,
        "content_audited_before_return": True,
    }


def build_minimal_source_pack(
    *,
    config: dict[str, Any],
    ledger: list[dict[str, Any]],
    current_recipe_dir: Path | None,
    reward_contract: dict[str, Any],
    field_expansion_report: dict[str, Any],
    control_contract: dict[str, Any],
    source_read_enabled: bool,
) -> dict[str, Any]:
    prompt_items: dict[str, dict[str, Any]] = {}
    store_items: dict[str, dict[str, Any]] = {}

    def add_json(logical_id: str, parsed: Any, *, kind: str) -> None:
        content = json.dumps(parsed, ensure_ascii=False, indent=2) + "\n"
        store_item = _source_store_item(logical_id, content, kind=kind, config=config)
        store_items[logical_id] = store_item
        prompt_items[logical_id] = _source_manifest_item(logical_id, store_item)

    def add_text(logical_id: str, content: str, *, kind: str, source: str) -> None:
        store_item = _source_store_item(logical_id, content, kind=kind, config=config, source=source)
        store_items[logical_id] = store_item
        prompt_items[logical_id] = _source_manifest_item(logical_id, store_item)

    add_json("contract/reward_inputs_contract.json", reward_contract, kind="generated_contract")
    add_json("contract/reward_inputs_field_expansion_report.json", field_expansion_report, kind="generated_contract")
    add_json("contract/control_effectiveness_contract.json", control_contract, kind="generated_contract")
    add_text(
        "examples/known_good_reward_terms.py",
        KNOWN_GOOD_REWARD_TERMS_SOURCE,
        kind="runner_owned_example",
        source="runner_owned_embedded_example",
    )

    recipe_source = _read_recipe_source(current_recipe_dir, "training_code/components.py") if current_recipe_dir is not None else None
    if recipe_source is not None:
        content, source = recipe_source
        add_text(
            "current_best/training_code/components.py",
            _prompt_facing_component_source(content),
            kind="current_recipe_source",
            source=source,
        )

    accepted = latest_accepted_record(ledger)
    if accepted is not None:
        decision = accepted.get("decision", {}) if isinstance(accepted.get("decision"), dict) else {}
        public_eval = prompt_history_public_result(decision.get("public_result"))
        if public_eval is not None:
            add_json("current_best/public_eval_summary.json", scrub_prompt_facing_value(public_eval), kind="public_eval_summary")

    recent_failures = _recent_public_failure_records(ledger)
    add_json("public_failures/recent_training_errors.json", recent_failures, kind="public_failure_records")

    pack = {
        "schema_version": "humanoid_agent.phase8e.source_pack.v1",
        "policy": {
            "scope": "training_side_public_only",
            "ids_are_closed_allowlist": True,
            "no_full_runner_adapter_by_default": True,
            "source_requests_service": (
                "enabled_closed_allowlist" if source_read_enabled else "disabled_in_this_author_call"
            ),
        },
        "items": prompt_items,
    }
    source_store = {
        "schema_version": "humanoid_agent.phase8e.source_store.v1",
        "policy": {
            "scope": "server_side_backing_store_not_serialized_into_prompt",
            "ids_match_prompt_source_pack": True,
            "content_audited_before_return": True,
        },
        "items": store_items,
    }
    assert_public_only_payload(pack, config=config, context="phase8e_source_pack")
    assert_prompt_facing_payload(pack, context="phase8e_source_pack")
    assert_public_only_payload(source_store, config=config, context="phase8e_source_store")
    assert_prompt_facing_payload(source_store, context="phase8e_source_store")
    return {"prompt_source_pack": pack, "source_store": source_store}


def _recent_public_failure_records(ledger: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for record in reversed(ledger):
        projected = project_ledger_record(record)
        if projected.get("accepted") is True:
            continue
        failures.append(projected)
        if len(failures) >= limit:
            break
    return list(reversed(failures))


def _read_recipe_source(root: Path | None, relative_path: str) -> tuple[str, str] | None:
    if root is None:
        return None
    rel = Path(relative_path)
    if rel.is_absolute() or any(part == ".." for part in rel.parts):
        raise Phase8EError(f"invalid source pack relative path: {relative_path}")
    root_resolved = root.resolve()
    path = (root_resolved / rel).resolve()
    if root_resolved != path and root_resolved not in path.parents:
        raise Phase8EError(f"source pack path escapes recipe root: {relative_path}")
    if path.is_symlink():
        raise Phase8EError(f"source pack refuses symlink: {relative_path}")
    if not path.is_file():
        return None
    if path.stat().st_size > SOURCE_PACK_MAX_TEXT_BYTES:
        raise Phase8EError(f"source pack item too large: {relative_path}")
    return path.read_text(encoding="utf-8"), f"current_recipe/{relative_path}"


def _source_store_item(
    logical_id: str,
    content: str,
    *,
    kind: str,
    config: dict[str, Any],
    source: str | None = None,
) -> dict[str, Any]:
    encoded = content.encode("utf-8")
    if len(encoded) > SOURCE_PACK_MAX_TEXT_BYTES:
        raise Phase8EError(f"source pack item too large: {logical_id}")
    _audit_source_pack_content(logical_id, content, config=config)
    item: dict[str, Any] = {
        "id": logical_id,
        "kind": kind,
        "format": "json" if logical_id.endswith(".json") else "text",
        "source": source or "runner_owned_generated",
        "content": content,
        "bytes": len(encoded),
        "content_sha256": hashlib.sha256(encoded).hexdigest(),
        "public_safe": True,
    }
    return item


def _source_manifest_item(logical_id: str, store_item: dict[str, Any]) -> dict[str, Any]:
    item = {
        "id": logical_id,
        "kind": store_item.get("kind"),
        "format": store_item.get("format"),
        "source": store_item.get("source"),
        "bytes": store_item.get("bytes"),
        "content_sha256": store_item.get("content_sha256"),
        "public_safe": store_item.get("public_safe"),
        "content_available": True,
        "summary": _source_item_summary(logical_id, store_item),
    }
    see = _source_item_see_ref(logical_id)
    if see is not None:
        item["see"] = see
    return item


def _source_item_see_ref(logical_id: str) -> str | None:
    return {
        "contract/reward_inputs_contract.json": (
            "required_candidate_package.component_contract.reward_inputs_contract"
        ),
        "contract/reward_inputs_field_expansion_report.json": (
            "required_candidate_package.component_contract.reward_inputs_field_expansion_report"
        ),
        "contract/control_effectiveness_contract.json": (
            "required_candidate_package.component_contract.control_effectiveness_contract"
        ),
    }.get(logical_id)


def _source_item_summary(logical_id: str, store_item: dict[str, Any]) -> str:
    if logical_id.startswith("contract/"):
        return "Contract body is already present once in required_candidate_package.component_contract."
    if logical_id == "examples/known_good_reward_terms.py":
        return "Runnable public RewardTerm and ScheduleTerm example source available by source-read."
    if logical_id == "current_best/training_code/components.py":
        return "Prompt-facing current-best component source available by source-read."
    if logical_id == "current_best/public_eval_summary.json":
        return "Prompt-safe current-best public evaluation summary available by source-read."
    if logical_id == "public_failures/recent_training_errors.json":
        return "Recent prompt-safe failed-round feedback records available by source-read."
    return f"Audited source item available by source-read ({store_item.get('kind')})."


def _audit_source_pack_content(logical_id: str, content: str, *, config: dict[str, Any]) -> None:
    assert_public_only_payload(content, config=config, context=f"source_pack:{logical_id}")
    assert_prompt_facing_payload(content, context=f"source_pack:{logical_id}")
    lowered = content.lower()
    hits = [token for token in SOURCE_PACK_FORBIDDEN_TOKENS if token.lower() in lowered]
    if hits:
        raise Phase8EError(f"source pack item contains forbidden source tokens in {logical_id}: {hits}")


def _prompt_facing_component_source(content: str) -> str:
    if contains_inactive_prompt_token(content):
        return _prompt_facing_source_omitted()
    return content


def _prompt_facing_source_omitted() -> str:
    return (
        '"""Prompt-facing source view omitted because this source references training surfaces '
        'that are not active in the current runner contract."""\n'
    )
