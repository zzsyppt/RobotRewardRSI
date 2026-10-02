"""Offline regression of the eight P3 final frames that lost task evidence.

Replays the production frame boundary and provider request serializer over saved
sections. The first and last requests must belong to the same author episode.
This does not rerun the historical author, change its scores, or reread raw NPZs.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig,
)
from true_rl_agent.prompting.authoring_frame import (
    build_authoring_context_frame, project_sections_for_authoring_frame,
    with_final_prompt_budget,
)
from true_rl_agent.prompting.payload import PROMPT_PAYLOAD_SECTION_ORDER
from true_rl_agent.prompting.source_catalog import (
    PROMPT_SOURCE_SECTION_KEYS, finalize_prompt_source_sections,
    has_inline_current_authoring_failed_source,
)
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def contains_behavior(value: object) -> bool:
    if isinstance(value, dict):
        behavior = value.get("behavior")
        available = isinstance(behavior, dict) and ("profile" in behavior or "transcripts" in behavior)
        return available or any(contains_behavior(v) for v in value.values())
    if isinstance(value, list):
        return any(contains_behavior(v) for v in value)
    return False


def rebuild(first: dict, last: dict) -> dict:
    if first["task"] != last["task"] or first["candidate_capabilities"] != last["candidate_capabilities"]:
        raise ValueError("episode contracts drifted between its first and final requests")
    if not has_inline_current_authoring_failed_source(last):
        raise ValueError("the omitted-evidence case is not a current failed-source repair")
    # Restore only sections from this episode's first input. The last request
    # remains authoritative for the exact failed source, repair and lineage.
    sections = deepcopy(first)
    sections.update(deepcopy(last))
    sections.pop("authoring_context_frame", None)
    sections, omissions = project_sections_for_authoring_frame(sections,
        frame_kind="static_repair", section_order=PROMPT_PAYLOAD_SECTION_ORDER)
    sections.update(finalize_prompt_source_sections(
        {key: sections[key] for key in PROMPT_SOURCE_SECTION_KEYS}))
    sections["authoring_context_frame"] = build_authoring_context_frame(
        frame_kind="static_repair", section_values=sections,
        section_order=PROMPT_PAYLOAD_SECTION_ORDER, omitted_sections=omissions)
    payload = {key: sections[key] for key in PROMPT_PAYLOAD_SECTION_ORDER if key in sections}
    payload["authoring_context_frame"] = with_final_prompt_budget(
        payload["authoring_context_frame"], prompt_payload=payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "PLAN/feedback_sys/P3/results/author_final_bundle/author_run_r5/authoring")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results/repair_frame_rebuild")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model="offline-serialization-only", output_dir=args.output / "unused-provider-output"))
    rows = []
    for episode in sorted(args.input.glob("*/lane-*")):
        paths = sorted(episode.glob("authoring_attempts/*/prompt_payload.json"))
        if not paths:
            continue
        first, last = read(paths[0]), read(paths[-1])
        if "public_feedback_summary" in last:
            continue
        before = [digest(path) for path in paths]
        payload = rebuild(first, last)
        request = provider._chat_completion_request(payload)
        text = request["messages"][1]["content"]
        delivered = json.loads(text)
        for section in ("public_feedback_summary", "selection_guidance", "exploration_directive"):
            if delivered[section] != first[section]:
                raise ValueError(f"lost original episode evidence/focus: {episode.name}/{section}")
        for section in ("source_payloads", "provider_or_runtime_repair_feedback"):
            if delivered[section] != last[section]:
                raise ValueError(f"changed failed source or exact repair: {episode.name}/{section}")
        budget = delivered["authoring_context_frame"]["budget_metadata"]
        if budget["actual_prompt_payload_bytes"] != len(text.encode("utf-8")):
            raise ValueError("serialized payload byte receipt mismatch")
        validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=delivered)
        if "public_history" in delivered:
            raise ValueError("repair retained optional history instead of the bounded task view")
        if before != [digest(path) for path in paths]:
            raise ValueError("historical source request was modified")
        relative = episode.relative_to(args.input)
        target = args.output / relative / "rebuilt_prompt_payload.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text + "\n", encoding="utf-8")
        rows.append({
            "episode": relative.as_posix(), "task_id": first["task"]["task_id"],
            "arm": "T" if contains_behavior(first["public_feedback_summary"]) else "A",
            "first_request": paths[0].relative_to(ROOT).as_posix(), "first_sha256": before[0],
            "last_request": paths[-1].relative_to(ROOT).as_posix(), "last_sha256": before[-1],
            "failure_category": last["provider_or_runtime_repair_feedback"]["failure_category"],
            "rebuilt_request": target.relative_to(ROOT).as_posix(), "rebuilt_sha256": digest(target),
            "original_prompt_bytes": last["authoring_context_frame"]["budget_metadata"]["actual_prompt_payload_bytes"],
            "rebuilt_prompt_bytes": len(text.encode("utf-8")),
            "task_sections_bytes": budget["provider_task_sections"]["serialized_bytes"],
            "task_sections_limit": budget["provider_task_sections"]["max_serialized_bytes"],
            "feedback_and_focus_exact": True, "failed_source_and_repair_exact": True,
            "behavior_present": contains_behavior(delivered["public_feedback_summary"]),
            "historical_requests_unchanged": True,
        })
    if len(rows) != 8 or sum(row["arm"] == "T" for row in rows) != 6:
        raise ValueError("expected the fixed eight historical omissions: A=2, T=6")
    report = {"status": "PASS", "scope": "offline_frame_and_provider_serialization_regression",
        "paid_calls": 0, "history_regraded": False,
        "common_prompt_rule": "historical contracts retained to isolate delivery; new shared guidance tested separately",
        "cases": rows}
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "cases": len(rows),
        "max_rebuilt_prompt_bytes": max(row["rebuilt_prompt_bytes"] for row in rows),
        "max_task_sections_bytes": max(row["task_sections_bytes"] for row in rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
