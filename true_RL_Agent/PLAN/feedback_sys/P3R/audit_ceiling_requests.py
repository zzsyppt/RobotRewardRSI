"""Read-only audit of saved R.0-b author requests; no provider calls or rebuilds."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_path(ref, base):
    path = Path(ref["path"])
    if not path.is_absolute():
        path = base / path
    if sha(path) != ref["sha256"].removeprefix("sha256:"):
        raise ValueError(f"frozen input hash differs: {path}")
    return path


def payload_report(payload, *, initial, focus, provider):
    from true_rl_agent.prompt_serialization import canonical_prompt_json
    from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt
    frame = payload["authoring_context_frame"]
    budget = frame["budget_metadata"]
    task_receipt = validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=payload)
    encoded = canonical_prompt_json(payload).encode("utf-8")
    request = provider._chat_completion_request(payload)  # Serialization only; no transport/credential lookup.
    user_text = request["messages"][1]["content"]
    checks = {
        "complete_feedback_matches_initial": payload.get("public_feedback_summary") == initial.get("public_feedback_summary"),
        "registered_focus_retained": focus in payload.get("selection_guidance", []),
        "selection_guidance_matches_initial": payload.get("selection_guidance") == initial.get("selection_guidance"),
        "recorded_payload_bytes_match": budget["actual_prompt_payload_bytes"] == len(encoded),
        "provider_user_is_saved_payload": user_text.encode("utf-8") == encoded,
    }
    return {"frame_kind": frame["frame_kind"], "checks": checks,
        "payload_sha256": hashlib.sha256(encoded).hexdigest(), "payload_bytes": len(encoded),
        "request_sha256": hashlib.sha256(canonical_prompt_json(request).encode("utf-8")).hexdigest(),
        "request_bytes": len(canonical_prompt_json(request).encode("utf-8")),
        "task_capacity_receipt": task_receipt,
        "visible_source_ids": [item.get("logical_id") for item in payload.get("source_payloads", [])],
        "source_boundary": "Validate the saved public payload; do not require inline failed source when task policy excludes it."}


def audit(args):
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.memory import PromptPayloadRef
    from true_rl_agent.prompt_artifacts import RoundAuthoringJournal, validate_authoring_budget_usage, verify_prompt_payload
    from true_rl_agent.prompt_serialization import canonical_prompt_json
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    stage = read(args.stage)
    data_path = bound_path(stage["data_ref"], args.stage.parent)
    data = read(data_path)
    old_path = bound_path(data["p3_cases"], data_path.parent)
    old_cases = {row["case_id"]: row for row in read(old_path)["cases"]}
    expected = {(row["case_id"], row["variant"]): row for row in stage["first_prompts"]}
    p, budget = stage["provider"], stage["budget"]
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model=p["requested_model"], output_dir=args.output.parent / "unused_no_candidates",
        base_url=p["base_url"], endpoint=p["endpoint"], max_tokens=budget["max_output_tokens"],
        reasoning_mode=p["reasoning_mode"], reasoning_effort=p["reasoning_effort"], temperature=p["temperature"],
        empty_final_reasoning_retry=p.get("empty_final_reasoning_retry", False)))
    episodes, attempts, failures = [], [], []
    source_matches = project_source_snapshot(args.root) == stage["production_source_snapshot"]
    if not source_matches:
        failures.append({"round_id": None, "errors": ["audit source differs from the frozen provider serialization source"]})
    for slot in stage["schedule"]:
        round_id = f"{slot['generation_id']}-{slot['lane_id']}"
        episode = {**slot, "round_id": round_id, "status": "NOT_COMPLETE", "errors": []}
        episodes.append(episode)
        try:
            frozen = expected[(slot["case_id"], slot["variant"])]
            initial = read(args.stage.parent / frozen["path"])
            if hashlib.sha256(canonical_prompt_json(initial).encode()).hexdigest() != frozen["canonical_prompt_sha256"]:
                raise ValueError("frozen first-prompt content hash differs")
            round_dir = args.results / "authoring" / slot["generation_id"] / slot["lane_id"]
            snapshot = RoundAuthoringJournal(round_dir, round_id=round_id).snapshot()
            contract = read(round_dir / "authoring_budget.json")
            source_usage = validate_authoring_budget_usage(list(snapshot.completed_attempt_records),
                budget_contract=contract, expected_round_id=round_id, reserved_attempt_count=snapshot.reserved_attempt_count)
            if (contract["max_provider_attempts"] != budget["max_attempts_per_author_episode"]
                    or contract["max_source_requests_per_round"] != budget["max_source_requests_per_author_episode"]):
                raise ValueError("saved episode budget differs from frozen stage")
            episode.update(provider_attempts_reserved=snapshot.reserved_attempt_count, **source_usage,
                provider_attempt_limit=contract["max_provider_attempts"], source_request_limit=contract["max_source_requests_per_round"])
            for attempt in snapshot.attempts:
                row = {**slot, "round_id": round_id, "attempt": attempt["sequence"],
                       "journal_status": attempt["status"], "errors": []}
                attempts.append(row)
                record = attempt.get("provider_attempt") or {}
                usage = record.get("provider_usage") or {}
                row.update(provider_status=record.get("provider_status"), response_mode=record.get("provider_response_mode"),
                    verifier_status=record.get("verifier_status"), provider_failure=record.get("provider_failure"),
                    provider_usage=usage, response_receipt=record.get("provider_response_receipt"),
                    usage_known=all(isinstance(usage.get(k), int) for k in ("request_units", "response_units", "total_units")))
                if attempt.get("prompt_payload") is None:
                    row.update(frame_kind="NO_SAVED_PROMPT", status="NOT_VERIFIABLE")
                    continue
                try:
                    ref = PromptPayloadRef.from_mapping(attempt["prompt_payload"])
                    payload = verify_prompt_payload(ref)
                    row.update(prompt_ref=ref.to_public_dict(), **payload_report(payload, initial=initial,
                        focus=old_cases[slot["case_id"]]["common_focus"], provider=provider))
                    receipt = record.get("provider_response_receipt") or {}
                    if receipt:
                        row["checks"]["response_requested_settings_match"] = (
                            receipt.get("requested_model") == p["requested_model"]
                            and receipt.get("requested_output_units") == budget["max_output_tokens"]
                            and receipt.get("requested_reasoning_effort") == p["reasoning_effort"])
                    if row["usage_known"]:
                        row["checks"]["actual_input_within_frozen_reserve"] = usage["request_units"] <= budget["input_reserve"]
                        row["actual_output_exceeds_requested_cap"] = usage["response_units"] > budget["max_output_tokens"]
                    if attempt["sequence"] == 1:
                        row["checks"]["first_payload_matches_freeze"] = row["payload_sha256"] == frozen["canonical_prompt_sha256"]
                        row["checks"]["first_request_matches_freeze"] = row["request_sha256"] == frozen["request_sha256"]
                    row["errors"] = [key for key, passed in row["checks"].items() if not passed]
                    row["status"] = "FAIL" if row["errors"] else "PASS"
                except (ValueError, TypeError, KeyError, OSError) as exc:
                    row.update(status="FAIL", errors=[str(exc)])
            result_path = args.results / "results" / f"{round_id}.json"
            if result_path.is_file():
                terminal = read(result_path)
                if any(terminal[key] != slot[key] for key in ("case_id", "variant", "repetition", "generation_id", "lane_id")):
                    raise ValueError("terminal identity differs from frozen schedule")
                episode.update(status="COMPLETE", stop_reason=terminal["stop_reason"], terminal_sha256=sha(result_path))
            if any(row["errors"] for row in attempts if row["round_id"] == round_id):
                episode["errors"].append("actual_request_checks_failed")
        except (ValueError, TypeError, KeyError, OSError) as exc:
            episode["errors"].append(str(exc))
        if episode["errors"]:
            failures.append({"round_id": round_id, "errors": episode["errors"]})
    unknown = [{"round_id": row["round_id"], "attempt": row["attempt"], "journal_status": row["journal_status"]}
               for row in attempts if not row["usage_known"]]
    unverified = [row for row in attempts if row.get("status") != "PASS"]
    incomplete = [row for row in episodes if row["status"] != "COMPLETE"]
    status = "FAIL" if failures else "INCOMPLETE" if incomplete or unverified else "PASS_WITH_UNKNOWN_USAGE" if unknown else "PASS"
    groups = []
    for key in sorted({(r["case_id"], r["variant"], r.get("frame_kind", "UNVERIFIED")) for r in attempts}):
        group = [r for r in attempts if (r["case_id"], r["variant"], r.get("frame_kind", "UNVERIFIED")) == key]
        groups.append({"case_id": key[0], "variant": key[1], "frame_kind": key[2], "attempts": len(group),
            "request_checks_passed": sum(r.get("status") == "PASS" for r in group),
            "usage_known": sum(r["usage_known"] for r in group),
            "known_input_units": sum(r["provider_usage"]["request_units"] for r in group if r["usage_known"]),
            "known_output_units": sum(r["provider_usage"]["response_units"] for r in group if r["usage_known"])})
    return {"schema_version": "feedback_p3r.actual_request_audit.v1", "status": status, "api_calls": 0,
        "stage_sha256": sha(args.stage), "auditor_sha256": sha(Path(__file__)),
        "production_source_matches_freeze": source_matches,
        "scope": "Saved journal prompts and actual usage only; reconstructed request serialization is not a captured HTTP body or semantic/behavioral acceptance.",
        "denominators": {"planned_episodes": len(stage["schedule"]), "completed_terminals": len(episodes) - len(incomplete),
            "reserved_attempts_seen": len(attempts), "request_checks_passed": sum(r.get("status") == "PASS" for r in attempts),
            "unknown_usage_attempts": len(unknown), "terminal_outcomes": dict(Counter(r.get("stop_reason", "NOT_COMPLETE") for r in episodes))},
        "groups": groups, "unknown_usage": unknown, "failures": failures, "episodes": episodes, "attempts": attempts,
        "limitations": ["Failed/no-candidate terminals remain in denominators.",
            "No new token estimate, journal, response, candidate, or private-source expectation is manufactured."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "src"))
    report = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "denominators", "failures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
