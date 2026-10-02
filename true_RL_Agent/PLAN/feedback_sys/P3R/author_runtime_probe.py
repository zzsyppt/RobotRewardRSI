"""Bounded P3-R.0-a replay of eight immutable P3 first prompts.

prepare/inspect are offline. run requires a separately reviewed stage card;
all provider attempts use the production journal and budget guard. No repair,
source follow-up, simulator, candidate training, or automatic retry is run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
P3 = Path(__file__).resolve().parent.parent / "P3"
CASES = ("stair-01", "stair-03", "push-02", "push-04")
PRICES = {"cache_hit": 0.006, "cache_miss": 0.3, "output": 1.2}
INPUT_RESERVE = 40000
DEFAULT_OUTPUT_CAP = 32768


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def config(output: Path, stage: dict[str, Any]):
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleProviderConfig
    p = stage["provider"]
    return OpenAICompatibleProviderConfig(
        model=p["requested_model"], output_dir=output,
        base_url=p["base_url"], endpoint=p["endpoint"],
        api_key_env=p["api_key_env"], env_file=Path(p["env_file"]),
        max_tokens=stage["budget"]["max_output_tokens"], timeout_s=p["timeout_s"],
        max_response_bytes=p["max_response_bytes"],
        reasoning_mode=p["reasoning_mode"], reasoning_effort=p["reasoning_effort"],
        empty_final_reasoning_retry=False, temperature=p["temperature"],
    )


def prepare(output: Path, *, max_output_tokens: int = DEFAULT_OUTPUT_CAP, usd_cap: float = 3.0) -> dict[str, Any]:
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider
    from true_rl_agent.prompt_serialization import canonical_prompt_json
    old_stage = read(P3 / "author_stage.json")
    frozen = read(P3 / "author_freeze_api_r7.json")
    stage = {
        "schema_version": "feedback_p3r.runtime_probe.v1",
        "status": "PREPARED_NOT_RELEASED",
        "provider": {**old_stage["provider"], "reasoning_effort": "high", "max_output_tokens": max_output_tokens},
        "budget": {"usd_cap": usd_cap, "max_provider_requests": 8,
                   "input_reserve": INPUT_RESERVE, "max_output_tokens": max_output_tokens,
                   "prices_usd_per_million": PRICES,
                   "pricing_checked_date": "2026-09-29",
                   "pricing_source": "https://api-docs.deepseek.com/quick_start/pricing/",
                   "worst_request_usd": (INPUT_RESERVE * PRICES["cache_miss"] + max_output_tokens * PRICES["output"]) / 1e6},
        "definition": {
            "first_complete": "Valid complete author response (candidate_draft or source_request), nonempty final, finish_reason != length.",
            "direct_candidate": "Reported separately; materialization is not static admission or semantic correctness.",
            "pass": "8/8 first_complete; no retries or setting search within these eight opportunities.",
            "source_request": "Count as complete response, but do not execute the requested follow-up in R.0-a.",
            "static_admission": "Separate inspection; not this runtime gate.",
        },
        "source": {"p3_freeze_sha256": sha((P3 / "author_freeze_api_r7.json").read_bytes()),
                   "p3_stage_sha256": sha((P3 / "author_stage.json").read_bytes())},
        "system_prompt_historical_equivalence": "PENDING_REMOTE_R7_CHECK",
        "pricing_basis": "Conservative peak accounting, not the provider invoice.",
    }
    stage["provider"]["empty_final_reasoning_retry"] = False
    if max_output_tokens <= 0 or stage["budget"]["worst_request_usd"] * 8 > usd_cap:
        raise ValueError("eight requests cannot fit the declared positive output and USD budget")
    provider = OpenAICompatibleCandidateProvider(config(output / "unused_candidates", stage))
    all_checks, selected = [], []
    for row in frozen["first_prompts"]:
        path = P3 / "author_preflight_r5" / "prompts" / f"{row['case_id']}_{row['arm']}.json"
        raw = path.read_bytes()
        payload = json.loads(raw)
        text = canonical_prompt_json(payload)
        if sha(raw) != row["sha256"] or sha(text.encode()) != row["canonical_prompt_sha256"]:
            raise ValueError(f"P3 frozen input changed: {path.name}")
        all_checks.append({"case_id": row["case_id"], "arm": row["arm"], "matches_p3_freeze": True})
        if row["case_id"] not in CASES:
            continue
        request = provider._chat_completion_request(payload)
        if request["messages"][1]["content"] != text:
            raise ValueError("provider serialization changed the frozen user message")
        target = output / "prompts" / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        selected.append({"case_id": row["case_id"], "arm": row["arm"],
                         "path": str(target.relative_to(output)).replace("\\", "/"),
                         "file_sha256": sha(raw), "user_sha256": sha(text.encode()),
                         "system_sha256": sha(request["messages"][0]["content"].encode()),
                         "request_sha256": sha(canonical_prompt_json(request).encode()),
                         "user_bytes": len(text.encode()),
                         "message_bytes": sum(len(m["content"].encode()) for m in request["messages"]),
                         "whole_request_bytes": len(canonical_prompt_json(request).encode())})
    if len(selected) != 8 or len(all_checks) != 24:
        raise ValueError("expected eight selected and 24 checked first prompts")
    random.Random(2026092901).shuffle(selected)
    for index, row in enumerate(selected, 1):
        row["slot"] = index
        row["generation_id"] = f"p3r-runtime-20260929-generation-{(index + 1) // 2:03d}"
        row["lane_id"] = "lane-a" if index % 2 else "lane-b"
    stage["first_prompts"] = selected
    stage["all_p3_first_prompt_checks"] = all_checks
    stage["script_sha256"] = sha(Path(__file__).read_bytes())
    write(output / "stage.json", stage)
    return stage


def inspect(stage_path: Path, output: Path) -> dict[str, Any]:
    from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
    stage = read(stage_path)
    price = stage["budget"]["prices_usd_per_million"]
    input_reserve = stage["budget"]["input_reserve"]
    output_cap = stage["budget"]["max_output_tokens"]
    rows = []
    for spec in stage["first_prompts"]:
        generation, lane = spec["generation_id"], spec["lane_id"]
        journal = RoundAuthoringJournal(output / "authoring" / generation / lane, round_id=f"{generation}-{lane}")
        for attempt in journal.snapshot().attempts:
            record = attempt.get("provider_attempt") or {}
            usage = record.get("provider_usage") or {}
            receipt = record.get("provider_response_receipt") or {}
            known = all(isinstance(usage.get(k), int) for k in ("request_units", "response_units"))
            inputs = usage["request_units"] if known else input_reserve
            outputs = usage["response_units"] if known else output_cap
            hit, miss = usage.get("request_cache_hit_units"), usage.get("request_cache_miss_units")
            cache_known = isinstance(hit, int) and isinstance(miss, int) and hit + miss == inputs
            cost = ((hit * price["cache_hit"] + miss * price["cache_miss"] if cache_known else inputs * price["cache_miss"]) + outputs * price["output"]) / 1e6
            rows.append({"case_id": spec["case_id"], "arm": spec["arm"],
                         "journal_status": attempt["status"], "usage_known": known,
                         "input_tokens_actual_or_reserved": inputs, "output_tokens_actual_or_reserved": outputs,
                         "usd_peak_actual_or_reserved": cost,
                         "provider_status": record.get("provider_status"),
                         "provider_failure": record.get("provider_failure"),
                         "first_complete": record.get("first_complete", False),
                         "direct_candidate": record.get("provider_response_mode") == "candidate_draft",
                         "receipt": receipt})
    complete = sum(row["first_complete"] for row in rows)
    total = sum(r["usd_peak_actual_or_reserved"] for r in rows)
    violations = []
    if total > stage["budget"]["usd_cap"]:
        violations.append("stage_usd_exceeded")
    if any(r["usage_known"] and r["input_tokens_actual_or_reserved"] > input_reserve for r in rows):
        violations.append("actual_input_exceeded_reserve")
    if any(r["receipt"].get("response_model") not in {None, stage["provider"]["current_canonical_model"]} for r in rows):
        violations.append("unexpected_response_model")
    deviations = [{"case_id": r["case_id"], "arm": r["arm"], "actual_output_tokens": r["output_tokens_actual_or_reserved"]}
                  for r in rows if r["usage_known"] and r["output_tokens_actual_or_reserved"] > output_cap]
    return {"status": "COMPLETE" if len(rows) == 8 else "PARTIAL",
            "attempts": len(rows), "first_complete": complete,
            "direct_candidate_count": sum(r["direct_candidate"] for r in rows),
            "source_request_count": sum(r["first_complete"] and not r["direct_candidate"] for r in rows),
            "denominator": 8, "runtime_gate": "PASS" if len(rows) == complete == 8 and not violations else "NOT_PASSED",
            "usd_peak_actual_or_reserved": total, "violations": violations,
            "output_request_deviations": deviations,
            "semantic_value_gate": "NOT_EVALUATED", "static_admission": "NOT_EVALUATED", "rows": rows}


def run(stage_path: Path, output: Path) -> dict[str, Any]:
    from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider
    from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
    from true_rl_agent.agent_core.provider_errors import ProviderCandidateError
    from true_rl_agent.prompt_artifacts import RoundAuthoringJournal, verify_prompt_payload
    from true_rl_agent.prompt_serialization import canonical_prompt_json
    stage = read(stage_path)
    if stage["status"] != "FROZEN_API_RELEASED" or not stage.get("resource_allocation_ref"):
        raise ValueError("reviewed resource allocation and explicit frozen API release required")
    if stage["script_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("probe script changed after preparation")
    if stage["system_prompt_historical_equivalence"] != "VERIFIED_R7":
        raise ValueError("historical system prompt equivalence not yet verified")
    caps = {"attempts_per_lane_per_generation": 1,
            "input_units_per_attempt_reserve": stage["budget"]["input_reserve"],
            "output_units_per_attempt_cap": stage["budget"]["max_output_tokens"]}
    with acquire_process_file_lock(output / ".runtime_probe.lock", blocking=False):
        for spec in stage["first_prompts"]:
            generation, lane = spec["generation_id"], spec["lane_id"]
            round_id = f"{generation}-{lane}"
            journal = RoundAuthoringJournal(output / "authoring" / generation / lane, round_id=round_id)
            if journal.snapshot().attempts:
                # Including an unknown response: its one opportunity is spent.
                continue
            report = inspect(stage_path, output)
            if report["usd_peak_actual_or_reserved"] + stage["budget"]["worst_request_usd"] > stage["budget"]["usd_cap"]:
                raise RuntimeError("remaining runtime probe USD cannot cover one full request")
            path = stage_path.parent / spec["path"]
            if sha(path.read_bytes()) != spec["file_sha256"]:
                raise ValueError("frozen prompt file changed")
            payload = read(path)
            provider = OpenAICompatibleCandidateProvider(config(output / "provider_candidates" / generation / lane, stage))
            request = provider._chat_completion_request(payload)
            if sha(canonical_prompt_json(request).encode()) != spec["request_sha256"]:
                raise ValueError("actual request differs from frozen system/user/settings")
            provider.validate_credential()
            guarded = CostGuardedProvider(delegate=provider, output_dir=output, generation_id=generation,
                                          lane_id=lane, provider_caps=caps, max_generations=4)
            with journal.acquire_lease():
                journal.bind_budget(max_provider_attempts=1, max_source_requests_per_round=0,
                                    max_source_ids_per_request=2, max_source_request_bytes_per_round=0)
                handle = journal.begin_attempt(payload)
                payload = verify_prompt_payload(handle.prompt_payload_ref)
                record = {"round_id": round_id, "attempt": 1, "attempt_id": handle.attempt_id}
                service_stop = False
                try:
                    response = guarded.next_authoring_response(prompt_payload=payload)
                    receipt = dict(response.provider_receipt or {})
                    record.update(provider_status="pass", provider_response_mode=response.response_mode,
                                  provider_usage=dict(response.provider_usage or {}), provider_response_receipt=receipt,
                                  first_complete=receipt.get("finish_reason") != "length" and receipt.get("response_text_utf8_bytes", 0) > 0)
                    if response.response_mode == "candidate_draft":
                        record["candidate"] = response.candidate.to_public_dict()
                    else:
                        record["logical_ids"] = list(response.logical_ids)
                except ProviderCandidateError as exc:
                    record.update(provider_status="fail", first_complete=False,
                                  provider_failure=exc.to_public_dict(), provider_usage=dict(exc.provider_usage),
                                  provider_response_receipt=dict(exc.provider_receipt))
                    service_stop = exc.requires_service_recovery
                journal.complete_attempt(handle, record, continuation_state={"disposition": "terminal", "stop_reason": "single_request_probe"})
            report = inspect(stage_path, output)
            write(output / "summary.json", report)
            print(json.dumps({"case_id": spec["case_id"], "arm": spec["arm"], "first_complete": record["first_complete"],
                              "attempts": report["attempts"], "usd_peak_actual_or_reserved": report["usd_peak_actual_or_reserved"]}), flush=True)
            known = all(isinstance(record.get("provider_usage", {}).get(k), int) for k in ("request_units", "response_units"))
            if service_stop or not known or report["violations"]:
                raise RuntimeError("provider service/usage requires review; response recorded, no automatic resubmit")
    return inspect(stage_path, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "inspect", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_OUTPUT_CAP,
                        help="prepare only; run/inspect use the sealed stage budget")
    parser.add_argument("--usd-cap", type=float, default=3.0,
                        help="prepare only; independent allocation available to this probe")
    args = parser.parse_args()
    stage_path = args.stage or args.output / "stage.json"
    if args.mode == "prepare":
        if stage_path.exists():
            raise ValueError("use a fresh preparation directory; never overwrite an existing stage")
        stage = prepare(args.output, max_output_tokens=args.max_output_tokens, usd_cap=args.usd_cap)
        print(json.dumps({"status": stage["status"], "first_prompts": len(stage["first_prompts"]),
                          "all_p3_prompt_checks": len(stage["all_p3_first_prompt_checks"]),
                          "worst_eight_requests_usd": stage["budget"]["worst_request_usd"] * 8}))
    else:
        report = inspect(stage_path, args.output) if args.mode == "inspect" else run(stage_path, args.output)
        write(args.output / "summary.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
