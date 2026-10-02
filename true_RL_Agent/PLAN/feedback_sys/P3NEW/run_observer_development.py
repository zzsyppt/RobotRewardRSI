"""Send one frozen old-media development request, without automatic retry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.evaluation.visual_observer import (
    ObserverBudget, audit_observer_output, call_observer, prepare_observer_request,
)
import prepare_observer_development as preparation


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stair", "push"), required=True)
    parser.add_argument("--secret", type=Path, required=True)
    args = parser.parse_args()
    packet = preparation.OUT
    release = json.loads((packet / "release.json").read_text(encoding="utf-8"))
    assert release["status"] == "TWO_DEVELOPMENT_REQUESTS_RELEASED"
    for name, digest in release["source_hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    target = packet / args.task
    request_bytes = (target / "request.json").read_bytes()
    assert hashlib.sha256(request_bytes).hexdigest() == release["request_hashes"][args.task]
    receipt_path = target / "receipt.json"
    if receipt_path.exists():
        raise RuntimeError("This request is already reserved; inspect its receipt, do not retry")
    windows, _ = (preparation.stair_windows if args.task == "stair" else preparation.push_windows)()
    prepared = prepare_observer_request(
        windows, model=preparation.MODEL, max_tokens=4096,
        generation_options={"thinking": {"type": "disabled"}},
    )
    assert prepared["request_bytes"] == request_bytes
    key = None
    for line in args.secret.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip() == "DEEPSEEK_API_KEY":
            key = value.strip().strip('"').strip("'")
    if not key:
        raise RuntimeError("Configured DeepSeek key missing")
    started = datetime.now(timezone.utc).isoformat()

    def receipt_sink(receipt):
        save(receipt_path, {**receipt, "started_utc": started,
            "recorded_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "old_replay_development_only", "formal_confirmation": False})

    receipt = call_observer(
        prepared, windows,
        budget=ObserverBudget(max_pixels=12_000_000, max_image_bytes=4_000_000,
                              max_cost_usd=0.05, max_calls=1),
        reserved_cost_usd=preparation.PER_CALL_RESERVATION_USD,
        url="https://api.deepseek.com/chat/completions", api_key=key,
        receipt_sink=receipt_sink, timeout_s=60,
        expected_fingerprint=preparation.FINGERPRINT,
    )
    response = receipt["response"]
    content = response["choices"][0]["message"].get("content", "")
    try:
        output = json.loads(content)
        audit = audit_observer_output(output, windows)
    except (ValueError, TypeError, AttributeError):
        audit = {"raw_content": content, "passed": False, "violations": ["output_json"],
                 "semantic_support": "not_evaluated"}
    audit.update(formal_confirmation=False, media_scope="independent_replay",
                 same_run_numeric_checker_used=False, attached_to_numeric_full=False)
    save(target / "mechanical_audit.json", audit)
    usage = receipt.get("usage") or {}
    cost = None
    if all(type(usage.get(k)) is int for k in ("prompt_tokens", "completion_tokens")):
        hit = usage.get("prompt_cache_hit_tokens", 0)
        miss = usage.get("prompt_cache_miss_tokens", usage["prompt_tokens"] - hit)
        cost = (hit * .006 + miss * .3 + usage["completion_tokens"] * 1.2) / 1_000_000
    summary = {"task": args.task, "actual_calls": 1, "retry_calls": 0,
               "fingerprint": receipt.get("fingerprint"), "usage": usage,
               "mechanical_passed": audit["passed"], "semantic_support": "not_evaluated",
               "formal_confirmation": False, "peak_cost_ceiling_usd": cost,
               "cost_is_provider_invoice": False}
    save(target / "result.json", summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
