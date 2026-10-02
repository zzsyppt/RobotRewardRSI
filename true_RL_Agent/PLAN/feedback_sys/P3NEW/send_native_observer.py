"""Send one explicitly released native request; never retry.

CLI: --prepared DIR --release release.json --secret specified.env [--timeout-s 180]
The release JSON must contain schema_version=p3new.native_observer_release.v1,
status=RELEASED_FOR_ONE_CALL, scope=old_development_not_n5, request_sha256,
prepared_card_sha256, model=deepseek-flash, expected_fingerprint, and
reserved_cost_usd=0.05. Author/freeze that card after reviewing prepared evidence.
The secret file must contain exactly one DEEPSEEK_API_KEY=... assignment.
Receipt and compact transition journal stay beside the sealed input files;
response/ holds raw provider JSON, unchanged answer text, and mechanical audit.
No API price is inferred from the reservation; unreported actual fees stay null.

N5 uses --prepared REPORT_DIR --episode-alias e1 (then explicitly e2).
Its p3new.native_observer_report_release.v1 card has status
RELEASED_FOR_TWO_CALLS, scope=n5_confirmation, report_card_sha256, packets,
selection_sha256, extraction_sha256, capture_card_sha256, report_limits,
model, expected_fingerprint, and reserved_cost_usd=0.05. These fields bind the
prepared report.json and both packets. Each invocation rebuilds cumulative
reservations from existing receipts; timeout/rejection never refunds them.
--aggregate NEW.json only archives the combined audit, without submission.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from prepare_native_observer import (
    FINGERPRINT, MODEL, RESERVATION_USD, build, digest, read_json, require, report_budget,
)
from true_rl_agent.evaluation.visual_observer import (
    SCHEMA_VERSION, ObserverBudget, audit_observer_output, call_observer,
)


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def read_secret(path: Path) -> str:
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        if separator and name.strip() == "DEEPSEEK_API_KEY":
            values.append(value.strip().strip("\"'"))
    require(len(values) == 1 and bool(values[0]), "specified secret must define one nonempty DEEPSEEK_API_KEY")
    return values[0]


def audit_response(receipt: dict, built: dict) -> tuple[str | None, dict]:
    content = None
    answer = None
    try:
        value = receipt["response"]["choices"][0]["message"]["content"]
        require(isinstance(value, str), "provider answer content is not a string")
        content = value
        answer = json.loads(content)
        require(isinstance(answer, dict), "observer answer is not an object")
        audited = audit_observer_output(answer, built["windows"], numeric_check=built["numeric_check"])
    except Exception as exc:
        audited = {"schema_version": SCHEMA_VERSION, "raw_output": answer, "observations": [],
                   "passed": False, "violations": ["answer_parse_or_audit_error"],
                   "error_type": type(exc).__name__, "semantic_support": "not_evaluated",
                   "raw_answer_preserved_in": "raw_response.json"}
    return content, audited


def send(prepared_dir: Path, release_path: Path, secret_path: Path, *,
         timeout_s: int = 180, transport=None, _report_context=None) -> dict:
    """Use production submission; injectable transport is only for offline CPU tests."""
    packet = prepared_dir.resolve()
    receipt_path, journal_path = packet / "receipt.json", packet / "receipt_journal.jsonl"
    response_dir = packet / "response"
    require(not any(path.exists() for path in (receipt_path, journal_path, response_dir)),
            "receipt or response already exists; this packet cannot be submitted again")
    require(type(timeout_s) is int and timeout_s > 0, "timeout must be positive")
    release_bytes = release_path.read_bytes()
    release = json.loads(release_bytes)
    card = read_json(packet / "card.json")
    scope = "n5_confirmation" if _report_context else "old_development_not_n5"
    if _report_context is None:
        require(release.get("schema_version") == "p3new.native_observer_release.v1"
                and release.get("status") == "RELEASED_FOR_ONE_CALL"
                and release.get("scope") == scope, "explicit development release is required")
        binding = release
    else:
        binding = _report_context["packet"]
    require(binding.get("prepared_card_sha256") == digest(packet / "card.json")
            and binding.get("request_sha256") == card["request_sha256"] == digest(packet / "request.json"),
            "release/card/request hashes differ")
    require(release.get("model") == card["model"] == MODEL
            and release.get("expected_fingerprint") == card["expected_fingerprint"] == FINGERPRINT
            and release.get("reserved_cost_usd") == RESERVATION_USD, "release settings differ")
    private = read_json(packet / "private_sidecar.json")
    alias = binding["episode_alias"] if _report_context else None
    built = build(Path(private["source_export"]), packet / "selection.json", episode_alias=alias)
    require(built["prepared"]["request_bytes"] == (packet / "request.json").read_bytes(),
            "rebuilt request bytes differ from sealed request")
    for name in ("card", "private_sidecar", "mechanical_audit", "alignment"):
        require(built[name] == read_json(packet / (name + ".json")), "rebuilt evidence differs: " + name)
    # All evidence/release checks precede reading the explicitly named secret.
    api_key = read_secret(secret_path)
    release_sha = hashlib.sha256(release_bytes).hexdigest()
    driver_sha = digest(Path(__file__))

    def receipt_sink(receipt: dict) -> None:
        stamped = {**receipt, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": scope, "formal_confirmation": bool(_report_context),
            "release_sha256": release_sha, "prepared_card_sha256": binding["prepared_card_sha256"],
            "sender_sha256": driver_sha, "actual_api_cost_usd": None,
            "actual_cost_status": "unknown; preserve provider response/usage without inventing a fee"}
        if _report_context:
            stamped.update(episode_alias=alias, report_card_sha256=release["report_card_sha256"])
        serialized = json_text(stamped)
        if receipt["status"] == "reserved":
            # Exclusive creation is the one-shot guard, before production transport.
            with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(serialized)
            journal_mode = "x"
        else:
            next_path = packet / "receipt.next.json"
            next_path.write_text(serialized, encoding="utf-8")
            next_path.replace(receipt_path)
            journal_mode = "a"
        transition = {key: value for key, value in stamped.items()
                      if key not in {"request", "private_sidecar", "response"}}
        with journal_path.open(journal_mode, encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(transition, ensure_ascii=False, allow_nan=False) + "\n")

    receipt = call_observer(built["prepared"], built["windows"],
        budget=(_report_context["budget"] if _report_context else
                ObserverBudget(max_pixels=50_000_000, max_image_bytes=50_000_000,
                               max_cost_usd=RESERVATION_USD, max_calls=1)),
        reserved_cost_usd=RESERVATION_USD, url="https://api.deepseek.com/chat/completions",
        api_key=api_key, receipt_sink=receipt_sink, transport=transport,
        timeout_s=timeout_s, expected_fingerprint=FINGERPRINT)
    response_dir.mkdir(exist_ok=False)
    (response_dir / "release.json").write_bytes(release_bytes)
    (response_dir / "raw_response.json").write_text(json_text(receipt["response"]), encoding="utf-8")
    content, audited = audit_response(receipt, built)
    if content is not None:
        (response_dir / "raw_answer.txt").write_bytes(content.encode("utf-8"))
    (response_dir / "mechanical_audit.json").write_text(json_text(audited), encoding="utf-8")
    result = {"status": receipt["status"], "scope": scope, "formal_confirmation": bool(_report_context),
        "request_sha256": receipt["request_sha256"], "release_sha256": release_sha,
        "raw_response_sha256": digest(response_dir / "raw_response.json"),
        "usage": receipt.get("usage"), "actual_fingerprint": receipt.get("fingerprint"),
        "fingerprint_matches": receipt["fingerprint_matches"], "mechanical_audit_passed": audited["passed"],
        "reserved_cost_usd": RESERVATION_USD, "actual_api_cost_usd": None,
        "actual_cost_status": "unknown; reservation is not spend; provider response/usage retained",
        "alignment_not_checked": built["alignment"]["not_checked"]}
    (response_dir / "result.json").write_text(json_text(result), encoding="utf-8")
    return result


def rebuild_report(report_dir: Path, release_path: Path):
    """Read the two frozen packets and reconstruct reservations, never refunding failures."""
    root = report_dir.resolve()
    card = read_json(root / "report.json")
    release = read_json(release_path)
    require(card["schema_version"] == "p3new.native_observer_report.v1"
            and card["scope"] == "n5_confirmation", "N5 report card required")
    require(release.get("schema_version") == "p3new.native_observer_report_release.v1"
            and release.get("status") == "RELEASED_FOR_TWO_CALLS"
            and release.get("scope") == "n5_confirmation", "explicit two-packet N5 release required")
    require(release["report_card_sha256"] == digest(root / "report.json")
            and release["packets"] == card["packets"]
            and release["selection_sha256"] == card["selection_sha256"]
            and release["extraction_sha256"] == card["extraction_sha256"]
            and release["capture_card_sha256"] == card["capture_card"]["sha256"]
            and release["report_limits"] == card["report_limits"], "report release hashes/limits differ")
    require(card["report_limits"] == {"images": 24, "subframes": 96, "pixels": 50_000_000,
            "image_bytes": 100_000_000, "calls": 2, "reserved_cost_usd": 0.10}, "report limits differ")
    require(release["model"] == card["model"] == MODEL
            and release["expected_fingerprint"] == card["expected_fingerprint"] == FINGERPRINT
            and release["reserved_cost_usd"] == card["reserved_cost_usd_per_call"] == RESERVATION_USD,
            "released observer settings differ")
    for name in ("capture_card", "selection_card"):
        require(digest(Path(card[name]["path"])) == card[name]["sha256"], "frozen input changed: " + name)
    require([row["episode_alias"] for row in card["packets"]] == ["e1", "e2"]
            and [row["directory"] for row in card["packets"]] == ["e1", "e2"], "fixed packet order differs")
    budget, planned, items, gap = report_budget(), report_budget(), [], False
    for index, binding in enumerate(card["packets"]):
        packet = root / binding["directory"]
        private = read_json(packet / "private_sidecar.json")
        built = build(Path(private["source_export"]), packet / "selection.json", episode_alias=binding["episode_alias"])
        require(binding["prepared_card_sha256"] == digest(packet / "card.json")
                and binding["request_sha256"] == digest(packet / "request.json")
                and built["prepared"]["request_bytes"] == (packet / "request.json").read_bytes()
                and digest(packet / "selection.json") == card["selection_sha256"]
                and built["card"]["extraction_sha256"] == card["extraction_sha256"], "frozen packet differs")
        for name in ("card", "private_sidecar", "mechanical_audit", "alignment"):
            require(built[name] == read_json(packet / (name + ".json")), "rebuilt packet differs: " + name)
        require(planned.reserve(built["windows"], RESERVATION_USD) == card["cumulative_reservations"][index],
                "planned cumulative reservation differs")
        receipt_path = packet / "receipt.json"
        receipt = read_json(receipt_path) if receipt_path.exists() else None
        if receipt is None:
            gap = True
        else:
            require(not gap, "later packet has receipt before earlier packet")
            require(receipt["status"] in {"reserved", "received", "transport_error"}
                    and receipt["request_sha256"] == binding["request_sha256"]
                    and receipt["release_sha256"] == digest(release_path)
                    and receipt["report_card_sha256"] == release["report_card_sha256"]
                    and receipt["prepared_card_sha256"] == binding["prepared_card_sha256"]
                    and receipt["episode_alias"] == binding["episode_alias"], "receipt belongs to another report/call")
            require(budget.reserve(built["windows"], RESERVATION_USD) == receipt["budget"],
                    "receipt cumulative reservation differs")
        items.append({"binding": binding, "packet": packet, "built": built, "receipt": receipt})
    return card, budget, items


def send_report(report_dir: Path, release_path: Path, secret_path: Path, *, episode_alias: str,
                timeout_s: int = 180, transport=None) -> dict:
    """Submit exactly the next explicitly named packet, with the whole report's budget."""
    card, budget, items = rebuild_report(report_dir, release_path)
    pending = [item for item in items if item["receipt"] is None]
    require(pending and pending[0]["binding"]["episode_alias"] == episode_alias,
            "select the first unsubmitted alias; previous calls cannot be retried")
    item = pending[0]
    return send(item["packet"], release_path, secret_path, timeout_s=timeout_s, transport=transport,
                _report_context={"packet": item["binding"], "budget": budget})


def aggregate_report(report_dir: Path, release_path: Path, output: Path) -> dict:
    """Retain original per-call audits, including malformed containers and all observations."""
    require(not output.exists(), "aggregate output must be fresh")
    card, budget, items = rebuild_report(report_dir, release_path)
    observations, violations, calls, raw = [], [], [], []
    for index, item in enumerate(items):
        packet, built, receipt = item["packet"], item["built"], item["receipt"]
        alias, prefix = item["binding"]["episode_alias"], f"calls[{index}]"
        call = {"episode_alias": alias, "request_sha256": item["binding"]["request_sha256"],
                "receipt_sha256": digest(packet / "receipt.json") if receipt else None}
        if receipt is None or receipt["status"] != "received":
            call["status"] = receipt["status"] if receipt else "not_submitted"
            violations.append(prefix + ":" + call["status"])
            calls.append(call)
            raw.append(None)
            continue
        audit_path = packet / "response/mechanical_audit.json"
        require(audit_path.exists(), "received packet is missing its preserved audit; finish offline response archival first")
        audited = read_json(audit_path)
        content, rebuilt_audit = audit_response(receipt, built)
        require(audited == rebuilt_audit
                and read_json(packet / "response/raw_response.json") == receipt["response"],
                "archived response or per-call audit differs from original receipt")
        if content is not None:
            require((packet / "response/raw_answer.txt").read_bytes() == content.encode("utf-8"),
                    "archived raw answer differs from original receipt")
        call.update(status="received", audit=audited, audit_sha256=digest(audit_path),
                    raw_response_sha256=digest(packet / "response/raw_response.json"))
        raw.append(deepcopy(audited.get("raw_output")))
        if not receipt.get("eligible_for_confirmation"):
            violations.append(prefix + ":receipt_not_eligible_for_confirmation")
        violations.extend(prefix + ":" + value for value in audited.get("violations", []))
        for record in audited.get("observations", []):
            row = deepcopy(record)
            row["reference"] = prefix + "." + row["reference"]
            row["source_call_index"] = index
            observations.append(row)
        calls.append(call)
    result = {"schema_version": SCHEMA_VERSION, "scope": "n5_confirmation",
        "report_card_sha256": digest(report_dir / "report.json"), "calls": calls,
        "raw_output": {"calls": raw}, "observations": observations, "violations": violations,
        "passed": not violations and not any(row["violations"] for row in observations),
        "semantic_support": "not_evaluated", "all_raw_observations_retained": True,
        "calls_reserved": budget.calls, "reserved_cost_usd": budget.reserved_cost_usd}
    output.write_text(json_text(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--secret", type=Path)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--episode-alias", choices=("e1", "e2"), help="N5: --prepared is the report directory")
    parser.add_argument("--aggregate", type=Path, help="N5: write combined audit offline, no submission")
    args = parser.parse_args()
    try:
        if args.aggregate:
            result = aggregate_report(args.prepared, args.release, args.aggregate)
            print(json.dumps({key: result[key] for key in ("passed", "calls_reserved", "semantic_support")}))
            return 0 if result["passed"] else 1
        require(args.secret is not None, "submission requires an explicitly named secret")
        result = (send_report(args.prepared, args.release, args.secret, episode_alias=args.episode_alias,
                    timeout_s=args.timeout_s) if args.episode_alias else
                  send(args.prepared, args.release, args.secret, timeout_s=args.timeout_s))
    except Exception as exc:
        # Exception strings/tracebacks can contain transport credentials or content.
        print(json.dumps({"status": "NOT_COMPLETED", "error_type": type(exc).__name__,
                          "inspect_receipt_if_present": True, "automatic_retry": False}))
        return 1
    print(json.dumps({key: result[key] for key in (
        "status", "request_sha256", "fingerprint_matches", "mechanical_audit_passed", "actual_api_cost_usd")}))
    return 0 if result["fingerprint_matches"] and result["mechanical_audit_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
