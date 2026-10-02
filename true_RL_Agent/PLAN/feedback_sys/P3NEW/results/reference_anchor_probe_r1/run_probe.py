"""Exact-input CPU preflight; --execute dispatches the registered single call."""
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile

from prepare_probe import HERE, BASE, runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    prepared = HERE / "final_input/prepared"
    card = runner.read(prepared / "prepare.json")
    payload = runner.read(prepared / "request.http_body.json")
    expected = runner.read(HERE / "expectations.json")
    for name, source in (("runner", card["runner"]), ("coverage", card["coverage_helper"])):
        runner.require(runner.ref(source["file"])["sha256"] == expected[name + "_sha256"], name + " source drift")
    protocol = next(json.loads(b["text"]) for b in payload["messages"][1]["content"]
                    if b["type"] == "text" and json.loads(b["text"]).get("role") == "review_protocol")
    runner.require(runner.sha(runner.coverage.canonical(protocol["data"])) == expected["protocol_data_canonical_sha256"],
                   "protocol meaning differs from forty-packet verification")
    nonprotocol = [b for b in payload["messages"][1]["content"] if b["type"] != "text" or
                   json.loads(b["text"]).get("role") not in {"review_protocol", "reference_catalog"}]
    runner.require(runner.sha(runner.encoded(nonprotocol)) == expected["nonprotocol_blocks_sha256"],
                   "raw evidence differs from forty-packet verification")
    runner.require(card["packet_id"] == "episode-01" and card["judge_id"] == "deepseek" and
                   card["stage"] == "facts" and card["png_count"] == 0 and card["max_output_tokens"] == 32768,
                   "probe identity or budget differs")
    rebuilt, _, rebuilt_card = runner.build(card["manifest"]["file"], "deepseek")
    runner.require(rebuilt_card == card and rebuilt == (prepared / "request.http_body.json").read_bytes(),
                   "prepared input drift")

    archive_path = BASE / "n5_execution_20260930/control/numeric_materials_r1/token_capacity_r1/deepseek_v4_tokenizer.zip"
    archive_bytes = archive_path.read_bytes()
    runner.require(runner.sha(archive_bytes) == "e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495", "cached tokenizer ZIP drift")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        candidates = [archive.read(name) for name in archive.namelist() if name.endswith("tokenizer.json")]
    candidates = [data for data in candidates if runner.sha(data) == "89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614"]
    runner.require(len(candidates) == 1, "tokenizer member differs")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_str(candidates[0].decode("utf-8"))
    texts = []
    for message in payload["messages"]:
        content = message["content"]
        texts.extend([content] if isinstance(content, str) else [b["text"] for b in content if b["type"] == "text"])
    text_tokens = sum(len(tokenizer.encode(text).ids) for text in texts)
    runner.require(text_tokens < 360000 and text_tokens + 73728 < 450000, "actual text input over reservation")
    ledger_path = BASE / "n5_execution_20260930_r2/budget_ledger.working.json"
    ledger = runner.read(ledger_path)
    charged = sum(row["charged_usd"] for row in ledger["entries"])
    runner.require(ledger["limit_usd"] == 15 and charged + .15 <= 15, "shared budget unavailable")
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "READY_ONE_DEVELOPMENT_CALL", "request": runner.ref(prepared / "request.http_body.json"),
        "prepare": runner.ref(prepared / "prepare.json"), "expectations": runner.ref(HERE / "expectations.json"),
        "registration": runner.ref(HERE / "REGISTRATION_ZH.md"), "controller": runner.ref(Path(__file__)),
        "text_tokens": text_tokens, "reserved_total_proxy": text_tokens + 73728,
        "text_tokenizer": runner.ref(archive_path), "protocol_data_matches_verified_packet": True,
        "original_materials_match_verified_packet": True, "body_serialization_note":
        "Protocol JSON file serialization differs locally/remotely; its data and nonprotocol blocks match. This count binds the actual remote body.",
        "budget_before": runner.ref(ledger_path), "charged_before_usd": charged, "reserved_usd": .15,
        "model_calls_by_preflight": 0, "semantic_qualification": False}
    output_name = "dispatch.json" if args.execute else "actual_request_preflight.json"
    runner.save(HERE / "final_input" / output_name, record, exclusive=True)
    print(json.dumps({k: record[k] for k in ("status", "text_tokens", "charged_before_usd", "reserved_usd")}), flush=True)
    if not args.execute:
        return
    release = runner.read(prepared / "release.draft.json")
    release.update(status="RELEASED_FOR_ONE_CALL", input_token_reservation=360000, reserved_usd=.15,
        prices_usd_per_million={"cache_hit": .006, "cache_miss": .3, "output": 1.2},
        pricing_source="https://api-docs.deepseek.com/quick_start/pricing/",
        budget={"ledger_file": str(ledger_path), "ledger_sha256": record["budget_before"]["sha256"],
                "budget_id": ledger["budget_id"], "limit_usd": 15})
    release_path = HERE / "final_input/release.json"
    runner.save(release_path, release, exclusive=True)
    receipt = runner.run(prepared, release_path, Path("/data/yuyangliu/secrets/deepseek.env"))
    runner.save(HERE / "final_input/budget_after.json", runner.read(ledger_path), exclusive=True)
    print(json.dumps({k: receipt.get(k) for k in ("status", "complete", "incomplete_reasons", "usage",
        "usage_cost_at_registered_rates_usd", "budget_used_after_usd")}), flush=True)


if __name__ == "__main__":
    main()
