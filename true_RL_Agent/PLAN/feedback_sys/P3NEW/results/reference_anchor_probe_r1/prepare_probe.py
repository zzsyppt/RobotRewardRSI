"""Prepare one old R2 packet for reference-contract development; zero calls."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
FROZEN = BASE / "n5_confirmation_source_r2"
sys.path.insert(0, str(HERE / "source/PLAN/feedback_sys/P3NEW"))
sys.path.insert(1, str(FROZEN / "src"))
import run_confirmation_review as runner


def main():
    old = BASE / "n5_confirmation_20260930_r2/numeric_materials/packets/episode-01/facts.manifest.json"
    old_prepared = BASE / "n5_confirmation_20260930_r2/numeric_materials/prepared/episode-01/deepseek"
    target = HERE / "final_input"
    output = target / "prepared"
    if target.exists():
        raise ValueError("development preparation already exists; preserve it")
    target.mkdir()
    manifest = runner.read(old)
    manifest.update(reference_contract=runner.coverage.ANCHOR_CONTRACT,
                    purpose="development_reference_anchor_probe_r1_no_qualification")
    protocol = HERE / "source/PLAN/feedback_sys/P3NEW/N5_FACT_REVIEW_PROTOCOL_ZH.md"
    original_protocol = next(item for item in manifest["materials"] if item["role"] == "review_protocol")
    original_protocol_path, _ = runner.referenced(old.parent, original_protocol)
    # Preserve packet-specific modality/scope clauses while replacing only the
    # shared protocol. Match the full forty-packet offline verification input.
    runner.save(target / "protocol.json", {**runner.read(original_protocol_path),
        **runner.packet_protocol(protocol), "status": "REFERENCE_ANCHOR_DEVELOPMENT_NOT_FROZEN"}, exclusive=True)
    for item in manifest["materials"]:
        original_path, _ = runner.referenced(old.parent, item)
        item["file"] = str(original_path)
        if item["role"] == "review_protocol":
            item.update(runner.ref(target / "protocol.json"))
    manifest_path = target / "facts.manifest.json"
    runner.save(manifest_path, manifest, exclusive=True)
    card = runner.prepare(manifest_path, "deepseek", output)
    request = runner.read(output / "request.http_body.json")
    previous = runner.read(old_prepared / "request.http_body.json")

    def unchanged_blocks(payload):
        return [block for block in payload["messages"][1]["content"]
                if block["type"] != "text" or json.loads(block["text"]).get("role") not in
                {"review_protocol", "reference_catalog"}]

    if unchanged_blocks(request) != unchanged_blocks(previous):
        raise ValueError("non-protocol original materials differ")
    receipt = {"status": "PREPARED_NO_CALL", "source_manifest": runner.ref(old),
        "old_request": runner.ref(old_prepared / "request.http_body.json"),
        "protocol": runner.ref(protocol), "registration": runner.ref(HERE / "REGISTRATION_ZH.md"),
        "controller": runner.ref(Path(__file__)),
        "prepared": runner.ref(output / "prepare.json"), "reference_catalog": runner.ref(output / "reference_catalog.json"),
        "non_protocol_materials_identical": True, "model_calls": 0,
        "new_request_sha256": card["request_sha256"], "new_request_bytes": card["request_bytes"],
        "runner": card["runner"], "coverage_helper": card["coverage_helper"]}
    runner.save(target / "preparation_receipt.json", receipt, exclusive=True)
    print(json.dumps({k: receipt[k] for k in ("status", "non_protocol_materials_identical", "model_calls", "new_request_bytes")}))


if __name__ == "__main__":
    main()
