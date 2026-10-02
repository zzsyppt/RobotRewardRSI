"""Check packet instruction extraction and original evidence preservation offline."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
sys.path.insert(0, str(TOOLS))
# Reuse the already installed same-runtime Pillow; no installation or API.
sys.path.append("C:/Users/lover/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages")
import run_confirmation_review as runner


def main():
    protocol = TOOLS / "N5_FACT_REVIEW_PROTOCOL_ZH.md"
    extracted = runner.packet_protocol(protocol)
    assert len(extracted["operative_rules"]) == 8
    assert "sealed_fact_id" in extracted["operative_rules"]["single_packet"]
    assert "## root" not in json.dumps(extracted)
    packed = HERE / "packed_final"
    actual = runner.read(packed / "materials/review_protocol.json")
    assert all(actual[k] == v for k, v in extracted.items())
    numeric_units = sorted((packed / "packets").glob("*/facts.manifest.json"))
    assert len(numeric_units) == 36
    for path in numeric_units:
        manifest = runner.read(path)
        assert all(row["role"] in runner.RAW_ROLES for row in manifest["materials"])
        for row in manifest["materials"]:
            runner.referenced(path.parent, row)
    old = TOOLS / "results/confirmation_media_rootreview_r1/materials/media-03"
    original = runner.read(old / "facts.manifest.json")
    revised = json.loads(json.dumps(original))
    protocol_path = HERE / "media_protocol.json"
    runner.save(protocol_path, {**extracted, "modality": "media with same-episode full-time related numerics"})
    for item in revised["materials"]:
        if item["role"] == "review_protocol":
            item.update(runner.ref(protocol_path))
        else:
            source = old / item["file"].split("/media-03/", 1)[1]
            assert runner.sha(source.read_bytes()) == item["sha256"]
            item["file"] = str(source)
    manifest_path = HERE / "media_facts.manifest.json"
    runner.save(manifest_path, revised)
    answers = [runner.build(manifest_path, judge) for judge in ("deepseek", "gpt6-sol")]
    payloads = [json.loads(row[0]) for row in answers]
    assert payloads[0]["messages"][1]["content"] == payloads[1]["messages"][1]["content"]
    assert all(row[2]["png_count"] == 12 and row[2]["review_text_inventory"] is None for row in answers)
    receipt = {"status": "MECHANICAL_DEVELOPMENT_PASS_NOT_N5", "api_calls": 0,
        "numeric_packets": len(numeric_units), "judge_rule_blocks": 8,
        "original_media_pngs": 12, "same_media_evidence_for_both_judges": True,
        "all_original_media_sha_preserved": True, "facts_have_no_reports_or_inventory": True,
        "source_refs": [runner.ref(TOOLS / name) for name in (
            "N5_FACT_REVIEW_PROTOCOL_ZH.md", "run_confirmation_review.py",
            "prepare_confirmation_materials.py", "prepare_confirmation_media.py")],
        "media_request_receipts": [row[2] for row in answers],
        "limits": "Old development evidence, offline packing only; not a model reading or semantic/capacity pass."}
    runner.save(HERE / "verification.json", receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k not in {"source_refs", "media_request_receipts"}}))


if __name__ == "__main__":
    main()
