"""Offline v2 presentation check on all 40 existing R2 facts packets.

Only this new output directory is written. Original requests, materials and
answers remain untouched. No model API, training, GPU, or sealing is invoked.
"""
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[5]
HERE = ROOT / "PLAN/feedback_sys/P3NEW"
OUT = Path(__file__).resolve().parent
ORIGINAL = HERE / "results/n5_confirmation_execution_r2"
V1 = HERE / "results/reference_anchor_development_r1"
refs, checks = {}, []


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def watch(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    refs[str(path)] = {"file": str(path), "sha256": sha(raw), "bytes": len(raw)}
    return raw


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode())


def check(name, valid):
    checks.append({"check": name, "status": "PASS" if valid else "FAIL"})
    if not valid:
        raise AssertionError(name)


def ordered(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def text_size(request):
    texts = []
    for message in request["messages"]:
        content = message["content"]
        texts.extend([content] if isinstance(content, str) else [block["text"] for block in content if block["type"] == "text"])
    sizes = [len(value.encode()) for value in texts]
    return {"all_message_text_utf8_bytes": sum(sizes), "largest_text_block_utf8_bytes": max(sizes), "text_block_count": len(sizes)}


def image_pairs(request):
    parts = request["messages"][1]["content"]
    return [(parts[index - 1], block) for index, block in enumerate(parts) if block["type"] == "image_url"]


def unsplit_texts(request):
    roles = {"raw_image", "review_protocol", "focus_questions", "own_sealed_facts"}
    return [block for block in request["messages"][1]["content"]
            if block["type"] == "text" and json.loads(block["text"]).get("role") in roles]


def main():
    if (OUT / "receipt.json").exists() or (OUT / "packets").exists():
        raise RuntimeError("refuse overwriting audit output")
    watch(Path(__file__))
    for name in ("run_confirmation_review.py", "confirmation_review_coverage.py", "N5_FACT_REVIEW_PROTOCOL_ZH.md"):
        save(OUT / "executed_source" / name, watch(HERE / name))
    spec = importlib.util.spec_from_file_location("v2_offline_runner", HERE / "run_confirmation_review.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    coverage = runner.coverage
    protocol = runner.packet_protocol(HERE / "N5_FACT_REVIEW_PROTOCOL_ZH.md")
    manifests = sorted((ORIGINAL / "sol_facts").glob("episode-*/facts.manifest.json")) + sorted((ORIGINAL / "sol_facts").glob("focus-*/facts.manifest.json")) + sorted((ORIGINAL / "sol_facts_media_bundled").glob("media-*/facts.manifest.json"))
    check("40 original facts manifests", len(manifests) == 40)
    packets, images = [], []
    for ordinal, path in enumerate(manifests, 1):
        old = json.loads(watch(path))
        pid, base = old["packet_id"], OUT / "packets" / old["packet_id"]
        old_body = watch(path.parent / "prepared/request.http_body.json")
        old_card = json.loads(watch(path.parent / "prepared/prepare.json"))
        for name in ("judge_instruction.txt", "release.draft.json"):
            watch(path.parent / "prepared" / name)
        rebuilt, _, _ = runner.build(path, old_card["judge_id"])
        check(pid + "/legacy HTTP bytes unchanged", rebuilt == old_body and sha(old_body) == old_card["request_sha256"])
        same_material_v1 = copy.deepcopy(old)
        same_material_v1["reference_contract"] = coverage.ANCHOR_CONTRACT
        for item in same_material_v1["materials"]:
            original_path = (path.parent / item["file"]).resolve()
            raw = watch(original_path)
            check(pid + "/input " + item["id"], sha(raw) == item["sha256"])
            if item["role"] == "review_protocol":
                current = {**json.loads(raw), **protocol, "status": "REFERENCE_BLOCK_DEVELOPMENT_NOT_FROZEN"}
                updated = base / "review_protocol.json"
                save(updated, current)
                item.update(file=str(updated.resolve()), sha256=sha(updated.read_bytes()))
            else:
                item["file"] = str(original_path)
        v1_manifest = base / "same_material_v1.manifest.json"
        save(v1_manifest, same_material_v1)
        new = {**same_material_v1, "reference_contract": coverage.BLOCK_CONTRACT}
        v2_manifest = base / "facts.manifest.json"
        save(v2_manifest, new)
        v1_body, _, _ = runner.build(v1_manifest, "deepseek")
        save(base / "same_material_v1.request.http_body.json", v1_body)
        v1_request = json.loads(v1_body)
        requests, cards = {}, {}
        for judge in ("deepseek", "gpt6-sol"):
            prepared = base / judge / "prepared"
            cards[judge] = runner.prepare(v2_manifest, judge, prepared)
            requests[judge] = json.loads((prepared / "request.http_body.json").read_bytes())
        request = requests["deepseek"]
        check(pid + "/two reviewers receive identical user blocks", request["messages"][1] == requests["gpt6-sol"]["messages"][1])
        v1_materials, v1_inventory = coverage.request_materials(v1_request)
        v2_materials, v2_inventory = coverage.request_materials(request)
        check(pid + "/all reconstructed values types and original key order", ordered(v1_materials) == ordered(v2_materials) and v1_inventory == v2_inventory is None)
        catalog = coverage.request_reference_catalog(request)
        archived_catalog = json.loads(watch(V1 / "packets" / pid / "deepseek/prepared/reference_catalog.json"))
        check(pid + "/same current v1 and archived v1 catalog identities", catalog == coverage.request_reference_catalog(v1_request) == archived_catalog)
        check(pid + "/excluded messages remain byte identical", unsplit_texts(request) == unsplit_texts(v1_request))
        check(pid + "/PNG blocks and preceding original labels retain byte order", image_pairs(request) == image_pairs(v1_request) == image_pairs(json.loads(old_body)))
        expected_images = [item for item in new["materials"] if item["role"] == "raw_image"]
        emitted = image_pairs(request)
        check(pid + "/PNG counts", len(emitted) == len(expected_images) == cards["deepseek"]["png_count"])
        for (_, block), item in zip(emitted, expected_images):
            raw = base64.b64decode(block["image_url"]["url"].split(",", 1)[1])
            check(pid + "/unchanged PNG " + item["id"], raw == Path(item["file"]).read_bytes())
            images.append({"packet_id": pid, "material_id": item["id"], "sha256": sha(raw)})
        material_blocks = [json.loads(block["text"]) for block in request["messages"][1]["content"]
                           if block["type"] == "text" and json.loads(block["text"]).get("role") == "reference_material_block"]
        for block in material_blocks:
            locator = coverage.normalize_reference({"anchor_id": block["anchor_id"]}, v2_materials, catalog)
            check(pid + "/anchor adjacent value " + block["anchor_id"], locator == block["locator"] and ordered(coverage.resolve_reference(locator, v2_materials)) == ordered(block["value"]))
        record = {"packet_id": pid, "old_manifest": refs[str(path.resolve())],
                  "old_request": refs[str((path.parent / "prepared/request.http_body.json").resolve())],
                  "v2_manifest": runner.ref(v2_manifest), "same_material_v1_request": runner.ref(base / "same_material_v1.request.http_body.json"),
                  "all_checks_passed": True, "catalog_items": len(catalog["items"]), "value_blocks": len(material_blocks), "png_count": len(emitted),
                  "catalog_canonical_sha256": sha(coverage.canonical(catalog)),
                  "sizes": {"original_legacy": {"request_bytes": len(old_body), **text_size(json.loads(old_body))},
                            "same_material_v1": {"request_bytes": len(v1_body), **text_size(v1_request)},
                            "v2": {"request_bytes": cards["deepseek"]["request_bytes"], **text_size(request)}},
                  "requests": {judge: {"sha256": card["request_sha256"], "bytes": card["request_bytes"]} for judge, card in cards.items()}}
        save(base / "comparison.json", record)
        packets.append(record)
        if ordinal % 10 == 0:
            print(json.dumps({"completed_packets": ordinal, "original_png_count": len(images)}), flush=True)
    check("48 original PNGs", len(images) == 48)
    check("all read input files remain unchanged", all(sha(Path(path).read_bytes()) == value["sha256"] for path, value in refs.items()))
    maxima = {kind: {field: {"packet_id": max(packets, key=lambda row: row["sizes"][kind][field])["packet_id"],
                            "value": max(row["sizes"][kind][field] for row in packets)}
                     for field in ("request_bytes", "all_message_text_utf8_bytes", "largest_text_block_utf8_bytes")}
              for kind in ("original_legacy", "same_material_v1", "v2")}
    result = {"schema_version": "p3new.reference_blocks_development.v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "PASS_ENGINEERING_ONLY", "packet_count": len(packets), "legacy_http_bodies_byte_identical": len(packets),
              "png_count": len(images), "png_unique_sha256": len({item["sha256"] for item in images}),
              "all_reconstructed_values_types_and_order_unchanged": True, "all_v1_catalog_ids_unchanged": True,
              "all_reviewers_user_blocks_identical": True, "all_original_inputs_unchanged": True,
              "size_maxima": maxima, "input_refs": list(refs.values()), "packets": packets, "checks": checks,
              "model_calls": 0, "gpu_calls": 0, "semantic_score_assigned": False, "original_answer_edited": False,
              "facts_seal_created": False, "token_measurement": "NOT_MEASURED_IN_THIS_OFFLINE_ENCODING_CHECK",
              "limits": ["Facts carrier round trips only; report boundary cases are covered separately by synthetic unit tests.",
                         "Canonical data SHA verifies reconstructed JSON content, not original-file bytes or semantic support.",
                         "The current protocol is shared by the v1/v2 encoding comparison; historic protocol and HTTP artifacts remain unchanged.",
                         "No evidence of improved model compliance, semantic accuracy, necessity or completeness is claimed."],
              "python_executable": sys.executable}
    save(OUT / "receipt.json", result)
    print(json.dumps({key: result[key] for key in ("status", "packet_count", "png_count", "size_maxima")}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
