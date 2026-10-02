"""Seal completed r2 answers and compare labels; no provider or scoring calls."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


BASE = Path(__file__).resolve().parent
EXPECTED_REMOTE = {
    "deepseek_receipt.json": "a94e27935245fbe6eeb08dab0f16975ad80f03315b942e2149e489fdb01b5d16",
    "deepseek_response.txt": "2cadb73aa85db7b1aceda70d6773ba57f922c3711ccf913c89158901f4538c85",
    "deepseek_seal.json": "3385be8828e9db27a31d8448b648effc8c9e5c67c4bc18435fb9cce7eab7b808",
}
EXPECTED_GPT = "817ce9c65cb1e7b04d7bc3f0e925a98416515dda7f290a03c9f76dcb7be0ffbc"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name: str):
    return json.loads((BASE / name).read_bytes())


def save_new(name: str, value) -> None:
    with (BASE / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    # Check provenance and frozen inputs before looking at either label sequence.
    actual_remote = {name: digest(BASE / name) for name in EXPECTED_REMOTE}
    assert actual_remote == EXPECTED_REMOTE
    assert digest(BASE / "gpt6_sol_response.json") == EXPECTED_GPT
    manifest = read("manifest.json")
    release = read("release.json")
    assert release["status"] == "FROZEN_CALIBRATION_RELEASED"
    assert digest(BASE / "manifest.json") == release["material_manifest_sha256"]
    assert digest(BASE / "request.http_body.json") == release["request_sha256"]
    material_hashes = {}
    for entry in manifest["allowlist"]:
        source = BASE / "materials" / entry["file"]
        assert source.stat().st_size == entry["bytes"]
        material_hashes[entry["file"]] = digest(source)
        assert material_hashes[entry["file"]] == entry["sha256"]
    request = read("request.http_body.json")
    supplied = json.loads(request["messages"][1]["content"])
    for name in material_hashes:
        assert supplied[name] == read("materials/" + name)
    rubric = read("materials/rubric.json")
    inputs = read("materials/items.json")
    ids = [item["id"] for item in inputs]
    assert len(ids) == len(set(ids)) == manifest["item_count"] == 20
    receipt = read("deepseek_receipt.json")
    ds_seal = read("deepseek_seal.json")
    assert receipt["status"] == "received" and receipt["calls"] == 1
    assert receipt["request_sha256"] == release["request_sha256"]
    assert receipt["model_matches"] and receipt["fingerprint_matches"]
    assert receipt["input_within_reservation"]
    assert receipt["response"]["choices"][0]["finish_reason"] == "stop"
    assert (BASE / "deepseek_response.txt").read_bytes() == receipt["response"]["choices"][0]["message"]["content"].encode("utf-8")
    assert ds_seal["answer_sha256"] == EXPECTED_REMOTE["deepseek_response.txt"]
    assert ds_seal["receipt_sha256"] == EXPECTED_REMOTE["deepseek_receipt.json"]
    usage = receipt["response"]["usage"]
    prices = release["prices_usd_per_million"]
    api_ceiling = (
        usage["prompt_cache_hit_tokens"] * prices["cache_hit"]
        + usage["prompt_cache_miss_tokens"] * prices["cache_miss"]
        + usage["completion_tokens"] * prices["output"]
    ) / 1_000_000
    assert abs(api_ceiling - 0.0360321) < 1e-12
    assert abs(api_ceiling - receipt["peak_cost_ceiling_usd"]) < 1e-12

    gpt = read("gpt6_sol_response.json")
    gpt_ids = [item["id"] for item in gpt["items"]]
    assert len(gpt_ids) == len(set(gpt_ids))
    save_new("gpt6_sol_seal.json", {
        "json_valid": True,
        "item_count": len(gpt_ids),
        "ids_equal": gpt_ids == ids,
        "answer_sha256": EXPECTED_GPT,
        "registration_sha256": digest(BASE / "gpt_reviewer_registration.json"),
        "compared_with_other_reviewer": False,
        "sealed_utc": now(),
        "actual_cost": "unavailable; not zero",
        "actual_fingerprint": "unavailable",
        "api_calls_during_sealing": 0,
    })
    save_new("transfer_verification.json", {
        "verified_utc": now(),
        "remote_host": "humanoid-server",
        "remote_directory": "/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1/PLAN/feedback_sys/P3NEW/review_calibration_r2",
        "remote_sha256": EXPECTED_REMOTE,
        "local_sha256": actual_remote,
        "all_transferred_bytes_match": True,
        "frozen_manifest_matches_release": True,
        "frozen_request_matches_release": True,
        "frozen_material_sha256": material_hashes,
        "request_material_objects_match_frozen_files": True,
        "deepseek_content_matches_saved_answer_exactly": True,
        "api_calls_during_sealing_or_comparison": 0,
    })

    # Both original files and their seal cards now exist. Only equality is scored.
    ds = read("deepseek_response.txt")
    ds_ids = [item["id"] for item in ds["items"]]
    assert len(ds_ids) == len(set(ds_ids))
    assert set(ds_ids) <= set(ids) and set(gpt_ids) <= set(ids)
    ds_by_id = {item["id"]: item for item in ds["items"]}
    gpt_by_id = {item["id"]: item for item in gpt["items"]}
    rows = []
    for item_id in ids:
        d = ds_by_id.get(item_id, {})
        g = gpt_by_id.get(item_id, {})
        rows.append({
            "id": item_id,
            "deepseek_kind": d.get("kind"),
            "gpt_kind": g.get("kind"),
            "deepseek_verdict": d.get("verdict"),
            "gpt_verdict": g.get("verdict"),
            "verdict_agrees": bool(d.get("verdict") and g.get("verdict") and d["verdict"] == g["verdict"]),
            "kind_agrees": bool(d.get("kind") and g.get("kind") and d["kind"] == g["kind"]),
        })
    verdict_n = sum(row["verdict_agrees"] for row in rows)
    kind_n = sum(row["kind_agrees"] for row in rows)
    n = len(ids)
    minimum = rubric["agreement"]["minimum_verdict_agreement"]
    comparison = {
        "status": "RAW_CALIBRATION_COMPARISON_NOT_ADJUDICATED",
        "compared_utc": now(),
        "formal_confirmation": False,
        "originals_sealed_before_comparison": True,
        "gpt_sha256": EXPECTED_GPT,
        "deepseek_sha256": EXPECTED_REMOTE["deepseek_response.txt"],
        "n": n,
        "verdict_agreement": verdict_n / n,
        "kind_agreement": kind_n / n,
        "verdict_agreement_count": verdict_n,
        "kind_agreement_count": kind_n,
        "verdict_disagreement_ids": [r["id"] for r in rows if not r["verdict_agrees"]],
        "kind_disagreement_ids": [r["id"] for r in rows if not r["kind_agrees"]],
        "minimum_verdict_agreement": minimum,
        "verdict_agreement_gate_passed": verdict_n / n >= minimum,
        "missing_response_counts_as_disagreement": True,
        "r1_raw_verdict_agreement_retained": 0.7,
        "revision_limit": manifest["revision_limit"],
        "peak_cost_ceiling_usd": api_ceiling,
        "cost_is_invoice": False,
        "gpt_backend_cost": "unavailable; not zero",
        "api_calls_during_sealing_or_comparison": 0,
        "rows": rows,
    }
    save_new("raw_comparison.json", comparison)
    # Recheck originals and frozen inputs after all outputs have been written.
    assert digest(BASE / "gpt6_sol_response.json") == EXPECTED_GPT
    assert {name: digest(BASE / name) for name in EXPECTED_REMOTE} == EXPECTED_REMOTE
    assert digest(BASE / "manifest.json") == release["material_manifest_sha256"]
    assert digest(BASE / "request.http_body.json") == release["request_sha256"]
    print(json.dumps({key: value for key, value in comparison.items() if key != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
