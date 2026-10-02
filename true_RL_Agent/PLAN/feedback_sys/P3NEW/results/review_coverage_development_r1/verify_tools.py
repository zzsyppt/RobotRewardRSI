"""CPU-only tool acceptance using old evidence and explicitly synthetic answers/cards."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

BASE = Path(__file__).resolve().parents[2]
HERE = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(BASE))
import confirmation_review_coverage as coverage
import prepare_confirmation_materials as packer
import run_confirmation_review as runner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = {}
    old = BASE / "results/semantic_execution_scope_r2"
    old_request, old_answer = old / "inputs/request.http_body.json", old / "output/raw_answer.json"
    watched = {str(p): sha(p) for p in (old_request, old_answer)}
    request = runner.read(old_request)
    original_audit = coverage.audit_answer(runner.read(old_answer), request, stage="report")
    runner.save(HERE / "old_r2_mechanical_audit.json", original_audit, exclusive=True)
    checks["old_complete_true_does_not_bypass_missing_coverage_or_bad_pointers"] = not original_audit["review_complete"] and bool(original_audit["missing_inventory_ids"]) and any(isinstance(e, dict) and e.get("issue") == "unresolvable_reference" for e in original_audit["errors"])
    prepared = HERE / "prepared_old_r2"
    card = runner.prepare(old / "report.manifest.json", "gpt6-sol", prepared,
        BASE / "results/capacity_actual_facts_r1/gpt6-sol_episode-02_facts.seal.json")
    new = runner.read(prepared / "request.http_body.json")
    materials, inventory = coverage.request_materials(new)
    old_materials, _ = coverage.request_materials(request)
    checks["all_original_material_envelopes_preserved"] = old_materials == materials
    checks["inventory_deterministic_and_all_originals_resolve"] = inventory == coverage.build_inventory(materials) and all(coverage.resolve_reference(i["locator"], materials) == i["original"] for i in inventory["items"])
    checks["no_formal_event_scalar_claim_expansion"] = len(inventory["items"]) == 39 and not any("/events/" in i["locator"]["json_pointer"] for i in inventory["items"])
    answer = {"judge_id": "gpt6-sol", "packet_id": "episode-02", "stage": "report", "complete": True,
              "synthetic_only": True, "findings": {"claims": [], "inventory_review": [], "fact_matches": [
                  {"sealed_fact_id": fact["fact_id"], "status": "missing", "reason": "SYNTHETIC cardinality only"}
                  for fact in materials["own_sealed_facts"]["data"]["findings"]["necessary_facts"]]}}
    for i, item in enumerate(inventory["items"]):
        cid = f"synthetic-{i}"
        answer["findings"]["claims"].append({"claim_id": cid, "report_locator": item["locator"],
            "verdict": "insufficient", "reason": "SYNTHETIC format check, no semantic judgment", "evidence_refs": []})
        answer["findings"]["inventory_review"].append({"inventory_id": item["inventory_id"], "status": "reviewed",
            "claim_ids": [cid], "reason": "SYNTHETIC complete format only; verdict deliberately insufficient"})
    passed = coverage.audit_answer(answer, new, stage="report")
    checks["synthetic_mechanical_complete_does_not_claim_semantic_support"] = passed["review_complete"] and passed["semantic_support_proven"] is False
    for name in ("missing_item", "duplicate_item", "unreviewed_item", "bad_union_pointer", "wrong_evidence_hash", "missing_fact_match", "other_unreviewed"):
        changed = deepcopy(answer)
        f = changed["findings"]
        if name == "missing_item": f["inventory_review"].pop()
        elif name == "duplicate_item": f["inventory_review"].append(deepcopy(f["inventory_review"][0]))
        elif name == "unreviewed_item": f["inventory_review"][0].update(status="unreviewed", claim_ids=[])
        elif name == "bad_union_pointer": f["claims"][0]["report_locator"] = {**inventory["items"][0]["locator"], "json_pointer": "/data/shared/{semantics,contract}"}
        elif name == "missing_fact_match": f["fact_matches"].pop()
        elif name == "other_unreviewed": f["unreviewed_locators"] = [inventory["items"][0]["locator"]]
        else: f["claims"][0]["evidence_refs"] = [{"material_id": "related-raw", "sha256": "0"*64, "json_pointer": "/data/clock"}]
        result = coverage.audit_answer(changed, new, stage="report")
        checks[name + "_prevents_complete"] = not result["review_complete"]
        if name == "unreviewed_item": checks["explicit_unreviewed_is_valid_record_not_missing"] = result["mechanical_valid"]
    coverage_counts = []
    source = BASE / "results/n5_review_capacity_development_r2"
    for p in sorted((source / "prepared_report_capacity").glob("*/gpt6-sol/request.http_body.json")):
        ms, _ = coverage.request_materials(runner.read(p))
        inv = coverage.build_inventory(ms)
        assert all(coverage.resolve_reference(i["locator"], ms) == i["original"] for i in inv["items"])
        coverage_counts.append({"packet": p.parent.parent.name, "items": len(inv["items"])})
    checks["all_36_old_episode_and_focus_inventory_references_resolve"] = len(coverage_counts) == 36
    facts_path = source / "packets/episode-02/facts.manifest.json"
    facts_body, _, facts_card = runner.build(facts_path, "gpt6-sol")
    fm, fi = coverage.request_materials(json.loads(facts_body))
    checks["facts_have_no_report_inventory_or_arithmetic"] = fi is None and facts_card["review_text_inventory"] is None and not any(m["role"] in coverage.REPORT_TEXT_ROLES | {"arithmetic_audit"} for m in fm.values())
    # Actual old raw files, copied only to a bounded temporary development fixture.
    # These fake confirmation labels exercise preparation code, never cohort admission.
    with TemporaryDirectory(prefix="synthetic_formal_", dir=HERE) as temp:
        temp = Path(temp).resolve()
        assert temp.is_relative_to(HERE.resolve())
        raw = temp / "raw"
        shutil.copytree(BASE / "results/raw_export_development_related_v1", raw)
        protocol = temp / "protocol.md"
        shutil.copyfile(BASE / "N5_FACT_REVIEW_PROTOCOL_ZH.md", protocol)
        capture = temp / "capture.json"
        runner.save(capture, {"status": "FROZEN", "synthetic_only": True, "review_protocol": {"sha256": sha(protocol)}})
        manifest = runner.read(raw / "manifest.json")
        manifest.update(purpose="confirmation", frozen_inputs={"capture_card": {"sha256": sha(capture)}, "review_protocol": {"sha256": sha(protocol)}})
        runner.save(raw / "manifest.json", manifest)
        output = temp / "prepared"
        packer.pack(raw, output, protocol, capture, profile="related_v1", max_output_tokens=32768)
        protocol_view = runner.read(output / "materials/review_protocol.json")
        formal_receipt = runner.read(output / "packing_receipt.json")
        checks["synthetic_formal_pack_uses_frozen_body_and_status"] = formal_receipt["purpose"] == "confirmation" and len(formal_receipt["packets"]) == 36 and "operative_protocol" in protocol_view and "operative_draft" not in protocol_view and protocol_view["status"] == "FROZEN_FOR_CONFIRMATION_NOT_RELEASED"
        report_out = temp / "report.manifest.json"
        full_item = next(m for m in runner.read(old / "report.manifest.json")["materials"] if m["role"] == "full_derivation")
        report = packer.prepare_report_manifest(output / "packets/episode-02/facts.manifest.json", [full_item], report_out)
        facts = runner.read(output / "packets/episode-02/facts.manifest.json")
        checks["formal_report_assembly_keeps_all_facts_and_no_seal_fabrication"] = len(report["materials"]) == len(facts["materials"])+1 and report["facts_manifest_sha256"] == sha(output / "packets/episode-02/facts.manifest.json") and "own_facts_seal" not in report
        try: packer.pack_capacity_reports(raw, output, temp / "unused")
        except ValueError as e: checks["formal_synthetic_capacity_rejected"] = "development-only" in str(e)
        manifest["frozen_inputs"]["review_protocol"]["sha256"] = "0"*64
        runner.save(raw / "manifest.json", manifest)
        try: packer.pack(raw, temp / "rejected", protocol, capture, profile="related_v1")
        except ValueError as e: checks["formal_hash_mismatch_rejected_before_output"] = "exact frozen" in str(e) and not (temp / "rejected").exists()
    checks["old_frozen_input_and_answer_unchanged"] = all(sha(Path(p)) == h for p, h in watched.items())
    result = {"status": "PASS_MECHANICAL_TOOL_CHECKS" if all(checks.values()) else "FAIL", "synthetic_answers_are_not_review_votes": True,
        "checks": checks, "old_inputs": watched, "coverage_counts": coverage_counts,
        "new_prepared_request": {"sha256": card["request_sha256"], "bytes": card["request_bytes"], "inventory": card["review_text_inventory"]},
        "source_sha256": {name: sha(BASE / name) for name in ("run_confirmation_review.py", "prepare_confirmation_materials.py", "confirmation_review_coverage.py")},
        "new_api_calls": 0, "gpu_jobs": 0, "ssh_calls": 0, "formal_cohort_qualified": False}
    runner.save(HERE / "verification.json", result, exclusive=True)
    print(json.dumps({"status": result["status"], "checks": len(checks), "failed": [k for k,v in checks.items() if not v], "receipt_sha256": sha(HERE / "verification.json")}))


if __name__ == "__main__":
    main()
