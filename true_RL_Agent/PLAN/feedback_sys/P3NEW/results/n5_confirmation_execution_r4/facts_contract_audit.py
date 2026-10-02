"""Read-only R4 facts structure audit; never a semantic score, seal, or release.

The frozen protocol is prose, not a newly invented JSON schema. Slash notation
accepts its literal key or the corresponding separate keys. Explicit null is
preserved for inapplicable fields; deciding applicability remains manual.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parent
P = E.parent / "n5_confirmation_source_r4/source/PLAN/feedback_sys/P3NEW"
PROTOCOL = P / "N5_FACT_REVIEW_PROTOCOL_ZH.md"
HELPER = P / "confirmation_review_coverage.py"
FINDINGS_FIELDS = ("unit_id", "materials_seen", "coverage", "necessary_facts",
                   "major_visible_forms", "unknowns", "missing_materials")
FACT_FIELDS = ("fact_id", "unit_id", "question_bundle", "statement", "body",
               "modality", "predicate", "polarity", "scope", "evidence_refs",
               "necessary_reason")


def ref(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "size_bytes": len(raw)}


def audit_contract(prepared_path, answer_path):
    """Inspect only the explicitly supplied completed answer and original input."""
    prepared_path, answer_path = Path(prepared_path), Path(answer_path)
    request_path = prepared_path.parent / "request.http_body.json"
    prepared = json.loads(prepared_path.read_bytes())
    request = json.loads(request_path.read_bytes())
    answer = json.loads(answer_path.read_bytes())
    errors, notices, nullable = [], [], []

    def error(issue, locator="", **detail):
        errors.append({"issue": issue, "locator": locator, **detail})

    request_ref, helper_ref = ref(request_path), ref(HELPER)
    if prepared.get("stage") != "facts" or prepared.get("capacity_facts_fixture"):
        error("requires_original_nonfixture_facts_prepare")
    if request_ref["sha256"] != prepared.get("request_sha256") or request_ref["size_bytes"] != prepared.get("request_bytes"):
        error("request_differs_from_prepared_bytes")
    if helper_ref["sha256"] != prepared.get("coverage_helper", {}).get("sha256"):
        error("frozen_coverage_helper_differs_from_prepared")
    if errors:
        raise ValueError(errors)
    spec = importlib.util.spec_from_file_location("r4_frozen_facts_coverage", HELPER)
    coverage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coverage)
    materials, _ = coverage.request_materials(request)
    units = {m["data"]["unit_id"] for m in materials.values()
             if m.get("role") == "focus_questions" and isinstance(m.get("data"), dict)
             and "unit_id" in m["data"]}
    expected_unit = next(iter(units)) if len(units) == 1 else prepared["packet_id"]
    if not isinstance(answer, dict):
        error("answer_not_object")
        answer = {}
    for key, expected in (("judge_id", prepared["judge_id"]),
                          ("packet_id", prepared["packet_id"]), ("stage", "facts")):
        if answer.get(key) != expected:
            error("envelope_identity_differs", "/" + key)
    if answer.get("complete") is not True:
        error("answer_not_declared_complete", "/complete")
    findings = answer.get("findings")
    if not isinstance(findings, dict):
        error("findings_not_object", "/findings")
        findings = {}
    for key in FINDINGS_FIELDS:
        if key not in findings:
            error("findings_field_missing", "/findings/" + key)
    if findings.get("unit_id") != expected_unit:
        error("findings_unit_differs_from_supplied_questions", "/findings/unit_id")
    for key in ("necessary_facts", "major_visible_forms", "unknowns", "missing_materials"):
        if key in findings and not isinstance(findings[key], list):
            error("declared_array_not_list", "/findings/" + key)
    facts = findings.get("necessary_facts")
    if not isinstance(facts, list):
        facts = []
    if not facts:
        error("zero_necessary_facts", "/findings/necessary_facts")
    seen = set()
    for i, fact in enumerate(facts):
        loc = f"/findings/necessary_facts/{i}"
        if not isinstance(fact, dict):
            error("necessary_fact_not_object", loc)
            continue
        for key in FACT_FIELDS:
            if key not in fact:
                error("necessary_fact_field_missing", loc + "/" + key)
        # interval OR an event anchor can locate a fact; both ratio operands are
        # required when the numerator/denominator notation is split into keys.
        if not any(key in fact for key in ("interval/event_anchor", "interval", "event_anchor")):
            error("necessary_fact_location_field_missing", loc)
        if "numerator/denominator" not in fact and not {"numerator", "denominator"} <= fact.keys():
            error("necessary_fact_ratio_fields_missing", loc)
        fid = fact.get("fact_id")
        if not isinstance(fid, str) or not fid.strip() or fid in seen:
            error("invalid_or_duplicate_fact_id", loc + "/fact_id")
        else:
            seen.add(fid)
        if fact.get("unit_id") != expected_unit:
            error("fact_unit_differs_from_supplied_questions", loc + "/unit_id")
        for key in ("statement", "necessary_reason"):
            if not isinstance(fact.get(key), str) or not fact[key].strip():
                error("necessary_fact_text_missing", loc + "/" + key)
        if fact.get("evidence_refs") == []:
            notices.append({"issue": "empty_fact_evidence_requires_manual_applicability_review", "locator": loc + "/evidence_refs"})
        nullable.extend(loc + "/" + k for k, v in fact.items() if v is None)
    reference_audit = coverage.audit_answer(answer, request, stage="facts")
    if not reference_audit.get("review_complete"):
        error("frozen_reference_audit_incomplete")
    media_present = any(m.get("role") == "raw_image" for m in materials.values())
    if not media_present and isinstance(findings.get("major_visible_forms"), list) and findings["major_visible_forms"]:
        error("nonempty_visible_forms_without_supplied_media", "/findings/major_visible_forms")
    return {
        "schema_version": "p3new.r4_facts_contract_audit.v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "scope": "n5_confirmation_readonly_structure_not_semantic_acceptance",
        "inputs": {"prepared": ref(prepared_path), "request": request_ref,
                   "answer": ref(answer_path), "protocol": ref(PROTOCOL), "coverage_helper": helper_ref},
        "judge_id": prepared["judge_id"], "packet_id": prepared["packet_id"], "stage": "facts",
        "protocol_lines": {"zero_facts": 50, "fact_fields": 57, "mechanical_limits": 106,
                           "catalog": 111, "findings_and_missing_media": 112, "complete": 141},
        "necessary_fact_count": len(facts), "unique_nonempty_fact_ids": len(seen),
        "media_present": media_present, "errors": errors, "notices": notices,
        "explicit_null_field_locators": nullable,
        "frozen_reference_audit": reference_audit,
        "structural_contract_complete": not errors,
        "manual_review_remaining": [
            "Five question bundles and relevant episode scope actually addressed, without a prescribed fact count per bundle.",
            "Coverage/materials_seen/unknowns/missing_materials accurately declare missing observations and unread scope.",
            "With no media, the answer explicitly explains that media was not supplied; no fixed phrase is imposed.",
            "Null fields, evidence, necessity, polarity and process localization are appropriate and semantically supported."],
        "semantic_support_proven": False, "global_80_facts_barrier_proven": False,
        "creates_seal_or_release": False, "model_calls": 0,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", required=True, type=Path, help="Original prepare.json (request alongside)")
    ap.add_argument("--answer", required=True, type=Path, help="Original raw_answer.json; never rewritten")
    ap.add_argument("--output", required=True, type=Path, help="New audit JSON, exclusive creation")
    args = ap.parse_args()
    result = audit_contract(args.prepared, args.answer)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "structural_contract_complete": result["structural_contract_complete"],
                      "necessary_fact_count": result["necessary_fact_count"], "error_count": len(result["errors"])}))
    return 0 if result["structural_contract_complete"] else 2


if __name__ == "__main__":
    sys.exit(main())
