"""Check saved P1c HTTP bytes, original claims, frozen E1 layers and failures."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior
from true_rl_agent.tasks.stair_climbing.trajectory_report import (
    build_stair_trajectory_report, project_trajectory_card, require_trajectory_phase_binding,
)
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resign(value):
    value["report_projection"].pop("content_hash", None)
    value["report_projection"]["content_hash"] = canonical_json_hash(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, required=True)
    args = parser.parse_args()
    folder = args.capture
    full, receipt = read(folder / "behavior_full.json"), read(folder / "capture_receipt.json")
    public, audit = full["public_report"], full["audit"]
    assert project_stair_behavior(full) == public
    card_manifest = read(receipt["card_reference"]["manifest_path"])
    card = read(Path(receipt["card_reference"]["manifest_path"]).parent / "policy_card.json")
    original = {c["claim_id"]: c for c in card["claims"]}
    assert set(original) == set(audit["claim_trace"])
    assert all(c == audit["claim_trace"][cid]["claim"] for cid, c in original.items())
    included = set(audit["included_claim_ids"])
    omitted = {r["claim_id"] for r in audit["omitted_claims"]}
    assert not included & omitted and included | omitted == set(original)
    assert set(audit["required_claim_ids"]) <= included
    rows = {r["id"]: r for s in SENTENCE_SECTIONS for r in public["report"][s]}
    assert all(rows[cid]["text"].endswith(original[cid]["text"]) for cid in included)
    frames = []
    for frame in receipt["frames"]:
        body = Path(frame["http_body"]["path"]).read_bytes()
        user = Path(frame["user_payload"]["path"]).read_bytes()
        assert json.loads(body)["messages"][-1]["content"].encode("utf-8") == user
        assert json.loads(user)["public_feedback_summary"]["behavior"]["profile"] == public
        for kind in ("http_body", "user_payload"):
            ref = frame[kind]
            assert file_sha256(Path(ref["path"])) == ref["sha256"]
            assert Path(ref["path"]).stat().st_size == ref["size_bytes"]
        frames.append({"frame": frame["frame"], "report_exact": True, "bytes_and_hashes_exact": True})

    # Concrete claim -> operand -> pinned P1a byte locator / embedded numeric
    # record mapping. The detailed records contain the original raw references.
    records = json.loads(audit["derivation_records_json"])
    mapping = []
    for cid, claim in original.items():
        operands = []
        for slot, op in claim["operands"].items():
            rid = op["record_id"]
            assert rid in records
            link = audit["evidence_links"].get(rid)
            operands.append({"slot": slot, **op, "p1a_byte_locator": link,
                "source_file": audit["source_files"][link["file_id"]] if link else None,
                "embedded_record": {"artifact": "projection_audit.json", "field": "derivation_records_json", "record_id": rid},
                "raw_reference_note": "Follow this record's evidence_ref/metric evidence_refs and scope; primary cohort member IDs bind the denominator."})
        mapping.append({"claim_id": cid, "included_in_author": cid in included,
            "original_text": claim["text"], "public_row": rows.get(cid), "scope": claim["scope"],
            "scope_record_id": claim["scope_record_id"], "source_manifest_sha256": audit["claim_trace"][cid]["source_manifest_sha256"],
            "operands": operands})
    write(folder / "claim_to_evidence.json", mapping)

    raw, frozen = read(args.raw_audit), read(args.frozen)
    assert raw["passed"] and frozen["passed"]
    assert raw["card_manifest_sha256"] == receipt["card_reference"]["manifest_sha256"]
    assert frozen["reports"]["necessary"]["counts"] == {"pass": 247}
    assert frozen["reports"]["full"]["counts"] == {"pass": 309}
    selection = read(ROOT / "PLAN/feedback_sys/NEW_SYS/P0/e1_selection.json")
    related = []
    for item in raw["necessary_coverage"]:
        assert set(item["claim_ids"]) <= set(original)
        author_ids = sorted(set(item["claim_ids"]) & included)
        related.append({"fact_id": item["fact_id"], "complete_audit_claim_ids": item["claim_ids"],
            "author_related_claim_ids": author_ids,
            "author_scope": "related_claim_only_not_all_fact_fields" if author_ids else "audit_only_not_in_author_prompt",
            "complete_audit_status": item["status"], "detail_note": item["detail_note"]})
    cases = []
    for episode in selection["episodes"]:
        eid = episode["episode_id"]
        facts = [r for r in related if r["fact_id"].startswith(eid + ".")]
        expanded = [r for r in frozen["reports"]["full"]["facts"] if r["fact_id"].startswith(eid + ".")]
        assert facts and expanded
        cases.append({"episode_id": eid, "source_id": episode["source_id"],
            "necessary_facts": len(facts), "full_facts": len(expanded),
            "necessary_facts_with_related_author_claim": sum(bool(r["author_related_claim_ids"]) for r in facts),
            "author_claim_ids_for_this_episode": [cid for cid in included if original[cid]["scope"] == eid],
            "full_audit_present": True, "author_has_all_facts": False})
    coverage = {"schema_version": "stair.p1c.e1_delivery_coverage.v1", "selection_unchanged": True,
        "selection_sha256": file_sha256(ROOT / "PLAN/feedback_sys/NEW_SYS/P0/e1_selection.json"),
        "necessary_gate": {"pass": 247, "total": 247}, "full_gate": {"pass": 309, "total": 309},
        "author_claims": len(included), "complete_audit_claims": len(original),
        "author_has_all_309_facts": False, "automatic_author_audit_retrieval": False,
        "cases": cases, "necessary_fact_to_claims": related,
        "full_fact_record_map": [{k: row[k] for k in ("fact_id", "status", "record_ids")} for row in frozen["reports"]["full"]["facts"]],
        "raw_audit_sha256": file_sha256(args.raw_audit), "frozen_verification_sha256": file_sha256(args.frozen)}
    write(folder / "e1_delivery_coverage.json", coverage)

    failures = []
    def rejects(name, fn):
        try:
            fn()
        except ValueError as error:
            failures.append({"name": name, "status": "REJECTED", "error": str(error)})
        else:
            raise AssertionError("negative case accepted: " + name)

    phase = read(folder / "public_dev_result.json")
    numeric = phase["task_evidence"]["numeric_evidence"]
    identity = load_numeric_evidence_manifest(numeric)["identity"]
    for key in ("job_id", "checkpoint_hash", "task_id", "seed_namespace", "evaluation_plan_hash", "evaluation_binding", "splits"):
        bad = {**identity, key: "different-source"}
        rejects("pinned_loader_" + key, lambda: build_stair_trajectory_report(receipt["card_reference"], numeric_reference=numeric, expected_identity=bad))
    rejects("pinned_card_manifest_hash", lambda: build_stair_trajectory_report(
        {**receipt["card_reference"], "manifest_sha256": "0"*64}, numeric_reference=numeric, expected_identity=identity))
    for key in ("checkpoint_hash", "evaluation_job_id", "seed_namespace", "projection_hash", "evaluation_plan_hash"):
        bad = deepcopy(full)
        old = bad["public_report"]["domain"][key]
        bad["public_report"]["domain"][key] = ("0" if old[0] != "0" else "1") + old[1:]
        resign(bad["public_report"])
        rejects("full_source_" + key, lambda: project_stair_behavior(bad))
    # Even editing the copied source_domain cannot override the original P1b source binding.
    bad = deepcopy(full)
    bad["public_report"]["domain"]["projection_hash"] = "0"*64
    bad["audit"]["source_domain"]["projection_hash"] = "0"*64
    resign(bad["public_report"])
    rejects("full_primary_source_projection", lambda: project_stair_behavior(bad))
    tight = audit["required_report_bytes"]
    rejects("mandatory_one_byte_short", lambda: project_trajectory_card(card, records,
        domain=public["domain"], card_hash=public["card_id"][5:], max_bytes=tight-1))
    write(folder / "negative_cases.json", failures)
    result = {"status": "PASS", "frames": frames, "included_claims": len(included),
        "omitted_claims": len(omitted), "report_bytes": report_bytes(public), "e1_cases": len(cases),
        "negative_cases_rejected": len(failures), "author_has_all_309_facts": False,
        "source_card_bytes": card_manifest["artifacts"]["policy_card.json"]["bytes"]}
    write(folder / "delivery_audit_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
