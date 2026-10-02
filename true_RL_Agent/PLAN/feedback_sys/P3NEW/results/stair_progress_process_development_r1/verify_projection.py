"""Reproject the checked historical measurements, including bounded media fixtures."""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from true_rl_agent.evaluation.binding import canonical_json_bytes
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.process_report import derive_stair_process_behavior, project_stair_process_report
from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
from tests.unit.test_task_visual_observation import reviewed_fixture

BASE = ROOT / "PLAN/feedback_sys/P3NEW"
OUT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name):
    spec = importlib.util.spec_from_file_location(name, BASE / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def main():
    receipt = json.loads((OUT / "rebuild_receipt.json").read_bytes())
    behavior = "src/true_rl_agent/tasks/stair_climbing/behavior.py"
    assert sha(ROOT / behavior) == receipt["source_sha256"][behavior]
    assert sha(OUT / "measured_with_progress.json") == receipt["outputs"]["measured_with_progress.json"]["sha256"]
    measured = json.loads((OUT / "measured_with_progress.json").read_bytes())
    full = derive_stair_process_behavior(measured)
    public = project_stair_process_report(full)
    visual, delivery = module("audit_visual_projection_offline"), module("audit_delivery_offline")
    cases = []
    for text_char in ("形", chr(0x20BB7)):
        audit = visual.synthetic_audit("stair", full)
        for row in audit["raw_output"]["episodes"][0]["observations"]:
            row.update(text=text_char * 200, uncertainty=text_char * 120)
        for row in audit["observations"]:
            row["raw_observation"].update(text=text_char * 200, uncertainty=text_char * 120)
        # This is an explicitly synthetic schema/byte fixture, not expert endorsement.
        augmented = attach_stair_visual_observations(full, reviewed_fixture(audit), observer_version="synthetic_capacity_only")
        assert augmented["episodes"] == full["episodes"]
        shown = project_stair_process_report(augmented)
        parent, _ = delivery.synthetic_s4("stair", shown)
        for field in ("process", "counterexample", "gaps"):
            assert parent["report"][field] == public["report"][field]
        displayed = parent["report"]["visual"][0]["text"].count(text_char)
        cases.append({"bytes_per_character": len(text_char.encode()), "text_characters": 200,
            "uncertainty_characters": 120, "all_observations": 3, "displayed_characters": displayed,
            "public_bytes": len(canonical_json_bytes(shown)), "with_parent_bytes": len(canonical_json_bytes(parent)),
            "all_numerics_process_counterexample_gaps_preserved": True,
            "visual_sentence_retained": displayed == 320})
        assert cases[-1]["with_parent_bytes"] <= 8192
    for name, value in (("full.json", full), ("public.json", public)):
        (OUT / name).write_bytes(canonical_json_bytes(value))
        receipt["outputs"][name] = {"sha256": sha(OUT / name), "bytes": (OUT / name).stat().st_size}
    receipt["source_sha256"]["src/true_rl_agent/tasks/stair_climbing/process_report.py"] = sha(ROOT / "src/true_rl_agent/tasks/stair_climbing/process_report.py")
    receipt["projection_refreshed_after_raw_rebuild"] = True
    (OUT / "rebuild_receipt.json").write_bytes(canonical_json_bytes(receipt))
    result = {"status": "PASS" if all(c["visual_sentence_retained"] for c in cases) else "VISUAL_OMITTED_BY_CAPACITY",
        "synthetic_media_and_review_only": True, "semantic_qualification": False,
        "numeric_report_bytes": len(canonical_json_bytes(public)), "byte_limit": 8192,
        "cases": cases, "api_calls": 0, "gpu_jobs": 0,
        "full_sha256": sha(OUT / "full.json"), "public_sha256": sha(OUT / "public.json")}
    (OUT / "projection_check.json").write_bytes(canonical_json_bytes(result))
    print(json.dumps(result))
    # Reuse the established production frame/HTTP serialization audit; no sender.
    phase = delivery.read(BASE / "results/delivery_offline_r1/stair/public_dev_result.json")
    phase["task_evidence"]["behavior"] = public
    # Keep the runner's canonical attachment name in this derived phase.
    delivery.write(OUT / "behavior_full.json", full)
    phase["task_evidence"]["behavior_full"] = {
        "schema_version": "true_rl_agent.process_full_ref.v1", "file": "behavior_full.json",
        "sha256": sha(OUT / "full.json"), "size_bytes": (OUT / "full.json").stat().st_size}
    phase["task_evidence_hash"] = canonical_json_hash(phase["task_evidence"])
    phase_ref = delivery.write(OUT / "public_dev_result.json", phase)
    confirmation = delivery.module(ROOT / "PLAN/feedback_sys/P3R/confirmation_context.py", "progress_confirmation")
    exp = confirmation._p3(ROOT)
    exp.bound_read, originals = delivery.bound_reader()
    spec = delivery.read(ROOT / "PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_stair.json")
    case = {"task": "stair", "checkpoint_hash": spec["checkpoint_hash"],
            "common_focus": "Read outcomes, complete process relations, compatible success and observation gaps."}
    comp, builder, feedback = confirmation.build_context(case, "T", spec,
        diagnostic_phase_ref=phase_ref, diagnostic_plan_ref=delivery.ref(delivery.ORIGINALS / "stair/public_dev_job.json"),
        allowed_seed_cohort_hash=spec["validated_evaluation_binding"]["seed_cohort_hash"],
        root=ROOT, base=ROOT, output=OUT / "delivery_context", development_fixture=True, exp=exp)
    assert feedback["behavior"]["profile"] == public
    plain = delivery.audit_frames("stair", full, public, builder, comp, OUT / "delivery_plain")
    # The last capacity case is the four-byte fixture; reuse its reviewed object.
    with_visual, generation = delivery.synthetic_s4("stair", shown)
    delivery.install_report(builder, feedback, "stair", with_visual, generation)
    rich = delivery.audit_frames("stair", augmented, with_visual, builder, comp,
        OUT / "delivery_visual_parent", generation=generation, base_phase_full=full)
    assert all(delivery.ref(item["path"])["sha256"] == item["sha256"] for item in originals)
    delivery.write(OUT / "delivery_check.json", {"status": "PASS_OFFLINE_HTTP_SERIALIZATION",
        "not_a_real_provider_send": True, "network_calls": 0,
        "plain": plain, "synthetic_visual_and_parent": rich, "originals_unchanged": True})
    print(json.dumps({"offline_delivery": "PASS", "plain_sentence_checks": plain["sentence_checks"],
        "visual_parent_sentence_checks": rich["sentence_checks"]}))


if __name__ == "__main__":
    main()
