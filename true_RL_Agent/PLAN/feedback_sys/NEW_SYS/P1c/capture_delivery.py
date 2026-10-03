"""R4 P1c acceptance harness: production finalizer/context/HTTP bytes, no send.

Uses existing qualified-context replay helpers. All new artifacts stay under
the explicitly chosen output directory; no rollout, training or model runs.
"""
from copy import deepcopy
from datetime import datetime, timezone
from functools import partial
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.dont_write_bytecode = True

from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig, UrllibJsonTransport,
)
from true_rl_agent.agent_core.public_safety import assert_public_only_prompt_payload
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt
from true_rl_agent.runner.specs import EvalPhaseJobSpec
from true_rl_agent.tasks.stair_climbing.phased_evaluation import finalize_stair_eval_phase_summary
from true_rl_agent.tasks.stair_climbing.trajectory_report import require_trajectory_phase_binding


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return ref(path)


class RequestCaptured(Exception):
    pass


class CaptureOpener:
    def open(self, request, *, timeout):
        self.body = request.data
        raise RequestCaptured()


def capture(payload, output):
    opener = CaptureOpener()
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model="offline-request-capture", base_url="https://request-capture.invalid",
        api_key="offline-placeholder", output_dir=output / "unused", max_tokens=32768,
        reasoning_mode="disabled"), transport=UrllibJsonTransport())
    with patch("urllib.request.build_opener", return_value=opener):
        try:
            provider.next_authoring_response(prompt_payload=payload)
        except RequestCaptured:
            pass
        else:
            raise AssertionError("production transport did not reach capture boundary")
    request = json.loads(opener.body)
    user = request["messages"][-1]["content"].encode("utf-8")
    if json.loads(user) != payload:
        raise AssertionError("serialized provider payload differs")
    return opener.body, user


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    D = load_module(ROOT / "PLAN/feedback_sys/P3NEW/audit_delivery_offline.py", "p1c_old_delivery_helpers")
    C = load_module(ROOT / "PLAN/feedback_sys/P3R/confirmation_context.py", "p1c_context_helpers")
    inputs = ROOT / "PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r4/engineering_report_revision_r1/delivery/inputs"
    item = read(inputs / "delivery_inputs.ready.json")["tasks"]["stair"]
    provenance = []

    def mapped(reference, path):
        actual = ref(path)
        if actual["sha256"] != reference["sha256"]:
            raise ValueError("historical input hash mismatch")
        provenance.append({"original": reference, "local": actual})
        return {**actual, **({"pointer": reference["pointer"]} if "pointer" in reference else {})}

    reader, originals = D.bound_reader()
    plan_ref = mapped(item["plan"], inputs / "stair.public_dev_job.json")
    old_phase_ref = mapped(item["phase"], inputs / "stair.public_dev_result.json")
    case_ref = mapped(item["qualified_case"], ROOT / ("PLAN/" + item["qualified_case"]["path"].split("/PLAN/", 1)[1]))
    spec_ref = mapped(item["qualified_spec"], ROOT / ("PLAN/" + item["qualified_spec"]["path"].split("/PLAN/", 1)[1]))
    case, spec = deepcopy(reader(case_ref, ROOT)), reader(spec_ref, ROOT)
    case["common_focus"] = "Read the exact source-bound trajectory report. Preserve denominators, unknowns, sampled time and capability limits."
    exp = C._p3(ROOT)
    exp.bound_read = reader
    original_spec = deepcopy(spec)
    original_spec.pop("diagnostic_public_dev_phase", None)
    comp, *_ = exp.reference_context(original_spec, root=ROOT, base=ROOT, output=output / "preflight_context")
    raw_plan = read(plan_ref["path"])
    plan = C._plan(raw_plan)
    phase_spec = EvalPhaseJobSpec(job_id=raw_plan["spec"]["job_id"], evaluation_plan=plan,
                                phase=EvaluationPhase.PUBLIC_DEV, metadata=raw_plan["metadata"])
    old_phase = read(old_phase_ref["path"])
    source = read(ROOT / "PLAN/feedback_sys/NEW_SYS/P0/source_lock.json")["sources"][0]
    if (source["identity"]["job_id"] != phase_spec.job_id
            or source["identity"]["evaluation_plan_hash"] != phase_spec.evaluation_plan_hash):
        raise ValueError("P0 source lock differs from R4 job/plan")
    numeric = {**old_phase["task_evidence"]["numeric_evidence"],
               "manifest_path": str(ROOT / source["manifest_relative_path"])}
    if numeric["manifest_sha256"] != source["manifest_sha256"]:
        raise ValueError("R4 phase and numeric source differ")
    card_path = ROOT / "outputs/trajectory_p1b/repair_b1/final_card/manifest.json"
    card_reference = {"manifest_path": str(card_path),
                      "manifest_sha256": "3bfd19515481faa6d9f273426e6b14d141fe616642536d15a6a55c2b2f9990f0"}
    # Explicit CPU replay handoff; this is not a newly executed runner job.
    handoff = {"status": "postprocessing", "task_evidence": deepcopy(old_phase["task_evidence"]),
               "environment_source_preflight": old_phase["environment_source_preflight"],
               "backend_execution_environment_verification": old_phase["backend_execution_environment_verification"],
               "evaluation_execution_cost": {"splits": {"public_dev": {}}}}
    handoff["task_evidence"]["numeric_evidence"] = numeric
    finalizer = partial(finalize_stair_eval_phase_summary, config=comp.task_plugin.config,
                        trajectory_card_ref=card_reference)
    finalized = finalizer(phase_spec, handoff, output_dir=output)
    full = read(output / "behavior_full.json")
    public = finalized["task_evidence"]["behavior"]
    phase = deepcopy(old_phase)
    phase["task_evidence"] = finalized["task_evidence"]
    phase["task_evidence_hash"] = finalized["task_evidence_hash"]
    phase_ref = write(output / "public_dev_result.json", phase)
    require_trajectory_phase_binding(public, job_id=phase["job_id"], checkpoint_hash=phase["checkpoint_hash"],
        evaluation_plan_hash=phase["evaluation_plan_hash"], evaluation_binding_hash=phase["evaluation_binding_hash"],
        archive_hash=numeric["manifest_sha256"], seed_namespace=source["identity"]["seed_namespace"])
    write(output / "public_report.json", public)
    write(output / "projection_audit.json", full["audit"])
    comp, builder, feedback = C.build_context(case, "T", spec, root=ROOT, base=ROOT,
        output=output / "context", diagnostic_phase_ref=phase_ref, diagnostic_plan_ref=plan_ref,
        allowed_seed_cohort_hash=source["identity"]["evaluation_binding"]["seed_cohort_hash"],
        development_fixture=True, exp=exp)
    if feedback["behavior"]["profile"] != public:
        raise AssertionError("task prompt adapter changed card")
    frames, sentence_checks = [], []
    for name, payload in D.make_frames(builder, comp, output / "frame_fixtures").items():
        body, user = capture(payload, output)
        assert_public_projection_safe(json.loads(user))
        assert_public_only_prompt_payload(json.loads(user))
        directory = output / name
        directory.mkdir()
        (directory / "request.http_body.json").write_bytes(body)
        (directory / "user_payload.json").write_bytes(user)
        delivered = json.loads(user)["public_feedback_summary"]["behavior"]["profile"]
        if delivered != public:
            raise AssertionError("final serialized payload lost or changed report")
        for section in SENTENCE_SECTIONS:
            for row in public["report"][section]:
                fragment = json.dumps(row["text"], ensure_ascii=False)[1:-1].encode("utf-8")
                wire_fragment = json.dumps(fragment.decode("utf-8"), ensure_ascii=False)[1:-1].encode("utf-8")
                if user.find(fragment) < 0 or body.find(wire_fragment) < 0:
                    raise AssertionError("sentence absent from final serialization")
                sentence_checks.append({"frame": name, "section": section, **row,
                    "user_byte_offset": user.find(fragment), "http_byte_offset": body.find(wire_fragment),
                    "original_claim": full["audit"]["claim_trace"].get(row["id"]),
                    "exact_sentence_preserved": True})
        budget = payload["authoring_context_frame"]["budget_metadata"]
        validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=payload)
        if budget["actual_prompt_payload_bytes"] != len(user):
            raise AssertionError("payload byte receipt mismatch")
        frames.append({"frame": name, "http_body": ref(directory / "request.http_body.json"),
            "user_payload": ref(directory / "user_payload.json"), "budget": budget,
            "report_bytes": report_bytes(delivered), "report_limit_bytes": 8192})
    write(output / "sentence_delivery.json", sentence_checks)
    receipt = {"status": "PASS_LOCAL_PRODUCTION_SERIALIZATION_CAPTURE", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "card_reference": card_reference, "card_id": public["card_id"], "schema_version": public["schema_version"],
        "domain": public["domain"], "included_claim_count": len(full["audit"]["included_claim_ids"]),
        "omitted_claim_count": len(full["audit"]["omitted_claims"]), "frames": frames,
        "source_provenance": provenance, "qualification_sources": originals,
        "cpu_finalizer_cost": finalized["evaluation_execution_cost"],
        "request_capture_boundary": "production UrllibJsonTransport creates Request.data; local opener raises before network",
        "model_name_field": "offline-request-capture (request configuration only; no model executed)",
        "network_calls": 0, "model_calls": 0, "training_runs": 0, "provider_usage": None,
        "author_understanding": "not_tested; subsequent P3 boundary",
        "online_default": "legacy process report; trajectory card is explicit opt-in via existing finalizer injection",
        "qualification_unchanged": True, "original_phase_unchanged": file_sha256(Path(old_phase_ref["path"])) == old_phase_ref["sha256"],
        "limits": ["CPU replay finalizer invocation, not a new runner/GPU execution", "R4 diagnostic observation is not a generation lane",
                   "Source-followup is a local allowlisted source request; repair frames use explicitly synthetic errors and no model response",
                   "Author code/source-catalog relationship is not established by the trajectory card", "No DeepSeek or other live model validation"]}
    write(output / "capture_receipt.json", receipt)
    print(json.dumps({"status": receipt["status"], "report_bytes": report_bytes(public),
        "included_claims": receipt["included_claim_count"], "omitted_claims": receipt["omitted_claim_count"],
        "frames": [{"frame": f["frame"], "http_bytes": f["http_body"]["size_bytes"], "user_bytes": f["user_payload"]["size_bytes"]} for f in frames]}, indent=2))


if __name__ == "__main__":
    main()
