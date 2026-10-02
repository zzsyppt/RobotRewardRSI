"""One-off offline recovery of the already received Stair call; never resubmit."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
SOURCE = BASE / "source_observer_v3_r8"
HELPERS = BASE / "text_author_smoke_r1/PLAN/feedback_sys/P3NEW"
ORIGINAL = HELPERS / "results/native_author_smoke_r3"
SHARED = HELPERS / "results/text_author_smoke_r1"
ROUND = "p3new-text-smoke-20260929-generation-003-lane-a"
ROUND_DIR = SHARED / "authoring/p3new-text-smoke-20260929-generation-003/lane-a"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE / "src"))
import true_rl_agent
sys.path.append(str(HELPERS))
import prepare_native_author_smoke as P
sys.path.insert(0, str(Path(true_rl_agent.__file__).resolve().parents[2] / "src"))
from true_rl_agent.agent_core import authoring_loop as A
from true_rl_agent.agent_core import openai_compatible_provider as O
from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.candidates.verifier import VerificationResult
from true_rl_agent.prompt_artifacts import AuthoringAttemptHandle, RoundAuthoringJournal, verify_prompt_payload
from true_rl_agent.runner.source_snapshot import project_source_snapshot


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def originals():
    paths = [p for root in (ORIGINAL, SHARED, SOURCE) for p in root.rglob("*") if p.is_file()]
    paths.append(HELPERS / "send_native_author_smoke.py")
    return {str(p): sha(p) for p in sorted(paths)}


def same_originals(before):
    assert all(Path(p).is_file() and sha(p) == h for p, h in before.items()), "original file drift"


def main(append):
    assert Path(true_rl_agent.__file__).resolve().parents[2] == SOURCE
    before = originals()
    stage_path = ORIGINAL / "stage.released.json"
    stage = P.D.read(stage_path)
    row = stage["episodes"][0]
    assert row["case_id"] == "stair-native-dev" and row["round_id"] == ROUND
    assert stage["production_source_root"] == str(SOURCE)
    assert project_source_snapshot(SOURCE) == P.D.read(P.checked(stage["production_source_snapshot"]))
    terminal_path = ORIGINAL / "results" / (ROUND + ".json")
    terminal = P.D.read(terminal_path)
    assert terminal["delivery_status"] == "FAILED" and terminal["exception_class"] == "IsolatedWorkerProtocolError"
    assert terminal["transport_invocations"] == 1
    transport = ORIGINAL / "transport/stair-native-dev"
    receipt = P.D.read(transport / "response_receipt.json")
    raw = P.D.read(transport / "raw_response.json")
    assert raw == json.loads(P.checked(receipt["raw_response"]).read_bytes())
    request_path = P.checked(receipt["request"])
    request = P.D.read(request_path)
    assert request_path.read_bytes() == P.checked(row["initial_http_body"]).read_bytes()
    assert raw["usage"] == receipt["usage"] and raw["model"] == receipt["response_model"]
    proposal = json.loads(raw["choices"][0]["message"]["content"])
    package, = (ORIGINAL / "provider_candidates/stair-native-dev").iterdir()
    assert (package / "training_code/components.py").read_bytes() == proposal["components_py"].encode("utf-8")
    assert P.D.read(package / "operation_diff.json")["declared_changed_components"] == proposal["declared_changed_components"]
    candidate = CandidatePackageRef(proposal["candidate_id"], str(package),
        CandidateChangeSet(frozenset(proposal["declared_changed_components"])),
        {"provider": "openai_compatible_chat", "declared_change_set_provisional": True})
    journal = RoundAuthoringJournal(ROUND_DIR, round_id=ROUND)
    snapshot = journal.snapshot()
    assert len(snapshot.attempts) == 1 and snapshot.attempts[0]["status"] == "interrupted_after_prompt"
    prompt_ref, = snapshot.prompt_payload_refs
    prompt = verify_prompt_payload(prompt_ref)
    assert json.loads(request["messages"][-1]["content"]) == prompt
    material = P.checked(row["material"]).parent
    comp, builder, _ = P.context_for("stair", material, P.D.read(material / "provenance.json"),
        OUT / ("context_append" if append else "context_verify"))
    rules = comp.candidate_admission_policy.authoring_prompt_rules()
    built = A.build_authoring_prompt(context_builder=builder, records=[], repair_feedback=None,
        latest_failed_candidate=None, authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
        candidate_verifier_prompt_rules=rules)
    assert built.provider_payload == prompt
    parent_source, parent_hash, role = select_authoring_anchor_source(prompt)
    assert role == "current_best" and parent_hash == row["qualified_parent_source_sha256"]
    verification = comp.candidate_admission_policy.verify(candidate)
    delta = (comp.candidate_admission_policy.verify_authoring_delta(candidate=candidate,
        verification=verification, parent_source=parent_source) if verification.status == "pass" else None)
    usage = O.normalize_provider_usage(raw["usage"])
    transport_impl = O.UrllibJsonTransport(max_response_bytes=stage["provider"]["max_response_bytes"])
    provider_receipt = O._provider_response_receipt(raw, request_payload=request,
        finish_reason=O._provider_finish_reason(raw), transport_policy_id=transport_impl.transport_policy_id,
        response_max_bytes=transport_impl.max_response_bytes)
    report = {"status": "OFFLINE_RECEIVED_CANDIDATE_RECHECK", "original_delivery_status": "FAILED",
        "api_calls": 0, "credential_contents_read": False, "gpu_jobs": 0,
        "admission": verification.to_public_dict(), "delta": delta.to_public_dict() if delta else None,
        "raw_provider_usage": raw["usage"], "normalized_usage": usage, "provider_response_receipt": provider_receipt,
        "stage_ref": P.D.ref(stage_path), "original_terminal_ref": P.D.ref(terminal_path),
        "transport_receipt_ref": P.D.ref(transport / "response_receipt.json"),
        "candidate_source_sha256": sha(package / "training_code/components.py"),
        "production_source_root": str(SOURCE), "spawn_source_path_first": sys.path[0]}
    same_originals(before)
    P.D.write(OUT / ("append_original_hashes.json" if append else "verify_original_hashes.json"), before)
    if not append:
        P.D.write(OUT / "offline_verification.json", report)
        print(json.dumps({"status": report["status"], "admission": verification.status,
                          "delta": delta.status if delta else None, "api_calls": 0}))
        return
    # Record the original infrastructure failure, separately from the new CPU recheck.
    error = VerificationResult("fail", "original_verifier_worker_import_error",
        {"exception_class": "IsolatedWorkerProtocolError", "stage": "original_verifier_spawn",
         "original_call_completed_verification": False})
    feedback = {"schema_version": "true_rl_agent.provider_repair_feedback.v1",
        "failure_category": "verifier_error", "verifier_status": "fail", "verifier_reason": error.reason,
        "verifier_details": dict(error.details), "instructions": ["Terminal infrastructure interruption; no retry or candidate edit."]}
    static_failure = A.build_static_authoring_failure(candidate=candidate, feedback=feedback)
    attempt = {"round_id": ROUND, "attempt_id": "attempt_000001", "attempt": 1,
        "provider_status": "pass", "provider_response_mode": proposal["response_mode"],
        "provider_usage": usage, "provider_response_receipt": provider_receipt,
        **A._candidate_attempt_public_record(candidate, error)}
    state = A._AuthoringContinuation(round_id=ROUND, disposition="terminal", repair_feedback=feedback,
        response_feedback=None, latest_failed_candidate=candidate, frame_kind=None, source_payloads=(),
        source_repository_prompt_ref=prompt_ref, source_request_count=0, source_request_bytes_returned=0,
        selected_prompt_payload_ref=prompt_ref, selected_round_public_context=built.round_public_context,
        selected_parent_program_source_checksum=built.parent_program_source_checksum,
        selected_candidate=candidate, last_provider_failure=None, stop_reason="static_failed", static_failure=static_failure).to_mapping()
    A._restore_authoring_continuation(state, expected_round_id=ROUND)
    handle = AuthoringAttemptHandle(ROUND, "attempt_000001", 1, prompt_ref,
        str(ROUND_DIR / "authoring_attempts/attempt_000001/attempt_result.json"))
    # Use production OS locks without rewriting their existing owner-note bytes.
    with acquire_process_file_lock(SHARED / ".text_author_smoke.lock", blocking=False):
        with acquire_process_file_lock(ROUND_DIR / ".authoring.lease", blocking=False):
            assert not Path(handle.result_path).exists()
            same_originals(before)
            journal.complete_attempt(handle, attempt, continuation_state=state)
            restored = journal.snapshot()
            assert len(restored.attempts) == 1 and restored.attempts[0]["status"] == "completed"
            A._restore_authoring_continuation(restored.latest_continuation_state, expected_round_id=ROUND)
    same_originals(before)
    cost_card = dict(stage)
    cost_card["budget"] = {**stage["budget"], "stage_usd": 4.0}
    accounting = P.R.usage_report(SHARED, cost_card, P.C._p3(P.ROOT))
    assert accounting["attempts"] == 5 and not accounting["violations"]
    assert all(a["usage_known"] and a["status"] == "completed" for a in accounting["attempts_detail"])
    P.D.write(OUT / "recovered_usage.json", accounting)
    report.update(status="RECOVERED_TERMINAL_VERIFIER_ERROR_NO_RESUBMISSION",
        continuation_stop_reason="static_failed", failure_category="verifier_error",
        appended_attempt_ref=P.D.ref(Path(handle.result_path)), original_files_unchanged=len(before),
        shared_usage_ref=P.D.ref(OUT / "recovered_usage.json"))
    P.D.write(OUT / "recovery.json", report)
    print(json.dumps({"status": report["status"], "admission": verification.status,
        "delta": delta.status if delta else None, "known_calls": accounting["attempts"],
        "original_files_unchanged": len(before), "api_calls": 0}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true")
    main(parser.parse_args().append)
