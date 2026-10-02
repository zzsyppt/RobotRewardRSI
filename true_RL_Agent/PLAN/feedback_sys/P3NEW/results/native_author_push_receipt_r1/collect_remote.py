"""Read-only remote receipt verification; stream selected exact originals as tar."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tarfile
from decimal import Decimal

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
BASE = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929")
SOURCE = BASE / "source_observer_v3_r8"
HELPERS = BASE / "text_author_smoke_r1/PLAN/feedback_sys/P3NEW"
OUT = HELPERS / "results/native_author_smoke_r3"
SHARED = HELPERS / "results/text_author_smoke_r1"
ROUND = "p3new-text-smoke-20260929-generation-003-lane-b"
JOURNAL = SHARED / "authoring/p3new-text-smoke-20260929-generation-003/lane-b"
TRANSPORT = OUT / "transport/push-native-dev"
sys.path.insert(0, str(SOURCE / "src"))
sys.path.append(str(HELPERS))
spec = importlib.util.spec_from_file_location("sealed_sender", HELPERS / "send_native_author_smoke_after_recovery.py")
sender = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sender)
from true_rl_agent.agent_core import openai_compatible_provider as O
from true_rl_agent.prompt_artifacts import AuthoringEvidenceRef, RoundAuthoringJournal, verify_authoring_evidence


def sha(p):
    with p.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


stage_path = OUT / "stage.runtime_recovered.json"
stage = sender.D.read(stage_path)
sender.validate(stage, stage_path, OUT, released=True)
assert sender.validate_prior_recovery(stage, stage_path, OUT, SHARED, "push-native-dev")
terminal_path = OUT / "results" / (ROUND + ".json")
terminal = sender.D.read(terminal_path)
assert terminal["stage_ref"] == sender.D.ref(stage_path)
assert terminal["driver_ref"] == stage["execution_script"]
assert terminal["delivery_status"] == "PASS" and terminal["stop_reason"] == "candidate_verified"
assert terminal["train_status"] == "not_run" and terminal["transport_invocations"] == 1
assert all(v["status"] == "PASS" for v in terminal["postcall_validation"].values())
row = stage["episodes"][1]
request = sender.D.read(TRANSPORT / "request.http_body.json")
assert (TRANSPORT / "request.http_body.json").read_bytes() == sender.P.checked(row["initial_http_body"]).read_bytes()
assert request["messages"][-1]["content"].encode() == sender.P.checked(row["initial_payload"]).read_bytes()
assert sender.D.read(TRANSPORT / "user_payload.json") == json.loads(request["messages"][-1]["content"])
raw = sender.D.read(TRANSPORT / "raw_response.json")
receipt = sender.D.read(TRANSPORT / "response_receipt.json")
assert raw == sender.D.read(sender.P.checked(receipt["raw_response"]))
assert sender.P.checked(receipt["request"]) == TRANSPORT / "request.http_body.json"
assert raw["usage"] == receipt["usage"] and raw["model"] == receipt["response_model"]
snapshot = RoundAuthoringJournal(JOURNAL, round_id=ROUND).snapshot()
assert len(snapshot.attempts) == 1 and snapshot.attempts[0]["status"] == "completed"
attempt, = snapshot.completed_attempt_records
assert terminal["authoring_checks"] == [attempt]
assert attempt["provider_usage"] == O.normalize_provider_usage(raw["usage"])
transport = O.UrllibJsonTransport(max_response_bytes=stage["provider"]["max_response_bytes"])
expected_receipt = O._provider_response_receipt(raw, request_payload=request,
    finish_reason=O._provider_finish_reason(raw), transport_policy_id=transport.transport_policy_id,
    response_max_bytes=transport.max_response_bytes)
assert attempt["provider_response_receipt"] == expected_receipt
prompt_ref, = snapshot.prompt_payload_refs
evidence = verify_authoring_evidence(AuthoringEvidenceRef.from_private_mapping(terminal["authoring_evidence_ref"]),
    expected_round_id=ROUND, expected_selected_prompt=prompt_ref)
assert len(evidence["attempts"]) == 1
assert sender.D.read(Path(prompt_ref.path)) == sender.D.read(TRANSPORT / "user_payload.json")
assert attempt["verifier_status"] == "pass" and attempt["component_smoke"]["status"] == "pass"
assert attempt["anti_degenerate_assessment"]["status"] == "pass"
package = Path(terminal["candidate_private_path"])
proposal = json.loads(raw["choices"][0]["message"]["content"])
assert proposal["candidate_id"] == attempt["candidate_id"] == "push_recovery_margin_v4"
assert (package / "training_code/components.py").read_bytes() == proposal["components_py"].encode()
assert "sha256:" + sha(package / "training_code/components.py") == attempt["candidate_program_source_checksum"]
assert len(list(JOURNAL.glob("authoring_attempts/*"))) == 1
assert len(list((OUT / "provider_candidates/push-native-dev").iterdir())) == 1
assert not (TRANSPORT / "transport_terminal.json").exists()
usage = sender.accounting(stage, SHARED)
sender.check_usage(usage, stage, expected_count=6)
assert usage == sender.D.read(OUT / "push-native-dev.usage.json") == sender.D.read(OUT / "usage.json")
u, prices = raw["usage"], stage["budget"]["prices_usd_per_million"]
cost = sum(Decimal(u[k]) * Decimal(str(prices[p])) for k, p in [
    ("prompt_cache_hit_tokens", "cache_hit"), ("prompt_cache_miss_tokens", "cache_miss"),
    ("completion_tokens", "output")]) / Decimal(1000000)
shared_cost = sum(Decimal(str(r["usd_peak"])) for r in usage["attempts_detail"])
assert abs(shared_cost - Decimal("0.200609736")) < Decimal("0.000000000001")

# A complete index of the native stage, shared history, frozen source, and all
# explicitly frozen inputs; large source archives/media remain at the source.
paths = {p for root in (OUT, SHARED, SOURCE) for p in root.rglob("*") if p.is_file()}
paths.update(sender.P.checked(ref) for ref in stage["frozen_inputs"])
paths.add(terminal_path)
index = {str(p): {"sha256": sha(p), "size_bytes": p.stat().st_size} for p in sorted(paths)}
copies = {}
for root, prefix in ((TRANSPORT, "transport"), (package, "candidate"), (JOURNAL, "journal")):
    copies.update({p: prefix + "/" + p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()})
copies.update({terminal_path: "terminal.json", stage_path: "cards/stage.runtime_recovered.json",
    sender.P.checked(stage["previous_release_ref"]): "cards/stage.released.json",
    sender.P.checked(stage["prior_case_recovery"]): "cards/stair_recovery.json",
    sender.P.checked(stage["execution_script"]): "cards/send_native_author_smoke_after_recovery.py",
    sender.P.checked(stage["production_source_snapshot"]): "cards/production_source_snapshot.json",
    OUT / "push-native-dev.usage.json": "shared_author_usage.json",
    SHARED / "usage.json": "historical_four_author_usage.json"})
for p, name in copies.items():
    if str(p) not in index:
        index[str(p)] = {"sha256": sha(p), "size_bytes": p.stat().st_size}
    index[str(p)]["local_original"] = name
report = {"status": "READ_ONLY_RECEIPT_AUDIT_PASS", "case_id": "push-native-dev",
    "scope": "old_development_not_n5", "new_api_calls": 0, "gpu_jobs": 0, "credential_contents_read": False,
    "delivery_status": terminal["delivery_status"], "stop_reason": terminal["stop_reason"],
    "train_status": terminal["train_status"], "training_evidence_scope": "saved author-only terminal; no RL training/eval performed by this audit",
    "actual_transport_invocations": 1, "provider_attempts": 1, "response_model": raw["model"],
    "system_fingerprint": raw["system_fingerprint"], "raw_usage": u,
    "candidate_id": proposal["candidate_id"], "candidate_source_sha256": sha(package / "training_code/components.py"),
    "static_verifier": "pass", "authoring_delta": "pass", "anti_degenerate": "pass", "component_smoke": "pass",
    "static_check_scope": "verified saved production receipts; candidate was not rerun during this collection",
    "source_root": str(SOURCE), "source_snapshot_matches": True,
    "costs_usd": {"push_actual": str(cost), "stair_actual": "0.025491636",
        "shared_author_ledger_6_known_calls": "0.200609736", "shared_author_historical_4_calls": "0.127089852",
        "p3new_before_native_15_calls": "0.255664596", "p3new_before_native_baseline_source": "parent supplied frozen cumulative baseline",
        "p3new_total_17_known_calls": str(Decimal("0.255664596") + Decimal("0.025491636") + cost)},
    "shared_author_usage_violations": usage["violations"], "indexed_files": len(index), "copied_originals": len(copies),
    "index_scope": [str(OUT), str(SHARED), str(SOURCE), "stage.frozen_inputs exact files"]}
with tarfile.open(fileobj=sys.stdout.buffer, mode="w|") as archive:
    for p, name in sorted(copies.items(), key=lambda x: x[1]):
        assert sha(p) == index[str(p)]["sha256"]
        archive.add(p, arcname=name, recursive=False)
    for name, value in (("audit.json", report), ("hash_index.json", index)):
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        entry = tarfile.TarInfo(name)
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
