"""Six fixed DS facts, sequential, one attempt each, after root's recovery audit.

Prepared for review only. Running this script dispatches APIs; no retry, report,
facts seal or automatic continuation is implemented. Existing evidence is kept.
"""
import argparse
from decimal import Decimal
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import run_registered_recovery_once as io
import prepare_facts_offline as prep

C, S = io.C, io.S
P = S / "PLAN/feedback_sys/P3NEW"
OUT = C / "amendment_r1_remaining_ds_facts"
STOP = C / "STOP_FACTS_DISPATCH_AMENDMENT_R1"
PACKETS = ("focus-01", "focus-07", "media-01", "media-02", "media-03", "media-04")
AMENDMENT_SHA = "c08ace063b3ae6a34300308cba856e82294bbe25fdfc0573865a192557f1bf34"
PREPARATION_SHA = "2090a5ee711207b31b56cd77e907821b38137f201908b87858199f882d5df9f0"
RECOVERY_SHA = "890f427c903c5918f67fdde5e41f9c09d696e2c5565dec408aff5e29b7d04f50"
CHECKER_SHA = "018078561cdc869562a4b5379a63f647418d450e6990cc75beb9c58b715dd5fd"
STRUCTURE_SHA = "3aa8f452ced834823dc433175ed9e99fd4438f9cca93138f774fc84003885ca3"
COMPARISON_SHA = "c115eb16e4d4c6187cc32c931096164be0261f55b1b433308c3be1b0b2e6be73"
ROOT_GATE_SHA = "a5685e26ab5c9ead2ce741ac17a248ed57d2d188e485e8feb30403f03b290665"


def check(reference, path=None):
    path = Path(path or reference.get("path", reference.get("file", "")))
    actual = io.ref(path)
    prep.require(actual["sha256"] == reference["sha256"], "SHA mismatch: " + str(path))
    if "size_bytes" in reference:
        prep.require(actual["size_bytes"] == reference["size_bytes"], "Size mismatch: " + str(path))
    return path


def prerequisites(structure_path, structure_sha):
    prep.require(not OUT.exists(), "Preserve the prior batch; no automatic retry/resume")
    prep.require(not STOP.exists(), "Amendment dispatch stop marker exists")
    source = prep.verify_source(S, C / "runtime_source_manifest.json")
    amendment_path = check({"sha256": AMENDMENT_SHA}, C / "runtime_limit_amendment_r1.json")
    preparation_path = check({"sha256": PREPARATION_SHA}, C / "amendment_r1_facts_preparation/preparation_receipt.json")
    amendment, preparation = io.read(amendment_path), io.read(preparation_path)
    slots = {r["packet_id"]: r for r in preparation["slots"]}
    registered = {r["packet_id"]: r for r in amendment["DeepSeek_delta"]["additional_facts_slots"]}
    prep.require(set(slots) == set(registered) == {"episode-13", *PACKETS}, "Registered seven-slot set differs")
    recovery_dir = C / "amendment_r1_facts_preparation/episode-13/prepared"
    recovery_path = check({"sha256": RECOVERY_SHA}, recovery_dir / "receipt.json")
    recovery = io.read(recovery_path)
    execution_path = C / "amendment_r1_recovery/execution_receipt.json"
    execution = io.read(execution_path)
    check(execution["runner_receipt"], recovery_path)
    prep.require(execution["returncode"] == 0 and execution["complete"] is True
                 and recovery["complete"] is True and recovery["finish_reason"] == "stop", "Recovery did not complete")
    comparison_path = check({"sha256": COMPARISON_SHA}, C / "amendment_r1_recovery/only_max_tokens_changed.json")
    comparison = io.read(comparison_path)
    prep.require(comparison["status"] == "PASS_ONLY_MAX_TOKENS_CHANGED"
                 and comparison["amendment"]["sha256"] == AMENDMENT_SHA
                 and comparison["cpu_preparation"]["sha256"] == PREPARATION_SHA
                 and comparison["whole_messages_images_and_catalog_unchanged"] is True, "Recovery HTTP comparison differs")
    row = slots["episode-13"]
    for name in ("original_request", "new_request", "new_prepare", "reference_catalog"):
        check(row[name])
    prep.require(comparison["old_request_sha256"] == row["original_request"]["sha256"]
                 and comparison["new_request_sha256"] == recovery["request_sha256"] == row["new_request"]["sha256"],
                 "Recovery comparison does not bind actual request")
    old, new = io.read(row["original_request"]["path"]), io.read(row["new_request"]["path"])
    prep.require(old["max_tokens"] == 32768 and new == {**old, "max_tokens": 65536}, "Recovery HTTP differs beyond output cap")
    prep.require(structure_sha == STRUCTURE_SHA, "Not the actual root recovery structure audit")
    structure_path = check({"sha256": structure_sha}, structure_path)
    structure = io.read(structure_path)
    prep.require(structure["packet_id"] == "episode-13" and structure["judge_id"] == "deepseek"
                 and structure["stage"] == "facts" and structure["structural_contract_complete"] is True
                 and not structure["errors"] and structure["frozen_reference_audit"]["review_complete"] is True,
                 "Root recovery structure/reference audit incomplete")
    for name, reference in (("answer", recovery["content"]), ("request", row["new_request"]), ("prepared", row["new_prepare"])):
        prep.require(structure["inputs"][name]["sha256"] == reference["sha256"], "Root audit input differs: " + name)
    root_gate_path = check({"sha256": ROOT_GATE_SHA}, C / "amendment_r1_recovery/root_gate.json")
    root_gate = io.read(root_gate_path)
    prep.require(root_gate["status"] == "COMPLETE_TRANSPORT_AND_STRUCTURE_ONLY_NOT_SEMANTIC_PASS"
                 and root_gate["amendment"]["sha256"] == AMENDMENT_SHA
                 and root_gate["same_message_content_roles_images_order"] is True
                 and root_gate["original_R4_disposition"] == "INCONCLUSIVE", "Root recovery gate differs")
    check(root_gate["receipt"], recovery_path)
    check(root_gate["structure_audit"], structure_path)
    check(root_gate["only_max_tokens_changed"], comparison_path)
    check(recovery["content"])
    check(recovery["answer_coverage"])
    prep.require(io.read(recovery["answer_coverage"]["file"])["review_complete"] is True, "Recovery frozen coverage incomplete")
    checker_path = check({"sha256": CHECKER_SHA}, C / "facts_contract_audit.py")
    spec = importlib.util.spec_from_file_location("amended_facts_checker", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    # Only runtime path binding changes; original checker bytes remain pinned.
    checker.PROTOCOL, checker.HELPER = P / "N5_FACT_REVIEW_PROTOCOL_ZH.md", P / "confirmation_review_coverage.py"
    for path in (checker.PROTOCOL, checker.HELPER, P / "run_confirmation_review.py"):
        check(source[path.relative_to(S).as_posix()], path)
    proposal_path = C / "facts_dispatch_preparation/deepseek_reservation_proposal.json"
    proposal = {r["packet_id"]: r for r in io.read(proposal_path)["rows"]}
    check({"sha256": prep.CARD_SHA}, C / "capture_card.json")
    budget = io.read(C / "capture_card.json")["budget"]
    prep.require(budget["api_phase_limit_usd"] == 15 and budget["budget_id"] == amendment["budget"]["budget_id"], "Phase budget differs")
    for packet in PACKETS:
        row, declaration = slots[packet], registered[packet]
        for name in ("original_request", "new_prepare", "new_request", "reference_catalog"):
            check(row[name])
        old, new = io.read(row["original_request"]["path"]), io.read(row["new_request"]["path"])
        prep.require(old["max_tokens"] == 32768 and new == {**old, "max_tokens": 65536}, "HTTP delta differs: " + packet)
        prep.require(declaration["maximum_new_attempts"] == 1 and declaration["purpose"] == "first_request"
                     and declaration["original_request"]["sha256"] == row["original_request"]["sha256"]
                     and proposal[packet]["request"]["sha256"] == row["original_request"]["sha256"], "Slot/proposal binding differs")
        reserve = proposal[packet]["suggested_input_token_reservation"]
        prep.require(reserve == declaration["input_token_reservation_existing_basis"], "Input reserve changed")
        prices = budget["prices_usd_per_million"]
        worst = (Decimal(reserve) * max(Decimal(str(prices[k])) for k in ("cache_hit", "cache_miss"))
                 + Decimal(65536) * Decimal(str(prices["output"]))) / Decimal(1000000)
        prep.require(worst <= Decimal("0.20"), "0.20 does not cover this registered input/output cap")
        prepared = Path(row["new_prepare"]["path"]).parent
        prep.require(not any((prepared / n).exists() for n in ("receipt.json", "raw_response.json", "raw_answer.txt")),
                     "Slot already has execution evidence: " + packet)
    return slots, proposal, budget, checker, {
        "amendment": io.ref(amendment_path), "preparation": io.ref(preparation_path),
        "recovery": io.ref(recovery_path), "recovery_execution": io.ref(execution_path),
        "recovery_http_comparison": io.ref(comparison_path), "root_structure_audit": io.ref(structure_path),
        "root_gate": io.ref(root_gate_path),
        "checker": io.ref(checker_path), "checker_bound_protocol": io.ref(checker.PROTOCOL),
        "checker_bound_helper": io.ref(checker.HELPER), "input_reservations": io.ref(proposal_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-structure-audit", required=True, type=Path)
    parser.add_argument("--recovery-structure-sha256", required=True)
    args = parser.parse_args()
    slots, proposal, budget, checker, evidence = prerequisites(args.recovery_structure_audit, args.recovery_structure_sha256)
    OUT.mkdir()
    io.save(OUT / "prerequisites.json", {"checked_at_utc": io.now(), "script": io.ref(__file__), "evidence": evidence,
                                      "scope": "six_first_attempt_facts_only_under_registered_runtime_amendment"})
    results, failure = [], None
    runner, ledger = P / "run_confirmation_review.py", C / "budget_ledger.working.json"
    for packet in PACKETS:
        if STOP.exists():
            failure = "STOP_FACTS_DISPATCH_AMENDMENT_R1 exists before " + packet
            break
        slot = OUT / packet
        slot.mkdir()
        result = {"packet_id": packet, "judge_id": "deepseek", "stage": "facts", "driver_dispatched": False}
        try:
            row = slots[packet]
            for name in ("new_prepare", "new_request", "reference_catalog"):
                check(row[name])
            prepared = Path(row["new_prepare"]["path"]).parent
            release = io.read(prepared / "release.draft.json")
            prep.require(release["output_token_reservation"] == 65536, "Release output differs")
            release.update(status="RELEASED_FOR_ONE_CALL", input_token_reservation=proposal[packet]["suggested_input_token_reservation"],
                           reserved_usd=0.20, prices_usd_per_million=budget["prices_usd_per_million"], pricing_source=budget["pricing_source"])
            release["budget"] = {"ledger_file": str(ledger), "ledger_sha256": io.ref(ledger)["sha256"],
                                 "budget_id": budget["budget_id"], "limit_usd": 15}
            release_path = slot / "release.json"
            io.save(release_path, release)
            prep.require(not STOP.exists(), "Stop marker appeared before dispatch")
            argv = [sys.executable, "-B", str(runner), "run", "--prepared", str(prepared), "--release", str(release_path),
                    "--secret", "/data/yuyangliu/secrets/deepseek.env"]
            io.save(slot / "dispatch.json", {"started_at_utc": io.now(), "argv": argv, "release": io.ref(release_path),
                                            "request": row["new_request"], "amendment": evidence["amendment"]})
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONPATH=os.pathsep.join(map(str, (S / "src", S, P))))
            result["driver_dispatched"] = True
            with (slot / "stdout.log").open("xb") as stdout, (slot / "stderr.log").open("xb") as stderr:
                code = subprocess.run(argv, cwd=S, env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr).returncode
            result["returncode"] = code
            receipt_path = prepared / "receipt.json"
            receipt = io.read(receipt_path)
            result["runner_receipt"] = io.ref(receipt_path)
            result["request_attempted"] = receipt.get("request_attempted")
            prep.require(code == 0 and receipt["complete"] is True and receipt["finish_reason"] == "stop"
                         and receipt["request_sha256"] == row["new_request"]["sha256"], "Frozen runner incomplete")
            answer_path = check(receipt["content"])
            audit = checker.audit_contract(prepared / "prepare.json", answer_path)
            io.save(slot / "facts_contract_audit.json", audit)
            result["structure_audit"] = io.ref(slot / "facts_contract_audit.json")
            prep.require(audit["structural_contract_complete"] is True and not audit["errors"], "Facts contract incomplete")
            result.update(status="COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED", ledger_after=io.ref(ledger))
        except Exception as error:
            failure = type(error).__name__ + ": " + str(error)
            result.update(status="FAILED_STOP_NEW_DISPATCH", error=failure)
            if not STOP.exists():
                try:
                    with STOP.open("x", encoding="utf-8") as f:
                        f.write(io.now() + " " + packet + " " + failure + "\n")
                except FileExistsError:
                    pass
        result["finished_at_utc"] = io.now()
        io.save(slot / "return.json", result)
        results.append(result)
        if failure:
            break
    final = {"status": "STOPPED_INCOMPLETE" if failure else "SIX_FACTS_COMPLETE_NOT_SEALED", "finished_at_utc": io.now(),
             "scope": "same_cohort_runtime_amendment_not_original_R4_PASS", "failure": failure, "results": results,
             "completed_slots": sum(r["status"] == "COMPLETED_TRANSPORT_AND_STRUCTURE_NOT_SEALED" for r in results),
             "automatic_retries": 0, "facts_seals": 0, "report_calls": 0, "ledger": io.ref(ledger)}
    io.save(OUT / "execution_receipt.json", final)
    print({k: final[k] for k in ("status", "completed_slots", "failure")})
    return 1 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
