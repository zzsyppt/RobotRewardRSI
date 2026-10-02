"""Offline static admission of existing R.0-a candidates; no API or repair.

Restores each task's already-qualified P3 context, then applies its production
verifier and authoring-delta check against the actual frozen first prompt.
This report is separate from, and never changes, the runtime completion gate.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(args):
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.prompt_artifacts import PromptPayloadRef, RoundAuthoringJournal, verify_prompt_payload
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    spec = importlib.util.spec_from_file_location("p3_runtime_verification_context", args.p3_tools / "author_experiment.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    cases = {c["case_id"]: c for c in read(args.p3_cases)["cases"]}
    stage = read(args.stage)
    contexts, rows = {}, []
    for row in stage["first_prompts"]:
        generation, lane = row["generation_id"], row["lane_id"]
        journal = RoundAuthoringJournal(args.runtime_output / "authoring" / generation / lane, round_id=f"{generation}-{lane}")
        attempts = journal.snapshot().attempts
        result = {"case_id": row["case_id"], "arm": row["arm"], "generation_id": generation,
                  "lane_id": lane, "static_admission": "NOT_AVAILABLE"}
        if not attempts or attempts[-1]["status"] != "completed":
            result["reason"] = "provider_attempt_not_completed"
            rows.append(result)
            continue
        attempt = attempts[-1]
        record = attempt["provider_attempt"]
        if record.get("provider_response_mode") != "candidate_draft":
            result.update(static_admission="NOT_APPLICABLE", reason=record.get("provider_response_mode", "no_candidate"))
            rows.append(result)
            continue
        public_candidate = record["candidate"]
        candidate_id = public_candidate["candidate_id"]
        matches = list((args.runtime_output / "provider_candidates" / generation / lane).glob(f"*_{candidate_id}"))
        if len(matches) != 1:
            raise ValueError(f"candidate package is not uniquely bound to its runtime slot: {row['case_id']}")
        candidate = CandidatePackageRef(candidate_id=candidate_id, path=str(matches[0]),
            change_set=CandidateChangeSet(frozenset(public_candidate["changed_components"])))
        payload = verify_prompt_payload(PromptPayloadRef.from_mapping(attempt["prompt_payload"]))
        parent_source, parent_hash, role = select_authoring_anchor_source(payload)
        old = cases[row["case_id"]]
        context_spec = exp.bound_read(old["author_context"], args.p3_cases.parent)
        context_key = old["author_context"]["sha256"]
        if context_key not in contexts:
            contexts[context_key] = exp.reference_context(context_spec, root=args.root, base=args.p3_cases.parent,
                output=args.output.parent / "static_contexts" / old["task"])
        comp, best, _, qualified_source, *_ = contexts[context_key]
        if role != "current_best" or parent_source != qualified_source or best.checkpoint_hash != old["checkpoint_hash"]:
            raise ValueError("static verification parent differs from qualified and prompt-visible parent")
        policy = comp.candidate_admission_policy
        verification = policy.verify(candidate)
        base_status = verification.status
        if base_status == "pass":
            verification = policy.verify_authoring_delta(candidate=candidate, verification=verification, parent_source=parent_source)
        result.update(candidate_id=candidate_id, candidate_source_sha256=hashlib.sha256((matches[0] / "training_code" / "components.py").read_bytes()).hexdigest(),
            prompt_checksum=attempt["prompt_payload"]["checksum"], parent_source_sha256=parent_hash,
            static_package_status=base_status, static_admission=verification.status.upper(),
            verification=verification.to_public_dict(), cpu_smoke="NOT_RUN", train_status="NOT_RUN")
        rows.append(result)
    snapshot = project_source_snapshot(args.root)
    report = {"schema_version": "feedback_p3r.runtime_candidate_static.v1",
        "scope": "Production task static verification and parent-bound authoring-delta admission only; CPU smoke and semantics not assessed.",
        "runtime_gate_mutated": False, "api_calls": 0, "candidate_count": sum(r.get("candidate_id") is not None for r in rows),
        "passed": sum(r["static_admission"] == "PASS" for r in rows), "failed": sum(r["static_admission"] == "FAIL" for r in rows),
        "not_applicable": sum(r["static_admission"] == "NOT_APPLICABLE" for r in rows),
        "pending": sum(r["static_admission"] == "NOT_AVAILABLE" for r in rows),
        "validation_source_hash": canonical_json_hash(snapshot),
        "validation_source_scope": "Independent full production source closure; not a claim that the original provider-only runtime snapshot contained these resources.",
        "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--p3-tools", type=Path, default=Path(__file__).resolve().parent.parent / "P3")
    parser.add_argument("--p3-cases", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--runtime-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.root / "src"))
    report = verify(args)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
