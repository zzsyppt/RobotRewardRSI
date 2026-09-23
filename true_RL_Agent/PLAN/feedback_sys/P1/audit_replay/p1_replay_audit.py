"""Audit driver: replay a real generation session through the shared host and prompt builder.

Modeled on PLAN/2new_world/P6/audit_shared_public_history.py. Read-only over the
session; outputs go to a scratch directory. Used to compare the P1 working tree
against the fe96455d baseline on real ledgers (formal c6 Stair, attempt_08 Push).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--task", choices=("stair", "push"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", default="")
    parser.add_argument("--label", default="run")
    parser.add_argument("--context", choices=("scratch", "session"), default="scratch")
    parser.add_argument("--memory-variant", choices=("full", "sparse", "sparse_notop", "token"), default="full")
    args = parser.parse_args()
    root = args.root.resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(root / "src"), str(root)]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"label": args.label, "root": str(root), "task": args.task, "source": str(args.source),
              "generation_record_sha256": hashlib.sha256(
                  (root / "src/true_rl_agent/agent_core/generation_record.py").read_bytes()).hexdigest()[:12]}
    try:
        run(args, root, output, report)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()[-3000:]
    (output / "report.json").write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k not in ("traceback",)}, default=str), flush=True)


def run(args, root, output, report):
    from true_rl_agent.agent_core.generation_contract import GenerationSessionContract
    from true_rl_agent.agent_core.generation_runtime import generation_public_history
    from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, load_ledger, primary_ledger_epoch_hash
    from true_rl_agent.candidates.prompt_projection import assert_public_only_prompt_payload
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
    from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
    from true_rl_agent.memory.context_builder import PublicContextBuilder
    from true_rl_agent.prompt_serialization import canonical_prompt_json, canonical_prompt_json_bytes

    if args.task == "stair":
        from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider as Provider
    else:
        from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider as Provider

    source = args.source.resolve()
    genesis = read(source / "session_genesis.json")
    manifest = read(source / "resolved_cell_manifest.json")
    ledger = load_ledger(source / "ledger.jsonl")
    provider = Provider()
    if args.context == "session":
        # Production resume path: read-only against the frozen session manifest.
        context = TaskCompositionContext(project_root=root, output_dir=source, read_only=True)
    else:
        context = TaskCompositionContext(project_root=root, output_dir=output / "composition")
    comp = provider.resolve(
        values={**provider.cli_defaults(), "cell_variant_id": manifest["cell_variant_id"]},
        context=context,
    )
    report["composition_context"] = args.context
    task_spec = comp.task_plugin.task_spec
    protocol = task_spec.acceptance_protocol
    anchor = genesis["anchor"]
    public, reference = anchor["public_result"], anchor["reference_manifest"]
    binding = EvaluationBindingV1.from_mapping(genesis["evaluation_binding"])
    best = CurrentBestEvalStats.from_reference_public_result_mapping(
        reference_id=reference["reference_id"], reference_manifest_hash=reference["manifest_hash"],
        program_source_checksum=reference["source_checksum"],
        training_binding_hash=anchor["training_binding"]["binding_hash"],
        capability_profile_hash=reference["capability_profile_hash"],
        checkpoint_hash=anchor["checkpoint_hash"], public_result=public,
        acceptance_protocol=protocol, expected_evaluation_binding=binding,
    )
    evidence = {"anchor_kind": "reference_program", "public_result": public,
                "public_result_hash": canonical_json_hash(public)}
    comp.task_plugin.bind_genesis_evidence(evidence)
    reference_source = comp.reference_program_artifact.source_text
    assert "sha256:" + hashlib.sha256(reference_source.encode()).hexdigest() == reference["source_checksum"]
    contract = GenerationSessionContract.from_mapping(read(source / "generation_session_contract.json"))
    phased = comp.task_plugin.phased_evaluation_adapter()
    foundation = SimpleNamespace(composition=comp, phased_adapter=phased, session_contract=contract)
    genesis_view = SimpleNamespace(initial_current_best=best,
        session_genesis_hash=genesis["session_genesis_hash"],
        expected_initial_policy_state_hash=anchor["initial_policy_state_hash"])
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter,
        task_prompt_adapter=comp.task_plugin.prompt_adapter(), phased_evaluation_adapter=phased,
        acceptance_protocol=protocol,
        initial_current_best=best, session_genesis_hash=genesis["session_genesis_hash"],
        candidate_capability_profile=task_spec.candidate_capability_profile,
        initial_reference_program_source=reference_source, initial_anchor_public_evidence=evidence,
        expected_initial_policy_state_hash=anchor["initial_policy_state_hash"],
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(genesis["session_genesis_hash"]),
        ledger_authoring_mode=PROVIDER_AUTHORING_MODE, generation_session_contract=contract)
    report["adapter"] = type(phased).__name__
    report["ledger_records"] = len(ledger)
    report["ledger_schema"] = sorted({r["schema_version"] for r in ledger})

    report["memory_variant"] = args.memory_variant
    if args.memory_variant != "full":
        _apply_memory_variant(args.memory_variant)
        _frames_only(args, output, report, builder, comp, ledger, assert_public_only_prompt_payload,
                     canonical_prompt_json_bytes)
        return

    # 1. Full-ledger replay through the production history projection.
    started = time.monotonic()
    history = generation_public_history(foundation, genesis=genesis_view, ledger=ledger)
    report["full_replay"] = {"elapsed_s": round(time.monotonic() - started, 2), "records": len(history.records)}
    checked = equal = 0
    lane_measurements = []
    for raw, rec in zip(ledger, history.records, strict=True):
        for t_raw, t in zip(raw["trials"], rec["generation"]["trials"], strict=True):
            for phase in ("public_dev", "public_accept"):
                if t_raw.get(phase) is not None:
                    checked += 1
                    equal += t_raw[phase]["public_summary"] == t[phase]["public_summary"]
            ras = t.get("runtime_activation_summary")
            lane_measurements.append({
                "generation": rec["generation"]["generation_index"], "lane": t["lane_id"],
                "trial_status": t["trial_status"],
                "runtime_activation_summary_keys": sorted(ras.keys()) if isinstance(ras, dict) else ras,
            })
    report["phase_summaries_checked"] = checked
    report["phase_summaries_equal_to_ledger"] = equal
    report["lane_runtime_summary_sample"] = lane_measurements[:4]
    report["current_best_after"] = history.current_best.checkpoint_hash

    # 2. Prompt frames: bytes, memory retention, determinism.
    report["frames"] = []
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    for next_generation in frames:
        prefix = ledger[:next_generation - 1]
        kwargs = dict(authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
                      candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules())
        started = time.monotonic()
        payload = builder.build_prompt(prefix, **kwargs).provider_payload
        elapsed = time.monotonic() - started
        payload_again = builder.build_prompt(prefix, **kwargs).provider_payload
        assert_public_only_prompt_payload(payload)
        memory = payload["public_history"]["experiment_memory"]
        experiments = memory.get("experiments", [])
        rows = [row for e in experiments for row in e.get("trial_outcomes", [])]
        runtime_feedback = payload.get("runtime_feedback_summary")
        item = {"next_generation": next_generation, "build_s": round(elapsed, 2),
                "prompt_bytes": canonical_prompt_json_bytes(payload),
                "memory_bytes": canonical_prompt_json_bytes(memory),
                "omitted_experiment_count": memory.get("omitted_experiment_count"),
                "experiments_retained": len(experiments),
                "trial_outcome_columns": memory.get("trial_outcome_columns"),
                "trial_rows": len(rows), "row_widths": sorted({len(r) for r in rows}),
                "summarized_entries": sum(1 for e in experiments if "candidate_id" not in e and e.get("trial_outcomes")),
                "last_row_measurements": [r[-1] for r in rows[-2:]] if rows and isinstance(rows[-1][-1], dict) else None,
                "top_runtime_observation": experiments[-1].get("runtime_observation") if experiments else None,
                "runtime_feedback_status": runtime_feedback.get("status") if isinstance(runtime_feedback, dict) else None,
                "deterministic": payload == payload_again}
        (output / f"payload_before_g{next_generation:02d}.json").write_text(canonical_prompt_json(payload), encoding="utf-8")
        report["frames"].append(item)
        print(json.dumps(item, default=str), flush=True)

    # 3. Real-data tamper: only the ledger's public_summary changes.
    tampered = copy.deepcopy(ledger[:2])
    trial = next(t for t in tampered[0]["trials"] if t.get("public_dev"))
    summary = trial["public_dev"]["public_summary"]
    target = _first_numeric_path(summary)
    _set_path(summary, target, _get_path(summary, target) + 0.001)
    try:
        generation_public_history(foundation, genesis=genesis_view, ledger=tampered)
        report["summary_only_tamper"] = {"path": target, "rejected": False}
    except ValueError as exc:
        report["summary_only_tamper"] = {"path": target, "rejected": True, "error": str(exc)}


def _apply_memory_variant(name):
    """Monkeypatch alternative encodings of the per-lane measurement column (audit only)."""
    from true_rl_agent.candidates import mechanism
    from true_rl_agent.agent_core import experiment_memory as em
    if name in ("sparse", "sparse_notop"):
        original = mechanism.RuntimeActivationEvidence.measurement_summary

        def sparse(self):
            return {k: v for k, v in original(self).items() if v is not None}
        mechanism.RuntimeActivationEvidence.measurement_summary = sparse
    if name in ("sparse_notop", "token"):
        orig_compact = em._compact_entry

        def compact_no_top(entry):
            entry = dict(entry)
            entry.pop("runtime_observation", None)
            return orig_compact(entry)
        em._compact_entry = compact_no_top
    if name == "token":
        orig_outcome = em._trial_outcome

        def token_outcome(trial):
            row = orig_outcome(trial)
            m = row[-1]
            if m.get("status") != "available":
                tok = "not_available"
            elif m.get("step_calls") == 0:
                tok = "not_executed"
            elif m.get("reward_term_count") == 0:
                tok = "no_terms"
            elif m.get("reward_term_calls") == 0:
                tok = "terms_not_called"
            elif m.get("nonzero_reward_contribution") is False:
                tok = "executed_zero"
            elif m.get("nonzero_reward_contribution") is True:
                tok = "executed_nonzero"
            else:
                tok = "executed_value_unknown"
            row[-1] = tok
            return row
        em._trial_outcome = token_outcome


def _frames_only(args, output, report, builder, comp, ledger, assert_public_only_prompt_payload, nbytes):
    report["frames"] = []
    for next_generation in [int(x) for x in args.frames.split(",") if x.strip()]:
        prefix = ledger[:next_generation - 1]
        payload = builder.build_prompt(prefix,
            authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
            candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules()).provider_payload
        assert_public_only_prompt_payload(payload)
        memory = payload["public_history"]["experiment_memory"]
        experiments = memory.get("experiments", [])
        rows = [row for e in experiments for row in e.get("trial_outcomes", [])]
        item = {"variant": args.memory_variant, "next_generation": next_generation,
                "prompt_bytes": nbytes(payload), "memory_bytes": nbytes(memory),
                "omitted_experiment_count": memory.get("omitted_experiment_count"),
                "experiments_retained": len(experiments), "trial_rows": len(rows),
                "last_rows_measure": [r[-1] for r in rows[-2:]],
                "has_top_runtime_observation": any("runtime_observation" in e for e in experiments)}
        report["frames"].append(item)
        print(json.dumps(item, default=str), flush=True)


def _first_numeric_path(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            found = _first_numeric_path(item, path + (key,))
            if found:
                return found
    elif isinstance(value, float):
        return path
    return None


def _get_path(value, path):
    for key in path:
        value = value[key]
    return value


def _set_path(value, path, new):
    for key in path[:-1]:
        value = value[key]
    value[path[-1]] = new


if __name__ == "__main__":
    main()
