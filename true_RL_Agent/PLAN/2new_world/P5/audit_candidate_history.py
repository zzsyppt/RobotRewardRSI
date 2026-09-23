"""Read immutable candidate/train receipts and run only the CPU reward fixture."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]


def main():
    import torch
    from true_rl_agent.candidates.component_source_semantics import project_single_component_operation_delta
    from true_rl_agent.candidates.verifier import StaticCandidateVerifierPolicy
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.evaluation.subject import ExperimentSubject
    from true_rl_agent.tasks.stair_climbing.candidate_admission import _duplicate_parent_reward_term_issue
    from true_rl_agent.tasks.stair_climbing.component_smoke import StairClimbingComponentSmokeAdapter
    from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
    from true_rl_agent.tasks.stair_climbing.reward_program import qualify_stair_diagnostic_parent

    output = Path(__file__).parent / "audit_candidate_history.json"
    report = {"purpose": "CPU fixture observations versus saved training samples; no new training or acceptance", "trials": []}
    baseline_path = ROOT / ".tmp/p5_component_semantics_baseline.py"
    spec = importlib.util.spec_from_file_location("true_rl_agent.candidates._p5_baseline_semantics", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    report["pre_p5_analyzer_sha256"] = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    provider = StairClimbingCompositionProvider()
    for suffix in ("c6ab8ba", "83824cd"):
        root = ROOT / "outputs" / ("formal_go2_stair_15round_20260901_" + suffix)
        manifest = json.loads((root / "resolved_cell_manifest.json").read_text())
        comp = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": manifest["cell_variant_id"]},
            context=TaskCompositionContext(project_root=ROOT, output_dir=output.parent / "audit_candidate_composition"))
        adapter = StairClimbingComponentSmokeAdapter(comp.task_plugin.config, comp.resolved_cell_manifest)
        inputs = adapter.build_reward_inputs()
        reference = comp.reference_program_artifact.source_text
        sources = {"sha256:" + hashlib.sha256(reference.encode()).hexdigest(): reference}
        for record in map(json.loads, (root / "ledger.jsonl").read_text().splitlines()):
            artifacts = {a["kind"]: a for a in record["private_artifacts"]}
            for trial in record["trials"]:
                if not trial["train"]:
                    continue
                candidate = trial["candidate"]
                package = Path(artifacts[trial["lane_id"] + "_candidate_artifact"]["uri"])
                source_bytes = (package / "training_code/components.py").read_bytes()
                assert "sha256:" + hashlib.sha256(source_bytes).hexdigest() == candidate["program_source_checksum"]
                source = source_bytes.decode("utf-8")
                sources[candidate["program_source_checksum"]] = source
                delta = trial["mechanism_operation_delta"]
                parent = sources[delta["anchor_source_checksum"]]
                # Compare the pre-P5 and current analyzer in the SAME Python.
                # Original Python 3.11 AST hashes are historical identity, not
                # rewritten to match a Python 3.12 re-analysis.
                current_delta = project_single_component_operation_delta(candidate_source=source, anchor_source=parent)
                assert current_delta == baseline.project_single_component_operation_delta(candidate_source=source, anchor_source=parent)
                program = qualify_stair_diagnostic_parent(contract=adapter.contract,
                    subject=ExperimentSubject.candidate(candidate["candidate_id"]),
                    subject_manifest_hash=candidate["subject_manifest_hash"], source_bytes=source_bytes,
                    expected_source_checksum=candidate["program_source_checksum"], source_name=str(package / "training_code/components.py"),
                    static_verifier_policy=StaticCandidateVerifierPolicy(), trusted_reference_source_text=reference)
                component = delta.get("after_component") or delta["before_component"]
                name = component["mechanism_id"].removeprefix("RewardTerm:")
                term = next(t for t in program.reward_terms if t.name == name)
                raw = term.fn(inputs)
                clipped = raw.clamp(min=term.clip_min, max=term.clip_max) if term.clip_min is not None or term.clip_max is not None else raw
                saturation = torch.zeros_like(raw, dtype=torch.bool)
                if term.clip_min is not None:
                    saturation |= raw < term.clip_min
                if term.clip_max is not None:
                    saturation |= raw > term.clip_max
                receipt_ref = artifacts[trial["lane_id"] + "_train_receipt"]
                blob = Path(receipt_ref["uri"]).read_bytes()
                assert "sha256:" + hashlib.sha256(blob).hexdigest() == receipt_ref["checksum"]
                summary = json.loads(blob)["train_result"]["summary"]
                runtime = next(t for t in summary["runtime_activation_summary"]["component_activation_records"] if t["name"] == name)
                report["trials"].append({"run": suffix, "generation": record["generation_index"], "lane": trial["lane_id"],
                    "candidate_id": candidate["candidate_id"], "status": trial["trial_status"], "source_checksum": candidate["program_source_checksum"],
                    "mechanism": name, "fixture_raw": raw.tolist(), "fixture_weighted": (clipped * term.weight).tolist(),
                    "fixture_nonzero_count": int(torch.count_nonzero(clipped * term.weight)), "fixture_clipped_count": int(saturation.sum()),
                    "train_max_abs_weighted_contribution": runtime["max_abs_weighted_contribution"],
                    "current_admission_issue": _duplicate_parent_reward_term_issue(candidate_source=source, parent_source=parent),
                    "source_bytes_unchanged": True, "pre_p5_analysis_unchanged": True,
                    "historical_delta_hash_matches_current_python": current_delta == delta})
    rows = report["trials"]
    report["summary"] = {"trained_trials": len(rows),
        "saved_training_zero_terms": [r["candidate_id"] for r in rows if r["train_max_abs_weighted_contribution"] == 0],
        "current_fixture_zero_or_all_clipped": sum(r["fixture_nonzero_count"] == 0 or r["fixture_clipped_count"] == 5 for r in rows),
        "current_numeric_admission_rejections": [r["candidate_id"] for r in rows if r["current_admission_issue"]],
        "all_source_bytes_and_pre_p5_analysis_unchanged": all(r["source_bytes_unchanged"] and r["pre_p5_analysis_unchanged"] for r in rows)}
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"]))


if __name__ == "__main__":
    main()
