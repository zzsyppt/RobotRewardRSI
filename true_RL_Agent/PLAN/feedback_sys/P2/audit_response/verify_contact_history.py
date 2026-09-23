"""Read-only CPU audit of saved Stair v2 contact-history operands.

This checks a stored sensor predicate, not collision surfaces or causality.
Full decoder/outcome/summary equivalence remains the paired Stair verifier's job.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def verify_contact_history(reference):
    import numpy as np
    import torch
    from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
    from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
    from true_rl_agent.tasks.stair_climbing.numeric_capture import STAIR_NUMERIC_PROFILE, STAIR_NUMERIC_PROFILE_V1
    from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest

    manifest = load_numeric_evidence_manifest(reference)
    context = manifest["metadata"]
    report = {"profile": context["profile"], "identity": manifest["identity"], "reference": reference}
    if context["profile"] == STAIR_NUMERIC_PROFILE_V1:
        return {**report, "status": "NOT_RECORDED", "contact_force_history": {"status": "not_recorded"}}
    if context["profile"] != STAIR_NUMERIC_PROFILE:
        raise ValueError("unsupported Stair numeric profile")
    declaration = context["contact_force_history"]
    if declaration.get("status") != "recorded":
        raise ValueError("v2 must declare recorded contact history")
    header_contract = {key: value for key, value in declaration.items() if key not in {"status", "array_axes"}}
    projection = stair_runtime_projection_from_manifest(ResolvedCellManifestV1.from_mapping(context["resolved_cell_manifest"]))
    signals = context["semantics"]["episode_outcomes"]["signals"]
    threshold = float(signals["forbidden_contact"]["force_threshold_n"])
    support_threshold = float(signals["slip"]["contact_force_threshold"])
    term = signals["forbidden_contact"]["manager_term"]
    counts = dict(valid_control_frames=0, invalid_padding_frames=0, predicate_mismatches=0,
                  manager_contact_frames=0, history_contact_frames=0, manager_contact_terminal_frames=0,
                  instantaneous_zero_history_nonzero_terminal_frames=0,
                  instantaneous_below_threshold_history_above_terminal_frames=0,
                  support_history_vs_instantaneous_disagree_body_frames=0,
                  valid_support_body_frames=0, nonzero_history_in_invalid_padding=0)
    first_mismatches, dtype_sets = [], {}
    max_chunk_bytes, total_raw_bytes = 0, 0
    for metadata, arrays in iter_numeric_evidence(reference):
        if metadata["transition_header"].get("contact_force_history") != header_contract:
            raise ValueError("chunk history declaration differs")
        valid = torch.from_numpy(arrays["valid"])
        if valid.ndim != 2:
            raise ValueError("valid must have control-step/environment axes")
        for name, body_count in (("support_contact_forces_w_history", projection.support_count),
                                 ("forbidden_contact_forces_w_history", projection.forbidden_contact_count)):
            if name not in arrays or arrays[name].shape != (*valid.shape, projection.contact_history_length, body_count, 3):
                raise ValueError("history array missing or axes differ from bound roles")
            if arrays[name].dtype.kind != "f" or not np.isfinite(arrays[name]).all():
                raise ValueError("contact history must contain finite floating-point vectors")
            counts["nonzero_history_in_invalid_padding"] += int(np.count_nonzero(arrays[name][~arrays["valid"]]))
            dtype_sets.setdefault(name, set()).add(str(arrays[name].dtype))
        # Same operation order as the installed illegal_contact predicate:
        # norm xyz -> max over history -> threshold -> any bound forbidden body.
        history_norm = torch.linalg.norm(torch.from_numpy(arrays["forbidden_contact_forces_w_history"]), dim=-1)
        history_contact = (history_norm.max(dim=2).values > threshold).any(dim=2)
        instantaneous_norm = torch.linalg.norm(torch.from_numpy(arrays["forbidden_contact_forces_w"]), dim=-1)
        instantaneous_contact = (instantaneous_norm > threshold).any(dim=2)
        instant_zero = (instantaneous_norm == 0).all(dim=2)
        terms = metadata["termination_term_order"]
        recorded_term = torch.from_numpy(arrays[f"termination_{terms.index(term)}"])
        mismatch = valid & (history_contact != recorded_term)
        terminal = valid & recorded_term & torch.from_numpy(arrays["done_mask"])
        support_history = torch.linalg.norm(torch.from_numpy(arrays["support_contact_forces_w_history"]), dim=-1).max(dim=2).values > support_threshold
        support_instant = torch.linalg.norm(torch.from_numpy(arrays["support_contact_forces_w"]), dim=-1) > support_threshold
        counts["valid_control_frames"] += int(valid.sum())
        counts["invalid_padding_frames"] += int((~valid).sum())
        counts["predicate_mismatches"] += int(mismatch.sum())
        counts["manager_contact_frames"] += int((valid & recorded_term).sum())
        counts["history_contact_frames"] += int((valid & history_contact).sum())
        counts["manager_contact_terminal_frames"] += int(terminal.sum())
        counts["instantaneous_zero_history_nonzero_terminal_frames"] += int((terminal & instant_zero & history_contact).sum())
        counts["instantaneous_below_threshold_history_above_terminal_frames"] += int((terminal & ~instantaneous_contact & history_contact).sum())
        counts["valid_support_body_frames"] += int(valid.sum()) * projection.support_count
        counts["support_history_vs_instantaneous_disagree_body_frames"] += int((valid.unsqueeze(-1) & (support_history != support_instant)).sum())
        for step_index, env_index in mismatch.nonzero().tolist()[:max(0, 8-len(first_mismatches))]:
            first_mismatches.append({"run_index": metadata["run_index"], "control_step": metadata["step_start"] + step_index,
                                     "env_index": env_index, "history_max_N": float(history_norm[step_index, env_index].max()),
                                     "recorded_term": bool(recorded_term[step_index, env_index])})
        chunk_bytes = sum(array.nbytes for array in arrays.values())
        max_chunk_bytes = max(max_chunk_bytes, chunk_bytes)
        total_raw_bytes += chunk_bytes
    directory = Path(reference["manifest_path"]).parent
    archive_bytes = sum(path.stat().st_size for path in directory.iterdir() if path.is_file())
    passed = (counts["valid_control_frames"] > 0 and counts["predicate_mismatches"] == 0
              and counts["nonzero_history_in_invalid_padding"] == 0
              and max_chunk_bytes <= 64 * 1024**2 and archive_bytes <= 2 * 1024**3)
    return {**report, "status": "PASS" if passed else "FAIL", "contact_force_history": declaration,
            "resolved_roles": context["runtime_projection"]["resolved_roles"],
            "forbidden_threshold_N": threshold, "support_threshold_N": support_threshold,
            "counts": counts, "first_mismatches": first_mismatches,
            "dtypes": {name: sorted(values) for name, values in dtype_sets.items()},
            "resources": {"max_chunk_raw_bytes": max_chunk_bytes, "total_raw_bytes": total_raw_bytes,
                          "archive_disk_bytes": archive_bytes},
            "boundaries": ["A terminal count here is the nonexclusive manager term plus done mask, not primary_reason classification.",
                           "Support disagreement compares two saved operands; it does not change the existing instantaneous decoder semantics.",
                           "No contact pair, contact surface, per-slot timestamp or causal explanation is inferred.",
                           "Run verify_stair_pair.py separately for complete outcome/summary equivalence and on/off timing/RSS."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--on-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.project_root.resolve() / "src"))
    result = json.loads((args.on_dir / "public_dev_result.json").read_text(encoding="utf-8"))
    report = verify_contact_history(result["task_evidence"]["numeric_evidence"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
