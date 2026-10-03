"""Reusable numeric evidence and card artifact writers, also used by offline CLIs."""
from __future__ import annotations

from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

from true_rl_agent.evaluation.trajectory_translation.contracts import CATALOG_VERSION, EXTRACTOR_VERSION
from true_rl_agent.evaluation.trajectory_translation.measurement import NUMERIC_CONFIG, aggregate_fractions, measure_episode
from true_rl_agent.evaluation.trajectory_translation.cohort import ACTIVITY_VERSION, COHORT_VERSION, POOLED_ZERO_TOLERANCE
from true_rl_agent.evaluation.trajectory_translation.report import CARD_VERSION, TEMPLATE_VERSION, SELECTION_VERSION, build_card, file_sha256, load_bundle, render_markdown
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import iter_stair_episodes

ROOT = Path(__file__).resolve().parents[4]

LIMITATIONS = {
    "contact": "Current net normal force threshold markers are sampled readouts, not uninterrupted physical contact or load shares.",
    "contact_force_scope": "Only bound support and forbidden bodies are recorded; source.input_inventory lists their exact entities. All other nonfoot body forces are unrecorded.",
    "history": "Stored history slots have no observed timestamps/validity; not a substep trajectory or integral.",
    "motion": "Historical link-position and COM-velocity reference mapping is unverified; no exact derivative equivalence.",
    "outcome": "Archived task evaluator is authoritative; numeric processes do not assign causes or change outcomes.",
    "deferred_obligations": ["periodic_phase_qualification", "nominal_or_effective_joint_limit_binding",
                             "independent_per_geometry_terrain_validation", "historical_link_COM_mapping"],
    "missing_capabilities": ["raw_action", "complete_policy_observation", "realized_command_series", "joint_velocity",
                             "torque_and_saturation", "actual_reward_components_and_gates", "sensor_air_time_first_contact",
                             "actual_mass_and_whole_body_momentum", "tangential_force_and_contact_objects",
                             "all_nonfoot_body_contact_forces", "parent_archive"],
    "not_implemented": ["policy_card", "author_projection", "query_AST", "visual_analysis", "automatic_unknown_pattern_discovery"],
}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def summarize_cohort(summaries: list[dict[str, Any]], support: tuple[str, ...]) -> dict[str, Any]:
    """Audit arithmetic for this one source only, not a policy-card aggregator."""
    groups = {}
    for group in ("all", "success", "failure"):
        members = [s for s in summaries if group == "all" or s["success"] == (group == "success")]
        norm = np.zeros(len(support))
        z = np.zeros(len(support))
        cell_norm = np.zeros(len(support))
        cell_z = np.zeros(len(support))
        combination = np.zeros(1 << len(support), dtype=np.int64)
        for s in members:
            f = s["measurements"]["force_summary"]
            norm += f["sample_sum_norm_n"]
            z += f["sample_sum_signed_fz_n"]
            cell_norm += f["rectangle_norm_ns"]
            cell_z += f["rectangle_signed_fz_ns"]
            combination += s["measurements"]["contact_combinations"]["counts"]
        def share(values):
            return (values / values.sum()).tolist() if abs(values.sum()) > 1e-12 else None
        equal_force = {}
        for key in ("norm_share", "signed_fz_share"):
            vals = [s["measurements"]["force_summary"][key] for s in members
                    if all(v is not None for v in s["measurements"]["force_summary"][key])]
            equal_force[key] = {"episode_equal_mean": np.mean(vals, axis=0).tolist() if vals else None,
                                "valid_episode_count": len(vals), "unknown_episode_count": len(members) - len(vals)}
        groups[group] = {"episodes": len(members), "valid_rows": sum(s["rows"] for s in members),
            "marker_fractions": {entity: aggregate_fractions([s["measurements"][f"{entity}.marker"] for s in members]) for entity in support},
            "on_counts": [sum(s["measurements"][f"{entity}.marker"]["true_count"] for s in members) for entity in support],
            "sample_sum_norm_n": norm.tolist(), "sample_sum_signed_fz_n": z.tolist(),
            "sample_cell_weighted_norm_share": share(cell_norm), "sample_cell_weighted_signed_fz_share": share(cell_z),
            "force_episode_equal": equal_force, "combination_counts": combination.tolist(),
            "joint_force_valid_rows": sum(s["measurements"]["force_summary"]["joint_valid_count"] for s in members),
            "weighting": "episode_equal and sample_cell_duration_weighted reported separately; no physical impulse claim"}
    return {"scope": "one_source_whole_recorded_episodes_no_stage_restriction", "support_order": list(support),
            "outcome_counts": dict(sorted(Counter(s["primary_reason"] for s in summaries).items())), "groups": groups}


def build_evidence(reference: dict[str, Any], output: Path, source_id: str,
                   selection: set[tuple[int, int]] | None = None) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    partial = output / "evidence.jsonl.partial"
    counts: Counter = Counter()
    summaries = []
    matched = set()
    episode_ids = set()
    source = None
    support = ()
    scanned_episodes = scanned_rows = 0
    with partial.open("x", encoding="utf-8", newline="\n") as stream:
        for episode in iter_stair_episodes(reference, source_id=source_id):
            scanned_episodes += 1
            scanned_rows += len(episode.steps)
            if selection is not None and (episode.run_index, episode.env_index) not in selection:
                continue
            matched.add((episode.run_index, episode.env_index))
            if episode.episode_id in episode_ids:
                raise ValueError("duplicate episode identity")
            episode_ids.add(episode.episode_id)
            source, support = episode.source, episode.entities["support"]
            header = {"record_type": "episode", "episode_id": episode.episode_id,
                "display_name": f"{source_id}.r{episode.run_index}.e{episode.env_index}",
                "source": source, "run_index": episode.run_index, "env_index": episode.env_index,
                "entities": episode.entities, "task": episode.task,
                "coverage": {"first_step": int(episode.steps[0]), "last_step": int(episode.steps[-1]),
                    "valid_rows": len(episode.steps), "dt_s": episode.dt_s, "raw": "processed_all_supplied_valid_rows",
                    "media": "not_inspected", "model_inspection": "none", "semantic": "independent_acceptance_required"}}
            stream.write(json.dumps(header, separators=(",", ":"), allow_nan=False) + "\n")
            counts["episode"] += 1
            measurements = {}
            ids = set()
            records = measure_episode(episode)
            for record in records:
                if record["record_id"] in ids:
                    raise ValueError("duplicate evidence record identity")
                ids.add(record["record_id"])
                stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
                counts[record["record_type"]] += 1
                if record["record_type"] == "measurement":
                    measurements[record["key"]] = record["value"]
            for record in records:
                operands = record["value"].get("operand_record_ids", [])
                if not all(operand in ids for operand in operands):
                    raise ValueError("unresolved temporal relation operand")
            summaries.append({"rows": len(episode.steps), "success": episode.task["outcome"]["success"],
                              "primary_reason": episode.task["outcome"]["primary_reason"], "measurements": measurements})
    if source is None or (selection is not None and matched != selection):
        raise ValueError("requested episodes are absent; no complete artifact published")
    partial.replace(output / "evidence.jsonl")
    code_paths = [Path(__file__), *(ROOT / "src/true_rl_agent/evaluation/trajectory_translation").glob("*.py"),
                  ROOT / "src/true_rl_agent/tasks/stair_climbing/trajectory_adapter.py"]
    manifest = {"schema_version": "trajectory.numeric_manifest.v1", "complete": True,
        "extractor_version": EXTRACTOR_VERSION, "catalog_version": CATALOG_VERSION, "numeric_config": NUMERIC_CONFIG,
        "source_manifest_ref": reference, "source_id": source_id, "acceptance_split": "public_dev",
        "selection": sorted([list(pair) for pair in selection]) if selection is not None else "all_episodes",
        "scanned_episodes": scanned_episodes, "scanned_valid_rows": scanned_rows,
        "emitted_episodes": len(summaries), "emitted_valid_rows": sum(s["rows"] for s in summaries),
        "record_counts": dict(counts), "limitations": {**LIMITATIONS, **(
            {"history": "Not recorded in Stair numeric v1; unknown, no history events or slots. Manager termination is independent."}
            if source["contact_force_history"].get("status") == "not_recorded" else {})},
        "evaluation_uses": ["development_regression", "descriptive_audit"], "prospective_confirmation": False,
        "cohort_numeric_summary": summarize_cohort(summaries, support),
        "evidence": {"path": "evidence.jsonl", "sha256": _sha(output / "evidence.jsonl"),
                     "bytes": (output / "evidence.jsonl").stat().st_size},
        "code_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): _sha(path) for path in code_paths}}
    _write_json(output / "manifest.json", manifest)
    return manifest



def peak_memory_bytes() -> int:
    if sys.platform == "win32":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                        "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                        "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]
        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(MemoryCounters), ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return counters.PeakWorkingSetSize
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _load(path: Path, expected: str) -> dict:
    if file_sha256(path) != expected:
        raise ValueError("P1a manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    return load_bundle(path, expected, iter_stair_episodes(manifest["source_manifest_ref"], source_id=manifest["source_id"]))


def build_card_artifacts(primary_path: Path, primary_sha: str, example_inputs: list[tuple[Path, str]],
          selection_path: Path | None, selection_sha: str | None, output: Path) -> dict:
    started = time.perf_counter()
    if selection_path is None:
        if selection_sha is not None or example_inputs:
            raise ValueError("automatic selection takes only its own complete archive")
        selected = {"episodes": [], "source_manifest_hashes": {}}
        selection_ref = {"mode": "current_archive_deterministic_examples.v1", "frozen_examples": False}
    else:
        if file_sha256(selection_path) != selection_sha:
            raise ValueError("frozen selection hash mismatch")
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        selection_ref = {"path": str(selection_path.resolve()), "sha256": selection_sha}
    output.mkdir(parents=True, exist_ok=False)
    primary = _load(primary_path, primary_sha)
    if primary["manifest"]["selection"] != "all_episodes":
        raise ValueError("primary card requires the full supplied archive, not an example subset")
    examples = [_load(path, sha) for path, sha in example_inputs]
    card, evidence = build_card(primary, examples, selected)
    with (output / "evidence.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for record in evidence:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    _write_json(output / "policy_card.json", card)
    with (output / "policy_card_zh.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(render_markdown(card))
    artifacts = {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                 for name in ("evidence.jsonl", "policy_card.json", "policy_card_zh.md")}
    code = [Path(__file__), ROOT / "src/true_rl_agent/evaluation/trajectory_translation/cohort.py",
            ROOT / "src/true_rl_agent/evaluation/trajectory_translation/report.py",
            *(ROOT / "src/true_rl_agent/evaluation/trajectory_translation" / name for name in ("contracts.py", "measurement.py", "temporal.py")),
            ROOT / "src/true_rl_agent/tasks/stair_climbing/trajectory_adapter.py", ROOT / "src/true_rl_agent/runner/numeric_evidence.py"]
    manifest = {"schema_version": "trajectory.card_manifest.v1", "complete": True,
                "card_version": CARD_VERSION, "cohort_version": COHORT_VERSION, "activity_version": ACTIVITY_VERSION,
                "template_version": TEMPLATE_VERSION, "selection_version": SELECTION_VERSION,
                "acceptance_split": "public_dev", "catalog_version": card["catalog_version"],
                "selection_ref": selection_ref,
                "source_files": {b["manifest"]["evidence"]["sha256"]: b["input"] for b in [primary, *examples]},
                "primary_binding": primary["binding"], "primary_episodes": len(primary["summaries"]),
                "primary_valid_rows": sum(e["coverage"]["valid_rows"] for e in primary["summaries"]),
                "example_source_bindings": [b["binding"] for b in examples],
                "example_source_episodes": sum(len(b["summaries"]) for b in examples),
                "raw_source_revalidation": "every declared chunk checked by task adapter, including non-selected example-source episodes",
                "processing_completeness": "all primary episodes/all enabled descriptor inputs; per-descriptor unknown remains explicit",
                "evaluation_uses": ["development_regression", "descriptive_audit"], "prospective_confirmation": False,
                "coverage": {"raw": "whole available primary archive", "media": "not_inspected", "model_inspection": "none",
                             "semantic": "requires independent necessary-fact and all-sentence audit"},
                "aggregation_config": {"default": "episode_equal", "pooled": "separate sample-cell weighted operands",
                                       "pool_zero_tolerance": POOLED_ZERO_TOLERANCE, "pool_units": "N*s for force; sample_cell_s for occupancy",
                                       "episode_force_share_eligibility": "inherited P1a sum_N threshold"},
                "artifacts": artifacts, "claim_count": len(card["claims"]), "selected_example_count": len(card["selected_examples"]),
                "cost": {"wall_seconds": time.perf_counter() - started, "peak_process_memory_bytes": peak_memory_bytes(),
                         "external_model_calls": 0, "api_cost": 0, "training_runs": 0},
                "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): file_sha256(p) for p in code},
                "author_projection": card["author_projection"], "review_status": card["review_status"]}
    _write_json(output / "manifest.json", manifest)
    return manifest
