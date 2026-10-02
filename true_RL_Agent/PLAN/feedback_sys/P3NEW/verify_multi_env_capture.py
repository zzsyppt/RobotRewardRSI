"""CPU-only r3/r4 development capture audit. Never writes into either capture.

Compare every valid first-episode numeric field and original outcome, then bind
the six sparse r4 media episodes to their own numeric rows. This is engineering
evidence, not N5, physical ground truth, or an independent implementation of the
production task alignment formulas. Run only after the capture has completed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np

STEPS = [1, 20, 40, 80, 160, 320]
ENVS = {"stair": [282, 0], "push": [19, 55]}
STATIC = {"terrain_env_origins_w", "terrain_types"}
IDENTITY = ("task_id", "subject_kind", "subject_id", "subject_manifest_hash",
            "program_source_checksum", "training_binding_hash", "evaluation_binding",
            "checkpoint_hash", "resolved_cell_manifest_hash")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, evidence):
    path = Path(path).resolve()
    evidence[str(path)] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def source_inventory(root):
    return {str(p.relative_to(root)).replace("\\", "/"): sha(p)
            for folder in ("src", "scripts", "configs") for p in sorted((root / folder).rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}


def read_runs(root, task, evidence):
    summary = read(root / "evaluation/public_dev/summary.json", evidence)
    require(summary.get("status") == "pass" and not summary.get("errors"), "capture summary not complete/pass")
    sidecar = read(root / "video/recordings.json", evidence)
    paths = [root / "numeric_evidence/manifest.json"] if task == "stair" else sorted(root.glob("numeric/public_dev-*/manifest.json"))
    runs = {}
    for path in paths:
        manifest = read(path, evidence)
        require(manifest.get("complete") is True, "numeric manifest incomplete")
        meta = manifest["metadata"]
        seeds = meta["seeds"] if task == "stair" else [{"run_index": manifest["identity"]["run_index"], "outcomes": meta["runtime_episode_records"]}]
        for seed in seeds:
            run = seed["run_index"]
            require(run not in runs, "duplicate numeric run")
            records = [r for r in sidecar["recordings"] if (r["seed"] == seed["runner_seed"] if task == "stair" else r["run_index"] == run)]
            require(bool(records) and len({r["seed"] for r in records}) == 1, "media seed/run binding missing or ambiguous")
            runs[run] = {"path": path, "manifest": manifest, "outcomes": seed["outcomes"],
                         "seed": records[0]["seed"], "records": records, "seed_metadata": seed}
            if task == "push":
                s = next(x for x in summary["task_evidence"]["seed_summaries"] if x["run"]["run_index"] == run)
                require(s["status"] == "pass" and s["numeric_evidence_ref"]["manifest_sha256"] == evidence[str(path.resolve())], "Push phase/numeric binding mismatch")
                require(s["per_episode"] == seed["outcomes"], "Push phase/raw outcomes mismatch")
    require(sorted(runs) == [1, 2, 3], "expected exactly three completed numeric runs")
    return runs, sidecar


def load_arrays(item, run, task, evidence):
    """Read all actual columns, not just columns known to a report reducer."""
    parts, static, schema, chunk_rows = defaultdict(list), {}, None, []
    chunks = [c for c in item["manifest"]["chunks"] if task == "push" or c["metadata"]["run_index"] == run]
    for chunk in chunks:
        path = item["path"].parent / chunk["file"]
        digest = sha(path)
        evidence[str(path.resolve())] = digest
        require(digest == chunk["sha256"], f"NPZ hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            require(set(archive.files) == set(chunk["arrays"]), "NPZ field inventory mismatch")
            dynamic = set(archive.files) - STATIC
            require(schema is None or schema == dynamic, "numeric fields differ across chunks")
            schema = dynamic
            valid = archive["valid"]
            require(valid.dtype == np.bool_ and valid.ndim == 2, "invalid validity mask")
            for name in archive.files:
                a = archive[name]
                require(chunk["arrays"][name] == {"shape": list(a.shape), "dtype": a.dtype.str}, "NPZ field schema mismatch")
                if name in STATIC:
                    require(name not in static or np.array_equal(static[name], a), "static field changed within run")
                    static[name] = a.copy()
                else:
                    require(a.shape[:2] == valid.shape, f"not a step/env column: {name}")
                    require(not parts[name] or (parts[name][0].dtype == a.dtype and parts[name][0].shape[1:] == a.shape[1:]), "column axes/dtype changed")
                    parts[name].append(a.copy())
            chunk_rows.append({"file": str(path), "sha256": digest, "rows": len(valid)})
    require(bool(chunks), "no numeric chunks")
    arrays = {name: np.concatenate(values) for name, values in parts.items()}
    require(arrays["valid"].shape[1] == len(item["outcomes"]), "numeric/outcome env count mismatch")
    clock = "episode_length" if task == "stair" else "control_step"
    for env, terminal in enumerate(item["outcomes"]):
        valid = arrays["valid"][:, env]
        t = terminal["episode_steps"]
        require(np.array_equal(np.flatnonzero(valid), np.arange(t)), f"valid first-episode rows are not 1..T: run={run}, env={env}")
        require(np.array_equal(arrays[clock][valid, env], np.arange(1, t + 1)), "valid clock does not end at own T")
        if task == "push":
            require(terminal["env_id"] == env and np.all(arrays["episode_index"][valid, env] == terminal["episode_index"]), "Push episode identity mismatch")
        for name, a in arrays.items():
            require(a.dtype.kind != "f" or np.isfinite(a[valid, env]).all(), f"nonfinite valid field: {name}")
    return arrays, static, chunk_rows


def compare_array(a, b):
    equal = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    result = {"identical": equal, "baseline_shape": list(a.shape), "candidate_shape": list(b.shape),
              "baseline_dtype": a.dtype.str, "candidate_dtype": b.dtype.str}
    if not equal and a.shape == b.shape and a.size:
        indices = np.argwhere(a != b)
        result["different_values"] = len(indices)
        result["first_different_index"] = indices[0].tolist() if len(indices) else None
        if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
            result["max_abs_difference"] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
    return result


def compare_run(left, right, la, ra, ls, rs):
    fields = []
    for name in sorted(set(la) | set(ra)):
        row = {"field": name, "identical": name in la and name in ra, "mismatched_envs": []}
        hashes = [hashlib.sha256(), hashlib.sha256()]
        count = [0, 0]
        for env in range(max(len(left["outcomes"]), len(right["outcomes"]))):
            values = []
            for i, arrays in enumerate((la, ra)):
                a = arrays[name][arrays["valid"][:, env], env] if name in arrays and env < arrays["valid"].shape[1] else None
                values.append(a)
                if a is not None:
                    hashes[i].update(canonical([env, a.dtype.str, list(a.shape)]))
                    hashes[i].update(a.tobytes())
                    count[i] += a.size
            check = compare_array(*values) if all(v is not None for v in values) else {"identical": False, "missing_field_or_env": True}
            if not check["identical"]:
                row["identical"] = False
                row["mismatched_envs"].append({"env": env, **check})
        row.update(valid_value_counts=count, baseline_valid_sha256=hashes[0].hexdigest(), candidate_valid_sha256=hashes[1].hexdigest())
        fields.append(row)
    static = {name: compare_array(ls[name], rs[name]) if name in ls and name in rs else {"identical": False, "missing": True} for name in sorted(set(ls) | set(rs))}
    terminal_differences = []
    for env in range(max(len(left["outcomes"]), len(right["outcomes"]))):
        a = left["outcomes"][env] if env < len(left["outcomes"]) else {}
        b = right["outcomes"][env] if env < len(right["outcomes"]) else {}
        if a != b:
            terminal_differences.append({"env": env, "changed_top_level_fields": sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)),
                                         "baseline_T": a.get("episode_steps"), "candidate_T": b.get("episode_steps")})
    return {"identical": all(r["identical"] for r in fields) and all(r["identical"] for r in static.values()) and not terminal_differences,
            "episode_counts": [len(left["outcomes"]), len(right["outcomes"])], "fields": fields, "static_fields": static,
            "terminal_identical": not terminal_differences, "terminal_differences": terminal_differences,
            "baseline_terminal_sha256": hashlib.sha256(canonical(left["outcomes"])).hexdigest(),
            "candidate_terminal_sha256": hashlib.sha256(canonical(right["outcomes"])).hexdigest()}


def check_media(root, task, item, arrays, align, evidence):
    from PIL import Image

    records = item["records"]
    require(sorted(r["env_index"] for r in records) == sorted(ENVS[task]), "expected two distinct registered envs per run")
    rows, files = [], []
    for record in records:
        env = record["env_index"]
        terminal = item["outcomes"][env]
        t = terminal["episode_steps"]
        path = root / "video" / record["file"]
        media = read(path, evidence)
        require(evidence[str(path.resolve())] == record["sha256"] and media == record["manifest"], "snapshot manifest/sidecar mismatch")
        require(record["episode_outcome"] == terminal and record["terminal_control_step"] == t == record["frame_count"] == media["episode_control_step_count"], "media bound to wrong terminal")
        require(media["env_index"] == env and media["seed"] == item["seed"], "snapshot manifest identity mismatch")
        require(media["selected_control_steps"] == STEPS and media["snapshot_interval_steps"] is None and media["sampling_mode"] == "fixed_steps", "not the registered sparse r4 capture")
        require(media["resolution"] == [1920, 1080], "wrong manifest resolution")
        require(media.get("other_envs_visually_hidden") is True, "visibility declaration missing")
        require(record.get("episode_evidence_source") == "pre_reset_task_state", "media timing declaration missing")
        samples = media["samples"]
        actual = [s["control_step"] for s in samples]
        require(actual == [s for s in STEPS if s <= t], "actual sample steps do not equal registered steps clipped by own T")
        require(record["includes_terminal_frame"] is (t in actual), "inaccurate terminal frame flag")
        require(media["captured_step_count"] == len(samples) and media["snapshot_count"] == record["snapshot_count"] == len(samples) * 2, "snapshot counts mismatch")
        image_rows = []
        for sample in samples:
            state = sample["state_before"]
            require(state == sample["state_after"], "render advanced or changed recorded selected state/clock")
            require([v["view_id"] for v in sample["views"]] == ["v0", "v1"], "missing/duplicate/reordered views")
            require(sample["views"][0]["camera_eye_w"] != sample["views"][1]["camera_eye_w"], "two views have the same camera eye")
            for view in sample["views"]:
                for key in ("eye", "lookat"):
                    require(np.allclose(view[f"camera_{key}_w"], np.asarray(state["root_pos_w"]) + view[f"camera_{key}_offset"], rtol=0, atol=1e-7), "camera tracking metadata does not follow this env root")
                image_path = root / "video" / view["file"]
                digest = sha(image_path)
                require(digest == view["sha256"], "PNG hash mismatch")
                with Image.open(image_path) as image:
                    require(image.format == "PNG" and image.size == (1920, 1080) and image.mode == "RGB", "PNG format/resolution/mode mismatch")
                    image.verify()
                files.append(str(image_path.resolve()))
                image_rows.append({"step": sample["control_step"], "view_id": view["view_id"], "file": str(image_path), "sha256": digest, "bytes": image_path.stat().st_size})
        valid = arrays["valid"][:, env]
        if task == "stair":
            projection = item["manifest"]["metadata"]["runtime_projection"]
            result = align(media, control_step=arrays["episode_length"][valid, env],
                           episode_length=arrays["episode_length"][valid, env], base_pos_w=arrays["base_pos_w"][valid, env],
                           base_quat_w=arrays["base_quat_w"][valid, env], joint_pos=arrays["joint_pos"][valid, env],
                           numeric_joint_names=projection["joint_names"], kinematic_base_source=projection["kinematic_base_source"])
        else:
            result = align(media, control_step=arrays["control_step"][valid, env], pose=arrays["pose"][valid, env], planar_motion=arrays["planar_motion"][valid, env])
        rows.append({"env": env, "seed": item["seed"], "T": t, "actual_steps": actual,
                     "includes_terminal_frame": record["includes_terminal_frame"], "snapshot_manifest": str(path),
                     "snapshot_manifest_sha256": record["sha256"], "images": image_rows, "numeric_overlap": result})
    return rows, files


def verify(args, result):
    evidence = result["input_sha256"]
    card = read(args.capture_card, evidence)
    before_sources = [source_inventory(p) for p in (args.baseline_source, args.source)]
    changed = {p: {"previous": before_sources[0].get(p), "current": before_sources[1].get(p)}
               for p in sorted(set(before_sources[0]) | set(before_sources[1])) if before_sources[0].get(p) != before_sources[1].get(p)}
    result["source_comparison"] = {"file_counts": [len(s) for s in before_sources], "changed": changed,
                                    "matches_frozen_capture_card": changed == card["changed_production_from_r3"]}
    require(changed == card["changed_production_from_r3"], "frozen source difference is not the registered four files")
    sys.path.insert(0, str(args.source / "src"))
    task_module = f"true_rl_agent.tasks.{'stair_climbing' if args.task == 'stair' else 'push_recovery'}.visual_observation"
    module = importlib.import_module(task_module)
    require(Path(module.__file__).resolve().is_relative_to(args.source.resolve()), "alignment helper imported from wrong source")
    result["alignment_helper"] = {"file": module.__file__, "sha256": sha(module.__file__), "independent_oracle": False}
    align = getattr(module, f"align_{args.task}_snapshot_samples")
    left, _ = read_runs(args.baseline, args.task, evidence)
    right, sidecar = read_runs(args.candidate, args.task, evidence)
    require(len(sidecar["recordings"]) == 6, "expected exactly six media run/env records")
    files = []
    result["runs"] = []
    for run in (1, 2, 3):
        a, b = left[run], right[run]
        keys = IDENTITY + (("runtime_dynamics_hash", "resolved_protocol_hash", "task_numeric_schema", "split") if args.task == "push" else ("capability_profile_hash", "seed_namespace", "splits"))
        bindings = {k: a["manifest"]["identity"].get(k) == b["manifest"]["identity"].get(k) and k in a["manifest"]["identity"] for k in keys}
        require(all(bindings.values()) and a["seed"] == b["seed"], "policy/protocol/cohort/seed binding changed")
        la, ls, lc = load_arrays(a, run, args.task, evidence)
        ra, rs, rc = load_arrays(b, run, args.task, evidence)
        comparison = compare_run(a, b, la, ra, ls, rs)
        media, images = check_media(args.candidate, args.task, b, ra, align, evidence)
        files.extend(images)
        result["runs"].append({"run_index": run, "seed": a["seed"], "identity_equal": bindings,
                               "excluded_from_trajectory_equality": {k: [a["manifest"]["identity"].get(k), b["manifest"]["identity"].get(k)] for k in ("job_id", "evaluation_plan_hash")},
                               "baseline_chunks": lc, "candidate_chunks": rc, "numeric_comparison": comparison, "media": media})
    require(len(files) == len(set(files)), "media filenames collide across run/env/view")
    actual_files = {str(p.resolve()) for p in (args.candidate / "video").glob("*.png")}
    require(set(files) == actual_files, "unreferenced or missing PNG after per-env terminal pruning")
    result["image_count"] = len(files)
    result["source_unchanged_during_audit"] = before_sources == [source_inventory(p) for p in (args.baseline_source, args.source)]
    result["input_unchanged_during_audit"] = all(sha(p) == digest for p, digest in evidence.items())
    require(result["source_unchanged_during_audit"] and result["input_unchanged_during_audit"], "inputs/source changed during audit")
    result["all_valid_numeric_fields_and_terminals_identical"] = all(r["numeric_comparison"]["identical"] for r in result["runs"])
    result["all_media_numeric_overlaps_aligned"] = all(m["numeric_overlap"]["status"] == "overlap_aligned" for r in result["runs"] for m in r["media"])
    result["status"] = "pass" if result["all_valid_numeric_fields_and_terminals_identical"] and result["all_media_numeric_overlaps_aligned"] else "difference_observed"


def self_test():
    import unittest

    class ExactComparisonTests(unittest.TestCase):
        def test_dtype_and_single_bit_difference(self):
            a = np.array([1.0], dtype=np.float32)
            self.assertFalse(compare_array(a, a.astype(np.float64))["identical"])
            self.assertFalse(compare_array(a, np.nextafter(a, np.float32(2)))["identical"])

        def test_padding_is_excluded_but_terminal_is_compared(self):
            a = {"valid": np.array([[True], [True], [False]]), "x": np.array([[1.], [2.], [0.]], dtype=np.float32)}
            b = {k: v.copy() for k, v in a.items()}
            b["x"][2, 0] = 99
            item = {"outcomes": [{"episode_steps": 2, "success": True}]}
            self.assertTrue(compare_run(item, item, a, b, {}, {})["identical"])
            b["x"][1, 0] = 3
            self.assertFalse(compare_run(item, item, a, b, {}, {})["identical"])
            b["x"][1, 0] = 2
            other = {"outcomes": [{"episode_steps": 2, "success": False}]}
            self.assertFalse(compare_run(item, other, a, b, {}, {})["identical"])

        def test_missing_field_is_not_pass(self):
            a = {"valid": np.array([[True]]), "x": np.array([[1]])}
            item = {"outcomes": [{"episode_steps": 1}]}
            self.assertFalse(compare_run(item, item, a, {"valid": a["valid"]}, {}, {})["identical"])

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExactComparisonTests))
    return 0 if result.wasSuccessful() else 1


def main():
    if "--self-test" in sys.argv:
        return self_test()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("stair", "push"), required=True)
    for name in ("baseline", "candidate", "baseline-source", "source", "capture-card", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "refusing to overwrite an earlier audit receipt")
    for root in (args.baseline, args.candidate, args.baseline_source, args.source):
        require(not args.output.resolve().is_relative_to(root.resolve()), "output must be outside original capture/source trees")
    result = {"schema_version": "p3new.multi_env_capture_audit.v1", "task": args.task,
              "started_at_utc": datetime.now(timezone.utc).isoformat(), "script_sha256": sha(__file__),
              "baseline": str(args.baseline), "candidate": str(args.candidate), "input_sha256": {},
              "scope": "old_development_r3_vs_r4_all_valid_first_episodes_and_six_r4_media_episodes",
              "limits": ["No N5, API, GPU, or new collection; originals are read-only.",
                         "Task overlap checker is reused from frozen production; it is not an independent physical oracle.",
                         "PNG checks cover file integrity/size and recorded camera/state metadata; human pixel/full-body review is separate.",
                         "Push numeric manifest omits raw seed IDs: binding uses run_index, cohort hash, and same-run media seed.",
                         "A trajectory difference prevents a noninterference conclusion; alone it does not identify its cause.",
                         "Invalid post-first-episode padding is excluded from numeric equality."]}
    code = 0
    try:
        verify(args, result)
        code = 0 if result["status"] == "pass" else 1
    except Exception as exc:
        result.update(status="incomplete_or_failed", error_type=type(exc).__name__, error=str(exc))
        code = 2
    result["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    result["script_unchanged_during_audit"] = sha(__file__) == result["script_sha256"]
    if not result["script_unchanged_during_audit"]:
        result["status"], code = "incomplete_or_failed", 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha(args.output), "images": result.get("image_count"), "error": result.get("error")}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
