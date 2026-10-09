"""CPU-only, lossless selected-environment export from completed development runs.

Run beside the source archives, then transfer only the export directory. This is
an extraction receipt, not a new production numeric archive or an alignment PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from true_rl_agent.runner.numeric_evidence import (  # noqa: E402
    iter_numeric_evidence, load_numeric_evidence_manifest,
)
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1  # noqa: E402


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def read_json(path: Path):
    payload = path.read_bytes()
    return json.loads(payload), payload


def copy_bytes(source: Path, destination: Path, expected: str | None = None) -> dict:
    payload = source.read_bytes()
    digest = sha(payload)
    if expected is not None and digest != expected:
        raise ValueError(f"source hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(payload)
    return {"source_path": str(source), "sha256": digest, "size_bytes": len(payload)}


def copy_png(source: Path, destination: Path, expected: str, *, storage: str = "copy") -> dict:
    if storage == "copy":
        return copy_bytes(source, destination, expected)
    if storage != "hardlink" or source.suffix.lower() != ".png" or destination.suffix.lower() != ".png":
        raise ValueError("hardlink storage is only supported for PNG files")
    payload = source.read_bytes()
    if sha(payload) != expected:
        raise ValueError(f"source hash mismatch: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Same-filesystem requirement and existing-destination errors propagate;
    # silently copying would defeat the caller's disk budget.
    os.link(source, destination)
    linked = destination.read_bytes()
    if len(linked) != len(payload) or sha(linked) != expected:
        raise ValueError(f"linked image hash mismatch: {destination}")
    return {"source_path": str(source), "sha256": expected, "size_bytes": len(linked),
            "storage": "hardlink", "shared_inode": True, "independent_backup": False}


def local_media(root: Path, name: str) -> Path:
    # Production manifests store filenames, not arbitrary external locators.
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise ValueError("media entry must be a filename")
    return root / name


def joint_mapping(*sources: tuple[str, dict]) -> dict:
    found = []

    def visit(value, path):
        if isinstance(value, dict):
            resolved = value.get("resolved_runtime")
            if isinstance(resolved, dict) and "joint_ids" in resolved:
                found.append({"source_pointer": path + "/resolved_runtime/joint_ids",
                              "joint_ids": resolved["joint_ids"]})
            for key, child in value.items():
                if isinstance(child, (dict, list)):
                    visit(child, path + "/" + key)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                if isinstance(child, (dict, list)):
                    visit(child, path + "/" + str(index))

    for label, source in sources:
        visit(source, label)
    if found and any(row["joint_ids"] != found[0]["joint_ids"] for row in found):
        raise ValueError("conflicting recorded Stair joint_ids")
    return {"status": "recorded" if found else "not_recorded_in_supplied_sources",
            "joint_ids": found[0]["joint_ids"] if found else None, "sources": found}


def extract(summary_path: Path, recordings_path: Path, output: Path, stdout_log: Path | None = None,
            resolved_cell_manifest: Path | None = None, *, png_storage: str = "copy") -> dict:
    if png_storage not in {"copy", "hardlink"}:
        raise ValueError("PNG storage must be copy or hardlink")
    summary_path, recordings_path = summary_path.resolve(), recordings_path.resolve()
    summary, _ = read_json(summary_path)
    recordings, _ = read_json(recordings_path)
    if summary.get("status") != "pass" or summary.get("evaluation_phase") != "public_dev":
        raise ValueError("a completed public_dev phase summary is required")
    task = {"unitree_stair_climbing": "stair", "unitree_push_recovery": "push"}[summary["task_id"]]
    if resolved_cell_manifest is not None:
        cell = ResolvedCellManifestV1.from_mapping(read_json(resolved_cell_manifest)[0])
        if cell.manifest_hash != summary["resolved_cell_manifest_hash"]:
            raise ValueError("resolved cell source differs from the original run")
    evidence = summary["task_evidence"]
    references = ([evidence["numeric_evidence"]] if task == "stair" else
                  [seed["numeric_evidence_ref"] for seed in evidence["seed_summaries"]])
    archives = []
    for reference in references:
        manifest = load_numeric_evidence_manifest(reference)
        identity = manifest["identity"]
        for key in ("job_id", "task_id", "evaluation_plan_hash", "evaluation_binding",
                    "resolved_cell_manifest_hash"):
            if identity[key] != summary[key]:
                raise ValueError("numeric source differs from summary: " + key)
        if task == "stair":
            if recordings["execution"]["checkpoint_hash"] != identity["checkpoint_hash"]:
                raise ValueError("Stair media checkpoint mismatch")
            if recordings["resolved_cell_manifest"]["manifest_hash"] != identity["resolved_cell_manifest_hash"]:
                raise ValueError("Stair media resolved manifest mismatch")
            for key in ("seed_namespace", "backend_execution_environment_verification"):
                if recordings["execution"][key] != identity[key]:
                    raise ValueError("Stair media execution differs: " + key)
        else:
            if (recordings["checkpoint"]["checkpoint_hash"] != identity["checkpoint_hash"] or
                    recordings["resolved_cell_manifest_hash"] != identity["resolved_cell_manifest_hash"]):
                raise ValueError("Push media checkpoint/resolved manifest mismatch")
        archives.append({"reference": reference, "manifest": manifest, "targets": []})

    targets, seen = [], set()
    for recording in recordings["recordings"]:
        if recording.get("capture_mode") != "selected_rgb_snapshots" or recording["split"] != "public_dev":
            raise ValueError("only native public_dev selected snapshots can be exported")
        env = recording["env_index"]
        candidates = []
        for archive in archives:
            manifest = archive["manifest"]
            if task == "stair":
                for seed in manifest["metadata"]["seeds"]:
                    if seed["runner_seed"] == recording["seed"] and seed["split"] == recording["split"]:
                        candidates.append((archive, seed["run_index"], seed))
            elif (manifest["identity"]["run_index"] == recording["run_index"] and
                  manifest["identity"]["split"] == recording["split"]):
                candidates.append((archive, recording["run_index"], None))
        if len(candidates) != 1:
            raise ValueError("recording does not identify one numeric run")
        archive, run, seed = candidates[0]
        metadata = archive["manifest"]["metadata"]
        outcome = seed["outcomes"][env] if task == "stair" else metadata["runtime_episode_records"][env]
        if outcome != recording["episode_outcome"]:
            raise ValueError("media outcome differs from the numeric run/environment")
        if (run, env) in seen:
            raise ValueError("duplicate run/environment recording")
        seen.add((run, env))
        snapshot_path = local_media(recordings_path.parent, recording["file"])
        snapshot, payload = read_json(snapshot_path)
        if sha(payload) != recording["sha256"] or snapshot != recording["manifest"]:
            raise ValueError("snapshot manifest differs from recordings receipt")
        if (snapshot["schema_version"] not in {"native_rgb_snapshots.v1", "native_rgb_snapshots.v2", "native_rgb_snapshots.v3", "native_rgb_snapshots.v4"} or
                snapshot["env_index"] != env or snapshot["seed"] != recording["seed"] or
                snapshot["episode_control_step_count"] != recording["frame_count"]):
            raise ValueError("snapshot identity/interval mismatch")
        target = {"run_index": run, "env_index": env, "runner_seed": recording["seed"],
                  "recording": recording, "snapshot": snapshot, "snapshot_path": snapshot_path,
                  "outcome": outcome, "seed_metadata": seed, "archive": archive,
                  "parts": {}, "static": {}, "source_slices": [], "row_count": 0}
        archive["targets"].append(target)
        targets.append(target)

    expected_runs = ({seed["run_index"] for seed in archives[0]["manifest"]["metadata"]["seeds"]}
                     if task == "stair" else {item["manifest"]["identity"]["run_index"] for item in archives})
    if not targets or {target["run_index"] for target in targets} != expected_runs:
        raise ValueError("recordings do not cover the numeric source's complete run set")

    # Stream every original archive once through the production hash/dtype reader.
    # Unselected environments/chunks are checked but never copied to the export.
    for archive in archives:
        manifest = archive["manifest"]
        for chunk_index, (chunk_meta, arrays) in enumerate(iter_numeric_evidence(archive["reference"])):
            chunk = manifest["chunks"][chunk_index]
            run = chunk_meta["run_index"] if task == "stair" else manifest["identity"]["run_index"]
            for target in archive["targets"]:
                if target["run_index"] != run:
                    continue
                env = target["env_index"]
                valid = arrays["valid"][:, env]
                rows = np.flatnonzero(valid)
                if task == "stair" and chunk_meta["static_present"]:
                    target["static"] = {name: arrays[name][env].copy()
                                        for name in ("terrain_env_origins_w", "terrain_types")}
                if not len(rows):
                    continue
                if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
                    raise ValueError("first-episode valid rows are not contiguous")
                names = (("valid", "base_pos_w", "base_quat_w", "joint_pos", "episode_length", "done_mask")
                         if task == "stair" else ("valid", "pose", "planar_motion", "control_step", "episode_index"))
                selected = {name: arrays[name][rows, env].copy() for name in names}
                if task == "stair":
                    selected["control_step"] = selected["episode_length"].copy()
                    expected_steps = chunk_meta["step_start"] + rows
                else:
                    expected_steps = chunk_meta["frame_start"] + rows + 1
                    episode = manifest["metadata"]["episode_indices"][env]
                    if not np.all(selected["episode_index"] == episode):
                        raise ValueError("valid rows crossed the recorded first episode")
                if not np.array_equal(selected["control_step"], expected_steps):
                    raise ValueError("numeric control step does not match the source chunk clock")
                start = target["row_count"]
                for name, values in selected.items():
                    target["parts"].setdefault(name, []).append(values)
                target["row_count"] += len(rows)
                target["source_slices"].append({"chunk_index": chunk_index, "file": chunk["file"],
                    "sha256": chunk["sha256"], "stored_bytes": chunk["stored_bytes"],
                    "chunk_metadata": chunk_meta, "source_rows_start": int(rows[0]),
                    "source_rows_stop_exclusive": int(rows[-1]) + 1,
                    "output_rows_start": start, "output_rows_stop_exclusive": target["row_count"],
                    "environment_axis_index": env})

    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": "p3new.selected_dual_view_export.v1", "task": task,
              "scope": "lossless selected first-episode arrays and original native snapshots",
              "state_alignment_status": "not_checked_by_extractor", "source_files": {}, "episodes": []}
    if png_storage == "hardlink":
        report["png_storage"] = {"mode": "hardlink", "shared_inode": True, "independent_backup": False,
            "source_and_export_must_remain_immutable": True,
            "scope": "PNG only; metadata copies and extracted NPZ files are independent"}
    report["source_files"]["source_summary.json"] = copy_bytes(summary_path, output / "source_summary.json")
    report["source_files"]["video/recordings.json"] = copy_bytes(recordings_path, output / "video/recordings.json")
    if resolved_cell_manifest is not None:
        report["source_files"]["resolved_cell_manifest.json"] = copy_bytes(
            resolved_cell_manifest, output / "resolved_cell_manifest.json")
    if stdout_log is not None:
        report["source_files"]["source_stdout.log"] = copy_bytes(stdout_log.resolve(), output / "source_stdout.log")
        report["stdout_scope"] = "original log retained for manual name/index inspection; no inferred joint_ids"
    for index, archive in enumerate(archives):
        name = f"numeric_sources/archive_{index:02d}.manifest.json"
        reference = archive["reference"]
        report["source_files"][name] = copy_bytes(Path(reference["manifest_path"]), output / name,
                                                  reference["manifest_sha256"])
        archive["exported_manifest"] = name
    mapping = joint_mapping(("source_summary", summary), ("recordings", recordings),
                            ("numeric_manifest", archives[0]["manifest"])) if task == "stair" else None
    for target in targets:
        arrays = {name: np.concatenate(parts) for name, parts in target["parts"].items()}
        if not arrays or not np.array_equal(arrays["control_step"], np.arange(1, target["row_count"] + 1)):
            raise ValueError("export does not contain a contiguous first episode starting at step 1")
        if target["row_count"] != target["outcome"]["episode_steps"]:
            raise ValueError("export length differs from the original episode outcome")
        if task == "stair":
            if not target["static"]:
                raise ValueError("Stair source terrain origin is missing")
            arrays.update(target["static"])
        name = f"run{target['run_index']:02d}_env{target['env_index']:04d}.npz"
        with (output / name).open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        recording, snapshot = target["recording"], target["snapshot"]
        manifest_name = "video/" + recording["file"]
        report["source_files"][manifest_name] = copy_bytes(target["snapshot_path"], output / manifest_name,
                                                           recording["sha256"])
        pairs, pairs_seen = [], set()
        for sample in snapshot["samples"]:
            step = sample["control_step"]
            if not 1 <= step <= min(recording["frame_count"], target["row_count"]):
                raise ValueError("actual snapshot step is outside the scored interval")
            source_slice = next(item for item in target["source_slices"]
                                if item["output_rows_start"] <= step - 1 < item["output_rows_stop_exclusive"])
            for view in sample["views"]:
                pair = (step, view["view_id"])
                if pair in pairs_seen:
                    raise ValueError("duplicate actual step/view sample")
                pairs_seen.add(pair)
                image_name = "video/" + view["file"]
                report["source_files"][image_name] = copy_png(
                    local_media(recordings_path.parent, view["file"]), output / image_name, view["sha256"],
                    storage=png_storage)
                pairs.append({"control_step": step, "view_id": view["view_id"], "file": image_name,
                    "sha256": view["sha256"], "extracted_row": step - 1,
                    "source_chunk_index": source_slice["chunk_index"],
                    "source_chunk_row": source_slice["source_rows_start"] + step - 1 - source_slice["output_rows_start"]})
        archive = target["archive"]
        metadata = archive["manifest"]["metadata"]
        pairing_metadata = ({"coordinates": metadata["coordinates"],
                             "runtime_projection": metadata["runtime_projection"],
                             "joint_mapping": mapping,
                             "control_step_source": "lossless copy of episode_length; checked against chunk.step_start+row",
                             "step_dt_s": target["seed_metadata"]["step_dt"]} if task == "stair" else
                            {key: metadata[key] for key in ("coordinates", "pose_fields", "planar_motion_fields", "time", "capture_point")})
        report["episodes"].append({"run_index": target["run_index"], "env_index": target["env_index"],
            "runner_seed": target["runner_seed"], "first_episode_steps": target["row_count"],
            "numeric_file": name, "numeric_sha256": sha((output / name).read_bytes()),
            "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str} for key, value in arrays.items()},
            "numeric_reference": archive["reference"], "exported_source_manifest": archive["exported_manifest"],
            "numeric_identity": archive["manifest"]["identity"], "pairing_metadata": pairing_metadata,
            "source_slices": target["source_slices"], "snapshot_manifest": manifest_name,
            "snapshot_pairs": pairs, "episode_outcome": target["outcome"]})
    (output / "extraction.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--summary", type=Path, required=True, help="Completed production public_dev phase summary")
    parser.add_argument("--recordings", type=Path, required=True, help="Same-run native video/recordings.json")
    parser.add_argument("--output", type=Path, required=True, help="Fresh small export directory")
    parser.add_argument("--png-storage", choices=("copy", "hardlink"), default="copy",
                        help="PNG storage only; hardlink requires the same filesystem and immutable originals, not a backup")
    parser.add_argument("--stdout-log", type=Path, help="Optional original log copied for manual joint-name/index inspection")
    parser.add_argument("--resolved-cell-manifest", type=Path,
                        help="Original run composition proof, required for asset-bound Push body geometry")
    args = parser.parse_args()
    report = extract(args.summary, args.recordings, args.output, args.stdout_log, args.resolved_cell_manifest,
                     png_storage=args.png_storage)
    print(json.dumps({"output": str(args.output.resolve()), "episodes": len(report["episodes"]),
                      "snapshots": sum(len(row["snapshot_pairs"]) for row in report["episodes"]),
                      "state_alignment_status": report["state_alignment_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
