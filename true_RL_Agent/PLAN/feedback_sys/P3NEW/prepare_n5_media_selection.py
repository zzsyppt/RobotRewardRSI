"""Offline N5 media selection only; no capture, observer request or release.

Freeze this script and the capture card before new outcomes are available.
The card's two preselected envs across three runs define the entire media pool.
This experiment chooses one raw success and one raw failure by fixed hash;
task-owned functions then choose at most six actual dual-view steps per episode.
The existing observer preparer must still verify full numeric/state alignment.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
if (PROJECT / "src").is_dir():
    sys.path.insert(0, str(PROJECT / "src"))

from true_rl_agent.evaluation.visual_observer import read_native_snapshot_media
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.visual_observation import select_stair_media_steps
from true_rl_agent.tasks.push_recovery.visual_observation import select_push_media_steps
from true_rl_agent.tasks.push_recovery.failure_classifier import recovery_outcome_class
from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed

RULE = "p3new-n5-observer-episode-v1"


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def inside(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root), "source path escapes export")
    return path


def choose_episodes(task, candidates):
    """Experiment-owned outcome/hash rule; no report, residual or visual ranking."""
    require(task in {"stair", "push"}, "unsupported task")
    pool = []
    for source in candidates:
        row = deepcopy(source)
        run, env = row["run_index"], row["env_index"]
        require(type(run) is int and run > 0 and type(env) is int and env >= 0, "invalid run/env")
        row["rank_sha256"] = hashlib.sha256(f"{RULE}|{task}|{run}|{env}".encode("utf-8")).hexdigest()
        outcome, group = row.get("outcome", {}), None
        if row["eligibility"] == "complete":
            if task == "stair":
                reason = outcome.get("primary_reason")
                if outcome.get("success") is True and reason == "success":
                    group = "success"
                elif outcome.get("success") is False and isinstance(reason, str) and reason not in {"", "success", "censored"}:
                    group = "failure"
            else:
                group = recovery_outcome_class(outcome)
        exclusion = (row["eligibility"] if row["eligibility"] != "complete" else
                     "no_push_control" if task == "push" and outcome.get("control_only") is True else
                     "unclassified_or_inconsistent_raw_outcome" if group is None else None)
        row.update(outcome_class=group, selected=False, exclusion_reason=exclusion,
                   selection_reason="ineligible_or_unclassified" if group is None else "higher_hash_within_outcome")
        pool.append(row)
    require(len({(r["run_index"], r["env_index"]) for r in pool}) == len(pool), "duplicate pool identity")
    selected, missing = [], []
    for group in ("success", "failure"):
        eligible = [row for row in pool if row["outcome_class"] == group]
        if not eligible:
            missing.append(group)
            continue
        row = min(eligible, key=lambda r: (r["rank_sha256"], r["run_index"], r["env_index"]))
        row.update(selected=True, selection_reason="minimum_frozen_hash_within_" + group)
        selected.append(row)
    return {"selected": selected, "pool": sorted(pool, key=lambda r: (r["run_index"], r["env_index"])),
            "missing_classes": missing}


def prepare(export, full_path, capture_card_path, output):
    export, full_path, capture_card_path, output = (Path(p).resolve() for p in
        (export, full_path, capture_card_path, output))
    require(not output.exists(), "output must be new; preserve prior selections")
    capture = read(capture_card_path)
    extraction = read(export / "extraction.json")
    require(extraction["schema_version"] == "p3new.selected_dual_view_export.v1", "unsupported export")
    task = extraction["task"]
    spec, profile = capture["tasks"][task], capture["capture_profile"]
    require(profile["public_dev_seed_runs"] == 3 and profile["envs_per_task_per_seed"] == 2
            and profile["views_per_control_step"] == 2 and profile["resolution"] == [1920, 1080]
            and profile["snapshot_interval_steps"] == 1, "capture profile is not the frozen six-episode dual-view design")
    envs = [row["selected_env"] for row in spec["media_env_selection"]]
    require(len(envs) == len(set(envs)) == 2, "capture card must preselect two distinct envs")
    expected = {(run, env) for run in range(1, 4) for env in envs}
    exported = {(row["run_index"], row["env_index"]): row for row in extraction["episodes"]}
    require(len(exported) == len(extraction["episodes"]) and set(exported) <= expected,
            "export contains duplicate or unregistered media episodes")
    # Verify all metadata and small NPZ exports. Only selected PNG bytes are read.
    for name, ref in extraction["source_files"].items():
        if not name.endswith(".png"):
            path = inside(export, name)
            require(path.stat().st_size == ref["size_bytes"] and digest(path) == ref["sha256"], "source hash differs: " + name)
    summary, recordings = read(export / "source_summary.json"), read(export / "video/recordings.json")
    require(summary["status"] == "pass" and summary["evaluation_phase"] == "public_dev", "completed public_dev required")
    require(summary["task_id"] == {"stair": "unitree_stair_climbing", "push": "unitree_push_recovery"}[task], "task identity differs")
    full_ref = summary["task_evidence"]["behavior_full"]
    require(digest(full_path) == full_ref["sha256"] and full_path.stat().st_size == full_ref["size_bytes"], "full source differs from phase summary")
    full = read(full_path)
    sources, candidates = {}, []
    for run, env in sorted(expected):
        key = (run, env)
        if key not in exported:
            candidates.append({"run_index": run, "env_index": env, "outcome": {}, "eligibility": "missing_capture"})
            continue
        ep = exported[key]
        reference = ep["numeric_reference"]
        refs = ([summary["task_evidence"]["numeric_evidence"]] if task == "stair" else
                [row["numeric_evidence_ref"] for row in summary["task_evidence"]["seed_summaries"]])
        require(reference in refs, "numeric archive absent from original summary")
        manifest = load_numeric_evidence_manifest({**reference, "manifest_path": str(inside(export, ep["exported_source_manifest"]))})
        identity, meta = manifest["identity"], manifest["metadata"]
        require(identity == ep["numeric_identity"], "export numeric identity differs")
        for field in ("job_id", "task_id", "evaluation_plan_hash", "evaluation_binding", "resolved_cell_manifest_hash"):
            require(identity[field] == summary[field], "numeric/summary binding differs: " + field)
        require(identity["checkpoint_hash"] == "sha256:" + spec["checkpoint"]["sha256"], "capture checkpoint differs")
        matches = [r for r in recordings["recordings"] if Path(r["file"]).name == Path(ep["snapshot_manifest"]).name]
        require(len(matches) == 1, "snapshot must bind one original recording")
        record = matches[0]
        require(record["split"] == "public_dev" and record["env_index"] == env and record["seed"] == ep["runner_seed"], "recording env/seed differs")
        if task == "stair":
            seed, = [r for r in meta["seeds"] if r["run_index"] == run]
            require(seed["runner_seed"] == record["seed"] and seed["split"] == "public_dev"
                    and identity["seed_namespace"] == spec["cohort"]["namespace"], "Stair run/cohort differs")
            raw = seed["outcomes"][env]
            require(recordings["resolved_cell_manifest"]["manifest_hash"] == identity["resolved_cell_manifest_hash"]
                    and all(recordings["execution"][field] == identity[field] for field in
                            ("checkpoint_hash", "seed_namespace", "backend_execution_environment_verification")),
                    "Stair media execution binding differs")
            require(full["domain"]["archive_hash"] == reference["manifest_sha256"]
                    and full["domain"]["checkpoint_hash"] == identity["checkpoint_hash"], "Stair full identity differs")
            item, = [r for r in full["episodes"] if r["run"] == run and r["slot"] == env]
            require(item["dt_s"] == seed["step_dt"] and item["primary_reason"] == raw["primary_reason"], "Stair full clock/outcome differs")
        else:
            require(record["run_index"] == run == identity["run_index"]
                    and record["seed"] == derive_runner_seed(base_seed=spec["cohort"]["eval_base_seed"],
                        split_name="public_dev", run_index=run), "Push run/cohort differs")
            raw, = [r for r in meta["runtime_episode_records"] if r["env_id"] == env]
            require(recordings["checkpoint"]["checkpoint_hash"] == identity["checkpoint_hash"]
                    and recordings["resolved_cell_manifest_hash"] == identity["resolved_cell_manifest_hash"]
                    and record["resolved_protocol_hash"] == identity["resolved_protocol_hash"], "Push media execution binding differs")
            index, = [i for i, r in enumerate(full["sources"]) if r["manifest_sha256"] == reference["manifest_sha256"]]
            require(full["binding"]["checkpoint_hash"] == identity["checkpoint_hash"], "Push full checkpoint differs")
            item, = [r for r in full["transcripts"] if r["source"] == index and r["env_id"] == env and r["episode_index"] == raw["episode_index"]]
        require(raw == record["episode_outcome"] == ep["episode_outcome"], "original first-episode outcomes differ")
        npz = inside(export, ep["numeric_file"])
        require(digest(npz) == ep["numeric_sha256"], "numeric export hash differs")
        with np.load(npz, allow_pickle=False) as archive:
            steps, valid = archive["control_step"].tolist(), archive["valid"]
            require(valid.shape == (raw["episode_steps"],) and valid.all(), "incomplete numeric valid mask")
            if task == "push":
                require(np.all(archive["episode_index"] == raw["episode_index"]), "Push numeric episode differs")
        total = raw["episode_steps"]
        snapshot_path = inside(export, ep["snapshot_manifest"])
        require(digest(snapshot_path) == record["sha256"], "snapshot hash differs")
        snapshot = read(snapshot_path)
        require(snapshot == record["manifest"] and snapshot["env_index"] == env and snapshot["seed"] == record["seed"], "snapshot identity differs")
        require(steps == list(range(1, total + 1)) == [s["control_step"] for s in snapshot["samples"]]
                and total == ep["first_episode_steps"] == record["frame_count"] == record["terminal_control_step"]
                and snapshot["episode_control_step_count"] == total and record["includes_terminal_frame"] is True,
                "complete actual first..T numeric/media clocks required")
        require(snapshot["schema_version"] in {"native_rgb_snapshots.v3", "native_rgb_snapshots.v4"} and snapshot["snapshot_interval_steps"] == 1
                and snapshot["sampling_mode"] == "interval_from_step_1" and snapshot["resolution"] == [1920, 1080]
                and snapshot["snapshot_count"] == 2 * total, "unexpected native profile")
        views = sorted(v["view_id"] for v in snapshot["samples"][0]["views"])
        require(len(views) == len(set(views)) == 2 and all(sorted(v["view_id"] for v in s["views"]) == views
                for s in snapshot["samples"]), "actual dual views differ")
        require((item["steps"] if task == "stair" else item["final_control_step"]) == total, "full episode endpoint differs")
        sources[key] = (ep, item, raw, meta, steps, snapshot_path, record["sha256"])
        candidates.append({"run_index": run, "env_index": env, "runner_seed": record["seed"],
                           "outcome": raw, "eligibility": "complete"})
    chosen = choose_episodes(task, candidates)
    episodes, details = [], []
    for row in chosen["selected"]:
        ep, item, raw, meta, available, snapshot_path, snapshot_sha = sources[(row["run_index"], row["env_index"])]
        selection = (select_stair_media_steps(item, available_steps=available) if task == "stair" else
            select_push_media_steps(item, runtime_episode_record=raw, available_steps=available,
                                    step_dt_s=meta["time"]["step_dt_s"]))
        steps = selection["selected_steps"]
        _, images = read_native_snapshot_media(snapshot_path, manifest_sha256=snapshot_sha, selected_steps=steps)
        require(len(steps) <= 6 and len(images) == 2 * len(steps) <= 12, "per-episode image bound exceeded")
        alias = "e1" if row["outcome_class"] == "success" else "e2"
        episodes.append({"alias": alias, "run_index": row["run_index"], "env_index": row["env_index"], "selected_steps": steps})
        details.append({"alias": alias, "runner_seed": row["runner_seed"], "numeric_identity": ep["numeric_identity"],
            "numeric_reference": ep["numeric_reference"], "numeric_npz_sha256": ep["numeric_sha256"],
            "snapshot_manifest_sha256": snapshot_sha, "selection": selection,
            "image_count": len(images), "image_bytes": sum(len(i.data) for i in images),
            "image_pixels": sum(i.width * i.height for i in images),
            "images": [{"step": i.displayed_steps[0], "view_id": i.view_id, "sha256": hashlib.sha256(i.data).hexdigest()} for i in images]})
    image_count = sum(d["image_count"] for d in details)
    require(image_count <= 24, "report image bound exceeded")
    complete_pool = all(r["eligibility"] == "complete" for r in chosen["pool"])
    value = {"schema_version": "p3new.native_observer_selection.v2", "scope": "n5_confirmation",
        "purpose": "Frozen outcome/hash media subset with task-owned process and rule-independent step coverage",
        "source": {"extraction_sha256": digest(export / "extraction.json"),
            "description": "One source-bound complete six-episode capture pool; no report-quality selection",
            "capture_card_sha256": digest(capture_card_path), "full_sha256": full_ref["sha256"]},
        "episodes": episodes}
    selection_bytes = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    card = {"status": "SELECTION_PREPARED_NOT_RELEASED", "observer_profile": "READY_FOR_PREPARATION" if complete_pool and not chosen["missing_classes"] else "INCONCLUSIVE",
        "task": task, "episode_rule": RULE, "hash_input": RULE + "|{task}|{run_index}|{env_index}",
        "pool": chosen["pool"], "missing_classes": chosen["missing_classes"], "pool_complete": complete_pool,
        "selected_details": details, "report_episode_count": len(episodes), "report_image_count": image_count,
        "remaining_report_image_capacity": 24 - image_count,
        "full_numeric_snapshot_state_alignment": "NOT_CHECKED_HERE_REQUIRED_BY_OBSERVER_PREPARE",
        "serialized_request_body_limit": "NOT_CHECKED_HERE_REQUIRED_BY_OBSERVER_PREPARE",
        "provider_requests": 0, "gpu_jobs": 0, "release": False,
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(),
        "source_files": {"capture_card": {"path": str(capture_card_path), "sha256": digest(capture_card_path)},
            "full": {"path": str(full_path), "sha256": full_ref["sha256"]}, "export": str(export)},
        "driver_sha256": digest(Path(__file__)),
        "task_selector_sha256": {fn.__module__: digest(sys.modules[fn.__module__].__file__)
            for fn in (select_stair_media_steps, select_push_media_steps)}}
    output.mkdir(parents=True, exist_ok=False)
    (output / "selection.json").write_bytes(selection_bytes)
    (output / "card.json").write_text(json.dumps(card, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--capture-card", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    card = prepare(args.export, args.full, args.capture_card, args.output)
    print(json.dumps({k: card[k] for k in ("status", "observer_profile", "task", "report_episode_count", "report_image_count")}))


if __name__ == "__main__":
    main()
