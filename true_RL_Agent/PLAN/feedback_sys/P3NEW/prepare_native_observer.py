"""Prepare native observer requests offline; never send them.

Selection JSON: {"schema_version":"p3new.native_observer_selection.v1",
"scope":"old_development_not_n5", "purpose":"specific development question",
"source":{"extraction_sha256":"...", "description":"original development source"},
"episodes":[{"run_index":1,"env_index":282,"selected_steps":[1,20,40]}]}.
The complete interval-1, terminal-inclusive capture requirement belongs to this
development profile, not to the shared media reader or observation contract.

N5 v2 selection preserves e1/e2 and limits each to six actual dual-view steps.
Use --capture-card CARD --selection SELECTION_DIR to prepare both packets and
the cumulative report card. This creates no release and makes no API call.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

import true_rl_agent  # noqa: E402

from true_rl_agent.evaluation.visual_observer import (  # noqa: E402
    IDENTITY_FIELDS, LAYOUT_VERSION, MediaWindow, ObserverBudget,
    prepare_observer_request, read_native_snapshot_media,
)
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest  # noqa: E402
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1  # noqa: E402
from true_rl_agent.tasks.stair_climbing.visual_observation import (  # noqa: E402
    STAIR_MEASUREMENT_CLAIMS, bind_stair_root_frame,
    align_stair_snapshot_samples, stair_numeric_check, stair_s0_for_window, stair_s0_definitions,
    stair_pose_facts, stair_body_image_points,
)
from true_rl_agent.tasks.push_recovery.visual_observation import (  # noqa: E402
    PUSH_S0_DEFINITIONS, align_push_snapshot_samples,
    push_measurement_claims, push_numeric_check, push_s0_for_window,
    push_pose_facts,
)
from true_rl_agent.tasks.push_recovery.body_geometry import (  # noqa: E402
    bind_push_body_anchors, push_body_geometry_definitions, push_body_geometry_s0,
    push_body_geometry_facts, push_body_image_points,
)

SOURCE_ROOT = Path(true_rl_agent.__file__).resolve().parents[2]

MODEL = "deepseek-flash"
FINGERPRINT = "aeb56401ca74e127821c4f9126dcb669"
RESERVATION_USD = 0.05
MAX_TOKENS = 32768
GENERATION_OPTIONS = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
# Provider transport limit, separate from the shared observer's image budget.
# https://api-docs.deepseek.com/guides/vision/ (checked 2026-09-29)
REQUEST_BODY_LIMIT_BYTES = 48 * 1024 * 1024


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_bytes())


def inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("export locator leaves the supplied export directory")
    return path


def require(condition, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def selection_episode(selection: dict, episode_alias: str | None) -> tuple[dict, str, int]:
    """Preserve the old one-episode contract; N5 addresses its frozen aliases."""
    choices = selection["episodes"]
    old = (selection.get("schema_version"), selection.get("scope")) == (
        "p3new.native_observer_selection.v1", "old_development_not_n5")
    if old:
        require(episode_alias is None and isinstance(choices, list) and len(choices) == 1,
                "old-development selects exactly one episode without an alias override")
        return choices[0], "e1", 12
    require((selection.get("schema_version"), selection.get("scope")) == (
        "p3new.native_observer_selection.v2", "n5_confirmation"), "unsupported selection scope/version")
    require(isinstance(choices, list) and len(choices) == 2
            and [row.get("alias") for row in choices] == ["e1", "e2"],
            "N5 requires both frozen episode aliases in e1/e2 order")
    require(episode_alias in {"e1", "e2"}, "N5 requires an explicit episode_alias")
    require(len({(row["run_index"], row["env_index"]) for row in choices}) == 2,
            "N5 aliases must identify two distinct episodes")
    return next(row for row in choices if row["alias"] == episode_alias), episode_alias, 6


def report_budget() -> ObserverBudget:
    return ObserverBudget(max_pixels=50_000_000, max_image_bytes=100_000_000,
                          max_cost_usd=0.10, max_calls=2)


def build(export: Path, selection_path: Path, *, episode_alias: str | None = None) -> dict:
    """Rebuild exact request bytes and same-run checks without writing or sending."""
    export, selection_path = export.resolve(), selection_path.resolve()
    selection_bytes = selection_path.read_bytes()
    selection = json.loads(selection_bytes)
    choice, alias, max_steps = selection_episode(selection, episode_alias)
    require(isinstance(selection.get("purpose"), str) and selection["purpose"].strip(), "selection purpose is required")
    source = selection["source"]
    require(isinstance(source.get("description"), str) and source["description"].strip(), "selection source description is required")
    extraction_path = export / "extraction.json"
    extraction_sha = digest(extraction_path)
    require(source["extraction_sha256"] == extraction_sha, "selection belongs to another export receipt")
    require(type(choice["run_index"]) is int and choice["run_index"] > 0
            and type(choice["env_index"]) is int and choice["env_index"] >= 0, "invalid run/env selection")
    steps = choice["selected_steps"]
    require(isinstance(steps, list) and 1 <= len(steps) <= max_steps
            and all(type(step) is int and step > 0 for step in steps)
            and steps == sorted(set(steps)), "selected steps exceed the profile or are not distinct/increasing")
    extracted = read_json(extraction_path)
    require(extracted["schema_version"] == "p3new.selected_dual_view_export.v1", "unsupported export receipt")
    task = extracted["task"]
    require(task in {"stair", "push"}, "unsupported development task")
    require({"source_summary.json", "video/recordings.json"} <= set(extracted["source_files"]),
            "export is missing original summary/recordings hash receipts")
    # Verify only local exported bytes, never reopen the large remote NPZ chunks.
    for name, receipt in extracted["source_files"].items():
        path = inside(export, name)
        require(path.stat().st_size == receipt["size_bytes"] and digest(path) == receipt["sha256"],
                "exported original source differs: " + name)
    for row in extracted["episodes"]:
        require(digest(inside(export, row["numeric_file"])) == row["numeric_sha256"], "extracted NPZ hash differs")
    candidates = [row for row in extracted["episodes"]
                  if (row["run_index"], row["env_index"]) == (choice["run_index"], choice["env_index"])]
    require(len(candidates) == 1, "selection does not identify one exported episode")
    episode = candidates[0]
    summary = read_json(export / "source_summary.json")
    recordings = read_json(export / "video/recordings.json")
    require(summary["status"] == "pass" and summary["evaluation_phase"] == "public_dev", "completed public_dev evidence is required")
    require(summary["task_id"] == {"stair": "unitree_stair_climbing", "push": "unitree_push_recovery"}[task], "task differs from export")
    reference = episode["numeric_reference"]
    references = ([summary["task_evidence"]["numeric_evidence"]] if task == "stair" else
                  [row["numeric_evidence_ref"] for row in summary["task_evidence"]["seed_summaries"]])
    require(reference in references, "numeric reference is absent from original phase summary")
    manifest = load_numeric_evidence_manifest({**reference,
        "manifest_path": str(inside(export, episode["exported_source_manifest"]))})
    numeric_id, metadata = manifest["identity"], manifest["metadata"]
    require(numeric_id == episode["numeric_identity"], "export numeric identity differs from sealed manifest")
    for key in ("job_id", "task_id", "evaluation_plan_hash", "evaluation_binding", "resolved_cell_manifest_hash"):
        require(numeric_id[key] == summary[key], "summary/numeric identity mismatch: " + key)
    recording_matches = [row for row in recordings["recordings"]
                         if Path(row["file"]).name == Path(episode["snapshot_manifest"]).name]
    require(len(recording_matches) == 1, "snapshot does not identify one recording")
    recording = recording_matches[0]
    env, run = episode["env_index"], episode["run_index"]
    require(recording["split"] == "public_dev" and recording["env_index"] == env
            and recording["seed"] == episode["runner_seed"], "recording run/environment/seed mismatch")
    if task == "stair":
        seeds = [row for row in metadata["seeds"] if row["run_index"] == run]
        require(len(seeds) == 1 and seeds[0]["runner_seed"] == recording["seed"]
                and seeds[0]["split"] == recording["split"], "Stair numeric seed binding differs")
        outcome = seeds[0]["outcomes"][env]
        require(recordings["resolved_cell_manifest"]["manifest_hash"] == numeric_id["resolved_cell_manifest_hash"], "Stair media manifest differs")
        for key in ("checkpoint_hash", "seed_namespace", "backend_execution_environment_verification"):
            require(recordings["execution"][key] == numeric_id[key], "Stair media execution differs: " + key)
    else:
        require(recording["run_index"] == run == numeric_id["run_index"]
                and numeric_id["split"] == "public_dev", "Push run identity differs")
        require(recordings["checkpoint"]["checkpoint_hash"] == numeric_id["checkpoint_hash"]
                and recordings["resolved_cell_manifest_hash"] == numeric_id["resolved_cell_manifest_hash"]
                and recording["resolved_protocol_hash"] == numeric_id["resolved_protocol_hash"], "Push media identity differs")
        outcome = metadata["runtime_episode_records"][env]
    require(outcome == recording["episode_outcome"] == episode["episode_outcome"], "first-episode metadata differs")
    with np.load(inside(export, episode["numeric_file"]), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    require(set(arrays) == set(episode["arrays"]), "extracted array fields differ")
    for name, array in arrays.items():
        require(episode["arrays"][name] == {"shape": list(array.shape), "dtype": array.dtype.str}, "extracted shape/dtype differs: " + name)
    total = outcome["episode_steps"]
    require(total == episode["first_episode_steps"] and np.array_equal(arrays["control_step"], np.arange(1, total + 1))
            and arrays["valid"].shape == (total,) and arrays["valid"].all(), "NPZ must cover every valid first-episode step")
    offset = 0
    for piece in episode["source_slices"]:
        chunk = manifest["chunks"][piece["chunk_index"]]
        require(piece["sha256"] == chunk["sha256"] and piece["file"] == chunk["file"]
                and piece["chunk_metadata"] == chunk["metadata"] and piece["environment_axis_index"] == env,
                "extraction chunk provenance differs from original manifest")
        start, end = piece["source_rows_start"], piece["source_rows_stop_exclusive"]
        stop = piece["output_rows_stop_exclusive"]
        require(piece["output_rows_start"] == offset and stop - offset == end - start, "extraction source/output row ranges differ")
        clock_start = chunk["metadata"]["step_start"] if task == "stair" else chunk["metadata"]["frame_start"] + 1
        require(np.array_equal(arrays["control_step"][offset:stop], np.arange(clock_start + start, clock_start + end)), "source row clock differs")
        if task == "stair":
            require(chunk["metadata"]["run_index"] == run, "Stair source chunk belongs to another run")
        offset = stop
    require(offset == total, "extraction source rows are incomplete")
    if task == "push":
        require(np.all(arrays["episode_index"] == metadata["episode_indices"][env]), "Push array crossed first episode")

    snapshot, images = read_native_snapshot_media(inside(export, episode["snapshot_manifest"]),
        manifest_sha256=recording["sha256"], selected_steps=steps)
    require(snapshot == recording["manifest"] and snapshot["env_index"] == env
            and snapshot["seed"] == recording["seed"], "original snapshot identity differs")
    require(snapshot["schema_version"] in {"native_rgb_snapshots.v3", "native_rgb_snapshots.v4"}
            and snapshot["snapshot_interval_steps"] == 1 and snapshot["sampling_mode"] == "interval_from_step_1",
            "this development profile requires native v3/v4 interval-1 sampling")
    require(recording["includes_terminal_frame"] is True and recording["frame_count"] == total
            and snapshot["episode_control_step_count"] == total
            and [sample["control_step"] for sample in snapshot["samples"]] == list(range(1, total + 1)),
            "this development profile requires full step 1..T including terminal")
    view_ids = sorted(view["view_id"] for view in snapshot["samples"][0]["views"])
    require(len(view_ids) == len(set(view_ids)) == 2
            and all(sorted(view["view_id"] for view in sample["views"]) == view_ids for sample in snapshot["samples"])
            and snapshot["snapshot_count"] == 2 * total, "each actual control step must have the same two views")
    require(snapshot["resolution"] == [1920, 1080] and len(images) == 2 * len(steps), "use original 1920x1080 dual-view PNGs")
    body_anchors = None
    if task == "stair":
        projection = metadata["runtime_projection"]
        alignment = align_stair_snapshot_samples(snapshot, control_step=arrays["control_step"],
            base_pos_w=arrays["base_pos_w"], base_quat_w=arrays["base_quat_w"],
            joint_pos=arrays["joint_pos"], episode_length=arrays["episode_length"],
            joint_ids=episode["pairing_metadata"]["joint_mapping"]["joint_ids"],
            numeric_joint_names=projection["joint_names"], kinematic_base_source=projection["kinematic_base_source"])
        profile_path = inside(SOURCE_ROOT / "configs/robots", projection["robot_profile_id"] + ".json")
        frame_path = profile_path.with_suffix(".body_frame.json")
        root_frame = bind_stair_root_frame(read_json(frame_path) if frame_path.exists() else None,
            runtime_projection=projection, robot_profile=read_json(profile_path))
        definitions, claims = stair_s0_definitions(root_frame), STAIR_MEASUREMENT_CLAIMS
    else:
        alignment = align_push_snapshot_samples(snapshot, control_step=arrays["control_step"],
            pose=arrays["pose"], planar_motion=arrays["planar_motion"])
        definitions, claims = PUSH_S0_DEFINITIONS, push_measurement_claims(metadata["parameters"]["recovery_height_m"])
        if snapshot["schema_version"] == "native_rgb_snapshots.v4":
            require("resolved_cell_manifest.json" in extracted["source_files"],
                    "v4 Push body geometry requires the original run resolved cell proof")
            cell = ResolvedCellManifestV1.from_mapping(read_json(export / "resolved_cell_manifest.json"))
            require(cell.manifest_hash == numeric_id["resolved_cell_manifest_hash"],
                    "body geometry cell proof differs from the numeric run")
            profile_id = cell.robot_profile["robot_profile_id"]
            profile_path = inside(SOURCE_ROOT / "configs/robots", profile_id + ".json")
            anchor_path = profile_path.with_suffix(".body_anchors.json")
            body_anchors = bind_push_body_anchors(read_json(anchor_path) if anchor_path.exists() else None,
                runtime_projection={"robot_profile_id": profile_id,
                    "robot_profile_hash": cell.robot_profile["profile_hash"],
                    "body_names": snapshot["articulation_body_names"]},
                robot_profile=read_json(profile_path))
            definitions += push_body_geometry_definitions(body_anchors)
    require(alignment["status"] == "overlap_aligned" and len(alignment["samples"]) == total,
            "complete snapshot-to-numeric overlap alignment did not pass")
    native_bodies = snapshot["articulation_body_names"]
    require(isinstance(native_bodies, list) and native_bodies
            and all(isinstance(name, str) and name for name in native_bodies)
            and len(native_bodies) == len(set(native_bodies)), "recorded articulation body names are required")
    body_names = tuple(dict.fromkeys(("root", *native_bodies)))
    bound = {"session_id": summary["job_id"], "task_id": numeric_id["task_id"],
        "robot_id": summary["resolved_cell_identity"]["robot_profile_id"],
        "protocol_id": numeric_id["resolved_protocol_hash"] if task == "push" else metadata["semantics"]["task_protocol_hash"],
        "source_id": reference["manifest_sha256"], "checkpoint_id": numeric_id["checkpoint_hash"],
        "episode_id": f"r{run}.e{env}", "run_id": str(run)}
    private_identity = {**bound, **{"numeric_" + key: bound[key] for key in IDENTITY_FIELDS},
        "run_index": run, "env_index": env, "runner_seed": recording["seed"],
        "is_replay": selection["scope"] != "n5_confirmation",
        "numeric_relation": "same_run_numeric_collected_with_media",
        "snapshot_manifest_sha256": recording["sha256"], "extraction_sha256": extraction_sha,
        "recordings_sha256": digest(export / "video/recordings.json")}
    if task == "stair":
        private_identity["root_frame_binding"] = ({"status": "asset_verified_specification_bound",
            "specification": root_frame,
            "frame_spec_file": {"path": str(frame_path), "sha256": digest(frame_path)},
            "robot_profile_file": {"path": str(profile_path), "sha256": digest(profile_path)}}
            if root_frame is not None else {"status": "missing", "reason": "no verified frame specification; forward scalar omitted"})
    elif body_anchors is not None:
        private_identity["body_anchor_binding"] = {"status": "asset_verified_specification_bound",
            "specification": body_anchors, "resolved_cell_manifest_hash": cell.manifest_hash,
            "spec_file_sha256": digest(anchor_path), "robot_profile_file_sha256": digest(profile_path)}
    window = MediaWindow(alias=alias, camera_version=snapshot["camera_tracking_version"],
        layout_version=LAYOUT_VERSION, roi_version="original_full_rgb.v1", body_names=body_names,
        images=images, identity=private_identity, measurement_claims=claims, s0_definitions=definitions,
        visibility="selected native frames; omitted steps are not displayed")
    scalars = (stair_s0_for_window(window, identity=bound, control_step=arrays["control_step"],
                 base_pos_w=arrays["base_pos_w"], base_quat_w=arrays["base_quat_w"],
                 terrain_origin_w=arrays["terrain_env_origins_w"], root_frame=root_frame) if task == "stair" else
               push_s0_for_window(window, identity=bound, control_step=arrays["control_step"],
                 pose=arrays["pose"], planar_motion=arrays["planar_motion"]))
    if task == "stair":
        window = replace(window, body_image_points=stair_body_image_points(window, snapshot, root_frame=root_frame))
    else:
        scalars += push_body_geometry_s0(window, snapshot, binding=body_anchors)
        window = replace(window, body_image_points=push_body_image_points(window, snapshot, binding=body_anchors))
    window = replace(window, s0=scalars)
    window = replace(window, measured_facts=(stair_pose_facts(window) if task == "stair"
                                             else push_pose_facts(window) + push_body_geometry_facts(window, binding=body_anchors)))
    numeric_check = (stair_numeric_check(identity=bound, control_step=arrays["control_step"],
                        base_pos_w=arrays["base_pos_w"], base_quat_w=arrays["base_quat_w"]) if task == "stair" else
                     push_numeric_check(identity=bound, control_step=arrays["control_step"],
                        pose=arrays["pose"], planar_motion=arrays["planar_motion"],
                        recovery_height_m=metadata["parameters"]["recovery_height_m"]))
    prepared = prepare_observer_request([window], model=MODEL, max_tokens=MAX_TOKENS,
        generation_options=GENERATION_OPTIONS,
        extra_blind_tokens=(reference["manifest_sha256"], numeric_id["checkpoint_hash"], summary["job_id"]))
    budget = ObserverBudget(max_pixels=50_000_000, max_image_bytes=50_000_000,
                            max_cost_usd=RESERVATION_USD, max_calls=1)
    reservation = budget.reserve([window], RESERVATION_USD)
    request = prepared["request_bytes"]
    require(len(request) <= REQUEST_BODY_LIMIT_BYTES,
            "native image request exceeds DeepSeek's 48 MiB HTTP body limit; choose fewer original frames")
    request_sha = hashlib.sha256(request).hexdigest()
    mechanical = {"passed": True, "source_files_hash_checked": len(extracted["source_files"]),
        "extracted_npz_hash_checked": len(extracted["episodes"]), "original_chunk_bytes_reopened": False,
        "original_chunk_hash_and_row_provenance": "verified against copied sealed numeric manifest",
        "profile": "old_development_interval1_two_views_terminal_inclusive_1920x1080",
        "complete_numeric_alignment": alignment["status"], "aligned_control_steps": total,
        "request_text_and_image_audit": prepared["audit"], "reservation": reservation,
        "manual_pixel_quality_review": "pending; mechanical preparation does not establish visual readability"}
    card = {"status": "PREPARED_NOT_SENT", "sent": False, "provider_requests": 0,
        "scope": selection["scope"], "purpose": selection["purpose"], "task": task,
        "model": MODEL, "thinking": "enabled", "reasoning_effort": "high", "max_tokens": MAX_TOKENS,
        "expected_fingerprint": FINGERPRINT, "actual_fingerprint": None,
        "request_sha256": request_sha, "request_bytes": len(request),
        "provider_request_body_limit_bytes": REQUEST_BODY_LIMIT_BYTES,
        "selection_sha256": hashlib.sha256(selection_bytes).hexdigest(), "extraction_sha256": extraction_sha,
        "selected_steps": steps, "view_ids": view_ids, "reservation": reservation,
        "actual_api_cost_usd": 0, "reservation_is_not_spend": True,
        "driver_sha256": digest(Path(__file__)),
        "dependencies": {str(Path(module.__file__).resolve()): digest(Path(module.__file__)) for module in (
            sys.modules[prepare_observer_request.__module__], sys.modules[align_stair_snapshot_samples.__module__],
            sys.modules[align_push_snapshot_samples.__module__])}}
    private = {"request_sha256": request_sha, "expected_fingerprint": FINGERPRINT,
        "prepare_private_sidecar": prepared["private_sidecar"], "selection": selection,
        "source_export": str(export), "numeric_reference": reference, "numeric_identity": numeric_id,
        "snapshot_manifest": episode["snapshot_manifest"], "numeric_file": episode["numeric_file"],
        "numeric_sha256": episode["numeric_sha256"], "alignment_not_checked": alignment["not_checked"]}
    if episode_alias is not None:
        card["episode_alias"] = private["episode_alias"] = alias
        mechanical["profile"] = "n5_interval1_two_views_terminal_inclusive_1920x1080"
    return {"prepared": prepared, "windows": (window,), "numeric_check": numeric_check,
        "selection_bytes": selection_bytes, "card": card, "private_sidecar": private,
        "mechanical_audit": mechanical, "alignment": alignment}


def prepare(export: Path, selection_path: Path, output: Path, *, episode_alias: str | None = None) -> dict:
    output = output.resolve()
    require(not output.exists(), "output must be a fresh directory")
    built = build(export, selection_path, episode_alias=episode_alias)
    output.mkdir(parents=True, exist_ok=False)
    (output / "request.json").write_bytes(built["prepared"]["request_bytes"])
    (output / "selection.json").write_bytes(built["selection_bytes"])
    for name in ("private_sidecar", "mechanical_audit", "alignment", "card"):
        (output / (name + ".json")).write_text(
            json.dumps(built[name], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return built["card"]


def prepare_report(export: Path, selection_dir: Path, capture_card: Path, output: Path) -> dict:
    """Freeze both packets and their cumulative reservation offline, without release."""
    export, selection_dir, capture_card, output = (
        path.resolve() for path in (export, selection_dir, capture_card, output))
    require(not output.exists(), "report output must be fresh")
    selection_path = selection_dir / "selection.json"
    selection, selection_card = read_json(selection_path), read_json(selection_dir / "card.json")
    selection_episode(selection, "e1")
    require(selection["scope"] == "n5_confirmation"
            and selection_card["observer_profile"] == "READY_FOR_PREPARATION"
            and selection_card["pool_complete"] is True and not selection_card["missing_classes"],
            "N5 requires a complete pool and both outcome classes")
    require(selection_card["selection_sha256"] == digest(selection_path)
            and selection["source"]["capture_card_sha256"] == digest(capture_card),
            "selection or capture hash differs")
    built = [build(export, selection_path, episode_alias=alias) for alias in ("e1", "e2")]
    budget = report_budget()
    totals = [budget.reserve(item["windows"], RESERVATION_USD) for item in built]
    output.mkdir(parents=True, exist_ok=False)
    packets = []
    for alias, item in zip(("e1", "e2"), built):
        packet = output / alias
        packet.mkdir()
        (packet / "request.json").write_bytes(item["prepared"]["request_bytes"])
        (packet / "selection.json").write_bytes(item["selection_bytes"])
        for name in ("private_sidecar", "mechanical_audit", "alignment", "card"):
            (packet / (name + ".json")).write_text(
                json.dumps(item[name], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        packets.append({"episode_alias": alias, "directory": alias,
            "prepared_card_sha256": digest(packet / "card.json"),
            "request_sha256": item["card"]["request_sha256"]})
    value = {"schema_version": "p3new.native_observer_report.v1", "status": "PREPARED_NOT_RELEASED",
        "scope": "n5_confirmation", "task": built[0]["card"]["task"], "packets": packets,
        "selection_sha256": digest(selection_path), "extraction_sha256": digest(export / "extraction.json"),
        "capture_card": {"path": str(capture_card), "sha256": digest(capture_card)},
        "selection_card": {"path": str(selection_dir / "card.json"), "sha256": digest(selection_dir / "card.json")},
        "model": MODEL, "expected_fingerprint": FINGERPRINT, "reserved_cost_usd_per_call": RESERVATION_USD,
        "report_limits": {"images": 24, "subframes": 96, "pixels": 50_000_000,
            "image_bytes": 100_000_000, "calls": 2, "reserved_cost_usd": 0.10},
        "cumulative_reservations": totals, "provider_requests": 0, "automatic_retry": False}
    (output / "report.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--export", dest="export_dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-alias", choices=("e1", "e2"))
    parser.add_argument("--capture-card", type=Path,
                        help="Prepare both N5 packets; --selection is the selection directory")
    args = parser.parse_args()
    if args.capture_card:
        require(args.episode_alias is None, "report preparation selects both aliases")
        card = prepare_report(args.export_dir, args.selection, args.capture_card, args.output)
        print(json.dumps({key: card[key] for key in ("status", "task", "provider_requests")}))
        return 0
    card = prepare(args.export_dir, args.selection, args.output, episode_alias=args.episode_alias)
    print(json.dumps({key: card[key] for key in ("status", "task", "request_sha256", "request_bytes", "provider_requests")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
