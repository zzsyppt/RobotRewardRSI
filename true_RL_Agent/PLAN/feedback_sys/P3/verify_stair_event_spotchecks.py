"""Bounded independent raw-array oracle for new P3 Stair event intervals.

Acceptance artifact only. No production behavior, transition decoder, geometry
function, or tracker is imported. Fixed cases cover positive and absent events,
short unconfirmed spans, success and failure. Regular flat-tread clearance is
checked from the archived course dimensions; C1/C2 clearance is outside scope.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


CASES = ("r1.e0", "r1.e48", "r1.e57", "r1.e108", "r1.e135", "r1.e171")
FIELDS = ("base_pos_w", "support_pos_w", "support_lin_vel_w",
          "support_contact_forces_w", "episode_length")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def spans(mask, required, code, body):
    """Vector-boundary run lengths, independent of the production state machine."""
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [{"code": code, "body": body, "start": int(a + 1), "end": int(b),
             "confirmed": int(a + required) if b - a >= required else None}
            for a, b in zip(changes[::2], changes[1::2], strict=True)]


def flat_clearance(metadata, episode, positions, origin):
    """Exact archived regular treads only; never approximate C2 as a flat stair."""
    suite = metadata["evaluation_suite"]
    cohort, template_id = episode["cell"].split(".", 1)
    template = next(t for t in suite["cell_templates"] if t["template_id"] == template_id)
    geometry = suite.get("cohort_geometry_overrides", {}).get(cohort, {}).get(template_id, template["geometry"])
    if geometry["kind"] != "regular":
        return None, {"status": "NOT_CHECKED", "reason": "independent clearance oracle limited to regular flat treads"}
    terrain = metadata["training_profile"]["terrain"]
    n = int(terrain["step_count"])
    start, depth = float(terrain["start_platform_length_m"]), float(geometry["tread_depth_m"])
    height = float(geometry["riser_height_m"])
    # Each riser begins after the start platform and the preceding treads.
    # Equality is assigned to the lower surface, matching the stated projection.
    x = positions[:, :, 0] - float(origin[0])
    edges = start / 2 + np.arange(n, dtype=np.float64) * depth
    levels = np.sum(x[:, :, None] > edges, axis=-1)
    # Accumulation preserves the course's per-riser construction precision.
    heights = np.concatenate(([0.], np.cumsum(np.repeat(height, n))))
    clearance = positions[:, :, 2] - float(origin[2]) - heights[levels]
    return clearance, {"status": "CHECKED", "kind": "regular", "step_count": n,
                       "riser_height_m": height, "tread_depth_m": depth,
                       "start_platform_length_m": start, "cohort": cohort}


def audit_episode(metadata, episode, arrays, origin, names):
    signals = metadata["semantics"]["episode_outcomes"]["signals"]
    require(np.array_equal(arrays["episode_length"], np.arange(1, episode["steps"] + 1)),
            episode["episode_id"] + ": recorded first-episode clock/coverage differs")
    force = arrays["support_contact_forces_w"].astype(np.float64)
    velocity = arrays["support_lin_vel_w"].astype(np.float64)
    loaded = np.sqrt(np.sum(force * force, axis=-1)) > signals["slip"]["contact_force_threshold"]
    speed = np.sqrt(np.sum(velocity[:, :, :2] ** 2, axis=-1))
    slip = loaded & (speed > signals["slip"]["speed_threshold"])
    rising = ~loaded & (velocity[:, :, 2] > 0.05)
    lowering = ~loaded & (velocity[:, :, 2] < -0.05)
    # Twenty-five elapsed control steps contain twenty-six sampled endpoints.
    # Subtracting a fixed terrain origin cancels; use raw world x directly.
    x = arrays["base_pos_w"][:, 0].astype(np.float64)
    window = math.ceil(0.5 / episode["dt_s"])
    stalled = np.zeros(len(x), dtype=bool)
    for stop in range(window, len(x)):
        stalled[stop] = np.max(x[stop-window:stop+1]) - x[stop-window] < 0.02
    expected = spans(stalled, 1, "low_progress", "root")
    codes = {"low_progress", "slip_body", "unloaded_rising", "unloaded_lowering"}
    clearance, geometry = flat_clearance(metadata, episode, arrays["support_pos_w"].astype(np.float64), origin)
    if clearance is not None:
        codes.add("low_clearance_body")
    for index, name in enumerate(names):
        expected.extend(spans(slip[:, index], signals["slip"]["consecutive_control_steps"], "slip_body", name))
        expected.extend(spans(rising[:, index], 3, "unloaded_rising", name))
        expected.extend(spans(lowering[:, index], 3, "unloaded_lowering", name))
        if clearance is not None:
            low = ~loaded[:, index] & (clearance[:, index] < signals["insufficient_clearance"]["threshold"])
            expected.extend(spans(low, signals["insufficient_clearance"]["consecutive_control_steps"], "low_clearance_body", name))
    key = lambda event: (event["code"], event["body"], event["start"])
    actual = sorted((e for e in episode["events"] if e["code"] in codes), key=key)
    expected.sort(key=key)
    require(expected == actual, episode["episode_id"] + ": raw oracle intervals/confirmations differ")
    return {"status": "PASS", "episode_id": episode["episode_id"], "cell": episode["cell"],
            "outcome_label_from_full_view": episode["primary_reason"], "valid_steps": len(x),
            "checked_codes": sorted(codes), "independent_clearance_scope": geometry,
            "interval_counts": dict(Counter(e["code"] for e in expected)),
            "confirmed_counts": dict(Counter(e["code"] for e in expected if e["confirmed"] is not None)),
            "unconfirmed_counts": dict(Counter(e["code"] for e in expected if e["confirmed"] is None)),
            "oracle_intervals": expected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=Path, required=True)
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--expected-full-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    report = {"schema_version": "true_rl_agent.feedback_p3.stair_raw_event_spotchecks.v1",
              "status": "FAIL", "command": [sys.executable, *sys.argv], "verifier_sha256": sha(__file__),
              "phase_file_sha256": sha(args.phase), "full_file_sha256": sha(args.full),
              "fixed_case_ids": list(CASES), "datasets": []}
    try:
        require(report["full_file_sha256"] == args.expected_full_sha256, "full evidence differs from supplied frozen hash")
        phase, full = read(args.phase), read(args.full)
        reference = phase["task_evidence"]["numeric_evidence"]
        manifest_path = Path(reference["manifest_path"])
        require(sha(manifest_path) == reference["manifest_sha256"] == full["domain"]["archive_hash"], "archive binding differs")
        manifest = read(manifest_path)
        require(manifest["complete"] is True, "raw archive incomplete")
        require(manifest["identity"]["checkpoint_hash"] == phase["checkpoint_hash"] == full["domain"]["checkpoint_hash"], "checkpoint binding differs")
        require(manifest["identity"]["splits"] == ["public_dev"], "unexpected evidence split")
        metadata = manifest["metadata"]
        names = metadata["runtime_projection"]["resolved_roles"]["support_effector"]
        require(names == full["semantics"]["support_bodies"], "named support order differs from archive")
        require(full["semantics"]["thresholds"]["low_progress_window_s"] == 0.5
                and full["semantics"]["thresholds"]["low_progress_gain_m"] == 0.02
                and full["semantics"]["thresholds"]["unloaded_vertical_speed_mps"] == 0.05
                and full["semantics"]["thresholds"]["unloaded_motion_steps"] == 3, "new event specification changed")
        selected = {e["episode_id"]: e for e in full["episodes"] if e["episode_id"] in CASES}
        require(set(selected) == set(CASES), "fixed episode missing")
        storage = {name: {key: [] for key in FIELDS} for name in CASES}
        origins, chunks = {}, []
        for chunk in manifest["chunks"]:
            if chunk["metadata"]["run_index"] != 1:
                continue
            path = manifest_path.parent / chunk["file"]
            require(sha(path) == chunk["sha256"], "raw chunk hash differs: " + chunk["file"])
            chunks.append({"file": chunk["file"], "sha256": chunk["sha256"]})
            with np.load(path, allow_pickle=False) as raw:
                for name, episode in selected.items():
                    index = episode["slot"]
                    if "terrain_env_origins_w" in raw:
                        origins[name] = raw["terrain_env_origins_w"][index].copy()
                        template = metadata["evaluation_suite"]["cell_templates"][int(raw["terrain_types"][index])]
                        require(episode["cell"] == "public_dev." + template["template_id"],
                                name + ": raw terrain assignment differs from full view")
                        require(episode["dt_s"] == metadata["evaluation_suite"]["geometry_policy"]["control_dt_s"],
                                name + ": control period differs from archived suite")
                    valid = raw["valid"][:, index]
                    for field in FIELDS:
                        storage[name][field].append(raw[field][:, index][valid])
        for name in CASES:
            arrays = {key: np.concatenate(parts, axis=0) for key, parts in storage[name].items()}
            report["datasets"].append(audit_episode(metadata, selected[name], arrays, origins[name], names))
        # Non-vacuity checks are specific to this frozen six-case acceptance set.
        counts = Counter(e["code"] for row in report["datasets"] for e in row["oracle_intervals"] if e["confirmed"] is not None)
        require(all(counts[code] > 0 for code in ("low_progress", "slip_body", "unloaded_rising", "unloaded_lowering", "low_clearance_body")), "chosen cases do not exercise each confirmed event")
        require(any(not row["interval_counts"].get("low_progress", 0) for row in report["datasets"]), "no absent low-progress counterexample")
        report.update(status="PASS", archive_manifest_sha256=reference["manifest_sha256"],
                      raw_chunks=chunks, named_supports=names,
                      nonvacuity={"confirmed_event_counts": dict(counts),
                                  "absent_low_progress_cases": [row["episode_id"] for row in report["datasets"]
                                                                 if not row["interval_counts"].get("low_progress", 0)],
                                  "short_unconfirmed_span_counts": dict(Counter(
                                      e["code"] for row in report["datasets"] for e in row["oracle_intervals"]
                                      if e["confirmed"] is None))},
                      total_valid_control_steps=sum(row["valid_steps"] for row in report["datasets"]),
                      total_intervals=sum(len(row["oracle_intervals"]) for row in report["datasets"]))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
    report["elapsed_seconds"] = time.perf_counter() - start
    report["boundaries"] = [
        "Six fixed run-1 Go2 episodes; this is an interval-level spotcheck, not exhaustive physical semantic validation.",
        "All valid raw steps of each selected first episode are checked; invalid padded samples are excluded.",
        "No production package is imported. NumPy/raw arithmetic independently reconstructs active masks, boundaries, and streak confirmations.",
        "Regular flat-tread clearance is independently checked for r1.e0 and r1.e171; C1/C2 clearance is explicitly not checked here.",
        "Outcome labels are retained for sampling description; this tool does not independently reclassify success/failure.",
        "No causal attribution, author value, video validation, or post-first-arrival claim follows from these comparisons.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "error", "total_valid_control_steps", "total_intervals", "elapsed_seconds")}), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
