"""Independent CPU development check; expected segments use NumPy masks, not production."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.tasks.stair_climbing.behavior import _progress_process
from true_rl_agent.tasks.stair_climbing.process_report import _progress_outline


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent(samples, events):
    """Connected components of x < prefix maximum; separate from production state machine."""
    a = np.asarray(samples, dtype=np.float64)
    if not len(a):
        return []
    under = a[:, 0] < np.maximum.accumulate(a[:, 0])
    starts = np.flatnonzero(under & ~np.r_[False, under[:-1]])
    ends = np.flatnonzero(under & ~np.r_[under[1:], False])
    result = []
    for start, last in zip(starts, ends):
        peak = int(start - 1)
        trough = int(start + np.argmin(a[start:last+1, 0]))
        recovery = int(last + 1) if last + 1 < len(a) else None
        end = recovery if recovery is not None else int(last)
        extrema = {}
        for column, name in enumerate(("roll_rad", "pitch_rad", "up_z"), 1):
            values = a[peak:end+1, column]
            lo, hi = int(values.argmin()) + peak, int(values.argmax()) + peak
            extrema[name] = [round(float(a[lo, column]), 6), lo+1,
                             round(float(a[hi, column]), 6), hi+1]
        overlaps = [i for i, event in enumerate(events)
                    if event["code"] in ("tilt", "low_progress")
                    and not (event["end"] < peak+1 or event["start"] > end+1)]
        result.append({"peak_step": peak+1, "trough_step": trough+1,
            "recovery_step": None if recovery is None else recovery+1,
            "observed_end_step": end+1,
            "peak_x_m": round(float(a[peak, 0]), 6), "trough_x_m": round(float(a[trough, 0]), 6),
            "end_x_m": round(float(a[end, 0]), 6), "drop_m": round(float(a[peak, 0]-a[trough, 0]), 6),
            "attitude_extrema": extrema, "overlapping_root_event_indices": overlaps})
    return result


def main():
    production = [ROOT / f"src/true_rl_agent/tasks/stair_climbing/{name}.py"
                  for name in ("behavior", "process_report")]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in production}
    base = HERE.parent / "n5_review_capacity_development_r2"
    rows, hashes, errors = [], {}, []
    for ordinal in range(1, 13):
        packet = base / f"prepared_report_capacity/episode-{ordinal:02}/gpt6-sol/request.http_body.json"
        hashes[str(packet.relative_to(ROOT))] = sha(packet)
        messages = json.loads(packet.read_bytes())["messages"][1]["content"]
        blocks = {v["role"]: v["data"] for v in (json.loads(m["text"]) for m in messages)}
        raw = blocks["raw_numeric"]
        old = blocks["full_derivation"]["complete_episode_derivation"]
        assert raw["episode"]["task"] == "stair"
        npz = base / "materials/raw/episodes" / raw["episode"]["episode_id"] / "original_values.npz"
        hashes[str(npz.relative_to(ROOT))] = sha(npz)
        assert sha(npz) == raw["original_values_npz"]["sha256"]
        with np.load(npz, allow_pickle=False) as arrays:
            x = arrays["raw__base_pos_w"][:, 0].astype(float) - float(arrays["static__terrain_env_origins_w"][0])
            q = arrays["raw__base_quat_w"].astype(float)
        w, qx, qy, qz = q.T
        roll = np.arctan2(2*(w*qx+qy*qz), 1-2*(qx*qx+qy*qy))
        pitch = np.arcsin(np.clip(2*(w*qy-qz*qx), -1, 1))
        up = 1-2*(qx*qx+qy*qy)
        samples = np.column_stack((x, roll, pitch, up)).tolist()
        events = deepcopy(old["events"])
        original_inputs = deepcopy((samples, events))
        expected = independent(samples, events)
        actual = _progress_process(samples, events)
        checks = {"all_segment_fields_equal": actual["excursions"] == expected,
                  "input_samples_and_events_unchanged": (samples, events) == original_inputs}
        prefix = np.maximum.accumulate(x)
        trough = int(np.argmax(prefix-x))
        peak = int(np.argmax(x[:trough+1]))
        legacy = [peak+1, trough+1, round(float(x[peak]-x[trough]), 6)]
        checks["old_max_x_retreat_equals_independent_raw"] = legacy == old["root_motion"]["max_x_retreat"]
        enhanced = {**old, "progress_process": actual}
        text, refs = _progress_outline(enhanced)
        terminal = [i for i, row in enumerate(expected) if row["recovery_step"] is None]
        ranked = sorted(range(len(expected)), key=lambda i: (-expected[i]["drop_m"], expected[i]["peak_step"]))
        selected = sorted((terminal + [i for i in ranked if i not in terminal])[:2],
                          key=lambda i: expected[i]["peak_step"])
        indices = [int(ref.rsplit("/", 1)[-1]) for ref in refs if "/excursions/" in ref]
        checks["public_selected_indices_equal"] = indices == selected
        checks["public_shown_denominator_explicit"] = f"{len(selected)}/{len(expected)} positive x-excursions shown" in text
        checks["public_terminal_unreturned_retained"] = not terminal or terminal[0] in selected
        checks["all_event_refs_are_existing_root_events"] = all(
            events[i]["body"] == "root" for row in actual["excursions"] for i in row["overlapping_root_event_indices"])
        if not all(checks.values()):
            errors.append({"episode": raw["episode"]["episode_id"], "checks": checks})
        rows.append({"packet": f"episode-{ordinal:02}", "raw_episode": raw["episode"]["episode_id"],
            "logical_id": old["episode_id"], "steps": len(x), "outcome": old["primary_reason"],
            "checks": checks, "old_max_x_retreat": legacy, "excursions": expected,
            "public_selected_indices": selected, "public_text": text})

    # Fixed hand-written step witnesses; no production-derived expected values.
    cases = [
        ("equal_peaks_and_troughs", [0,1,1,0,0,1,1,.5,1], [(3,4,6),(7,8,9)]),
        ("terminal_return", [0,1,0,1], [(2,3,4)]),
        ("terminal_unreturned", [0,1,0,0], [(2,3,None)]),
        ("all_flat", [1,1,1], []),
        ("single_sample", [1], []),
    ]
    synthetic = []
    for name, xvalues, witnesses in cases:
        samples = [[value, -i*.01, i*.02, 1.] for i, value in enumerate(xvalues)]
        event = [{"body": "root", "code": "tilt", "start": 1, "end": len(xvalues), "confirmed": None}]
        expected = independent(samples, event)
        actual = _progress_process(samples, event)["excursions"]
        tuples = [(r["peak_step"],r["trough_step"],r["recovery_step"]) for r in actual]
        passed = tuples == witnesses and actual == expected
        synthetic.append({"case": name, "hand_written_step_witnesses": witnesses, "pass": passed})
        if not passed:
            errors.append({"case": name})
    assert all(sha(ROOT / path) == digest for path, digest in source_hashes.items())
    assert all(sha(ROOT / path) == digest for path, digest in hashes.items())
    result = {"status": "DEVELOPMENT_NUMERIC_CHECK_PASS" if not errors else "DEVELOPMENT_NUMERIC_CHECK_FAIL",
        "not_n5_qualification": True, "semantic_review_vote": False, "provider_requests": 0, "gpu_runs": 0,
        "source_hashes": source_hashes, "input_hashes": hashes, "inputs_unchanged": True,
        "independent_method": "NumPy prefix maximum then connected components of strict-below mask; raw float32 NPZ operands cast to float64; quaternion Euler independently computed; all exported numeric fields compared at existing six-decimal serialization precision",
        "scope_limit": "Calls actual production progress helper and public outline with original ordered events; does not regenerate the full archive or requalify event predicates/sensor physics. Old global retreat is independently checked against frozen full derivation, whose source logic remains separate.",
        "episodes_checked": len(rows), "segments_checked": sum(len(r["excursions"]) for r in rows),
        "terminal_unreturned_episodes": sum(any(e["recovery_step"] is None for e in r["excursions"]) for r in rows),
        "errors": errors, "episodes": rows, "synthetic_boundary_cases": synthetic}
    output = HERE / "independent_check.json"
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({k: result[k] for k in ("status", "episodes_checked", "segments_checked", "terminal_unreturned_episodes", "errors")}))
    print("sha256=" + sha(output))


if __name__ == "__main__":
    main()
