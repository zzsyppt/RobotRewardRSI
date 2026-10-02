"""Nonblind development audit of one frozen old episode; never changes scores."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
SOURCE = HERE.parent / "semantic_execution_scope_r2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(mask):
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]]) + 1
    ends = np.flatnonzero(mask & np.r_[~mask[1:], True]) + 1
    return [[int(a), int(b)] for a, b in zip(starts, ends)]


def main():
    request_path = SOURCE / "inputs/request.http_body.json"
    answer_path = SOURCE / "output/raw_answer.json"
    watched = [request_path, answer_path, PROJECT / "src/true_rl_agent/tasks/stair_climbing/behavior.py",
               PROJECT / "src/true_rl_agent/tasks/stair_climbing/process_report.py"]
    hashes = {str(p): sha(p) for p in watched}
    request, answer = (json.loads(p.read_bytes()) for p in (request_path, answer_path))
    blocks = {row["role"]: row for row in (json.loads(b["text"]) for b in request["messages"][1]["content"])}
    raw, full = blocks["raw_numeric"]["data"], blocks["full_derivation"]["data"]
    steps = raw["episode"]["steps"]

    def field(section, name):
        value = raw[section][name]
        if "by_c_order_component" in value:
            columns = [np.fromstring(component, sep=",") for component in value["by_c_order_component"]]
        else:
            columns = []
            for intervals in value["inclusive_constant_intervals_by_component"]:
                column = np.full(steps, np.nan)
                for first, last, number in intervals:
                    column[first-1:last] = number
                assert np.isfinite(column).all()
                columns.append(column)
        assert all(len(column) == steps for column in columns)
        # Restore the declared raw dtype before arithmetic; decimal text is a
        # round-trip transport, not an independent float64 measurement.
        return np.column_stack(columns).reshape((steps, *value["component_shape"])).astype(value["dtype"]).astype(float)

    x = field("raw_columns", "base_pos_w")[:, 0]
    q = field("raw_columns", "base_quat_w")
    w, qx, qy, qz = q.T
    roll = np.arctan2(2*(w*qx+qy*qz), 1-2*(qx*qx+qy*qy))
    pitch = np.arcsin(np.clip(2*(w*qy-qz*qx), -1, 1))
    supplied = field("direct_columns", "root_roll_pitch_rad")
    assert np.allclose(np.column_stack((roll, pitch)), supplied, atol=1e-14, rtol=0)
    excursions, peak, begin, trough = [], 0, None, None
    for index in range(1, steps):
        if x[index] >= x[peak]:
            if begin is not None:
                excursions.append({"peak_step": begin+1, "trough_step": trough+1, "return_step": index+1,
                    "drop_m": float(x[begin]-x[trough]), "peak_x_w_m": float(x[begin]),
                    "trough_x_w_m": float(x[trough]), "return_x_w_m": float(x[index])})
                begin = trough = None
            peak = index
        else:
            if begin is None:
                begin, trough = peak, index
            elif x[index] < x[trough]:
                trough = index
    if begin is not None:
        excursions.append({"peak_step": begin+1, "trough_step": trough+1, "return_step": None,
                           "drop_m": float(x[begin]-x[trough])})
    for row in excursions:
        first, last = row["peak_step"]-1, row["return_step"] or steps
        row["signed_context"] = {
            "interval_inclusive": [first+1, last],
            "roll_min_max_rad": [float(roll[first:last].min()), float(roll[first:last].max())],
            "pitch_min_max_rad": [float(pitch[first:last].min()), float(pitch[first:last].max())],
            "pitch_negative_threshold_intersections": [[a+first, b+first] for a, b in spans(pitch[first:last] < -.65)],
        }
    gain = np.full(steps, np.nan)
    for index in range(25, steps):
        gain[index] = x[index-25:index+1].max()-x[index-25]
    forces = field("direct_columns", "support_force_norm_n")
    velocities = field("raw_columns", "support_lin_vel_w")
    loaded_motion = (forces > 1.) & (np.linalg.norm(velocities[:, :, :2], axis=-1) > .5)
    events = full["complete_episode_derivation"]["events"]
    rows = [dict(zip(events["columns"], row)) for row in events["rows"]]
    selected_ids = {"e02-f" + number for number in ("02", "04", "07", "08", "09", "11", "14", "15", "20", "23")}
    result = {"status": "NONBLIND_DEVELOPMENT_AUDIT_NOT_N5_NOT_RESCORING", "inputs": hashes,
        "scope": "same frozen request raw operands; no visual evidence, no original sensor-physics proof",
        "material_block_refs": {role: {k: row[k] for k in ("material_id", "sha256") if k in row}
                                for role, row in blocks.items()},
        "raw_episode": raw["episode"]["episode_id"], "steps": steps,
        "pitch_recomputed_from_original_quaternion_matches_supplied_direct_values": True,
        "negative_pitch_below_minus_0_65_intervals": spans(pitch < -.65),
        "positive_pitch_above_0_65_intervals": spans(pitch > .65),
        "abs_roll_above_0_65_intervals": spans(np.abs(roll) > .65),
        "low_gain_window_endpoints": spans(gain < .02),
        "low_gain_definition": "inclusive 26-sample window spanning 25 intervals =0.5s, max(x)-start_x<0.02m; not zero motion",
        "excursions": excursions,
        "excursion_definition_for_this_audit": "below a preceding observed running maximum until first x>=that maximum; latest equal peak before descent, earliest equal trough; no minimum-amplitude filter; descriptive, not a new production rule",
        "largest_two_excursions": sorted(excursions, key=lambda row: -row["drop_m"])[:2],
        "front_loaded_motion_intervals_overlapping_140_145": {body: [s for s in spans(loaded_motion[:, i]) if s[0]<=145 and s[1]>=140]
                                                            for i, body in enumerate(("FL_foot", "FR_foot"))},
        "full_root_events": [{"original_index": i, **row} for i, row in enumerate(rows) if row["code"] in ("tilt", "low_progress")],
        "full_root_motion": full["complete_episode_derivation"]["root_motion"],
        "all_forbidden_instant_norms_zero": bool((field("direct_columns", "forbidden_force_norm_n") == 0).all()),
        "all_forbidden_buffer_max_norms_zero": bool((field("direct_columns", "forbidden_buffer_max_force_norm_n") == 0).all()),
        "raw_discrete": raw["raw_discrete"], "contact_history_definition": raw["definitions"]["contact_history"],
        "original_target_facts": [f for f in blocks["own_sealed_facts"]["data"]["findings"]["necessary_facts"] if f["fact_id"] in selected_ids],
        "original_target_matches_not_modified": [m for m in answer["findings"]["fact_matches"] if m["sealed_fact_id"] in selected_ids],
        "old_facts_qualification": "not established; claims checked here are not accepted as ground truth by self-report",
        "provider_requests": 0, "gpu_runs": 0}
    assert all(sha(p) == hashes[str(p)] for p in watched)
    with (HERE / "evidence.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"excursions": len(excursions), "largest_two": result["largest_two_excursions"],
                      "evidence_sha256": sha(HERE / "evidence.json"), "source_files_unchanged": True}))


if __name__ == "__main__":
    main()
