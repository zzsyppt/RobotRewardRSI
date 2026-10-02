"""Independent NumPy checks of new Stair motion fields on the old development pool.

No simulator/provider calls. --derive uses the one production decoder once;
the verification below computes its own quaternion, motion and interval oracle.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.stair_climbing.behavior import build_stair_behavior, project_stair_behavior
from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
from true_rl_agent.prompting.authoring_frame import build_authoring_context_frame, with_final_prompt_budget
from true_rl_agent.prompting.payload import PROMPT_PAYLOAD_SECTION_ORDER
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt

MANIFEST = HERE / "results/stair_numeric_dev_r1/numeric_evidence/manifest.json"
FULL = HERE / "results/stair_motion_v2_full.json"
OLD_FULL = HERE.parent / "P3/results/author_final_bundle/author_blind_review/reviewer/evidence-stair.json"
PROMPT = HERE / "runtime_probe_r1/prompts/stair-01_T.json"
SOURCE = ROOT / "src/true_rl_agent/tasks/stair_climbing/behavior.py"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def intervals(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return list(zip(starts.tolist(), ends.tolist()))


def check_close(actual, expected, label):
    assert np.allclose(actual, expected, atol=1.01e-6, rtol=0), (label, actual, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--derive", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    frozen = {str(path): sha(path) for path in (MANIFEST, OLD_FULL, PROMPT, SOURCE)}
    assert frozen[str(MANIFEST)] == "22fb749620c4e614cbb4df4fbfdb82b70d76c8f1cd81a3eda50de465d9fb3e4a"
    if args.derive:
        ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(MANIFEST), "manifest_sha256": sha(MANIFEST)}
        FULL.write_text(json.dumps(build_stair_behavior(ref), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    full, manifest, old_full = read(FULL), read(MANIFEST), read(OLD_FULL)
    assert full["schema_version"] == "true_rl_agent.stair.behavior.full.v2"
    episodes = {(e["run"], e["slot"]): e for e in full["episodes"]}
    assert len(episodes) == 864
    signals = manifest["metadata"]["semantics"]["episode_outcomes"]["signals"]
    force_threshold = signals["slip"]["contact_force_threshold"]
    body_names = full["semantics"]["support_bodies"]
    assert body_names == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]  # this archived robot, not production
    by_run = defaultdict(list)
    for chunk in manifest["chunks"]:
        path = MANIFEST.parent / chunk["file"]
        assert sha(path) == chunk["sha256"]
        by_run[chunk["metadata"]["run_index"]].append(path)
    oracle, sensitivity = {}, {str(t): 0 for t in (.045, .05, .055)}
    checked_spans = 0
    for run, paths in by_run.items():
        buffers = defaultdict(list)
        origins = None
        for path in paths:
            with np.load(path, allow_pickle=False) as arrays:
                if "terrain_env_origins_w" in arrays:
                    origins = arrays["terrain_env_origins_w"].copy()
                for field in ("base_pos_w", "base_quat_w", "support_lin_vel_w", "support_contact_forces_w", "valid", "episode_length"):
                    buffers[field].append(arrays[field])
        arrays = {field: np.concatenate(parts) for field, parts in buffers.items()}
        for slot in range(arrays["valid"].shape[1]):
            mask = arrays["valid"][:, slot]
            a = {key: value[:, slot][mask] for key, value in arrays.items()}
            count = len(a["episode_length"])
            assert np.array_equal(a["episode_length"], np.arange(1, count+1))
            episode = episodes[(run, slot)]
            assert count == episode["steps"]
            x_position = a["base_pos_w"][:, 0].astype(float)-float(origins[slot, 0])
            w,x,y,z = a["base_quat_w"].astype(float).T
            roll = np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y))
            pitch = np.arcsin(np.clip(2*(w*y-z*x), -1, 1))
            up_z = 1-2*(x*x+y*y)
            columns = {"x_m": x_position, "roll_rad": roll, "pitch_rad": pitch, "up_z": up_z}
            motion = episode["root_motion"]
            check_close(motion["first"], [v[0] for v in columns.values()], (run, slot, "first"))
            check_close(motion["last"], [v[-1] for v in columns.values()], (run, slot, "last"))
            for field, values in columns.items():
                low, high = int(np.argmin(values)), int(np.argmax(values))
                check_close(motion["extrema"][field], [values[low], low+1, values[high], high+1], (run, slot, field))
            drops = np.maximum.accumulate(x_position)-x_position
            trough = int(np.argmax(drops)); peak = int(np.argmax(x_position[:trough+1]))
            check_close(motion["max_x_retreat"], [peak+1, trough+1, drops[trough]], (run, slot, "retreat"))
            negative = np.flatnonzero(up_z < 0)
            assert motion["first_up_z_negative_step"] == (int(negative[0])+1 if len(negative) else None)
            unloaded = np.linalg.norm(a["support_contact_forces_w"].astype(float), axis=-1) <= force_threshold
            speed = np.linalg.norm(a["support_lin_vel_w"].astype(float), axis=-1)
            required = int(np.ceil(.5 / episode["dt_s"]))
            expected = []
            for body_index, body in enumerate(body_names):
                for start, end in intervals(unloaded[:, body_index] & (speed[:, body_index] <= .05)):
                    event = {"code": "unloaded_low_motion", "body": body, "start": start+1, "end": end+1,
                        "confirmed": start+required if end-start+1 >= required else None,
                        "measured": {"max_speed_mps": round(float(speed[start:end+1, body_index].max()), 6),
                                     "root_x_gain_m": round(float(x_position[end]-x_position[start]), 6)}}
                    expected.append(event)
            expected.sort(key=lambda e: (e["start"], e["body"], e["code"]))
            actual = [e for e in episode["events"] if e["code"]=="unloaded_low_motion"]
            assert actual == expected, (run, slot, "low-motion intervals")
            checked_spans += len(expected)
            before = any(e["confirmed"] is not None and e["confirmed"] < count for e in expected)
            assert before == episode["before_terminal"]["unloaded_low_motion"]
            oracle[(run, slot)] = {"low_motion_before_terminal": before, "up_negative": bool(len(negative)),
                "retreat": round(float(drops[trough]), 6),
                **{axis+"_range": [round(float(columns[axis].min()),6), round(float(columns[axis].max()),6)]
                   for axis in ("roll_rad", "pitch_rad")}}
            for threshold in (.045, .05, .055):
                found = any(start+required < count and end-start+1 >= required for b in range(len(body_names))
                    for start, end in intervals(unloaded[:, b] & (speed[:, b] <= threshold)))
                sensitivity[str(threshold)] += int(found)
    view = project_stair_behavior(full)
    assert project_stair_behavior(view) == view and size(view) <= 8192
    totals = [dict(zip(view["profile"]["stratum_totals"]["columns"], row, strict=True)) for row in view["profile"]["stratum_totals"]["rows"]]
    assert sum(row["n"] for row in totals) == 864
    for row in totals:
        group = [e for e in episodes.values() if e["stratum"]==row["stratum"]]
        assert row["n"] == len(group)
        assert row["unloaded_low_motion"] == sum(oracle[(e["run"], e["slot"])]["low_motion_before_terminal"] for e in group)
    contrasts = [dict(zip(view["contrasts"]["columns"], values, strict=True)) for values in view["contrasts"]["rows"]]
    failure_strata = {e["stratum"] for e in episodes.values() if e["primary_reason"]!="success"}
    assert {row["stratum"] for row in contrasts} >= failure_strata
    for row in contrasts:
        for side, reason in (("failed", row["reason"]), ("success", "success")):
            group = [e for e in episodes.values() if (e["cell"], e["run"], e["primary_reason"]) == (row["cell"], row["run_ordinal"], reason)]
            counts = dict(zip(view["contrasts"]["side_columns"], row[side], strict=True))
            assert counts["n"] == len(group)
            assert counts["up_z_negative_n"] == sum(oracle[(e["run"], e["slot"])]["up_negative"] for e in group)
            assert counts["max_x_retreat_m"] == max((oracle[(e["run"], e["slot"])]["retreat"] for e in group), default=None)
            for axis in ("roll_rad_range", "pitch_rad_range"):
                ranges = [oracle[(e["run"], e["slot"])][axis] for e in group]
                assert counts[axis] == ([min(r[0] for r in ranges), max(r[1] for r in ranges)] if ranges else None)
    by_id = {e["episode_id"]: e for e in episodes.values()}
    for row in view["transcripts"]:
        source = by_id[row["episode_id"]]
        assert row["root_motion"] == source["root_motion"]
        for event in row["events"]:
            alias = {"loaded_point_motion_body": "slip_body", "loaded_point_motion_any": "slip_any"}.get(event["code"], event["code"])
            assert {**event, "code": alias} in source["events"]
    # Original outcomes and old four condition counts remain unchanged. Old
    # per-body interval confirmation may contain the already fixed v1 duration bug.
    old_by_id = {e["episode_id"]: e for e in old_full["episodes"]}
    for key, row in by_id.items():
        old = old_by_id[key]
        assert (row["primary_reason"], row["steps"]) == (old["primary_reason"], old["steps"])
        assert {k:row["before_terminal"][k] for k in old["before_terminal"]} == old["before_terminal"]
    prompt = read(PROMPT)
    legacy_view = prompt["public_feedback_summary"]["behavior"]["profile"]
    assert project_stair_behavior(legacy_view) == legacy_view
    old_v2 = read(HERE / "results/stair_projection_v2_real_view.json")
    assert project_stair_behavior(old_v2) == old_v2
    temporary = deepcopy(prompt)
    temporary["public_feedback_summary"]["behavior"]["profile"] = view
    old_frame = temporary.pop("authoring_context_frame")
    temporary["authoring_context_frame"] = build_authoring_context_frame(frame_kind=old_frame["frame_kind"],
        section_values=temporary, section_order=PROMPT_PAYLOAD_SECTION_ORDER, omitted_sections=old_frame["omitted_sections"])
    temporary["authoring_context_frame"] = with_final_prompt_budget(temporary["authoring_context_frame"], prompt_payload=temporary)
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model="offline-only", output_dir=HERE / "results/unused-provider-output"))
    request = provider._chat_completion_request(temporary)
    text = request["messages"][1]["content"]
    delivered = json.loads(text); budget = delivered["authoring_context_frame"]["budget_metadata"]
    validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=delivered)
    assert budget["actual_prompt_payload_bytes"] == len(text.encode())
    assert frozen == {path:sha(Path(path)) for path in frozen}
    write(HERE / "results/stair_motion_v3_view.json", view)
    report = {"status":"PASS", "scope":"old development numeric pool; not independent semantics/value acceptance",
        "source_hashes": frozen, "full_sha256":sha(FULL), "script_sha256":sha(Path(__file__)), "episodes":864,
        "independent_low_motion_spans_checked": checked_spans, "low_motion_before_terminal_episodes": sensitivity["0.05"],
        "speed_threshold_sensitivity_episode_counts":sensitivity, "thresholds_unchanged_after_check":True,
        "view_bytes":size(view), "view_schema":view["schema_version"], "stratum_totals":totals,
        "protected_failure_strata":sorted(failure_strata), "retained_contrasts":contrasts,
        "representatives":view["transcripts"], "profile_reason_rows_omitted":view["profile"]["omitted_group_rows"],
        "size_probe":{"user_json_bytes":len(text.encode()), "http_request_json_bytes":size(request),
            "task_sections_bytes":budget["provider_task_sections"]["serialized_bytes"],
            "source_public_result_hash_not_rebound":True, "never_delivered":True},
        "old_public_v1_v2_idempotent":True, "old_outcomes_and_four_condition_counts_unchanged":True,
        "api_calls":0,"gpu_jobs":0,"seconds":time.perf_counter()-started}
    write(HERE / "results/stair_motion_v2_validation.json",report)
    print(json.dumps({k:report[k] for k in ("status","episodes","view_bytes","independent_low_motion_spans_checked","speed_threshold_sensitivity_episode_counts","seconds")}))


if __name__=="__main__":
    main()
