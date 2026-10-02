#!/usr/bin/env python3
"""Independent offline P1a acceptance against byte-frozen P0 facts.

Only this verifier reads the reference ledger. No production imports, raw-array
recomputation, source selection changes, or expected-value substitution occur.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NECESSARY_SHA = "ce3a6115793eb4e182f3bb5641f163dc9d7858e3f4682b9ee99806ab637d7c74"
FULL_SHA = "661c6782dc198d8367dc20c6b42b4a2d4d30b9d6d333dcebbf72d4e60528f009"
SUPPORT = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
MISSING = object()
MAPPING_NOTES = [
    "Lookup uses source manifest SHA256 + run + environment, never display names.",
    "Canonical robot_0.support/body entity IDs map to frozen role names after exact entity-order validation.",
    "Half-open production step intervals map to inclusive frozen last_step by subtracting one.",
    "Run cell duration, observed span and recorder boundary brackets remain separate; physical duration is null only when explicitly unknown.",
    "Summary float rounding is accepted only within each frozen fact's absolute/relative tolerance; integers, booleans and indices are exact.",
    "Frozen summary tail rows are selected by step from the longer production tail; full ledger checks every frozen tail row.",
    "Derived summaries use emitted counts/intervals only (occupancy, means, extrema, sign tests); no raw-array calculation fills missing product values.",
    "Sparse outcome counters may omit zero classes; cohort denominator and every episode count are checked independently.",
    "Frozen counterexample check=confirmed_by_raw_operands is an oracle receipt: compare its boolean proposition against emitted numeric/temporal records, without claiming new raw verification.",
    "Frozen low-speed threshold_source prose ('P0 development descriptive threshold, not task success/failure rule') maps explicitly to measurement_config.low_speed_and_progress_threshold_source=development_calibration; the 0.05 scalar and M04 XY/XYZ definition are checked independently, not inferred from threshold magnitude.",
    "Full-ledger missing_channels/existing_temporal_channels are array-inventory claims, not omitted numeric operations. Product limitations and evidence refs express most corresponding concepts but do not serialize those exact complete inventories; they remain explicit unsupported fields rather than inferred from absent records.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_constant)


def load_frozen(root: Path) -> tuple[dict, dict]:
    necessary_path = root / "PLAN/feedback_sys/NEW_SYS/P0/necessary_facts.json"
    full_path = root / "outputs/trajectory_p0/reference/full_reference_facts.json"
    for path, expected in ((necessary_path, NECESSARY_SHA), (full_path, FULL_SHA)):
        if sha256(path) != expected:
            raise ValueError(f"frozen reference SHA mismatch: {path}")
    necessary, full = read_json(necessary_path), read_json(full_path)
    if len(necessary["episodes"]) != 18 or sum(len(e["facts"]) for e in necessary["episodes"]) != 246:
        raise ValueError("frozen necessary fact inventory differs from 18 episodes / 246 facts")
    return necessary, full


def compare_value(expected: Any, actual: Any, tolerance: dict, path: str = "value") -> tuple[list, int]:
    """Return all differences, including accepted float roundoff and extensions."""
    if actual is MISSING:
        return [{"path": path, "status": "unsupported", "reason": "required output field absent", "expected": expected}], 0
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [{"path": path, "status": "fail", "reason": "object type differs", "expected": expected, "actual": actual}], 0
        changes, fields = [], 0
        for key, value in expected.items():
            delta, count = compare_value(value, actual.get(key, MISSING), tolerance, f"{path}.{key}")
            changes.extend(delta)
            fields += count
        for key in sorted(actual.keys() - expected.keys()):
            changes.append({"path": f"{path}.{key}", "status": "extension", "reason": "additional output field", "actual": actual[key]})
        return changes, fields
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(expected) != len(actual):
            return [{"path": path, "status": "fail", "reason": "array type or length differs", "expected": expected, "actual": actual}], 0
        changes, fields = [], 0
        for index, (lhs, rhs) in enumerate(zip(expected, actual, strict=True)):
            delta, count = compare_value(lhs, rhs, tolerance, f"{path}[{index}]")
            changes.extend(delta)
            fields += count
        return changes, fields
    if type(expected) is float:
        numeric = type(actual) in (float, int) and math.isfinite(actual)
        passed = numeric and math.isclose(expected, actual, rel_tol=tolerance["relative"], abs_tol=tolerance["absolute"])
        if passed and expected == actual:
            return [], 1
        return [{"path": path, "status": "accepted_roundoff" if passed else "fail",
                 "reason": "within frozen numeric tolerance" if passed else "numeric value outside frozen tolerance or wrong type",
                 "expected": expected, "actual": actual}], int(passed)
    passed = type(expected) is type(actual) and expected == actual
    return ([] if passed else [{"path": path, "status": "fail", "reason": "exact value/type differs",
                               "expected": expected, "actual": actual}]), int(passed)


def _identity(header: dict) -> tuple[str, int, int]:
    if (type(header["run_index"]) is not int or header["run_index"] < 1
            or type(header["env_index"]) is not int or header["env_index"] < 0):
        raise ValueError("episode run/environment indices must be exact integers")
    return header["source"]["manifest_sha256"], header["run_index"], header["env_index"]


def load_output(directory: Path, wanted: set[tuple]) -> tuple[dict, dict]:
    manifest = read_json(directory / "manifest.json")
    if manifest["schema_version"] != "trajectory.numeric_manifest.v1" or manifest["complete"] is not True:
        raise ValueError("incomplete or unsupported evidence manifest")
    if manifest["acceptance_split"] != "public_dev":
        raise ValueError("non-public output is not accepted")
    evidence = manifest["evidence"]
    if evidence["path"] != "evidence.jsonl":
        raise ValueError("unexpected evidence path")
    path = directory / evidence["path"]
    if sha256(path) != evidence["sha256"] or path.stat().st_size != evidence["bytes"]:
        raise ValueError("evidence bytes or SHA mismatch")
    selected, seen_episodes, counts = {}, set(), Counter()
    current = None
    current_ids: set[str] = set()
    row_count = 0
    with path.open(encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, 1):
            row = json.loads(line, parse_constant=_reject_constant)
            kind = row["record_type"]
            counts[kind] += 1
            if kind == "episode":
                current = row
                identity = _identity(row)
                if identity in seen_episodes:
                    raise ValueError(f"duplicate episode identity at line {lineno}")
                seen_episodes.add(identity)
                source = row["source"]
                if (source["manifest_sha256"] != manifest["source_manifest_ref"]["manifest_sha256"]
                        or source["source_id"] != manifest["source_id"]
                        or source["acceptance_split"] != "public_dev"
                        or not source["trial_id"] or not source["checkpoint_hash"]):
                    raise ValueError("episode source identity differs from evidence manifest")
                current_ids = set()
                row_count += row["coverage"]["valid_rows"]
                if identity in wanted:
                    selected[identity] = {"header": row, "records": []}
            else:
                if current is None or row["episode_id"] != current["episode_id"]:
                    raise ValueError(f"record outside its episode at line {lineno}")
                if row["record_id"] in current_ids:
                    raise ValueError(f"duplicate evidence record at line {lineno}")
                current_ids.add(row["record_id"])
                ref = row["evidence_ref"]
                if (ref["manifest_sha256"] != current["source"]["manifest_sha256"]
                        or ref["run_index"] != current["run_index"]
                        or any(c["env_index"] != current["env_index"] for c in ref["chunks"])):
                    raise ValueError(f"record source reference differs from episode at line {lineno}")
                identity = _identity(current)
                if identity in wanted:
                    selected[identity]["records"].append(row)
    if dict(counts) != manifest["record_counts"] or len(seen_episodes) != manifest["emitted_episodes"] or row_count != manifest["emitted_valid_rows"]:
        raise ValueError("evidence content counts differ from manifest")
    return manifest, selected


def run_value(row: dict | None, compact: bool) -> dict | None:
    if row is None:
        return None
    first, stop = row["steps_half_open"]
    if compact:
        return {"steps": [first, stop - 1], "n": row["sample_count"],
                "cell_s": row["sample_cell_duration_proxy_s"], "span_s": row["observed_span_s"],
                "left_censored": row["left_censored"], "right_censored": row["right_censored"],
                "onset_s": row["onset_range_s"], "end_s": row["end_range_s"]}
    value = {"first_step": first, "last_step": stop - 1, "sample_count": row["sample_count"],
             "marker_cell_duration_s": row["sample_cell_duration_proxy_s"],
             "observed_endpoint_span_s": row["observed_span_s"],
             "left_censored": row["left_censored"], "right_censored": row["right_censored"],
             "onset_boundary_s": row["onset_range_s"], "end_boundary_s": row["end_range_s"],
             "unknown_boundary": row["left_censored"] or row["right_censored"]}
    if row["boundary_clock"] == "recorder_readout":
        value["boundary_interval_semantics"] = "(previous sample, first sample] / (last sample, next sample]"
    if row["physical_continuity"] == "unknown":
        value["physical_continuous_duration_s"] = None
    return value


class EpisodeView:
    def __init__(self, output: dict):
        self.header, self.records = output["header"], output["records"]
        self.measurements = {r["key"]: r for r in self.records if r["record_type"] == "measurement"}
        self.used: set[str] = set()
        support = self.header["entities"]["support"]
        if support != [f"robot_0.support.{name}" for name in SUPPORT]:
            raise ValueError("frozen support role order does not match output")
        if self.header["entities"]["forbidden"] != ["robot_0.body.base"]:
            raise ValueError("frozen forbidden-body order does not match output")

    def measurement(self, key: str) -> dict:
        record = self.measurements[key]
        self.used.add(record["record_id"])
        return record["value"]

    def runs(self, key: str, state: str) -> list[dict]:
        rows = [r for r in self.records if r["key"] == key and r["record_type"] == "state_interval" and r["value"]["state"] == state]
        self.used.update(r["record_id"] for r in rows)
        return [r["value"] for r in rows]

    def events(self, key: str, after: bool) -> list[int]:
        rows = [r for r in self.records if r["key"] == key + ".transition" and r["record_type"] == "event" and r["value"]["after"] is after]
        self.used.update(r["record_id"] for r in rows)
        return [r["value"]["step"] for r in rows]

    def mapping(self, template: str, expected: dict, compact: bool) -> dict:
        self.used.clear()
        h = self.header
        coverage = h["coverage"]
        if template == "source_entity_clock":
            self.used.add(h["episode_id"])
            value = {"source_id": h["source"]["source_id"], "episode_id": h["display_name"],
                     "support_order": SUPPORT, "quat_order": h["source"]["coordinates"]["base_quat_w"],
                     "dt_s": coverage["dt_s"], "first_time_s": coverage["first_step"] * coverage["dt_s"],
                     "last_time_s": coverage["last_step"] * coverage["dt_s"], "valid_count": coverage["valid_rows"]}
            if h["source"]["coverage"]["sample_phase"] == "pre_reset_control_step_end":
                value["sample_phase"] = "post_step_pre_reset"
            if coverage["raw"] == "processed_all_supplied_valid_rows":
                value["invalid_padding_excluded"] = True
            return value
        if template == "archived_result_and_endpoint":
            v = self.measurement("task_endpoint")
            return {"archived_outcome": v["outcome"], "terminal_step": v["terminal_step"],
                    "terminal_time_s": v["terminal_time_s"], "terminal_npz_flags": v["terminal_flags"]}
        if template in ("root_progress", "required_return_counterexample"):
            v = self.measurement("root_progress")
            if v["progress_direction_world"] != [1.0, 0.0, 0.0]:
                raise ValueError("world-x reference cannot compare another progress direction")
            d = v["drawdown"]
            if template == "required_return_counterexample":
                t = self.measurement("task_endpoint")
                return {"peak_step": d["peak_step"], "trough_step": d["trough_step"], "return_step": d["return_step"],
                        "drawdown_m": d["max_drawdown_m"], "success": t["outcome"]["success"],
                        "check": d["peak_step"] < d["trough_step"] < d["return_step"] and d["max_drawdown_m"] > 0 and t["outcome"]["success"]}
            return {**{k: v[k] for k in ("net_world_xyz_m", "sampled_path_xyz_m", "sampled_path_xy_m")},
                    "positive_x_variation_m": v["positive_progress_variation_m"], "negative_x_variation_m": v["negative_progress_variation_m"],
                    "max_x_drawdown_m": d["max_drawdown_m"], "drawdown_peak_step": d["peak_step"],
                    "drawdown_trough_step": d["trough_step"], "first_recovered_peak_step": d["return_step"]}
        if template == "body_up":
            v = self.measurement("body_up")
            runs = self.runs("body_up_negative", "true")
            raw = v["raw_body_up_diagnostic"]
            diagnostic = {"first_up_z": raw["first"], "last_up_z": raw["last"], "minimum_up_z": raw["minimum"]}
            if "first_negative_step" in raw:
                diagnostic["first_negative_step"] = raw["first_negative_step"]
            return {"first_up_z": v["up_z"]["first"], "last_up_z": v["up_z"]["last"], "minimum_up_z": v["up_z"]["minimum"],
                    "minimum_step": v["up_z"]["minimum_step"], "first_negative_step": runs[0]["steps_half_open"][0] if runs else None,
                    "negative_sample_count": v["negative"]["true_count"], "negative_runs": [run_value(r, compact) for r in runs],
                    "max_quaternion_norm_error": v["max_quaternion_norm_error"],
                    "raw_vs_normalized_up_z_max_abs_delta": v["raw_vs_normalized_up_z_max_abs_delta"], "raw_body_up_diagnostic": diagnostic}
        if template.endswith(".markers"):
            entity = template.split(".")[0]
            key = f"robot_0.support.{entity}.marker"
            v = self.measurement(key)
            on, off = self.runs(key, "true"), self.runs(key, "false")
            onsets, offsets = self.events(key, True), self.events(key, False)
            return {"body_index": SUPPORT.index(entity), "on_count": v["true_count"], "off_count": v["false_count"],
                    "occupancy": v["fraction"], "observed_onset_steps": onsets, "observed_offset_steps": offsets,
                    "on_runs": [run_value(r, compact) for r in on], "off_runs": [run_value(r, compact) for r in off],
                    "longest_off_run": run_value(max(off, key=lambda r: r["sample_count"], default=None), compact),
                    "complete_off_run_count": sum(r["complete_run"] for r in off),
                    "observed_onset_count": len(onsets), "observed_offset_count": len(offsets),
                    "shortest_observed_off_run_samples": min((r["sample_count"] for r in off), default=None)}
        if template.endswith(".low_speed_off"):
            key = "robot_0.support." + template
            v = self.measurement(key)
            longest = v["longest"]
            self.runs(key, "true")
            if longest is not None:
                longest = {**{k: longest[k] for k in ("max_speed_xyz_mps", "median_speed_xyz_mps", "max_speed_xy_mps", "median_speed_xy_mps", "max_force_norm_n", "minimum_up_z", "maximum_up_z")},
                           "longest_run": run_value(longest["interval"], compact)}
            config = self.measurements[key]["measurement_config"]
            result = {"speed_threshold_mps": config["low_horizontal_speed_mps"], "longest": longest}
            if "low_speed_and_progress_threshold_source" in config:
                result["threshold_source"] = config["low_speed_and_progress_threshold_source"]
            if "definition" in config:
                result["predicate_definition"] = config["definition"]
            return result
        if template == "force_semantics":
            v = self.measurement("force_summary")
            count = v["joint_valid_count"]
            return {"sample_count": count, **{k: v[k] for k in ("sample_sum_norm_n", "sample_sum_signed_fz_n", "norm_share", "signed_fz_share")},
                    "mean_norm_n": [x / count for x in v["sample_sum_norm_n"]],
                    "mean_signed_fz_n": [x / count for x in v["sample_sum_signed_fz_n"]],
                    "norm_denominator_n": sum(v["sample_sum_norm_n"]), "signed_fz_denominator_n": sum(v["sample_sum_signed_fz_n"])}
        if template == "contact_combinations":
            v = self.measurement(template)
            return {"encoding_version": v["encoding_version"], "bit_order": [x.removeprefix("robot_0.support.") for x in v["bit_order"]],
                    "bit_weights": v["bit_weights"], "counts_0_to_15": v["counts"], "all_off_count": v["counts"][0],
                    "all_off_fraction": v["counts"][0] / v["valid_count"], "FL_FR_RR_only_count": v["counts"][11],
                    "FL_FR_RR_only_fraction": v["counts"][11] / v["valid_count"]}
        if template == "body_frame_anchors":
            rows = [self.measurement(f"robot_0.support.{s}.relative_position") for s in SUPPORT]
            first = [r["first_body_xyz_m"] for r in rows]
            value = {"first_body_xyz_m": first, "last_body_xyz_m": [r["last_body_xyz_m"] for r in rows],
                     "first_left_y_positive": first[0][1] > 0 and first[2][1] > 0,
                     "first_right_y_negative": first[1][1] < 0 and first[3][1] < 0,
                     "first_front_x_positive": first[0][0] > 0 and first[1][0] > 0,
                     "first_rear_x_negative": first[2][0] < 0 and first[3][0] < 0}
            if all("mean_body_xyz_m" in r for r in rows):
                value["mean_body_xyz_m"] = [r["mean_body_xyz_m"] for r in rows]
            return value
        if template == "short_terminal_events":
            key = "robot_0.body.base"
            current, history = self.measurement(key + ".current_marker"), self.measurement(key + ".history_diagnostic")
            v = self.measurement("task_endpoint")
            tail_steps = expected["tail_steps"]
            indices = [v["tail_steps"].index(step) for step in tail_steps]
            value = {"forbidden_body_order": ["base"], "current_forbidden_on_count": current["true_count"],
                     "current_forbidden_on_runs": [run_value(r, compact) for r in self.runs(key + ".current_marker", "true")],
                     "history_any_forbidden_on_count": history["true_count"],
                     "history_any_forbidden_on_runs": [run_value(r, compact) for r in self.runs(key + ".history_any_marker", "true")],
                     "tail_steps": [v["tail_steps"][i] for i in indices],
                     "tail_flags": {k: [values[i] for i in indices] for k, values in v["tail_flags"].items()}}
            for field in ("tail_support_norms_n", "tail_forbidden_norms_n", "tail_forbidden_history_norms_n"):
                if field in v:
                    value[field] = [v[field][i] for i in indices]
            return value
        if template == "history_and_missing_channels":
            rows = [self.measurement(f"robot_0.support.{s}.history_diagnostic") for s in SUPPORT]
            value = {"history_any_on_counts": [r["true_count"] for r in rows],
                     "history_any_on_current_off_counts": [r["history_any_on_current_off_count"] for r in rows],
                     "current_on_history_all_off_counts": [r["current_on_history_all_off_count"] for r in rows],
                     "history_length": h["source"]["contact_force_history"]["history_length"],
                     "history_full_physics_coverage": any(r["full_physics_coverage"] for r in rows)}
            for out, field in (("history_slot_timestamps", "slot_timestamps"), ("history_slot_validity", "slot_validity")):
                if all(r[field] == "not_recorded" for r in rows):
                    value[out] = "not recorded"
            return value
        if template == "required_inverted_counterexample":
            runs = self.runs("body_up_negative", "true")
            first = runs[0]
            front, relation_checks = [], []
            for s in SUPPORT[:2]:
                key = f"robot_0.support.{s}.low_speed_off"
                v = self.measurement(key)["longest"]["interval"]
                self.runs(key, "true")
                front.append({"entity": s, "run": run_value(v, compact)})
                relations = [r for r in self.records if r["key"] == "body_up_negative.onset_before." + key and r["record_type"] == "temporal_relation"]
                self.used.update(r["record_id"] for r in relations)
                # This frozen counterexample has one inverted run and one long
                # front-foot run; verify a relation targets that exact pair.
                targets = {r["record_id"] for r in self.records if r["record_type"] == "state_interval" and r["key"] == key and r["value"] == v}
                relation_checks.append(any(r["value"]["status"] == "true" and r["value"]["operand_record_ids"][1] in targets for r in relations)
                                       and v["observed_span_s"] > 5)
            return {"first_negative_step": first["steps_half_open"][0], "first_negative_time_s": first["first_observed_s"],
                    "front_low_speed_off_runs": front, "check": all(relation_checks)}
        raise KeyError(f"unmapped frozen template: {template}")


def cohort_mapping(manifest: dict, expected: dict) -> dict:
    c = manifest["cohort_numeric_summary"]
    entities = c["support_order"]
    if entities != [f"robot_0.support.{s}" for s in SUPPORT]:
        raise ValueError("cohort support order differs")
    groups = {}
    for name, v in c["groups"].items():
        count = v["joint_force_valid_rows"]
        groups[name] = {"episodes": v["episodes"], "valid_samples": v["valid_rows"],
            **{k: v[k] for k in ("on_counts", "sample_sum_norm_n", "sample_sum_signed_fz_n", "combination_counts")},
            "marker_occupancy": [v["marker_fractions"][e]["sample_cell_duration_weighted_fraction"] for e in entities],
            "norm_share": v["sample_cell_weighted_norm_share"], "signed_fz_share": v["sample_cell_weighted_signed_fz_share"],
            "mean_norm_n": [x / count for x in v["sample_sum_norm_n"]],
            "mean_signed_fz_n": [x / count for x in v["sample_sum_signed_fz_n"]]}
    return {"primary_outcome_counts": {k: c["outcome_counts"].get(k, 0) for k in set(expected["primary_outcome_counts"]) | set(c["outcome_counts"])},
            "support_order": SUPPORT, "groups": groups}


def result(fact_id: str, expected: dict, actual: dict, tolerance: dict, ids: list[str]) -> dict:
    changes, fields = compare_value(expected, actual, tolerance)
    for change in changes:
        if change["status"] != "unsupported":
            continue
        if change["path"] == "value.missing_channels":
            change["reason"] = (
                "Complete P0 absence inventory is not serialized. manifest.limitations names raw_action, "
                "joint_velocity, torque_and_saturation, actual_reward_components_and_gates, "
                "tangential_force_and_contact_objects, actual_mass_and_whole_body_momentum, parent_archive "
                "and deferred nominal_or_effective_joint_limit_binding. All-nonfoot-body-force absence "
                "is not explicit; only the bound forbidden entity base is listed. This field is an "
                "inventory assertion, not an unimplemented numeric measurement.")
        elif change["path"] == "value.existing_temporal_channels":
            change["reason"] = (
                "P0 sample-export array inventory is not serialized as a list. Thirteen numeric/flag "
                "channels are dependencies in evidence_ref.channels; valid/padding and sample clock "
                "are expressed in header/source coverage. P0 episode_length/step_index and derived "
                "source_chunk_index/source_chunk_row have clock/chunk/row locator counterparts, "
                "but those structured locators are not claimed as an exact source-array inventory.")
    statuses = {d["status"] for d in changes}
    status = "fail" if "fail" in statuses else "unsupported" if "unsupported" in statuses else "pass"
    return {"fact_id": fact_id, "status": status, "compared_scalar_fields": fields,
            "record_ids": sorted(ids), "tolerance": tolerance, "differences": changes,
            "actual_mapped_value": actual}


def verify(root: Path, directories: list[Path]) -> dict:
    necessary, full = load_frozen(root)
    wanted = {(e["raw_reference"]["manifest_sha256"], e["raw_reference"]["run_index"], e["raw_reference"]["env_index"]) for e in necessary["episodes"]}
    outputs, manifests = {}, {}
    for directory in directories:
        manifest, selected = load_output(directory, wanted)
        source = manifest["source_manifest_ref"]["manifest_sha256"]
        if source in manifests or outputs.keys() & selected.keys():
            raise ValueError("duplicate input source/episode")
        manifests[source] = manifest
        outputs.update(selected)
    reports = {}
    for label, ledger in (("necessary", necessary), ("full", full)):
        rows = []
        for episode in ledger["episodes"]:
            raw = episode["raw_reference"]
            identity = raw["manifest_sha256"], raw["run_index"], raw["env_index"]
            view = EpisodeView(outputs[identity]) if identity in outputs else None
            for fact in episode["facts"]:
                template = fact["template"] if label == "necessary" else fact["fact_id"].removeprefix(episode["episode_id"] + ".")
                expected = dict(fact["value"])
                if template.endswith(".low_speed_off") and "threshold_source" in expected:
                    if expected["threshold_source"] != "P0 development descriptive threshold, not task success/failure rule":
                        raise ValueError("unrecognized frozen threshold-source meaning")
                    expected["threshold_source"] = "development_calibration"
                    # The frozen M04 formula explicitly separates XY predicate
                    # from XYZ diagnostics; check that product declaration too.
                    expected["predicate_definition"] = "M04 XY predicate; XYZ diagnostics only"
                if template.startswith("required_"):
                    if expected["check"] != "confirmed_by_raw_operands":
                        raise ValueError("unrecognized frozen counterexample receipt")
                    expected["check"] = True
                tolerance = necessary["fact_templates"][template]["tolerance"] if label == "necessary" else fact["tolerance"]
                if view is None:
                    actual, ids = {}, []
                    unavailable = "selected source/run/environment is absent from supplied output"
                else:
                    unavailable = None
                    try:
                        actual = view.mapping(template, expected, label == "necessary")
                    except KeyError as error:
                        # A missing record/value is a failed coverage obligation,
                        # not permission to recompute it from the oracle or raw NPZ.
                        actual = {}
                        unavailable = f"required output record/value missing: {error}"
                    ids = list(view.used)
                # Report only the frozen projection; field renames/added output
                # metadata are documented in mapping notes, not false failures.
                actual = {k: v for k, v in actual.items() if k in expected}
                comparison = result(episode["episode_id"] + "." + template, expected, actual, tolerance, ids)
                if unavailable is not None:
                    comparison["unavailable_reason"] = unavailable
                rows.append(comparison)
        for fact in ledger["cohort_facts"]:
            manifest = manifests.get(fact["operands"]["manifest_sha256"])
            actual = cohort_mapping(manifest, fact["value"]) if manifest else {}
            ids = ["manifest:cohort_numeric_summary:" + fact["operands"]["manifest_sha256"]] if manifest else []
            rows.append(result(fact["fact_id"], fact["value"], actual, fact["tolerance"], ids))
        reports[label] = {"counts": dict(Counter(r["status"] for r in rows)), "fact_count": len(rows),
                          "compared_scalar_fields": sum(r["compared_scalar_fields"] for r in rows), "facts": rows}
    return {"schema_version": "trajectory.p1a.independent_comparison.v1", "reference_sha256": {"necessary": NECESSARY_SHA, "full": FULL_SHA},
            "output_sources": [{"source_id": m["source_id"], "source_manifest_sha256": s, "evidence_sha256": m["evidence"]["sha256"],
                                "episodes": m["emitted_episodes"], "valid_rows": m["emitted_valid_rows"]} for s, m in manifests.items()],
            "mapping_notes": MAPPING_NOTES, "reports": reports,
            "pass_criterion": "All 247 frozen necessary facts pass; no full-ledger numeric/temporal mismatch. Full-ledger unsupported fields remain listed and are not claimed as validated.",
            "full_ledger_coverage_complete": reports["full"]["counts"] == {"pass": 309},
            "passed": reports["necessary"]["counts"] == {"pass": 247} and reports["full"]["counts"].get("fail", 0) == 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, action="append", required=True, help="P1a output directory; repeat for R4 and legacy")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = verify(args.project_root, args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"passed": report["passed"], **{label: {k: v for k, v in value.items() if k != "facts"} for label, value in report["reports"].items()}}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
