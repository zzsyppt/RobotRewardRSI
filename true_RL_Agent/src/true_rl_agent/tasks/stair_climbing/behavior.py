"""Stair-owned deterministic behavior views over sealed public-dev evidence.

Intervals describe measured conditions, never collision causes or reward fixes.
The full offline view and the bounded author projection use the same derivation.
"""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
import json
import math
from typing import Any, Mapping

from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from true_rl_agent.tasks.stair_climbing.numeric_capture import recompute_stair_numeric_evidence
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
from true_rl_agent.tasks.stair_climbing.transition_decoder import _roll_pitch_from_wxyz
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile

LEGACY_FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v1"
FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v2"
LEGACY_PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v1"
PREVIOUS_PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v2"
PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v3"
LEGACY_RULE_VERSION = "stair.observed_motion.v1"
RULE_VERSION = "stair.observed_motion.v2"
MAX_PUBLIC_BYTES = 8 * 1024
WINDOW_S = 1.0
LOW_PROGRESS_WINDOW_S = 0.5
LOW_PROGRESS_DISTANCE_M = 0.02
LIFT_SPEED_MPS = 0.05
UNLOADED_LOW_SPEED_MPS = 0.05
UNLOADED_LOW_MOTION_S = 0.5
_PROFILE_EVENTS = ("tilt", "slip_any", "low_clearance_any", "low_progress")
_PUBLIC_KEYS = {"schema_version", "domain", "semantics", "profile", "contrasts", "transcripts", "omitted"}
_DOMAIN_KEYS = {"checkpoint_hash", "task_protocol_hash", "suite_hash", "projection_hash", "source_hash",
                "cohort_hash", "episode_keys_hash", "reward_input_schema_hash", "rule_version", "archive_hash", "numeric_profile", "history_status"}


def _size(value):
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _rounded(value):
    return round(float(value), 6)


def _norm(vector):
    return math.sqrt(sum(float(value) ** 2 for value in vector))


PROGRESS_PROCESS_VERSION = "stair.root_progress.v1"


def _progress_process(samples, events):
    """Measured drawdown/re-attainment episodes, without an action classifier.

    A return to the prior running maximum closes an excursion. The latest
    equal maximum starts the next one; a terminal tail remains right-censored.
    All positive excursions survive here, including small oscillations.
    """
    spans, peak, trough = [], 0, None
    for index in range(1, len(samples)):
        if samples[index][0] >= samples[peak][0]:
            if trough is not None:
                spans.append((peak, trough, index, index))
            peak, trough = index, None
        elif trough is None or samples[index][0] < samples[trough][0]:
            trough = index
    if trough is not None:
        spans.append((peak, trough, None, len(samples) - 1))
    rows = []
    for peak, trough, recovered, end in spans:
        extrema = {}
        for column, name in enumerate(("roll_rad", "pitch_rad", "up_z"), 1):
            low = min(range(peak, end + 1), key=lambda i: samples[i][column])
            high = max(range(peak, end + 1), key=lambda i: samples[i][column])
            extrema[name] = [_rounded(samples[low][column]), low + 1,
                             _rounded(samples[high][column]), high + 1]
        rows.append({"peak_step": peak + 1, "trough_step": trough + 1,
            "recovery_step": None if recovered is None else recovered + 1,
            "observed_end_step": end + 1,
            "peak_x_m": _rounded(samples[peak][0]), "trough_x_m": _rounded(samples[trough][0]),
            "end_x_m": _rounded(samples[end][0]),
            "drop_m": _rounded(samples[peak][0] - samples[trough][0]),
            "attitude_extrema": extrema,
            "overlapping_root_event_indices": [i for i, e in enumerate(events)
                if e["code"] in {"tilt", "low_progress"}
                and e["start"] <= end + 1 and e["end"] >= peak + 1]})
    return {"version": PROGRESS_PROCESS_VERSION, "excursions": rows}


class _Episode:
    def __init__(self, cell, run, index, signals, names, has_history):
        self.cell, self.run, self.index = cell, run, index
        self.signals, self.names, self.has_history = signals, names, has_history
        self.dt = cell.spec.control_dt_s
        self.windows = deque()
        self.open = {}
        self.events = []
        self.stats = {}
        self.anchor = {"status": "not_reached", "step": None}
        self.previous_progress = None
        self.ranges = {"roll_rad": [math.inf, -math.inf], "pitch_rad": [math.inf, -math.inf],
                       "progress_m": [math.inf, -math.inf]}
        self.steps = 0
        self.body_ranges = {name: [math.inf, 0., 0.] for name in names}
        self.slow_measurements = {}
        self.root_motion = {"first": None, "last": None, "extrema": {},
                            "max_x_retreat": [1, 1, 0.], "first_up_z_negative_step": None}
        self.x_peak = None
        self.progress_samples = []
        self.required = {"tilt": signals["base_orientation_instability"]["consecutive_control_steps"],
                         "slip_any": signals["slip"]["consecutive_control_steps"],
                         "low_clearance_any": signals["insufficient_clearance"]["consecutive_control_steps"],
                         "low_progress": 1}

    def _condition(self, code, body, active, step, required=1):
        key = (code, body)
        stat = self.stats.setdefault(key, {"first": None, "confirmed": None, "count": 0,
                                          "longest": 0, "last": None, "streak": 0})
        stat["streak"] = stat["streak"] + 1 if active else 0
        if active:
            stat["first"] = step if stat["first"] is None else stat["first"]
            stat["last"] = step
            stat["count"] += 1
            stat["longest"] = max(stat["longest"], stat["streak"])
            if stat["streak"] >= required and stat["confirmed"] is None:
                stat["confirmed"] = step
            if key not in self.open:
                self.open[key] = (step, required)
        elif key in self.open:
            self._close(key, step - 1)

    def _close(self, key, end):
        start, required = self.open.pop(key)
        event = {"code": key[0], "body": key[1], "start": start, "end": end,
                 "confirmed": start + required - 1 if end-start+1 >= required else None}
        if key[0] == "unloaded_low_motion":
            first_x, last_x, speed = self.slow_measurements.pop(key[1])
            event["measured"] = {"max_speed_mps": _rounded(speed), "root_x_gain_m": _rounded(last_x-first_x)}
        self.events.append(event)

    def observe(self, transition, raw, outcome):
        t, i, signals = transition, self.index, self.signals
        step = t.episode_step
        self.steps = step
        quaternion = raw["base_quat_w"][i].tolist()
        roll, pitch = _roll_pitch_from_wxyz(quaternion)
        up_z = 1. - 2. * (float(quaternion[1])**2 + float(quaternion[2])**2)
        sample = [float(t.progress_m), roll, pitch, up_z]
        self.progress_samples.append(sample)
        motion = self.root_motion
        if motion["first"] is None:
            motion["first"] = [_rounded(value) for value in sample]
        motion["last"] = [_rounded(value) for value in sample]
        for name, value in zip(("x_m", "roll_rad", "pitch_rad", "up_z"), sample, strict=True):
            bounds = motion["extrema"].setdefault(name, [value, step, value, step])
            if value < bounds[0]:
                bounds[:2] = [value, step]
            if value > bounds[2]:
                bounds[2:] = [value, step]
        if up_z < 0 and motion["first_up_z_negative_step"] is None:
            motion["first_up_z_negative_step"] = step
        if self.x_peak is None or t.progress_m > self.x_peak[1]:
            self.x_peak = (step, t.progress_m)
        retreat = self.x_peak[1] - t.progress_m
        if retreat > motion["max_x_retreat"][2]:
            motion["max_x_retreat"] = [self.x_peak[0], step, retreat]
        for name, value in (("roll_rad", roll), ("pitch_rad", pitch), ("progress_m", t.progress_m)):
            self.ranges[name][0] = min(self.ranges[name][0], value)
            self.ranges[name][1] = max(self.ranges[name][1], value)
        if self.anchor["status"] == "not_reached" and t.progress_m >= self.cell.start_platform_length_m/2:
            self.anchor = {"status": "already_beyond_first_sample" if self.previous_progress is None else "crossed",
                           "step": None if self.previous_progress is None else step}
        self.previous_progress = t.progress_m
        self.windows.append((step, t.progress_m))
        window_steps = int(math.ceil(LOW_PROGRESS_WINDOW_S / self.dt))
        while self.windows and self.windows[0][0] < step-window_steps:
            self.windows.popleft()
        low_progress = (step-self.windows[0][0] == window_steps
                        and max(value for _, value in self.windows)-self.windows[0][1] < LOW_PROGRESS_DISTANCE_M)
        slip = [contact and speed > signals["slip"]["speed_threshold"]
                for contact, speed in zip(t.support_contacts, t.support_planar_speeds_mps, strict=True)]
        low = [not contact and clearance < signals["insufficient_clearance"]["threshold"]
               for contact, clearance in zip(t.support_contacts, t.support_surface_clearances_m, strict=True)]
        conditions = {"tilt": max(abs(roll), abs(pitch)) > signals["base_orientation_instability"]["threshold"],
                      "slip_any": any(slip), "low_clearance_any": any(low), "low_progress": low_progress}
        for code, active in conditions.items():
            self._condition(code, "root" if code in {"tilt", "low_progress"} else "any_support",
                            active, step, self.required[code])
        self._condition("forbidden_manager", "forbidden_role", t.forbidden_contact, step)
        self._condition("no_instant_support", "all_support", not any(t.support_contacts), step)
        for j, body in enumerate(self.names):
            self.body_ranges[body][0] = min(self.body_ranges[body][0], t.support_surface_clearances_m[j])
            self.body_ranges[body][1] = max(self.body_ranges[body][1], t.support_planar_speeds_mps[j])
            self.body_ranges[body][2] = max(self.body_ranges[body][2], abs(float(raw["support_lin_vel_w"][i, j, 2])))
            self._condition("contact_instant", body, t.support_contacts[j], step)
            self._condition("slip_body", body, slip[j], step, self.required["slip_any"])
            self._condition("low_clearance_body", body, low[j], step, self.required["low_clearance_any"])
            vz = float(raw["support_lin_vel_w"][i, j, 2])
            speed = _norm(raw["support_lin_vel_w"][i, j].tolist())
            slow = not t.support_contacts[j] and speed <= UNLOADED_LOW_SPEED_MPS
            if slow:
                first_x, _, prior_speed = self.slow_measurements.get(body, (t.progress_m, t.progress_m, 0.))
                self.slow_measurements[body] = (first_x, t.progress_m, max(prior_speed, speed))
            self._condition("unloaded_low_motion", body, slow, step,
                            int(math.ceil(UNLOADED_LOW_MOTION_S/self.dt)))
            self._condition("unloaded_rising", body, not t.support_contacts[j] and vz > LIFT_SPEED_MPS, step, 3)
            self._condition("unloaded_lowering", body, not t.support_contacts[j] and vz < -LIFT_SPEED_MPS, step, 3)
            if self.has_history:
                history = raw["support_contact_forces_w_history"][i, :, j].tolist()
                contact = max(_norm(vector) for vector in history) > signals["slip"]["contact_force_threshold"]
                self._condition("contact_history", body, contact, step)
        if self.has_history:
            import torch
            history = raw["forbidden_contact_forces_w_history"][i]
            predicate = bool((torch.linalg.vector_norm(history, dim=-1)
                              > signals["forbidden_contact"]["force_threshold_n"]).any())
            self._condition("forbidden_history", "forbidden_role", predicate, step)
        if outcome is None:
            return None
        for key in list(self.open):
            self._close(key, step)
        self.events.sort(key=lambda event: (event["start"], event["body"], event["code"]))
        before = {}
        window_present = {}
        window_first = max(1, step-int(math.ceil(WINDOW_S/self.dt))+1)
        for code in _PROFILE_EVENTS:
            stat = self.stats[(code, "root" if code in {"tilt", "low_progress"} else "any_support")]
            before[code] = stat["confirmed"] is not None and stat["confirmed"] < step
            window_present[code] = any(event["code"] == code and event["start"] < step
                                       and event["end"] >= window_first for event in self.events)
        before["unloaded_low_motion"] = any(event["code"] == "unloaded_low_motion"
            and event["confirmed"] is not None and event["confirmed"] < step for event in self.events)
        motion["extrema"] = {name: [_rounded(row[0]), row[1], _rounded(row[2]), row[3]]
                             for name, row in motion["extrema"].items()}
        motion["max_x_retreat"][2] = _rounded(motion["max_x_retreat"][2])
        return {"episode_id": f"r{self.run}.e{i}", "run": self.run, "slot": i, "cell": self.cell.cell_id,
                "stratum": self.cell.spec.stratum_id, "primary_reason": outcome.primary_reason,
                "steps": step, "dt_s": self.dt, "anchor": self.anchor,
                "ranges": {key: [_rounded(x) for x in values] for key, values in self.ranges.items()},
                "body_extrema": [[name, *[_rounded(x) for x in values]] for name, values in self.body_ranges.items()],
                "root_motion": motion,
                "progress_process": _progress_process(self.progress_samples, self.events),
                "events": self.events, "before_terminal": before, "window_present": window_present,
                "coverage": {"first_step": 1, "last_step": step, "terminal_included": True,
                             "window_first_step": window_first}}


def build_stair_behavior(reference: Mapping[str, Any], *, expected_identity: Mapping[str, Any] | None = None) -> dict:
    """Full independently recomputed view; no raw paths/seeds enter its public projection."""
    archive = load_numeric_evidence_manifest(reference)
    metadata, identity = archive["metadata"], archive["identity"]
    if expected_identity is not None and any(identity.get(k) != v for k, v in expected_identity.items()):
        raise ValueError("Stair behavior archive identity mismatch")
    if identity.get("splits") != ["public_dev"]:
        raise ValueError("Stair behavior requires bound public_dev evidence")
    manifest = ResolvedCellManifestV1.from_mapping(metadata["resolved_cell_manifest"])
    projection = stair_runtime_projection_from_manifest(manifest)
    training = StairUnifiedTrainingProfile.from_mapping(metadata["training_profile"])
    suite = StairUnifiedEvaluationSuite.from_mapping(metadata["evaluation_suite"], training_profile=training)
    cells = suite.private_cell_assignment("public_dev")
    signals = metadata["semantics"]["episode_outcomes"]["signals"]
    has_history = metadata.get("contact_force_history", {}).get("status") == "recorded"
    episodes, active = [], {}
    def observe(run, transition, raw, outcome):
        key = (run, transition.env_index)
        state = active.setdefault(key, None)
        if state is None:
            state = active[key] = _Episode(cells[transition.env_index], run, transition.env_index, signals,
                                         projection.support_body_names, has_history)
        completed = state.observe(transition, raw, outcome)
        if completed is not None:
            episodes.append(completed)
            del active[key]
    result = recompute_stair_numeric_evidence(reference, observe_transition=observe)
    if active or len(episodes) != result["episode_count"]:
        raise ValueError("Stair behavior episode coverage mismatch")
    episodes.sort(key=lambda episode: (episode["run"], episode["slot"]))
    source = identity.get("environment_source_preflight", {}).get("source_manifest_hash")
    backend_source = identity.get("backend_execution_environment_verification", {}).get("actual_manifest_hash")
    domain = {"checkpoint_hash": identity["checkpoint_hash"],
              "task_protocol_hash": metadata["semantics"]["task_protocol_hash"],
              "suite_hash": canonical_json_hash(metadata["evaluation_suite"]),
              "projection_hash": projection.projection_hash,
              "reward_input_schema_hash": projection.reward_input_schema_hash_vnext,
              "source_hash": canonical_json_hash([source, backend_source]) if source and backend_source else None,
              "cohort_hash": canonical_json_hash({"namespace": identity.get("seed_namespace"),
                  "runs": [[seed["run_index"], seed["runner_seed"], seed["split"]] for seed in metadata["seeds"]]}),
              "episode_keys_hash": canonical_json_hash([[e["run"], e["slot"], e["cell"]] for e in episodes]),
              "rule_version": RULE_VERSION, "archive_hash": reference["manifest_sha256"],
              "numeric_profile": metadata["profile"], "history_status": "recorded" if has_history else "not_recorded"}
    semantics = {
        "kinematic_body": projection.kinematic_base_source, "support_bodies": list(projection.support_body_names),
        "forbidden_bodies": list(projection.forbidden_contact_body_names),
        "coordinates": "positions relative to terrain origin; velocities and forces world; quaternion wxyz; roll/pitch signed Euler rad",
        "time": "one-based control steps; edge lies after previous sample and by current sample; conditions starting at 1 are left censored; terminal included",
        "contact": "instant norm > threshold; history max norm is a separate buffered predicate, without slot timestamps",
        "body_extrema_columns": ["body", "min_reference_surface_clearance_m", "max_planar_speed_mps", "max_abs_world_vz_mps"],
        "thresholds": {"contact_N": signals["slip"]["contact_force_threshold"],
                       "forbidden_N": signals["forbidden_contact"]["force_threshold_n"], "slip_mps": signals["slip"]["speed_threshold"],
                       "clearance_m": signals["insufficient_clearance"]["threshold"], "tilt_rad": signals["base_orientation_instability"]["threshold"],
                       "slip_steps": signals["slip"]["consecutive_control_steps"], "clearance_steps": signals["insufficient_clearance"]["consecutive_control_steps"],
                       "tilt_steps": signals["base_orientation_instability"]["consecutive_control_steps"],
                       "low_progress_window_s": LOW_PROGRESS_WINDOW_S, "low_progress_gain_m": LOW_PROGRESS_DISTANCE_M,
                       "unloaded_vertical_speed_mps": LIFT_SPEED_MPS, "unloaded_motion_steps": 3},
        "rules": {"tilt": "root abs roll or pitch above threshold", "slip_any": "any instant-loaded support moves fast; global consecutive streak",
                  "low_clearance_any": "any instant-unloaded reference point below actual C1/C2 surface clearance threshold; global streak",
                  "slip_body": "same slip condition, consecutive for this body only", "low_clearance_body": "same clearance condition, consecutive for this body only",
                  "low_progress": "maximum root x in preceding window minus window start x below gain threshold",
                  "unloaded_rising": "instant-unloaded reference point world vz positive above threshold",
                  "unloaded_lowering": "instant-unloaded reference point world vz negative beyond threshold",
                  "contact_instant": "this support instantaneous force norm above contact threshold",
                  "contact_history": "this support maximum buffered force norm above contact threshold; not an exact touchdown time",
                  "no_instant_support": "all support instantaneous contact predicates false",
                  "forbidden_manager": "existing task forbidden termination term active",
                  "forbidden_history": "buffered forbidden force predicate, independently retained; manager term determines termination"},
        "limits": "reference point is not toe/sole center; root is not necessarily forbidden body; no contact surface or causality; first-arrival closure censors later behavior",
        "actionability": {"tilt": ["instant_only", "roll", "pitch"],
            "slip_body": ["instant_only", "support_contact_force", "support_lin_vel"],
            "slip_any": ["instant_only", "any(slip_body predicate)"],
            "contact_history": ["available", "support_contact", "support_contact_force_history"],
            "unloaded_motion": ["instant_only", "support_contact_force", "support_lin_vel"],
            "streaks": ["diagnostic_only: global/body/vertical streaks need absent pose/velocity history; force buffer insufficient"],
            "forbidden_manager": ["available", "forbidden_contact"],
            "forbidden_history": ["buffer_not_in_ABI; instant_proxy_only", "forbidden_contact_force"],
            "low_clearance_body": ["exact_surface_not_in_ABI", "height_scan", "support_pos_relative_to_terrain_origin"],
            "contact_edge": (["available", "support_first_contact", "support_last_air_time"] if projection.support_transition_inputs else ["not_in_reward_ABI"]),
            "low_progress": ["window_not_in_ABI; instant_proxies", "forward_progress_m", "base_lin_vel_yaw"],
            "authority": ["pure reward; no new state/input or gate/terrain/PPO"]}}
    semantics["thresholds"].update(unloaded_low_speed_mps=UNLOADED_LOW_SPEED_MPS,
                                   unloaded_low_motion_s=UNLOADED_LOW_MOTION_S)
    semantics["rules"]["unloaded_low_motion"] = (
        "one support: instant force norm<=contact_N AND world 3D point speed<=unloaded_low_speed_mps; "
        "confirmed after ceil(unloaded_low_motion_s/dt_s) samples")
    semantics["actionability"]["unloaded_low_motion"] = [
        "instant_proxy_only; streak not in ABI", "support_contact_force", "support_lin_vel"]
    semantics["root_motion"] = (
        "first/last=[x_m,roll_rad,pitch_rad,up_z]; extrema values=[min,step,max,step], earliest ties; "
        "max_x_retreat=[prior_peak_step,later_trough_step,drop_m]; up_z=1-2(qx^2+qy^2), body up axis world z; "
        "Euler angles wrap, not accumulated rotation. Negative up_z is an orientation observation, not a contact/cause. "
        "unloaded_low_motion measured.max_speed_mps and root_x_gain_m cover that interval; not physical separation or whole-body stillness.")
    semantics["progress_process"] = (
        "stair.root_progress.v1: all positive drawdowns from the running root-x maximum, no amplitude threshold. "
        "Peak is the latest equal maximum; trough is the earliest minimum before the first sample x>=peak_x. "
        "recovery_step means only re-attaining that x, not balance recovery, monotonic motion or a new attempt. "
        "Null recovery_step means not re-attained through observed_end_step=terminal, not permanent failure. "
        "attitude_extrema cover peak..observed_end inclusive, signed Euler angles wrap; overlapping_root_event_indices "
        "refer to the complete events array, preserve unconfirmed intervals and establish overlap, not causality. "
        "Excursion count includes small oscillations; six-decimal display rounding does not define the segmentation.")
    return {"schema_version": FULL_SCHEMA, "domain": domain, "semantics": semantics,
            "episodes": episodes, "summary_hash": canonical_json_hash(result["summary"])}


# Presentation aliases only: the full archive and its detector rules do not change.
_PUBLIC_CODES = {"slip_any": "loaded_point_motion_any", "slip_body": "loaded_point_motion_body"}
_PUBLIC_EVENTS = tuple(_PUBLIC_CODES.get(code, code) for code in _PROFILE_EVENTS)
_WINDOW_EVENTS = tuple(code for code in _PROFILE_EVENTS if code != "slip_any")


def _profile(episodes, *, enhanced=False):
    event_codes = (*_PROFILE_EVENTS, "unloaded_low_motion") if enhanced else _PROFILE_EVENTS
    window_codes = () if enhanced else _WINDOW_EVENTS
    public_events = tuple(_PUBLIC_CODES.get(code, code) for code in event_codes)
    groups = defaultdict(list)
    for episode in episodes:
        groups[(episode["stratum"], episode["primary_reason"])].append(episode)
    profile = {"episode_count": len(episodes),
            "columns": ["stratum", "primary_reason", "n", "root_anchor_crossed", *public_events,
                        "min_steps", "max_steps", *["window_"+code for code in window_codes]],
            "event_count_scope": "stratum_totals covers all episodes; reason rows may be budget-omitted. Counts are episodes out of row n, conditions may overlap. Plain columns: confirmed before terminal; window columns: any presence in last 1s, no streak requirement. Short episodes have shorter exposure. Saturated window_slip_any omitted, not zero.",
            "rows": [[stratum, reason, len(group), sum(e["anchor"]["status"] == "crossed" for e in group),
                      *[sum(e["before_terminal"][code] for e in group) for code in event_codes],
                      min(e["steps"] for e in group), max(e["steps"] for e in group),
                      *[sum(e["window_present"][code] for e in group) for code in window_codes]]
                     for (stratum, reason), group in sorted(groups.items())]}
    total_columns = ["n", "success", "root_anchor_crossed", *public_events,
                     *["window_"+code for code in window_codes]]
    totals = defaultdict(lambda: dict.fromkeys(total_columns, 0))
    for values in profile["rows"]:
        row = dict(zip(profile["columns"], values, strict=True))
        row["success"] = row["n"] if row["primary_reason"] == "success" else 0
        for column in total_columns:
            totals[row["stratum"]][column] += row[column]
    profile["stratum_totals"] = {
        "columns": ["stratum", *total_columns],
        "rows": [[stratum, *[counts[key] for key in total_columns]] for stratum, counts in sorted(totals.items())]}
    profile.update(omitted_group_rows=0, omitted_group_episodes=0)
    return profile


def _failure_groups(episodes):
    failures, successes = defaultdict(list), defaultdict(list)
    for episode in episodes:
        cell_run = (episode["cell"], episode["run"])
        if episode["primary_reason"] == "success":
            successes[cell_run].append(episode)
        else:
            failures[(episode["primary_reason"], *cell_run)].append(episode)
    reason_n = defaultdict(int)
    for (reason, _, _), group in failures.items():
        reason_n[reason] += len(group)
    # Dominant outcome first; within it prefer a matched, well represented group.
    ordered = sorted(failures, key=lambda key: (-reason_n[key[0]], key[0],
        not bool(successes.get(key[1:])), -len(failures[key]), key[1], key[2]))
    return [(key, failures[key], successes.get(key[1:], [])) for key in ordered]


def _contrasts(episodes, *, enhanced=False):
    groups = _failure_groups(episodes)
    def counts(group):
        return {"n": len(group), "confirmed": {
            _PUBLIC_CODES.get(code, code): sum(e["before_terminal"][code] for e in group)
            for code in ((*_PROFILE_EVENTS, "unloaded_low_motion") if enhanced else _PROFILE_EVENTS)}}
    rows = [{"reason": key[0], "cell": key[1], "run_ordinal": key[2],
             "failed": counts(failed), "success": counts(success),
             "status": "matched" if success else "missing_same_cell_run_success"}
            for key, failed, success in groups]
    result = {"failure_group_count": len(groups),
            "matched_group_count": sum(bool(success) for _, _, success in groups),
            "missing_success_group_count": sum(not success for _, _, success in groups),
            "rows": rows,
            "scope": "Same policy/cell/run, each failure reason separate. confirmed values are episode counts out of that side n; overlap allowed. Association only, unequal episode exposure. n=0 success means missing, not a zero rate."}
    if enhanced:
        result["side_columns"] = ["n", *_PUBLIC_EVENTS, "unloaded_low_motion", "up_z_negative_n",
                                  "max_x_retreat_m", "roll_rad_range", "pitch_rad_range"]
        for row, (_, failed, success) in zip(rows, groups, strict=True):
            row["stratum"] = failed[0]["stratum"]
            for side, group in (("failed", failed), ("success", success)):
                record = row[side]
                row[side] = [record["n"], *record["confirmed"].values(),
                    sum(e["root_motion"]["first_up_z_negative_step"] is not None for e in group),
                    max((e["root_motion"]["max_x_retreat"][2] for e in group), default=None),
                    *[[min(e["root_motion"]["extrema"][axis][0] for e in group),
                       max(e["root_motion"]["extrema"][axis][2] for e in group)] if group else None
                      for axis in ("roll_rad", "pitch_rad")]]
        result["scope"] = (
            "Counts/n: confirmed before terminal; up_z_negative_n includes terminal. Retreat/ranges are group extrema, "
            "not typical or a shared sequence. Same cell/run; overlap/unequal exposure, not causal. n=0 success means missing, not a zero rate.")
    return result


def _event_priority(event, terminal_step):
    root_process = (event["body"] == "root" and event["code"] in {"tilt", "low_progress"}
                    and event["confirmed"] is not None)
    return (event["code"] == "forbidden_manager", root_process,
            event["confirmed"] is not None, event["end"] == terminal_step,
            event["end"] - event["start"] + 1, -event["start"], event["code"], event["body"])


def _process_core(events, terminal_step):
    """One formal end and one confirmed interval per root process condition."""
    core = []
    for code in ("forbidden_manager", "tilt", "low_progress"):
        matches = [event for event in events if event["code"] == code
                   and (code == "forbidden_manager"
                        or (event["body"] == "root" and event["confirmed"] is not None))]
        if matches:
            core.append(max(matches, key=lambda event: _event_priority(event, terminal_step)))
    slow = [event for event in events if event["code"] == "unloaded_low_motion"
            and event["confirmed"] is not None]
    if slow:
        # The overlap is an observed coexistence, never an explanation of low progress.
        progress = [event for event in events if event["code"] == "low_progress"]
        def slow_priority(event):
            overlap = sum(max(0, min(event["end"], other["end"])-max(event["start"], other["start"])+1)
                          for other in progress)
            return overlap, event["end"]-event["start"]+1, -event["start"], event["body"]
        core.append(max(slow, key=slow_priority))
    return core


def _representatives(episodes, *, enhanced=False):
    groups = _failure_groups(episodes)
    def typical(group):
        patterns = defaultdict(list)
        for episode in group:
            patterns[tuple(episode["before_terminal"][code] for code in ("tilt", "low_progress"))].append(episode)
        # A frequent measured conjunction, not a learned semantic class. Stable
        # ordering resolves ties; duration only ranks within the chosen group.
        pattern = min(patterns, key=lambda key: (-len(patterns[key]), key))
        if enhanced:
            group = patterns[pattern]
        ordered = sorted(group, key=lambda e: (e["steps"], e["run"], e["slot"]))
        chosen = ordered[(len(ordered)-1)//2]
        selection = {"tilt": pattern[0], "low_progress": pattern[1], "pattern_n": len(patterns[pattern]),
                     "group_n": sum(map(len, patterns.values()))}
        return chosen, selection
    if groups:
        _, failed, success = groups[0]
        selected = [typical(failed)]
        if success:
            selected.append(typical(success))
    else:
        selected = [typical(episodes)] if episodes else []
    result = []
    for episode, selection in selected:
        formal_spans = {(event["start"], event["end"]) for event in episode["events"]
                        if event["code"] == "forbidden_manager"}
        candidates = [event for event in episode["events"]
                      if event["code"] not in {"contact_instant", "contact_history"}
                      and not (event["code"] == "forbidden_history"
                               and (event["start"], event["end"]) in formal_spans)]
        core = _process_core(candidates, episode["steps"])
        optional = sorted((event for event in candidates if event not in core),
                          key=lambda event: _event_priority(event, episode["steps"]), reverse=True)
        chosen = core + optional[:8-len(core)]
        chosen = sorted(chosen, key=lambda event: (event["start"], event["end"], event["body"], event["code"]))
        transcript = {key: deepcopy(episode[key]) for key in
                       ("episode_id", "cell", "stratum", "primary_reason", "steps", "dt_s", "anchor", "ranges", "body_extrema", "coverage")}
        transcript.update({"events": deepcopy(chosen), "omitted_events": len(episode["events"])-len(chosen),
                         "omitted_body_extrema": 0,
                         "relationship": "index_example" if episode is selected[0][0] else "same_cell_run"})
        if enhanced:
            # All transcripts span observed step 1..steps including terminal.
            # The old last-1s coverage duplicates that span and an omitted view.
            for key in ("ranges", "body_extrema", "omitted_body_extrema", "coverage"):
                transcript.pop(key)
            transcript.update(root_motion=deepcopy(episode["root_motion"]), process_selection=selection)
        result.append(transcript)
    return result


def project_stair_behavior(value: Mapping[str, Any]) -> dict:
    """Closed, bounded, portable public view; legacy absence is handled by callers."""
    if value.get("schema_version") in {"true_rl_agent.stair.behavior.full.v3", "true_rl_agent.stair.behavior.public.v4"}:
        from .process_report import project_stair_process_report
        return project_stair_process_report(value)
    if value.get("schema_version") in {LEGACY_FULL_SCHEMA, FULL_SCHEMA}:
        enhanced = value["schema_version"] == FULL_SCHEMA
        episodes = value["episodes"]
        contrasts = _contrasts(episodes, enhanced=enhanced)
        transcripts = _representatives(episodes, enhanced=enhanced)
        result = {"schema_version": PUBLIC_SCHEMA if enhanced else PREVIOUS_PUBLIC_SCHEMA,
                  "domain": deepcopy(value["domain"]), "semantics": deepcopy(value["semantics"]),
                  "profile": _profile(episodes, enhanced=enhanced), "contrasts": contrasts, "transcripts": transcripts,
                  "omitted": {"episodes": len(episodes)-len(transcripts), "contrast_rows": 0,
                              "selection": "dominant failure reason; matched cell/run then largest failure group; median-length failure and same-group success only. Contrasts keep primary group and at least one matched group if available; other groups may be omitted."}}
        # The legacy public view does not display the new progress objects.
        result["semantics"].pop("progress_process", None)
        # Describe the same rules once, briefly. Keep their thresholds, units,
        # body binding and ABI availability; no event or count is redefined.
        result["semantics"]["rules"] = {
            "tilt": "abs(root roll/pitch)>tilt_rad",
            "slip_any": "any loaded support speed>slip_mps; global streak",
            "low_clearance_any": "any unloaded point clearance<C1/C2 threshold; global streak",
            "slip_body": "slip streak per named body",
            "low_clearance_body": "clearance streak per named body",
            "low_progress": "full-window max(root x)-initial x<gain threshold",
            "unloaded_rising": "unloaded world vz>vertical threshold",
            "unloaded_lowering": "unloaded world vz<-vertical threshold",
            "contact_instant": "instant force norm>contact_N",
            "contact_history": "buffer max norm>contact_N; slot times unknown",
            "no_instant_support": "all instant contacts false",
            "forbidden_manager": "task terminal predicate",
            "forbidden_history": "buffer force predicate; not termination authority",
        }
        result["semantics"].update(
            coordinates="terrain-relative position; world velocity/force; wxyz quaternion; signed Euler roll/pitch rad",
            time="1-based control step; edge in (previous,current]; start=1 left-censored; terminal included; end=terminal is observation cutoff, not natural cessation",
            contact="instant norm>contact_N; buffered max separate; no slot times",
            limits="point != toe/sole center; fast loaded-point translation does not prove surface slip; low progress != still body; root may differ from forbidden body; surface/causality unknown; first arrival censors later motion",
        )
        if enhanced:
            result["semantics"].pop("body_extrema_columns", None)
            result["semantics"]["rules"]["unloaded_low_motion"] = value["semantics"]["rules"]["unloaded_low_motion"]
            result["omitted"]["selection"] = (
                "Per stratum: dominant reason, matched then largest cell/run. Examples: modal (tilt,low_progress), "
                "then median duration. pattern_n/group_n counts that pair, not full timeline. Other groups omitted.")
            result["profile"]["event_count_scope"] = (
                "All episodes in stratum_totals. Counts out of row n; confirmed before terminal, overlap/unequal exposure. "
                "Reason rows may be omitted; last-1s presence omitted.")
            result["semantics"]["root_motion"] = (
                "first/last=[x_m,roll_rad,pitch_rad,up_z]; extrema=[min,step,max,step], earliest ties. "
                "max_x_retreat=[prior_peak_step,trough_step,drop_m]. up_z=1-2(qx^2+qy^2), body up axis world z; "
                "Euler wraps. Event measurements use same interval.")
            result["semantics"]["time"] = (
                "Observed steps 1..steps include terminal; edge in (previous,current]; start=1 left-censored; end=terminal is cutoff, not cessation")
        for section in ("rules", "actionability"):
            result["semantics"][section] = {
                _PUBLIC_CODES.get(code, code): detail
                for code, detail in result["semantics"][section].items()}
        result["semantics"]["rules"]["loaded_point_motion_any"] = (
            "any reference point with instant force norm>contact_N and planar world speed>loaded_point_speed_mps; global streak may alternate bodies")
        result["semantics"]["rules"]["loaded_point_motion_body"] = (
            "instant loaded-point predicate for one named body; confirmed only after loaded_point_steps consecutive steps")
        result["semantics"]["actionability"]["loaded_point_motion_any"] = [
            "instant_only", "any(loaded_point_motion_body predicate)"]
        if enhanced:
            result["semantics"]["rules"]["loaded_point_motion_any"] = (
                "any point: instant force norm>contact_N AND planar world speed>loaded_point_speed_mps; global streak may alternate bodies")
            result["semantics"]["actionability"]["loaded_point_motion_any"] = [
                "instant_only", "support_contact_force", "support_lin_vel"]
            result["semantics"]["actionability"]["streaks"] = [
                "diagnostic_only: streaks need absent pose/velocity history; force buffer insufficient"]
        thresholds = result["semantics"]["thresholds"]
        thresholds["loaded_point_speed_mps"] = thresholds.pop("slip_mps")
        thresholds["loaded_point_steps"] = thresholds.pop("slip_steps")
        for transcript in transcripts:
            for event in transcript["events"]:
                event["code"] = _PUBLIC_CODES.get(event["code"], event["code"])
        if enhanced:
            # Definitions are a view too: retain profile rules and every displayed
            # interval rule, without repeating a catalogue of unselected events.
            visible_codes = {*_PUBLIC_EVENTS, "unloaded_low_motion",
                             *(event["code"] for row in transcripts for event in row["events"])}
            result["semantics"]["rules"] = {key: detail for key, detail in result["semantics"]["rules"].items()
                                            if key in visible_codes}
            result["semantics"]["actionability"] = {
                key: detail for key, detail in result["semantics"]["actionability"].items()
                if key in visible_codes or key in {"streaks", "authority"}}
            result["semantics"]["limits"] = (
                "point != toe/sole; loaded-point motion does not prove surface slip; low progress != still body; "
                "unloaded low speed != physical separation/freeze; root may differ from forbidden body; no surface/causality.")
        # Protect the primary failure group, including an explicit missing
        # success result. If it has no match, retain a distinct matched group too.
        protected_rows = contrasts["rows"][:1]
        if enhanced:
            by_stratum = defaultdict(list)
            for row in contrasts["rows"]:
                by_stratum[row["stratum"]].append(row)
            for rows in by_stratum.values():
                reason_n = defaultdict(int)
                for row in rows:
                    reason_n[row["reason"]] += row["failed"][0]
                selected = min(rows, key=lambda row: (-reason_n[row["reason"]], row["reason"],
                    row["status"] != "matched", -row["failed"][0], row["cell"], row["run_ordinal"]))
                if selected not in protected_rows:
                    protected_rows.append(selected)
        first_matched = next((row for row in contrasts["rows"] if row["status"] == "matched"), None)
        if first_matched is not None and first_matched not in protected_rows:
            protected_rows.append(first_matched)
        if enhanced:
            columns = ["stratum", "reason", "cell", "run_ordinal", "failed", "success", "status"]
            contrasts["columns"] = columns
            contrasts["rows"] = [[row[key] for key in columns] for row in contrasts["rows"]]
            protected_rows = [[row[key] for key in columns] for row in protected_rows]
        while _size(result) > MAX_PUBLIC_BYTES and len(contrasts["rows"]) > len(protected_rows):
            index = next(i for i in range(len(contrasts["rows"])-1, -1, -1)
                         if contrasts["rows"][i] not in protected_rows)
            contrasts["rows"].pop(index)
            result["omitted"]["contrast_rows"] += 1
        # Keep every stratum's complete totals; detailed reason rows are a view
        # over those totals. Preserve the selected process and its success class.
        profile = result["profile"]
        protected_groups = set() if enhanced else {(row["stratum"], row["primary_reason"]) for row in transcripts}
        while _size(result) > MAX_PUBLIC_BYTES:
            optional = [row for row in profile["rows"] if tuple(row[:2]) not in protected_groups]
            if not optional:
                break
            row = min(optional, key=lambda row: (row[2], row[0], row[1]))
            profile["rows"].remove(row)
            profile["omitted_group_rows"] += 1
            profile["omitted_group_episodes"] += row[2]
        # Extrema repeat the full-episode ranges and are secondary to process
        # intervals and a matched comparison. Never trim all comparison rows.
        for row in transcripts:
            if _size(result) <= MAX_PUBLIC_BYTES:
                break
            if "body_extrema" in row:
                row["omitted_body_extrema"] = len(row["body_extrema"])
                row["body_extrema"] = []
        while _size(result) > MAX_PUBLIC_BYTES:
            with_detail = [(row, [event for event in row["events"]
                                  if event not in _process_core(row["events"], row["steps"])])
                           for row in transcripts if len(row["events"]) > 2]
            with_detail = [(row, optional) for row, optional in with_detail if optional]
            if not with_detail:
                break
            row, optional = max(with_detail, key=lambda item: _size(item[0]["events"]))
            row["events"].remove(min(optional, key=lambda event: _event_priority(event, row["steps"])))
            row["omitted_events"] += 1
            if enhanced:
                visible = {*_PUBLIC_EVENTS, "unloaded_low_motion",
                           *(event["code"] for row in transcripts for event in row["events"])}
                for section in ("rules", "actionability"):
                    result["semantics"][section] = {key: detail for key, detail in result["semantics"][section].items()
                        if key in visible or key in {"streaks", "authority"}}
        if enhanced and not result["semantics"]["rules"].keys() & {"unloaded_rising", "unloaded_lowering"}:
            thresholds.pop("unloaded_vertical_speed_mps", None)
            thresholds.pop("unloaded_motion_steps", None)
    elif value.get("schema_version") in {LEGACY_PUBLIC_SCHEMA, PREVIOUS_PUBLIC_SCHEMA, PUBLIC_SCHEMA}:
        result = deepcopy(dict(value))
    else:
        raise ValueError("Stair behavior schema mismatch")
    if set(result) != _PUBLIC_KEYS or set(result["domain"]) != _DOMAIN_KEYS:
        raise ValueError("Stair behavior fields mismatch")
    enhanced = result["schema_version"] == PUBLIC_SCHEMA
    if result["domain"]["rule_version"] != (RULE_VERSION if enhanced else LEGACY_RULE_VERSION):
        raise ValueError("Stair behavior rule version mismatch")
    if result["domain"]["history_status"] not in {"recorded", "not_recorded"}:
        raise ValueError("Stair behavior history status mismatch")
    legacy = result["schema_version"] == LEGACY_PUBLIC_SCHEMA
    profile = result["profile"]
    n = profile["episode_count"]
    records = _profile_records(result)
    if type(n) is not int or n <= 0 or sum(row["n"] for row in records) + profile.get("omitted_group_episodes", 0) != n:
        raise ValueError("Stair behavior episode denominator mismatch")
    for row in records:
        count_keys = set(row) - {"stratum", "primary_reason", "min_steps", "max_steps"}
        if any(type(row[key]) is not int or not 0 <= row[key] <= row["n"] for key in count_keys) or row["n"] <= 0:
            raise ValueError("Stair behavior profile counts mismatch")
    if result["omitted"]["episodes"] + len(result["transcripts"]) != n:
        raise ValueError("Stair behavior transcript coverage mismatch")
    profile_keys = {"episode_count", "columns", "event_count_scope", "rows"}
    if set(profile) != profile_keys | ({"indexed_labels"} if legacy else {
            "stratum_totals", "omitted_group_rows", "omitted_group_episodes"}):
        raise ValueError("Stair behavior profile fields mismatch")
    if not legacy:
        totals = profile["stratum_totals"]
        expected = {"stratum", "n", "success", "root_anchor_crossed", *_PUBLIC_EVENTS,
                    *["window_"+code for code in (() if enhanced else _WINDOW_EVENTS)]}
        if enhanced:
            expected.add("unloaded_low_motion")
        if set(totals["columns"]) != expected or len(totals["columns"]) != len(expected):
            raise ValueError("Stair behavior stratum total columns mismatch")
        complete = [dict(zip(totals["columns"], row, strict=True)) for row in totals["rows"]]
        if (sum(row["n"] for row in complete) != n
                or len({row["stratum"] for row in complete}) != len(complete)
                or any(type(count) is not int or not 0 <= count <= row["n"]
                       for row in complete for key, count in row.items() if key != "stratum")):
            raise ValueError("Stair behavior stratum total counts mismatch")
    if set(result["omitted"]) != {"episodes", "contrast_rows", "selection"}:
        raise ValueError("Stair behavior omission fields mismatch")
    contrast = result["contrasts"]
    if legacy:
        if set(contrast) != {"matched_cell_run_count", "all_cell_run_count", "columns", "rows", "missing_group_count", "scope"}:
            raise ValueError("Stair behavior contrast fields mismatch")
        if (contrast["matched_cell_run_count"] + contrast["missing_group_count"] != contrast["all_cell_run_count"]
                or len(contrast["rows"]) + result["omitted"]["contrast_rows"] != contrast["matched_cell_run_count"]):
            raise ValueError("Stair behavior contrast coverage mismatch")
    else:
        if set(contrast) != {"failure_group_count", "matched_group_count", "missing_success_group_count", "rows", "scope"} | ({"side_columns", "columns"} if enhanced else set()):
            raise ValueError("Stair behavior contrast fields mismatch")
        if (contrast["matched_group_count"] + contrast["missing_success_group_count"] != contrast["failure_group_count"]
                or len(contrast["rows"]) + result["omitted"]["contrast_rows"] != contrast["failure_group_count"]):
            raise ValueError("Stair behavior contrast coverage mismatch")
        if enhanced:
            expected = {"stratum", "reason", "cell", "run_ordinal", "failed", "success", "status"}
            if len(contrast["columns"]) != len(expected) or set(contrast["columns"]) != expected:
                raise ValueError("Stair behavior contrast group columns mismatch")
        for values in contrast["rows"]:
            row = dict(zip(contrast["columns"], values, strict=True)) if enhanced else values
            for side in ("failed", "success"):
                if enhanced:
                    columns = contrast["side_columns"]
                    expected = {"n", *_PUBLIC_EVENTS, "unloaded_low_motion", "up_z_negative_n",
                                "max_x_retreat_m", "roll_rad_range", "pitch_rad_range"}
                    if len(columns) != len(expected) or set(columns) != expected:
                        raise ValueError("Stair behavior contrast columns mismatch")
                    counts = dict(zip(columns, row[side], strict=True))
                    maximum = counts.pop("max_x_retreat_m")
                    if (counts["n"] == 0 and maximum is not None) or (counts["n"] and (maximum is None or maximum < 0)):
                        raise ValueError("Stair behavior contrast motion coverage mismatch")
                    for axis in ("roll_rad_range", "pitch_rad_range"):
                        bounds = counts.pop(axis)
                        if ((counts["n"] == 0 and bounds is not None)
                                or (counts["n"] and (not isinstance(bounds, list) or len(bounds) != 2 or bounds[0] > bounds[1]))):
                            raise ValueError("Stair behavior contrast range coverage mismatch")
                    condition_counts = {key: value for key, value in counts.items() if key != "n"}
                else:
                    counts = row[side]
                    condition_counts = counts["confirmed"]
                    if set(condition_counts) != set(_PUBLIC_EVENTS):
                        raise ValueError("Stair behavior contrast columns mismatch")
                if (type(counts["n"]) is not int or counts["n"] < (1 if side == "failed" else 0)
                        or any(type(value) is not int or not 0 <= value <= counts["n"]
                               for value in condition_counts.values())):
                    raise ValueError("Stair behavior contrast counts mismatch")
            success_n = dict(zip(contrast["side_columns"], row["success"], strict=True))["n"] if enhanced else row["success"]["n"]
            expected_status = "matched" if success_n else "missing_same_cell_run_success"
            if row["status"] != expected_status:
                raise ValueError("Stair behavior contrast missingness mismatch")
    for transcript in result["transcripts"]:
        expected = {"episode_id", "cell", "stratum", "primary_reason", "steps", "dt_s", "anchor",
                    "events", "omitted_events", "relationship"}
        expected |= ({"root_motion", "process_selection"} if enhanced else
                     {"ranges", "body_extrema", "coverage"} | (set() if legacy else {"omitted_body_extrema"}))
        if set(transcript) != expected:
            raise ValueError("Stair behavior transcript fields mismatch")
        if transcript["relationship"] not in {"index_example", "same_cell_run", "unmatched_context"}:
            raise ValueError("Stair behavior transcript relationship mismatch")
        for event in transcript["events"]:
            expected = {"code", "body", "start", "end", "confirmed"}
            if enhanced and event["code"] == "unloaded_low_motion":
                expected.add("measured")
            if set(event) != expected:
                raise ValueError("Stair behavior event fields mismatch")
            if not 1 <= event["start"] <= event["end"] <= transcript["steps"]:
                raise ValueError("Stair behavior event time mismatch")
            if event["confirmed"] is not None and not event["start"] <= event["confirmed"] <= event["end"]:
                raise ValueError("Stair behavior confirmation time mismatch")
    if _size(result) > MAX_PUBLIC_BYTES:
        raise ValueError("Stair behavior public byte budget exceeded")
    from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
    assert_public_projection_safe(result)
    return result


def compare_stair_behavior(parent, candidate, *, expected_parent_checkpoint_hash: str | None) -> dict:
    new_versions = {"true_rl_agent.stair.behavior.full.v3", "true_rl_agent.stair.behavior.public.v4"}
    if any(isinstance(value, Mapping) and value.get("schema_version") in new_versions for value in (parent, candidate)):
        from .process_report import compare_stair_process_reports
        return compare_stair_process_reports(parent, candidate, expected_parent_checkpoint_hash=expected_parent_checkpoint_hash)
    if not isinstance(parent, Mapping) or not isinstance(candidate, Mapping) or not expected_parent_checkpoint_hash:
        return {"status": "not_available", "reason": "missing_bound_parent_or_behavior"}
    left, right = project_stair_behavior(parent), project_stair_behavior(candidate)
    if left["domain"]["checkpoint_hash"] != expected_parent_checkpoint_hash:
        return {"status": "not_comparable", "reason": "parent_checkpoint_mismatch"}
    if not left["domain"]["source_hash"] or not right["domain"]["source_hash"]:
        return {"status": "not_comparable", "reason": "execution_source_unavailable"}
    unequal = [key for key in ("task_protocol_hash", "suite_hash", "projection_hash", "source_hash", "cohort_hash", "episode_keys_hash", "reward_input_schema_hash", "rule_version")
               if left["domain"][key] != right["domain"][key]]
    if unequal:
        return {"status": "not_comparable", "reason": "comparison_domain_mismatch", "fields": unequal}
    available = []
    for item in (left, right):
        available.append({ _PUBLIC_CODES.get(column, column) for column in item["profile"]["columns"]
                          if column not in {"stratum", "primary_reason", "n", "root_anchor_crossed", "min_steps", "max_steps"}})
    common = [column for column in (*_PUBLIC_EVENTS, "unloaded_low_motion", *["window_"+code for code in _PROFILE_EVENTS])
              if column in available[0] and column in available[1]]
    by_side = []
    for item in (left, right):
        counts = defaultdict(lambda: [0]*(2+len(common)))
        legacy = item["schema_version"] == LEGACY_PUBLIC_SCHEMA
        totals = item["profile"].get("stratum_totals")
        records = (_profile_records(item) if legacy else
                   [dict(zip(totals["columns"], row, strict=True)) for row in totals["rows"]])
        for row in records:
            entry = counts[row["stratum"]]
            entry[0] += row["n"]
            entry[1] += (row["n"] if row["primary_reason"] == "success" else 0) if legacy else row["success"]
            for index, column in enumerate(common, start=2):
                entry[index] += row[column]
        by_side.append(counts)
    rows = []
    for stratum in sorted(set(by_side[0]) | set(by_side[1])):
        a, b = by_side[0].get(stratum), by_side[1].get(stratum)
        rows.append({"stratum": stratum, "parent": a, "candidate": b,
                     "rate_delta": ([_rounded(b[i]/b[0]-a[i]/a[0]) for i in range(1, 2+len(common))] if a and b else None)})
    return {"status": "available", "parent_checkpoint_hash": expected_parent_checkpoint_hash,
            "candidate_checkpoint_hash": right["domain"]["checkpoint_hash"],
            "columns": ["n", "success", *common], "rows": rows,
            "not_compared_columns": sorted(available[0] ^ available[1]),
            "scope": "same-domain stratum distributions; only equivalent present columns; not step-aligned or causal; unequal episode exposure retained",
            "history_comparison": "not_in_common_profile; never impute missing history"}


def _profile_records(value):
    """Resolve each version's labels and columns before validation/comparison."""
    profile = value["profile"]
    columns = profile["columns"]
    legacy = value["schema_version"] == LEGACY_PUBLIC_SCHEMA
    expected = {"stratum", "primary_reason", "n", "root_anchor_crossed", "min_steps", "max_steps"}
    expected.update(_PROFILE_EVENTS if legacy else _PUBLIC_EVENTS)
    if value["schema_version"] == PUBLIC_SCHEMA:
        expected.add("unloaded_low_motion")
    else:
        expected.update("window_"+code for code in (_PROFILE_EVENTS if legacy else _WINDOW_EVENTS))
    if len(columns) != len(expected) or set(columns) != expected:
        raise ValueError("Stair behavior profile columns mismatch")
    records = []
    for row in profile["rows"]:
        if len(row) != len(columns):
            raise ValueError("Stair behavior profile counts mismatch")
        record = {_PUBLIC_CODES.get(column, column): value for column, value in zip(columns, row, strict=True)}
        if legacy:
            for field in ("stratum", "primary_reason"):
                record[field] = profile["indexed_labels"][field][record[field]]
        records.append(record)
    return records
