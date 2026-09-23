"""Scoring for the bounded P6 sequence diagnostic; not a production protocol."""
from __future__ import annotations


def sequence_outcome(steps, stable, events, *, failure_label, final_stable,
                     window_steps, dwell_steps):
    """Measure one recovery after the last of exactly two verified pushes."""
    invalid = sum(not event["injection_verified"] for event in events)
    if failure_label is None and len(events) != 2:
        invalid += abs(2 - len(events))
    recovery = None
    if len(events) == 2 and not invalid:
        last = events[-1]["control_step"]
        dwell = 0
        previous = last
        for step, value in zip(steps, stable, strict=True):
            if step <= last:
                continue
            if step > last + window_steps:
                break
            dwell = (dwell + 1 if step == previous + 1 else 1) if value else 0
            previous = step
            if dwell >= dwell_steps:
                recovery = step
                break
    if invalid:
        label = "invalid_push_injection"
    elif failure_label is not None:
        label = failure_label
    elif recovery is not None and final_stable:
        label = "success_recovered"
    else:
        label = "not_recovered"
    return {"success": label == "success_recovered", "failure_label": label,
            "sequence_recovered": recovery is not None,
            "sequence_recovery_control_step": recovery,
            "invalid_injection_count": invalid}


def raw_conditions(position, quaternion_xyzw, linear, angular, *, initial_xy, origin, parameters):
    """Numpy-only reconstruction from pre-reset physical state."""
    import numpy as np
    p = parameters
    x, y, z, w = quaternion_xyzw.T
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x*x + y*y))
    pitch = np.arcsin(np.clip(2 * (w*y - z*x), -1, 1))
    height = position[:, 2] - origin[2]
    drift = np.linalg.norm(position[:, :2] - initial_xy, axis=1)
    stable = ((np.linalg.norm(linear[:, :2], axis=1) <= p["recovery_velocity_mps"])
              & (np.linalg.norm(angular, axis=1) <= p["recovery_angular_velocity_radps"])
              & (np.abs(roll) <= p["recovery_tilt_rad"])
              & (np.abs(pitch) <= p["recovery_tilt_rad"])
              & (height >= p["recovery_height_m"]))
    failures = ((pitch > p["fall_tilt_rad"], "fall_forward"),
                (pitch < -p["fall_tilt_rad"], "fall_backward"),
                (np.abs(roll) > p["fall_tilt_rad"], "fall_side"),
                (height < p["fall_height_m"], "height_collapse"))
    if p["max_displacement_m"] is not None:
        failures += ((drift > p["max_displacement_m"], "out_of_bounds"),)
    failure = next((name for mask, name in failures if bool(mask.any())), None)
    return stable.tolist(), failure
