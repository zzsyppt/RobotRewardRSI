"""Pure scoring for the bounded P6 external-force diagnostic, not qualification."""

from __future__ import annotations


def _post_force_recovery(samples, end, window_steps, dwell_steps):
    previous, dwell = end, 0
    for step, value in samples:
        if step <= end:
            continue
        if step > end + window_steps:
            break
        dwell = (dwell + 1 if step == previous + 1 else 1) if value else 0
        previous = step
        if dwell >= dwell_steps:
            return step
    return None


def force_outcome(steps, stable, events, planned_count, *, failure_label,
                  final_stable, window_steps, dwell_steps):
    """Require a fresh post-force dwell for every planned pulse.

    ``end_control_step`` is the final sample with force applied. Recovery uses
    samples end+1 through end+window, inclusive. Returned recovery steps align
    with the supplied events; partial or unverified events have no recovery.
    ``injection_verified`` describes the substeps actually executed, separately
    from completion: a fall may interrupt a correctly applied pulse. A genuine
    failed injection remains invalid even when a physical failure owns the label.
    The caller supplies chronologically ordered samples and nonoverlapping
    pulse/recovery windows from the frozen diagnostic declaration.
    """
    if planned_count < 1 or window_steps < 1 or dwell_steps < 1:
        raise ValueError("planned_count, window_steps and dwell_steps must be positive")
    samples = list(zip(steps, stable, strict=True))
    invalid = max(0, len(events) - planned_count)
    if failure_label is None:
        invalid += max(0, planned_count - len(events))
    recoveries = []
    for event in events:
        completed = event["completed"]
        verified = event["injection_verified"]
        invalid += int(not verified or (not completed and failure_label is None))
        recovery = None
        if completed and verified:
            recovery = _post_force_recovery(
                samples, event["end_control_step"], window_steps, dwell_steps)
        recoveries.append(recovery)
    recovered = sum(step is not None for step in recoveries)
    if failure_label is not None:
        label = failure_label
    elif invalid:
        label = "invalid_push_injection"
    elif recovered == planned_count and final_stable:
        label = "success_recovered"
    else:
        label = "not_recovered"
    return {
        "success": failure_label is None and not invalid
        and recovered == planned_count and bool(final_stable),
        "failure_label": label,
        "recovered_event_count": recovered,
        "event_recovery_control_steps": recoveries,
        "invalid_injection_count": invalid,
    }


def force_sequence_outcome(steps, stable, events, planned_count, *, failure_label,
                           final_stable, window_steps, dwell_steps):
    """Recover once after all planned finite pulses; a single pulse is unchanged."""
    if planned_count < 1 or window_steps < 1 or dwell_steps < 1:
        raise ValueError("planned_count, window_steps and dwell_steps must be positive")
    samples = list(zip(steps, stable, strict=True))
    invalid = max(0, len(events) - planned_count)
    if failure_label is None:
        invalid += max(0, planned_count - len(events))
    invalid += sum(not e["injection_verified"] or
                   (not e["completed"] and failure_label is None) for e in events)
    complete = len(events) == planned_count and all(e["completed"] for e in events)
    recovery = None
    if complete and not invalid:
        recovery = _post_force_recovery(
            samples, events[-1]["end_control_step"], window_steps, dwell_steps)
    if failure_label is not None:
        label = failure_label
    elif invalid:
        label = "invalid_push_injection"
    elif recovery is not None and final_stable:
        label = "success_recovered"
    else:
        label = "not_recovered"
    return {"success": label == "success_recovered", "failure_label": label,
            "sequence_recovered": recovery is not None,
            "sequence_recovery_control_step": recovery,
            "invalid_injection_count": invalid}
