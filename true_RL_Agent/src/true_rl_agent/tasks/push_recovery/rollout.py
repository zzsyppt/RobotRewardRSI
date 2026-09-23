"""Task-owned Push rollout aggregation and pre-reset native recording."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from true_rl_agent.tasks.push_recovery.isaac_config import public_runtime_dynamics
from true_rl_agent.tasks.push_recovery.runtime import push_runtime_state
from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed, public_seed_record
from true_rl_agent.tasks.push_recovery.numeric_capture import PushNumericCapture, PushNumericCaptureOptions

PUSH_ROLLOUT_SCHEMA_VERSION = "true_rl_agent.push_recovery.rollout.v2"
PUSH_FORCE_ROLLOUT_SCHEMA_VERSION = "true_rl_agent.push_recovery.rollout.v3"


def aggregate_recovery_episodes(records: list[dict[str, Any]], *, force_sequence: bool = False) -> dict[str, Any]:
    counts = Counter(record["failure_label"] for record in records)
    eligible = [record for record in records if not record["control_only"]]
    success = sum(record["success"] for record in eligible)
    total = len(records)
    strata = {}
    for name in ("low", "medium", "high", "repeated"):
        rows = [record for record in eligible if record.get("stratum") == name]
        successes = sum(record["success"] for record in rows)
        strata[name] = {"episodes": len(rows), "successes": successes,
            "success_rate": successes / len(rows) if rows else None}
    result = {"episodes": total, "success_count": success, "successes": success,
        "stratum_results": strata,
        "success_rate": success / len(eligible) if eligible else None,
        "failure_breakdown": {"total": total, "counts": dict(sorted(counts.items())),
            "rates": {label: count / total for label, count in sorted(counts.items())} if total else {},
            "dominant_failure": next((label for label, _ in counts.most_common() if label not in {"success_recovered", "no_push_control"}), None)},
        "eligible_episode_count": len(eligible),
        "pushed_episode_count": sum(record["push_event_count"] > 0 for record in eligible),
        "planned_push_event_count": sum(record["planned_push_count"] for record in eligible),
        "push_event_count": sum(record["push_event_count"] for record in records),
        "invalid_injection_count": sum(record["invalid_injection_count"] for record in records),
        "control_episode_count": total - len(eligible),
        "control_success_count": sum(record["control_success"] for record in records)}
    if force_sequence:
        result.update(
            completed_push_event_count=sum(record["completed_push_event_count"] for record in records),
            verified_push_event_count=sum(record["verified_push_event_count"] for record in records),
            sequence_recovered_count=sum(record["sequence_recovered"] for record in eligible),
        )
    else:
        result["recovered_event_count"] = sum(record["recovered_event_count"] for record in records)
    return result


def _finite(value: Any) -> bool:
    import torch
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(item) for item in value)
    return bool(torch.isfinite(value).all()) if isinstance(value, torch.Tensor) else True


def run_physical_push_rollout(env: Any, config: dict[str, Any], args: Any, *,
    inference_policy: Any = None, split_name: str, run_index: int,
    numeric_capture: PushNumericCaptureOptions | None = None) -> dict[str, Any]:
    import torch
    base = env.unwrapped
    seed = derive_runner_seed(base_seed=args.seed, split_name=split_name, run_index=run_index)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base.seed(seed)
    state = push_runtime_state(base)
    state.begin_evaluation(split_name, seed)
    reset = env.reset()
    obs = reset[0] if isinstance(reset, tuple) else reset
    recorder = getattr(args, "_push_video_recorder", None)
    force_sequence = "push_evaluation_protocol" in config
    if recorder is not None:
        recorder.start(f"{split_name}-{run_index:02d}", seed, task_records_steps=force_sequence)
    num_envs, device = int(base.num_envs), base.device
    max_steps = args.max_steps if args.max_steps is not None else int(base.max_episode_length)
    previous_action = torch.zeros((num_envs, base.action_manager.total_action_dim), device=device)
    action_sum = torch.zeros(num_envs, device=device)
    action_rate_sum, action_samples = torch.zeros_like(action_sum), torch.zeros_like(action_sum)
    observation_finite, reward_finite, calls = _finite(obs), True, 0
    previous_capture = getattr(base, "_push_pre_reset_capture", None)
    previous_numeric_capture = getattr(state, "numeric_capture", None)
    numeric = PushNumericCapture(numeric_capture, state, split_name=split_name, run_index=run_index) if numeric_capture is not None else None
    state.numeric_capture = numeric
    if recorder is not None and force_sequence:
        def capture_frame() -> None:
            if not bool(state.closed[recorder.options.env_index]):
                recorder.record_step()
        base._push_pre_reset_capture = capture_frame
    try:
        for _ in range(int(max_steps)):
            if inference_policy is not None:
                with torch.inference_mode():
                    actions = inference_policy(obs)
            elif args.policy == "random":
                actions = torch.randn_like(previous_action) * (args.action_scale if args.action_scale is not None else 1.0)
            else:
                actions = torch.zeros_like(previous_action)
            active = (~state.closed).to(dtype=action_sum.dtype)
            action_sum += torch.linalg.vector_norm(actions, dim=1) * active
            if calls:
                action_rate_sum += torch.linalg.vector_norm(actions - previous_action, dim=1) * active
            action_samples += active
            previous_action = actions.clone()
            result = env.step(actions)
            if not isinstance(result, tuple) or len(result) not in {4, 5}:
                raise RuntimeError("Push rollout requires the native or RSL-RL step result")
            obs, rewards = result[0], result[1]
            calls += 1
            observation_finite &= _finite(obs)
            reward_finite &= _finite(rewards)
            if bool(state.closed.all()):
                break
    finally:
        state.numeric_capture = previous_numeric_capture
        # Do not retain the last frame's tensors after an opt-in archive ends.
        if numeric is not None:
            for name in ("_numeric_pose", "_numeric_conditions", "_numeric_planar_motion"):
                if hasattr(state, name):
                    delattr(state, name)
        if recorder is not None and force_sequence:
            if previous_capture is None:
                del base._push_pre_reset_capture
            else:
                base._push_pre_reset_capture = previous_capture
    records = state.episode_records()
    for record in records:
        index = record["env_id"]
        count = max(1.0, float(action_samples[index]))
        record["mean_action_l2"] = float(action_sum[index]) / count
        record["mean_action_rate_l2"] = float(action_rate_sum[index]) / count
    aggregate = aggregate_recovery_episodes(records, force_sequence=force_sequence)
    if recorder is not None:
        outcome = records[recorder.options.env_index]
        record = recorder.finish(frame_count=outcome["episode_steps"] if force_sequence else max(0, outcome["episode_steps"] - 1))
        record.update(episode_outcome=outcome, episode_evidence_source="pre_reset_task_state",
            split=split_name, run_index=run_index, resolved_protocol_hash=state.parameters["resolved_protocol_hash"])
        if force_sequence:
            record.update(capture_point="after_physics_before_termination_and_reset", includes_terminal_frame=True)
    valid = observation_finite and reward_finite and aggregate["invalid_injection_count"] == 0 and bool(state.closed.all())
    means = {"mean_episode_steps": "episode_steps", "mean_min_base_height_m": "min_base_height_m",
        "mean_max_abs_pitch_rad": "max_abs_pitch_rad", "mean_max_abs_roll_rad": "max_abs_roll_rad",
        "mean_final_xy_drift_m": "max_xy_drift_m", "mean_final_planar_speed_mps": "final_planar_speed_mps",
        "mean_action_l2": "mean_action_l2", "mean_action_rate_l2": "mean_action_rate_l2"}
    if force_sequence:
        means["mean_max_xy_drift_m"] = means.pop("mean_final_xy_drift_m")
    secondary = {name: sum(record[key] for record in records) / len(records) for name, key in means.items()}
    secondary["recovery_window_success_rate"] = aggregate["success_rate"]
    seed_count = {"public_dev": args.public_seeds, "public_accept": args.public_accept_seeds,
        "sealed_validation": args.sealed_validation_seeds}[split_name]
    seed_record = public_seed_record(split_name=split_name, run_index=run_index, split_seed_count=seed_count)
    result = {"schema_version": PUSH_FORCE_ROLLOUT_SCHEMA_VERSION if force_sequence else PUSH_ROLLOUT_SCHEMA_VERSION,
        "status": "pass" if valid else "fail",
        "public_score": aggregate["success_rate"], "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_code_loaded_in_eval": False, "candidate_package_loaded": False, "candidate_package_loaded_in_eval": False,
        "run": {**seed_record, "subject_kind": args.evaluation_subject.kind, "subject_id": args.evaluation_subject.subject_id,
            "task_id": args.task_id, "round_id": args.round},
        "task": {"mechanism": state.parameters["mechanism"], "diagnostic_control_only": state.control_only},
        "configuration": public_runtime_dynamics(config),
        "environment": {"num_envs": num_envs, "step_dt": float(base.step_dt), "max_steps": int(max_steps)},
        "execution_cost": {"evaluation_env_count": num_envs, "vector_env_step_calls": calls,
            "closed_first_episode_control_steps": sum(record["episode_steps"] for record in records)},
        "evidence": {"reset_ok": True, "observation_finite": observation_finite, "reward_finite": reward_finite,
            "events_recorded_at_injection": True, "outcomes_captured_before_reset": True,
            "resolved_protocol_hash": state.parameters["resolved_protocol_hash"],
            "all_first_episodes_closed": bool(state.closed.all())},
        "aggregate": aggregate, "secondary_metrics": secondary, "per_episode": records,
        "metrics": {"success_rate": aggregate["success_rate"], "failure_breakdown": aggregate["failure_breakdown"]},
        "errors": [] if valid else [{"type": "PushPhysicalEvidenceInvalid", "message": "injection, finite-state or episode closure check failed"}]}
    if numeric is not None:
        result["numeric_evidence_ref"] = numeric.complete(records)
    return result
