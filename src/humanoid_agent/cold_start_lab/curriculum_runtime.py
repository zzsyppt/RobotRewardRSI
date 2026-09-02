"""Runner-owned execution of compiled Phase 8E curriculum schedules."""

from __future__ import annotations

from typing import Any


RUNTIME_SCHEMA = "humanoid_agent.phase8e.compiled_curriculum.runtime.v1"


class Phase8ECompiledCurriculumRuntime:
    """Apply a compiled train-only curriculum to a VecEnv by progress."""

    def __init__(self, *, curriculum: dict[str, Any], planned_training_steps: int) -> None:
        self.curriculum = curriculum
        self.planned_training_steps = max(1, int(planned_training_steps))
        self.enabled = bool(curriculum.get("enabled", False))
        self.stages = [stage for stage in curriculum.get("stages", []) if isinstance(stage, dict)]
        self.step_calls = 0
        self.last_applied_stage: dict[str, Any] | None = None
        self.last_apply_report: dict[str, Any] | None = None
        self.stage_transitions: list[dict[str, Any]] = []
        self.stage_step_counts: dict[str, int] = {}

    @classmethod
    def from_training_config(
        cls,
        config: dict[str, Any],
        *,
        planned_training_steps: int,
    ) -> "Phase8ECompiledCurriculumRuntime":
        raw = config.get("runner_curriculum")
        curriculum = raw if isinstance(raw, dict) else {"enabled": False, "stages": []}
        return cls(curriculum=curriculum, planned_training_steps=planned_training_steps)

    def public_summary(self) -> dict[str, Any]:
        return {
            "schema_version": "humanoid_agent.phase8e.compiled_curriculum.summary.v1",
            "enabled": self.enabled,
            "execution_mode": self.curriculum.get("execution_mode"),
            "stage_count": len(self.stages),
            "capabilities": self.curriculum.get("capabilities", []),
            "dr_execution": self.curriculum.get("dr_execution"),
        }

    def runtime_summary(self) -> dict[str, Any]:
        return {
            "schema_version": RUNTIME_SCHEMA,
            "enabled": self.enabled,
            "planned_training_steps": self.planned_training_steps,
            "step_calls": self.step_calls,
            "last_applied_stage": self._public_stage(self.last_applied_stage),
            "last_apply_report": self.last_apply_report,
            "stage_transitions": list(self.stage_transitions),
            "stage_step_counts": dict(self.stage_step_counts),
        }

    def apply_before_step(self, env: Any) -> None:
        if not self.enabled or not self.stages:
            return
        progress = min(1.0, self.step_calls / self.planned_training_steps)
        stage = self._stage_for_progress(progress)
        report = self._apply_stage_push(env, stage)
        name = str(stage.get("name", "unnamed_stage"))
        if self.last_applied_stage is None or self.last_applied_stage.get("name") != name:
            self.stage_transitions.append(
                {
                    "step": self.step_calls,
                    "progress_fraction": round(progress, 6),
                    "stage": name,
                }
            )
        self.last_applied_stage = stage
        self.last_apply_report = report
        self.stage_step_counts[name] = self.stage_step_counts.get(name, 0) + 1

    def mark_step_complete(self) -> None:
        self.step_calls += 1

    def _stage_for_progress(self, progress: float) -> dict[str, Any]:
        chosen = self.stages[-1]
        for stage in self.stages:
            raw_range = stage.get("progress_range")
            if not isinstance(raw_range, list | tuple) or len(raw_range) != 2:
                continue
            start = float(raw_range[0])
            end = float(raw_range[1])
            if start <= progress < end or (progress >= 1.0 and end >= 1.0):
                chosen = stage
                break
        return chosen

    def _apply_stage_push(self, env: Any, stage: dict[str, Any]) -> dict[str, Any]:
        push = stage.get("push")
        if not isinstance(push, dict):
            return {"status": "skipped", "reason": "stage_has_no_push_payload"}
        base = _unwrap_env(env)
        term = _attr_path(base, ("cfg", "events", "push_robot"))
        if term is None:
            return {"status": "not_applied", "reason": "push_robot_term_missing"}
        return _apply_push_to_term(term, push)

    @staticmethod
    def _public_stage(stage: dict[str, Any] | None) -> dict[str, Any] | None:
        if stage is None:
            return None
        return {
            "name": stage.get("name"),
            "progress_range": stage.get("progress_range"),
            "capabilities": stage.get("capabilities", []),
            "push": stage.get("push"),
            "robustness": stage.get("robustness"),
        }


class Phase8EVecEnvCurriculumWrapper:
    """Proxy a VecEnv and apply runner-compiled curriculum before each step."""

    def __init__(self, env: Any, runtime: Phase8ECompiledCurriculumRuntime) -> None:
        self.env = env
        self.phase8e_curriculum_runtime = runtime

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def step(self, actions: Any) -> tuple[Any, ...]:
        self.phase8e_curriculum_runtime.apply_before_step(self.env)
        try:
            result = self.env.step(actions)
        finally:
            self.phase8e_curriculum_runtime.mark_step_complete()
        return result


def _apply_push_to_term(term: Any, push: dict[str, Any]) -> dict[str, Any]:
    velocity = push.get("velocity_range_mps")
    interval = push.get("interval_range_s")
    if not isinstance(velocity, dict):
        return {"status": "not_applied", "reason": "missing_velocity_range_mps"}
    try:
        x_range = _pair(velocity["x"])
        y_range = _pair(velocity["y"])
        interval_range = _pair(interval if interval is not None else [1.0, 1.0])
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "not_applied", "reason": f"invalid_push_payload:{type(exc).__name__}"}

    try:
        term.interval_range_s = interval_range
        params = getattr(term, "params")
        params["velocity_range"]["x"] = x_range
        params["velocity_range"]["y"] = y_range
    except Exception as exc:  # noqa: BLE001 - runtime instrumentation should report, not crash training.
        return {"status": "not_applied", "reason": f"push_term_mutation_failed:{type(exc).__name__}"}
    return {
        "status": "applied",
        "path": "env.unwrapped.cfg.events.push_robot",
        "interval_range_s": list(interval_range),
        "velocity_range_mps": {"x": list(x_range), "y": list(y_range)},
    }


def _unwrap_env(env: Any) -> Any:
    current = env
    for _ in range(8):
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            return current
        current = unwrapped
    return current


def _attr_path(obj: Any, path: tuple[str, ...]) -> Any:
    current = obj
    for name in path:
        current = getattr(current, name, None)
        if current is None:
            return None
    return current


def _pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        raise ValueError(f"expected pair, got {value!r}")
    return (float(value[0]), float(value[1]))
