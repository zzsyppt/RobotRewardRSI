"""Dispersed training-step samples with fixed budgets; tasks own meanings."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter


class TrainingNumericCapture:
    """Observe an existing training run without driving its loop or rewards.

    Each disjoint third of the declared run has at most eight dispersed samples.
    The early first point is anchored at step one to retain startup evidence
    when training fails early. These observations are not whole-run means or
    exact first activation times. Short runs sample all available steps.
    """

    def __init__(
        self, root: Path, *, identity: Mapping[str, Any], semantics: Mapping[str, Any],
        planned_control_steps: int, num_envs: int, steps_per_iteration: int,
        source_step_offset: int, initial_learning_iteration: int = 0,
        reported_iteration: Callable[[], int] | None = None,
        window_steps: int = 8, sampled_env_count: int = 16,
    ) -> None:
        for name, value in (("planned_control_steps", planned_control_steps), ("num_envs", num_envs),
                            ("steps_per_iteration", steps_per_iteration), ("window_steps", window_steps),
                            ("sampled_env_count", sampled_env_count)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if window_steps > 8 or sampled_env_count > 16:
            raise ValueError("training capture is bounded to 8 steps per window and 16 environments")
        for value in (source_step_offset, initial_learning_iteration):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError("training capture offsets must be non-negative integers")
        self.writer = NumericEvidenceWriter(root, identity)
        self.semantics = dict(semantics)
        self.planned_control_steps = planned_control_steps
        self.num_envs = num_envs
        self.steps_per_iteration = steps_per_iteration
        self.source_step_offset = source_step_offset
        self.initial_learning_iteration = initial_learning_iteration
        self.reported_iteration = reported_iteration
        self.env_ids = np.arange(min(num_envs, sampled_env_count), dtype=np.int64)
        edges = (0, planned_control_steps // 3, 2 * planned_control_steps // 3, planned_control_steps)
        self.windows = []
        for index, name in enumerate(("early", "middle", "late")):
            start, stop = edges[index] + 1, edges[index + 1]
            length = max(0, stop - start + 1)
            width = min(window_steps, length)
            planned_steps = [start + ((2 * j + 1) * length - 1) // (2 * width)
                             for j in range(width)]
            if index == 0 and planned_steps:
                planned_steps[0] = 1
            self.windows.append({"name": name, "start_step": start, "end_step": stop,
                                 "planned_sample_steps": width, "planned_steps": planned_steps,
                                 "sampled_steps": []})
        self._scheduled = {step: window for window in self.windows for step in window["planned_steps"]}
        self.observed_control_steps = 0
        self._reference: dict[str, Any] | None = None

    def wants_step(self, source_step: int) -> bool:
        step = source_step - self.source_step_offset
        return step in self._scheduled

    def observe(self, source_step: int, arrays: Mapping[str, Any] | None,
                *, metadata: Mapping[str, Any] | None = None) -> None:
        """Called once after each successful reward dispatch, including gaps."""
        if self._reference is not None:
            raise RuntimeError("training capture already finalized")
        step = source_step - self.source_step_offset
        if step != self.observed_control_steps + 1:
            raise ValueError("training capture observed nonconsecutive control steps")
        self.observed_control_steps = step
        window = self._scheduled.get(step)
        if window is None or arrays is None:
            return
        sampled: dict[str, np.ndarray] = {}
        for key, value in arrays.items():
            if hasattr(value, "detach"):
                detached = value.detach()
                if detached.ndim < 1 or detached.shape[0] != self.num_envs:
                    raise ValueError("training sample must preserve its environment axis")
                array = detached[:len(self.env_ids)].cpu().numpy().copy()
            else:
                value = np.asarray(value)
                if value.ndim < 1 or value.shape[0] != self.num_envs:
                    raise ValueError("training sample must preserve its environment axis")
                array = value[:len(self.env_ids)].copy()
            sampled[key] = array
        if not sampled:
            return
        local_iteration, step_in_iteration = divmod(step - 1, self.steps_per_iteration)
        sampled["env_id"] = self.env_ids.copy()
        self.writer.write_chunk(sampled, {
            "kind": "training_sample", "window": window["name"],
            "source_control_step": source_step, "training_control_step": step,
            "local_rollout_iteration": local_iteration,
            "learning_iteration_from_control_steps": self.initial_learning_iteration + local_iteration,
            "step_in_rollout": step_in_iteration + 1,
            "runner_reported_learning_iteration": self.reported_iteration() if self.reported_iteration else None,
            "task": dict(metadata or {}),
        })
        window["sampled_steps"].append(step)

    def finish(self, *, training_status: str) -> dict[str, Any]:
        """Publish observation coverage, independently of training success."""
        if self._reference is not None:
            return self._reference
        windows = []
        for window in self.windows:
            count = len(window["sampled_steps"])
            planned = window["planned_sample_steps"]
            status = ("not_scheduled" if not planned else
                      "sampled" if count == planned else
                      "partial" if count else
                      "not_reached" if self.observed_control_steps < window["planned_steps"][0] else "not_sampled")
            windows.append({**window, "status": status, "sampled_control_steps": count,
                            "sampled_env_transitions": count * len(self.env_ids),
                            "planned_sampled_env_transitions": planned * len(self.env_ids)})
        coverage = {
            "schema_version": "true_rl_agent.training_numeric_capture.v2",
            "sampling_strategy": "step_one_anchor_then_equal_bin_centers_within_disjoint_training_thirds",
            "training_status": training_status,
            "planned_control_steps": self.planned_control_steps,
            "observed_control_steps": self.observed_control_steps,
            "source_step_offset_excluded": self.source_step_offset,
            "steps_per_iteration": self.steps_per_iteration,
            "initial_learning_iteration": self.initial_learning_iteration,
            "num_envs": self.num_envs, "sampled_env_ids": self.env_ids.tolist(),
            "env_selection": "fixed_first_environment_ids_no_rng",
            "scope": "fixed_windows_and_environment_subset_not_whole_training_distribution",
            "windows": windows, "semantics": self.semantics,
        }
        # An empty index is an explicit no-sample record, not a fabricated step.
        self.writer.write_chunk({"sampled_control_step": np.asarray(
            [step for window in self.windows for step in window["sampled_steps"]], dtype=np.int64)},
            {"kind": "sampling_index"})
        reference = self.writer.complete(coverage)
        self._reference = {"status": "complete", "reference": reference, "coverage": coverage}
        return self._reference


class TrainingNumericAttachment:
    """Keep optional capture failures outside the reward/training result."""

    def __init__(self, create: Callable[[], TrainingNumericCapture],
                 bind: Callable[[TrainingNumericAttachment], None]) -> None:
        self.recorder: TrainingNumericCapture | None = None
        self._result: dict[str, Any] | None = None
        self._run("initialization", lambda: self._initialize(create, bind))

    @property
    def active(self) -> bool:
        return self.recorder is not None and self._result is None

    def _initialize(self, create, bind) -> None:
        self.recorder = create()
        bind(self)

    def _run(self, stage: str, operation: Callable[[], Any]) -> Any:
        if self._result is not None:
            return None
        try:
            return operation()
        except Exception as exc:
            root = exc
            while root.__cause__ is not None:
                root = root.__cause__
            self._result = {"status": "incomplete", "stage": stage,
                            "error_type": type(root).__name__[:64],
                            "error_message": str(root)[:512]}
            logging.getLogger(__name__).warning(
                "Training numeric attachment failed during %s: %s: %s", stage,
                self._result["error_type"], self._result["error_message"])
            return None

    def observe(self, source_step: int, sample: Callable[[], tuple[Mapping[str, Any], Mapping[str, Any]]]) -> None:
        def record():
            arrays, metadata = sample() if self.recorder.wants_step(source_step) else (None, None)
            self.recorder.observe(source_step, arrays, metadata=metadata)
        self._run("sampling", record)

    def finish(self, *, training_status: str) -> dict[str, Any]:
        result = self._run("finalization", lambda: self.recorder.finish(training_status=training_status))
        if result is not None:
            self._result = result
        return self._result


def finish_training_numeric_capture(capture: TrainingNumericAttachment | None,
                                    *, training_status: str, requested: bool = False) -> dict[str, Any]:
    if capture is None:
        return {"status": "not_started" if requested else "not_enabled"}
    return capture.finish(training_status=training_status)
