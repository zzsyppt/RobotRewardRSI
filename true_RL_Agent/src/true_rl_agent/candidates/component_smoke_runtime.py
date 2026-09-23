"""Spawn-isolated runtime smoke checks for candidate components."""

from __future__ import annotations

import math
import multiprocessing as mp
import queue as queue_module
import time
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.candidates.artifact import SealedCandidateArtifactRef
from true_rl_agent.candidates.authoring_public_result import public_authoring_result
from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm
from true_rl_agent.candidates.isolated_worker import (
    IsolatedWorkerProtocolError,
    IsolatedWorkerStartupError,
)
from true_rl_agent.candidates.loader import load_training_components
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.tasks.component_smoke import (
    ComponentSmokeAdapter,
    ComponentSmokeFixture,
)

COMPONENT_SMOKE_SCHEMA_VERSION = "true_rl_agent.component_smoke_result.v2"
DEFAULT_COMPONENT_SMOKE_TIMEOUT_S = 2.0
DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S = 2.0
DEFAULT_COMPONENT_SMOKE_WORKER_STARTUP_TIMEOUT_S = 60.0

_public_result = public_authoring_result


def run_component_smoke_check(
    candidate: CandidatePackageRef | SealedCandidateArtifactRef,
    smoke_adapter: ComponentSmokeAdapter | None,
    *,
    timeout_s: float = DEFAULT_COMPONENT_SMOKE_TIMEOUT_S,
    worker_startup_timeout_s: float = DEFAULT_COMPONENT_SMOKE_WORKER_STARTUP_TIMEOUT_S,
    load_timeout_s: float = DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a task-owned lightweight smoke check over candidate callables."""

    timeout_s = _positive_finite_timeout(timeout_s, "timeout_s")
    worker_startup_timeout_s = _positive_finite_timeout(
        worker_startup_timeout_s,
        "worker_startup_timeout_s",
    )
    load_timeout_s = _positive_finite_timeout(load_timeout_s, "load_timeout_s")
    if smoke_adapter is None:
        return _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "skip",
                "reason": "task_smoke_adapter_not_provided",
                "task_id": "not_provided",
                "checked_reward_terms": 0,
                "checked_diagnostics": 0,
            }
        )
    fixture = smoke_adapter.build_component_smoke_fixture()
    if fixture is None:
        return _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "skip",
                "reason": "task_smoke_fixture_not_available",
                "task_id": smoke_adapter.task_id,
                "checked_reward_terms": 0,
                "checked_diagnostics": 0,
            }
        )
    candidate_dir = _candidate_dir(candidate)
    if candidate_dir is None:
        return _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "skip",
                "reason": "candidate_package_source_not_local",
                "task_id": fixture.task_id,
                "checked_reward_terms": 0,
                "checked_diagnostics": 0,
                "fixture_summary": dict(fixture.public_summary),
            }
        )
    return _run_component_smoke_with_timeout(
        candidate_dir,
        fixture,
        timeout_s=timeout_s,
        worker_startup_timeout_s=worker_startup_timeout_s,
        load_timeout_s=load_timeout_s,
    )


def _run_component_smoke_with_timeout(
    candidate_dir: Path,
    fixture: ComponentSmokeFixture,
    *,
    timeout_s: float,
    worker_startup_timeout_s: float,
    load_timeout_s: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    progress_queue = ctx.Queue()
    worker_ready = ctx.Event()
    worker_start = ctx.Event()
    process = ctx.Process(
        target=_component_smoke_worker,
        args=(
            str(candidate_dir),
            fixture,
            result_queue,
            progress_queue,
            worker_ready,
            worker_start,
        ),
    )
    process.daemon = True
    process.start()
    try:
        startup_deadline = time.monotonic() + worker_startup_timeout_s
        while process.is_alive() and not worker_ready.is_set():
            remaining = startup_deadline - time.monotonic()
            if remaining <= 0:
                break
            worker_ready.wait(timeout=min(0.05, remaining))
        if not worker_ready.is_set():
            if process.is_alive():
                raise IsolatedWorkerStartupError(
                    "component smoke worker startup timed out"
                )
            process.join()
            raise IsolatedWorkerProtocolError(
                "component smoke worker exited before declaring readiness"
            )

        load_timeout_s = max(load_timeout_s, timeout_s)
        latest_progress: Mapping[str, Any] = {"phase": "component_load"}
        deadline = time.monotonic() + load_timeout_s
        active_timeout_s = load_timeout_s
        result_received = False
        result: Any = None
        worker_start.set()
        while process.is_alive():
            try:
                result = result_queue.get_nowait()
            except queue_module.Empty:
                pass
            else:
                result_received = True
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                queued_progress = _latest_queue_item(progress_queue)
                if queued_progress:
                    latest_progress = queued_progress
                    if latest_progress.get("phase") in {"reward", "diagnostic"}:
                        active_timeout_s = timeout_s
                        deadline = time.monotonic() + timeout_s
                        continue
                break
            try:
                progress = progress_queue.get(timeout=min(0.05, remaining))
            except queue_module.Empty:
                continue
            if not isinstance(progress, Mapping):
                continue
            latest_progress = dict(progress)
            if latest_progress.get("phase") in {"reward", "diagnostic"}:
                active_timeout_s = timeout_s
                deadline = time.monotonic() + timeout_s
        if result_received:
            # Drain the result before requiring worker exit. A multiprocessing
            # Queue feeder can otherwise block worker teardown on pipe backpressure.
            process.join(1.0)
        elif process.is_alive():
            return _component_smoke_timeout_result(
                fixture=fixture,
                latest_progress=latest_progress,
                timeout_s=active_timeout_s,
            )
        else:
            process.join()
            try:
                result = result_queue.get(timeout=0.25)
            except queue_module.Empty:
                return _public_result(
                    {
                        "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                        "status": "fail",
                        "reason": "component_smoke_worker_failed",
                        "task_id": fixture.task_id,
                        "checked_reward_terms": 0,
                        "checked_diagnostics": 0,
                        "fixture_summary": dict(fixture.public_summary),
                        "error_type": "WorkerExit",
                    }
                )
        if isinstance(result, Mapping):
            return _public_result(dict(result))
        return _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "fail",
                "reason": "component_smoke_worker_failed",
                "task_id": fixture.task_id,
                "checked_reward_terms": 0,
                "checked_diagnostics": 0,
                "fixture_summary": dict(fixture.public_summary),
                "error_type": "WorkerResultInvalid",
            }
        )
    finally:
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)


def _component_smoke_worker(
    candidate_dir: str,
    fixture: ComponentSmokeFixture,
    result_queue: Any,
    progress_queue: Any,
    worker_ready: Any,
    worker_start: Any,
) -> None:
    # Spawn has already restored the task fixture, including Tensor imports.
    # Only candidate loading and callable execution enter the shorter budgets.
    worker_ready.set()
    worker_start.wait()
    try:
        _put_progress(progress_queue, {"phase": "component_load"})
        components = load_training_components(Path(candidate_dir))
    except Exception as exc:  # noqa: BLE001 - public-safe conversion happens below.
        result_queue.put(
            _public_result(
                {
                    "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                    "status": "fail",
                    "reason": "component_load_failed",
                    "task_id": fixture.task_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "checked_reward_terms": 0,
                    "checked_diagnostics": 0,
                    "fixture_summary": dict(fixture.public_summary),
                }
            )
        )
        return

    reward_results: list[dict[str, Any]] = []
    for term in components.reward_terms:
        _put_progress(progress_queue, {"phase": "reward", "name": term.name, "kind": "RewardTerm"})
        reward_result = _smoke_reward_term(term, fixture)
        reward_results.append(reward_result)
        if reward_result["status"] == "fail":
            result_queue.put(
                _public_result(
                    {
                        "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                        "status": "fail",
                        "reason": reward_result["reason"],
                        "task_id": fixture.task_id,
                        "checked_reward_terms": len(reward_results),
                        "checked_diagnostics": 0,
                        "fixture_summary": dict(fixture.public_summary),
                        "reward_results": reward_results,
                    }
                )
            )
            return

    diagnostic_results: list[dict[str, Any]] = []
    for term in components.diagnostics:
        _put_progress(progress_queue, {"phase": "diagnostic", "name": term.name, "kind": "DiagnosticTerm"})
        diagnostic_result = _smoke_diagnostic_term(term, fixture)
        diagnostic_results.append(diagnostic_result)
        if diagnostic_result["status"] == "fail":
            result_queue.put(
                _public_result(
                    {
                        "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                        "status": "fail",
                        "reason": diagnostic_result["reason"],
                        "task_id": fixture.task_id,
                        "checked_reward_terms": len(reward_results),
                        "checked_diagnostics": len(diagnostic_results),
                        "fixture_summary": dict(fixture.public_summary),
                        "reward_results": reward_results,
                        "diagnostic_results": diagnostic_results,
                    }
                )
            )
            return

    advisory_rewards = [
        item for item in reward_results if item["status"] == "advisory"
    ]
    result_queue.put(
        _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "advisory" if advisory_rewards else "pass",
                "reason": (
                    "reward_term_suppressed_by_declared_clip_and_weight"
                    if advisory_rewards
                    else "component_smoke_passed"
                ),
                "task_id": fixture.task_id,
                "checked_reward_terms": len(reward_results),
                "checked_diagnostics": len(diagnostic_results),
                "fixture_summary": dict(fixture.public_summary),
                "reward_results": reward_results,
                "diagnostic_results": diagnostic_results,
                "advisory_reward_term_names": [
                    str(item["name"]) for item in advisory_rewards
                ],
            }
        )
    )


def _put_progress(progress_queue: Any, item: Mapping[str, Any]) -> None:
    try:
        progress_queue.put_nowait(dict(item))
    except Exception:
        return


def _latest_queue_item(items: Any) -> Mapping[str, Any]:
    latest: Mapping[str, Any] = {}
    while True:
        try:
            item = items.get_nowait()
        except queue_module.Empty:
            return latest
        if isinstance(item, Mapping):
            latest = dict(item)


def _component_smoke_timeout_result(
    *,
    fixture: ComponentSmokeFixture,
    latest_progress: Mapping[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    phase = latest_progress.get("phase")
    if phase == "reward":
        reward = {
            "name": latest_progress.get("name", "unknown_reward_term"),
            "kind": "RewardTerm",
            "status": "fail",
            "reason": "reward_callable_timeout",
            "timeout_s": timeout_s,
        }
        return _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "fail",
                "reason": "reward_callable_timeout",
                "task_id": fixture.task_id,
                "checked_reward_terms": 1,
                "checked_diagnostics": 0,
                "fixture_summary": dict(fixture.public_summary),
                "reward_results": [reward],
            }
        )
    if phase == "diagnostic":
        diagnostic = {
            "name": latest_progress.get("name", "unknown_diagnostic_term"),
            "kind": "DiagnosticTerm",
            "status": "fail",
            "reason": "diagnostic_callable_timeout",
            "timeout_s": timeout_s,
        }
        return _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "fail",
                "reason": "diagnostic_callable_timeout",
                "task_id": fixture.task_id,
                "checked_reward_terms": 0,
                "checked_diagnostics": 1,
                "fixture_summary": dict(fixture.public_summary),
                "diagnostic_results": [diagnostic],
            }
        )
    return _public_result(
        {
            "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
            "status": "fail",
            "reason": "component_load_timeout",
            "task_id": fixture.task_id,
            "checked_reward_terms": 0,
            "checked_diagnostics": 0,
            "fixture_summary": dict(fixture.public_summary),
            "timeout_s": timeout_s,
        }
    )


def _positive_finite_timeout(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{field_name} must be a positive finite number")
    return float(value)


def _smoke_reward_term(term: RewardTerm, fixture: ComponentSmokeFixture) -> dict[str, Any]:
    try:
        value = term.fn(fixture.reward_inputs)
    except Exception as exc:  # noqa: BLE001 - converted to public-safe feedback.
        return _public_result(
            {
                "name": term.name,
                "kind": "RewardTerm",
                "status": "fail",
                "reason": "reward_callable_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    try:
        output = _output_summary(value)
    except Exception as exc:  # noqa: BLE001 - converted to public-safe feedback.
        return _public_result(
            {
                "name": term.name,
                "kind": "RewardTerm",
                "status": "fail",
                "reason": "reward_output_unsupported",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    if output["shape"] != list(fixture.expected_reward_output_shape):
        return _public_result(
            {
                "name": term.name,
                "kind": "RewardTerm",
                "status": "fail",
                "reason": "reward_output_wrong_shape",
                "expected_shape": list(fixture.expected_reward_output_shape),
                "observed_shape": output["shape"],
            }
        )
    if not output["finite"]:
        return _public_result(
            {
                "name": term.name,
                "kind": "RewardTerm",
                "status": "fail",
                "reason": "reward_output_nonfinite",
                "observed_shape": output["shape"],
            }
        )
    effective_max_abs = _declared_effective_max_abs(
        output["values"],
        term=term,
    )
    if not math.isfinite(effective_max_abs):
        return _public_result(
            {
                "name": term.name,
                "kind": "RewardTerm",
                "status": "fail",
                "reason": "reward_effective_output_nonfinite",
                "observed_shape": output["shape"],
            }
        )
    raw_values = output["values"]
    observations = {
        "observed_sample_count": len(raw_values),
        "observed_raw_nonzero_count": sum(abs(value) > 1.0e-12 for value in raw_values),
        "observed_clip_saturation_count": sum(
            (term.clip_min is not None and value < term.clip_min)
            or (term.clip_max is not None and value > term.clip_max)
            for value in raw_values),
        "coverage": "finite_fixture_only_not_a_proof_of_universal_effect",
    }
    cases = fixture.public_summary.get("sample_cases", ())
    if len(cases) == len(raw_values) and len(cases) <= 8:
        observations["sample_case_outputs"] = [
            {"case": case, "raw": raw, "weighted_after_clip": (
                max(term.clip_min if term.clip_min is not None else -math.inf,
                    min(term.clip_max if term.clip_max is not None else math.inf, raw)) * float(term.weight))}
            for case, raw in zip(cases, raw_values, strict=True)]
    # A zero raw value can be valid on one synthetic fixture (for example, a
    # tracking error at its target). Advise only when declared runtime
    # semantics erase a signal the callable actually produced.
    if output["max_abs"] > 0.0 and effective_max_abs == 0.0:
        return _public_result(
            {
                "name": term.name,
                "kind": "RewardTerm",
                "status": "advisory",
                "reason": "reward_term_suppressed_by_declared_clip_and_weight",
                "observed_shape": output["shape"],
                "raw_max_abs": output["max_abs"],
                "effective_max_abs_after_clip_and_weight": effective_max_abs,
                "declared_weight": float(term.weight),
                "declared_clip_min": term.clip_min,
                "declared_clip_max": term.clip_max,
                "blocking": False,
                **observations,
            }
        )
    return _public_result(
        {
            "name": term.name,
            "kind": "RewardTerm",
            "status": "pass",
            "reason": "reward_callable_returned_finite_expected_shape",
            "observed_shape": output["shape"],
            "raw_max_abs": output["max_abs"],
            "effective_max_abs_after_clip_and_weight": effective_max_abs,
            **observations,
        }
    )


def _smoke_diagnostic_term(term: DiagnosticTerm, fixture: ComponentSmokeFixture) -> dict[str, Any]:
    try:
        term.fn(fixture.reward_inputs)
    except Exception as exc:  # noqa: BLE001 - converted to public-safe repair feedback.
        return _public_result(
            {
                "name": term.name,
                "kind": "DiagnosticTerm",
                "status": "fail",
                "reason": "diagnostic_callable_failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
    return _public_result(
        {
            "name": term.name,
            "kind": "DiagnosticTerm",
            "status": "pass",
            "reason": "diagnostic_callable_executed",
        }
    )


def _output_summary(value: Any) -> dict[str, Any]:
    shape, flattened = _shape_and_values(value)
    return {
        "shape": list(shape),
        "finite": all(math.isfinite(item) for item in flattened),
        "max_abs": max((abs(item) for item in flattened), default=0.0),
        "values": flattened,
    }


def _declared_effective_max_abs(
    values: tuple[float, ...],
    *,
    term: RewardTerm,
) -> float:
    weighted: list[float] = []
    for raw_value in values:
        value = raw_value
        if term.clip_min is not None and value < term.clip_min:
            value = float(term.clip_min)
        if term.clip_max is not None and value > term.clip_max:
            value = float(term.clip_max)
        weighted.append(value * float(term.weight))
    return max((abs(value) for value in weighted), default=0.0)


def _shape_and_values(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if _is_torch_tensor(value):
        detached = value.detach()
        if str(detached.device) != "cpu":
            detached = detached.cpu()
        return (
            tuple(int(item) for item in detached.shape),
            tuple(float(item) for item in detached.reshape(-1).tolist()),
        )
    if isinstance(value, int | float):
        return (), (float(value),)
    if isinstance(value, list | tuple):
        if not value:
            return (0,), ()
        if all(isinstance(item, int | float) for item in value):
            return (len(value),), tuple(float(item) for item in value)
        if all(isinstance(row, list | tuple) for row in value):
            rows = list(value)
            cols = len(rows[0]) if rows else 0
            flattened: list[float] = []
            for row in rows:
                if len(row) != cols or not all(isinstance(item, int | float) for item in row):
                    raise TypeError("nested reward output must be rectangular numeric values")
                flattened.extend(float(item) for item in row)
            return (len(rows), cols), tuple(flattened)
    raise TypeError(f"unsupported reward output type: {type(value).__name__}")


def _candidate_dir(
    candidate: CandidatePackageRef | SealedCandidateArtifactRef,
) -> Path | None:
    if "://" in candidate.path:
        return None
    path = Path(candidate.path)
    if not path.is_dir():
        return None
    return path


def _is_torch_tensor(value: Any) -> bool:
    value_type = type(value)
    return (
        value_type.__module__.startswith("torch")
        and hasattr(value, "shape")
        and callable(getattr(value, "detach", None))
        and callable(getattr(value, "reshape", None))
    )
