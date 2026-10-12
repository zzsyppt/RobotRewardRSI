"""Production Stair rollout loop consuming only runner-owned recorder evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable

from true_rl_agent.tasks.stair_climbing.episode_evidence import (
    StairEpisodeOutcome,
    StairEpisodeTracker,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_evaluation_summary,
)
from true_rl_agent.tasks.stair_climbing.evaluation_semantics import (
    StairEvaluationSemantics,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_TRANSITION_EXTRAS_KEY,
)
from true_rl_agent.tasks.stair_climbing.runtime_projection import (
    StairRuntimeProjection,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    StairEvaluationSeedResolver,
    StairOrderedEvaluationSeedResolver,
    StairPrivateSeedResolver,
)
from true_rl_agent.tasks.stair_climbing.transition_decoder import (
    STAIR_TRANSITION_HISTORY_CHUNK_STEPS,
    decode_stair_transition_batch,
    materialize_stair_transition_history,
)


class StairEvaluationRuntimeError(RuntimeError):
    """Raised when the fixed evaluation rollout cannot be completed."""


@dataclass(frozen=True)
class StairSplitRolloutExecution:
    """One split's outcomes plus control steps observed at the env boundary."""

    per_seed_outcomes: tuple[tuple[StairEpisodeOutcome, ...], ...]
    vector_env_step_calls: int

    def __post_init__(self) -> None:
        if not self.per_seed_outcomes or any(
            not seed_outcomes for seed_outcomes in self.per_seed_outcomes
        ):
            raise ValueError("Stair split execution outcomes cannot be empty")
        env_counts = {len(seed_outcomes) for seed_outcomes in self.per_seed_outcomes}
        if len(env_counts) != 1:
            raise ValueError("Stair split execution env count must be fixed across seeds")
        if (
            isinstance(self.vector_env_step_calls, bool)
            or not isinstance(self.vector_env_step_calls, int)
            or self.vector_env_step_calls <= 0
        ):
            raise ValueError("Stair split vector env step calls must be positive")

    @property
    def seed_count(self) -> int:
        return len(self.per_seed_outcomes)

    @property
    def evaluation_env_count_per_seed(self) -> int:
        return len(self.per_seed_outcomes[0])

    @property
    def simulated_env_control_steps(self) -> int:
        return self.vector_env_step_calls * self.evaluation_env_count_per_seed

    @property
    def closed_first_episode_control_steps(self) -> int:
        return sum(
            outcome.episode_steps
            for seed_outcomes in self.per_seed_outcomes
            for outcome in seed_outcomes
        )


@dataclass(frozen=True)
class _StairSeedRolloutExecution:
    outcomes: tuple[StairEpisodeOutcome, ...]
    vector_env_step_calls: int


def run_all_stair_splits(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1,
    semantics: StairEvaluationSemantics,
    seed_resolver: StairPrivateSeedResolver | StairEvaluationSeedResolver,
    seed_namespace: str,
) -> dict[str, dict[str, Any]]:
    """Run every fixed split and return public-safe aggregate summaries."""

    if isinstance(seed_resolver, StairOrderedEvaluationSeedResolver):
        raise TypeError(
            "ordered Stair splits require split-local environments; use phased evaluation"
        )

    split_counts = {
        "public_dev": seed_resolver.evaluation_protocol.public_seed_count,
        "public_accept": seed_resolver.evaluation_protocol.public_accept_seed_count,
        "sealed_validation": (
            seed_resolver.evaluation_protocol.sealed_validation_seed_count
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for split, seed_count in split_counts.items():
        per_seed_outcomes = run_stair_split_outcomes(
            env=env,
            inference_policy=inference_policy,
            protocol=protocol,
            semantics=semantics,
            seed_resolver=seed_resolver,
            split=split,
            seed_count=seed_count,
            seed_namespace=seed_namespace,
        )
        result[split] = build_stair_evaluation_summary(
            protocol,
            semantics=semantics,
            evaluation_protocol=seed_resolver.evaluation_protocol,
            split=split,
            per_seed_outcomes=per_seed_outcomes,
            evaluation_profile=(
                seed_resolver.evaluation_profile
                if isinstance(seed_resolver, StairEvaluationSeedResolver)
                else None
            ),
        )
    return result


def run_stair_split_outcomes(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1 | None,
    semantics: StairEvaluationSemantics,
    seed_resolver: (
        StairPrivateSeedResolver
        | StairEvaluationSeedResolver
        | StairOrderedEvaluationSeedResolver
    ),
    split: str,
    seed_count: int,
    seed_namespace: str,
    progress_callback: Callable[[int, int], None] | None = None,
    runtime_projection: StairRuntimeProjection | None = None,
) -> list[list[StairEpisodeOutcome]]:
    """Collect one closed outcome cohort for an explicit seed domain."""

    execution = run_stair_split_execution(
        env=env,
        inference_policy=inference_policy,
        protocol=protocol,
        semantics=semantics,
        seed_resolver=seed_resolver,
        split=split,
        seed_count=seed_count,
        seed_namespace=seed_namespace,
        progress_callback=progress_callback,
        runtime_projection=runtime_projection,
    )
    return [list(seed_outcomes) for seed_outcomes in execution.per_seed_outcomes]


def run_stair_split_execution(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1 | None,
    semantics: StairEvaluationSemantics,
    seed_resolver: (
        StairPrivateSeedResolver
        | StairEvaluationSeedResolver
        | StairOrderedEvaluationSeedResolver
    ),
    split: str,
    seed_count: int,
    seed_namespace: str,
    progress_callback: Callable[[int, int], None] | None = None,
    runtime_projection: StairRuntimeProjection | None = None,
    on_seed_reset: Callable[[int, int], None] | None = None,
    on_seed_completed: Callable[[tuple[StairEpisodeOutcome, ...]], None] | None = None,
    numeric_capture: Any | None = None,
) -> StairSplitRolloutExecution:
    """Collect outcomes and exact runner-owned control-step cost for one split."""

    if isinstance(seed_count, bool) or not isinstance(seed_count, int) or seed_count <= 0:
        raise ValueError("Stair evaluation seed_count must be a positive integer")
    outcomes: list[tuple[StairEpisodeOutcome, ...]] = []
    vector_env_step_calls = 0
    for run_index in range(1, seed_count + 1):
        seed_execution = _execute_stair_seed_rollout(
            env=env,
            inference_policy=inference_policy,
            protocol=protocol,
            semantics=semantics,
            seed_resolver=seed_resolver,
            split=split,
            run_index=run_index,
            seed_namespace=seed_namespace,
            runtime_projection=runtime_projection,
            on_seed_reset=on_seed_reset,
            numeric_capture=numeric_capture,
        )
        outcomes.append(seed_execution.outcomes)
        vector_env_step_calls += seed_execution.vector_env_step_calls
        if on_seed_completed is not None:
            on_seed_completed(seed_execution.outcomes)
        if progress_callback is not None:
            progress_callback(run_index, seed_count)
    return StairSplitRolloutExecution(
        per_seed_outcomes=tuple(outcomes),
        vector_env_step_calls=vector_env_step_calls,
    )


def run_stair_seed_rollout(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1 | None,
    semantics: StairEvaluationSemantics,
    seed_resolver: (
        StairPrivateSeedResolver
        | StairEvaluationSeedResolver
        | StairOrderedEvaluationSeedResolver
    ),
    split: str,
    run_index: int,
    seed_namespace: str,
    runtime_projection: StairRuntimeProjection | None = None,
) -> list[StairEpisodeOutcome]:
    """Collect exactly the first complete episode for every evaluation env."""

    execution = _execute_stair_seed_rollout(
        env=env,
        inference_policy=inference_policy,
        protocol=protocol,
        semantics=semantics,
        seed_resolver=seed_resolver,
        split=split,
        run_index=run_index,
        seed_namespace=seed_namespace,
        runtime_projection=runtime_projection,
    )
    return list(execution.outcomes)


def _execute_stair_seed_rollout(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1 | None,
    semantics: StairEvaluationSemantics,
    seed_resolver: (
        StairPrivateSeedResolver
        | StairEvaluationSeedResolver
        | StairOrderedEvaluationSeedResolver
    ),
    split: str,
    run_index: int,
    seed_namespace: str,
    runtime_projection: StairRuntimeProjection | None = None,
    on_seed_reset: Callable[[int, int], None] | None = None,
    numeric_capture: Any | None = None,
) -> _StairSeedRolloutExecution:
    """Collect one seed and retain the exact number of env.step invocations."""

    if not callable(inference_policy):
        raise TypeError("Stair evaluation inference_policy must be callable")
    runner_seed = seed_resolver.derive_seed(
        split=split,
        run_index=run_index,
        namespace=seed_namespace,
    )
    _set_runner_seed(runner_seed)
    observation, _ = _reset_env(env, runner_seed)
    _assert_finite_tree(observation, "Stair reset observation")
    if on_seed_reset is not None:
        on_seed_reset(run_index, runner_seed)

    if isinstance(seed_resolver, StairOrderedEvaluationSeedResolver):
        assignment = seed_resolver.private_bin_assignment(split)
        cell_assignment = seed_resolver.private_cell_assignment(split)
    else:
        assignment = seed_resolver.private_bin_assignment()
        cell_assignment = (
            seed_resolver.private_cell_assignment()
            if isinstance(seed_resolver, StairEvaluationSeedResolver)
            else None
        )
    base_env = getattr(env, "unwrapped", env)
    step_dt = float(getattr(base_env, "step_dt", 0.0))
    if isinstance(seed_resolver, StairEvaluationSeedResolver):
        horizon_s = seed_resolver.evaluation_profile.max_episode_horizon_s
    elif isinstance(seed_resolver, StairOrderedEvaluationSeedResolver):
        horizon_s = max(
            cell.episode_horizon_s
            for cell in seed_resolver.evaluation_suite.runtime_cells(split)
        )
    else:
        if protocol is None:
            raise TypeError("legacy Stair evaluation requires StairProtocolV1")
        horizon_s = float(protocol.to_dict()["evaluation"]["episode_length_s"])
    if step_dt <= 0.0:
        raise StairEvaluationRuntimeError("Stair environment step_dt must be positive")
    max_steps_float = horizon_s / step_dt
    max_steps = int(round(max_steps_float))
    if abs(max_steps_float - max_steps) > 1.0e-7 or max_steps <= 0:
        raise StairEvaluationRuntimeError(
            "Stair episode horizon is not an integral control-step count"
        )

    trackers = [
        StairEpisodeTracker(protocol, semantics, env_index=index, bin_id=bin_id,
                            evaluation_cell=cell_assignment[index] if cell_assignment is not None else None)
        for index, bin_id in enumerate(assignment)
    ]
    outcomes: list[StairEpisodeOutcome | None] = [None] * len(trackers)
    active = [True] * len(trackers)
    if numeric_capture is not None:
        numeric_capture.begin_seed(run_index=run_index, runner_seed=runner_seed, split=split, step_dt=step_dt)

    def consume_chunk(host_steps: tuple[Mapping[str, Any], ...], done_steps: tuple[Any, ...]) -> None:
        valid_steps: list[list[bool]] = []
        for raw_transition, done_mask in zip(host_steps, done_steps, strict=True):
            if not any(active):
                break
            valid_steps.append(list(active))
            transitions = decode_stair_transition_batch(
                raw_transition, wrapper_done_mask=done_mask, protocol=protocol, semantics=semantics,
                expected_bin_assignment=assignment, expected_cell_assignment=cell_assignment,
                runtime_projection=runtime_projection,
            )
            for env_index, transition in enumerate(transitions):
                if active[env_index]:
                    outcome = trackers[env_index].observe(transition)
                    if outcome is not None:
                        outcomes[env_index], active[env_index] = outcome, False
        if numeric_capture is not None and valid_steps:
            numeric_capture.record_chunk(host_steps[:len(valid_steps)], done_steps[:len(valid_steps)], valid_steps)

    raw_history: list[Mapping[str, Any]] = []
    wrapper_done_history: list[Any] = []
    seen_done = None
    finite_status = None
    vector_env_step_calls = 0
    for step_index in range(max_steps):
        actions = _invoke_inference_policy(inference_policy, observation)
        finite_status = _accumulate_finite_tree(actions, finite_status)
        observation, reward, done_mask, extras = _step_env(env, actions)
        vector_env_step_calls += 1
        finite_status = _accumulate_finite_tree(observation, finite_status)
        finite_status = _accumulate_finite_tree(reward, finite_status)
        if not isinstance(extras, Mapping):
            raise StairEvaluationRuntimeError("Stair env extras must be an object")
        raw_transition = extras.get(STAIR_EVAL_TRANSITION_EXTRAS_KEY)
        if not isinstance(raw_transition, Mapping):
            raise StairEvaluationRuntimeError(
                "Stair production recorder did not publish pre-reset evidence"
            )
        done_snapshot = _clone_done_mask(done_mask)
        raw_history.append(dict(raw_transition))
        wrapper_done_history.append(done_snapshot)
        if len(raw_history) >= STAIR_TRANSITION_HISTORY_CHUNK_STEPS:
            host_chunk, done_chunk = _materialize_history_chunk(
                raw_history,
                wrapper_done_history,
            )
            consume_chunk(host_chunk, done_chunk)
        seen_done = (
            done_snapshot
            if seen_done is None
            else seen_done | done_snapshot
        )
        done_poll_interval = (
            STAIR_TRANSITION_HISTORY_CHUNK_STEPS
            if _is_cuda_tensor(done_snapshot)
            else 1
        )
        should_poll_done = (
            (step_index + 1) % done_poll_interval == 0
            or step_index + 1 == max_steps
        )
        if should_poll_done and _all_true(
            seen_done,
            "Stair cumulative done mask",
        ):
            break
    _assert_accumulated_finite(finite_status)
    host_chunk, done_chunk = _materialize_history_chunk(
        raw_history,
        wrapper_done_history,
    )
    consume_chunk(host_chunk, done_chunk)

    if any(active) or any(outcome is None for outcome in outcomes):
        raise StairEvaluationRuntimeError(
            "Stair first-episode coverage did not close within the fixed horizon"
        )
    if numeric_capture is not None:
        numeric_capture.finish_seed(outcomes)
    return _StairSeedRolloutExecution(
        outcomes=tuple(outcome for outcome in outcomes if outcome is not None),
        vector_env_step_calls=vector_env_step_calls,
    )


def _materialize_history_chunk(
    raw_history: list[Mapping[str, Any]],
    wrapper_done_history: list[Any],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Any, ...]]:
    if not raw_history:
        return (), ()
    host_history, host_wrapper_done = materialize_stair_transition_history(
        raw_history,
        wrapper_done_masks=wrapper_done_history,
    )
    raw_history.clear()
    wrapper_done_history.clear()
    return host_history, host_wrapper_done


def _invoke_inference_policy(
    inference_policy: Callable[[Any], Any],
    observation: Any,
) -> Any:
    try:
        import torch
    except ImportError:
        return inference_policy(observation)
    with torch.inference_mode():
        return inference_policy(observation)


def _clone_done_mask(value: Any) -> Any:
    try:
        return value.detach().clone()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise StairEvaluationRuntimeError(
            "Stair wrapper done mask cannot be snapshotted"
        ) from exc


def _all_true(value: Any, context: str) -> bool:
    try:
        result = value.all()
        item = getattr(result, "item", None)
        return bool(item() if callable(item) else result)
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        raise StairEvaluationRuntimeError(
            f"{context} does not support exact reduction"
        ) from exc


def _is_cuda_tensor(value: Any) -> bool:
    device = getattr(value, "device", None)
    return (
        isinstance(device, str) and device.startswith("cuda")
    ) or str(getattr(device, "type", "")) == "cuda"


def _set_runner_seed(seed: int) -> None:
    try:
        import torch
    except ImportError as exc:
        raise StairEvaluationRuntimeError("Stair evaluation requires torch") from exc
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reset_env(env: Any, seed: int) -> tuple[Any, Mapping[str, Any]]:
    try:
        result = env.reset(seed=seed)
    except TypeError:
        _seed_environment(env, seed)
        result = env.reset()
    if isinstance(result, tuple) and len(result) == 2:
        observation, extras = result
        if not isinstance(extras, Mapping):
            raise StairEvaluationRuntimeError("Stair reset extras must be an object")
        return observation, extras
    return result, {}


def _seed_environment(env: Any, seed: int) -> None:
    for target in (env, getattr(env, "unwrapped", None)):
        seed_method = getattr(target, "seed", None)
        if not callable(seed_method):
            continue
        try:
            seed_method(seed)
        except TypeError:
            seed_method(seed=seed)
        return
    raise StairEvaluationRuntimeError("Stair environment has no seed API")


def _step_env(env: Any, actions: Any) -> tuple[Any, Any, Any, Mapping[str, Any]]:
    result = env.step(actions)
    if not isinstance(result, tuple):
        raise StairEvaluationRuntimeError("Stair env.step must return a tuple")
    if len(result) == 4:
        observation, reward, done, extras = result
        return observation, reward, done.bool(), extras
    if len(result) == 5:
        observation, reward, terminated, truncated, extras = result
        return observation, reward, terminated.bool() | truncated.bool(), extras
    raise StairEvaluationRuntimeError("Stair env.step return arity is unsupported")


def _assert_finite_tree(value: Any, context: str) -> None:
    try:
        import torch
    except ImportError as exc:
        raise StairEvaluationRuntimeError("Stair evaluation requires torch") from exc
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all().item()):
            raise StairEvaluationRuntimeError(f"{context} contains NaN/Inf")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_finite_tree(item, context)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_finite_tree(item, context)


def _accumulate_finite_tree(value: Any, current: Any) -> Any:
    try:
        import torch
    except ImportError:
        return current
    if isinstance(value, torch.Tensor):
        status = torch.isfinite(value).all()
        return status if current is None else current & status
    if isinstance(value, Mapping):
        for item in value.values():
            current = _accumulate_finite_tree(item, current)
        return current
    if isinstance(value, (tuple, list)):
        for item in value:
            current = _accumulate_finite_tree(item, current)
    return current


def _assert_accumulated_finite(value: Any) -> None:
    if value is None:
        return
    try:
        if not bool(value.item()):
            raise StairEvaluationRuntimeError(
                "Stair rollout actions, observations, or rewards contain NaN/Inf"
            )
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        if isinstance(exc, StairEvaluationRuntimeError):
            raise
        raise StairEvaluationRuntimeError(
            "Stair rollout finite status cannot be materialized"
        ) from exc


__all__ = [
    "StairEvaluationRuntimeError",
    "StairSplitRolloutExecution",
    "run_all_stair_splits",
    "run_stair_split_execution",
    "run_stair_split_outcomes",
    "run_stair_seed_rollout",
]
