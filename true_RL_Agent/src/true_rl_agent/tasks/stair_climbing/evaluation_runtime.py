"""Production Stair rollout loop consuming only runner-owned recorder evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from true_rl_agent.tasks.stair_climbing.episode_evidence import (
    StairEpisodeOutcome,
    StairEpisodeTracker,
)
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    build_stair_split_summary,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.runtime_profile import (
    STAIR_EVAL_TRANSITION_EXTRAS_KEY,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import StairPrivateSeedResolver
from true_rl_agent.tasks.stair_climbing.transition_decoder import (
    decode_stair_transition_batch,
)


class StairEvaluationRuntimeError(RuntimeError):
    """Raised when the fixed evaluation rollout cannot be completed."""


def run_all_stair_splits(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1,
    seed_resolver: StairPrivateSeedResolver,
) -> dict[str, dict[str, Any]]:
    """Run every fixed split and return public-safe aggregate summaries."""

    split_counts = {
        "public_dev": seed_resolver.evaluation_protocol.public_seed_count,
        "public_accept": seed_resolver.evaluation_protocol.public_accept_seed_count,
        "sealed_validation": (
            seed_resolver.evaluation_protocol.sealed_validation_seed_count
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for split, seed_count in split_counts.items():
        per_seed_outcomes = [
            run_stair_seed_rollout(
                env=env,
                inference_policy=inference_policy,
                protocol=protocol,
                seed_resolver=seed_resolver,
                split=split,
                run_index=run_index,
            )
            for run_index in range(1, seed_count + 1)
        ]
        result[split] = build_stair_split_summary(
            protocol,
            evaluation_protocol=seed_resolver.evaluation_protocol,
            split=split,
            per_seed_outcomes=per_seed_outcomes,
        )
    return result


def run_stair_seed_rollout(
    *,
    env: Any,
    inference_policy: Callable[[Any], Any],
    protocol: StairProtocolV1,
    seed_resolver: StairPrivateSeedResolver,
    split: str,
    run_index: int,
) -> list[StairEpisodeOutcome]:
    """Collect exactly the first complete episode for every evaluation env."""

    if not callable(inference_policy):
        raise TypeError("Stair evaluation inference_policy must be callable")
    runner_seed = seed_resolver.derive_seed(split=split, run_index=run_index)
    _set_runner_seed(runner_seed)
    observation, _ = _reset_env(env, runner_seed)
    _assert_finite_tree(observation, "Stair reset observation")

    assignment = seed_resolver.private_bin_assignment()
    trackers = [
        StairEpisodeTracker(protocol, env_index=index, bin_id=bin_id)
        for index, bin_id in enumerate(assignment)
    ]
    outcomes: list[StairEpisodeOutcome | None] = [None] * len(trackers)
    active = [True] * len(trackers)
    base_env = getattr(env, "unwrapped", env)
    step_dt = float(getattr(base_env, "step_dt", 0.0))
    horizon_s = float(protocol.to_dict()["evaluation"]["episode_length_s"])
    if step_dt <= 0.0:
        raise StairEvaluationRuntimeError("Stair environment step_dt must be positive")
    max_steps_float = horizon_s / step_dt
    max_steps = int(round(max_steps_float))
    if abs(max_steps_float - max_steps) > 1.0e-7 or max_steps <= 0:
        raise StairEvaluationRuntimeError(
            "Stair episode horizon is not an integral control-step count"
        )

    for _ in range(max_steps):
        actions = inference_policy(observation)
        _assert_finite_tree(actions, "Stair policy actions")
        observation, reward, done_mask, extras = _step_env(env, actions)
        _assert_finite_tree(observation, "Stair step observation")
        _assert_finite_tree(reward, "Stair step reward")
        if not isinstance(extras, Mapping):
            raise StairEvaluationRuntimeError("Stair env extras must be an object")
        raw_transition = extras.get(STAIR_EVAL_TRANSITION_EXTRAS_KEY)
        if not isinstance(raw_transition, Mapping):
            raise StairEvaluationRuntimeError(
                "Stair production recorder did not publish pre-reset evidence"
            )
        transitions = decode_stair_transition_batch(
            raw_transition,
            wrapper_done_mask=done_mask,
            protocol=protocol,
            expected_bin_assignment=assignment,
        )
        for env_index, transition in enumerate(transitions):
            if not active[env_index]:
                continue
            outcome = trackers[env_index].observe(transition)
            if outcome is not None:
                outcomes[env_index] = outcome
                active[env_index] = False
        if not any(active):
            break

    if any(active) or any(outcome is None for outcome in outcomes):
        raise StairEvaluationRuntimeError(
            "Stair first-episode coverage did not close within the fixed horizon"
        )
    return [outcome for outcome in outcomes if outcome is not None]


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


__all__ = [
    "StairEvaluationRuntimeError",
    "run_all_stair_splits",
    "run_stair_seed_rollout",
]
