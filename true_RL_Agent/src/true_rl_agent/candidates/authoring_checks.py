"""Authoring-time static checks for candidate packages."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import multiprocessing as mp
import queue as queue_module
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from true_rl_agent.agent_core.context import sanitize_prompt_public_context
from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm
from true_rl_agent.candidates.loader import load_training_components
from true_rl_agent.candidates.schema import CandidatePackageRef
from true_rl_agent.prompting.public_safety import assert_public_only_prompt_payload
from true_rl_agent.prompting.source_catalog import (
    CURRENT_BEST_SOURCE_ID,
    LATEST_FAILED_SOURCE_ID,
)
from true_rl_agent.tasks.component_smoke import (
    ComponentSmokeAdapter,
    ComponentSmokeFixture,
    SmokeTensor,
)

COMPONENT_SMOKE_SCHEMA_VERSION = "true_rl_agent.component_smoke_result.v1"
ANTI_DEGENERATE_SCHEMA_VERSION = "true_rl_agent.anti_degenerate_assessment.v1"
STATIC_AUTHORING_FAILURE_SCHEMA_VERSION = "true_rl_agent.static_authoring_failure.v1"
DEFAULT_COMPONENT_SMOKE_TIMEOUT_S = 2.0
DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S = 2.0
_NON_MECHANISM_DELTA_CLASSIFICATIONS = frozenset(
    {
        "identical_or_format_only",
        "parameter_only_delta",
        "rationale_only_delta",
        "parameter_and_rationale_only_delta",
    }
)


def run_component_smoke_check(
    candidate: CandidatePackageRef,
    smoke_adapter: ComponentSmokeAdapter | None,
    *,
    timeout_s: float = DEFAULT_COMPONENT_SMOKE_TIMEOUT_S,
) -> dict[str, Any]:
    """Run a task-owned lightweight smoke check over candidate callables."""

    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
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
    return _run_component_smoke_with_timeout(candidate_dir, fixture, timeout_s=float(timeout_s))


def assess_anti_degenerate_candidate(
    candidate: CandidatePackageRef,
    prompt_payload: Mapping[str, Any],
    *,
    evaluated_history_sources: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Assess candidate source against current best and runner-owned evaluated history."""

    current_best_source, current_best_hash = _current_best_source_from_prompt(prompt_payload)
    candidate_source = _read_candidate_components_source(candidate)
    if candidate_source is None:
        return _public_result(
            {
                "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
                "status": "pass",
                "classification": "unknown",
                "reason": "candidate_source_not_available",
                "compares_to": current_best_hash,
                "blocking": False,
            }
        )
    if current_best_source is None:
        current_best_assessment = _public_result(
            {
                "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
                "status": "pass",
                "classification": "unknown",
                "reason": "current_best_source_not_inline",
                "compares_to": None,
                "blocking": False,
            }
        )
    else:
        current_best_assessment = assess_anti_degenerate_source(
            candidate_source=candidate_source,
            current_best_source=current_best_source,
            current_best_sha256=current_best_hash,
        )
        if current_best_assessment.get("blocking") is True:
            return current_best_assessment

    history_assessment = _assess_against_evaluated_history(
        candidate_source=candidate_source,
        evaluated_history_sources=evaluated_history_sources,
    )
    if history_assessment is not None:
        return history_assessment

    result = dict(current_best_assessment)
    result["comparison_scope"] = "current_best_and_evaluated_history"
    result["evaluated_history_source_count"] = len(evaluated_history_sources)
    return _public_result(result)


def assess_anti_degenerate_source(
    *,
    candidate_source: str,
    current_best_source: str,
    current_best_sha256: str | None = None,
) -> dict[str, Any]:
    """Classify obvious no-op or parameter-only source changes."""

    candidate_digest = hashlib.sha256(candidate_source.encode("utf-8")).hexdigest()
    current_digest = current_best_sha256 or hashlib.sha256(current_best_source.encode("utf-8")).hexdigest()
    try:
        candidate_tree = ast.parse(candidate_source)
        current_tree = ast.parse(current_best_source)
    except SyntaxError as exc:
        return _public_result(
            {
                "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
                "status": "pass",
                "classification": "unknown",
                "reason": "source_parse_failed",
                "error_type": type(exc).__name__,
                "compares_to": current_digest,
                "candidate_source_sha256": candidate_digest,
                "blocking": False,
            }
        )

    candidate_ast = ast.dump(candidate_tree, include_attributes=False)
    current_ast = ast.dump(current_tree, include_attributes=False)
    if candidate_ast == current_ast:
        return _anti_degenerate_result(
            classification="identical_or_format_only",
            status="fail",
            reason="candidate_source_is_identical_or_format_only",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=True,
            feedback="Rewrite the candidate mechanism instead of resubmitting the current-best source.",
        )

    candidate_numeric_ast = ast.dump(_normalize_numeric_constants(candidate_tree), include_attributes=False)
    current_numeric_ast = ast.dump(_normalize_numeric_constants(current_tree), include_attributes=False)
    if candidate_numeric_ast == current_numeric_ast:
        return _anti_degenerate_result(
            classification="parameter_only_delta",
            status="warning",
            reason="candidate_changes_only_numeric_constants",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=False,
            feedback="Numeric-only edits are weak evidence; prefer a code-level reward mechanism change.",
        )

    candidate_string_ast = ast.dump(_normalize_string_constants(candidate_tree), include_attributes=False)
    current_string_ast = ast.dump(_normalize_string_constants(current_tree), include_attributes=False)
    if candidate_string_ast == current_string_ast:
        return _anti_degenerate_result(
            classification="rationale_only_delta",
            status="warning",
            reason="candidate_changes_only_strings_or_rationale",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=False,
            feedback="Rationale-only edits do not create a new training mechanism.",
        )

    candidate_non_mechanism_ast = ast.dump(
        _normalize_non_mechanism_constants(candidate_tree),
        include_attributes=False,
    )
    current_non_mechanism_ast = ast.dump(
        _normalize_non_mechanism_constants(current_tree),
        include_attributes=False,
    )
    if candidate_non_mechanism_ast == current_non_mechanism_ast:
        return _anti_degenerate_result(
            classification="parameter_and_rationale_only_delta",
            status="warning",
            reason="candidate_changes_only_numeric_and_string_constants",
            compares_to=current_digest,
            candidate_digest=candidate_digest,
            blocking=False,
            feedback=(
                "Numeric and rationale-only edits do not create a new training mechanism."
            ),
        )

    return _anti_degenerate_result(
        classification="mechanism_delta_present",
        status="pass",
        reason="candidate_source_structure_differs_from_current_best",
        compares_to=current_digest,
        candidate_digest=candidate_digest,
        blocking=False,
        feedback="Mechanism delta is present; runner-owned eval still decides acceptance.",
    )


def build_component_smoke_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    smoke_result: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = {
        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
        "status": "repair_required",
        "previous_candidate_id": candidate.candidate_id,
        "previous_candidate_changed_components": sorted(candidate.change_set.changed_components),
        "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
        "previous_candidate_source_delivery": "see_source_catalog",
        "failure_category": "component_smoke_error",
        "component_smoke_result": dict(smoke_result),
        "instructions": [
            "Fix only the candidate training_code/components.py so every reward and diagnostic callable executes on the task-owned RewardInputs fixture.",
            "Return vector-like finite reward outputs with the shape described by the prompt reward_inputs_contract.",
            "Diagnostic callables must execute without errors on the same task-owned fixture.",
            "Do not change runner, eval protocol, training budget, seed policy, checkpoints, PPO, or acceptance rules.",
        ],
    }
    return _public_result(feedback)


def build_anti_degenerate_repair_feedback(
    *,
    candidate: CandidatePackageRef,
    assessment: Mapping[str, Any],
) -> dict[str, Any]:
    feedback = {
        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
        "status": "rewrite_required",
        "previous_candidate_id": candidate.candidate_id,
        "previous_candidate_changed_components": sorted(candidate.change_set.changed_components),
        "previous_candidate_source_logical_id": LATEST_FAILED_SOURCE_ID,
        "previous_candidate_source_delivery": "see_source_catalog",
        "failure_category": "anti_degenerate_warning",
        "anti_degenerate_assessment": dict(assessment),
        "instructions": [
            "Rewrite the candidate to introduce a real code-level reward mechanism delta.",
            "Do not resubmit identical, formatting-only, or rationale-only source.",
            (
                "Do not repeat a previously evaluated mechanism by changing only "
                "metadata strings or numeric constants."
            ),
            "Keep runner, eval, seed, budget, checkpoint, PPO, curriculum, and acceptance rules unchanged.",
        ],
    }
    return _public_result(feedback)


def build_static_authoring_failure(
    *,
    candidate: CandidatePackageRef,
    feedback: Mapping[str, Any],
) -> dict[str, Any]:
    return _public_result(
        {
            "schema_version": STATIC_AUTHORING_FAILURE_SCHEMA_VERSION,
            "candidate_id": candidate.candidate_id,
            "candidate_changed_components": sorted(candidate.change_set.changed_components),
            "failure_category": feedback.get("failure_category", "static_authoring_error"),
            "feedback": dict(feedback),
        }
    )


def _run_component_smoke_with_timeout(
    candidate_dir: Path,
    fixture: ComponentSmokeFixture,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=1)
    progress_queue = ctx.Queue()
    process = ctx.Process(
        target=_component_smoke_worker,
        args=(str(candidate_dir), fixture, result_queue, progress_queue),
    )
    process.daemon = True
    process.start()
    latest_progress: Mapping[str, Any] = {}
    load_deadline = time.monotonic() + max(
        DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S,
        timeout_s,
    )
    callable_deadline: float | None = None
    while process.is_alive():
        deadline = callable_deadline or load_deadline
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            progress = progress_queue.get(timeout=min(0.05, remaining))
        except queue_module.Empty:
            continue
        if not isinstance(progress, Mapping):
            continue
        latest_progress = dict(progress)
        if (
            callable_deadline is None
            and latest_progress.get("phase") in {"reward", "diagnostic"}
        ):
            callable_deadline = time.monotonic() + timeout_s
    if process.is_alive():
        queued_progress = _latest_queue_item(progress_queue)
        if queued_progress:
            latest_progress = queued_progress
        process.terminate()
        process.join(1.0)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(1.0)
        return _component_smoke_timeout_result(
            fixture=fixture,
            latest_progress=latest_progress,
            timeout_s=(
                timeout_s
                if callable_deadline is not None
                else max(DEFAULT_COMPONENT_SMOKE_LOAD_TIMEOUT_S, timeout_s)
            ),
        )
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


def _component_smoke_worker(
    candidate_dir: str,
    fixture: ComponentSmokeFixture,
    result_queue: Any,
    progress_queue: Any,
) -> None:
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

    result_queue.put(
        _public_result(
            {
                "schema_version": COMPONENT_SMOKE_SCHEMA_VERSION,
                "status": "pass",
                "reason": "component_smoke_passed",
                "task_id": fixture.task_id,
                "checked_reward_terms": len(reward_results),
                "checked_diagnostics": len(diagnostic_results),
                "fixture_summary": dict(fixture.public_summary),
                "reward_results": reward_results,
                "diagnostic_results": diagnostic_results,
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
    return _public_result(
        {
            "name": term.name,
            "kind": "RewardTerm",
            "status": "pass",
            "reason": "reward_callable_returned_finite_expected_shape",
            "observed_shape": output["shape"],
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
    }


def _shape_and_values(value: Any) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if isinstance(value, SmokeTensor):
        return value.shape, value.flatten()
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


def _candidate_dir(candidate: CandidatePackageRef) -> Path | None:
    if "://" in candidate.path:
        return None
    path = Path(candidate.path)
    if not path.is_dir():
        return None
    return path


def _read_candidate_components_source(candidate: CandidatePackageRef) -> str | None:
    candidate_dir = _candidate_dir(candidate)
    if candidate_dir is None:
        return None
    try:
        return (candidate_dir / "training_code" / "components.py").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _current_best_source_from_prompt(prompt_payload: Mapping[str, Any]) -> tuple[str | None, str | None]:
    payloads = prompt_payload.get("source_payloads")
    if not isinstance(payloads, list):
        return None, None
    for item in payloads:
        if not isinstance(item, Mapping):
            continue
        if item.get("logical_id") != CURRENT_BEST_SOURCE_ID:
            continue
        content = item.get("content")
        if not isinstance(content, str):
            return None, None
        digest = item.get("content_sha256")
        return content, digest if isinstance(digest, str) else None
    return None, None


def _normalize_numeric_constants(tree: ast.AST) -> ast.AST:
    parsed = ast.parse(ast.unparse(tree))

    class Visitor(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            if isinstance(node.value, int | float) and not isinstance(node.value, bool):
                return ast.copy_location(ast.Constant(value=0), node)
            return node

    return ast.fix_missing_locations(Visitor().visit(parsed))


def _normalize_string_constants(tree: ast.AST) -> ast.AST:
    parsed = ast.parse(ast.unparse(tree))

    class Visitor(ast.NodeTransformer):
        def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
            if isinstance(node.value, str):
                return ast.copy_location(ast.Constant(value=""), node)
            return node

    return ast.fix_missing_locations(Visitor().visit(parsed))


def _normalize_non_mechanism_constants(tree: ast.AST) -> ast.AST:
    return _normalize_string_constants(_normalize_numeric_constants(tree))


def _assess_against_evaluated_history(
    *,
    candidate_source: str,
    evaluated_history_sources: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    checked_count = 0
    for historical in reversed(tuple(evaluated_history_sources)):
        source = historical.get("content")
        if not isinstance(source, str):
            continue
        checked_count += 1
        content_sha256 = historical.get("content_sha256")
        assessment = assess_anti_degenerate_source(
            candidate_source=candidate_source,
            current_best_source=source,
            current_best_sha256=(
                content_sha256 if isinstance(content_sha256, str) else None
            ),
        )
        equivalence = assessment.get("classification")
        if equivalence not in _NON_MECHANISM_DELTA_CLASSIFICATIONS:
            continue
        result = {
            "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
            "status": "fail",
            "classification": "historical_mechanism_repeat",
            "reason": "candidate_repeats_previously_evaluated_mechanism",
            "equivalence_classification": equivalence,
            "comparison_scope": "evaluated_history",
            "compares_to": assessment.get("compares_to"),
            "candidate_source_sha256": assessment.get("candidate_source_sha256"),
            "matched_round_id": historical.get("round_id"),
            "matched_candidate_id": historical.get("candidate_id"),
            "evaluated_history_source_count": len(evaluated_history_sources),
            "checked_history_source_count": checked_count,
            "blocking": True,
            "feedback_for_author": (
                "This mechanism was already evaluated. Use its recorded outcome to "
                "introduce a structurally different code-level mechanism."
            ),
        }
        return _public_result(result)
    return None


def _anti_degenerate_result(
    *,
    classification: str,
    status: str,
    reason: str,
    compares_to: str,
    candidate_digest: str,
    blocking: bool,
    feedback: str,
) -> dict[str, Any]:
    return _public_result(
        {
            "schema_version": ANTI_DEGENERATE_SCHEMA_VERSION,
            "status": status,
            "classification": classification,
            "reason": reason,
            "compares_to": compares_to,
            "candidate_source_sha256": candidate_digest,
            "blocking": blocking,
            "feedback_for_author": feedback,
        }
    )


def _public_result(value: Mapping[str, Any]) -> dict[str, Any]:
    clean = sanitize_prompt_public_context(dict(value))
    serialized = json.loads(json.dumps(clean, ensure_ascii=False))
    assert_public_only_prompt_payload(serialized)
    return serialized
