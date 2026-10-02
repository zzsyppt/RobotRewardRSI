"""A separate public-dev observation of an already bound Stair policy.

These phases have no acceptance authority. The original qualified evaluation
and its sealed evidence remain unchanged; this module does not qualify a policy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, Mapping

from true_rl_agent.evaluation.binding import EvaluationBindingV1
from true_rl_agent.evaluation.phase_assembly import require_phase_result_matches_plan
from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
from true_rl_agent.runner.job_attempts import JobAttemptExecutionScope
from true_rl_agent.runner.specs import EvalJobSpec, EvalPhaseJobSpec
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID, StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import STAIR_EVAL_PURPOSE_PUBLIC_DEV_DIAGNOSTIC
from true_rl_agent.tasks.stair_climbing.phased_evaluation import canonical_stair_phase_result_summary
from true_rl_agent.tasks.stair_climbing.seed_policy import (
    STAIR_FORMAL_EVAL_NAMESPACE, diagnostic_seed_namespace,
)

if TYPE_CHECKING:
    from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter


def _binding(config: StairClimbingTaskConfig, namespace: str) -> EvaluationBindingV1:
    if not config.is_unified_protocol:
        raise ValueError("Stair public-dev diagnostics require the unified protocol")
    return EvaluationBindingV1(
        eval_protocol_hash=config.evaluation_protocol().protocol_hash,
        acceptance_protocol_hash=config.acceptance_protocol.protocol_hash,
        task_protocol_hash=config.task_protocol_hash,
        seed_cohort_hash=config.seed_resolver().namespace_cohort_hash(namespace),
    )


def prepare_stair_public_dev_diagnostic(
    *, config: StairClimbingTaskConfig,
    backend_adapter: IsaacLabRslRlBackendAdapter,
    formal_eval_plan: EvalJobSpec,
    job_id: str, round_id: str, summary_path: str,
    attempt_scope: JobAttemptExecutionScope | None = None,
    numeric_evidence_dir: str | None = None,
    numeric_evidence_max_bytes: int = 2 * 1024**3,
    capture_numeric: bool = True,
    diagnostic_cohort: str = "v1",
) -> EvalPhaseJobSpec:
    """Rebind one executable formal plan to the declared diagnostic cohort.

    The caller restores and validates the policy's qualification separately.
    The returned phase uses the existing runner, lease and CPU finalizer.
    """
    namespace = diagnostic_seed_namespace(diagnostic_cohort)
    formal_eval_plan.execution_checkpoint()
    if (formal_eval_plan.task_id != STAIR_TASK_ID
            or formal_eval_plan.evaluation_protocol() != config.evaluation_protocol()
            or formal_eval_plan.evaluation_binding != _binding(config, STAIR_FORMAL_EVAL_NAMESPACE)):
        raise ValueError("Stair diagnostic source plan differs from the formal task domain")
    if job_id == formal_eval_plan.job_id:
        raise ValueError("Stair diagnostic evaluation requires a new job identity")
    plan = replace(formal_eval_plan, job_id=job_id,
        evaluation_binding=_binding(config, namespace),
        attempt_scope=None)
    plan = backend_adapter.rematerialize_eval_plan(
        eval_plan=plan, round_id=round_id, summary_path=summary_path)
    argv = [*plan.metadata["cli_argv"], "--stair-eval-purpose", STAIR_EVAL_PURPOSE_PUBLIC_DEV_DIAGNOSTIC]
    if diagnostic_cohort != "v1":
        argv.extend(("--diagnostic-cohort", diagnostic_cohort))
    plan = replace(plan, metadata={**plan.metadata, "cli_argv": argv,
        "cli_command": " ".join(argv), "stair_eval_purpose": STAIR_EVAL_PURPOSE_PUBLIC_DEV_DIAGNOSTIC})
    return backend_adapter.materialize_eval_phase(
        eval_plan=plan, phase=EvaluationPhase.PUBLIC_DEV, attempt_scope=attempt_scope,
        capture_numeric=capture_numeric, numeric_evidence_dir=numeric_evidence_dir,
        numeric_evidence_max_bytes=numeric_evidence_max_bytes)


def project_stair_public_dev_diagnostic(
    *, config: StairClimbingTaskConfig, eval_plan: EvalJobSpec,
    phase_result: EvaluationPhaseResult,
    diagnostic_cohort: str = "v1",
) -> Mapping[str, Any]:
    """Validate the new plan/phase before exposing its public observation.

    This is not a PublicEvalResult and must not replace the qualified anchor.
    Formal assembly continues to reject this diagnostic cohort binding.
    """
    namespace = diagnostic_seed_namespace(diagnostic_cohort)
    if (eval_plan.task_id != STAIR_TASK_ID
            or eval_plan.evaluation_binding != _binding(config, namespace)
            or phase_result.phase is not EvaluationPhase.PUBLIC_DEV):
        raise ValueError("Stair diagnostic projection requires its public_dev domain")
    require_phase_result_matches_plan(phase_result, eval_plan)
    summary = canonical_stair_phase_result_summary(
        config=config, expected_phase=EvaluationPhase.PUBLIC_DEV, result=phase_result,
        expected_seed_namespace=namespace)
    if "behavior" in phase_result.task_evidence:
        from true_rl_agent.tasks.stair_climbing.behavior import project_stair_behavior

        summary = {**summary, "behavior": project_stair_behavior(phase_result.task_evidence["behavior"])}
    return summary
