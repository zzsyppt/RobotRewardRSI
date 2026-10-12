"""Stair cohort preparation and version reading for shared final execution."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace

from true_rl_agent.agent_core.generation_finalization import (
    FINAL_PLAN_SCHEMA_VERSION, GenerationFinalPlan, publish_final_plan,
)
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.json_contract import load_strict_json_object
from true_rl_agent.tasks.stair_climbing.config import STAIR_TASK_ID, StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE
from true_rl_agent.tasks.stair_climbing.study_holdout import StairStudyHoldout


def _build_plan(*, config, holdout, required_closed_generations):
    bound = config.bind_final_holdout(holdout)
    suite = bound.ordered_evaluation_suite
    return GenerationFinalPlan(
        session_id=holdout.study_id, task_id=STAIR_TASK_ID,
        task_protocol_hash=config.task_protocol_hash,
        evaluation_protocol_hash=config.evaluation_protocol().protocol_hash,
        seed_cohort_hash=bound.seed_resolver().namespace_cohort_hash(STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE),
        phase=EvaluationPhase.SEALED_VALIDATION,
        seed_count_per_job=config.sealed_validation_seed_count,
        required_closed_generations=required_closed_generations,
        task_commitment={
            "schema_version": "true_rl_agent.task.stair_climbing.final_cohort.v1",
            "source_evaluation_suite_hash": holdout.source_suite_hash,
            "evaluation_suite_hash": suite.suite_hash,
            "sealed_cohort_hash": suite.cohort_hash("sealed_validation"),
            "confirmation_namespace": STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE,
            "cell_count": len(suite.cells_for_cohort("sealed_validation")),
            "episodes_per_seed_per_cell": suite.episodes_per_seed_per_cell,
            "study_holdout_commitment": holdout.commitment,
            "replication_of": holdout.replication_of,
        },
    )


def prepare_stair_final_plan(
    *, output_dir: Path, session_id: str, config: StairClimbingTaskConfig,
    required_closed_generations: int, resume_existing: bool,
    replication_source: Path | None = None,
) -> GenerationFinalPlan:
    if not config.requires_study_holdout or config.protocol_state != "active":
        raise ValueError("new Stair sessions require the active study holdout contract")
    private_path = output_dir / "final_confirmation" / "study_holdout.json"
    replication = StairStudyHoldout.load(replication_source) if replication_source is not None else None
    if private_path.exists():
        holdout = StairStudyHoldout.load(private_path)
    elif resume_existing:
        raise FileNotFoundError("resumed Stair session is missing its committed study holdout")
    else:
        holdout = StairStudyHoldout.create(
            path=private_path, study_id=session_id,
            source_suite=config.ordered_evaluation_suite,
            training_profile=config.unified_training_profile,
            replication=replication,
        )
    if holdout.study_id != session_id or (replication is not None and holdout.replication_of != replication.commitment):
        raise ValueError("Stair study differs from the requested session or replication")
    plan = _build_plan(config=config, holdout=holdout, required_closed_generations=required_closed_generations)
    publish_final_plan(output_dir, plan, require_existing=resume_existing)
    return plan


def bind_stair_final_config(config, *, plan, plan_path):
    if not isinstance(plan, GenerationFinalPlan):
        # Historical fields keep their original meaning in the version reader.
        from true_rl_agent.tasks.stair_climbing.final_confirmation import bind_stair_generation_final_config

        return bind_stair_generation_final_config(config, plan=plan, plan_path=plan_path)
    holdout = StairStudyHoldout.load(
        plan_path.with_name("study_holdout.json"),
        expected_commitment=plan.task_commitment["study_holdout_commitment"],
    )
    expected = _build_plan(config=config, holdout=holdout,
                           required_closed_generations=plan.required_closed_generations)
    if expected != plan:
        raise ValueError("Stair final plan differs from its frozen task cohort")
    return config.bind_final_holdout(holdout)


def read_stair_final_plan(path, *, expected_session_id=None, expected_config=None):
    value = load_strict_json_object(path, require_canonical=True, trailing_newline=True)
    if value.get("schema_version") != FINAL_PLAN_SCHEMA_VERSION:
        from true_rl_agent.tasks.stair_climbing.final_confirmation import load_stair_generation_final_confirmation_plan

        return load_stair_generation_final_confirmation_plan(
            path, expected_session_id=expected_session_id, expected_config=expected_config,
        )
    plan = GenerationFinalPlan.from_mapping(value)
    if plan.task_id != STAIR_TASK_ID or (expected_session_id is not None and plan.session_id != expected_session_id):
        raise ValueError("Stair final plan task or session differs")
    if expected_config is not None:
        bind_stair_final_config(expected_config, plan=plan, plan_path=path)
    return plan


def bind_stair_final_job(eval_plan, *, config, plan, plan_path):
    bind_stair_final_config(config, plan=plan, plan_path=plan_path)
    from true_rl_agent.tasks.stair_climbing.entrypoint_contract import STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION

    argv = [*eval_plan.metadata["cli_argv"], "--stair-final-plan", plan_path.as_posix(),
            "--stair-eval-purpose", STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION]
    return replace(eval_plan, metadata={**eval_plan.metadata, "cli_argv": argv,
        "cli_command": " ".join(argv), "stair_eval_purpose": STAIR_EVAL_PURPOSE_FINAL_CONFIRMATION})


def summarize_stair_final_comparison(*, config, plan, plan_path, reference, final,
                                     objective_family_mask):
    from true_rl_agent.tasks.stair_climbing.phased_evaluation import canonical_stair_phase_result_summary
    from true_rl_agent.tasks.stair_climbing.unified_scoring import build_unified_score_projection
    from true_rl_agent.evaluation.derived_statistics import DERIVED_STATISTICS_VERSION, sample_statistics

    bound = bind_stair_final_config(config, plan=plan, plan_path=plan_path)
    summaries = [canonical_stair_phase_result_summary(config=bound, expected_phase=plan.phase,
        result=result, expected_seed_namespace=STAIR_FINAL_CONFIRMATION_EVAL_NAMESPACE)
        for result in (reference, final)]
    projections = [build_unified_score_projection(summary, objective_family_mask=objective_family_mask)
                   for summary in summaries]
    scores = [projection["j_primary"]["seed_macro_scores"] for projection in projections]
    if any(len(values) != plan.seed_count_per_job for values in scores):
        raise ValueError("Stair final comparison seed count differs from its plan")
    deltas = [b - a for a, b in zip(*scores, strict=True)]
    return {"schema_version": "true_rl_agent.task.stair_climbing.final_comparison.v1",
        "summaries": dict(zip(("genesis", "final"), summaries, strict=True)),
        "scores": dict(zip(("genesis", "final"), projections, strict=True)),
        "paired_seed_deltas": deltas,
        "paired_statistics": sample_statistics(deltas, statistics_version=DERIVED_STATISTICS_VERSION)}
