"""Push's existing sealed-cohort comparison; new physical qualification is P6."""

from pathlib import Path

from true_rl_agent.agent_core.generation_finalization import GenerationFinalPlan, publish_final_plan
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin


def prepare_push_final_plan(
    *, output_dir: Path, session_id: str, config: PushRecoveryTaskConfig,
    required_closed_generations: int, resume_existing: bool,
) -> GenerationFinalPlan:
    plugin = PushRecoveryTaskPlugin(config)
    plan = GenerationFinalPlan(
        session_id=session_id, task_id=plugin.task_spec.task_id,
        task_protocol_hash=plugin.task_spec.resolved_task_protocol_hash(),
        evaluation_protocol_hash=config.evaluation_protocol().protocol_hash,
        seed_cohort_hash=config.seed_cohort_hash(),
        phase=EvaluationPhase.SEALED_VALIDATION,
        seed_count_per_job=config.sealed_validation_seed_count,
        required_closed_generations=required_closed_generations,
        task_commitment={
            "schema_version": "true_rl_agent.task.push_recovery.final_cohort.v1",
            "cohort_source": "existing_sealed_validation",
            "episodes_per_seed": config.eval_num_envs,
            "new_seed_generalization_claim": False,
            "new_geometry_generalization_claim": False,
        },
    )
    publish_final_plan(output_dir, plan, require_existing=resume_existing)
    return plan


def summarize_push_final_comparison(*, config, plan, reference, final):
    from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter
    from true_rl_agent.tasks.push_recovery.metrics import project_push_split
    from true_rl_agent.evaluation.derived_statistics import sample_statistics

    adapter = PushPhasedEvaluationAdapter(config, plan.task_protocol_hash)
    splits = []
    for result in (reference, final):
        if result.phase != plan.phase:
            raise ValueError("Push final comparison cohort differs from its plan")
        errors = []
        split = project_push_split(adapter._phase_summaries(result), split_name=plan.phase.value, errors=errors)
        if errors or split["score_stats"]["count"] != plan.seed_count_per_job:
            raise ValueError("Push final comparison is incomplete")
        splits.append(split)
    deltas = [b - a for a, b in zip(*(split["score_stats"]["values"] for split in splits), strict=True)]
    return {"schema_version": "true_rl_agent.task.push_recovery.final_comparison.v1",
        "summaries": dict(zip(("genesis", "final"), splits, strict=True)),
        "paired_seed_deltas": deltas,
        # Match the existing Push protocol's arithmetic; P4 does not revise it.
        "paired_statistics": sample_statistics(deltas, statistics_version=None),
        "new_seed_generalization_claim": False, "new_geometry_generalization_claim": False}
