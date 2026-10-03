"""Confirmation author contexts: old qualification, separate public observation.

All artifact refs are hash-bound relative to ``base``; nothing is collected here.
Push historically includes the seed cohort in its task protocol hash. Its new
seed therefore rebuilds that legitimate composition/binding, while task-owned
source admission preserves the original training identity. No new diagnostic
phase is assembled with old sealed evidence or installed as CurrentBest.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import importlib.util
from pathlib import Path
from typing import Any


def _p3(root: Path):
    path = root / "PLAN/feedback_sys/P3/author_experiment.py"
    spec = importlib.util.spec_from_file_location("p3_confirmation_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _plan(value):
    """Read the public eval-plan commitment saved by the replay entrypoints."""
    from true_rl_agent.evaluation.binding import EvaluationBindingV1
    from true_rl_agent.evaluation.subject import ExperimentSubject
    from true_rl_agent.runner.specs import CheckpointRef, EvalJobSpec, GpuProcessBinding

    value = value.get("spec", value)
    value = value.get("evaluation_plan", value)
    gpu = value.get("gpu_process_binding")
    plan = EvalJobSpec(
        **{key: value[key] for key in ("job_id", "task_id", "subject_manifest_hash",
            "program_source_checksum", "training_binding_hash", "capability_profile_hash",
            "execution_device", "public_seed_count", "public_accept_seed_count",
            "sealed_validation_seed_count", "eval_num_envs", "eval_protocol_version")},
        subject=ExperimentSubject(kind=value["subject"]["kind"], subject_id=value["subject"]["id"]),
        evaluation_binding=EvaluationBindingV1.from_mapping(value["evaluation_binding"]),
        checkpoint=CheckpointRef(**value["checkpoint"]),
        gpu_process_binding=(GpuProcessBinding(host_execution_device=value["execution_device"], **gpu) if gpu else None))
    if plan.to_public_dict() != value:
        raise ValueError("saved eval plan does not round-trip through the real plan contract")
    return plan


class DiagnosticFeedbackAdapter:
    """Keep qualification/source context intact; replace only feedback content."""
    def __init__(self, delegate, feedback, focus):
        self.delegate, self.feedback, self.focus = delegate, feedback, focus

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def project_prompt_public_history(self, **kwargs):
        projected = self.delegate.project_prompt_public_history(**kwargs)
        return replace(projected, public_score_source="original_qualification." + projected.public_score_source)

    def build_prompt_sections(self, **kwargs):
        from true_rl_agent.tasks.prompt_adapter import sanitize_task_prompt_sections

        sections = copy.deepcopy(self.delegate.build_prompt_sections(**kwargs))
        sections["public_feedback_summary"] = copy.deepcopy(self.feedback)
        sections["selection_guidance"] = [*sections["selection_guidance"],
            "The qualified parent and best_public_score retain the original qualification cohort. "
            "The separately identified diagnostic public-dev observation supplies the current feedback; "
            "it has no acceptance authority and must not be compared as a score improvement over that historical cohort.",
            self.focus]
        return sanitize_task_prompt_sections(sections)


def build_context(case, arm, spec, *, diagnostic_phase_ref, diagnostic_plan_ref,
                  allowed_seed_cohort_hash, root: Path, base: Path, output: Path,
                  diagnostic_eval_base_seed=None, source_manifest_ref=None,
                  training_summary_ref=None, development_fixture=False, exp=None):
    """Return ``(original_composition, PublicContextBuilder, feedback)``.

    ``spec`` is the original P3 reference context (its old diagnostic ref is
    ignored on a copy). Case carries task/checkpoint/common_focus. The Push
    branch also needs the registered seed and authenticated source-training
    manifest/receipt. ``development_fixture`` explicitly permits a historical
    cohort for CPU wiring checks only; production confirmation forbids it.
    """
    from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, primary_ledger_epoch_hash
    from true_rl_agent.evaluation.binding import canonical_json_hash
    from true_rl_agent.evaluation.phase_assembly import require_phase_result_matches_plan
    from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
    from true_rl_agent.memory.context_builder import PublicContextBuilder

    if arm not in {"A", "T"} or case["task"] != spec["task"]:
        raise ValueError("confirmation task or arm differs")
    diagnostic_as_of = case.get("as_of_cutoff")
    if not development_fixture and (not isinstance(diagnostic_as_of, str) or not diagnostic_as_of):
        raise ValueError("confirmation case requires the new observation's as-of cutoff")
    exp = exp or _p3(root)
    original_spec = copy.deepcopy(spec)
    original_spec.pop("diagnostic_public_dev_phase", None)
    comp, best, anchor, source, genesis_hash, initial_hash = exp.reference_context(
        original_spec, root=root, base=base, output=output)
    anchor_hash = canonical_json_hash(anchor)
    phase = EvaluationPhaseResult.from_private_mapping(exp.bound_read(diagnostic_phase_ref, base))
    plan = _plan(exp.bound_read(diagnostic_plan_ref, base))
    require_phase_result_matches_plan(phase, plan)
    expected = {"checkpoint_hash": best.checkpoint_hash,
        "program_source_checksum": best.program_source_checksum, "training_binding_hash": best.training_binding_hash,
        "capability_profile_hash": best.capability_profile_hash, "subject_manifest_hash": best.subject_manifest_hash,
        "task_id": comp.task_id, "eval_protocol_hash": best.eval_protocol_hash}
    if (phase.phase is not EvaluationPhase.PUBLIC_DEV or phase.subject.kind != "reference"
            or phase.subject.subject_id != best.anchor_id or case["checkpoint_hash"] != best.checkpoint_hash
            or any(getattr(phase, key) != value for key, value in expected.items())):
        raise ValueError("diagnostic phase differs from the qualified policy identity")
    binding = plan.evaluation_binding
    if (binding.seed_cohort_hash != allowed_seed_cohort_hash
            or binding.acceptance_protocol_hash != best.acceptance_protocol_hash):
        raise ValueError("diagnostic cohort or acceptance contract is not registered")
    if binding.seed_cohort_hash == best.seed_cohort_hash and not development_fixture:
        raise ValueError("confirmation requires a new cohort; historical data are development only")

    if spec["task"] == "stair":
        if binding.task_protocol_hash != best.task_protocol_hash:
            raise ValueError("Stair diagnostic task domain differs")
        if development_fixture and binding.seed_cohort_hash == best.seed_cohort_hash:
            projected = comp.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=phase)
        else:
            from true_rl_agent.tasks.stair_climbing.diagnostic_evaluation import project_stair_public_dev_diagnostic
            from true_rl_agent.tasks.stair_climbing.seed_policy import diagnostic_cohort_from_namespace

            projected = project_stair_public_dev_diagnostic(config=comp.task_plugin.config,
                eval_plan=plan, phase_result=phase,
                diagnostic_cohort=diagnostic_cohort_from_namespace(
                    phase.task_evidence["execution_identity"]["seed_namespace"]))
    elif spec["task"] == "push":
        from true_rl_agent.composition import TaskCompositionContext
        from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
        from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
        from true_rl_agent.tasks.push_recovery.checkpoint_admission import validate_push_reference_evaluation_source

        if diagnostic_eval_base_seed is None or source_manifest_ref is None or training_summary_ref is None:
            raise ValueError("Push diagnostic requires registered seed and original training source refs")
        provider = PushRecoveryCompositionProvider()
        diagnostic_comp = provider.resolve(values={**provider.cli_defaults(), **spec["composition_values"],
            "eval_base_seed": diagnostic_eval_base_seed},
            context=TaskCompositionContext(project_root=root, output_dir=output / "diagnostic_composition"))
        original_config, diagnostic_config = comp.task_plugin.config, diagnostic_comp.task_plugin.config
        # Only the declared seed and its resolved task identity may differ. The
        # production resolver regenerates ABI, physics, permissions and hashes.
        if replace(diagnostic_config, eval_base_seed=original_config.eval_base_seed,
                   resolved_cell_task_protocol_hash=original_config.resolved_cell_task_protocol_hash) != original_config:
            raise ValueError("Push diagnostic changed task settings beyond the declared seed")
        if binding != diagnostic_comp.backend_adapter.evaluation_binding_for(diagnostic_comp.task_plugin.task_spec):
            raise ValueError("Push phase binding differs from its registered seed composition")
        manifest = ResolvedCellManifestV1.from_mapping(exp.bound_read(source_manifest_ref, base))
        trained = exp.bound_read(training_summary_ref, base)
        source_binding = validate_push_reference_evaluation_source(manifest, diagnostic_comp.resolved_cell_manifest, trained)
        if (source_binding != best.training_binding_hash
                or trained["checkpoint"]["checksum"] != best.checkpoint_hash
                or trained["program_source_checksum"] != best.program_source_checksum):
            raise ValueError("Push diagnostic source admission differs from qualified training")
        projected = diagnostic_comp.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=phase)
        behavior = projected.get("behavior", {})
        if behavior.get("comparison_domain", {}).get("resolved_cell_manifest_hash") != diagnostic_comp.resolved_cell_manifest.manifest_hash:
            raise ValueError("Push diagnostic behavior differs from its resolved cell")
    else:
        raise ValueError("unknown confirmation task")
    if not isinstance(projected.get("behavior"), dict):
        raise ValueError("diagnostic phase did not seal measured behavior")
    behavior = projected["behavior"]
    if spec["task"] == "stair":
        if (behavior["domain"]["checkpoint_hash"] != best.checkpoint_hash
                or behavior["domain"]["archive_hash"] != phase.task_evidence["numeric_evidence"]["manifest_sha256"]):
            raise ValueError("Stair behavior differs from its phase checkpoint/archive")
    else:
        if any(behavior["binding"].get(key) != value for key, value in {
            "checkpoint_hash": best.checkpoint_hash, "program_source_checksum": best.program_source_checksum,
            "evaluation_plan_hash": phase.evaluation_plan_hash}.items()):
            raise ValueError("Push behavior differs from its phase policy/plan")
        archives = sorted(seed["numeric_evidence_ref"]["manifest_sha256"] for seed in phase.task_evidence["seed_summaries"])
        if archives != sorted(source["manifest_sha256"] for source in behavior["sources"]):
            raise ValueError("Push behavior differs from its phase archives")

    # Only a task-adapter input shape. It contains no historical accept/sealed
    # fields and is never passed to PublicContextBuilder or qualification.
    observation = {"status": "pass", "metrics": {"public_dev": copy.deepcopy(projected)}}
    task_input = {"anchor_kind": "reference_program", "public_result": observation,
        "public_result_hash": canonical_json_hash(observation)}
    if behavior.get("schema_version") == "true_rl_agent.stair.behavior.public.v5":
        task_input["trajectory_phase_binding"] = {"job_id": phase.job_id,
            "checkpoint_hash": phase.checkpoint_hash, "evaluation_plan_hash": phase.evaluation_plan_hash,
            "evaluation_binding_hash": phase.evaluation_binding_hash,
            "archive_hash": phase.task_evidence["numeric_evidence"]["manifest_sha256"],
            "seed_namespace": phase.task_evidence["execution_identity"]["seed_namespace"]}
    feedback = comp.task_plugin.prompt_adapter().build_prompt_sections(public_records=(),
        public_context={"initial_anchor_public_evidence": task_input})["public_feedback_summary"]
    feedback = copy.deepcopy(feedback)
    owner = feedback if spec["task"] == "stair" else feedback["public_dev"]
    if "behavior" not in owner:
        raise ValueError("default task feedback lost measured behavior")
    if arm == "A":
        owner.pop("behavior")
    feedback["diagnostic_observation"] = {"phase": "public_dev", "acceptance_authority": False,
        "evaluation_plan_hash": phase.evaluation_plan_hash, "evaluation_binding_hash": phase.evaluation_binding_hash,
        "seed_cohort_hash": binding.seed_cohort_hash, "phase_result_hash": canonical_json_hash(phase.to_private_dict()),
        "qualification_binding_hash": best.evaluation_binding_hash,
        "as_of_cutoff": diagnostic_as_of if diagnostic_as_of is not None else spec.get("diagnostic_as_of"),
        "scope": "same qualified policy; separate public-dev observation; historical qualification unchanged",
        "development_fixture": bool(development_fixture)}
    if spec["task"] == "stair":
        for entry in feedback.get("feedback_entries", []):
            entry.update(source_kind="diagnostic_public_dev", selection_basis=["diagnostic_same_qualified_policy"])
    else:
        feedback["source_selection"] = "diagnostic_same_qualified_policy"
    adapter = DiagnosticFeedbackAdapter(comp.task_plugin.prompt_adapter(), feedback, case["common_focus"])
    namespace = canonical_json_hash({"kind": "confirmation_author_context", "qualified_namespace": genesis_hash,
        "diagnostic_phase_sha256": diagnostic_phase_ref["sha256"], "plan_hash": phase.evaluation_plan_hash})
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter, task_prompt_adapter=adapter,
        phased_evaluation_adapter=comp.task_plugin.phased_evaluation_adapter(),
        acceptance_protocol=comp.task_plugin.task_spec.acceptance_protocol, initial_current_best=best,
        session_genesis_hash=namespace, candidate_capability_profile=comp.task_plugin.task_spec.candidate_capability_profile,
        initial_reference_program_source=source, initial_anchor_public_evidence=anchor,
        expected_initial_policy_state_hash=initial_hash, ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(namespace))
    if canonical_json_hash(anchor) != anchor_hash:
        raise ValueError("diagnostic feedback changed the qualification anchor")
    return comp, builder, feedback
