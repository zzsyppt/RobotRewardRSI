"""Push projections for the fixed generation phases.

The host runs and persists phases. This module owns success, motion checks,
selection and the existing Push holdout rule; it never schedules a job.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from true_rl_agent.evaluation.binding import canonical_json_hash, require_sha256_checksum, is_sha256_digest
from true_rl_agent.evaluation.phased_adapter import GenerationWorkload
from true_rl_agent.composition.cell_manifest import ResolvedCellManifestV1
from true_rl_agent.evaluation.phase_assembly import (
    bind_phase_receipts,
    merge_phase_costs,
    require_phase_result_matches_plan,
)
from true_rl_agent.evaluation.phases import (
    EvaluationAssembly, EvaluationPhase, EvaluationPhaseResult,
)
from true_rl_agent.evaluation.sealed_validation import build_sealed_validation_gate
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.eval_isolation import build_candidate_isolation_gate
from true_rl_agent.runner.specs import EvalJobSpec
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.phased_evaluation import PUSH_PHASE_EVIDENCE_SCHEMA
from true_rl_agent.tasks.push_recovery.metrics import (
    DEFAULT_MOTION_LIMITS,
    FORCE_ROLLOUT_SCHEMA,
    push_behavior_quality_gate,
    push_motion_quality_gate,
    project_push_split,
    build_push_public_result_from_summaries,
    build_push_sealed_rollout_evidence,
    push_recovery_evidence_gate,
    recovery_validation_contract,
)

@dataclass(frozen=True)
class PushPublicDevScreen:
    trial_id: str
    screen_role: str
    phase_result_hash: str
    evaluation_plan_hash: str
    checkpoint_hash: str
    evaluation_binding_hash: str
    execution_device: str
    resource_binding_hash: str | None
    train_receipt_checksum: str | None
    score_stats: Mapping[str, Any]
    behavior_passed: bool

    def to_selector_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.push_recovery.public_dev_screen.v1",
            **{name: copy.deepcopy(getattr(self, name)) for name in self.__dataclass_fields__},
            "acceptance_authority": False,
        }

    @property
    def evidence_hash(self) -> str:
        return canonical_json_hash(self.to_selector_dict())


@dataclass(frozen=True)
class PushSelection:
    selected_trial_id: str | None
    contract_hash: str
    screen_evidence_hashes: tuple[str, str]
    guard_reference_evidence_hash: str

    @property
    def reason(self) -> str:
        return "public_dev_success_rank" if self.selected_trial_id else "no_eligible_public_dev_trial"

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "true_rl_agent.task.push_recovery.selection.v1",
            "selected_trial_id": self.selected_trial_id,
            "contract_hash": self.contract_hash,
            "screen_evidence_hashes": list(self.screen_evidence_hashes),
            "guard_reference_evidence_hash": self.guard_reference_evidence_hash,
            "reason": self.reason,
            "acceptance_authority": False,
        }


@dataclass(frozen=True)
class PushPhasedEvaluationAdapter:
    config: PushRecoveryTaskConfig
    task_protocol_hash: str

    @property
    def parent_public_dev_source(self) -> Literal["genesis_receipt"]:
        return "genesis_receipt"

    def selection_contract(self) -> dict[str, Any]:
        payload = {
            "schema_version": "true_rl_agent.task.push_recovery.selection_contract.v1",
            "task_protocol_hash": self.task_protocol_hash,
            "evaluation_protocol_hash": self.config.evaluation_protocol().protocol_hash,
            "source_phase": "public_dev",
            "rank": "success_rate_descending_then_trial_id_ascending",
            "eligibility": "existing_push_motion_and_behavior_checks",
            "motion_limits": dict(DEFAULT_MOTION_LIMITS),
            "width": 2,
            "acceptance_authority": False,
        }
        if self.config.push_protocol is not None:
            payload.update(
                schema_version="true_rl_agent.task.push_recovery.selection_contract.v2",
                eligibility="complete_valid_planned_push_recovery_evidence",
                independent_validation=recovery_validation_contract(),
            )
            payload.pop("motion_limits")
        if self.config.push_evaluation_protocol is not None:
            payload.update(schema_version="true_rl_agent.task.push_recovery.selection_contract.v3",
                           eligibility="complete_verified_force_sequence_and_fresh_recovery_evidence")
        return {**payload, "contract_hash": canonical_json_hash(payload)}

    def cost_contract(self) -> dict[str, Any]:
        phases = {
            phase.value: {"max_jobs": 2 if phase is EvaluationPhase.PUBLIC_DEV else 1,
                          "seed_count_per_job": self.seed_count(phase)}
            for phase in EvaluationPhase
        }
        seed_units = sum(item["max_jobs"] * item["seed_count_per_job"] for item in phases.values())
        payload = {
            "schema_version": "true_rl_agent.task.push_recovery.phased_eval_cost_contract.v1",
            "task_protocol_hash": self.task_protocol_hash,
            "evaluation_protocol_hash": self.config.evaluation_protocol().protocol_hash,
            "topology": "two_dev_selected_accept_selected_sealed",
            "per_generation_phases": phases,
            "session_bootstrap": {},
            "max_session_bootstrap_seed_units": 0,
            "max_seed_units_per_generation": seed_units,
            "episodes_per_seed_unit": self.config.eval_num_envs,
            "max_episode_executions_per_generation": seed_units * self.config.eval_num_envs,
            "max_session_bootstrap_episode_executions": 0,
            "scientific_trial_count": 2,
            "sealed_query_count": 1,
            "execution_retry_reserve_included": False,
        }
        return {**payload, "contract_hash": canonical_json_hash(payload)}

    def seed_count(self, phase: EvaluationPhase) -> int:
        return {
            EvaluationPhase.PUBLIC_DEV: self.config.public_seed_count,
            EvaluationPhase.PUBLIC_ACCEPT: self.config.public_accept_seed_count,
            EvaluationPhase.SEALED_VALIDATION: self.config.sealed_validation_seed_count,
        }[phase]

    def generation_workload(self, *, cell_manifest: ResolvedCellManifestV1) -> GenerationWorkload:
        expected_protocol = (cell_manifest.base_task_protocol_hash if cell_manifest.uses_legacy_training_binding_v1
                             else cell_manifest.resolved_cell_task_protocol_hash)
        if expected_protocol != self.task_protocol_hash:
            raise ValueError("Push workload cell differs from its task protocol")
        projection = cell_manifest.robot_task_binding["task_projection"]
        facts = projection.get("push_abi_facts", projection.get("phase0_binding_facts"))
        control = facts["control"]
        episode_seconds = self.config.eval_runtime_dynamics()["environment"]["episode_length_s"]
        # Match the runtime's first-episode loop using this cell's frozen control
        # period; robot geometry and control facts never enter the shared host.
        steps = episode_seconds / control["control_dt_s"]
        max_steps = math.ceil(steps) if self.config.push_protocol is not None else int(round(steps))
        return GenerationWorkload(
            task_protocol_hash=self.task_protocol_hash,
            phased_evaluation_cost_contract_hash=self.cost_contract()["contract_hash"],
            train_steps_per_job=self.config.budget_steps,
            public_dev_seed_units_per_job=self.config.public_seed_count,
            public_accept_seed_units_per_job=self.config.public_accept_seed_count,
            sealed_validation_seed_units_per_job=self.config.sealed_validation_seed_count,
            max_eval_control_steps_per_seed_unit=self.config.eval_num_envs * max_steps,
        )

    def _phase_summaries(self, result: EvaluationPhaseResult) -> list[Mapping[str, Any]]:
        evidence = result.task_evidence
        if (
            result.task_id != "unitree_push_recovery"
            or result.eval_protocol_hash != self.config.evaluation_protocol().protocol_hash
            or set(evidence) not in ({"schema_version", "phase", "seed_summaries"},
                                     {"schema_version", "phase", "seed_summaries", "behavior"},
                                     {"schema_version", "phase", "seed_summaries", "behavior", "behavior_full"},
                                     {"schema_version", "phase", "seed_summaries", "behavior", "behavior_full", "trajectory_build"})
            or evidence["schema_version"] != PUSH_PHASE_EVIDENCE_SCHEMA
            or evidence["phase"] != result.phase.value
        ):
            raise ValueError("Push phase evidence identity mismatch")
        if "behavior" in evidence and result.phase is not EvaluationPhase.PUBLIC_DEV:
            raise ValueError("Push behavior evidence is public_dev-only")
        if "behavior_full" in evidence:
            full = evidence["behavior_full"]
            public = evidence["behavior"]
            if (not isinstance(full, Mapping) or not isinstance(public, Mapping)
                    or set(full) != {"schema_version", "file", "sha256", "size_bytes"}
                    or full.get("schema_version") != "true_rl_agent.process_full_ref.v1"
                    or full.get("file") != "behavior_full.json"
                    or not is_sha256_digest(full.get("sha256"))
                    or not isinstance(full.get("size_bytes"), int) or isinstance(full.get("size_bytes"), bool)
                    or full["size_bytes"] <= 0
                    or public.get("schema_version") not in {"true_rl_agent.push_recovery.behavior.public.v3",
                                                           "true_rl_agent.push_recovery.behavior.public.v4"}
                    or public.get("full_derivation_sha256") != full.get("sha256")):
                raise ValueError("Push process report differs from sealed full derivation")
        trajectory = evidence.get("behavior", {}).get("schema_version") == "true_rl_agent.push_recovery.behavior.public.v4"
        if trajectory != ("trajectory_build" in evidence) or (trajectory and "behavior_full" not in evidence):
            raise ValueError("Push trajectory requires its full and CPU build receipts")
        summaries = evidence["seed_summaries"]
        if not isinstance(summaries, list) or len(summaries) != self.seed_count(result.phase):
            raise ValueError("Push phase evidence seed count mismatch")
        if any(not isinstance(item, Mapping) for item in summaries):
            raise ValueError("Push phase summaries must be objects")
        if self.config.push_evaluation_protocol is not None and any(
            item.get("schema_version") != FORCE_ROLLOUT_SCHEMA for item in summaries
        ):
            raise ValueError("Push force evaluation requires v3 rollout evidence")
        errors: list[dict[str, Any]] = []
        project_push_split(summaries, split_name=result.phase.value, errors=errors)
        if errors:
            raise ValueError("Push phase contains incomplete or invalid rollout summaries")
        if trajectory:
            from .trajectory_pipeline import validate_build_receipt
            from .trajectory_report import require_phase_binding
            require_phase_binding(evidence["behavior"], self._trajectory_phase_binding(result))
            validate_build_receipt(evidence["trajectory_build"], evidence["behavior"])
        return summaries

    @staticmethod
    def _trajectory_phase_binding(result: EvaluationPhaseResult) -> dict:
        from .trajectory_report import IDENTITY_FIELDS
        return {**{k: getattr(result, k) for k in IDENTITY_FIELDS if k not in {"subject_kind", "subject_id"}},
                "subject_kind": result.subject.kind, "subject_id": result.subject.subject_id,
                "evaluation_binding_hash": result.evaluation_binding_hash,
                "manifest_sha256": sorted(s["numeric_evidence_ref"]["manifest_sha256"]
                                          for s in result.task_evidence["seed_summaries"])}

    def project_reference_public_dev_gate(self, *, phase_result):
        if self.config.push_protocol is None:
            return None
        if phase_result.phase is not EvaluationPhase.PUBLIC_DEV:
            raise ValueError("reference screening requires public-dev evidence")
        from true_rl_agent.tasks.push_recovery.reference_program import (
            reference_dev_evidence_passes, reference_qualification_contract,
        )

        summaries = self._phase_summaries(phase_result)
        dev = project_push_split(summaries, split_name="public_dev", errors=[])
        score = dev["score_stats"]["mean"]
        evidence = push_recovery_evidence_gate(summaries)
        contract = reference_qualification_contract(self.config)
        return {
            "status": "pass" if evidence["status"] == "pass" and reference_dev_evidence_passes(self.config, dev) else "fail",
            "qualification_contract_hash": canonical_json_hash(contract),
            "public_dev_score": score,
            **({"public_dev_reference_band": contract["public_dev_reference_band"]}
               if self.config.reference_admission_policy is None
               else {"reference_admission_policy": self.config.reference_admission_policy}),
            "recovery_evidence_status": evidence["status"],
        }

    def project_public_phase_summary(self, *, phase_result: EvaluationPhaseResult) -> Mapping[str, Any]:
        if phase_result.phase is EvaluationPhase.SEALED_VALIDATION:
            raise ValueError("sealed phase summaries cannot enter generation trial history")
        summary = project_push_split(self._phase_summaries(phase_result),
                                     split_name=phase_result.phase.value, errors=[])
        if "behavior" in phase_result.task_evidence:
            from true_rl_agent.tasks.push_recovery.behavior import project_push_behavior
            behavior = project_push_behavior(phase_result.task_evidence["behavior"])
            binding = behavior.get("domain", behavior.get("binding", {}))
            if binding["checkpoint_hash"] != phase_result.checkpoint_hash:
                raise ValueError("Push behavior checkpoint differs from its phase receipt")
            summary["behavior"] = behavior
        return summary

    def project_public_dev_screen(
        self, *, trial_id: str, phase_result: EvaluationPhaseResult,
        screen_role: str, train_result: TrainJobResult | None,
        train_receipt_checksum: str | None,
        mechanism_operation_delta: Mapping[str, Any] | None,
    ) -> PushPublicDevScreen:
        if phase_result.phase is not EvaluationPhase.PUBLIC_DEV:
            raise ValueError("Push selection requires public-dev evidence")
        if screen_role == "candidate":
            if not isinstance(train_result, TrainJobResult):
                raise ValueError("Push candidate screen requires completed training")
            require_sha256_checksum(train_receipt_checksum, "Push screen train receipt")
            if train_result.checkpoint.checksum != phase_result.checkpoint_hash or any(
                getattr(train_result, name) != getattr(phase_result, name)
                for name in ("subject", "subject_manifest_hash", "program_source_checksum",
                             "training_binding_hash", "capability_profile_hash", "execution_device")
            ):
                raise ValueError("Push candidate screen train/eval lineage mismatch")
        elif screen_role != "guard_reference" or train_result is not None or train_receipt_checksum is not None:
            raise ValueError("Push screen role or training evidence is invalid")
        summaries = self._phase_summaries(phase_result)
        if self.config.push_protocol is not None:
            behavior_passed = push_recovery_evidence_gate(summaries)["status"] == "pass"
        else:
            motion = push_motion_quality_gate(summaries, limits=DEFAULT_MOTION_LIMITS, prior_errors=[])
            behavior = push_behavior_quality_gate(public_dev_summaries=summaries,
                public_accept_summaries=[], limits=DEFAULT_MOTION_LIMITS, prior_errors=[])
            behavior_passed = motion["status"] == behavior["status"] == "pass"
        return PushPublicDevScreen(
            trial_id=trial_id, screen_role=screen_role,
            phase_result_hash=canonical_json_hash(phase_result.to_private_dict()),
            evaluation_plan_hash=phase_result.evaluation_plan_hash,
            checkpoint_hash=phase_result.checkpoint_hash,
            evaluation_binding_hash=phase_result.evaluation_binding_hash,
            execution_device=phase_result.execution_device,
            resource_binding_hash=phase_result.resource_binding_hash,
            train_receipt_checksum=train_receipt_checksum,
            score_stats=project_push_split(summaries, split_name="public_dev", errors=[])["score_stats"],
            behavior_passed=behavior_passed,
        )

    def select(self, screens: Sequence[PushPublicDevScreen], *, guard_reference: PushPublicDevScreen) -> PushSelection:
        ordered = sorted(screens, key=lambda item: item.trial_id)
        if len(ordered) != 2 or ordered[0].trial_id == ordered[1].trial_id:
            raise ValueError("Push selection requires two distinct trials")
        if guard_reference.screen_role != "guard_reference" or any(
            item.screen_role != "candidate"
            or item.evaluation_binding_hash != guard_reference.evaluation_binding_hash
            for item in ordered
        ):
            raise ValueError("Push selection screen role or comparison domain mismatch")
        eligible = [item for item in ordered if item.behavior_passed]
        winner = max(eligible, key=lambda item: item.score_stats["mean"]) if eligible else None
        return PushSelection(
            selected_trial_id=winner.trial_id if winner else None,
            contract_hash=self.selection_contract()["contract_hash"],
            screen_evidence_hashes=tuple(item.evidence_hash for item in ordered),
            guard_reference_evidence_hash=guard_reference.evidence_hash,
        )

    def project_public_accept_scores(self, *, phase_result):
        from true_rl_agent.evaluation.public_accept_scores import PublicAcceptScores

        if phase_result.phase is not EvaluationPhase.PUBLIC_ACCEPT:
            raise ValueError("score comparison requires public-accept evidence")
        summary = self.project_public_phase_summary(phase_result=phase_result)
        return PublicAcceptScores(score_stats={**summary["score_stats"],
            "eval_protocol_hash": self.config.evaluation_protocol().protocol_hash})

    def _assemble(self, eval_plan: EvalJobSpec, phase_results, phase_receipt_refs, *, trial_id, selection):
        if (eval_plan.task_id != "unitree_push_recovery"
            or eval_plan.evaluation_binding.task_protocol_hash != self.task_protocol_hash
            or eval_plan.evaluation_binding.acceptance_protocol_hash != self.config.acceptance_protocol().protocol_hash
            or eval_plan.evaluation_binding.seed_cohort_hash != self.config.seed_cohort_hash()):
            raise ValueError("Push evaluation plan differs from the task contract")
        by_phase = {}
        for result in phase_results:
            require_phase_result_matches_plan(result, eval_plan)
            if result.phase in by_phase:
                raise ValueError("Push assembly contains a duplicate phase")
            by_phase[result.phase] = result
        if set(by_phase) != set(EvaluationPhase):
            raise ValueError("Push assembly requires every evaluation phase")
        summaries = {phase: self._phase_summaries(result) for phase, result in by_phase.items()}
        first = by_phase[EvaluationPhase.PUBLIC_DEV]
        for result in by_phase.values():
            if any(getattr(result, name) != getattr(first, name) for name in (
                "environment_source_preflight", "backend_execution_environment_verification", "checkpoint_safety_report",
            )):
                raise ValueError("Push phase execution environment evidence differs")
        projected = build_push_public_result_from_summaries(
            public_dev_summaries=summaries[EvaluationPhase.PUBLIC_DEV],
            public_accept_summaries=summaries[EvaluationPhase.PUBLIC_ACCEPT],
            min_public_accept_count=self.config.public_accept_seed_count,
            acceptance_protocol=self.config.acceptance_protocol(),
            evaluation_protocol=self.config.evaluation_protocol(),
        )
        public_metrics = dict(projected.metrics)
        if "behavior" in first.task_evidence:
            public_metrics["public_dev"] = self.project_public_phase_summary(phase_result=first)
            if first.task_evidence["behavior"]["schema_version"] == "true_rl_agent.push_recovery.behavior.public.v4":
                public_metrics["trajectory_phase_binding"] = self._trajectory_phase_binding(first)
        sealed = build_push_sealed_rollout_evidence(
            summaries=summaries[EvaluationPhase.SEALED_VALIDATION],
            evaluation_protocol=self.config.evaluation_protocol(),
            checkpoint_hash=eval_plan.execution_checkpoint().checksum,
            evaluation_binding_hash=eval_plan.evaluation_binding.binding_hash,
            evaluation_subject=eval_plan.subject,
            subject_manifest_hash=eval_plan.subject_manifest_hash,
            program_source_checksum=eval_plan.program_source_checksum,
            training_binding_hash=eval_plan.training_binding_hash,
            capability_profile_hash=eval_plan.capability_profile_hash,
            acceptance_protocol=self.config.acceptance_protocol(),
            public_accept_score=projected.metrics["public_accept"]["score_stats"]["mean"],
        )
        summary = {
            "schema_version": ("true_rl_agent.push_recovery.eval_summary.v3"
                               if self.config.push_evaluation_protocol is not None
                               else "true_rl_agent.push_recovery.eval_summary.v2"
                               if self.config.push_protocol is not None
                               else "true_rl_agent.push_recovery.eval_summary.v1"),
            **eval_plan.job_identity().to_public_dict(),
            "evaluation_protocol": self.config.evaluation_protocol().to_public_dict(),
            "evaluation_binding": eval_plan.evaluation_binding.to_public_dict(),
            "status": projected.status, "public_score": projected.public_score,
            "metrics": public_metrics,
            "execution_device": first.execution_device,
            "environment_source_preflight": dict(first.environment_source_preflight),
            "backend_execution_environment_verification": dict(first.backend_execution_environment_verification),
            "checkpoint_safety_report": dict(first.checkpoint_safety_report),
            "evaluation_execution_cost": merge_phase_costs(by_phase),
            "sealed_validation_evidence": sealed,
            "sealed_validation_gate": build_sealed_validation_gate(
                rollout_evidence=sealed,
                checkpoint_safety_status=str(first.checkpoint_safety_report["status"]),
                checkpoint_hash_match_status=str(first.checkpoint_safety_report["checkpoint_hash_match_status"]),
                candidate_isolation_gate=build_candidate_isolation_gate(),
                expected_eval_protocol_hash=eval_plan.evaluation_protocol().protocol_hash,
                expected_evaluation_binding_hash=eval_plan.evaluation_binding.binding_hash,
            ),
            "candidate_code_loaded_in_eval": False,
            "candidate_package_loaded": False,
            "candidate_package_loaded_in_eval": False,
            "candidate_path_argument_present": False,
            "errors": [],
        }
        return EvaluationAssembly(
            evaluation_plan_hash=canonical_json_hash(eval_plan.to_public_dict()),
            selected_trial_id=trial_id, selection=selection,
            phase_receipts=bind_phase_receipts(eval_plan=eval_plan, by_phase=by_phase,
                                              phase_receipt_refs=phase_receipt_refs),
            summary=summary,
        )

    def assemble_full_summary(self, *, eval_plan, trial_id, selection_result,
                              selection_screens, guard_reference, phase_results, phase_receipt_refs):
        replayed = self.select(selection_screens, guard_reference=guard_reference)
        if replayed != selection_result or replayed.selected_trial_id != trial_id:
            raise ValueError("Push assembly selection replay mismatch")
        selected = next(item for item in selection_screens if item.trial_id == trial_id)
        dev = next(item for item in phase_results if item.phase is EvaluationPhase.PUBLIC_DEV)
        expected = self.project_public_dev_screen(
            trial_id=trial_id, phase_result=dev, screen_role="guard_reference",
            train_result=None, train_receipt_checksum=None, mechanism_operation_delta=None,
        ).to_selector_dict()
        actual = selected.to_selector_dict()
        for name in ("screen_role", "train_receipt_checksum"):
            expected.pop(name)
            actual.pop(name)
        if actual != expected:
            raise ValueError("Push selected screen differs from the evaluated phase")
        return self._assemble(eval_plan, phase_results, phase_receipt_refs,
                              trial_id=trial_id, selection=selection_result.to_public_dict())

    def assemble_reference_full_summary(self, *, eval_plan, phase_results, phase_receipt_refs):
        if eval_plan.subject.kind != "reference":
            raise ValueError("Push reference assembly requires a reference plan")
        return self._assemble(eval_plan, phase_results, phase_receipt_refs,
            trial_id=eval_plan.subject.subject_id, selection={
                "schema_version": "true_rl_agent.task.push_recovery.reference_eval_assembly.v1",
                "selected_trial_id": eval_plan.subject.subject_id,
                "selection_kind": "reference_program_qualification", "acceptance_authority": False,
            })
