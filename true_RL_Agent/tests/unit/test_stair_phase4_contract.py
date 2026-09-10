"""Phase 4 contracts for the unified Stair reference-owned objective mask."""

from __future__ import annotations

import ast
import copy
import inspect
import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import MappingProxyType

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.agent_core.experiment_memory import build_experiment_memory
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
)
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.exploration import StairExplorationProjector
from true_rl_agent.tasks.stair_climbing.memory import StairTaskMemoryAdapter
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.prompt_adapter import (
    StairClimbingPromptAdapter,
    _compact_required_feedback_entry,
    _latest_generation_trial_comparisons,
    _stair_unified_capacity_preflight_feedback,
)
from true_rl_agent.tasks.stair_climbing.reference_programs.official_g1_rough_v1.training_code import (
    components as legacy_official_components,
)
from true_rl_agent.tasks.stair_climbing.reference_programs.official_g1_rough_vnext_v1.training_code import (
    components as vnext_official_components,
)
from true_rl_agent.tasks.stair_climbing.reference_qualification import (
    qualify_stair_reference_public_result,
)
from true_rl_agent.tasks.stair_climbing.robot_binding import (
    resolve_stair_cell_manifest,
)
from true_rl_agent.tasks.stair_climbing.selection import (
    StairPublicDevScreenEvidence,
    classify_stair_selector_trial_eligibility,
)
from true_rl_agent.tasks.stair_climbing.unified_scoring import (
    build_reference_objective_family_mask,
    build_unified_reference_score_projection,
    build_unified_score_projection,
    build_unified_stair_prompt_feedback,
    build_unified_stair_summary,
    objective_family_mask_from_public_result,
    validate_objective_family_mask,
)


class StairPhase4ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        protocol_path = ROOT / "configs/tasks/stair_unified_protocol_v1.json"
        base = StairClimbingTaskConfig.defaults(protocol_path)
        manifest = resolve_stair_cell_manifest(
            cell_variant_id=base.cell_variant_id,
            config=base,
            plugin=StairClimbingTaskPlugin(base),
        )
        cls.config = StairClimbingTaskConfig.defaults(
            protocol_path,
            resolved_cell_task_protocol_hash=(
                manifest.resolved_cell_task_protocol_hash
            ),
        )
        cls.manifest = manifest
        cls.summary = cls._summary_for_split("public_accept")
        cls.public_dev_summary = cls._summary_for_split("public_dev")

    @classmethod
    def _summary_for_split(
        cls,
        split: str,
        *,
        success_counts: dict[str, int] | None = None,
        progress: dict[str, float] | None = None,
    ) -> dict[str, object]:
        suite = cls.config.ordered_evaluation_suite
        success_counts = success_counts or {
            "c0_id": 1,
            "c1_id": 1,
            "height_ood": 0,
            "tread_ood": 0,
            "joint_ood": 0,
            "phase_composition_ood": 3,
            "c1_conditional_frontier": 3,
            "c2_mechanism_ood": 0,
        }
        progress = progress or {
            "c0_id": 0.8,
            "c1_id": 0.8,
            "height_ood": 0.4,
            "tread_ood": 0.4,
            "joint_ood": 0.4,
            "phase_composition_ood": 0.9,
            "c1_conditional_frontier": 0.9,
            "c2_mechanism_ood": 0.1,
        }
        per_seed = []
        protocol = cls.config.evaluation_protocol()
        seed_count = {
            "public_dev": protocol.public_seed_count,
            "public_accept": protocol.public_accept_seed_count,
            "sealed_validation": protocol.sealed_validation_seed_count,
        }[split]
        for _ in range(seed_count):
            occurrences: Counter[str] = Counter()
            outcomes = []
            for env_index, cell in enumerate(
                suite.private_cell_assignment(split)
            ):
                occurrence = occurrences[cell.cell_id]
                occurrences[cell.cell_id] += 1
                succeeded = occurrence < success_counts[cell.stratum_id]
                fraction = progress[cell.stratum_id]
                outcomes.append(
                    StairEpisodeOutcome(
                        env_index=env_index,
                        bin_id=cell.cell_id,
                        episode_steps=cell.episode_horizon_control_steps,
                        primary_reason="success" if succeeded else "timeout",
                        success=succeeded,
                        diagnostic_flags=(
                            ("first_valid_top_arrival_achieved",)
                            if succeeded
                            else ()
                        ),
                        max_progress_m=fraction * cell.progress_goal_m,
                        progress_fraction=fraction,
                        active_terminal_terms=() if succeeded else ("time_out",),
                        evaluation_cell_id=cell.cell_id,
                    )
                )
            per_seed.append(outcomes)
        return build_unified_stair_summary(
            suite=suite,
            semantics=cls.config.evaluation_semantics,
            split=split,
            per_seed_outcomes=per_seed,
        )

    def test_reference_mask_excludes_only_floor_families(self) -> None:
        suite = self.config.ordered_evaluation_suite
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=self.summary,
        )
        self.assertEqual(mask["qualified_ood_families"], ["F_scalar", "F_c1"])
        self.assertEqual(
            validate_objective_family_mask(
                suite=suite,
                mask=mask,
                public_accept_summary=self.summary,
            ),
            mask,
        )
        projection = build_unified_score_projection(
            self.summary,
            objective_family_mask=mask,
        )
        self.assertEqual(
            projection["objective_family_mask_hash"],
            mask["objective_family_mask_hash"],
        )
        self.assertEqual(
            projection["qualified_ood_families"],
            ["F_scalar", "F_c1"],
        )
        self.assertAlmostEqual(
            projection["j_ood"]["score_stats"]["mean"],
            (
                self.summary["family_scores"]["F_scalar"]
                + self.summary["family_scores"]["F_c1"]
            )
            / 2.0,
        )

    def test_reference_mask_loader_accepts_frozen_genesis_public_result(self) -> None:
        def freeze(value: object) -> object:
            if isinstance(value, dict):
                return MappingProxyType(
                    {str(key): freeze(item) for key, item in value.items()}
                )
            if isinstance(value, list):
                return tuple(freeze(item) for item in value)
            return value

        suite = self.config.ordered_evaluation_suite
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=self.summary,
        )
        public_result = freeze(
            {"metrics": {"objective_family_mask": mask}}
        )

        self.assertEqual(
            objective_family_mask_from_public_result(
                suite=suite,
                public_result=public_result,
            ),
            mask,
        )

    def test_all_floor_reference_returns_diagnostic_failure_not_exception(self) -> None:
        suite = self.config.ordered_evaluation_suite
        floor_summary = self._summary_for_split(
            "public_accept",
            success_counts={
                "c0_id": 1,
                "c1_id": 1,
                "height_ood": 0,
                "tread_ood": 0,
                "joint_ood": 0,
                "phase_composition_ood": 0,
                "c1_conditional_frontier": 0,
                "c2_mechanism_ood": 0,
            },
            progress={
                "c0_id": 0.8,
                "c1_id": 0.8,
                "height_ood": 0.05,
                "tread_ood": 0.05,
                "joint_ood": 0.05,
                "phase_composition_ood": 0.05,
                "c1_conditional_frontier": 0.05,
                "c2_mechanism_ood": 0.05,
            },
        )
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=floor_summary,
        )
        self.assertEqual(mask["qualified_ood_families"], [])
        self.assertEqual(
            validate_objective_family_mask(
                suite=suite,
                mask=mask,
                public_accept_summary=floor_summary,
                require_nonempty=False,
            ),
            mask,
        )
        diagnostic_projection = build_unified_reference_score_projection(
            floor_summary,
            objective_family_mask=mask,
        )
        self.assertEqual(
            diagnostic_projection["aggregation_mode"],
            "diagnostic_all_declared_families_empty_reference_mask",
        )
        self.assertEqual(diagnostic_projection["qualified_ood_families"], [])
        self.assertEqual(
            diagnostic_projection["objective_family_mask_hash"],
            mask["objective_family_mask_hash"],
        )
        floor_dev_summary = self._summary_for_split(
            "public_dev",
            success_counts={
                "c0_id": 1,
                "c1_id": 1,
                "height_ood": 0,
                "tread_ood": 0,
                "joint_ood": 0,
                "phase_composition_ood": 0,
                "c1_conditional_frontier": 0,
                "c2_mechanism_ood": 0,
            },
            progress={
                "c0_id": 0.8,
                "c1_id": 0.8,
                "height_ood": 0.05,
                "tread_ood": 0.05,
                "joint_ood": 0.05,
                "phase_composition_ood": 0.05,
                "c1_conditional_frontier": 0.05,
                "c2_mechanism_ood": 0.05,
            },
        )
        feedback = build_unified_stair_prompt_feedback(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            summary=floor_dev_summary,
            objective_family_mask=mask,
        )
        self.assertEqual(
            feedback["generalization_summary"]["qualified_ood_families"],
            [],
        )
        self.assertEqual(
            feedback["generalization_summary"]["objective_family_mask_hash"],
            mask["objective_family_mask_hash"],
        )
        result = qualify_stair_reference_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=float(
                    diagnostic_projection["j_primary"]["score_stats"]["mean"]
                ),
                metrics={
                    "public_accept": floor_summary,
                    "objective_family_mask": mask,
                    "acceptance": {"required_pass_gates": []},
                },
            ),
        )
        self.assertFalse(result.passed)
        self.assertEqual(
            result.private_evidence["task_gate"]["ood_family_status"],
            "fail",
        )
        self.assertEqual(
            result.private_evidence["task_gate"]["reason"],
            "unified_stair_reference_readiness_not_met",
        )

    def test_exploration_compares_only_public_dev_to_public_dev_parent(self) -> None:
        mask = build_reference_objective_family_mask(
            suite=self.config.ordered_evaluation_suite,
            public_accept_summary=self.summary,
        )
        projection = build_unified_score_projection(
            self.public_dev_summary,
            objective_family_mask=mask,
        )
        candidate_score = float(projection["j_primary"]["score_stats"]["mean"])
        guard_names = tuple(
            guard.name for guard in self.config.acceptance_protocol.named_guards
        )
        candidate_guard_floor = min(
            float(projection[name]["score_stats"]["mean"])
            for name in guard_names
        )
        trial_id = "phase4-public-dev-trial"
        trial = {
            "trial_id": trial_id,
            "trial_status": "qualified_not_selected",
            "public_dev": {"public_summary": self.public_dev_summary},
            "public_accept": {"marker": "must_not_be_read"},
        }
        generation = {
            "selection": {"selected_trial_id": trial_id},
            "guard_reference": {
                "screen_role": "guard_reference",
                "acceptance_authority": False,
                "generalization_score_stats": {"mean": candidate_score - 0.05},
                "anchor_score_stats": {"mean": candidate_guard_floor},
            },
        }
        assessment = StairExplorationProjector(
            self.config,
            objective_family_mask=mask,
        ).assess_trial(
            generation=generation,
            trial=trial,
            parent=None,
        )

        self.assertEqual(assessment.observed_effect, "strong_positive")
        self.assertEqual(assessment.value_label, "high")
        self.assertAlmostEqual(
            assessment.public_evidence_summary["objective"]["delta"],
            0.05,
        )
        self.assertEqual(
            assessment.public_evidence_summary["source_split"],
            "public_dev",
        )
        self.assertNotIn("materiality_threshold", str(assessment.public_evidence_summary))

        trial["selection_screen"] = {
            "runtime_effect_evidence": {
                "schema_version": "true_rl_agent.task.stair_climbing.candidate_runtime_effect_evidence.v1",
                "status": "fail",
                "reason": "candidate_reward_operation_runtime_effect_not_observed",
                "acceptance_authority": False,
            }
        }
        runtime_failed = StairExplorationProjector(
            self.config,
            objective_family_mask=mask,
        ).assess_trial(
            generation=generation,
            trial=trial,
            parent=None,
        )
        self.assertEqual(runtime_failed.observed_effect, "negative")
        self.assertEqual(runtime_failed.value_label, "low")
        self.assertEqual(
            runtime_failed.reason_codes,
            ("candidate_reward_operation_runtime_effect_not_observed",),
        )

        generation["guard_reference"]["acceptance_authority"] = True
        rejected_reference = StairExplorationProjector(
            self.config,
            objective_family_mask=mask,
        ).assess_trial(
            generation=generation,
            trial=trial,
            parent=None,
        )
        self.assertEqual(rejected_reference.observed_effect, "unknown")
        self.assertEqual(
            rejected_reference.reason_codes,
            ("parent_public_dev_reference_missing",),
        )

    def test_legacy_v3_screen_evidence_keeps_its_exact_identity(self) -> None:
        legacy = StairPublicDevScreenEvidence(
            trial_id="legacy-screen",
            screen_role="guard_reference",
            train_receipt_checksum=None,
            runtime_effect_evidence=None,
            phase_result_hash="1" * 64,
            evaluation_plan_hash="2" * 64,
            task_id="unitree_stair_climbing",
            subject=ExperimentSubject.reference("legacy-reference"),
            checkpoint_hash="sha256:" + "3" * 64,
            eval_protocol_hash="4" * 64,
            evaluation_binding_hash="5" * 64,
            training_binding_hash="6" * 64,
            capability_profile_hash="7" * 64,
            execution_environment_hash="8" * 64,
            execution_device="cuda:0",
            resource_binding_hash=None,
            anchor_score_stats={"mean": 0.5},
            anchor_seed_macro_scores=(0.5,),
            generalization_score_stats={"mean": 0.4},
            generalization_seed_macro_scores=(0.4,),
            schema_version=(
                "true_rl_agent.task.stair_climbing."
                "public_dev_screen_evidence.v3"
            ),
        )
        payload = legacy.to_selector_dict()
        evidence_hash = legacy.evidence_hash

        self.assertNotIn("objective_family_mask_hash", payload)
        replayed = StairPublicDevScreenEvidence.from_mapping(payload)
        self.assertEqual(replayed.to_selector_dict(), payload)
        self.assertEqual(replayed.evidence_hash, evidence_hash)

    def test_vnext_reference_is_the_15_term_official_program_under_roles(self) -> None:
        legacy = legacy_official_components.build_training_components()
        vnext = vnext_official_components.build_training_components()
        expected = [(term.name, term.weight) for term in legacy.reward_terms]
        self.assertEqual(len(expected), 15)
        self.assertEqual(
            [(term.name, term.weight) for term in vnext.reward_terms],
            expected,
        )
        role_to_legacy = {
            "support_contact_time": "foot_contact_time",
            "support_air_time": "foot_air_time",
            "support_contact_force_history": "foot_contact_force_history",
            "support_lin_vel": "foot_lin_vel",
            "ankle_group_soft_pos_limits": "ankle_soft_joint_pos_limits",
            "ankle_group_pos": "ankle_joint_pos",
            "hip_yaw_roll_group_pos_error": "hip_yaw_roll_joint_pos_error",
            "arm_group_pos_error": "arm_joint_pos_error",
            "finger_group_pos_error": "finger_joint_pos_error",
            "torso_group_pos_error": "torso_joint_pos_error",
            "hip_knee_group_acc": "hip_knee_joint_acc",
            "locomotion_group_torque": "locomotion_joint_torque",
        }

        class NormalizeRoleFields(ast.NodeTransformer):
            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                self.generic_visit(node)
                node.attr = role_to_legacy.get(node.attr, node.attr)
                return node

        for legacy_term, vnext_term in zip(
            legacy.reward_terms,
            vnext.reward_terms,
            strict=True,
        ):
            expected_tree = ast.dump(
                ast.parse(inspect.getsource(legacy_term.fn)),
                include_attributes=False,
            )
            vnext_tree = NormalizeRoleFields().visit(
                ast.parse(inspect.getsource(vnext_term.fn))
            )
            self.assertEqual(
                ast.dump(
                    ast.fix_missing_locations(vnext_tree),
                    include_attributes=False,
                ),
                expected_tree,
                legacy_term.name,
            )

    def test_mask_replay_and_plugin_binding_reject_drift(self) -> None:
        suite = self.config.ordered_evaluation_suite
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=self.summary,
        )
        incoherent = copy.deepcopy(mask)
        incoherent["family_measurements"]["F_scalar"]["qualified"] = False
        payload = dict(incoherent)
        payload.pop("objective_family_mask_hash")
        incoherent["objective_family_mask_hash"] = canonical_json_hash(payload)
        with self.assertRaisesRegex(ValueError, "qualification predicate"):
            validate_objective_family_mask(suite=suite, mask=incoherent)

        plugin = StairClimbingTaskPlugin(
            self.config,
            resolved_cell_manifest=self.manifest,
        )
        plugin.bind_objective_family_mask(mask)
        self.assertEqual(plugin.objective_family_mask(), mask)
        memory = StairTaskMemoryAdapter(
            config=self.config,
            objective_family_mask_provider=plugin.objective_family_mask,
        )
        self.assertEqual(
            memory.exploration_projector().objective_family_mask,
            mask,
        )
        prompt = plugin.prompt_adapter()
        self.assertEqual(prompt.objective_family_mask_provider(), mask)
        different = copy.deepcopy(mask)
        different["family_measurements"]["F_c1"].update(
            {"score": 0.5, "macro_success_rate": 0.5, "qualified": True}
        )
        different["qualified_ood_families"] = ["F_scalar", "F_c1"]
        payload = dict(different)
        payload.pop("objective_family_mask_hash")
        different["objective_family_mask_hash"] = canonical_json_hash(payload)
        with self.assertRaisesRegex(ValueError, "already frozen"):
            plugin.bind_objective_family_mask(different)

    def test_prompt_objective_uses_the_frozen_mask(self) -> None:
        suite = self.config.ordered_evaluation_suite
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=self.summary,
        )
        feedback = build_unified_stair_prompt_feedback(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            summary=self.public_dev_summary,
            objective_family_mask=mask,
        )
        generalization = feedback["generalization_summary"]
        self.assertEqual(
            generalization["objective_family_mask_hash"],
            mask["objective_family_mask_hash"],
        )
        self.assertEqual(
            generalization["qualified_ood_families"],
            ["F_scalar", "F_c1"],
        )
        self.assertAlmostEqual(
            generalization["j_ood"],
            (
                self.public_dev_summary["family_scores"]["F_scalar"]
                + self.public_dev_summary["family_scores"]["F_c1"]
            )
            / 2.0,
        )
        compact = _compact_required_feedback_entry(
            {"feedback": feedback},
            evaluation_profile=None,
        )["feedback"]["generalization_summary"]
        self.assertEqual(
            compact["qualified_ood_families"],
            ["F_scalar", "F_c1"],
        )
        self.assertEqual(
            compact["objective_family_mask_hash"],
            mask["objective_family_mask_hash"],
        )

    def test_provider_view_uses_public_dev_without_acceptance_result_leak(
        self,
    ) -> None:
        suite = self.config.ordered_evaluation_suite
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=self.summary,
        )
        feedback = build_unified_stair_prompt_feedback(
            suite=suite,
            semantics=self.config.evaluation_semantics,
            summary=self.public_dev_summary,
            objective_family_mask=mask,
        )
        public_dev_score = feedback["generalization_summary"][
            "public_dev_objective_score"
        ]
        public_dev_guard_floor = min(
            feedback["generalization_summary"]["named_guard_scores"].values()
        )
        hidden_marker = "internal-acceptance-result-must-not-reach-provider"
        checksum = "sha256:" + "a" * 64
        public_result = {
            "status": "pass",
            "public_score": 0.987654321,
            "metrics": {
                "prompt_feedback": feedback,
                "public_dev": self.public_dev_summary,
                "candidate_code_loaded_in_eval": True,
                "public_accept": self.summary,
                "sealed_validation": {
                    "marker": hidden_marker,
                    "score": 0.87654321,
                },
                "unified_score": {"j_primary": {"score": 0.987654321}},
                "acceptance": {
                    "marker": hidden_marker,
                    "materiality_threshold": 0.05,
                },
            },
        }
        record = {
            "round_id": "phase4-generation-001",
            "task": {"task_id": "unitree_stair_climbing"},
            "evaluation_binding": {
                "task_protocol_hash": self.config.task_protocol_hash,
            },
            "candidate": {
                "candidate_id": "phase4-candidate-a",
                "changed_components": ["reward_callable"],
            },
            "candidate_surface": {"classification": "code_level"},
            "train_result": {"program_source_checksum": checksum},
            "runtime_activation_summary": {
                "reward_term_calls": 2,
                "nonzero_reward_contribution": True,
                "capability_activation_records": ["complete_reward_program"],
                "component_activation_records": [
                    {
                        "kind": "reward_term",
                        "name": "progress",
                        "max_abs_weighted_contribution": 1.0,
                    }
                ],
            },
            "public_result": public_result,
            "decision": {
                "outcome": "reject",
                "accepted": False,
                "reason": hidden_marker,
                "surface_classification": "code_level",
                "public_score": 0.987654321,
                "comparison": {
                    "score_source": "internal.public_accept.j_primary",
                    "candidate_score": 0.987654321,
                    "current_best_score": 0.9,
                    "delta": 0.087654321,
                    "threshold": 0.05,
                },
                "gate_statuses": {
                    "runtime_activation_gate": "pass",
                    "sealed_validation_gate": "fail",
                },
            },
            "generation": {
                "schema_version": (
                    "true_rl_agent.generation_public_history_projection.v3"
                ),
                "generation_id": "phase4-generation-001",
                "generation_index": 1,
                "generation_outcome": "formal_rejected",
                "parent_current_best": {"public_score": 0.9},
                "guard_reference": {
                    "screen_role": "guard_reference",
                    "acceptance_authority": False,
                    "generalization_score_stats": {"mean": public_dev_score},
                    "anchor_score_stats": {"mean": public_dev_guard_floor},
                    "marker": hidden_marker,
                },
                "formal": {
                    "schema_version": "true_rl_agent.generation_formal_projection.v1",
                    "selected_lane_id": "lane-a",
                    "selected_trial_id": "phase4-trial-a",
                    "sealed_phase_receipt_checksum": "sha256:" + "b" * 64,
                    "decision_outcome": "reject",
                    "accepted": False,
                },
                "selection": {
                    "selected_trial_id": "phase4-trial-a",
                    "reason": "public_dev_runtime_activation_filter",
                    "guard_regression_by_trial": {
                        "phase4-trial-a": 0.0,
                        "phase4-trial-b": 0.0,
                    },
                    "selection_eligible_trial_ids": [
                        "phase4-trial-a",
                    ],
                },
                "trials": [
                    {
                        "schema_version": "true_rl_agent.generation_trial.v1",
                        "lane_id": "lane-a",
                        "trial_id": "phase4-trial-a",
                        "trial_status": "formal_rejected",
                        "status_reason": hidden_marker,
                        "candidate": {
                            "candidate_id": "phase4-candidate-a",
                            "program_source_checksum": checksum,
                        },
                        "public_dev": {"public_summary": self.public_dev_summary},
                        "selection_screen": {
                            "runtime_effect_evidence": {
                                "schema_version": "true_rl_agent.task.stair_climbing.candidate_runtime_effect_evidence.v1",
                                "status": "pass",
                                "reason": "candidate_reward_operation_runtime_effect_observed",
                                "acceptance_authority": False,
                            }
                        },
                        "public_accept": {
                            "public_summary": self.summary,
                            "marker": hidden_marker,
                        },
                    }
                ],
            },
        }
        sibling = copy.deepcopy(record["generation"]["trials"][0])
        sibling["lane_id"] = "lane-b"
        sibling["trial_id"] = "phase4-trial-b"
        sibling["trial_status"] = "qualified_not_selected"
        sibling["candidate"] = {
            "candidate_id": "phase4-candidate-b",
            "program_source_checksum": "sha256:" + "d" * 64,
        }
        sibling["selection_screen"]["runtime_effect_evidence"] = {
            "schema_version": "true_rl_agent.task.stair_climbing.candidate_runtime_effect_evidence.v1",
            "status": "fail",
            "reason": "candidate_reward_operation_runtime_effect_not_observed",
            "acceptance_authority": False,
        }
        record["generation"]["trials"].append(sibling)
        adapter = StairClimbingPromptAdapter(
            self.config,
            resolved_cell_manifest=self.manifest,
            objective_family_mask_provider=lambda: mask,
        )

        projection = adapter.project_prompt_public_history(
            public_records=[record],
            initial_anchor_public_evidence={
                "anchor_kind": "reference_program",
                "public_result": public_result,
                "public_result_hash": "c" * 64,
            },
        )
        replayed_projection = adapter.project_prompt_public_history(
            public_records=projection.records,
            initial_anchor_public_evidence={
                "anchor_kind": "reference_program",
                "public_result": public_result,
                "public_result_hash": "c" * 64,
            },
        )
        self.assertEqual(replayed_projection, projection)
        projected = projection.records[0]
        self.assertEqual(projection.initial_anchor_score, public_dev_score)
        self.assertEqual(
            projection.public_score_source,
            "prompt_feedback.generalization_summary.public_dev_objective_score",
        )
        self.assertEqual(projected["decision"]["public_score"], public_dev_score)
        self.assertEqual(
            set(projected["public_result"]["metrics"]),
            {"prompt_feedback", "candidate_code_loaded_in_eval"},
        )
        self.assertEqual(
            projected["decision"]["gate_statuses"],
            {"runtime_activation_gate": "pass"},
        )
        self.assertEqual(
            projected["decision"]["reason"],
            "runner_did_not_accept_candidate",
        )
        self.assertNotIn("parent_current_best", projected["generation"])
        self.assertNotIn("guard_reference", projected["generation"])
        self.assertNotIn(
            "sealed_phase_receipt_checksum",
            projected["generation"]["formal"],
        )
        self.assertEqual(
            projected["generation"]["public_dev_parent_reference"],
            {
                "source_split": "public_dev",
                "acceptance_authority": False,
                "objective_score": public_dev_score,
                "guard_floor_score": public_dev_guard_floor,
            },
        )
        self.assertNotIn("public_accept", projected["generation"]["trials"][0])
        self.assertNotIn("status_reason", projected["generation"]["trials"][0])
        projected_serialized = json.dumps(projected, sort_keys=True)
        self.assertNotIn(hidden_marker, projected_serialized)
        self.assertNotIn('"public_accept"', projected_serialized)
        self.assertNotIn('"sealed_validation"', projected_serialized)
        self.assertNotIn("sealed_phase_receipt_checksum", projected_serialized)
        self.assertNotIn("0.987654321", projected_serialized)
        self.assertNotIn("0.87654321", projected_serialized)

        prompt_context = {
            "initial_anchor_public_evidence": {
                "anchor_kind": "reference_program",
                "public_result": public_result,
                "public_result_hash": "c" * 64,
            }
        }
        sections = adapter.build_prompt_sections(
            public_records=[record],
            public_context=prompt_context,
        )
        self.assertEqual(
            adapter.build_prompt_sections(
                public_records=projection.records,
                public_context=prompt_context,
            ),
            sections,
        )
        serialized = json.dumps(sections, sort_keys=True)
        self.assertNotIn(hidden_marker, serialized)
        comparisons = sections["public_feedback_summary"][
            "latest_generation_trial_comparisons"
        ]
        self.assertEqual(comparisons[0]["objective_score"], public_dev_score)
        self.assertEqual(comparisons[0]["objective_delta"], 0.0)
        self.assertEqual(comparisons[0]["selector_guard_regression"], 0.0)
        self.assertEqual(comparisons[0]["selector_guard_max_regression"], 0.05)
        self.assertTrue(comparisons[0]["selector_eligible"])
        self.assertEqual(
            comparisons[0]["generation_selector_reason"],
            "public_dev_runtime_activation_filter",
        )
        self.assertEqual(
            comparisons[0]["runtime_effect_evidence"]["status"],
            "pass",
        )
        self.assertEqual(comparisons[1]["selector_status"], "runtime_ineligible")
        self.assertEqual(comparisons[1]["observed_effect"], "negative")
        self.assertEqual(
            comparisons[1]["reason_codes"],
            ["candidate_reward_operation_runtime_effect_not_observed"],
        )
        preflight_comparisons = _stair_unified_capacity_preflight_feedback(
            self.config
        )["latest_generation_trial_comparisons"]
        self.assertEqual(len(preflight_comparisons), len(comparisons))
        for preflight, actual in zip(
            preflight_comparisons,
            comparisons,
            strict=True,
        ):
            self.assertEqual(set(preflight), set(actual))
            self.assertEqual(len(actual), 18)
        self.assertNotIn("materiality_threshold", comparisons[0])
        for entry in sections["public_feedback_summary"]["feedback_entries"]:
            self.assertEqual(entry["objective_score"], public_dev_score)
            self.assertNotIn("guard_score", entry)
        self.assertFalse(
            sections["public_eval_contract"]["prompt_score_semantics"][
                "acceptance_result_metrics_visible"
            ]
        )

        drifted_record = copy.deepcopy(record)
        drifted_record["public_result"]["metrics"]["prompt_feedback"][
            "generalization_summary"
        ]["public_dev_objective_score"] += 0.01
        with self.assertRaisesRegex(
            ValueError,
            "stored prompt feedback differs from public_dev",
        ):
            adapter.project_prompt_public_history(
                public_records=[drifted_record],
                initial_anchor_public_evidence=None,
            )

        memory = StairTaskMemoryAdapter(config=self.config).build_public_summary(
            [projected]
        )
        self.assertEqual(
            memory["public_dev_objective"]["latest"],
            public_dev_score,
        )
        self.assertEqual(
            memory["score_source"],
            "prompt_feedback.generalization_summary.public_dev_objective_score",
        )

    def test_provider_view_preserves_no_formal_generation_outcomes(self) -> None:
        hidden_reason = "private-runner-detail-must-not-reach-provider"
        adapter = StairClimbingPromptAdapter(
            self.config,
            resolved_cell_manifest=self.manifest,
        )
        for outcome in (
            "paired_authoring_incomplete",
            "candidate_execution_incomplete",
            "no_selection",
        ):
            with self.subTest(outcome=outcome):
                record = {
                    "round_id": f"phase4-generation-{outcome}",
                    "task": {"task_id": "unitree_stair_climbing"},
                    "candidate_surface": {"classification": "code_level"},
                    "decision": {
                        "outcome": "generation_head_retained",
                        "accepted": False,
                        "reason": hidden_reason,
                        "surface_classification": "code_level",
                    },
                    "generation": {
                        "schema_version": (
                            GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION
                        ),
                        "generation_id": f"phase4-generation-{outcome}",
                        "generation_index": 1,
                        "generation_outcome": outcome,
                        "selection": None,
                        "formal": None,
                        "trials": [],
                    },
                }

                projection = adapter.project_prompt_public_history(
                    public_records=[record],
                    initial_anchor_public_evidence=None,
                )
                replayed = adapter.project_prompt_public_history(
                    public_records=projection.records,
                    initial_anchor_public_evidence=None,
                )

                self.assertEqual(replayed, projection)
                projected = projection.records[0]
                self.assertEqual(projected["decision"]["reason"], outcome)
                self.assertEqual(
                    projected["generation"]["generation_outcome"],
                    outcome,
                )
                memory = build_experiment_memory(projection.records)
                self.assertEqual(memory["experiments"][0]["reason"], outcome)
                self.assertNotIn(
                    hidden_reason,
                    json.dumps(projected, sort_keys=True),
                )

    def test_unified_selectionless_generation_is_not_selector_feedback(self) -> None:
        cases = (
            (
                "paired_authoring_incomplete",
                [],
                True,
            ),
            (
                "candidate_execution_incomplete",
                [{"public_dev": {"public_summary": {}}}],
                True,
            ),
            (
                "candidate_execution_incomplete",
                [],
                False,
            ),
        )
        for outcome, trials, include_selection in cases:
            with self.subTest(
                outcome=outcome,
                include_selection=include_selection,
            ):
                generation = {
                    "schema_version": (
                        GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION
                    ),
                    "generation_id": "selectionless-generation",
                    "generation_outcome": outcome,
                    "trials": trials,
                }
                if include_selection:
                    generation["selection"] = None
                self.assertEqual(
                    _latest_generation_trial_comparisons(
                        [{"generation": generation}],
                        config=self.config,
                    ),
                    [],
                )

        malformed = {
            "schema_version": GENERATION_PUBLIC_HISTORY_PROJECTION_SCHEMA_VERSION,
            "generation_id": "malformed-selection-generation",
            "generation_outcome": "candidate_execution_incomplete",
            "trials": [],
            "selection": [],
        }
        with self.assertRaisesRegex(
            ValueError,
            "selector evidence must be an object or null",
        ):
            _latest_generation_trial_comparisons(
                [{"generation": malformed}],
                config=self.config,
            )

    def test_selector_eligibility_classifier_owns_boundary_semantics(self) -> None:
        self.assertEqual(
            classify_stair_selector_trial_eligibility(
                guard_regression=0.05,
                guard_max_regression=0.05,
                runtime_effect_status="pass",
            ),
            "eligible",
        )
        self.assertEqual(
            classify_stair_selector_trial_eligibility(
                guard_regression=0.0500001,
                guard_max_regression=0.05,
                runtime_effect_status="pass",
            ),
            "guard_ineligible",
        )
        self.assertEqual(
            classify_stair_selector_trial_eligibility(
                guard_regression=0.0,
                guard_max_regression=0.05,
                runtime_effect_status="fail",
            ),
            "runtime_ineligible",
        )
        self.assertEqual(
            classify_stair_selector_trial_eligibility(
                guard_regression=0.1,
                guard_max_regression=0.05,
                runtime_effect_status=None,
            ),
            "guard_and_runtime_ineligible",
        )

    def test_reference_qualification_binds_the_replayed_mask(self) -> None:
        suite = self.config.ordered_evaluation_suite
        mask = build_reference_objective_family_mask(
            suite=suite,
            public_accept_summary=self.summary,
        )
        projection = build_unified_score_projection(
            self.summary,
            objective_family_mask=mask,
        )
        result = qualify_stair_reference_public_result(
            config=self.config,
            public_result=PublicEvalResult(
                status="pass",
                public_score=float(projection["j_primary"]["score_stats"]["mean"]),
                metrics={
                    "public_accept": self.summary,
                    "objective_family_mask": mask,
                    "acceptance": {"required_pass_gates": []},
                },
            ),
        )
        self.assertTrue(result.passed)
        self.assertEqual(
            result.private_evidence["objective_family_mask_hash"],
            mask["objective_family_mask_hash"],
        )
        self.assertEqual(
            result.public_result.metrics["objective_family_mask"],
            mask,
        )


if __name__ == "__main__":
    unittest.main()
