from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.schema import CandidateChangeSet
from true_rl_agent.composition.base import TaskCompositionContext
from true_rl_agent.curriculum.capabilities import (
    build_default_curriculum_capability_registry,
)
from true_rl_agent.curriculum.compiler import CurriculumCompiler
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.tasks.push_recovery.candidate_surface import (
    PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY,
)
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairPhase0Bundle


class CurriculumPhase9Tests(unittest.TestCase):
    def test_capability_registry_marks_unwired_capabilities_not_effective(self) -> None:
        registry = build_default_curriculum_capability_registry()

        all_ids = {capability["id"] for capability in registry.to_public_list(include_ineffective=True)}
        prompt_effective_ids = {capability["id"] for capability in registry.prompt_effective_capabilities()}

        self.assertIn("reward_weight_scale", all_ids)
        self.assertIn("external_push_curriculum", all_ids)
        self.assertFalse(registry.get("external_push_curriculum").effective_now)
        self.assertNotIn("external_push_curriculum", prompt_effective_ids)
        with self.assertRaises(KeyError):
            registry.validate_ids(["invented_curriculum_capability"])

    def test_curriculum_compiler_explains_modes_and_disables_llm_proposal(self) -> None:
        registry = build_default_curriculum_capability_registry()
        default_config = {
            "train_distribution": "runner_owned_target_distribution",
            "push_distribution": "task_default_target_push",
            "budget_steps": 4096,
        }
        proposal = {
            "operation_diff": {
                "curriculum_design": {
                    "stage_list": [
                        {
                            "name": "numeric warmup",
                            "progress_range": [0.0, 1.0],
                            "capabilities": ["reward_weight_scale"],
                        }
                    ]
                }
            }
        }

        for mode in ("frozen", "runner_owned", "task_default"):
            compiled = CurriculumCompiler().compile(
                plan=CurriculumPlan(mode=mode),
                task_default_training_config=default_config,
                registry=registry,
                llm_proposal=proposal,
            )

            self.assertEqual(compiled.effective_plan.mode, mode)
            self.assertEqual(compiled.effective_training_config, default_config)
            self.assertEqual(compiled.proposal_status, "disabled_by_phase")
            self.assertIn("disabled_by_phase", " ".join(compiled.explanations))
            self.assertEqual(compiled.lineage["mode"], mode)
            self.assertEqual(
                compiled.lineage["task_default_config_sha256"],
                compiled.lineage["effective_config_sha256"],
            )

    def test_curriculum_compiler_rejects_unregistered_capability_even_when_disabled(self) -> None:
        registry = build_default_curriculum_capability_registry()

        with self.assertRaises(KeyError):
            CurriculumCompiler().compile(
                plan=CurriculumPlan(mode="task_default"),
                task_default_training_config={"train_distribution": "runner_owned_target_distribution"},
                registry=registry,
                llm_proposal={"capabilities": ["made_up_capability"]},
            )

    def test_curriculum_compiler_rejects_malformed_capability_fields(self) -> None:
        registry = build_default_curriculum_capability_registry()

        with self.assertRaises(ValueError):
            CurriculumCompiler().compile(
                plan=CurriculumPlan(mode="task_default"),
                task_default_training_config={"train_distribution": "runner_owned_target_distribution"},
                registry=registry,
                llm_proposal={"capabilities": ["reward_weight_scale", 123]},
            )

    def test_task_and_backend_expose_fixed_protocol_and_future_capability_status(self) -> None:
        provider = PushRecoveryCompositionProvider()
        composition = provider.resolve(
            values=provider.cli_defaults(),
            context=TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=PROJECT_ROOT / "outputs" / "test-composition",
            ),
        )
        task = composition.task_plugin.task_spec
        backend = composition.backend_adapter

        self.assertEqual(task.default_curriculum.mode, "task_default")
        self.assertFalse(task.default_curriculum.llm_controlled)
        self.assertEqual(task.fixed_train_protocol["baseline_train_distribution"], "runner_owned_target_distribution")
        self.assertFalse(backend.backend_spec.curriculum_capability_effective_now("external_push_curriculum"))

    def test_stair_extension_schema_fixture_is_disabled_and_schema_only(self) -> None:
        bundle = StairPhase0Bundle.load(PROJECT_ROOT)
        protocol = bundle.protocol.to_dict()
        activation = protocol["activation"]

        self.assertFalse(activation["runtime_plugin_enabled"])
        self.assertFalse(activation["training_command_enabled"])
        self.assertFalse(activation["llm_stair_height_progression_enabled"])
        self.assertIn("training_distribution", protocol["terrain"])
        self.assertIn("evaluation_bins", protocol["terrain"])
        self.assertEqual(bundle.descriptor.to_dict()["task_id"], "unitree_stair_climbing")

    def test_schedule_and_height_range_only_surfaces_are_parameter_only(self) -> None:
        policy = PUSH_RECOVERY_CANDIDATE_SURFACE_POLICY

        for component in ("reward_weight_scale", "height_range", "single_stage_numeric_schedule", "schedule"):
            assessment = policy.classify(CandidateChangeSet(frozenset({component})))
            self.assertEqual(assessment.classification, "parameter_only")
            self.assertFalse(assessment.eligible_for_policy_accept)


if __name__ == "__main__":
    unittest.main()
