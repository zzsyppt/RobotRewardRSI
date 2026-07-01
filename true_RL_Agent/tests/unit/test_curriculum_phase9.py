from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter, IsaacLabRslRlBackendConfig
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidateSurfacePolicy
from true_rl_agent.curriculum.capabilities import build_default_curriculum_capability_registry
from true_rl_agent.curriculum.compiler import CurriculumCompiler
from true_rl_agent.curriculum.schema import CurriculumPlan
from true_rl_agent.tasks.base import TaskInvariantSpec, TaskSpec


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
        task = TaskSpec(
            task_id="unitree_push_recovery",
            name="Unitree Push Recovery",
            supported_policy_initializations=frozenset({"random", "warm_start"}),
            invariants=TaskInvariantSpec({"fall_threshold": 0.30}),
            fixed_train_protocol={"baseline_train_distribution": "runner_owned_target_distribution"},
            fixed_eval_protocol={"eval_bins": ["public_dev", "trusted_eval"]},
        )
        backend = IsaacLabRslRlBackendAdapter(
            IsaacLabRslRlBackendConfig(
                isaac_lab_python="/opt/isaac/python.sh",
                train_entrypoint="train.py",
                eval_entrypoint="eval.py",
                default_budget_steps=1024,
                public_seed_count=2,
                trusted_seed_count=4,
            )
        )

        self.assertEqual(task.default_curriculum.mode, "task_default")
        self.assertFalse(task.default_curriculum.llm_controlled)
        self.assertEqual(task.fixed_train_protocol["baseline_train_distribution"], "runner_owned_target_distribution")
        self.assertFalse(backend.backend_spec.curriculum_capability_effective_now("external_push_curriculum"))

    def test_stair_extension_schema_fixture_is_disabled_and_schema_only(self) -> None:
        fixture_path = PROJECT_ROOT / "configs" / "tasks" / "stair_extension_schema.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))

        self.assertFalse(fixture["runtime_plugin_enabled"])
        self.assertFalse(fixture["llm_stair_height_progression_enabled"])
        self.assertIn("fixed_baseline_train_distribution", fixture)
        self.assertIn("fixed_eval_bins", fixture)
        self.assertIn("height_map_terrain_metadata", fixture)

    def test_schedule_and_height_range_only_surfaces_are_parameter_only(self) -> None:
        policy = CandidateSurfacePolicy()

        for component in ("reward_weight_scale", "height_range", "single_stage_numeric_schedule", "schedule"):
            assessment = policy.classify(CandidateChangeSet(frozenset({component})))
            self.assertEqual(assessment.classification, "parameter_only")
            self.assertFalse(assessment.eligible_for_policy_accept)


if __name__ == "__main__":
    unittest.main()
