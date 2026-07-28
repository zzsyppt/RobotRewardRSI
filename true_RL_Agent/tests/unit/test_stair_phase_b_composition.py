from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.composition.defaults import (  # noqa: E402
    build_application_composition_registry,
)
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionConfig,
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.runtime_profile import (  # noqa: E402
    STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FORMAL_EVAL_NAMESPACE,
    assert_seed_namespaces_disjoint,
)
from true_rl_agent.tasks.stair_climbing.staging import (  # noqa: E402
    build_stair_staging_composition_registry,
)
from true_rl_agent.training_programs import ReferenceProgramManifest  # noqa: E402


class StairPhaseBCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol_path = ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
        self.task_config = StairClimbingTaskConfig.defaults(self.protocol_path)
        self.reference = _reference_manifest(self.task_config)

    def test_default_registry_remains_push_only(self) -> None:
        registry = build_application_composition_registry()

        self.assertNotIn(STAIR_TASK_ID, registry.task_ids())
        self.assertEqual(registry.task_ids(), ("unitree_push_recovery",))

    def test_staging_registry_requires_real_reference_capability(self) -> None:
        with self.assertRaisesRegex(TypeError, "ReferenceProgramManifest"):
            build_stair_staging_composition_registry(  # type: ignore[arg-type]
                reference_program_manifest=None
            )

        wrong = replace(
            self.reference,
            capability_profile_hash="2" * 64,
            manifest_hash=None,
        )
        provider = StairClimbingCompositionProvider(reference_program_manifest=wrong)
        with self.assertRaisesRegex(ValueError, "capability profile mismatch"):
            resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=_context(),
            )

    def test_composition_resolves_once_to_one_consistent_snapshot(self) -> None:
        registry = build_stair_staging_composition_registry(
            reference_program_manifest=self.reference
        )
        provider = registry.get(STAIR_TASK_ID)
        values = provider.cli_defaults()

        composition = resolve_task_composition(
            provider=provider,
            values=values,
            context=_context(),
        )
        snapshot = composition.runtime_snapshot()
        backend = composition.backend_adapter.config
        task = composition.task_plugin.config

        self.assertEqual(snapshot["task_defaults"], values)
        self.assertEqual(
            snapshot["task_protocol_hash"],
            task.protocol_v2.protocol_hash,
        )
        self.assertEqual(
            backend.evaluation_binding.task_protocol_hash,
            task.protocol_v2.protocol_hash,
        )
        self.assertEqual(
            backend.evaluation_binding.seed_cohort_hash,
            task.seed_resolver().cohort_hash,
        )
        self.assertEqual(
            backend.train_runtime_dynamics_hash,
            task.train_runtime_profile().runtime_dynamics_hash,
        )
        self.assertEqual(
            task.train_runtime_profile().to_public_dict()["complete_reward_runtime"],
            STAIR_COMPLETE_REWARD_RUNTIME_STATUS,
        )
        self.assertEqual(composition.genesis_anchor_kind, "reference_program")
        self.assertIs(composition.reference_program_manifest, self.reference)

    def test_strict_config_rejects_coercion_and_relational_drift(self) -> None:
        defaults = StairClimbingCompositionConfig.defaults(project_root=ROOT)
        values = defaults.to_task_defaults()
        cases = {
            "runner_timeout_s": True,
            "public_seed_count": "3",
            "sealed_validation_seed_count": 7.0,
            "isaac_python": None,
            "train_entrypoint": 42,
        }
        for field_name, invalid in cases.items():
            with self.subTest(field_name=field_name):
                changed = dict(values)
                changed[field_name] = invalid
                with self.assertRaises((TypeError, ValueError)):
                    StairClimbingCompositionConfig.from_values(changed)

        with self.assertRaisesRegex(ValueError, "planned env steps mismatch"):
            StairClimbingCompositionConfig.from_values(
                {**values, "budget_steps": values["budget_steps"] + 1}
            )
        with self.assertRaisesRegex(ValueError, "does not match base protocol"):
            replace(self.task_config, eval_num_envs=49)
        with self.assertRaisesRegex(ValueError, "not an allowed V2 pilot budget"):
            replace(
                self.task_config,
                budget_steps=self.task_config.budget_steps * 2 + 1,
                train_max_iterations=self.task_config.train_max_iterations * 2 + 1,
            )

    def test_official_g1_agent_configuration_is_fully_bound(self) -> None:
        agent = self.task_config.training_agent_configuration()
        self.assertEqual(agent.policy["actor_hidden_dims"], (512, 256, 128))
        self.assertEqual(agent.policy["critic_hidden_dims"], (512, 256, 128))
        self.assertEqual(agent.algorithm["entropy_coef"], 0.008)
        spec = resolve_task_composition(
            provider=StairClimbingCompositionProvider(
                reference_program_manifest=self.reference
            ),
            values=StairClimbingCompositionConfig.defaults(
                project_root=ROOT
            ).to_task_defaults(),
            context=_context(),
        ).task_plugin.task_spec
        self.assertEqual(
            spec.fixed_train_protocol["agent_configuration_hash"],
            agent.agent_configuration_hash,
        )
        self.assertEqual(
            spec.fixed_train_protocol["private_training_seed_resolver_hash"],
            self.task_config.private_training_seed_resolver_hash(),
        )

    def test_private_seed_namespaces_are_unique_and_disjoint(self) -> None:
        resolver = self.task_config.seed_resolver()
        formal = [
            resolver.derive_seed(
                split=split,
                run_index=index,
                namespace=STAIR_FORMAL_EVAL_NAMESPACE,
            )
            for split, count in (
                ("public_dev", resolver.evaluation_protocol.public_seed_count),
                (
                    "public_accept",
                    resolver.evaluation_protocol.public_accept_seed_count,
                ),
                (
                    "sealed_validation",
                    resolver.evaluation_protocol.sealed_validation_seed_count,
                ),
            )
            for index in range(1, count + 1)
        ]
        self.assertEqual(len(formal), len(set(formal)))
        assert_seed_namespaces_disjoint(resolver)

        with self.assertRaisesRegex(ValueError, "contains duplicates"):
            assert_seed_namespaces_disjoint(_CollidingResolver())


def _reference_manifest(config: StairClimbingTaskConfig) -> ReferenceProgramManifest:
    return ReferenceProgramManifest(
        reference_id="stair-r0-phase-b-fixture",
        task_id=STAIR_TASK_ID,
        source_uri="artifact://phase-b-test/reference/components.py",
        source_checksum="sha256:" + "1" * 64,
        source_size_bytes=1,
        authorship_origin="official_upstream",
        upstream_project="IsaacLab",
        upstream_revision="phase-b-test-only",
        upstream_source_path="source/isaaclab_tasks/g1/rough_env_cfg.py",
        capability_profile_hash=config.protocol_v2.capability_profile.profile_hash,
    )


def _context() -> TaskCompositionContext:
    return TaskCompositionContext(
        project_root=ROOT,
        output_dir=ROOT / "outputs" / "phase-b-test",
    )


class _CollidingResolver:
    def _split_seed_count(self, split: str) -> int:
        return 1

    def derive_seed(self, *, split: str, run_index: int, namespace: str) -> int:
        return 7 if namespace == STAIR_FORMAL_EVAL_NAMESPACE else 8


if __name__ == "__main__":
    unittest.main()
