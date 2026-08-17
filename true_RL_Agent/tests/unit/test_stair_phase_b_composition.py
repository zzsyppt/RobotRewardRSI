from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.fakes.reference_programs import make_verified_stair_reference  # noqa: E402
from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.composition.defaults import (  # noqa: E402
    build_application_composition_registry,
)
from true_rl_agent.composition.lazy import (  # noqa: E402
    LazyTaskCompositionProvider,
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
from true_rl_agent.tasks.stair_climbing.session_start import (  # noqa: E402
    STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
    STAIR_REWARD_COLD_START_MODE,
)


class StairPhaseBCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.protocol_path = (
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"
        )
        self.task_config = StairClimbingTaskConfig.defaults(self.protocol_path)
        self.reference = make_verified_stair_reference(
            Path(self.temporary.name) / "reference",
            task_id=STAIR_TASK_ID,
            capability_profile_hash=(
                self.task_config.protocol_v2.capability_profile.profile_hash
            ),
        )

    def test_default_registry_registers_stair_lazily(self) -> None:
        registry = build_application_composition_registry()

        self.assertEqual(registry.default_task_id, "unitree_push_recovery")
        self.assertEqual(
            registry.task_ids(),
            ("unitree_push_recovery", STAIR_TASK_ID),
        )
        self.assertIsInstance(
            registry.get(STAIR_TASK_ID),
            LazyTaskCompositionProvider,
        )

    def test_stair_provider_validates_injected_reference_capability(self) -> None:
        with self.assertRaisesRegex(TypeError, "artifact is invalid"):
            StairClimbingCompositionProvider(  # type: ignore[arg-type]
                reference_program_artifact=self.reference.manifest
            )

        wrong = make_verified_stair_reference(
            Path(self.temporary.name) / "wrong-profile",
            task_id=STAIR_TASK_ID,
            capability_profile_hash="2" * 64,
        )
        provider = StairClimbingCompositionProvider(reference_program_artifact=wrong)
        with self.assertRaisesRegex(ValueError, "capability profile mismatch"):
            resolve_task_composition(
                provider=provider,
                values=provider.cli_defaults(),
                context=_context(),
            )

    def test_composition_resolves_once_to_one_consistent_snapshot(self) -> None:
        provider = StairClimbingCompositionProvider(
            reference_program_artifact=self.reference
        )
        values = provider.cli_defaults()

        composition = resolve_task_composition(
            provider=provider,
            values=values,
            context=_context(),
        )
        snapshot = composition.genesis_identity_snapshot()
        backend = composition.backend_adapter.config
        task = composition.task_plugin.config

        expected_snapshot_defaults = dict(values)
        expected_snapshot_defaults.pop("start_mode")
        self.assertEqual(snapshot["task_defaults"], expected_snapshot_defaults)
        self.assertEqual(
            snapshot["task_protocol_hash"],
            task.task_protocol_hash,
        )
        self.assertEqual(
            composition.task_plugin.task_spec.invariants.values[
                "protocol_status"
            ],
            task.protocol_v2.to_public_dict()["status"],
        )
        self.assertNotIn(
            "draft_status",
            composition.task_plugin.task_spec.invariants.values,
        )
        self.assertEqual(
            backend.evaluation_binding.task_protocol_hash,
            task.task_protocol_hash,
        )
        self.assertEqual(
            backend.evaluation_binding.seed_cohort_hash,
            task.seed_resolver().namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
        )
        self.assertEqual(
            composition.task_plugin.task_spec.fixed_eval_protocol[
                "seed_namespace"
            ],
            STAIR_FORMAL_EVAL_NAMESPACE,
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
        self.assertIs(composition.reference_program_artifact, self.reference)
        self.assertIs(
            composition.reference_program_manifest,
            self.reference.manifest,
        )
        self.assertEqual(
            snapshot["reference_program"]["verification_hash"],
            self.reference.verification_hash,
        )
        self.assertNotIn("candidate_admission_policy", snapshot)
        self.assertNotIn("authoring_capability_readiness", snapshot)

    def test_reward_cold_start_skips_reference_and_binds_initial_state(self) -> None:
        provider = StairClimbingCompositionProvider()
        values = dict(provider.cli_defaults())
        values["start_mode"] = STAIR_REWARD_COLD_START_MODE
        with patch(
            "true_rl_agent.tasks.stair_climbing.reference_program."
            "resolve_official_stair_reference_program",
            side_effect=AssertionError("cold start resolved the official reference"),
        ) as resolver:
            composition = resolve_task_composition(
                provider=provider,
                values=values,
                context=_context(),
            )

        resolver.assert_not_called()
        self.assertEqual(composition.genesis_anchor_kind, "bootstrap")
        self.assertIsNone(composition.reference_program_artifact)
        self.assertIsNone(composition.baseline_manifest)
        self.assertEqual(
            composition.bootstrap_initial_policy_state_hash,
            STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
        )
        snapshot = composition.genesis_identity_snapshot()
        self.assertEqual(
            snapshot["bootstrap_initial_policy_state_hash"],
            STAIR_FORMAL_INITIAL_POLICY_STATE_HASH,
        )
        self.assertEqual(
            snapshot["task_defaults"]["session_start_profile"]["profile_id"],
            STAIR_REWARD_COLD_START_MODE,
        )

        reference_provider = StairClimbingCompositionProvider(
            reference_program_artifact=self.reference
        )
        reference = resolve_task_composition(
            provider=reference_provider,
            values=reference_provider.cli_defaults(),
            context=_context(),
        )
        self.assertEqual(
            composition.task_plugin.task_spec.fixed_train_protocol,
            reference.task_plugin.task_spec.fixed_train_protocol,
        )
        self.assertEqual(
            composition.task_plugin.task_spec.fixed_eval_protocol,
            reference.task_plugin.task_spec.fixed_eval_protocol,
        )
        self.assertEqual(
            composition.task_plugin.task_spec.invariants,
            reference.task_plugin.task_spec.invariants,
        )
        self.assertEqual(
            composition.backend_adapter.config.evaluation_binding,
            reference.backend_adapter.config.evaluation_binding,
        )
        self.assertEqual(
            composition.task_plugin.task_spec.resolved_task_protocol_hash(),
            reference.task_plugin.task_spec.resolved_task_protocol_hash(),
        )

    def test_reward_cold_start_rejects_reference_and_score_override(self) -> None:
        provider = StairClimbingCompositionProvider(
            reference_program_artifact=self.reference
        )
        values = dict(provider.cli_defaults())
        values["start_mode"] = STAIR_REWARD_COLD_START_MODE
        with self.assertRaisesRegex(ValueError, "cannot carry an injected reference"):
            resolve_task_composition(
                provider=provider,
                values=values,
                context=_context(),
            )

        provider = StairClimbingCompositionProvider()
        values["initial_best_score"] = 0.1
        with self.assertRaisesRegex(ValueError, "profile initial_best_score"):
            resolve_task_composition(
                provider=provider,
                values=values,
                context=_context(),
            )

    def test_reference_role_does_not_rewrite_authorship_provenance(self) -> None:
        inherited = make_verified_stair_reference(
            Path(self.temporary.name) / "inherited-reference",
            task_id=STAIR_TASK_ID,
            capability_profile_hash=(
                self.task_config.protocol_v2.capability_profile.profile_hash
            ),
            authorship_origin="llm_generated",
        )
        provider = StairClimbingCompositionProvider(
            reference_program_artifact=inherited
        )
        composition = resolve_task_composition(
            provider=provider,
            values=provider.cli_defaults(),
            context=_context(),
        )

        self.assertEqual(composition.genesis_anchor_kind, "reference_program")
        self.assertEqual(
            composition.reference_program_manifest.authorship_origin,
            "llm_generated",
        )

    def test_strict_config_rejects_coercion_and_relational_drift(self) -> None:
        defaults = StairClimbingCompositionConfig.defaults(project_root=ROOT)
        values = defaults.to_task_defaults()
        cases = {
            "runner_timeout_s": True,
            "public_seed_count": "3",
            "sealed_validation_seed_count": 7.0,
            "isaac_python": None,
            "train_entrypoint": 42,
            "execution_device": "cuda",
            "start_mode": 1,
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
        with self.assertRaisesRegex(ValueError, "differs from fixed V2 pilot budget"):
            replace(
                self.task_config,
                budget_steps=self.task_config.budget_steps * 2 + 1,
                train_max_iterations=self.task_config.train_max_iterations * 2 + 1,
            )

    def test_official_g1_agent_configuration_is_fully_bound(self) -> None:
        production = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
        )
        agent = production.training_agent_configuration()
        self.assertEqual(agent.policy["actor_hidden_dims"], (512, 256, 128))
        self.assertEqual(agent.policy["critic_hidden_dims"], (512, 256, 128))
        self.assertEqual(agent.algorithm["entropy_coef"], 0.008)
        spec = resolve_task_composition(
            provider=StairClimbingCompositionProvider(
                reference_program_artifact=self.reference
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
            production.private_training_seed_resolver_hash(),
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
