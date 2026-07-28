from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.fake_components import (
    FakeBackendAdapter,
    FakeRunner,
    FakeTaskPlugin,
)
from true_rl_agent.composition import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionError,
    TaskCompositionProvider,
    TaskCompositionRegistry,
    resolve_task_composition,
)
from true_rl_agent.composition.cli import (
    TASK_RUNTIME_DEFAULT_KEYS,
    add_task_runtime_arguments,
    add_task_selection_argument,
    select_task_composition_provider,
)
from true_rl_agent.composition.defaults import (
    build_application_composition_registry,
)
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec,
    BaselineManifest,
    CheckpointProvenance,
)
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.push_recovery.composition import (
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin


class _MemoryAdapter:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def build_public_summary(
        self,
        public_records: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        return {"task_id": self.task_id, "record_count": len(public_records)}


class _Provider(TaskCompositionProvider):
    def __init__(
        self,
        task_id: str,
        defaults: Mapping[str, Any],
        composition: object | None = None,
    ) -> None:
        self._task_id = task_id
        self._defaults = dict(defaults)
        self._composition = composition

    @property
    def task_id(self) -> str:
        return self._task_id

    def cli_defaults(self) -> Mapping[str, Any]:
        return dict(self._defaults)

    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        _ = values, context
        return self._composition  # type: ignore[return-value]


def _defaults(*, marker: str, budget_steps: int) -> dict[str, Any]:
    values: dict[str, Any] = {
        "isaac_python": f"/runtime/{marker}/python",
        "train_entrypoint": f"scripts/{marker}_train.py",
        "eval_entrypoint": f"scripts/{marker}_eval.py",
        "budget_steps": budget_steps,
        "public_seed_count": 2,
        "public_accept_seed_count": 3,
        "sealed_validation_seed_count": 5,
        "eval_protocol_version": f"{marker}_eval_v1",
        "runtime_config": None,
        "train_profile": f"{marker}_train",
        "eval_profile": f"{marker}_eval",
        "train_seed": 42,
        "train_num_envs": 16,
        "train_max_iterations": 4,
        "train_steps_per_iteration": 8,
        "eval_num_envs": 7,
        "runner_timeout_s": 30,
    }
    assert set(values) == set(TASK_RUNTIME_DEFAULT_KEYS)
    return values


class CompositionTests(unittest.TestCase):
    def _parse(
        self,
        argv: list[str],
        registry: TaskCompositionRegistry,
    ) -> argparse.Namespace:
        provider = select_task_composition_provider(argv, registry)
        parser = argparse.ArgumentParser()
        add_task_selection_argument(
            parser,
            registry=registry,
            provider=provider,
        )
        add_task_runtime_arguments(parser, provider.cli_defaults())
        return parser.parse_args(argv)

    def _resolved_composition(
        self,
        *,
        memory_task_id: str = "fake_push_recovery",
        policy_initialization: PolicyInitialization | None = None,
        baseline_manifest: BaselineManifest | None = None,
    ) -> ResolvedTaskComposition:
        initialization = policy_initialization or PolicyInitialization.random()
        return ResolvedTaskComposition(
            task_id="fake_push_recovery",
            task_plugin=FakeTaskPlugin(),
            task_memory_adapter=_MemoryAdapter(memory_task_id),
            backend_adapter=FakeBackendAdapter(),
            runner=FakeRunner(),
            policy_initialization=initialization,
            genesis_anchor_kind=(
                "bootstrap"
                if initialization.kind == "random"
                else "policy_baseline"
            ),
            task_defaults={"budget_steps": 128},
            baseline_manifest=baseline_manifest,
        )

    def _baseline_manifest(
        self,
        *,
        baseline_id: str = "fake-baseline",
    ) -> BaselineManifest:
        return BaselineManifest(
            baseline_id=baseline_id,
            task_id="fake_push_recovery",
            checkpoint=CheckpointRef(
                "baseline://fake/policy.pt",
                "sha256:" + "1" * 64,
                "warm_start",
                sealed=True,
            ),
            provenance=CheckpointProvenance(
                source_run_id="fake-run",
                source_task_id="fake_push_recovery",
                created_by="test",
            ),
            compatibility=BaselineCompatibilitySpec(
                backend_id="fake_backend",
                robot="fake_robot",
                observation_schema_hash="fake_obs",
                action_schema_hash="fake_action",
                policy_architecture="fake_policy",
            ),
        )

    def test_two_stage_cli_uses_only_selected_task_defaults(self) -> None:
        registry = TaskCompositionRegistry(default_task_id="alpha")
        registry.register(_Provider("alpha", _defaults(marker="alpha", budget_steps=111)))
        registry.register(_Provider("beta", _defaults(marker="beta", budget_steps=222)))

        alpha = self._parse([], registry)
        beta = self._parse(["--task-id", "beta"], registry)
        overridden = self._parse(
            ["--task-id", "beta", "--budget-steps", "999"],
            registry,
        )

        self.assertEqual(alpha.budget_steps, 111)
        self.assertEqual(alpha.train_profile, "alpha_train")
        self.assertEqual(beta.budget_steps, 222)
        self.assertEqual(beta.train_profile, "beta_train")
        self.assertEqual(beta.isaac_python, "/runtime/beta/python")
        self.assertEqual(overridden.budget_steps, 999)
        self.assertEqual(overridden.eval_profile, "beta_eval")

    def test_registry_and_resolver_fail_closed_before_execution(self) -> None:
        registry = TaskCompositionRegistry(default_task_id="fake_push_recovery")
        valid = self._resolved_composition()
        provider = _Provider(
            "fake_push_recovery",
            _defaults(marker="fake", budget_steps=128),
            valid,
        )
        registry.register(provider)
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register(provider)
        with self.assertRaisesRegex(KeyError, "unknown task composition"):
            registry.get("missing")

        with tempfile.TemporaryDirectory() as tmp:
            context = TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=Path(tmp),
            )
            resolved = resolve_task_composition(
                provider=provider,
                values={},
                context=context,
            )
            self.assertIs(resolved, valid)
            with self.assertRaisesRegex(
                TaskCompositionError,
                "selected provider",
            ):
                resolve_task_composition(
                    provider=_Provider(
                        "other-task",
                        _defaults(marker="other", budget_steps=1),
                        valid,
                    ),
                    values={},
                    context=context,
                )
            with self.assertRaisesRegex(
                TaskCompositionError,
                "must return ResolvedTaskComposition",
            ):
                resolve_task_composition(
                    provider=_Provider(
                        "fake_push_recovery",
                        _defaults(marker="bad", budget_steps=1),
                        object(),
                    ),
                    values={},
                    context=context,
                )

    def test_resolved_snapshot_binds_memory_and_is_immutable(self) -> None:
        composition = self._resolved_composition()
        with self.assertRaises(TypeError):
            composition.task_defaults["budget_steps"] = 256
        with self.assertRaisesRegex(
            TaskCompositionError,
            "task memory adapter",
        ):
            self._resolved_composition(memory_task_id="other-task")

    def test_push_composition_rejects_noncanonical_value_types(self) -> None:
        provider = PushRecoveryCompositionProvider()
        cases = (
            ("runner_timeout_s", True, "runner_timeout_s"),
            ("public_seed_count", "3", "public_seed_count"),
            (
                "sealed_validation_seed_count",
                7.9,
                "sealed_validation_seed_count",
            ),
            ("isaac_python", None, "isaac_python"),
            ("train_entrypoint", 42, "train_entrypoint"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            context = TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=Path(tmp),
            )
            for field_name, invalid, error_field in cases:
                values = dict(provider.cli_defaults())
                values[field_name] = invalid
                with self.subTest(field_name=field_name, invalid=invalid):
                    with self.assertRaisesRegex(ValueError, error_field):
                        provider.resolve(values=values, context=context)

    def test_push_task_config_rejects_invalid_programmatic_values(self) -> None:
        for field_name, invalid in (
            ("budget_steps", True),
            ("eval_num_envs", 8.0),
            ("public_accept_seed_count", "3"),
            ("eval_base_seed", False),
            ("train_profile", None),
            ("runtime_config_path", Path("push.yaml")),
        ):
            with self.subTest(field_name=field_name, invalid=invalid):
                with self.assertRaisesRegex(ValueError, field_name):
                    PushRecoveryTaskConfig(**{field_name: invalid})
        for invalid_config in (False, {}, object()):
            with self.subTest(invalid_plugin_config=invalid_config):
                with self.assertRaisesRegex(TypeError, "plugin config"):
                    PushRecoveryTaskPlugin(invalid_config)

    def test_push_composition_rejects_inconsistent_training_budget(self) -> None:
        provider = PushRecoveryCompositionProvider()
        values = dict(provider.cli_defaults())
        values.update(
            budget_steps=65,
            train_num_envs=8,
            train_steps_per_iteration=8,
            train_max_iterations=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(
                ValueError,
                "planned env steps mismatch",
            ):
                provider.resolve(
                    values=values,
                    context=TaskCompositionContext(
                        project_root=PROJECT_ROOT,
                        output_dir=Path(tmp),
                    ),
                )

        with self.assertRaisesRegex(ValueError, "planned env steps mismatch"):
            PushRecoveryTaskConfig(
                budget_steps=65,
                train_num_envs=8,
                train_steps_per_iteration=8,
                train_max_iterations=1,
            )

    def test_push_snapshot_and_runtime_share_one_validated_config(self) -> None:
        provider = PushRecoveryCompositionProvider()
        values = dict(provider.cli_defaults())
        values.update(
            {
                "isaac_python": "/opt/isaac/python",
                "train_entrypoint": "scripts/custom_train.py",
                "eval_entrypoint": "scripts/custom_eval.py",
                "budget_steps": 192,
                "public_seed_count": 2,
                "public_accept_seed_count": 2,
                "sealed_validation_seed_count": 2,
                "train_profile": "small_push",
                "eval_profile": "no_push",
                "train_num_envs": 8,
                "train_max_iterations": 3,
                "train_steps_per_iteration": 8,
                "eval_num_envs": 4,
                "runner_timeout_s": 123,
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            composition = provider.resolve(
                values=values,
                context=TaskCompositionContext(
                    project_root=PROJECT_ROOT,
                    output_dir=Path(tmp),
                ),
            )

        task_config = composition.task_plugin.config
        backend_config = composition.backend_adapter.config
        self.assertEqual(dict(composition.task_defaults), values)
        self.assertEqual(
            composition.runner.allowed_executables,
            (values["isaac_python"],),
        )
        self.assertEqual(composition.runner.timeout_s, values["runner_timeout_s"])
        self.assertEqual(backend_config.isaac_lab_python, values["isaac_python"])
        self.assertEqual(backend_config.train_entrypoint, values["train_entrypoint"])
        self.assertEqual(backend_config.eval_entrypoint, values["eval_entrypoint"])
        for snapshot_key, task_field, backend_field in (
            ("budget_steps", "budget_steps", "default_budget_steps"),
            ("public_seed_count", "public_seed_count", "public_seed_count"),
            (
                "public_accept_seed_count",
                "public_accept_seed_count",
                "public_accept_seed_count",
            ),
            (
                "sealed_validation_seed_count",
                "sealed_validation_seed_count",
                "sealed_validation_seed_count",
            ),
            ("train_num_envs", "train_num_envs", "train_num_envs"),
            (
                "train_max_iterations",
                "train_max_iterations",
                "train_max_iterations",
            ),
            (
                "train_steps_per_iteration",
                "train_steps_per_iteration",
                "train_steps_per_iteration",
            ),
            ("eval_num_envs", "eval_num_envs", "eval_num_envs"),
        ):
            expected = values[snapshot_key]
            self.assertEqual(getattr(task_config, task_field), expected)
            self.assertEqual(getattr(backend_config, backend_field), expected)

    def test_genesis_anchor_is_bound_to_training_initialization(
        self,
    ) -> None:
        manifest = self._baseline_manifest()
        with self.assertRaisesRegex(
            TaskCompositionError,
            "bootstrap composition cannot carry",
        ):
            self._resolved_composition(baseline_manifest=manifest)

        warm_start = manifest.to_policy_initialization()
        self._resolved_composition(
            policy_initialization=warm_start,
            baseline_manifest=manifest,
        )
        different_provenance = replace(
            manifest,
            provenance=CheckpointProvenance(
                source_run_id="different-run",
                source_task_id="fake_push_recovery",
                created_by="test",
            ),
            manifest_hash=None,
        )
        self.assertEqual(
            different_provenance.baseline_id,
            manifest.baseline_id,
        )
        self.assertEqual(
            different_provenance.checkpoint,
            manifest.checkpoint,
        )
        self.assertNotEqual(
            different_provenance.manifest_hash,
            manifest.manifest_hash,
        )
        with self.assertRaisesRegex(
            TaskCompositionError,
            "does not match baseline manifest",
        ):
            self._resolved_composition(
                policy_initialization=warm_start,
                baseline_manifest=different_provenance,
            )
        with self.assertRaisesRegex(
            TaskCompositionError,
            "does not match baseline manifest",
        ):
            self._resolved_composition(
                policy_initialization=warm_start,
                baseline_manifest=self._baseline_manifest(
                    baseline_id="different-baseline"
                ),
            )

    def test_task_neutral_composition_modules_have_no_concrete_task_semantics(
        self,
    ) -> None:
        composition_root = SRC_ROOT / "true_rl_agent" / "composition"
        generic_files = (
            "base.py",
            "registry.py",
            "cli.py",
            "__init__.py",
        )
        forbidden = (
            "push_recovery",
            "stair_climbing",
            "unitree_push_recovery",
            "unitree_stair_climbing",
            "stair_height",
        )
        for filename in generic_files:
            source = (composition_root / filename).read_text(encoding="utf-8")
            with self.subTest(filename=filename):
                for marker in forbidden:
                    self.assertNotIn(marker, source)

    def test_phase1a_does_not_register_production_stair_task(self) -> None:
        registry = build_application_composition_registry()

        self.assertEqual(registry.task_ids(), ("unitree_push_recovery",))
        self.assertNotIn("unitree_stair_climbing", registry.task_ids())


if __name__ == "__main__":
    unittest.main()
