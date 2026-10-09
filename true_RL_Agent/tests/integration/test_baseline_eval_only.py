from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tests.fakes.backend_environment import (
    make_test_isaac_lab_execution_environment_manifest,
)
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.composition import (
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec,
    BaselineManifest,
    CheckpointProvenance,
)
from true_rl_agent.runner.jobs import dry_run_eval_command
from true_rl_agent.runner.specs import BaselineEvalJobSpec, CheckpointRef
from true_rl_agent.tasks.push_recovery.composition import (
    PUSH_RECOVERY_ACTION_SCHEMA_HASH,
    PUSH_RECOVERY_BACKEND_ID,
    PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH,
    PUSH_RECOVERY_POLICY_ARCHITECTURE,
    PUSH_RECOVERY_ROBOT,
    PushRecoveryCompositionProvider,
)


class BaselineEvalOnlyIntegrationTests(unittest.TestCase):
    def _manifest(self, **overrides: object) -> BaselineManifest:
        values: dict[str, object] = {
            "baseline_id": "official-g1-rough",
            "task_id": "unitree_push_recovery",
            "checkpoint": CheckpointRef(
                uri="baseline://official-g1-rough/policy.pt",
                checksum="sha256:" + "1" * 64,
                role="warm_start",
                sealed=True,
                metadata={"local_path": "private/checkpoints/official-g1-rough.pt"},
            ),
            "provenance": CheckpointProvenance(
                source_run_id="isaaclab-official-g1-rough",
                source_task_id="Isaac-Velocity-Rough-G1-v0",
                created_by="isaaclab",
            ),
            "compatibility": BaselineCompatibilitySpec(
                backend_id=PUSH_RECOVERY_BACKEND_ID,
                robot=PUSH_RECOVERY_ROBOT,
                observation_schema_hash=PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH,
                action_schema_hash=PUSH_RECOVERY_ACTION_SCHEMA_HASH,
                policy_architecture=PUSH_RECOVERY_POLICY_ARCHITECTURE,
            ),
        }
        values.update(overrides)
        return BaselineManifest(**values)

    def _composition(self, output_dir: Path):
        provider = PushRecoveryCompositionProvider()
        composition = resolve_task_composition(
            provider=provider,
            values={
                **provider.cli_defaults(),
                "cell_variant_id": "push.g1_29dof.legacy_v1",
            },
            context=TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=output_dir,
            ),
        )
        adapter = composition.backend_adapter
        assert isinstance(adapter, IsaacLabRslRlBackendAdapter)
        return replace(
            composition,
            backend_adapter=IsaacLabRslRlBackendAdapter(
                adapter.config,
                execution_environment_resolver=(
                    make_test_isaac_lab_execution_environment_manifest
                ),
            ),
        )

    def test_backend_materializes_baseline_eval_without_train_or_candidate_surface(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composition = self._composition(Path(tmp))
            manifest = self._manifest()

            spec = composition.backend_adapter.materialize_baseline_eval(
                session_id="session-001",
                task_spec=composition.task_plugin.task_spec,
                baseline_manifest=manifest,
            )

        self.assertIsInstance(spec, BaselineEvalJobSpec)
        self.assertEqual(spec.evaluation_subject().kind, "baseline")
        self.assertEqual(spec.subject_manifest_hash, manifest.manifest_hash)
        argv = list(spec.metadata["cli_argv"])
        self.assertEqual(argv[argv.index("--subject-kind") + 1], "baseline")
        self.assertEqual(
            argv[argv.index("--subject-id") + 1],
            manifest.baseline_id,
        )
        self.assertEqual(
            argv[argv.index("--subject-manifest-hash") + 1],
            manifest.manifest_hash,
        )
        self.assertNotIn("--candidate-id", argv)
        self.assertNotIn("--candidate", argv)
        self.assertNotIn("candidate_path", json.dumps(spec.to_public_dict()))
        self.assertNotIn("local_path", json.dumps(spec.to_public_dict()))
        command = dry_run_eval_command(spec)
        self.assertIn("--subject-kind baseline", command)
        self.assertIn(
            f"--subject-manifest-hash {manifest.manifest_hash}",
            command,
        )

    def test_baseline_incompatibility_fails_before_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composition = self._composition(Path(tmp))
            backend = composition.backend_adapter
            task_spec = composition.task_plugin.task_spec
            with self.assertRaisesRegex(
                ValueError,
                "checkpoint checksum must use",
            ):
                replace(
                    self._manifest().checkpoint,
                    checksum="sha256:not-a-digest",
                )
            cases = {
                "task": self._manifest(task_id="other-task"),
                "observation": self._manifest(
                    compatibility=replace(
                        self._manifest().compatibility,
                        observation_schema_hash="other-observation",
                    )
                ),
                "checkpoint_seal": self._manifest(
                    checkpoint=replace(
                        self._manifest().checkpoint,
                        sealed=False,
                    )
                ),
            }
            for label, manifest in cases.items():
                with self.subTest(label=label):
                    with self.assertRaisesRegex(
                        ValueError,
                        "baseline compatibility failed",
                    ):
                        backend.materialize_baseline_eval(
                            session_id="session-001",
                            task_spec=task_spec,
                            baseline_manifest=manifest,
                        )

    def test_task_binding_or_manifest_identity_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composition = self._composition(Path(tmp))
            backend = composition.backend_adapter
            assert isinstance(backend, IsaacLabRslRlBackendAdapter)
            for field_name in ("task_protocol_hash", "seed_cohort_hash"):
                with self.subTest(field_name=field_name):
                    wrong_binding = replace(
                        backend.config.evaluation_binding,
                        **{field_name: "f" * 64},
                    )
                    mismatched_backend = IsaacLabRslRlBackendAdapter(
                        replace(
                            backend.config,
                            evaluation_binding=wrong_binding,
                        )
                    )
                    with self.assertRaisesRegex(
                        ValueError,
                        field_name,
                    ):
                        mismatched_backend.materialize_baseline_eval(
                            session_id="session-001",
                            task_spec=composition.task_plugin.task_spec,
                            baseline_manifest=self._manifest(),
                        )

            canonical = self._manifest()
            with self.assertRaisesRegex(
                ValueError,
                "baseline manifest hash mismatch",
            ):
                replace(canonical, manifest_hash="e" * 64)

            with self.assertRaisesRegex(
                ValueError,
                "experiment subject_id",
            ):
                backend.materialize_baseline_eval(
                    session_id="../escape",
                    task_spec=composition.task_plugin.task_spec,
                    baseline_manifest=canonical,
                )


if __name__ == "__main__":
    unittest.main()
