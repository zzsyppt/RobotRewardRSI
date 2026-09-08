from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.materialize_job import main

from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.composition import (
    ResolvedTaskComposition,
    TaskCompositionContext,
    TaskCompositionRegistry,
)
from true_rl_agent.policy_baselines import (
    BaselineCompatibilitySpec,
    BaselineManifest,
    CheckpointProvenance,
)
from true_rl_agent.runner.specs import CheckpointRef
from true_rl_agent.tasks.push_recovery.composition import (
    PUSH_RECOVERY_ACTION_SCHEMA_HASH,
    PUSH_RECOVERY_BACKEND_ID,
    PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH,
    PUSH_RECOVERY_POLICY_ARCHITECTURE,
    PUSH_RECOVERY_ROBOT,
    PUSH_RECOVERY_TASK_ID,
    PushRecoveryCompositionProvider,
)


class _FixturePushProvider(PushRecoveryCompositionProvider):
    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        base = super().resolve(values=values, context=context)
        if not isinstance(base.backend_adapter, IsaacLabRslRlBackendAdapter):
            raise TypeError("test provider requires the Isaac Lab backend")
        return base


class _BaselinePushProvider(_FixturePushProvider):
    def resolve(
        self,
        *,
        values: Mapping[str, Any],
        context: TaskCompositionContext,
    ) -> ResolvedTaskComposition:
        base = super().resolve(values=values, context=context)
        manifest = BaselineManifest(
            baseline_id="official-g1-rough",
            task_id=PUSH_RECOVERY_TASK_ID,
            checkpoint=CheckpointRef(
                uri="baseline://official-g1-rough/policy.pt",
                checksum="sha256:" + "1" * 64,
                role="warm_start",
                sealed=True,
                metadata={"local_path": "private/official-g1-rough.pt"},
            ),
            provenance=CheckpointProvenance(
                source_run_id="isaaclab-official-g1-rough",
                source_task_id="Isaac-Velocity-Rough-G1-v0",
                created_by="isaaclab",
            ),
            compatibility=BaselineCompatibilitySpec(
                backend_id=PUSH_RECOVERY_BACKEND_ID,
                robot=PUSH_RECOVERY_ROBOT,
                observation_schema_hash=PUSH_RECOVERY_OBSERVATION_SCHEMA_HASH,
                action_schema_hash=PUSH_RECOVERY_ACTION_SCHEMA_HASH,
                policy_architecture=PUSH_RECOVERY_POLICY_ARCHITECTURE,
            ),
        )
        return replace(
            base,
            policy_initialization=manifest.to_policy_initialization(),
            genesis_anchor_kind="policy_baseline",
            baseline_manifest=manifest,
        )


class MaterializeJobScriptTests(unittest.TestCase):
    def _write_candidate(
        self,
        root: Path,
        *,
        candidate_id: str = "candidate-001",
        source: str | None = None,
    ) -> Path:
        candidate = root / candidate_id
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            "schema_version: true_rl_agent.candidate.v1\n"
            f"candidate_id: {candidate_id}\n"
            "entrypoint: training_code.components:build_training_components\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            source
            or (
                "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
                "def reward(inputs):\n"
                "    return inputs.base_height\n\n"
                "def build_training_components():\n"
                "    return TrainingComponents(\n"
                "        name='materialized-candidate',\n"
                "        reward_terms=[RewardTerm(name='height', weight=1.0, fn=reward)],\n"
                "    )\n"
            ),
            encoding="utf-8",
        )
        (candidate / "rationale.md").write_text("Materialize candidate.\n", encoding="utf-8")
        (candidate / "operation_diff.json").write_text("{}\n", encoding="utf-8")
        (candidate / "diagnostics.json").write_text("{}\n", encoding="utf-8")
        return candidate

    def test_materialize_job_rejects_unsupported_candidate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / "candidate-invalid"
            (candidate / "training_code").mkdir(parents=True)
            (candidate / "candidate_manifest.yaml").write_text(
                "schema_version: true_rl_agent.candidate.v999\n"
                "candidate_id: candidate-invalid\n"
                "entrypoint: training_code.components:build_training_components\n",
                encoding="utf-8",
            )
            (candidate / "training_code" / "components.py").write_text(
                "def build_training_components():\n    return None\n",
                encoding="utf-8",
            )
            (candidate / "rationale.md").write_text("Invalid manifest.\n", encoding="utf-8")
            (candidate / "operation_diff.json").write_text("{}\n", encoding="utf-8")
            (candidate / "diagnostics.json").write_text("{}\n", encoding="utf-8")

            with redirect_stderr(io.StringIO()), self.assertRaises(
                SystemExit
            ) as captured:
                main(
                    [
                        "--round-id",
                        "invalid-manifest",
                        "--candidate-id",
                        "candidate-invalid",
                        "--candidate-path",
                        str(candidate),
                        "--output-dir",
                        tmp,
                    ]
                )
            self.assertEqual(captured.exception.code, 2)

    def test_materialize_job_requires_an_explicit_candidate_package(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as captured,
        ):
            main(["--candidate-id", "candidate-001", "--output-dir", tmp])

        self.assertEqual(captured.exception.code, 2)

    def test_materialize_job_rejects_static_verifier_failure_before_export(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            candidate = self._write_candidate(
                Path(tmp),
                candidate_id="candidate-forbidden",
                source=(
                    "import os\n\n"
                    "def build_training_components():\n"
                    "    return None\n"
                ),
            )

            with (
                redirect_stdout(output),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as captured,
            ):
                main(
                    [
                        "--candidate-id",
                        "candidate-forbidden",
                        "--candidate-path",
                        str(candidate),
                        "--output-dir",
                        tmp,
                    ]
                )

        self.assertEqual(captured.exception.code, 2)
        self.assertEqual(output.getvalue(), "")

    def test_materialize_job_rejects_inconsistent_training_budget(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            self.assertRaisesRegex(ValueError, "planned env steps mismatch"),
        ):
            main(
                [
                    "--round-id",
                    "budget-mismatch",
                    "--candidate-id",
                    "candidate-001",
                    "--candidate-path",
                    "artifact://candidate-001",
                    "--output-dir",
                    tmp,
                    "--budget-steps",
                    "65",
                    "--train-num-envs",
                    "8",
                    "--train-max-iterations",
                    "1",
                    "--train-steps-per-iteration",
                    "8",
                ]
            )

    def test_materialized_job_exports_runner_execution_argv(self) -> None:
        output = io.StringIO()
        provider = _FixturePushProvider()
        registry = TaskCompositionRegistry(default_task_id=provider.task_id)
        registry.register(provider)

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(output):
            candidate = self._write_candidate(Path(tmp))
            exit_code = main(
                [
                    "--round-id",
                    "round-001",
                    "--candidate-id",
                    "candidate-001",
                    "--candidate-path",
                    str(candidate),
                    "--output-dir",
                    tmp,
                ],
                registry=registry,
            )

        payload = json.loads(output.getvalue())
        runner_execution = payload["runner_execution"]

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["eval"]["sealed_validation_seed_count"], 7)
        self.assertIn("eval_protocol_hash", payload["eval"])
        self.assertTrue(runner_execution["train"]["cli_argv"][0].endswith("/bin/python"))
        self.assertEqual(runner_execution["train"]["cli_argv"][1], "scripts/isaaclab_rsl_rl_train.py")
        self.assertIn("--candidate", runner_execution["train"]["cli_argv"])
        self.assertTrue(runner_execution["train"]["summary_path"].endswith("round-001/train/summary.json"))
        self.assertTrue(runner_execution["train"]["checkpoint_output_path"].endswith("round-001/checkpoints/best.pt"))
        self.assertTrue(runner_execution["eval"]["cli_argv"][0].endswith("/bin/python"))
        self.assertEqual(runner_execution["eval"]["cli_argv"][1], "scripts/isaaclab_rsl_rl_eval.py")
        self.assertIn("artifact://round-001/checkpoints/best.pt", runner_execution["eval"]["cli_argv"])
        self.assertNotIn("--checkpoint-sha256", runner_execution["eval"]["cli_argv"])
        self.assertNotIn("checkpoint", payload["eval"])
        self.assertEqual(
            payload["eval"]["checkpoint_plan"],
            {
                "state": "planned_output",
                "uri": "artifact://round-001/checkpoints/best.pt",
            },
        )
        self.assertIn("--subject-kind", runner_execution["eval"]["cli_argv"])
        self.assertEqual(
            runner_execution["eval"]["cli_argv"][
                runner_execution["eval"]["cli_argv"].index("--subject-kind") + 1
            ],
            "candidate",
        )
        self.assertIn("--subject-id", runner_execution["eval"]["cli_argv"])
        self.assertEqual(
            runner_execution["eval"]["cli_argv"][
                runner_execution["eval"]["cli_argv"].index("--subject-id") + 1
            ],
            "candidate-001",
        )
        self.assertNotIn("--candidate-id", runner_execution["eval"]["cli_argv"])
        self.assertNotIn("--candidate", runner_execution["eval"]["cli_argv"])
        self.assertTrue(runner_execution["eval"]["summary_path"].endswith("round-001/eval/summary.json"))
        self.assertNotIn("trusted_seed_count", json.dumps(runner_execution))
        self.assertNotIn("trusted_seed_count", json.dumps(payload["eval"]))
        self.assertNotIn(
            "--backend-execution-environment-manifest-hash",
            runner_execution["train"]["cli_argv"],
        )
        self.assertNotIn(
            "--backend-execution-environment-manifest-hash",
            runner_execution["eval"]["cli_argv"],
        )

    def test_materialize_baseline_eval_emits_no_train_or_candidate_job(self) -> None:
        provider = _BaselinePushProvider()
        registry = TaskCompositionRegistry(default_task_id=provider.task_id)
        registry.register(provider)
        output = io.StringIO()

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(output):
            exit_code = main(
                [
                    "--job-kind",
                    "baseline-eval",
                    "--round-id",
                    "session-001",
                    "--output-dir",
                    tmp,
                ],
                registry=registry,
            )

        payload = json.loads(output.getvalue())
        eval_argv = payload["runner_execution"]["eval"]["cli_argv"]
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            payload["schema_version"],
            "true_rl_agent.scripts.materialized_baseline_eval.v1",
        )
        self.assertNotIn("train", payload)
        self.assertNotIn("candidate_id", json.dumps(payload))
        self.assertEqual(
            eval_argv[eval_argv.index("--subject-kind") + 1],
            "baseline",
        )
        self.assertEqual(
            eval_argv[eval_argv.index("--subject-id") + 1],
            "official-g1-rough",
        )
        self.assertIn("--subject-manifest-hash", eval_argv)
        self.assertNotIn("--candidate-id", eval_argv)
        self.assertNotIn("--candidate-path", eval_argv)


if __name__ == "__main__":
    unittest.main()
