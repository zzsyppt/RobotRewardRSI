from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.isaaclab_stair_rsl_rl_eval import (  # noqa: E402
    _build_parser as build_eval_parser,
)
from scripts.isaaclab_stair_rsl_rl_eval import main as stair_eval_main  # noqa: E402
from scripts.isaaclab_stair_rsl_rl_train import (  # noqa: E402
    _build_parser as build_train_parser,
)
from scripts.isaaclab_stair_rsl_rl_train import main as stair_train_main  # noqa: E402

from tests.fakes.evaluation import make_verified_candidate_artifact  # noqa: E402
from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.runner.checkpoint import file_sha256  # noqa: E402
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.entrypoint_contract import (  # noqa: E402
    validate_stair_eval_invocation,
    validate_stair_train_invocation,
)
from true_rl_agent.training_programs import ReferenceProgramManifest  # noqa: E402


class StairEntrypointMaterializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = StairClimbingTaskConfig.defaults(
            ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
        )

    def test_materialized_jobs_use_stair_entrypoints_and_pass_pure_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            composition, jobs = self._materialize(Path(tmp), "stair-materialized")
            train_argv = jobs.train_spec.metadata["cli_argv"]
            eval_argv = jobs.eval_spec.metadata["cli_argv"]

            self.assertEqual(train_argv[1], "scripts/isaaclab_stair_rsl_rl_train.py")
            self.assertEqual(eval_argv[1], "scripts/isaaclab_stair_rsl_rl_eval.py")
            self.assertEqual(eval_argv.count("--program-source-checksum"), 1)
            self.assertNotIn("strong_push", " ".join(train_argv + eval_argv))
            train_args = build_train_parser().parse_args(train_argv[2:])
            train_context = validate_stair_train_invocation(
                train_args,
                project_root=ROOT,
            )
            self.assertEqual(
                train_context.training_binding.binding_hash,
                jobs.train_spec.training_binding.binding_hash,
            )
            self.assertEqual(
                train_context.candidate_artifact.artifact_hash,
                jobs.train_spec.candidate_artifact.artifact_hash,
            )
            self.assertEqual(
                composition.task_plugin.task_spec.task_id,
                STAIR_TASK_ID,
            )

            checkpoint = Path(tmp) / "runner-checkpoint.pt"
            checkpoint.write_bytes(b"phase-b-checkpoint-fixture")
            checkpoint_hash = file_sha256(checkpoint)
            self.assertIsNotNone(checkpoint_hash)
            bound_eval_argv = list(eval_argv[2:])
            checkpoint_index = bound_eval_argv.index("--checkpoint")
            bound_eval_argv[checkpoint_index + 1] = str(checkpoint)
            bound_eval_argv[checkpoint_index + 2 : checkpoint_index + 2] = [
                "--checkpoint-sha256",
                checkpoint_hash,
            ]
            eval_args = build_eval_parser(_FakeAppLauncher).parse_args(
                bound_eval_argv
            )
            eval_context = validate_stair_eval_invocation(
                eval_args,
                project_root=ROOT,
            )
            self.assertEqual(
                eval_context.evaluation_binding,
                jobs.eval_spec.evaluation_binding,
            )
            self.assertEqual(eval_context.checkpoint_hash, checkpoint_hash)

    def test_phase_b_train_entrypoint_fails_closed_after_complete_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-fail-closed")
            argv = jobs.train_spec.metadata["cli_argv"][2:]

            self.assertEqual(stair_train_main(argv), 1)
            summary_path = Path(jobs.train_spec.metadata["summary_path"])
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "fail")
            self.assertFalse(summary["formal_training_executed"])
            self.assertEqual(
                summary["runtime_activation_summary"]["reason"],
                "complete_reward_runtime_not_activated",
            )
            self.assertEqual(
                summary["training"]["budget_accounting"]["status"],
                "pass",
            )
            self.assertFalse(
                Path(jobs.train_spec.metadata["checkpoint_output_path"]).exists()
            )

    def test_eval_summary_is_written_before_simulation_app_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "eval-summary.json"
            argv = [
                "--job-id",
                "stair-eval-close-order",
                "--round",
                "phase-b",
                "--task-id",
                STAIR_TASK_ID,
                "--subject-kind",
                "candidate",
                "--subject-id",
                "candidate-close-order",
                "--subject-manifest-hash",
                "1" * 64,
                "--program-source-checksum",
                "sha256:" + "2" * 64,
                "--training-binding-hash",
                "3" * 64,
                "--checkpoint",
                str(Path(tmp) / "checkpoint.pt"),
                "--checkpoint-sha256",
                "sha256:" + "4" * 64,
                "--eval-protocol-version",
                "test",
                "--task-protocol-hash",
                "5" * 64,
                "--acceptance-protocol-hash",
                "6" * 64,
                "--seed-cohort-hash",
                "7" * 64,
                "--evaluation-binding-hash",
                "8" * 64,
                "--capability-profile-hash",
                "9" * 64,
                "--summary-path",
                str(summary_path),
                "--profile",
                "test",
                "--runtime-dynamics-hash",
                "sha256:" + "a" * 64,
                "--config",
                "test",
                "--public-seeds",
                "1",
                "--public-accept-seeds",
                "1",
                "--sealed-validation-seeds",
                "1",
                "--num-envs",
                "1",
                "--seed",
                "1",
                "--headless",
            ]
            context = SimpleNamespace(
                checkpoint_path=Path(tmp) / "checkpoint.pt",
                checkpoint_hash="sha256:" + "4" * 64,
            )
            checkpoint_report = SimpleNamespace(
                to_public_dict=lambda: {
                    "status": "pass",
                    "reason": "checkpoint_safe_to_load",
                }
            )
            with (
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval.validate_stair_eval_invocation",
                    return_value=context,
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval.build_checkpoint_safety_report",
                    return_value=checkpoint_report,
                ),
                patch(
                    "scripts.isaaclab_stair_rsl_rl_eval._run_eval",
                    return_value={"status": "pass", "public_score": 0.0},
                ),
            ):
                with self.assertRaisesRegex(SystemExit, "simulation closed"):
                    stair_eval_main(
                        argv,
                        app_launcher_class=_TerminatingFakeAppLauncher,
                    )

            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8"))["status"],
                "pass",
            )

    def test_preflight_rejects_binding_and_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, jobs = self._materialize(Path(tmp), "stair-drift")
            train_argv = list(jobs.train_spec.metadata["cli_argv"][2:])
            index = train_argv.index("--training-binding-hash")
            train_argv[index + 1] = "0" * 64
            train_args = build_train_parser().parse_args(train_argv)
            with self.assertRaisesRegex(ValueError, "training binding hash mismatch"):
                validate_stair_train_invocation(train_args, project_root=ROOT)

            train_argv = list(jobs.train_spec.metadata["cli_argv"][2:])
            index = train_argv.index("--config")
            train_argv[index + 1] = "configs/tasks/stair_protocol_v1.json"
            train_args = build_train_parser().parse_args(train_argv)
            with self.assertRaisesRegex(ValueError, "config path is not canonical"):
                validate_stair_train_invocation(train_args, project_root=ROOT)

    def _materialize(self, output_dir: Path, candidate_id: str):
        provider = StairClimbingCompositionProvider(
            reference_program_manifest=_reference_manifest(self.config)
        )
        composition = resolve_task_composition(
            provider=provider,
            values=provider.cli_defaults(),
            context=TaskCompositionContext(
                project_root=ROOT,
                output_dir=output_dir,
            ),
        )
        binding = composition.backend_adapter.evaluation_binding_for(
            composition.task_plugin.task_spec
        )
        jobs = composition.backend_adapter.materialize_jobs(
            round_id=candidate_id,
            task_spec=composition.task_plugin.task_spec,
            candidate=make_verified_candidate_artifact(candidate_id),
            policy_initialization=composition.policy_initialization,
            evaluation_binding=binding,
        )
        return composition, jobs


class _FakeAppLauncher:
    @staticmethod
    def add_app_launcher_args(parser) -> None:
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cpu")


class _TerminatingFakeAppLauncher(_FakeAppLauncher):
    def __init__(self, args) -> None:
        self.app = self

    def close(self) -> None:
        raise SystemExit("simulation closed")


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


if __name__ == "__main__":
    unittest.main()
