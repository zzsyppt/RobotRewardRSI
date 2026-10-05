"""Task-owned source binding and CPU preparation; no GPU or historical files."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from scripts.push_numeric_replay import build_parser, prepare
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.checkpoint_admission import validate_push_reference_evaluation_source
from true_rl_agent.tasks.push_recovery.config import PushRecoveryTaskConfig
from true_rl_agent.tasks.push_recovery.plugin import PushRecoveryTaskPlugin
from true_rl_agent.tasks.push_recovery.robot_binding import resolve_push_cell_manifest


class PushCheckpointAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.project = Path(__file__).resolve().parents[2]
        provider = PushRecoveryCompositionProvider()
        self.composition = provider.resolve(values={**provider.cli_defaults(), "cell_variant_id": "push.go2_12dof.recovery_v2",
            "execution_device": "cuda:0", "reference_admission_policy": None,
            "reference_program_id": "push_standing_recovery_stationary_reference_v3"},
            context=TaskCompositionContext(project_root=self.project, output_dir=self.root / "source"))
        c = self.composition
        binding = c.backend_adapter._training_binding(task_spec=c.task_plugin.task_spec, policy_initialization=c.policy_initialization)
        cfg = c.task_plugin.config
        self.trained = {"status": "pass", "task_id": c.task_id, "subject_kind": "reference",
            "subject_id": "push_standing_recovery_stationary_reference_v3", "subject_manifest_hash": "e" * 64,
            # A saved historical program is not required to equal today's source.
            "program_source_checksum": "sha256:" + "f" * 64,
            "resolved_cell_manifest_hash": c.resolved_cell_manifest.manifest_hash,
            "training_binding_hash": binding.binding_hash, "checkpoint": {"checksum": "sha256:" + "a" * 64},
            **{key: getattr(binding, key) for key in ("policy_architecture_hash", "agent_configuration_hash",
                "optimizer_training_profile_hash", "initialization_seed_schedule_hash", "capability_profile_hash")},
            "training": {"latest_checkpoint_iteration": cfg.train_max_iterations - 1,
                "budget_accounting": {"status": "pass", "budget_steps": cfg.budget_steps,
                "config_derived_total_timesteps": cfg.budget_steps,
                "planned_env_steps": cfg.budget_steps, "num_envs": cfg.train_num_envs,
                "num_steps_per_env": cfg.train_steps_per_iteration, "max_iterations": cfg.train_max_iterations}}}
        self.summary = self.root / "source_train.json"
        self.summary.write_text(json.dumps(self.trained), encoding="utf-8")

    def args(self, name, capture):
        return build_parser().parse_args(["--cell", "push.go2_12dof.recovery_v3", "--source-manifest",
            str(self.root / "source/resolved_cell_manifest.json"), "--checkpoint-summary", str(self.summary),
            "--checkpoint", str(self.root / "not-read-in-preflight.pt"), "--device", "cuda:0", "--output", str(self.root / name),
            "--mode", "preflight", "--capture", capture])

    def test_driver_and_composition_share_source_admission_and_explicit_off(self):
        on, phase, report = prepare(self.args("on", "on"))
        source = self.composition.resolved_cell_manifest
        self.assertNotEqual(source.manifest_hash, on.resolved_cell_manifest.manifest_hash)
        self.assertEqual(phase.training_binding_hash, self.trained["training_binding_hash"])
        self.assertEqual(phase.program_source_checksum, self.trained["program_source_checksum"])
        self.assertEqual(phase.subject_manifest_hash, self.trained["subject_manifest_hash"])
        self.assertFalse(report["checkpoint_bytes_checked"])
        self.assertEqual(report["checkpoint_runner_admission"], "not_executed")
        self.assertIn("--numeric-evidence-dir", phase.metadata["cli_argv"])
        self.assertTrue(on.backend_adapter.config.capture_public_dev_numeric)
        self.assertFalse(on.backend_adapter.config.capture_training_numeric)
        off, phase_off, _ = prepare(self.args("off", "off"))
        self.assertNotIn("--numeric-evidence-dir", phase_off.metadata["cli_argv"])
        argv = list(phase.metadata["cli_argv"])
        argv[argv.index("--training-binding-hash") + 1] = "b" * 64
        with self.assertRaisesRegex(ValueError, "relabeled"):
            on.runner.eval_checkpoint_admission_hook(phase.execution_checkpoint(), on.resolved_cell_manifest, tuple(argv))
        self.summary.write_text(json.dumps({**self.trained, "status": "fail"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bytes changed"):
            on.runner.eval_checkpoint_admission_hook(phase.execution_checkpoint(), on.resolved_cell_manifest, tuple(phase.metadata["cli_argv"]))

    def test_video_presentation_flags_reach_eval_argv_only(self):
        args = self.args("video", "off")
        args.video_dir = self.root / "video"
        args.video_env_index = 19
        args.video_camera_eye = [2.0, -2.0, 1.5]
        args.video_camera_lookat = [0.0, 0.0, 0.5]
        args.video_resolution = [640, 480]
        _, phase, _ = prepare(args)
        argv = phase.metadata["cli_argv"]
        self.assertEqual(argv[-15:], ["--video-dir", str(args.video_dir), "--video-env-index", "19",
            "--video-camera-eye", "2.0", "-2.0", "1.5", "--video-camera-lookat", "0.0", "0.0", "0.5",
            "--video-resolution", "640", "480"])
        self.assertEqual(phase.metadata["cli_command"], " ".join(argv))

    def test_training_dynamics_robot_direction_and_binding_are_not_relaxed(self):
        target, _, _ = prepare(self.args("target", "on"))
        source, current = self.composition.resolved_cell_manifest, target.resolved_cell_manifest
        cfg = target.task_plugin.config
        bad_cfg = replace(cfg, push_protocol=replace(cfg.push_protocol, high_velocity_scale=9), resolved_cell_task_protocol_hash=None)
        bad = resolve_push_cell_manifest(cell_variant_id=bad_cfg.cell_variant_id, config=bad_cfg, plugin=PushRecoveryTaskPlugin(bad_cfg))
        with self.assertRaisesRegex(ValueError, "training dynamics"):
            validate_push_reference_evaluation_source(source, bad, self.trained)
        with self.assertRaisesRegex(ValueError, "v2 to v3"):
            validate_push_reference_evaluation_source(current, source, self.trained)
        wrong_cfg = PushRecoveryTaskConfig.for_cell("push.g1_37dof.recovery_v3")
        wrong = resolve_push_cell_manifest(cell_variant_id=wrong_cfg.cell_variant_id, config=wrong_cfg, plugin=PushRecoveryTaskPlugin(wrong_cfg))
        with self.assertRaisesRegex(ValueError, "v2 to v3"):
            validate_push_reference_evaluation_source(source, wrong, self.trained)
        with self.assertRaisesRegex(ValueError, "cannot be reconstructed"):
            validate_push_reference_evaluation_source(source, current, {**self.trained, "training_binding_hash": "b" * 64})


if __name__ == "__main__":
    unittest.main()
