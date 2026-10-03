"""The real replay entrypoint owns only probe/rollout leases; no simulator."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import push_numeric_replay as replay
from tests.unit import test_push_checkpoint_admission as admission_fixture
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhaseResult
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager, GpuLeaseUnavailableError, PreGenesisGpuLeaseHolder
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.push_recovery.phased_adapter import PushPhasedEvaluationAdapter
from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed


class PushNumericReplayTests(unittest.TestCase):
    def setUp(self):
        self.fixture = admission_fixture.PushCheckpointAdmissionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def test_new_base_seed_changes_cohort_plan_and_real_direction_phase(self):
        from tests.unit.test_push_numeric_capture import PushNumericCaptureTests

        phases, manifests = [], []
        for name, seed in (("old", 1_000_000_000), ("new", 1_500_000_000)):
            args = self.fixture.args(name, "on")
            args.cell, args.eval_base_seed = "push.go2_12dof.recovery_v2", seed
            args.video_dir = self.root / name / "video"
            args.video_secondary_env_index, args.video_snapshot_interval_steps = 1, 1
            composition, phase, _ = replay.prepare(args)
            argv = phase.metadata["cli_argv"]
            self.assertEqual(argv[argv.index("--seed") + 1], str(seed))
            self.assertEqual(argv[argv.index("--video-secondary-env-index") + 1], "1")
            self.assertEqual(argv[argv.index("--video-snapshot-interval-steps") + 1], "1")
            phases.append(phase)
            manifests.append(composition.resolved_cell_manifest.to_dict())
        before, after = phases
        self.assertNotEqual(before.evaluation_binding.seed_cohort_hash, after.evaluation_binding.seed_cohort_hash)
        self.assertNotEqual(before.evaluation_plan_hash, after.evaluation_plan_hash)
        for field in ("subject", "subject_manifest_hash", "program_source_checksum", "training_binding_hash", "capability_profile_hash"):
            self.assertEqual(getattr(before, field), getattr(after, field))
        # Historical Push TaskSpec includes the cohort in its invariant hash.
        self.assertNotEqual(before.evaluation_binding.task_protocol_hash, after.evaluation_binding.task_protocol_hash)
        hash_fields = {"manifest_hash", "base_task_protocol_hash", "resolved_cell_task_protocol_hash"}
        self.assertEqual(*[{key: value for key, value in manifest.items() if key not in hash_fields} for manifest in manifests])
        self.assertEqual(before.evaluation_binding.acceptance_protocol_hash, after.evaluation_binding.acceptance_protocol_hash)
        fixture = PushNumericCaptureTests()
        fixture.setUp()
        self.addCleanup(fixture.tmp.cleanup)
        state = fixture.velocity.state
        angles = []
        for seed in (1_000_000_000, 1_500_000_000):
            state.begin_evaluation("public_dev", derive_runner_seed(base_seed=seed, split_name="public_dev", run_index=1))
            state.reset(None)
            angles.append(state.angle.clone())
        self.assertFalse(bool((angles[0] == angles[1]).all()))

    def test_entrypoint_releases_probe_and_rollout_lease_before_cpu_completion(self):
        checkpoint = self.root / "fixture.pt"
        checkpoint.write_bytes(b"synthetic checkpoint, never executed")
        self.fixture.trained["checkpoint"]["checksum"] = file_sha256(checkpoint)
        self.fixture.summary.write_text(json.dumps(self.fixture.trained), encoding="utf-8")
        output = self.root / "run"
        identity = GpuDeviceIdentity("cuda:1", "test-machine", "GPU-test")
        manager = GpuLeaseManager(lease_root=self.root / "leases", devices={"cuda:1": identity})
        holder = PreGenesisGpuLeaseHolder(session_id="other", genesis_identity_hash="a" * 64,
            session_contract_hash="b" * 64, lane_id="cuda:1", invocation_segment_id="other", logical_job_id="other")
        order = []

        def probe(*_args, **_kwargs):
            order.append("probe")
            with self.assertRaises(GpuLeaseUnavailableError):
                manager.acquire("cuda:1", holder=holder)
            return {"manifest_hash": "b" * 64}

        def rollout(_runner, phase, **_kwargs):
            order.append("rollout")
            with self.assertRaises(GpuLeaseUnavailableError):
                manager.acquire("cuda:1", holder=holder)
            self.assertEqual(phase.execution_device, "cuda:0")
            self.assertEqual(phase.gpu_process_binding.host_execution_device, "cuda:1")
            self.assertEqual(phase.gpu_process_binding.resource_binding_hash, identity.identity_hash)
            path = Path(phase.metadata["summary_path"])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")
            return EvaluationPhaseResult(
                job_id=phase.job_id, task_id=phase.task_id, phase=phase.phase,
                evaluation_plan_hash=phase.evaluation_plan_hash, subject=phase.subject,
                subject_manifest_hash=phase.subject_manifest_hash, program_source_checksum=phase.program_source_checksum,
                training_binding_hash=phase.training_binding_hash, capability_profile_hash=phase.capability_profile_hash,
                checkpoint_hash=phase.execution_checkpoint().checksum, eval_protocol_hash=phase.evaluation_protocol().protocol_hash,
                evaluation_binding_hash=phase.evaluation_binding.binding_hash, execution_device=phase.execution_device,
                resource_binding_hash=phase.attempt_scope.resource_binding_hash,
                task_evidence={}, execution_cost={}, environment_source_preflight={},
                backend_execution_environment_verification={}, checkpoint_safety_report={},
                gpu_process_binding=phase.gpu_process_binding.to_dict())

        def complete(_runner, _phase, result):
            order.append("cpu")
            with manager.acquire("cuda:1", holder=holder):
                self.assertEqual(list((output / "job_attempts").rglob("attempt_*.result.json")), [])
            return result

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(replay, "discover_nvidia_gpu_inventory", return_value=[SimpleNamespace(execution_device="cuda:1", identity=identity)]), \
                patch.object(replay, "GpuLeaseManager", return_value=manager), \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", side_effect=probe), \
                patch.object(SubprocessRunner, "_run_evaluation_phase", autospec=True, side_effect=rollout), \
                patch.object(SubprocessRunner, "_complete_evaluation_phase", autospec=True, side_effect=complete), \
                patch.object(PushPhasedEvaluationAdapter, "project_public_phase_summary", return_value={"fixture": True}):
            self.assertEqual(replay.main(["--cell", "push.go2_12dof.recovery_v2", "--source-manifest",
                str(self.root / "source/resolved_cell_manifest.json"), "--checkpoint-summary", str(self.fixture.summary),
                "--checkpoint", str(checkpoint), "--device", "cuda:1", "--output", str(output),
                "--mode", "run", "--capture", "off"]), 0)
        self.assertEqual(order, ["probe", "rollout", "cpu"])
        report = json.loads((output / "result.json").read_text())
        receipt = json.loads(Path(report["phase_attempt_receipts"][0]).read_text())
        result = json.loads((output / "public_dev_result.json").read_text())
        self.assertEqual(receipt["output_hash"], canonical_json_hash(result))
        self.assertAlmostEqual(report["eval_phase_gpu_lease_wall_duration_s"], receipt["resource_lease_wall_duration_ns"] / 1e9)

    def test_replay_cli_profile_defaults_and_normal_binding(self):
        args = self.fixture.args("profile", "on")
        self.assertEqual((args.behavior_feedback_profile, args.trajectory_report_max_bytes), ("legacy_process_v1", 12288))
        args.behavior_feedback_profile, args.trajectory_report_max_bytes = "trajectory_numeric_v1", 12287
        composition, phase, _ = replay.prepare(args)
        self.assertEqual(composition.runner.eval_phase_summary_finalizer.keywords["trajectory_report_max_bytes"], 12287)
        self.assertEqual(composition.task_defaults["behavior_feedback_profile"], "trajectory_numeric_v1")
        self.assertIn("--numeric-evidence-dir", phase.metadata["cli_argv"])
        self.assertIsNone(phase.gpu_process_binding)
        args.output = self.root / "off-profile"
        args.capture = "off"
        with self.assertRaisesRegex(ValueError, "requires --capture on"):
            replay.prepare(args)

    def test_replay_cli_rejects_bad_profile_and_budget_before_probe(self):
        import contextlib
        import io
        with contextlib.redirect_stderr(io.StringIO()):
            for flag, value in (("--behavior-feedback-profile", "unsupported"), ("--trajectory-report-max-bytes", "oops")):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    replay.build_parser().parse_args([flag, value])
        for budget in (0, -1, 12289):
            args = self.fixture.args("bad-" + str(budget), "on")
            args.behavior_feedback_profile, args.trajectory_report_max_bytes = "trajectory_numeric_v1", budget
            with patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment") as probe:
                with self.assertRaisesRegex(ValueError, "budget"):
                    replay.prepare(args)
                probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
