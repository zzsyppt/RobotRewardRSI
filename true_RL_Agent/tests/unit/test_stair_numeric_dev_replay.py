from __future__ import annotations

import copy
import argparse
import hashlib
import importlib.util
import json
import os
import io
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.phases import EvaluationPhaseResult
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.gpu_lease import (
    GpuDeviceIdentity, GpuLeaseManager, GpuLeaseUnavailableError, PreGenesisGpuLeaseHolder,
)
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.phased_adapter import StairPhasedEvaluationAdapter

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json"
spec = importlib.util.spec_from_file_location("stair_numeric_dev_replay", ROOT / "scripts/stair_numeric_dev_replay.py")
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class StairNumericDevReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        provider = StairClimbingCompositionProvider()
        self.composition = provider.resolve(
            values={**provider.cli_defaults(), "cell_variant_id": "stair.go2_12dof.unified_v3"},
            context=TaskCompositionContext(project_root=ROOT, output_dir=self.root / "composition"),
        )
        self.checkpoint = self.root / "fixture.pt"
        self.checkpoint.write_bytes(b"synthetic-policy-preflight-fixture-not-a-real-trained-policy")
        self.checksum = "sha256:" + hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.summary_path = self.root / "source_train_summary.json"
        self.summary = self._summary()
        self._save_summary()

    def _summary(self, *, source_index=-1, kind="reference"):
        config = self.composition.task_plugin.config
        binding = config.formal_training_binding()
        source = self.composition.runner.eval_checkpoint_admission_hook.manifests[source_index]
        # Explicit synthetic completion facts exercise validation only.
        summary = {
            "status": "pass", "formal_training_executed": True, "job_id": "fixture-source-train",
            "task_id": self.composition.task_id, "subject_kind": kind, "subject_id": "fixture-trained-policy",
            "subject_manifest_hash": "e" * 64, "program_source_checksum": "sha256:" + "d" * 64,
            "resolved_cell_manifest_hash": source.manifest_hash, "resolved_cell_identity": source.summary_identity(),
            "training_binding_hash": binding.binding_hash,
            "training": {
                "budget_steps": config.budget_steps, "num_envs": config.train_num_envs,
                "num_steps_per_env": config.train_steps_per_iteration, "max_iterations": config.train_max_iterations,
                "latest_checkpoint_iteration": config.train_max_iterations - 1,
                "initial_policy_state_hash": "c" * 64, "execution_contract": config.training_execution_contract(),
                "budget_accounting": build_budget_accounting(
                    budget_steps=config.budget_steps, num_envs=config.train_num_envs,
                    num_steps_per_env=config.train_steps_per_iteration, max_iterations=config.train_max_iterations,
                ),
            },
            "environment_source_preflight": {"status": "pass", "source_manifest_hash": "a" * 64,
                                             "task_protocol_hash": source.resolved_cell_task_protocol_hash},
            "backend_execution_environment_verification": {"status": "pass", "backend_id": "isaac_lab_rsl_rl",
                                                           "actual_manifest_hash": "b" * 64, "expected_manifest_hash": "b" * 64},
            "checkpoint": {"uri": "artifact://fixture/best.pt", "role": "train_output", "checksum": self.checksum,
                           "metadata": {"source_latest_checkpoint_sha256": self.checksum}},
        }
        for field in ("capability_profile_hash", "policy_architecture_hash", "agent_configuration_hash",
                      "optimizer_training_profile_hash", "initialization_seed_schedule_hash"):
            summary[field] = getattr(binding, field)
        return summary

    def _save_summary(self):
        self.summary_path.write_text(json.dumps(self.summary), encoding="utf-8")

    def _args(self, output, *, mode="preflight", capture="off", device="cuda:0"):
        return ["--checkpoint", str(self.checkpoint), "--train-summary", str(self.summary_path),
                "--protocol", str(PROTOCOL), "--output", str(output), "--device", device,
                "--isaac-python", sys.executable, "--mode", mode, "--capture-numeric", capture]

    def _result(self, phase):
        return EvaluationPhaseResult(
            job_id=phase.job_id, task_id=phase.task_id, phase=phase.phase,
            evaluation_plan_hash=phase.evaluation_plan_hash, subject=phase.subject,
            subject_manifest_hash=phase.subject_manifest_hash, program_source_checksum=phase.program_source_checksum,
            training_binding_hash=phase.training_binding_hash, capability_profile_hash=phase.capability_profile_hash,
            checkpoint_hash=phase.execution_checkpoint().checksum,
            eval_protocol_hash=phase.evaluation_protocol().protocol_hash,
            evaluation_binding_hash=phase.evaluation_binding.binding_hash,
            execution_device=phase.execution_device,
            resource_binding_hash=phase.attempt_scope.resource_binding_hash if phase.attempt_scope else None,
            gpu_process_binding=phase.gpu_process_binding.to_dict() if phase.gpu_process_binding else None,
            task_evidence={}, execution_cost={}, environment_source_preflight={},
            backend_execution_environment_verification={}, checkpoint_safety_report={},
        )

    def test_preflight_on_off_preserves_v1_identity_without_environment_probe_or_launch(self):
        before_summary, before_checkpoint = self.summary_path.read_bytes(), self.checkpoint.read_bytes()
        with patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", side_effect=AssertionError("probe")), \
                patch.object(replay, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU discovery")), \
                patch.object(SubprocessRunner, "run_eval_phase", side_effect=AssertionError("launch")):
            for capture in ("off", "on"):
                output = self.root / capture
                self.assertEqual(replay.main(self._args(output, capture=capture)), 0)
                report = json.loads((output / "result.json").read_text())
                job = json.loads((output / "public_dev_job.json").read_text())
                self.assertEqual(report["status"], "PREFLIGHT_PASS")
                self.assertEqual(report["planned_public_dev_episodes"], 864)
                self.assertEqual(report["phases_executed"], [])
                self.assertEqual(report["source_training_cell_identity"]["cell_variant_id"], "stair.go2_12dof.unified_v1")
                self.assertEqual(job["spec"]["evaluation_plan"]["subject"]["id"], self.summary["subject_id"])
                self.assertEqual("--numeric-evidence-dir" in job["metadata"]["cli_argv"], capture == "on")
                self.assertNotIn("gpu_process_binding", job["spec"]["evaluation_plan"])
                self.assertFalse(any(arg.startswith("--gpu-") for arg in job["metadata"]["cli_argv"]))
        self.assertEqual(before_summary, self.summary_path.read_bytes())
        self.assertEqual(before_checkpoint, self.checkpoint.read_bytes())

    def test_video_presentation_flags_reach_eval_argv_only(self):
        output = self.root / "video-preflight"
        flags = ["--video-dir", str(output / "video"), "--video-env-index", "3",
                 "--video-camera-eye", "2.0", "-2.0", "1.5",
                 "--video-camera-lookat", "0.0", "0.0", "0.5",
                 "--video-resolution", "640", "480", "--video-snapshot-interval-steps", "1",
                 "--video-secondary-env-index", "5"]
        self.assertEqual(replay.main(self._args(output) + flags), 0)
        job = json.loads((output / "public_dev_job.json").read_text())
        argv = job["metadata"]["cli_argv"]
        self.assertEqual(argv[-len(flags):], flags)
        self.assertEqual(job["metadata"]["cli_command"], " ".join(argv))
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "PREFLIGHT_PASS")

    def test_feedback_parser_matches_normal_composition_choices_and_defaults(self):
        provider_parser = argparse.ArgumentParser()
        StairClimbingCompositionProvider().add_cli_arguments(provider_parser)
        args = replay._parser().parse_args(self._args(self.root / "parse"))
        defaults = provider_parser.parse_args([])
        for name in ("behavior_feedback_profile", "trajectory_report_max_bytes"):
            self.assertEqual(getattr(args, name), getattr(defaults, name))
        provider_profile = next(a for a in provider_parser._actions if a.dest == "behavior_feedback_profile")
        cli_profile = next(a for a in replay._parser()._actions if a.dest == "behavior_feedback_profile")
        self.assertEqual(cli_profile.choices, provider_profile.choices)
        for flags in (["--behavior-feedback-profile", "unknown"], ["--trajectory-report-max-bytes", "1.5"]):
            with self.subTest(flags=flags), patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
                replay._parser().parse_args(self._args(self.root / "bad-parse") + flags)
            self.assertEqual(error.exception.code, 2)

    def test_feedback_cli_reaches_real_composition_finalizer_and_keeps_legacy_defaults(self):
        compositions = []
        resolve = StairClimbingCompositionProvider.resolve

        def capture(provider, *, values, context):
            composition = resolve(provider, values=values, context=context)
            compositions.append(composition)
            return composition

        with patch.object(StairClimbingCompositionProvider, "resolve", autospec=True, side_effect=capture), \
                patch.object(replay, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU discovery")), \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", side_effect=AssertionError("probe")), \
                patch.object(SubprocessRunner, "run_eval_phase", side_effect=AssertionError("launch")):
            for name, flags in [("legacy", []), ("explicit-legacy", ["--behavior-feedback-profile", "legacy_process_v1"]),
                                ("trajectory", ["--behavior-feedback-profile", "trajectory_numeric_v1", "--trajectory-report-max-bytes", "8000"])]:
                self.assertEqual(replay.main(self._args(self.root / name, capture="on") + flags), 0)
        legacy, explicit, trajectory = compositions
        self.assertEqual(legacy.task_defaults, explicit.task_defaults)
        self.assertNotIn("behavior_feedback_profile", legacy.task_defaults)
        self.assertNotIn("trajectory_report_max_bytes", legacy.task_defaults)
        for composition, profile, budget in [(legacy, "legacy_process_v1", 8192), (trajectory, "trajectory_numeric_v1", 8000)]:
            keywords = composition.runner.eval_phase_summary_finalizer.keywords
            self.assertEqual(keywords["behavior_feedback_profile"], profile)
            self.assertEqual(keywords["trajectory_report_max_bytes"], budget)
        self.assertEqual(trajectory.task_defaults["behavior_feedback_profile"], "trajectory_numeric_v1")
        self.assertEqual(legacy.task_plugin.config.formal_training_binding(), trajectory.task_plugin.config.formal_training_binding())
        self.assertEqual(legacy.task_plugin.config.evaluation_protocol(), trajectory.task_plugin.config.evaluation_protocol())

    def test_feedback_invalid_budget_uses_existing_composition_validation_before_gpu(self):
        with patch.object(replay, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU discovery")) as discover, \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", side_effect=AssertionError("probe")) as probe, \
                patch.object(SubprocessRunner, "run_eval_phase", side_effect=AssertionError("launch")) as launch, \
                patch.object(replay.traceback, "print_exc"):
            for profile, budget in [("trajectory_numeric_v1", 0), ("trajectory_numeric_v1", -1),
                                    ("trajectory_numeric_v1", 8193), ("legacy_process_v1", 8000)]:
                output = self.root / f"invalid-{profile}-{budget}"
                with self.subTest(profile=profile, budget=budget):
                    self.assertEqual(replay.main(self._args(output, mode="run") +
                        ["--behavior-feedback-profile", profile, "--trajectory-report-max-bytes", str(budget)]), 1)
                    result = json.loads((output / "result.json").read_text())
                    self.assertEqual(result["status"], "FAIL")
                    self.assertIn("budget", result["error"])
                    self.assertFalse((output / "public_dev_job.json").exists())
            discover.assert_not_called()
            probe.assert_not_called()
            launch.assert_not_called()

    def test_actual_cli_help_and_trajectory_preflight_accept_profile_options(self):
        command = [sys.executable, "-B", str(ROOT / "scripts/stair_numeric_dev_replay.py")]
        help_result = subprocess.run([*command, "--help"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--behavior-feedback-profile", help_result.stdout)
        self.assertIn("--trajectory-report-max-bytes", help_result.stdout)
        output = self.root / "actual-cli"
        result = subprocess.run([*command, *self._args(output, capture="on"),
            "--behavior-feedback-profile", "trajectory_numeric_v1", "--trajectory-report-max-bytes", "8192"],
            capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((output / "result.json").read_text())
        self.assertEqual(report["status"], "PREFLIGHT_PASS")
        self.assertEqual(report["planned_public_dev_episodes"], 864)
        self.assertEqual(report["phases_executed"], [])

    def test_named_diagnostic_cohort_preflight(self):
        self.test_diagnostic_preflight_uses_new_cohort_without_changing_source_identity("n5_r2_20260930")

    def test_diagnostic_preflight_uses_new_cohort_without_changing_source_identity(self, cohort="v1"):
        from true_rl_agent.tasks.stair_climbing.seed_policy import diagnostic_seed_namespace

        output = self.root / "diagnostic-preflight"
        flags = [] if cohort == "v1" else ["--diagnostic-cohort", cohort]
        with patch.object(replay, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU discovery")), \
                patch.object(SubprocessRunner, "run_eval_phase", side_effect=AssertionError("launch")):
            self.assertEqual(replay.main(self._args(output, capture="on") + ["--cohort", "diagnostic"] + flags), 0)
        job = json.loads((output / "public_dev_job.json").read_text())
        plan = job["spec"]["evaluation_plan"]
        config = self.composition.task_plugin.config
        self.assertEqual(plan["evaluation_binding"]["seed_cohort_hash"],
                         config.seed_resolver().namespace_cohort_hash(diagnostic_seed_namespace(cohort)))
        self.assertEqual(plan["subject"]["id"], self.summary["subject_id"])
        self.assertEqual(plan["training_binding_hash"], self.summary["training_binding_hash"])
        self.assertIn("--stair-eval-purpose", job["metadata"]["cli_argv"])
        argv = job["metadata"]["cli_argv"]
        self.assertEqual("--diagnostic-cohort" in argv, cohort != "v1")
        if cohort != "v1":
            self.assertEqual(argv[argv.index("--diagnostic-cohort") + 1], cohort)
        report = json.loads((output / "result.json").read_text())
        if cohort == "v1":
            self.assertNotIn("diagnostic_cohort", report)
            self.assertNotIn("seed_namespace", report)
        else:
            self.assertEqual(report["diagnostic_cohort"], cohort)
            self.assertEqual(report["seed_namespace"], diagnostic_seed_namespace(cohort))
        self.assertEqual(self.summary, json.loads(self.summary_path.read_text()))

    def test_named_diagnostic_cohort_is_rejected_in_formal_mode_before_output(self):
        output = self.root / "invalid-formal-cohort"
        with self.assertRaisesRegex(ValueError, "requires --cohort diagnostic"):
            replay.main(self._args(output) + ["--diagnostic-cohort", "n5_r2_20260930"])
        self.assertFalse(output.exists())

    def test_bad_completion_budget_binding_and_source_are_rejected(self):
        mutations = (
            lambda s: s.update(formal_training_executed=False),
            lambda s: s["training"].update(latest_checkpoint_iteration=100),
            lambda s: s["training"]["budget_accounting"].update(planned_env_steps=1),
            lambda s: s.update(training_binding_hash="a" * 64),
            lambda s: s.update(resolved_cell_manifest_hash="b" * 64),
            lambda s: s["resolved_cell_identity"].update(cell_variant_id="stair.go2_12dof.unified_v3"),
            lambda s: s["environment_source_preflight"].update(status="fail"),
        )
        for mutate in mutations:
            source = copy.deepcopy(self.summary)
            mutate(source)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                replay._completed_source(source, composition=self.composition, checkpoint_path=self.checkpoint)
        source = self._summary(source_index=1)
        source["environment_source_preflight"]["task_protocol_hash"] = self.composition.task_plugin.config.training_binding_task_protocol_hash
        with self.assertRaisesRegex(ValueError, "source training environment preflight"):
            replay._completed_source(source, composition=self.composition, checkpoint_path=self.checkpoint)
        self.checkpoint.write_bytes(b"different bytes")
        with self.assertRaisesRegex(ValueError, "checksum"):
            replay._completed_source(self.summary, composition=self.composition, checkpoint_path=self.checkpoint)

    def test_run_uses_production_phase_runner_and_keeps_candidate_authorship(self):
        self.summary = self._summary(source_index=1, kind="candidate")
        self._save_summary()
        calls = []

        def execute(runner, phase):
            calls.append(phase)
            self.assertEqual(lease_events, ["enter", "exit"])
            self.assertIsInstance(phase.attempt_scope, PreGenesisExecutionScope)
            self.assertIsInstance(runner.gpu_lease_manager, GpuLeaseManager)
            self._assert_process_binding(phase, identity)
            return self._result(phase)

        output = self.root / "run"
        identity = GpuDeviceIdentity(execution_device="cuda:1", machine_id="test-machine", physical_id="GPU-test")
        lease_events = []

        class Lease:
            receipt = {"fixture": True}

            def __enter__(self):
                lease_events.append("enter")
                return self

            def __exit__(self, *args):
                lease_events.append("exit")

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(replay, "discover_nvidia_gpu_inventory", return_value=[SimpleNamespace(execution_device="cuda:1", identity=identity)]), \
                patch.object(GpuLeaseManager, "acquire", return_value=Lease()) as acquire, \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", return_value={"manifest_hash": "b" * 64}), \
                patch.object(SubprocessRunner, "run_eval_phase", autospec=True, side_effect=execute), \
                patch.object(StairPhasedEvaluationAdapter, "project_public_phase_summary", return_value={"fixture": True}) as projection:
            self.assertEqual(replay.main(self._args(output, mode="run", capture="on", device="cuda:1")), 0)
        self.assertEqual(lease_events, ["enter", "exit"])
        self.assertIsInstance(acquire.call_args.kwargs["holder"], PreGenesisGpuLeaseHolder)
        self.assertEqual(acquire.call_args.args[0], "cuda:1")
        self.assertEqual(projection.call_count, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].subject.kind, "candidate")
        self.assertEqual(calls[0].subject.subject_id, self.summary["subject_id"])
        self.assertEqual(calls[0].program_source_checksum, self.summary["program_source_checksum"])
        self.assertEqual(calls[0].execution_checkpoint().metadata["resolved_cell_identity"]["cell_variant_id"], "stair.go2_12dof.unified_v2")
        result = json.loads((output / "public_dev_result.json").read_text())
        self.assertFalse(result["acceptance_authority"])
        self.assertEqual(result["phase"], "public_dev")
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "PASS")

    def _assert_process_binding(self, phase, identity):
        binding = phase.gpu_process_binding
        self.assertEqual(binding.host_execution_device, "cuda:1")
        self.assertEqual(binding.process_execution_device, "cuda:0")
        self.assertEqual(binding.resource_binding_hash, identity.identity_hash)
        self.assertEqual(phase.execution_device, "cuda:0")
        self.assertEqual(phase.attempt_scope.resource_binding_hash, identity.identity_hash)
        argv = phase.metadata["cli_argv"]
        expected = {"--device": "cuda:0", "--gpu-host-execution-device": "cuda:1",
                    "--gpu-process-execution-device": "cuda:0", "--gpu-resource-binding-hash": identity.identity_hash}
        for key, value in expected.items():
            self.assertEqual(argv.count(key), 1)
            self.assertEqual(argv[argv.index(key) + 1], value)

    def test_entrypoint_releases_probe_and_runner_lease_before_cpu_finalization(self):
        identity = GpuDeviceIdentity("cuda:1", "test-machine", "GPU-test")
        manager = GpuLeaseManager(lease_root=self.root / "leases", devices={"cuda:1": identity})
        holder = PreGenesisGpuLeaseHolder(session_id="other", genesis_identity_hash="a" * 64,
            session_contract_hash="b" * 64, lane_id="cuda:1", invocation_segment_id="check",
            logical_job_id="other")
        order = []
        output = self.root / "lease-boundary"

        def rollout(_runner, phase, **kwargs):
            order.append("rollout")
            self._assert_process_binding(phase, identity)
            context = kwargs["execution_context"]
            self.assertEqual(context.environment_overrides["CUDA_VISIBLE_DEVICES"], "GPU-test")
            self.assertIsNone(context.environment_overrides["NVIDIA_VISIBLE_DEVICES"])
            with self.assertRaises(GpuLeaseUnavailableError):
                manager.acquire("cuda:1", holder=holder)
            return self._result(phase)

        def finalize(_runner, phase, result):
            order.append("cpu")
            with manager.acquire("cuda:1", holder=holder):
                self.assertEqual(list((output / "job_attempts").rglob("attempt_*.result.json")), [])
            return result

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(replay, "discover_nvidia_gpu_inventory", return_value=[SimpleNamespace(execution_device="cuda:1", identity=identity)]), \
                patch.object(replay, "GpuLeaseManager", return_value=manager), \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", return_value={"manifest_hash": "b" * 64}), \
                patch.object(SubprocessRunner, "_run_evaluation_phase", autospec=True, side_effect=rollout), \
                patch.object(SubprocessRunner, "_complete_evaluation_phase", autospec=True, side_effect=finalize), \
                patch.object(replay, "project_stair_public_dev_diagnostic", return_value={"fixture": True}) as projection:
            self.assertEqual(replay.main(self._args(output, mode="run", capture="on", device="cuda:1") +
                ["--cohort", "diagnostic", "--diagnostic-cohort", "n5_r2_20260930"]), 0)
        self.assertEqual(projection.call_args.kwargs["diagnostic_cohort"], "n5_r2_20260930")
        self.assertEqual(order, ["rollout", "cpu"])
        report = json.loads((output / "result.json").read_text())
        receipt = json.loads(Path(report["phase_attempt_receipts"][0]).read_text())
        result = json.loads((output / "public_dev_result.json").read_text())
        self.assertEqual(receipt["output_hash"], canonical_json_hash(result))
        self.assertEqual(report["eval_phase_gpu_lease_wall_duration_s"], receipt["resource_lease_wall_duration_ns"] / 1e9)
        self.assertEqual(report["gpu_lease_wall_duration_s"], report["environment_probe_gpu_lease_wall_duration_s"]
                         + report["eval_phase_gpu_lease_wall_duration_s"])

    def test_run_releases_lease_when_environment_probe_fails(self):
        identity = GpuDeviceIdentity(execution_device="cuda:0", machine_id="test-machine", physical_id="GPU-test")
        released = []

        class Lease:
            receipt = {"fixture": True}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                released.append(True)

        output = self.root / "failed-run"
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(replay, "discover_nvidia_gpu_inventory", return_value=[SimpleNamespace(execution_device="cuda:0", identity=identity)]), \
                patch.object(GpuLeaseManager, "acquire", return_value=Lease()), \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", side_effect=RuntimeError("fixture probe failure")), \
                patch.object(SubprocessRunner, "run_eval_phase", side_effect=AssertionError("must not launch")), \
                patch.object(replay.traceback, "print_exc"):
            self.assertEqual(replay.main(self._args(output, mode="run")), 1)
        self.assertEqual(released, [True])
        self.assertEqual(json.loads((output / "result.json").read_text())["status"], "FAIL")

    def test_existing_output_cannot_be_overwritten(self):
        output = self.root / "exists"
        output.mkdir()
        with self.assertRaises(FileExistsError):
            replay.main(self._args(output))


if __name__ == "__main__":
    unittest.main()
