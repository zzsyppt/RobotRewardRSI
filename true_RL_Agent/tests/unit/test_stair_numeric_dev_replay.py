from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.composition import TaskCompositionContext
from true_rl_agent.evaluation.phases import EvaluationPhaseResult
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager, PreGenesisGpuLeaseHolder
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

    def _args(self, output, *, mode="preflight", capture="off"):
        return ["--checkpoint", str(self.checkpoint), "--train-summary", str(self.summary_path),
                "--protocol", str(PROTOCOL), "--output", str(output), "--device", "cuda:0",
                "--isaac-python", sys.executable, "--mode", mode, "--capture-numeric", capture]

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
        self.assertEqual(before_summary, self.summary_path.read_bytes())
        self.assertEqual(before_checkpoint, self.checkpoint.read_bytes())

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
            return EvaluationPhaseResult(
                job_id=phase.job_id, task_id=phase.task_id, phase=phase.phase,
                evaluation_plan_hash=phase.evaluation_plan_hash, subject=phase.subject,
                subject_manifest_hash=phase.subject_manifest_hash, program_source_checksum=phase.program_source_checksum,
                training_binding_hash=phase.training_binding_hash, capability_profile_hash=phase.capability_profile_hash,
                checkpoint_hash=phase.execution_checkpoint().checksum,
                eval_protocol_hash=phase.evaluation_protocol().protocol_hash,
                evaluation_binding_hash=phase.evaluation_binding.binding_hash,
                execution_device=phase.execution_device, resource_binding_hash=None,
                task_evidence={}, execution_cost={}, environment_source_preflight={},
                backend_execution_environment_verification={}, checkpoint_safety_report={},
            )

        output = self.root / "run"
        identity = GpuDeviceIdentity(execution_device="cuda:0", machine_id="test-machine", physical_id="GPU-test")
        lease_events = []

        class Lease:
            receipt = {"fixture": True}

            def __enter__(self):
                lease_events.append("enter")
                return self

            def __exit__(self, *args):
                lease_events.append("exit")

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                patch.object(replay, "discover_nvidia_gpu_inventory", return_value=[SimpleNamespace(execution_device="cuda:0", identity=identity)]), \
                patch.object(GpuLeaseManager, "acquire", return_value=Lease()) as acquire, \
                patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment", return_value={"manifest_hash": "b" * 64}), \
                patch.object(SubprocessRunner, "run_eval_phase", autospec=True, side_effect=execute), \
                patch.object(StairPhasedEvaluationAdapter, "project_public_phase_summary", return_value={"fixture": True}) as projection:
            self.assertEqual(replay.main(self._args(output, mode="run", capture="on")), 0)
        self.assertEqual(lease_events, ["enter", "exit"])
        self.assertIsInstance(acquire.call_args.kwargs["holder"], PreGenesisGpuLeaseHolder)
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
