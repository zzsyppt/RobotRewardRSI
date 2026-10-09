"""Real subprocess/lease wiring for CPU phase finalization, without a GPU."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest

from tests.unit import test_subprocess_runner as fixtures
from tests.unit.test_runner_p1_reliability import _holder, _scope
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import (
    EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION, EvaluationPhase,
)
from true_rl_agent.runner.base import RunnerExecutionError, RunnerFailureKind
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity, GpuLeaseManager
from true_rl_agent.runner.specs import EvalPhaseJobSpec
from true_rl_agent.runner.subprocess_runner import SubprocessRunner


class PhasePostprocessingTests(unittest.TestCase):
    def _fixture(self, root, *, status="postprocessing", numeric_capture=False):
        helper = fixtures.SubprocessRunnerTests()
        metadata = {
            "summary_path": str(root / "summary.json"),
            "backend_id": "isaac_lab_rsl_rl",
            "backend_execution_environment_manifest_hash": (
                fixtures.TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH
            ),
        }
        if numeric_capture:
            metadata["numeric_evidence_dir"] = str(root / "numeric_evidence")
            metadata["numeric_evidence_max_bytes"] = 1048576
        plan = fixtures._eval_job_spec(
            job_id="postprocessing-plan", task_id="unitree_push_recovery",
            candidate_id="postprocessing-candidate",
            checkpoint=helper._eval_checkpoint(root), public_seed_count=3,
            public_accept_seed_count=5, sealed_validation_seed_count=7,
            evaluation_binding=fixtures.make_evaluation_binding(
                public_seed_count=3, public_accept_seed_count=5,
                sealed_validation_seed_count=7,
            ), metadata=metadata,
        )
        identity = GpuDeviceIdentity("cuda:0", "test-machine", "GPU-TEST")
        manager = GpuLeaseManager(lease_root=root / "leases", devices={"cuda:0": identity})
        spec = EvalPhaseJobSpec(
            job_id="postprocessing-phase", evaluation_plan=plan,
            phase=EvaluationPhase.PUBLIC_DEV, metadata=metadata,
            attempt_scope=replace(_scope(root / "attempts"), resource_binding_hash=identity.identity_hash),
        )
        evidence = {"phase": "public_dev", "raw": "bound-rollout"}
        summary = {
            **helper._eval_summary_binding(spec),
            "schema_version": EVALUATION_PHASE_SUMMARY_SCHEMA_VERSION,
            "evaluation_phase": spec.phase.value,
            "evaluation_plan_hash": spec.evaluation_plan_hash,
            "status": status, "task_evidence": evidence,
            "task_evidence_hash": canonical_json_hash(evidence),
            "evaluation_execution_cost": {"splits": {"public_dev": {}}},
            "environment_source_preflight": {"status": "pass"},
            "backend_execution_environment_verification": dict(
                fixtures.TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
            ),
            "checkpoint_safety_report": helper._checkpoint_safety_report(),
            "candidate_code_loaded_in_eval": False, "candidate_package_loaded": False,
            "candidate_package_loaded_in_eval": False, "candidate_path_argument_present": False,
            "errors": [],
        }
        code = (
            "from pathlib import Path\n"
            f"Path({str(root / 'summary.json')!r}).write_text({json.dumps(summary)!r})\n"
            f"with Path({str(root / 'launches.txt')!r}).open('a') as f: f.write('launch\\n')\n"
        )
        argv = helper._checkpoint_cli_argv(plan.execution_checkpoint(), code)
        if numeric_capture:
            argv = [*argv, "--numeric-evidence-dir", metadata["numeric_evidence_dir"],
                    "--numeric-evidence-max-bytes", str(metadata["numeric_evidence_max_bytes"])]
        spec = fixtures._with_eval_argv(spec, argv)
        return spec, manager

    def test_child_exits_and_lease_releases_before_finalization_and_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, manager = self._fixture(root)
            cpu_sleep_s = 0.15
            finalizer_calls = []

            def finalize(actual_spec, summary, *, output_dir):
                self.assertEqual(actual_spec, spec)
                self.assertEqual(output_dir, root)
                self.assertEqual((root / "launches.txt").read_text(), "launch\n")
                self.assertEqual(list((root / "attempts").rglob("attempt_*.result.json")), [])
                # Acquiring the physical identity proves release of the real
                # process file lock, not just an accounting timestamp.
                with manager.acquire("cuda:0", holder=_holder(logical_job_id="next-job")):
                    time.sleep(cpu_sleep_s)
                finalizer_calls.append(1)
                changed = {**summary, "status": "pass"}
                changed["task_evidence"] = {**summary["task_evidence"], "derived": "complete"}
                changed["task_evidence_hash"] = canonical_json_hash(changed["task_evidence"])
                return changed

            runner = SubprocessRunner(
                (sys.executable,), timeout_s=10, gpu_lease_manager=manager,
                eval_phase_summary_finalizer=finalize,
            )
            started = time.perf_counter()
            result = runner.run_eval_phase(spec)
            elapsed = time.perf_counter() - started
            receipt = json.loads(next((root / "attempts").rglob("attempt_*.result.json")).read_text())
            self.assertEqual(result.task_evidence["derived"], "complete")
            self.assertEqual(receipt["output_hash"], canonical_json_hash(result.to_private_dict()))
            self.assertLess(receipt["resource_lease_wall_duration_ns"] / 1e9, elapsed - cpu_sleep_s)
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "pass")
            self.assertEqual(runner.run_eval_phase(spec).to_private_dict(), result.to_private_dict())
            self.assertEqual(finalizer_calls, [1])
            self.assertEqual((root / "launches.txt").read_text(), "launch\n")

    def test_cpu_failure_does_not_publish_or_repeat_gpu_rollout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, manager = self._fixture(root)

            def finalize(_spec, _summary, *, output_dir):
                self.assertEqual(output_dir, root)
                with manager.acquire("cuda:0", holder=_holder(logical_job_id="next-job")):
                    raise ValueError("numeric attachment incomplete")

            runner = SubprocessRunner(
                (sys.executable,), timeout_s=10, gpu_lease_manager=manager,
                eval_phase_summary_finalizer=finalize,
            )
            with self.assertRaisesRegex(RunnerExecutionError, "finalization failed") as raised:
                runner.run_eval_phase(spec)
            self.assertEqual(raised.exception.failure_kind, RunnerFailureKind.SUMMARY_INVALID)
            self.assertEqual((root / "launches.txt").read_text(), "launch\n")
            receipt = json.loads(next((root / "attempts").rglob("attempt_*.result.json")).read_text())
            self.assertIsNone(receipt["output_hash"])
            self.assertFalse(receipt["retryable"])
            self.assertEqual(json.loads((root / "summary.json").read_text())["status"], "postprocessing")

    def test_handoff_without_task_finalizer_is_not_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, manager = self._fixture(root)
            runner = SubprocessRunner((sys.executable,), timeout_s=10, gpu_lease_manager=manager)
            with self.assertRaises(RunnerExecutionError):
                runner.run_eval_phase(spec)
            receipt = json.loads(next((root / "attempts").rglob("attempt_*.result.json")).read_text())
            self.assertIsNone(receipt["output_hash"])

    def test_fresh_capture_cannot_skip_cpu_handoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            spec, manager = self._fixture(root, status="pass", numeric_capture=True)
            calls = []
            runner = SubprocessRunner((sys.executable,), timeout_s=10,
                gpu_lease_manager=manager,
                eval_phase_summary_finalizer=lambda *args, **kwargs: calls.append(args))
            with self.assertRaisesRegex(RunnerExecutionError, "CPU postprocessing handoff") as raised:
                runner.run_eval_phase(spec)
            self.assertEqual(raised.exception.failure_kind, RunnerFailureKind.SUMMARY_INVALID)
            self.assertEqual(calls, [])
            self.assertEqual((root / "launches.txt").read_text(), "launch\n")
            receipt = json.loads(next((root / "attempts").rglob("attempt_*.result.json")).read_text())
            self.assertIsNone(receipt["output_hash"])
            self.assertFalse(receipt["retryable"])


if __name__ == "__main__":
    unittest.main()
