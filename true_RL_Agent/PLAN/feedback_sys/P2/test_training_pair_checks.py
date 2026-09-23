"""Synthetic CPU tests of the P2 acceptance reader, never experiment evidence."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture

SPEC = importlib.util.spec_from_file_location("training_pair_reader", Path(__file__).with_name("verify_training_pair.py"))
READER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(READER)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


class TrainingPairReaderTests(unittest.TestCase):
    def pair(self, root, task):
        on, off = root / "on", root / "off"
        on.mkdir(); off.mkdir()
        iterations = 1 if task == "stair" else 3
        steps = iterations * 8
        offset = 1 if task == "stair" else 0
        identity = {"training_binding_hash": "a" * 64, "resolved_cell_manifest_hash": "b" * 64,
                    "resolved_cell_identity": {"cell_variant_id": task + ".go2.synthetic_fixture"},
                    "backend_execution_environment_verification": {"actual_manifest_hash": "c" * 64},
                    "initial_policy_state_hash": "d" * 64}
        program = "sha256:" + "e" * 64
        job = {"job_id": "synthetic-job", "task_id": "synthetic-task", "subject_kind": "reference", "subject_id": "synthetic-reference"}
        identity.update(task="stair_climbing" if task == "stair" else "push_recovery", job=job,
                        task_protocol_hash="f" * 64, training_seed=42)
        argv_protocol = "f" * 64
        if task == "stair":
            from true_rl_agent.composition.cell_manifest import materialize_resolved_cell_manifest
            from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
            from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
            from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
            config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
            manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id,
                config=config, plugin=StairClimbingTaskPlugin(config))
            manifest_path = materialize_resolved_cell_manifest(manifest, output_dir=root / "cell")
            argv_protocol = config.training_binding_task_protocol_hash
            job["task_id"] = manifest.task_id
            identity.update(task_protocol_hash=manifest.resolved_cell_task_protocol_hash,
                resolved_cell_manifest_hash=manifest.manifest_hash, resolved_cell_identity=manifest.summary_identity(),
                environment_source_preflight={"status": "pass", "source_manifest_hash": "c" * 64,
                    "task_protocol_hash": manifest.resolved_cell_task_protocol_hash})
            identity["program"] = {"program_source_checksum": program}
            identity.update(diagnostic_smoke=True, run_dir=str(on / "run"))
        else:
            identity["program_source_checksum"] = program
            identity.update(train_attempt_id="attempt-on", run_dir=str(on / "run/attempt-attempt-on"))
        capture = TrainingNumericCapture(on / "numeric", identity=identity,
            semantics={"terms": [{"index": 0, "name": "example", "weight": 2.0}]},
            planned_control_steps=steps, num_envs=16, steps_per_iteration=8, source_step_offset=offset)
        for step in range(1, steps + 1):
            raw = np.ones(16, dtype=np.float32) * step
            arrays = {"raw_term_000": raw, "weighted_term_000": raw * 2}
            if task == "stair":
                arrays["total_rate"] = raw * 2
            else:
                arrays.update(candidate_total=raw * 2, input_reward=raw, returned_reward=raw * 3)
            capture.observe(offset + step, arrays, metadata={"reward_weight_scale": 1.0})
        evidence = capture.finish(training_status="pass")
        for folder, flag in ((on, "on"), (off, "off")):
            checkpoint = folder / "smoke_checkpoint.pt"
            torch.save({"actor_state_dict": {"weight": torch.ones(2)}, "critic_state_dict": {"bias": torch.zeros(1)},
                "optimizer_state_dict": {"state": {0: {"step": torch.tensor(1.), "exp_avg": torch.ones(2)}}, "param_groups": [{"lr": 0.001}]},
                "iter": iterations - 1, "infos": {"different_run_folder": str(folder)}}, checkpoint)
            training = {"initial_policy_state_hash": "d" * 64, "num_envs": 16, "latest_checkpoint_iteration": iterations - 1,
                        "numeric_evidence": evidence if flag == "on" else {"status": "not_enabled"}}
            summary = {"status": "pass", "training": training, "runtime_activation_summary": {"step_calls": steps + offset},
                       "activation_preflight": {"real_transition_dispatcher_calls": offset}, "checkpoint": {"checksum": READER.sha(checkpoint)}}
            summary["backend_execution_environment_verification"] = identity["backend_execution_environment_verification"]
            if task == "stair":
                summary.update(original_training_job_identity=job, reward_program=identity["program"],
                               environment_source_preflight=identity["environment_source_preflight"])
            else:
                summary.update(**job, program_source_checksum=program, training_binding_hash="a" * 64,
                               run={"train_attempt_id": "attempt-" + flag})
            write(folder / "training_summary.json", summary)
            argv = ["python", "entry", "--summary-path", str(folder / "training_summary.json"),
                "--checkpoint-output", str(checkpoint), "--training-binding-hash", "a" * 64,
                "--resolved-cell-manifest-hash", identity["resolved_cell_manifest_hash"],
                "--backend-execution-environment-manifest-hash", "c" * 64]
            for key, value in job.items():
                argv.extend(("--" + key.replace("_", "-"), value))
            argv.extend(("--program-source-checksum", program, "--task-protocol-hash", argv_protocol, "--seed", "42"))
            if task == "push":
                argv.extend(("--run-dir", str(folder / "run")))
            else:
                argv.extend(("--resolved-cell-manifest-path", str(manifest_path),
                             "--base-task-protocol-hash", manifest.base_task_protocol_hash))
            if flag == "on":
                argv.append("--capture-training-numeric")
            write(folder / "executed_train_argv.json", argv)
            write(folder / "result.json", {"status": "PASS", "capture": flag, "task": task, "device": "cuda:0",
                "subject_kind": "reference", "source_argv_sha256": "sha256:" + "f" * 64, "program_source_checksum": program,
                "diagnostic_workload": {"env_steps": steps * 16}, "initial_policy_state_hash": "d" * 64,
                "numeric_evidence": training["numeric_evidence"], "wall_seconds": 21.0 if flag == "on" else 20.0,
                "training_summary_sha256": READER.sha(folder / "training_summary.json")})
            Path(str(folder) + ".time").write_text("peak_rss_kib=1000 elapsed_s=22.0 exit=0\n", encoding="utf-8")
        return on, off

    def test_two_task_pairs_compare_state_not_container_file_hash(self):
        for task in ("stair", "push"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as folder:
                on, off = self.pair(Path(folder), task)
                report = READER.verify(ROOT, on, off)
                self.assertEqual(report["status"], "PASS")
                self.assertFalse(report["checkpoint_states"]["file_hash_equal"])
                self.assertTrue(report["checkpoint_states"]["model_optimizer_exact"])
                self.assertGreater(report["cost"]["added_numeric_archive_bytes"], 0)

    def test_optimizer_change_fails_and_missing_rss_is_not_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            on, off = self.pair(Path(folder), "push")
            Path(str(on) + ".time").unlink()
            report = READER.verify(ROOT, on, off)
            self.assertEqual(report["status"], "INCOMPLETE_RESOURCE_MEASUREMENT")
            self.assertIsNone(report["cost"]["max_rss_bytes_delta"])
            checkpoint = off / "smoke_checkpoint.pt"
            state = torch.load(checkpoint, weights_only=True)
            state["optimizer_state_dict"]["state"][0]["exp_avg"][0] += 1
            torch.save(state, checkpoint)
            # Keep the synthetic receipts internally bound, so the failure is
            # optimizer invariance rather than merely a file checksum mismatch.
            summary = READER.read(off / "training_summary.json")
            summary["checkpoint"]["checksum"] = READER.sha(checkpoint)
            write(off / "training_summary.json", summary)
            result = READER.read(off / "result.json")
            result["training_summary_sha256"] = READER.sha(off / "training_summary.json")
            write(off / "result.json", result)
            report = READER.verify(ROOT, on, off)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("model_optimizer_exact", report["failed_checks"])

    def test_runtime_step_count_and_execution_config_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            on, off = self.pair(Path(folder), "stair")
            summary = READER.read(off / "training_summary.json")
            summary["runtime_activation_summary"]["step_calls"] -= 1
            write(off / "training_summary.json", summary)
            result = READER.read(off / "result.json")
            result["training_summary_sha256"] = READER.sha(off / "training_summary.json")
            write(off / "result.json", result)
            report = READER.verify(ROOT, on, off)
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("actual_training_steps", report["failed_checks"])

    def test_resource_limits_allow_boundary_and_reject_each_excess(self):
        limits = {"archive_bytes": 2 * 1024**3, "chunk_raw_bytes": [64 * 1024**2],
                  "paired_rss_delta": 512 * 1024**2}
        self.assertTrue(all(READER.resource_budget_checks(**limits).values()))
        for field, value, failed in (
            ("archive_bytes", limits["archive_bytes"] + 1, "numeric_archive_within_2_gib"),
            ("chunk_raw_bytes", [limits["chunk_raw_bytes"][0] + 1], "numeric_chunks_within_64_mib"),
            ("paired_rss_delta", limits["paired_rss_delta"] + 1, "paired_extra_rss_within_512_mib"),
        ):
            with self.subTest(field=field):
                checks = READER.resource_budget_checks(**{**limits, field: value})
                self.assertFalse(checks[failed])
                self.assertEqual(sum(not passed for passed in checks.values()), 1)
        missing = READER.resource_budget_checks(**{**limits, "paired_rss_delta": None})
        self.assertNotIn("paired_extra_rss_within_512_mib", missing)

    def test_actual_rss_excess_fails_pair_but_training_wall_ratio_is_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            on, off = self.pair(Path(folder), "push")
            result = READER.read(on / "result.json")
            result["wall_seconds"] = 200.0  # No invented 25% training-throughput gate.
            write(on / "result.json", result)
            within = READER.verify(ROOT, on, off)
            self.assertEqual(within["status"], "PASS")
            self.assertEqual(within["cost"]["wall_ratio"], 10.0)
            peak_kib = 1000 + 512 * 1024 + 1
            Path(str(on) + ".time").write_text(f"peak_rss_kib={peak_kib} elapsed_s=201.0 exit=0\n")
            over = READER.verify(ROOT, on, off)
            self.assertEqual(over["status"], "FAIL")
            self.assertIn("paired_extra_rss_within_512_mib", over["failed_checks"])


if __name__ == "__main__":
    unittest.main()
