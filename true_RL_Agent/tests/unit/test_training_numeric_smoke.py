from __future__ import annotations

import importlib.util
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import sys

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("training_numeric_smoke", ROOT / "scripts/training_numeric_smoke.py")
SMOKE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SMOKE)


class TrainingNumericSmokeTests(unittest.TestCase):
    def argv(self, *, task="push", subject="candidate"):
        script = "isaaclab_rsl_rl_train.py" if task == "push" else "isaaclab_stair_rsl_rl_train.py"
        return [sys.executable, str(ROOT / "scripts" / script), "--subject-kind", subject,
                "--init", "random", "--program-source-checksum", "sha256:" + "a" * 64,
                "--summary-path", "source-summary.json", "--checkpoint-output", "source.pt",
                "--num-envs", "16", "--num-steps-per-env", "8", "--max-iterations", "3", "--budget-steps", "384"]

    def test_prepare_preserves_admission_inputs_and_changes_only_capture_outputs(self):
        for task in ("push", "stair"):
            for subject in ("candidate", "reference"):
                source = self.argv(task=task, subject=subject) + ["--capture-training-numeric"]
                original = list(source)
                selected, output = SMOKE.prepare_argv({"metadata": {"cli_argv": source}}, output=Path("target"), capture=False)
                self.assertEqual(source, original)
                self.assertEqual(selected, task)
                self.assertNotIn("--capture-training-numeric", output)
                for option in ("--subject-kind", "--init", "--program-source-checksum", "--budget-steps"):
                    self.assertEqual(SMOKE.unique_option_value(output, option), SMOKE.unique_option_value(source, option))
                _, on = SMOKE.prepare_argv(output, output=Path("target"), capture=True)
                self.assertEqual(on.count("--capture-training-numeric"), 1)

    def test_wrong_entrypoint_formal_push_budget_and_warmstart_rejected(self):
        for option, value in (("--budget-steps", "49152000"), ("--init", "warm_start")):
            argv = self.argv()
            argv[argv.index(option) + 1] = value
            with self.assertRaises(ValueError):
                SMOKE.prepare_argv(argv, output=Path("unused"), capture=True)
        argv = self.argv()
        argv[1] = "unrelated.py"
        with self.assertRaises(ValueError):
            SMOKE.prepare_argv(argv, output=Path("unused"), capture=True)

    def test_preflight_never_discovers_gpu_or_launches_child(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "argv.json"
            source.write_text(json.dumps(self.argv()), encoding="utf-8")
            with patch.object(SMOKE, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU discovery")), \
                    patch.object(SMOKE, "run_owned_process", side_effect=AssertionError("child launch")):
                code = SMOKE.main(["--train-argv", str(source), "--output", str(root / "out"),
                                   "--capture", "on", "--device", "cuda:0"])
            self.assertEqual(code, 0)
            report = json.loads((root / "out/result.json").read_text())
            self.assertEqual(report["train_process_starts"], 0)
            self.assertFalse(report["formal_training_claim"])
            self.assertEqual(report["status"], "PREPARED_NOT_EXECUTED")

    def test_preflight_preserves_virtualenv_interpreter_when_resolve_points_to_base_python(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            interpreter = root / "venv/bin/python"
            interpreter.parent.mkdir(parents=True)
            interpreter.write_bytes(b"test executable location; never launched")
            base_python = Path(sys.executable).resolve()
            original_resolve = Path.resolve
            # Model the Linux venv/bin/python -> base-python symlink portably,
            # including on Windows hosts without symlink-creation privileges.
            def follow_virtualenv_link(path, *args, **kwargs):
                return base_python if path == interpreter else original_resolve(path, *args, **kwargs)
            source = root / "argv.json"
            source.write_text(json.dumps(self.argv()), encoding="utf-8")
            with patch.object(Path, "resolve", follow_virtualenv_link), \
                    patch.object(SMOKE, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU discovery")), \
                    patch.object(SMOKE, "run_owned_process", side_effect=AssertionError("child launch")):
                self.assertEqual(interpreter.resolve(), base_python)
                code = SMOKE.main(["--train-argv", str(source), "--output", str(root / "out"),
                    "--capture", "off", "--device", "cuda:0", "--isaac-python", str(interpreter)])
            self.assertEqual(code, 0)
            prepared = json.loads((root / "out/prepared_argv.json").read_text())
            self.assertEqual(prepared[0], str(interpreter))
            self.assertNotEqual(prepared[0], str(base_python))

    def test_stair_child_uses_qualified_smoke_and_never_claims_formal_training(self):
        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(device="cuda:0", headless=True, task_id="stair_climbing",
                base_task_protocol_hash="b" * 64, backend_execution_environment_manifest_hash="c" * 64,
                runtime_comparison_json=None, runtime_comparison_hash=None,
                summary_path=str(Path(folder) / "summary.json"), checkpoint_output=str(Path(folder) / "checkpoint.pt"),
                capture_training_numeric=True)
            context = SimpleNamespace(config=object(), agent_configuration=object(), identity=Mock(),
                                      training_binding=SimpleNamespace(binding_hash="d" * 64))
            context.identity.to_public_dict.return_value = {"subject_kind": "candidate"}
            entry = Mock()
            entry._build_parser.return_value.parse_args.return_value = args
            entry.require_execution_device.return_value = "cuda:0"
            entry.validate_stair_train_invocation.return_value = context
            entry.preflight_stair_execution_source_manifest.return_value.to_summary_dict.return_value = {"status": "pass"}
            entry.verify_current_isaac_lab_execution_environment.return_value = {"status": "pass"}
            entry._load_and_qualify_program.return_value.to_public_dict.return_value = {"source": "qualified"}
            entry.attach_gpu_runtime_evidence.side_effect = lambda fields, *_a, **_k: fields
            entry.close_isaac_app_on_success.return_value = 0
            execution = Mock()
            execution.to_summary_fields.return_value = {"status": "pass", "training_executed": True, "training": {"budget_steps": 128}}
            with patch.object(SMOKE.importlib.util, "spec_from_file_location", return_value=Mock()), \
                    patch.object(SMOKE.importlib.util, "module_from_spec", return_value=entry), \
                    patch.dict(sys.modules, {"isaaclab.app": SimpleNamespace(AppLauncher=Mock())}), \
                    patch("true_rl_agent.tasks.stair_climbing.training_diagnostics.load_stair_training_comparison", return_value=None), \
                    patch("true_rl_agent.tasks.stair_climbing.isaaclab_execution.execute_stair_training_smoke", return_value=execution) as run:
                self.assertEqual(SMOKE._stair_child(self.argv(task="stair")), 0)
            self.assertIs(run.call_args.kwargs["reward_program"], entry._load_and_qualify_program.return_value)
            self.assertTrue(run.call_args.kwargs["capture_training_numeric"])
            report = json.loads(Path(args.summary_path).read_text())
            self.assertFalse(report["formal_training_executed"])
            self.assertTrue(report["validation_training_executed"])

    def test_independent_saved_term_check_detects_bad_total(self):
        import numpy as np
        from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture
        for task in ("stair", "push"):
            for corrupt in (False, True):
                with tempfile.TemporaryDirectory() as folder:
                    capture = TrainingNumericCapture(Path(folder) / "numeric", identity={"test": True},
                        semantics={"terms": [{"index": 0, "name": "term", "weight": 2.0}]},
                        planned_control_steps=1, num_envs=2, steps_per_iteration=1, source_step_offset=0)
                    raw = np.asarray([1., 2.], dtype=np.float32)
                    actual = raw * 2 + (1 if corrupt else 0)
                    arrays = {"raw_term_000": raw, "weighted_term_000": raw * 2}
                    if task == "stair":
                        arrays["total_rate"] = actual
                    else:
                        arrays.update(candidate_total=actual, input_reward=raw, returned_reward=raw + actual)
                    capture.observe(1, arrays, metadata={"reward_weight_scale": 1.0})
                    evidence = capture.finish(training_status="pass")
                    if corrupt:
                        with self.assertRaisesRegex(ValueError, "sum"):
                            SMOKE.check_sampled_rewards(evidence, task=task)
                    else:
                        checked = SMOKE.check_sampled_rewards(evidence, task=task)
                        self.assertEqual(checked["verified_sampled_control_steps"], 1)
                        self.assertFalse(checked["ppo_post_manager_reward_observed"])

    def test_bound_coverage_identity_actual_index_and_denominators_are_checked(self):
        import numpy as np
        from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter, iter_numeric_evidence
        from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capture = TrainingNumericCapture(root / "original", identity={"job": {"job_id": "this-run"}},
                semantics={"terms": [{"index": 0, "name": "term", "weight": 2.0}]},
                planned_control_steps=24, num_envs=2, steps_per_iteration=8, source_step_offset=1)
            for step in range(1, 25):
                raw = np.asarray([1., 2.], dtype=np.float32)
                capture.observe(step + 1, {"raw_term_000": raw, "weighted_term_000": raw * 2,
                    "candidate_total": raw * 2, "input_reward": raw, "returned_reward": raw * 3},
                    metadata={"reward_weight_scale": 1.0})
            evidence = capture.finish(training_status="pass")
            expected_identity = {"job": {"job_id": "this-run"}}
            checked = SMOKE.check_sampled_rewards(evidence, task="push", expected_identity=expected_identity,
                                                  require_all_windows=True)
            self.assertEqual(checked["verified_sampled_env_transitions"], 48)
            self.assertEqual(checked["observed_training_env_transitions"], 48)
            self.assertTrue(checked["identity_matches"])
            with self.assertRaisesRegex(ValueError, "identity"):
                SMOKE.check_sampled_rewards(evidence, task="push", expected_identity={"job": {"job_id": "other"}})
            unbound = copy.deepcopy(evidence)
            unbound["coverage"]["num_envs"] += 1
            with self.assertRaisesRegex(ValueError, "bound numeric manifest"):
                SMOKE.check_sampled_rewards(unbound, task="push")
            original_chunks = list(iter_numeric_evidence(evidence["reference"]))
            for case in ("missing_sample", "wrong_index", "wrong_env", "wrong_count", "wrong_clock"):
                with self.subTest(case=case):
                    chunks = copy.deepcopy(original_chunks)
                    coverage = copy.deepcopy(evidence["coverage"])
                    if case == "missing_sample":
                        chunks.pop(0)
                    elif case == "wrong_index":
                        chunks[-1][1]["sampled_control_step"][0] = 99
                    elif case == "wrong_env":
                        chunks[0][1]["env_id"][0] = 1
                    elif case == "wrong_count":
                        coverage["windows"][0]["sampled_env_transitions"] += 1
                    else:
                        chunks[0][0]["source_control_step"] += 1
                    writer = NumericEvidenceWriter(root / case, expected_identity)
                    for metadata, arrays in chunks:
                        writer.write_chunk(arrays, metadata)
                    modified = {"status": "complete", "coverage": coverage, "reference": writer.complete(coverage)}
                    with self.assertRaises(ValueError):
                        SMOKE.check_sampled_rewards(modified, task="push", require_all_windows=True)

    def test_empty_archive_is_missing_arithmetic_and_cannot_pass_three_window_smoke(self):
        from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture
        with tempfile.TemporaryDirectory() as folder:
            capture = TrainingNumericCapture(Path(folder) / "numeric", identity={"test": True},
                semantics={"terms": []}, planned_control_steps=24, num_envs=2,
                steps_per_iteration=8, source_step_offset=0)
            evidence = capture.finish(training_status="fail")
            checked = SMOKE.check_sampled_rewards(evidence, task="push")
            self.assertIsNone(checked["raw_weighted_sum_arithmetic_matches"])
            with self.assertRaisesRegex(ValueError, "all three windows"):
                SMOKE.check_sampled_rewards(evidence, task="push", require_all_windows=True)

    def test_legacy_contiguous_v1_archive_is_not_reinterpreted_as_dispersed_v2(self):
        import numpy as np
        from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter
        with tempfile.TemporaryDirectory() as folder:
            writer = NumericEvidenceWriter(Path(folder) / "legacy", {"source": "synthetic-v1-contract"})
            windows = []
            index = []
            for name, start, end in (("early", 1, 8), ("middle", 45, 52), ("late", 89, 96)):
                steps = list(range(start, end + 1)); index.extend(steps)
                windows.append({"name": name, "start_step": start, "end_step": end,
                    "planned_sample_steps": 8, "sampled_steps": steps, "status": "sampled",
                    "sampled_control_steps": 8, "sampled_env_transitions": 16, "planned_sampled_env_transitions": 16})
                for step in steps:
                    raw = np.array([1., 2.], dtype=np.float32)
                    writer.write_chunk({"env_id": np.arange(2), "raw_term_000": raw,
                        "weighted_term_000": raw * 2, "total_rate": raw * 2},
                        {"kind": "training_sample", "window": name, "source_control_step": step + 1,
                         "training_control_step": step, "local_rollout_iteration": (step - 1) // 8,
                         "learning_iteration_from_control_steps": (step - 1) // 8,
                         "step_in_rollout": (step - 1) % 8 + 1, "task": {}})
            writer.write_chunk({"sampled_control_step": np.asarray(index)}, {"kind": "sampling_index"})
            coverage = {"schema_version": "true_rl_agent.training_numeric_capture.v1", "training_status": "pass",
                "planned_control_steps": 96, "observed_control_steps": 96, "steps_per_iteration": 8,
                "source_step_offset_excluded": 1, "initial_learning_iteration": 0,
                "sampled_env_ids": [0, 1], "num_envs": 2, "windows": windows,
                "semantics": {"terms": [{"index": 0, "weight": 2.0}]}}
            evidence = {"status": "complete", "coverage": coverage, "reference": writer.complete(coverage)}
            checked = SMOKE.check_sampled_rewards(evidence, task="stair", require_all_windows=True)
            self.assertEqual(checked["verified_sampled_control_steps"], 24)

    def test_dispersed_v2_checker_binds_long_plan_indices_and_rejects_changed_schedule(self):
        import numpy as np
        from true_rl_agent.runner.numeric_evidence import NumericEvidenceWriter, iter_numeric_evidence
        from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capture = TrainingNumericCapture(root / "v2", identity={},
                semantics={"terms": [{"index": 0, "weight": 2.0}]},
                planned_control_steps=48000, num_envs=2, steps_per_iteration=24, source_step_offset=0)
            raw = np.array([1., 2.], dtype=np.float32)
            for step in range(1, 48001):
                capture.observe(step, {"raw_term_000": raw, "weighted_term_000": raw * 2,
                                      "total_rate": raw * 2} if capture.wants_step(step) else None)
            evidence = capture.finish(training_status="pass")
            checked = SMOKE.check_sampled_rewards(evidence, task="stair", require_all_windows=True)
            self.assertEqual(checked["verified_sampled_control_steps"], 24)
            self.assertEqual(evidence["coverage"]["windows"][0]["planned_steps"],
                             [1, 3000, 5000, 7000, 9000, 11000, 13000, 15000])
            early_failure = TrainingNumericCapture(root / "early-failure", identity={},
                semantics={"terms": [{"index": 0, "weight": 2.0}]},
                planned_control_steps=48000, num_envs=2, steps_per_iteration=24, source_step_offset=0)
            for step in range(1, 101):
                early_failure.observe(step, {"raw_term_000": raw, "weighted_term_000": raw * 2,
                                            "total_rate": raw * 2} if early_failure.wants_step(step) else None)
            partial = early_failure.finish(training_status="fail")
            partial_checked = SMOKE.check_sampled_rewards(partial, task="stair")
            self.assertEqual(partial_checked["verified_sampled_control_steps"], 1)
            self.assertEqual(partial_checked["verified_sampled_env_transitions"], 2)
            self.assertEqual(partial_checked["observed_training_env_transitions"], 200)
            self.assertTrue(partial_checked["raw_weighted_sum_arithmetic_matches"])
            with self.assertRaisesRegex(ValueError, "all three windows"):
                SMOKE.check_sampled_rewards(partial, task="stair", require_all_windows=True)
            coverage = copy.deepcopy(evidence["coverage"])
            coverage["windows"][0]["planned_steps"][0] += 1
            writer = NumericEvidenceWriter(root / "wrong-plan", {})
            for metadata, arrays in iter_numeric_evidence(evidence["reference"]):
                writer.write_chunk(arrays, metadata)
            modified = {"status": "complete", "coverage": coverage, "reference": writer.complete(coverage)}
            with self.assertRaisesRegex(ValueError, "sample plan"):
                SMOKE.check_sampled_rewards(modified, task="stair", require_all_windows=True)

    def test_expected_identity_uses_executed_job_and_program_for_both_tasks(self):
        for task in ("push", "stair"):
            with self.subTest(task=task):
                argv = self.argv(task=task) + ["--job-id", "this-run", "--task-id", task + "-task",
                    "--subject-id", "subject", "--training-binding-hash", "b" * 64,
                    "--task-protocol-hash", "c" * 64, "--seed", "42", "--run-dir", "run"]
                job = {"job_id": "this-run", "task_id": task + "-task", "subject_kind": "candidate", "subject_id": "subject"}
                checksum = "sha256:" + "a" * 64
                summary = {**job, "program_source_checksum": checksum, "training_binding_hash": "b" * 64,
                    "run": {"train_attempt_id": "attempt"}, "original_training_job_identity": job,
                    "reward_program": {"program_source_checksum": checksum},
                    "training": {"initial_policy_state_hash": "d" * 64},
                    "backend_execution_environment_verification": {"status": "pass"}}
                expected = SMOKE.expected_training_numeric_identity(summary, task=task, argv=argv)
                self.assertEqual(expected["job"], job)
                self.assertEqual(expected["training_seed"], 42)
                self.assertEqual(expected["run_dir"], str(Path("run") / "attempt-attempt") if task == "push" else "run")
                key = "reward_program" if task == "stair" else None
                if key:
                    summary[key]["program_source_checksum"] = "sha256:" + "f" * 64
                else:
                    summary["program_source_checksum"] = "sha256:" + "f" * 64
                with self.assertRaisesRegex(ValueError, "program"):
                    SMOKE.expected_training_numeric_identity(summary, task=task, argv=argv)

    def test_stair_identity_separates_real_v3_cell_from_ancestor_training_protocol(self):
        from true_rl_agent.composition.cell_manifest import materialize_resolved_cell_manifest
        from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
        from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
        from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id,
            config=config, plugin=StairClimbingTaskPlugin(config))
        ancestor_protocol = config.training_binding_task_protocol_hash
        self.assertNotEqual(ancestor_protocol, manifest.resolved_cell_task_protocol_hash)
        with tempfile.TemporaryDirectory() as folder:
            path = materialize_resolved_cell_manifest(manifest, output_dir=Path(folder))
            argv = self.argv(task="stair") + ["--job-id", "stair-run", "--task-id", manifest.task_id,
                "--subject-id", "subject", "--training-binding-hash", "b" * 64,
                "--task-protocol-hash", ancestor_protocol, "--seed", "42",
                "--resolved-cell-manifest-path", str(path), "--resolved-cell-manifest-hash", manifest.manifest_hash,
                "--base-task-protocol-hash", manifest.base_task_protocol_hash]
            job = {"job_id": "stair-run", "task_id": manifest.task_id,
                   "subject_kind": "candidate", "subject_id": "subject"}
            summary = {"original_training_job_identity": job,
                "reward_program": {"program_source_checksum": "sha256:" + "a" * 64},
                "training": {"initial_policy_state_hash": "d" * 64},
                "backend_execution_environment_verification": {"status": "pass"},
                "environment_source_preflight": {"status": "pass", "source_manifest_hash": "e" * 64,
                    "task_protocol_hash": manifest.resolved_cell_task_protocol_hash}}
            expected = SMOKE.expected_training_numeric_identity(summary, task="stair", argv=argv)
            self.assertEqual(expected["task_protocol_hash"], manifest.resolved_cell_task_protocol_hash)
            self.assertEqual(expected["resolved_cell_identity"], manifest.summary_identity())
            self.assertEqual(expected["resolved_cell_manifest_hash"], manifest.manifest_hash)
            self.assertEqual(expected["training_binding_hash"], "b" * 64)
            self.assertEqual(expected["environment_source_preflight"], summary["environment_source_preflight"])
            wrong_hash = list(argv)
            wrong_hash[wrong_hash.index("--resolved-cell-manifest-hash") + 1] = "f" * 64
            with self.assertRaisesRegex(ValueError, "manifest argv hash mismatch"):
                SMOKE.expected_training_numeric_identity(summary, task="stair", argv=wrong_hash)
            summary["environment_source_preflight"]["task_protocol_hash"] = ancestor_protocol
            with self.assertRaisesRegex(ValueError, "preflight differs"):
                SMOKE.expected_training_numeric_identity(summary, task="stair", argv=argv)


if __name__ == "__main__":
    unittest.main()
