"""Reference CLI preflight and reload never spend an execution budget."""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
_SPEC = importlib.util.spec_from_file_location("qualify_reference_script", ROOT / "scripts/qualify_reference.py")
CLI = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(CLI)

from tests.fakes.backend_environment import TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST
from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.runner.gpu_lease import GpuDeviceIdentity

CELLS = ("push.g1_37dof.recovery_v3", "push.go2_12dof.recovery_v3")


def _argv(output: Path, mode: str, cell: str = CELLS[0]) -> list[str]:
    return ["--task-id", "unitree_push_recovery", "--cell-variant-id", cell,
            "--output-dir", str(output), "--session-id", "reference-cli-test",
            "--lease-root", str(output / "leases"), "--execution-device", "cuda:1",
            "--mode", mode]


def _files(output: Path):
    return {str(path.relative_to(output)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in output.rglob("*") if path.is_file()}


class QualifyReferenceScriptTests(unittest.TestCase):
    def test_stair_preflight_uses_only_the_shared_workload_interface(self):
        provider = CLI.build_application_composition_registry().get("unitree_stair_climbing")
        with tempfile.TemporaryDirectory() as directory:
            argv = _argv(Path(directory), "preflight")
            argv[argv.index("--task-id") + 1] = provider.task_id
            argv[argv.index("--cell-variant-id") + 1] = provider.cli_defaults()["cell_variant_id"]
            with patch.object(CLI, "prepare_session_genesis", side_effect=AssertionError("executed")), \
                 patch.object(CLI, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU queried")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(CLI.main(argv), 0)

    def test_both_cells_preflight_materializes_real_jobs_without_execution(self):
        for cell in CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                output = Path(directory)
                with patch.object(CLI, "prepare_session_genesis", side_effect=AssertionError("executed")), \
                     patch.object(CLI, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU queried")), \
                     redirect_stdout(io.StringIO()) as stream:
                    self.assertEqual(CLI.main(_argv(output, "preflight", cell)), 0)
                report = json.loads(stream.getvalue())
                self.assertEqual(report["cell_variant_id"], cell)
                self.assertEqual(len(report["train_job_spec_hash"]), 64)
                self.assertEqual(len(report["evaluation_plan_hash"]), 64)
                self.assertEqual(report["evaluation_phases"], ["public_dev", "public_accept", "sealed_validation"])
                contract = json.loads((output / "reference_qualification_execution.json").read_text())
                self.assertEqual((contract["train_jobs"], contract["eval_jobs"], contract["infrastructure_retries"]), (1, 3, 0))
                self.assertEqual(contract["workload"]["train_steps_per_job"], 500 * 24 * 4096)
                self.assertEqual(contract["max_reference_eval_seed_units"], 13)
                self.assertFalse((output / "session_genesis.json").exists())
                self.assertFalse((output / "reference_qualification_started.json").exists())

    def test_load_only_missing_or_invalid_genesis_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with redirect_stdout(io.StringIO()):
                CLI.main(_argv(output, "preflight"))
            for invalid in (False, True):
                with self.subTest(invalid=invalid):
                    if invalid:
                        (output / "session_genesis.json").write_text("{}", encoding="utf-8")
                    before = _files(output)
                    with patch.object(CLI, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU queried")), \
                         patch.object(CLI, "_preflight_jobs", side_effect=AssertionError("materialized")), \
                         self.assertRaises((FileNotFoundError, ValueError)):
                        CLI.main(_argv(output, "load-only"))
                    self.assertEqual(_files(output), before)

    def test_push_owned_cli_selection_reaches_preflight(self):
        from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol
        from true_rl_agent.tasks.push_recovery.reference_program import STATIONARY_REFERENCE_ID

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            protocol = replace(PushRecoveryProtocol.defaults(), high_velocity_scale=0.6)
            path = output / "protocol.json"
            path.write_text(json.dumps(protocol.to_dict()), encoding="utf-8")
            argv = _argv(output / "run", "preflight") + [
                "--eval-base-seed", "1000000000", "--reference-program-id", STATIONARY_REFERENCE_ID,
                "--push-protocol-path", str(path)]
            with patch.object(CLI, "_preflight_jobs", wraps=CLI._preflight_jobs) as materialize, \
                 patch.object(CLI, "prepare_session_genesis", side_effect=AssertionError("executed")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(CLI.main(argv), 0)
            composition = materialize.call_args.args[0]
            self.assertEqual(composition.task_plugin.config.eval_base_seed, 1000000000)
            self.assertEqual(composition.task_plugin.config.push_protocol, protocol)
            self.assertEqual(composition.reference_program_artifact.manifest.reference_id, STATIONARY_REFERENCE_ID)
            self.assertEqual(composition.task_defaults["push_evaluation_protocol"],
                             composition.task_plugin.config.push_evaluation_protocol.to_dict())

    def test_omitted_seed_defers_to_the_selected_cell_and_explicit_seed_wins(self):
        for cell in CELLS:
            for explicit_seed in (None, 17):
                with self.subTest(cell=cell, explicit_seed=explicit_seed), tempfile.TemporaryDirectory() as directory:
                    output = Path(directory)
                    argv = _argv(output, "preflight", cell)
                    if explicit_seed is not None:
                        argv.extend(("--eval-base-seed", str(explicit_seed)))
                    args = CLI.parse_args(argv)
                    self.assertEqual(args.eval_base_seed, explicit_seed)
                    composition = CLI.resolve_task_composition(provider=args._provider,
                        values={**args._provider.cli_defaults(), **vars(args)},
                        context=CLI.TaskCompositionContext(project_root=ROOT, output_dir=output))
                    self.assertEqual(composition.backend_adapter.config.eval_base_seed,
                                     1500000000 if explicit_seed is None else explicit_seed)

    def test_load_only_runner_rejects_every_execution_entrypoint(self):
        runner = CLI._LoadOnlyRunner()
        for entrypoint in ("run_train", "run_eval", "run_eval_phase", "run_baseline_eval"):
            with self.subTest(entrypoint=entrypoint), self.assertRaisesRegex(RuntimeError, "load-only"):
                getattr(runner, entrypoint)(None)

    def test_qualification_rejects_masked_gpu_ordinals_before_inventory_or_execution(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), \
             patch.object(CLI, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU queried")), \
             patch.object(CLI, "prepare_session_genesis", side_effect=AssertionError("executed")), \
             self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES unset"):
            CLI.main(_argv(Path(directory), "qualify"))

    def test_qualification_uses_direct_genesis_under_existing_gpu_lease_and_never_retries(self):
        identity = GpuDeviceIdentity(execution_device="cuda:1", machine_id="test-host", physical_id="test-gpu")
        gpu = SimpleNamespace(execution_device="cuda:1", identity=identity)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            def fail_after_admission(**kwargs):
                self.assertFalse(kwargs["require_existing"])
                self.assertTrue(kwargs["bind_backend_execution_environment"])
                for unsupported in ("gpu_process_binding", "generation_session_contract_ref",
                                    "reference_train_attempt_scope", "reference_eval_phase_attempt_scopes"):
                    self.assertNotIn(unsupported, kwargs)
                saved_job = CLI.load_strict_json_object(output / "reference_train_job.json")
                saved_env = CLI.load_strict_json_object(output / "backend_execution_environment.json")
                self.assertEqual(saved_job["metadata"]["backend_execution_environment_manifest_hash"], saved_env["manifest_hash"])
                admission = json.loads((output / "reference_qualification_started.json").read_text())
                self.assertEqual(admission["resource_binding_hash"], identity.identity_hash)
                self.assertEqual(admission["execution_device"], "cuda:1")
                raise RuntimeError("injected execution failure")

            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), \
                 patch.object(CLI, "discover_nvidia_gpu_inventory", return_value=[gpu]), \
                 patch.object(CLI, "isolate_orchestrator_process_group"), \
                 patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment",
                              return_value=TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST), \
                 patch.object(CLI, "prepare_session_genesis", side_effect=fail_after_admission) as prepare:
                with self.assertRaisesRegex(RuntimeError, "injected execution failure"):
                    CLI.main(_argv(output, "qualify"))
                with self.assertRaisesRegex(ValueError, "already attempted"):
                    CLI.main(_argv(output, "qualify"))
                self.assertEqual(prepare.call_count, 1)


    def _completed_source(self, parent):
        source, output = parent / "original", parent / "continuation"
        argv = _argv(source, "preflight")
        with redirect_stdout(io.StringIO()):
            CLI.main(argv)
        parsed = CLI.parse_args(argv)
        composition = CLI.resolve_task_composition(
            provider=parsed._provider, values={**parsed._provider.cli_defaults(), **vars(parsed)},
            context=CLI.TaskCompositionContext(project_root=ROOT, output_dir=source, read_only=True))
        environment = TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST
        CLI._preflight_jobs(composition, output=source, session_id=parsed.session_id, execution_environment=environment)
        contract = CLI.load_strict_json_object(source / "reference_qualification_execution.json")
        CLI._publish_json(source / "reference_qualification_started.json", {
            "contract_hash": CLI.canonical_json_hash(contract), "execution_device": "cuda:1", "resource_binding_hash": "a" * 64})
        job = CLI.load_strict_json_object(source / "reference_train_job.json")
        summary = Path(job["metadata"]["summary_path"])
        summary.parent.mkdir(parents=True)
        summary.write_text('{"status":"pass","training":{"max_iterations":500}}', encoding="utf-8")
        checkpoint = Path(job["metadata"]["checkpoint_output_path"])
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"original completed checkpoint")
        process = parent / "original_process.json"
        command = [sys.executable, str(ROOT / "scripts/qualify_reference.py"), *_argv(source, "qualify")]
        CLI._publish_json(process, {"argv": command, "argv_hash": CLI.canonical_json_hash(command),
            "cwd": str(ROOT), "elapsed_wall_s": 600.0, "wall_timeout_s": 14400,
            "started_at_unix": 100.0, "ended_at_unix": 700.0, "wrapper_retries": 0, "returncode": 1})
        result = SimpleNamespace(summary={"formal_training_executed": True},
            checkpoint=SimpleNamespace(metadata={"local_path": str(checkpoint)}, checksum=CLI.file_sha256(checkpoint)),
            to_public_dict=lambda: {"fixture": "verified completed result"})
        resume = [*_argv(output, "resume-evaluation"), "--source-output-dir", str(source),
            "--source-execution-environment", str(source / "backend_execution_environment.json"),
            "--prior-process-receipt", str(process)]
        return source, output, resume, result

    def test_resume_evaluation_preserves_original_evidence_and_load_only_rechecks_sources(self):
        identity = GpuDeviceIdentity(execution_device="cuda:1", machine_id="test-host", physical_id="test-gpu")
        gpu = SimpleNamespace(execution_device="cuda:1", identity=identity)
        genesis = SimpleNamespace(session_genesis_hash="b" * 64,
            initial_current_best=SimpleNamespace(to_public_dict=lambda: {}))
        with tempfile.TemporaryDirectory() as directory:
            source, output, argv, completed = self._completed_source(Path(directory))
            before = _files(source)
            new_environment = replace(CLI.IsaacLabExecutionEnvironmentManifest.from_mapping(
                TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST), manifest_hash=None,
                project_source={"schema_version": "true_rl_agent.project_source_snapshot.v1",
                                "files": {"src/host.py": "c" * 64}}).to_dict()
            def prepare(**kwargs):
                original_spec, result = kwargs["completed_reference_training"]
                self.assertIs(result, completed)
                self.assertEqual(Path(original_spec.metadata["summary_path"]).parents[3], source)
                self.assertEqual(original_spec.metadata["backend_execution_environment_manifest_hash"],
                                 TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST["manifest_hash"])
                CLI._publish_json(output / "session_genesis.json", {"fixture": "only the shared seam is mocked"})
                return genesis
            with patch.object(SubprocessRunner, "recover_completed_train", return_value=completed) as recover, \
                 patch.object(SubprocessRunner, "run_train", side_effect=AssertionError("training repeated")), \
                 patch.object(CLI, "prepare_session_genesis", side_effect=prepare), \
                 patch.object(CLI, "discover_nvidia_gpu_inventory", return_value=[gpu]), \
                 patch.object(CLI, "isolate_orchestrator_process_group"), \
                 patch.object(IsaacLabRslRlBackendAdapter, "resolve_session_execution_environment",
                              return_value=new_environment), \
                 patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), redirect_stdout(io.StringIO()):
                self.assertEqual(CLI.main(argv), 0)
                self.assertEqual(recover.call_count, 1)
            self.assertEqual(_files(source), before)
            contract = CLI.load_strict_json_object(output / "reference_qualification_execution.json")
            self.assertEqual((contract["train_jobs"], contract["reused_train_jobs"], contract["eval_jobs"]), (0, 1, 3))
            self.assertEqual(contract["evaluation_execution_environment_manifest_hash"], new_environment["manifest_hash"])
            self.assertNotEqual(new_environment["manifest_hash"], TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST["manifest_hash"])
            self.assertEqual((contract["prior_qualification_wall_s"], contract["remaining_qualification_wall_s"]), (600, 13800))
            current = _files(output)
            with patch.object(CLI, "prepare_session_genesis", return_value=genesis), \
                 patch.object(CLI, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU queried")), \
                 patch.object(SubprocessRunner, "recover_completed_train", side_effect=AssertionError("train recovered again")), \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(CLI.main(_argv(output, "load-only")), 0)
            self.assertEqual(_files(output), current)
            summary = source / "runner/reference-cli-test/train/summary.json"
            summary.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source bytes changed"), redirect_stdout(io.StringIO()):
                CLI.main(_argv(output, "load-only"))

    def test_resume_rejects_missing_or_changed_source_before_recovery(self):
        for failure in ("missing_marker", "changed_job", "existing_eval", "spent_budget", "wrong_source"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                source, output, argv, completed = self._completed_source(Path(directory))
                if failure == "missing_marker":
                    (source / "reference_qualification_started.json").unlink()
                elif failure == "changed_job":
                    (source / "reference_train_job.json").write_text("{}", encoding="utf-8")
                elif failure == "existing_eval":
                    path = source / "runner/reference-cli-test/eval/public_dev/summary.json"
                    path.parent.mkdir(parents=True)
                    path.write_text("{}", encoding="utf-8")
                else:
                    path = Path(directory) / "original_process.json"
                    data = CLI.load_strict_json_object(path)
                    if failure == "spent_budget":
                        data["elapsed_wall_s"] = 14400
                    else:
                        data["argv"][data["argv"].index("--session-id") + 1] = "wrong-source"
                        data["argv_hash"] = CLI.canonical_json_hash(data["argv"])
                    path.write_text(json.dumps(data), encoding="utf-8")
                with patch.object(SubprocessRunner, "recover_completed_train", side_effect=AssertionError("recovery admitted")), \
                     patch.object(CLI, "discover_nvidia_gpu_inventory", side_effect=AssertionError("GPU queried")), \
                     self.assertRaises((ValueError, FileNotFoundError)), redirect_stdout(io.StringIO()):
                    CLI.main(argv)
                self.assertFalse((output / "reference_qualification_started.json").exists())

    def test_original_missing_job_entity_is_reconstructed_without_source_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output, argv, completed = self._completed_source(Path(directory))
            (source / "reference_train_job.json").unlink()
            before = _files(source)
            args = CLI.parse_args(argv)
            composition = CLI.resolve_task_composition(
                provider=args._provider, values={**args._provider.cli_defaults(), **vars(args)},
                context=CLI.TaskCompositionContext(project_root=ROOT, output_dir=output))
            contract = CLI.load_strict_json_object(source / "reference_qualification_execution.json")
            with patch.object(SubprocessRunner, "recover_completed_train", return_value=completed), \
                 patch.object(SubprocessRunner, "run_train", side_effect=AssertionError("training repeated")):
                pair, receipt = CLI._load_completed_training(args, composition, contract)
            self.assertIs(pair[1], completed)
            self.assertEqual(receipt["train_job_source"], "reconstructed_from_verified_original_configuration")
            self.assertEqual(_files(source), before)
            for role in ("checkpoint", "execution_environment"):
                path = Path(receipt["source_files"][role]["path"])
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "source bytes changed"):
                    CLI._verify_import_receipt(receipt)
                path.write_bytes(original)
            missing = Path(receipt["source_files"]["checkpoint"]["path"])
            missing.unlink()
            with self.assertRaises(FileNotFoundError):
                CLI._verify_import_receipt(receipt)

    def test_resume_requires_explicit_sources_and_clean_disjoint_output(self):
        with tempfile.TemporaryDirectory() as directory, patch("sys.stderr", io.StringIO()):
            source, output, argv, _ = self._completed_source(Path(directory))
            with self.assertRaises(SystemExit):
                CLI.parse_args(_argv(output, "resume-evaluation"))
            output.mkdir()
            (output / "unrelated.txt").write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new clean output"):
                CLI.main(argv)


    def test_task_and_cell_must_be_explicit_and_retired_cells_cannot_create_sessions(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = _argv(Path(directory), "preflight")
            for name in ("--task-id", "--cell-variant-id"):
                index = argv.index(name)
                with self.subTest(argument=name), redirect_stdout(io.StringIO()), \
                     patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    CLI.parse_args(argv[:index] + argv[index + 2:])
            for retired in ("push.g1_37dof.recovery_v2", "push.go2_12dof.recovery_v2"):
                with self.subTest(retired_cell=retired), patch("sys.stderr", io.StringIO()), self.assertRaises(SystemExit):
                    CLI.parse_args(_argv(Path(directory), "preflight", retired))


if __name__ == "__main__":
    unittest.main()
