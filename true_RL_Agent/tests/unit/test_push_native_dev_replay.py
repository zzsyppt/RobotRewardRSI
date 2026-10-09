"""P6 native replay verifies retained evidence without executing a simulator."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
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
SPEC = importlib.util.spec_from_file_location("push_native_dev_replay", ROOT / "PLAN/2new_world/P6/push_native_dev_replay.py")
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)

from scripts import qualify_reference as CLI

from tests.fakes.backend_environment import (
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
    TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
)
from tests.unit import test_push_phased_evaluation as phase_fixtures
from tests.unit.test_push_force_consumers import episode
from true_rl_agent.tasks.push_recovery.metrics import FORCE_ROLLOUT_SCHEMA
from true_rl_agent.tasks.push_recovery.rollout import aggregate_recovery_episodes
from true_rl_agent.agent_core.trial_receipts import TrialReceiptJournal
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.subprocess_runner import SubprocessRunner


def files(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in root.rglob("*") if path.is_file()}


def fixture(root, cell, **options):
    formal = root / "formal"
    args = SimpleNamespace(formal_dir=formal, task_id="unitree_push_recovery", cell=cell,
        device="cuda:1", isaac_python=sys.executable, **options)
    extra = [part for key, value in options.items() for part in ("--" + key.replace("_", "-"), str(value))]
    with redirect_stdout(io.StringIO()):
        CLI.main(["--task-id", args.task_id, "--cell-variant-id", cell,
            "--execution-device", args.device, "--isaac-python", args.isaac_python,
            "--output-dir", str(formal), "--session-id", "native-fixture",
            "--lease-root", str(root / "leases"), "--mode", "preflight", *extra])
    contract = REPLAY.load_strict_json_object(formal / "reference_qualification_execution.json")
    (formal / "reference_qualification_started.json").write_text(json.dumps({
        "contract_hash": REPLAY.canonical_json_hash(contract), "execution_device": args.device,
        "resource_binding_hash": "e" * 64}), encoding="utf-8")
    composition, contract = REPLAY.load_composition(args)
    backend = composition.backend_adapter
    jobs = backend.materialize_reference_jobs(session_id=contract["session_id"],
        task_spec=composition.task_plugin.task_spec, reference=composition.reference_program_artifact,
        artifact_root=formal / "reference_program_artifacts", policy_initialization=composition.policy_initialization,
        backend_execution_environment_manifest_hash=TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH)
    train = jobs.train_spec
    checkpoint = Path(train.metadata["checkpoint_output_path"])
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"UNIT TEST CHECKPOINT; NEVER EXECUTED OR PROMOTED")
    binding = train.training_binding
    summary = {**train.job_identity().to_public_dict(), "status": "pass", "execution_device": args.device,
        "training_binding_hash": binding.binding_hash,
        **{name: getattr(binding, name) for name in ("policy_architecture_hash", "optimizer_training_profile_hash",
            "agent_configuration_hash", "initialization_seed_schedule_hash", "capability_profile_hash")},
        "program_source_checksum": train.program_source_checksum,
        "resolved_cell_manifest_hash": composition.resolved_cell_manifest.manifest_hash,
        "resolved_cell_identity": composition.resolved_cell_manifest.summary_identity(),
        "backend_execution_environment_verification": TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
        "checkpoint": {"uri": str(checkpoint), "checksum": REPLAY.file_sha256(checkpoint),
            "role": "train_output", "sealed": False, "metadata": {"local_path": str(checkpoint)}},
        "training": build_budget_accounting(budget_steps=49152000, num_envs=4096,
            num_steps_per_env=24, max_iterations=500)}
    summary_path = Path(train.metadata["summary_path"])
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    trained = composition.runner._run_train_once(train, process_log_dir=None, reuse_existing_summary=True)
    plan = REPLAY.bind_eval_spec_to_checkpoint(jobs.eval_spec, trained.checkpoint)
    commitment = {"schema_version": "true_rl_agent.session_genesis.reference_phase_receipts.v1",
        "session_id": contract["session_id"], "genesis_identity_hash": contract["genesis_identity_hash"],
        "reference_manifest_hash": composition.reference_program_artifact.manifest.manifest_hash,
        "evaluation_plan_hash": REPLAY.canonical_json_hash(plan.to_public_dict())}
    journal = TrialReceiptJournal(formal / "reference_program_artifacts/evaluation_receipts", contract={
        "schema_version": REPLAY.TRIAL_EXECUTION_CONTRACT_SCHEMA_VERSION,
        "round_id": contract["session_id"], "session_genesis_hash": contract["genesis_identity_hash"],
        "ledger_epoch_hash": REPLAY.canonical_json_hash(commitment),
        "purpose": "session_genesis_reference_phased_evaluation", "commitment": commitment})
    journal.bind_contract()
    helper = phase_fixtures.PushPhasedEvaluationTests()
    helper.config = composition.task_plugin.config
    helper.adapter = composition.task_plugin.phased_evaluation_adapter()
    rows = []
    for index in range(64):
        row = episode(success=False, env_id=index)
        row.update(stratum=("low", "medium", "high", "repeated")[index // 16],
            planned_push_count=2 if index >= 48 else 1, push_event_count=0,
            completed_push_event_count=0, verified_push_event_count=0,
            force_events=[], failure_label="height_collapse", min_base_height_m=.1)
        rows.append(row)
    seed = {"schema_version": FORCE_ROLLOUT_SCHEMA, "status": "pass", "per_episode": rows,
            "aggregate": aggregate_recovery_episodes(rows, force_sequence=True)}
    for phase in REPLAY.EvaluationPhase:
        spec = backend.materialize_eval_phase(eval_plan=plan, phase=phase)
        result = helper.phase(plan, phase, success=0)
        evidence = {**result.task_evidence, "seed_summaries": [copy.deepcopy(seed) for _ in range(helper.adapter.seed_count(phase))]}
        journal.publish_eval_phase(spec, replace(result, job_id=spec.job_id, task_evidence=evidence))
    return args, composition, contract, plan, journal


class PushNativeReplayTests(unittest.TestCase):
    def test_explicit_training_settings_preserve_force_evaluation_and_native_read(self):
        from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "original_protocol.json"
            path.write_text(json.dumps(PushRecoveryProtocol.defaults().to_dict()), encoding="utf-8")
            options = {"push_protocol_path": path, "eval_base_seed": 42,
                "reference_program_id": "push_standing_recovery_reference_v2"}
            args, composition, _, _, _ = fixture(root, REPLAY.CELLS[0], **options)
            self.assertEqual(composition.task_plugin.config.push_protocol, PushRecoveryProtocol.defaults())
            self.assertIsNotNone(composition.task_plugin.config.push_evaluation_protocol)
            self.assertIn("push_evaluation_protocol", composition.resolved_cell_manifest.robot_task_binding["task_projection"])
            before = files(args.formal_dir)
            extra = [part for key, value in options.items()
                     for part in ("--" + key.replace("_", "-"), str(value))]
            with redirect_stdout(io.StringIO()):
                code = REPLAY.main(["--formal-dir", str(args.formal_dir), "--task-id", args.task_id,
                    "--cell", args.cell, "--device", args.device, "--output", str(root / "native"),
                    "--video-env-index", "3", "--mode", "preflight", *extra])
            self.assertEqual(code, 0)
            job = REPLAY.load_strict_json_object(root / "native/native_dev_job.json")
            self.assertEqual(job["argv"][job["argv"].index("--seed") + 1], "42")
            self.assertEqual(files(args.formal_dir), before)

    def test_explicit_stationary_reference_protocol_and_fresh_seed_preserve_formal_identity(self):
        from true_rl_agent.tasks.push_recovery.protocol import PushRecoveryProtocol

        for cell in REPLAY.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                protocol = replace(PushRecoveryProtocol.defaults(), low_velocity_scale=.225,
                    medium_velocity_scale=.45, high_velocity_scale=.75)
                path = root / "declared_protocol.json"
                path.write_text(json.dumps(protocol.to_dict()), encoding="utf-8")
                options = {"push_protocol_path": path, "eval_base_seed": 98765,
                    "reference_program_id": "push_standing_recovery_stationary_reference_v3"}
                args, composition, contract, plan, _ = fixture(root, cell, **options)
                self.assertEqual(composition.task_plugin.config.push_protocol, protocol)
                self.assertEqual(composition.reference_program_artifact.manifest.reference_id, options["reference_program_id"])
                before = files(args.formal_dir)
                output = root / "native"
                extra = [part for key, value in options.items() for part in ("--" + key.replace("_", "-"), str(value))]
                with redirect_stdout(io.StringIO()):
                    code = REPLAY.main(["--formal-dir", str(args.formal_dir), "--task-id", args.task_id,
                        "--cell", cell, "--device", args.device, "--output", str(output),
                        "--video-env-index", "3", "--mode", "preflight", *extra])
                self.assertEqual(code, 0)
                job = REPLAY.load_strict_json_object(output / "native_dev_job.json")
                self.assertEqual(job["argv"][job["argv"].index("--seed") + 1], "98765")
                self.assertEqual(job["spec"]["evaluation_plan"]["checkpoint"], plan.to_public_dict()["checkpoint"])
                expected = REPLAY.load_strict_json_object(output / "expected_public_dev.json")
                self.assertEqual(len(expected["episodes"]), 3)
                self.assertTrue(expected["phase_receipt"]["checksum"].startswith("sha256:"))
                args.eval_base_seed = 42
                with self.assertRaises(ValueError):
                    REPLAY.load_composition(args)
                self.assertEqual(files(args.formal_dir), before)

    def test_episode_comparison_detects_nested_event_and_unrecorded_env_drift(self):
        from true_rl_agent.tasks.push_recovery.seed_policy import derive_runner_seed

        episodes = [[episode(env_id=env) for env in range(64)] for _ in range(3)]
        expected = {"episodes": episodes, "phase_receipt": {"checksum": "sha256:" + "a" * 64}, "phase_result_hash": "b" * 64}
        result = SimpleNamespace(phase=REPLAY.EvaluationPhase.PUBLIC_DEV,
            task_evidence={"seed_summaries": [{"per_episode": copy.deepcopy(rows)} for rows in episodes]})
        recordings = [{"run_index": index, "env_index": 4, "split": "public_dev",
            "seed": derive_runner_seed(base_seed=98765, split_name="public_dev", run_index=index),
            "episode_outcome": copy.deepcopy(episodes[index - 1][4])} for index in range(1, 4)]
        compare = lambda: REPLAY.compare_dev_replay(expected, result, recordings, env_index=4, base_seed=98765)
        self.assertEqual(compare()["status"], "PASS")
        result.task_evidence["seed_summaries"][1]["per_episode"][63]["force_events"][0]["actual_substeps"] -= 1
        self.assertEqual(compare()["mismatched_episodes"], [{"run_index": 2, "env_id": 63}])
        self.assertEqual(compare()["status"], "FAIL")
        recordings[0]["episode_outcome"]["force_events"][0]["delivered_impulse_world_ns"][0] = .5
        self.assertFalse(compare()["selected_media_matches_formal_episodes"])

    def test_continuation_no_go_replay_uses_imported_train_and_declared_eval_environment(self):
        # The CLI tests own source admission; this checks the diagnostic consumes
        # that helper's original result while preserving the actual phase plan.
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(SubprocessRunner, "_run_job", side_effect=AssertionError("process launched")):
            root = Path(directory)
            args, composition, original_contract, plan, _ = fixture(root, REPLAY.CELLS[0])
            jobs = composition.backend_adapter.materialize_reference_jobs(
                session_id=original_contract["session_id"], task_spec=composition.task_plugin.task_spec,
                reference=composition.reference_program_artifact, artifact_root=root / "verified-source",
                policy_initialization=composition.policy_initialization,
                backend_execution_environment_manifest_hash=TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH)
            trained = composition.runner.recover_completed_train(jobs.train_spec)
            source = root / "retained-training"
            source.mkdir()
            (source / "contract.json").write_text(json.dumps(original_contract), encoding="utf-8")
            (source / "summary.json").write_bytes(Path(jobs.train_spec.metadata["summary_path"]).read_bytes())
            (source / "process.json").write_text(json.dumps({"argv": [sys.executable, "scripts/qualify_reference.py",
                "--task-id", args.task_id, "--cell-variant-id", args.cell, "--execution-device", args.device,
                "--isaac-python", args.isaac_python, "--output-dir", str(source),
                "--session-id", original_contract["session_id"], "--lease-root", str(root / "leases"),
                "--mode", "qualify"]}), encoding="utf-8")
            imported = {"source_output_dir": str(source), "session_id": original_contract["session_id"],
                "genesis_identity_hash": original_contract["genesis_identity_hash"],
                "source_files": {key: {"path": str(source / name)} for key, name in (
                    ("execution_contract", "contract.json"), ("process", "process.json"),
                    ("execution_environment", "environment.json"), ("summary", "summary.json"))},
                "qualification_wall_cap_s": 10000, "prior_qualification_wall_s": 4000,
                "remaining_qualification_wall_s": 6000}
            contract = {**original_contract, "schema_version": "true_rl_agent.reference_qualification_execution.v2",
                "train_jobs": 0, "reused_train_jobs": 1,
                "completed_training_import_hash": REPLAY.canonical_json_hash(imported),
                "evaluation_execution_environment_manifest_hash": TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH,
                **{key: imported[key] for key in ("qualification_wall_cap_s",
                    "prior_qualification_wall_s", "remaining_qualification_wall_s")}}
            formal = args.formal_dir
            (formal / CLI.IMPORT_FILENAME).write_text(json.dumps(imported), encoding="utf-8")
            (formal / "reference_qualification_execution.json").write_text(json.dumps(contract), encoding="utf-8")
            (formal / "backend_execution_environment.json").write_text(json.dumps({
                "manifest_hash": TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST_HASH}), encoding="utf-8")
            (formal / "reference_qualification_started.json").write_text(json.dumps({
                "contract_hash": REPLAY.canonical_json_hash(contract), "execution_device": args.device}), encoding="utf-8")
            before = files(formal), files(source)
            output = root / "native"
            output.mkdir()
            with patch.object(REPLAY.qualification_cli, "_verify_import_receipt"), \
                 patch.object(type(composition.backend_adapter), "validate_session_execution_environment"), \
                 patch.object(REPLAY.qualification_cli, "_load_completed_training",
                              return_value=((jobs.train_spec, trained), imported)) as recover:
                current, declared = REPLAY.load_composition(args)
                recovered, evidence = REPLAY.source_plan(current, declared, formal=formal, output=output)
            self.assertEqual(recovered.to_public_dict(), plan.to_public_dict())
            self.assertEqual(evidence["train_summary_path"], str(source / "summary.json"))
            self.assertEqual(evidence["qualification_gate"]["status"], "fail")
            self.assertEqual(recover.call_args.args[0].output_dir, formal)
            self.assertEqual(recover.call_args.args[0].source_output_dir, source)
            self.assertEqual(recover.call_args.args[2], original_contract)
            self.assertEqual((files(formal), files(source)), before)
            self.assertEqual(REPLAY.video_job(current, recovered, output=output, env_index=3).evaluation_plan.eval_num_envs, 64)

    def test_both_cells_recover_no_go_and_materialize_same_checkpoint_and_full_dev(self):
        for cell in REPLAY.CELLS:
            with self.subTest(cell=cell), tempfile.TemporaryDirectory() as directory, \
                 patch.object(SubprocessRunner, "_run_job", side_effect=AssertionError("process launched")):
                root = Path(directory)
                args, composition, contract, original, _ = fixture(root, cell)
                before = files(args.formal_dir)
                output = root / "replay"
                output.mkdir()
                recovered, source = REPLAY.source_plan(composition, contract, formal=args.formal_dir, output=output)
                self.assertEqual(recovered.to_public_dict(), original.to_public_dict())
                self.assertEqual(source["qualification_gate"]["status"], "fail")
                spec = REPLAY.video_job(composition, recovered, output=output, env_index=48)
                argv = spec.metadata["cli_argv"]
                for option, value in (("--public-seeds", "3"), ("--public-accept-seeds", "3"),
                    ("--sealed-validation-seeds", "7"), ("--num-envs", "64"),
                    ("--seed", "1500000000"),
                    ("--evaluation-phase", "public_dev"), ("--video-env-index", "48"),
                    ("--checkpoint-sha256", original.execution_checkpoint().checksum)):
                    self.assertEqual(argv[argv.index(option) + 1], value)
                self.assertIsNone(spec.attempt_scope)
                self.assertIsNone(spec.gpu_process_binding)
                self.assertTrue(Path(spec.metadata["summary_path"]).is_relative_to(output))
                with redirect_stdout(io.StringIO()):
                    code = REPLAY.main(["--formal-dir", str(args.formal_dir),
                        "--task-id", args.task_id, "--cell", args.cell, "--device", args.device,
                        "--isaac-python", args.isaac_python, "--output", str(root / "cli-preflight"),
                        "--video-env-index", "48", "--mode", "preflight"])
                self.assertEqual(code, 0)
                report = REPLAY.load_strict_json_object(root / "cli-preflight/result.json")
                self.assertEqual(report["status"], "PREPARED")
                self.assertEqual(report["budget"]["episodes"], 192)
                self.assertEqual(report["phases_executed"], [])
                self.assertEqual(files(args.formal_dir), before)

    def test_missing_phase_checkpoint_drift_and_invalid_existing_genesis_never_fall_through(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(SubprocessRunner, "_run_job", side_effect=AssertionError("process launched")):
            root = Path(directory)
            args, composition, contract, plan, journal = fixture(root, REPLAY.CELLS[0])
            output = root / "replay"
            output.mkdir()
            sealed = journal.root / "eval_phases/sealed_validation.json"
            saved = sealed.read_bytes()
            sealed.unlink()
            with self.assertRaisesRegex(ValueError, "every completed formal phase"):
                REPLAY.source_plan(composition, contract, formal=args.formal_dir, output=output)
            sealed.write_bytes(saved)
            Path(plan.execution_checkpoint().metadata["local_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes"):
                REPLAY.video_job(composition, plan, output=output, env_index=0)
            (args.formal_dir / REPLAY.SESSION_GENESIS_FILENAME).write_text("{}", encoding="utf-8")
            with patch.object(REPLAY, "recover_unqualified_plan", side_effect=AssertionError("fallback used")), \
                 self.assertRaises(ValueError):
                REPLAY.source_plan(composition, contract, formal=args.formal_dir, output=output)

    def test_qualified_genesis_is_loaded_with_execution_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args, composition, contract, plan, journal = fixture(root, REPLAY.CELLS[1])
            (args.formal_dir / REPLAY.SESSION_GENESIS_FILENAME).write_text("verified in mocked loader", encoding="utf-8")
            source = root / "private.json"
            source.write_text("{}", encoding="utf-8")
            dev_spec = composition.backend_adapter.materialize_eval_phase(eval_plan=plan, phase=REPLAY.EvaluationPhase.PUBLIC_DEV)
            _, dev_ref = journal.load_eval_phase(dev_spec)
            dev_receipt = REPLAY.load_trial_eval_phase_receipt(dev_ref)
            verified = SimpleNamespace(reference_eval_plan=lambda: plan,
                session_genesis_hash="f" * 64, reference_eval_artifact_path=lambda: source,
                reference_eval_phase_receipt=lambda phase: dev_receipt)
            with patch.object(REPLAY, "prepare_session_genesis", return_value=verified) as load, \
                 patch.object(REPLAY, "recover_unqualified_plan", side_effect=AssertionError("fallback used")):
                recovered, evidence = REPLAY.source_plan(composition, contract, formal=args.formal_dir, output=root / "unused")
            self.assertIs(recovered, plan)
            self.assertEqual(evidence["source_kind"], "qualified_reference_genesis")
            self.assertEqual(evidence["expected_public_dev"]["phase_receipt"], dev_ref.to_dict())
            self.assertTrue(load.call_args.kwargs["require_existing"])
            self.assertFalse(load.call_args.kwargs["bind_backend_execution_environment"])
            with self.assertRaisesRegex(RuntimeError, "load-only"):
                load.call_args.kwargs["runner"].run_train(None)


if __name__ == "__main__":
    unittest.main()
