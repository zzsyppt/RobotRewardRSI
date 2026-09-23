"""A completed reference train can precede evaluation in a fresh execution directory."""

from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.unit import test_reference_genesis as fixtures
from true_rl_agent.agent_core.session_genesis import prepare_session_genesis
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION, GenerationSessionContract,
    publish_generation_session_contract,
)
from true_rl_agent.agent_core.session_genesis import session_genesis_identity_hash
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.specs import GpuProcessBinding
from true_rl_agent.agent_core.generation_preparation import GenerationLaunchConfig
from true_rl_agent.agent_core import generation_runtime
from true_rl_agent.runner.subprocess_runner import SubprocessRunner, RunnerExecutionError
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.checkpoint import file_sha256


class _EnvironmentBackend(fixtures._ReferenceBackend):
    def __init__(self, environment_hash):
        self.environment_hash = environment_hash

    def resolve_session_execution_environment(self, **kwargs):
        return {"manifest_hash": self.environment_hash}

    def validate_session_execution_environment(self, *, execution_environment):
        if execution_environment != {"manifest_hash": self.environment_hash}:
            raise ValueError("execution source drift")

    def materialize_reference_jobs(self, *, backend_execution_environment_manifest_hash, **kwargs):
        jobs = super().materialize_reference_jobs(**kwargs)
        metadata = {"backend_execution_environment_manifest_hash": backend_execution_environment_manifest_hash}
        def bind(spec):
            argv = [*spec.metadata["cli_argv"], "--backend-execution-environment-manifest-hash",
                    backend_execution_environment_manifest_hash]
            return replace(spec, metadata={**spec.metadata, **metadata, "cli_argv": argv, "cli_command": " ".join(argv)})
        return replace(
            jobs,
            train_spec=bind(jobs.train_spec),
            eval_spec=bind(jobs.eval_spec),
        )


class _CostPhasedReferenceRunner(fixtures._PhasedReferenceRunner):
    def run_eval_phase(self, eval_spec):
        result = super().run_eval_phase(eval_spec)
        return replace(result, execution_cost={"splits": {
            eval_spec.phase.value: {"seed_count": 1, "simulated_env_control_steps": 10}}})


class CompletedReferenceTrainingTests(unittest.TestCase):
    def test_real_runner_recovers_bound_reference_without_launch_or_source_writes(self):
        for case in ("bound", "wrong_binding", "unbound_claim"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task = fixtures._ReferenceTaskPlugin()
                _, reference = fixtures._reference_program_for_task(root, task)
                binding = GpuProcessBinding(host_execution_device="cuda:1", process_execution_device="cuda:0",
                                            resource_binding_hash="d" * 64)
                spec = fixtures._ReferenceBackend().materialize_reference_jobs(
                    session_id="original-reference", task_spec=task.task_spec, reference=reference,
                    artifact_root=root / "reference_program_artifacts", policy_initialization=PolicyInitialization.random(),
                    gpu_process_binding=(None if case == "unbound_claim" else binding),
                ).train_spec
                argv = [*spec.metadata["cli_argv"], "--num-envs", "1", "--num-steps-per-env", "1", "--max-iterations", "128"]
                spec = replace(spec, metadata={**spec.metadata, "cli_argv": argv, "cli_command": " ".join(argv)})
                checkpoint = Path(spec.metadata["checkpoint_output_path"])
                checkpoint.parent.mkdir(parents=True)
                checkpoint.write_bytes(b"original-completed-reference")
                budget = build_budget_accounting(budget_steps=128, num_envs=1, num_steps_per_env=1, max_iterations=128)
                summary = {
                    "job_id": spec.job_id, "task_id": spec.task_id,
                    "subject_kind": "reference", "subject_id": spec.subject.subject_id,
                    "subject_manifest_hash": spec.subject_manifest_hash, "status": "pass",
                    "training_binding_hash": spec.training_binding.binding_hash,
                    **{name: getattr(spec.training_binding, name) for name in (
                        "policy_architecture_hash", "optimizer_training_profile_hash", "agent_configuration_hash",
                        "initialization_seed_schedule_hash", "capability_profile_hash")},
                    "program_source_checksum": spec.program_source_checksum, "execution_device": "cuda:0",
                    "gpu_process_binding": (replace(binding, resource_binding_hash="e" * 64).to_dict()
                                            if case == "wrong_binding" else binding.to_dict()),
                    "owned_process_identity": {"schema_version": "true_rl_agent.runner.owned_process_identity.v1",
                                               "process_id": 123456, "process_group_id": 123456},
                    "training": {"budget_steps": 128, "num_envs": 1, "num_steps_per_env": 1, "max_iterations": 128,
                                 "latest_checkpoint_iteration": 127, "initial_policy_state_hash": "f" * 64,
                                 "budget_accounting": budget},
                    "checkpoint": {"uri": str(checkpoint), "checksum": file_sha256(checkpoint),
                                   "role": "train_output", "metadata": {"local_path": str(checkpoint)}},
                }
                path = Path(spec.metadata["summary_path"])
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(summary), encoding="utf-8")
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}
                runner = SubprocessRunner(allowed_executables=(argv[0],))
                with patch.object(SubprocessRunner, "_execute_with_attempts", side_effect=AssertionError("new attempt")), \
                     patch.object(SubprocessRunner, "_clear_existing_summary", side_effect=AssertionError("source write")), \
                     patch("true_rl_agent.runner.subprocess_runner.run_owned_process", side_effect=AssertionError("new process")):
                    if case == "bound":
                        recovered = runner.recover_completed_train(spec)
                        self.assertEqual(recovered.gpu_process_binding, binding)
                        self.assertEqual(recovered.job_id, spec.job_id)
                        self.assertTrue(recovered.summary["formal_training_executed"])
                        self.assertEqual(recovered.summary["runner_training_completion"]["source_summary_sha256"], file_sha256(path))
                        self.assertEqual(recovered.checkpoint.metadata["gpu_process_binding"], binding.to_dict())
                        self.assertEqual(recovered.summary["owned_process_identity"], summary["owned_process_identity"])
                    else:
                        message = "trusted job binding" if case == "wrong_binding" else "unbound summary cannot claim"
                        with self.assertRaisesRegex(RunnerExecutionError, message):
                            runner.recover_completed_train(spec)
                self.assertEqual(set(before), {p for p in root.rglob("*") if p.is_file()})
                for p, (content, mtime) in before.items():
                    self.assertEqual(p.read_bytes(), content)
                    self.assertEqual(p.stat().st_mtime_ns, mtime)

    def test_formal_host_routes_training_source_with_only_eval_attempt_scopes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            config = GenerationLaunchConfig(
                project_root=root, output_dir=root / "new", lease_root=root / "lease",
                provider_env_file=root / "provider.env", session_id="target", cell_variant_id="fixture",
                machine_id="fixture", gpu0_uuid="gpu0", gpu1_uuid="gpu1",
                max_allocated_active_gpu_hours=1, max_simulated_control_steps=1000, max_output_bytes=1000,
                source_reference_training_genesis_dir=root / "source",
                source_reference_training_session_id="source",
            )
            for changes in ({"source_reference_training_session_id": None},
                            {"source_reference_training_genesis_dir": config.output_dir},
                            {"source_reference_genesis_dir": root / "other", "source_reference_session_id": "other"}):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    replace(config, **changes)
            composition = SimpleNamespace(
                genesis_identity_snapshot=lambda: {"fixture": "host"},
                task_plugin=SimpleNamespace(bind_genesis_evidence=lambda value: None),
                backend_adapter=None, policy_initialization=None, baseline_manifest=None,
                reference_program_artifact=None, bootstrap_initial_best_score=None,
                bootstrap_initial_policy_state_hash=None,
            )
            foundation = SimpleNamespace(config=config, composition=composition, runner=None,
                session_contract=SimpleNamespace(contract_hash="a" * 64), session_contract_ref=None,
                cost_contract=SimpleNamespace(contract_hash="b" * 64, retry_train_jobs=1),
                gpu_bindings=(GpuProcessBinding(host_execution_device="cuda:0", process_execution_device="cuda:0",
                                              resource_binding_hash="c" * 64),), phased_adapter=None)
            with patch.object(generation_runtime, "prepare_session_genesis") as prepare:
                generation_runtime.prepare_generation_genesis(foundation)
            passed = prepare.call_args.kwargs
            self.assertEqual(passed["source_reference_training_genesis_dir"], root / "source")
            self.assertEqual(passed["source_reference_training_session_id"], "source")
            self.assertIsNone(passed["reference_train_attempt_scope"])
            self.assertEqual(set(passed["reference_eval_phase_attempt_scopes"]), set(EvaluationPhase))

    def fixture(self, root):
        old = root / "original"
        old.mkdir()
        task = fixtures._ReferenceTaskPlugin()
        _, reference = fixtures._reference_program_for_task(old, task)
        jobs = _EnvironmentBackend("a" * 64).materialize_reference_jobs(
            session_id="completed-reference", task_spec=task.task_spec, reference=reference,
            artifact_root=old / "reference_program_artifacts", policy_initialization=PolicyInitialization.random(),
            backend_execution_environment_manifest_hash="a" * 64,
        )
        result = fixtures._ReferenceRunner().run_train(jobs.train_spec)
        runner = _CostPhasedReferenceRunner()
        kwargs = {
            "output_dir": root / "continuation", "session_id": "completed-reference",
            "task_plugin": task, "backend_adapter": _EnvironmentBackend("b" * 64), "runner": runner,
            "policy_initialization": PolicyInitialization.random(), "reference_program_artifact": reference,
            "genesis_identity_snapshot": {
                "schema_version": "true_rl_agent.runtime_composition_snapshot.v3",
                "scientific": {"fixture": "unchanged-science", "budget_steps": jobs.train_spec.budget_steps},
                "execution_placement": {"project_root": str(root), "execution_device": "cuda:0"},
            },
            "reference_phased_evaluation_adapter": fixtures._ReferencePhasedAdapter(),
            "bind_backend_execution_environment": True,
        }
        return old, jobs.train_spec, result, runner, kwargs

    def test_preserves_original_train_and_uses_new_eval_environment_without_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            old, spec, result, runner, kwargs = self.fixture(Path(directory))
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in old.rglob("*") if path.is_file()}
            with patch.object(runner, "run_train", side_effect=AssertionError("retrained")):
                created = prepare_session_genesis(**kwargs, completed_reference_training=(spec, result))
                loaded = prepare_session_genesis(**kwargs, require_existing=True)
            self.assertEqual(loaded.session_genesis_hash, created.session_genesis_hash)
            self.assertEqual(runner.phase_calls, list(EvaluationPhase))
            self.assertEqual(runner.eval_calls, 0)
            self.assertEqual(runner.train_calls, 0)
            private = json.loads((kwargs["output_dir"] / "reference_eval_artifact.json").read_text())
            self.assertEqual(private["train_job"]["spec"], spec.to_public_dict())
            self.assertEqual(private["train_job"]["metadata"], dict(spec.metadata))
            self.assertEqual(private["train_result"]["summary"], result.summary)
            self.assertEqual(private["reference_artifact"]["path"], spec.reference_artifact.path)
            self.assertEqual(private["eval_job"]["metadata"]["backend_execution_environment_manifest_hash"], "b" * 64)
            self.assertEqual(loaded.artifact.backend_execution_environment, {"manifest_hash": "b" * 64})
            for path, (content, mtime) in before.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_changed_training_identity_or_unqualified_result_fails_before_eval(self):
        for case in ("identity", "completion", "checkpoint"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                _, spec, result, runner, kwargs = self.fixture(Path(directory))
                if case == "identity":
                    spec = replace(spec, job_id="different-session-train")
                    message = "does not match"
                elif case == "completion":
                    result = replace(result, summary={**result.summary, "formal_training_executed": False})
                    message = "did not execute formal training"
                else:
                    Path(result.checkpoint.metadata["local_path"]).write_bytes(b"changed")
                    message = "checkpoint bytes changed"
                with patch.object(runner, "run_train", side_effect=AssertionError("retrained")), \
                     self.assertRaisesRegex(ValueError, message):
                    prepare_session_genesis(**kwargs, completed_reference_training=(spec, result))
                self.assertEqual(runner.phase_calls, [])
                self.assertFalse((kwargs["output_dir"] / "session_genesis.json").exists())

    def source_fixture(self, root):
        _, _, _, runner, kwargs = self.fixture(root)
        kwargs.update(output_dir=root / "qualified-source", session_id="source-session")
        kwargs["backend_adapter"] = _EnvironmentBackend("a" * 64)
        self.bind_generation(kwargs, "d" * 64, training=True)
        with patch.object(runner, "run_train", wraps=runner.run_train) as trained:
            source = prepare_session_genesis(**kwargs)
            spec = trained.call_args.args[0]
        private = json.loads((kwargs["output_dir"] / "reference_eval_artifact.json").read_text())
        # Recover uses the original, retained result; no new training is allowed.
        from true_rl_agent.agent_core.session_genesis import _reference_train_result_from_private_artifact
        result = _reference_train_result_from_private_artifact(private, anchor=source.artifact.anchor, train_spec=spec)
        target = {**kwargs, "output_dir": root / "fresh-session", "session_id": "target-session",
                  "backend_adapter": _EnvironmentBackend("b" * 64),
                  "source_reference_training_genesis_dir": kwargs["output_dir"],
                  "source_reference_training_session_id": "source-session"}
        self.bind_generation(target, "e" * 64, training=False)
        return source, spec, result, runner, target

    def bind_generation(self, kwargs, resource_hash, *, training):
        contract = GenerationSessionContract(
            schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
            session_mode="development_generation_pilot", session_id=kwargs["session_id"],
            task_id=kwargs["task_plugin"].task_spec.task_id,
            backend_id=kwargs["backend_adapter"].backend_id,
            max_generations=1, max_trials=2, max_train_jobs=2,
            bound_contract_hashes={name: "a" * 64 for name in (
                "authoring_policy", "continuation_policy", "evaluation_cost", "evaluation_protocol",
                "exploration_projector", "final_confirmation_plan", "provider_policy", "resource_plan",
                "retry_policy", "selector", "session_cost")},
        )
        kwargs["generation_session_contract_ref"] = publish_generation_session_contract(kwargs["output_dir"], contract)
        kwargs["gpu_process_binding"] = GpuProcessBinding(
            host_execution_device="cuda:1", process_execution_device="cuda:0", resource_binding_hash=resource_hash)
        common = dict(journal_root=kwargs["output_dir"] / "job_attempts", session_id=kwargs["session_id"],
            genesis_identity_hash=session_genesis_identity_hash(kwargs["genesis_identity_snapshot"]),
            session_contract_hash=contract.contract_hash, lane_id="lane-a", budget_contract_hash="a" * 64,
            resource_binding_hash=resource_hash, attempt_budget_units=1, max_infrastructure_retries=0)
        kwargs["reference_train_attempt_scope"] = (
            PreGenesisExecutionScope(**common, invocation_segment_id="train") if training else None)
        kwargs["reference_eval_phase_attempt_scopes"] = {
            phase: PreGenesisExecutionScope(**common, invocation_segment_id=phase.value) for phase in EvaluationPhase}

    def test_new_session_reuses_original_training_and_reevaluates_with_new_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            source, spec, result, runner, kwargs = self.source_fixture(Path(directory))
            source_dir = kwargs["source_reference_training_genesis_dir"]
            before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in source_dir.rglob("*") if path.is_file()}
            runner.phase_calls.clear()
            with patch.object(runner, "recover_completed_train", create=True, return_value=result) as recover, \
                 patch.object(runner, "run_train", side_effect=AssertionError("retrained")):
                created = prepare_session_genesis(**kwargs)
                loaded = prepare_session_genesis(**kwargs, require_existing=True)
            self.assertEqual(recover.call_count, 2)
            self.assertEqual(recover.call_args.args[0].job_id, spec.job_id)
            self.assertEqual(runner.phase_calls, list(EvaluationPhase))
            self.assertEqual(created.session_genesis_hash, loaded.session_genesis_hash)
            self.assertNotEqual(created.session_genesis_hash, source.session_genesis_hash)
            private = json.loads((kwargs["output_dir"] / "reference_eval_artifact.json").read_text())
            original = json.loads((source_dir / "reference_eval_artifact.json").read_text())
            self.assertEqual(private["train_job"], original["train_job"])
            self.assertEqual(private["train_result"], original["train_result"])
            self.assertEqual(private["eval_job"]["spec"]["job_id"], "target-session-eval")
            self.assertEqual(private["eval_job"]["spec"]["gpu_process_binding"]["resource_binding_hash"], "e" * 64)
            provenance = created.artifact.runtime_snapshot["execution_placement"]["reference_training_source"]
            self.assertEqual(provenance["source_session_genesis_hash"], source.session_genesis_hash)
            self.assertFalse((kwargs["output_dir"] / "ledger.jsonl").exists())
            self.assertFalse((kwargs["output_dir"] / "runner" / "target-session" / "train").exists())
            from true_rl_agent.agent_core.generation_job_evidence import validated_job_cost_artifacts
            foundation = SimpleNamespace(config=SimpleNamespace(output_dir=kwargs["output_dir"]),
                cost_contract=SimpleNamespace(scientific_cost_basis=SimpleNamespace(train_steps_per_job=spec.budget_steps)))
            train_cost, eval_cost, _ = validated_job_cost_artifacts(foundation, genesis=created)
            self.assertEqual(train_cost, {})
            self.assertEqual(len(eval_cost), 3)
            self.assertTrue(all(job_id.startswith("target-session-") for job_id in eval_cost))
            for path, (content, mtime) in before.items():
                self.assertEqual(path.read_bytes(), content)
                self.assertEqual(path.stat().st_mtime_ns, mtime)
            no_source = {k: v for k, v in kwargs.items() if not k.startswith("source_reference_training_")}
            with self.assertRaisesRegex(ValueError, "training source drift"):
                prepare_session_genesis(**no_source, require_existing=True)

    def test_reuse_rejects_changed_science_or_unmatched_recovered_evidence(self):
        for case in ("science", "summary", "checkpoint"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                _, _, result, runner, kwargs = self.source_fixture(Path(directory))
                if case == "science":
                    snapshot = json.loads(json.dumps(kwargs["genesis_identity_snapshot"]))
                    snapshot["scientific"]["budget_steps"] += 1
                    kwargs["genesis_identity_snapshot"] = snapshot
                    expected = "scientific identity mismatch"
                elif case == "summary":
                    result = replace(result, summary={**result.summary, "changed": True})
                    expected = "sealed source evidence"
                else:
                    Path(result.checkpoint.metadata["local_path"]).write_bytes(b"changed")
                    expected = "checkpoint bytes changed"
                runner.phase_calls.clear()
                with patch.object(runner, "recover_completed_train", create=True, return_value=result), \
                     patch.object(runner, "run_train", side_effect=AssertionError("retrained")), \
                     self.assertRaisesRegex(ValueError, expected):
                    prepare_session_genesis(**kwargs)
                self.assertEqual(runner.phase_calls, [])

    def test_original_partial_directory_remains_forbidden(self):
        with tempfile.TemporaryDirectory() as directory:
            old, spec, result, runner, kwargs = self.fixture(Path(directory))
            kwargs["output_dir"] = old
            with self.assertRaisesRegex(ValueError, "runner state but no Genesis"):
                prepare_session_genesis(**kwargs, completed_reference_training=(spec, result))
            self.assertEqual(runner.phase_calls, [])
            self.assertEqual(runner.train_calls, 0)


if __name__ == "__main__":
    unittest.main()
