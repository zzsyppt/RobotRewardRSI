"""Historical launch recovery followed by current Q and a paused two-generation session."""

import json
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.fakes.backend_environment import (
    use_test_isaac_lab_execution_environment, TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST,
)
from tests.fakes.generation_runner import (
    _FixtureProvider, _isaac_assets_source_available, GPU_UUIDS, _write_candidate, _provider_usage,
)
from tests.integration.test_generation_session import (
    build_fixture_foundation, SharedStairRunner, SharedPushRunner, PushCandidateProvider,
)
from true_rl_agent.agent_core import generation_runtime
from true_rl_agent.agent_core.generation_application import run_generation_session
from true_rl_agent.agent_core.generation_job_evidence import validated_job_cost_artifacts
from true_rl_agent.agent_core.generation_operations import OperationAdmission
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.authoring_response import CandidateDraftResponse
from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateChangeSet
from true_rl_agent.runner.checkpoint import file_sha256
from true_rl_agent.runner.job_attempts import PreGenesisExecutionScope
from true_rl_agent.runner.subprocess_runner import SubprocessRunner
from true_rl_agent.agent_core.trial_receipts import _train_result_from_private_mapping
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider


class TwoGenerationPushProvider(PushCandidateProvider):
    def next_authoring_response(self, *, prompt_payload):
        # This scenario keeps the reference parent after G1. Exercise two distinct
        # complete programs under Push's replacement-authoring contract.
        candidate_id = f"fixture-g{self.generation_index}-{self.lane_id}"
        candidate_dir = self.output_dir / candidate_id
        expression = "inputs.base_height" if self.lane_id == "lane-a" else "inputs.roll * inputs.roll"
        if self.generation_index == 2:
            expression = "inputs.base_height.square()" if self.lane_id == "lane-a" else "inputs.pitch * inputs.pitch"
        name = f"fixture_g{self.generation_index}_{self.lane_id.replace('-', '_')}"
        source = ("from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
            f"def {name}(inputs):\n    return {expression} * {self.generation_index}\n\n"
            f"def build_training_components():\n    return TrainingComponents(name='{name}', "
            f"reward_terms=[RewardTerm(name='{name}', weight=0.1, fn={name})])\n")
        _write_candidate(candidate_dir, candidate_id=candidate_id, source=source)
        return CandidateDraftResponse(candidate=CandidatePackageRef(candidate_id=candidate_id,
            path=str(candidate_dir), change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={}), provider_usage=_provider_usage())


class PromptParentStairProvider(_FixtureProvider):
    def next_authoring_response(self, *, prompt_payload):
        current = prompt_payload["candidate_lineage"]["current_best"]
        self.reference_source = next(item["content"] for item in prompt_payload["source_payloads"]
                                     if item["logical_id"] == current["source_payload_ref"])
        return super().next_authoring_response(prompt_payload=prompt_payload)


class CompletedTrainingSessionTests(unittest.TestCase):
    def test_push_historical_train_current_q_and_two_generation_pause_resume(self):
        self._session(PushRecoveryCompositionProvider(), "push.g1_29dof.reference_v1",
                      SharedPushRunner, TwoGenerationPushProvider)

    @unittest.skipUnless(_isaac_assets_source_available(), "Stair source closure requires target Isaac installation")
    def test_stair_go2_historical_train_current_q_and_two_generation_pause_resume(self):
        self._session(StairClimbingCompositionProvider(), "stair.go2_12dof.unified_v3",
                      SharedStairRunner, PromptParentStairProvider)

    def _session(self, provider, cell, runner_type, candidate_provider):
        with tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
            root = Path(directory).resolve()
            original = build_fixture_foundation(root / "historical", provider=provider, cell=cell, max_generations=2)
            source_runner = runner_type(foundation=original)
            composition = original.composition
            environment = TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_MANIFEST
            old_gpu = replace(original.gpu_bindings[0], resource_binding_hash="e" * 64)
            binding = composition.backend_adapter.resolve_session_evaluation_binding(
                session_id=original.config.session_id, session_dir=original.config.output_dir,
                task_spec=composition.task_plugin.task_spec, allow_create=True)
            jobs = composition.backend_adapter.materialize_reference_jobs(
                session_id=original.config.session_id, task_spec=composition.task_plugin.task_spec,
                reference=composition.reference_program_artifact,
                artifact_root=original.config.output_dir / "reference_program_artifacts",
                policy_initialization=composition.policy_initialization, evaluation_binding=binding,
                backend_execution_environment_manifest_hash=environment["manifest_hash"], gpu_process_binding=old_gpu)
            source_result = source_runner._run_train_once(jobs.train_spec,
                context=SimpleNamespace(gpu_process_binding=old_gpu, physical_id=GPU_UUIDS[0]))
            launch_path = original.config.output_dir / "launch.json"
            launch_path.write_text(json.dumps({"status": "PASS", "returncode": 0,
                "argv": jobs.train_spec.metadata["cli_argv"], "job_spec": jobs.train_spec.to_public_dict(),
                "execution_environment": environment}), encoding="utf-8")

            def ref(path):
                return {"path": str(Path(path).resolve()), "sha256": file_sha256(Path(path))}

            descriptor = root / "source.json"
            descriptor.write_text(json.dumps({
                "schema_version": "true_rl_agent.completed_reference_training_source.v1",
                "source_cwd": str(original.config.project_root), "launch": ref(launch_path),
                "summary": ref(jobs.train_spec.metadata["summary_path"]),
                "checkpoint": ref(source_result.checkpoint.metadata["local_path"])}), encoding="utf-8")
            source_before = {p: p.read_bytes() for p in original.config.output_dir.rglob("*") if p.is_file()}

            def recover(_runner, spec):
                self.assertEqual(spec.to_public_dict(), jobs.train_spec.to_public_dict())
                return _train_result_from_private_mapping(source_runner._load_output("train", spec.job_id))

            admitted = []
            real_admission = OperationAdmission.__call__

            def admission(hook, scope, logical_job_id, job_kind, attempt_index):
                timeout = real_admission(hook, scope, logical_job_id, job_kind, attempt_index)
                if isinstance(scope, PreGenesisExecutionScope):
                    admitted.append((scope, {"logical_job_id": logical_job_id,
                                            "job_kind": job_kind, "attempt_index": attempt_index}))
                return timeout

            with patch.object(SubprocessRunner, "recover_completed_train", recover), \
                    patch.object(OperationAdmission, "__call__", admission):
                foundation = build_fixture_foundation(root / "current", provider=provider, cell=cell,
                    max_generations=2, pause=1, completed_training_source=descriptor)
                imported = foundation.completed_reference_training
                self.assertEqual(imported.train_spec.to_public_dict(), jobs.train_spec.to_public_dict())
                self.assertEqual(imported.train_result, source_result)
                with self.assertRaisesRegex(ValueError, "tuple differs"):
                    replace(imported, train_result=replace(imported.train_result,
                        summary={**imported.train_result.summary, "private_source_marker": "changed"})).validate()
                self.assertNotEqual(imported.train_spec.gpu_process_binding, foundation.gpu_bindings[0])
                self.assertFalse((foundation.config.output_dir / "job_attempts").exists())
                self.assertFalse((foundation.config.output_dir / "session_genesis.json").exists())
                current_runner = runner_type(foundation=foundation)
                current_runner.scenario = "accepted" if cell.startswith("stair.") else "no_selection"
                train = current_runner.run_train

                def candidate_train(spec):
                    self.assertEqual(spec.subject.kind, "candidate", "historical training must never execute")
                    return train(spec)

                current_runner.run_train = candidate_train
                foundation = replace(foundation, runner=current_runner)
                source = composition.reference_program_artifact.source_bytes.decode("utf-8")
                with patch.object(candidate_provider, "reference_source", source), patch.object(
                        generation_runtime, "OpenAICompatibleCandidateProvider", candidate_provider):
                    paused = run_generation_session(foundation)
                    self.assertEqual(paused["status"], "paused")
                    self.assertEqual(paused["closed_generation_count"], 1)
                    ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
                    self.assertEqual(ledger[0]["generation_outcome"], current_runner.scenario)
                    genesis_before = (foundation.config.output_dir / "session_genesis.json").read_bytes()
                    attempts_before = {p: p.read_bytes() for p in
                        (foundation.config.output_dir / "job_attempts").glob("*/attempt_*.reservation.json")}
                    resumed = build_fixture_foundation(root / "current", provider=provider, cell=cell,
                        max_generations=2, resume=True, completed_training_source=descriptor)
                    resumed = replace(resumed, runner=current_runner)
                    report = run_generation_session(resumed)
                self.assertEqual(report["checks"]["status"], "PASS")
                self.assertEqual(report["closed_generation_count"], 2)
                self.assertEqual(report["cost"]["actual"]["train_jobs"], 4)
                self.assertEqual(genesis_before, (foundation.config.output_dir / "session_genesis.json").read_bytes())
                self.assertTrue(all(p.read_bytes() == value for p, value in attempts_before.items()))
                self.assertEqual(len(admitted), 3)
                for scope, args in admitted:
                    self.assertEqual(args["job_kind"], "eval_phase")
                    self.assertEqual(args["attempt_index"], 0)
                    self.assertEqual(scope.journal_root, foundation.config.output_dir / "job_attempts")
                    self.assertEqual(scope.session_contract_hash, foundation.session_contract.contract_hash)
                    self.assertEqual(scope.budget_contract_hash, foundation.cost_contract.contract_hash)
                    self.assertEqual(scope.resource_binding_hash, foundation.gpu_bindings[0].resource_binding_hash)
                genesis = generation_runtime.prepare_generation_genesis(resumed, require_existing=True)
                train_steps, evals, _ = validated_job_cost_artifacts(resumed, genesis=genesis)
                self.assertNotIn(jobs.train_spec.job_id, train_steps)
                q_ids = {foundation.config.session_id + "-eval-" + phase
                         for phase in ("public-dev", "public-accept", "sealed-validation")}
                self.assertEqual({args["logical_job_id"] for _, args in admitted}, q_ids)
                self.assertEqual(set(train_steps), {
                    f"{foundation.config.session_id}-generation-{index:03d}-{lane}-train"
                    for index in (1, 2) for lane in ("lane-a", "lane-b")})
                self.assertTrue(q_ids <= set(evals))
                reservations = [json.loads(p.read_text(encoding="utf-8")) for p in
                    (foundation.config.output_dir / "job_attempts").glob("*/attempt_*.reservation.json")]
                self.assertFalse(any(p["logical_job_id"] == jobs.train_spec.job_id for p in reservations))
                audit = replace(resumed, config=replace(resumed.config, terminal_audit_only=True))
                self.assertEqual(run_generation_session(audit), report)
            self.assertTrue(all(p.read_bytes() == value for p, value in source_before.items()))


if __name__ == "__main__":
    unittest.main()
