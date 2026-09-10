"""Real composition/lifecycle/receipts with a fixed, attempt-backed runner."""

from __future__ import annotations

import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
from tests.fakes.generation_runner import (
    _AttemptBackedStairRunner, _FormalFixtureProvider, _write_fake_nvidia_smi,
    _nvidia_smi_name, FORMAL_SESSION_ID, GPU_UUIDS, _isaac_assets_source_available,
    _gpu_summary_fields, _checkpoint_report,
    _write_candidate, _provider_usage,
)
from tests.unit.test_generation_cost_contract import fixture_cost
from true_rl_agent.agent_core import generation_runtime, generation_application
from true_rl_agent.agent_core.generation_application import run_generation_session
from true_rl_agent.agent_core.generation_preparation import GenerationLaunchConfig, prepare_generation_foundation
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.generation_finalization import bind_stair_final_config
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.phased_evaluation import build_push_phase_summary
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.budget import build_budget_accounting
from true_rl_agent.runner.base import CandidateRuntimeError, RunnerExecutionError, RunnerFailureKind
from true_rl_agent.agent_core.authoring_response import CandidateDraftResponse
from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateChangeSet
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.generation_record import validate_generation_ledger_record
from tests.fakes.backend_environment import TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION

ROOT = Path(__file__).resolve().parents[2]


def _inject_train_failure(runner, spec):
    if spec.subject.kind != "candidate" or "lane-a" not in spec.job_id:
        return
    if runner.scenario not in {"runtime_failure", "infrastructure_failure"}:
        return
    runner._claim_actual("train", spec.job_id)
    if runner.scenario == "infrastructure_failure":
        raise RunnerExecutionError("fixture train timed out", failure_kind=RunnerFailureKind.TIMEOUT)
    contract = spec.candidate_failure_contract
    rule = next(rule for rule in contract.rules if rule.stage == runner.runtime_failure_stage)
    raise CandidateRuntimeError("fixture reward callable failed", public_details={
        "status": "fail", "subprocess_returncode": 1,
        "candidate_failure_contract": contract.to_public_dict(),
        "candidate_failure": contract.evidence(stage=rule.stage, reason_code=rule.reason_codes[0]).to_public_dict(),
    })


class SharedStairRunner(_AttemptBackedStairRunner):
    scenario = "accepted"
    runtime_failure_stage = "ppo_reward_execution"

    def _run_train_once(self, spec, *, context):
        _inject_train_failure(self, spec)
        return super()._run_train_once(spec, context=context)

    def _execution(self, *, config, eval_spec, split_success_thirds):
        if eval_spec.subject.kind == "candidate":
            split_success_thirds = {
                phase: ((3 if "lane-a" in eval_spec.job_id else 2)
                        if phase.value == "public_dev" and self.scenario != "no_selection" else
                        2 if self.scenario == "early_reject" else
                        0 if self.scenario == "no_selection" or
                        (self.scenario == "sealed_rejected" and phase.value == "sealed_validation") else 3)
                for phase in split_success_thirds
            }
        return super()._execution(config=config, eval_spec=eval_spec,
                                  split_success_thirds=split_success_thirds)

    def _evaluation_config(self, eval_spec):
        config = self.foundation.composition.task_plugin.config
        plan = self.foundation.final_confirmation_plan
        if eval_spec.evaluation_binding.seed_cohort_hash == plan.seed_cohort_hash:
            return bind_stair_final_config(config, plan=plan,
                plan_path=self.foundation.config.output_dir / "final_confirmation" / "plan.json")
        return config


class SharedPushRunner(_AttemptBackedStairRunner):
    scenario = "early_reject"
    runtime_failure_stage = "reward_execution"

    def _run_train_once(self, spec, *, context):
        _inject_train_failure(self, spec)
        return super()._run_train_once(spec, context=context)

    def _train_summary(self, spec, *, binding, physical_uuid):
        config = self.foundation.composition.task_plugin.config
        summary = {"schema_version": "true_rl_agent.fixture.push_train.v1",
            **spec.job_identity().to_public_dict(), "status": "pass", "errors": [],
            "execution_device": binding.process_execution_device,
            "formal_training_executed": True,
            "training": {"initial_policy_state_hash": spec.expected_initial_policy_state_hash or "a" * 64,
                "budget_accounting": build_budget_accounting(budget_steps=config.budget_steps,
                    num_envs=config.train_num_envs, num_steps_per_env=config.train_steps_per_iteration,
                    max_iterations=config.train_max_iterations)},
            **_gpu_summary_fields(spec.job_id, binding, physical_uuid)}
        if spec.candidate_artifact is not None:
            import torch
            from true_rl_agent.tasks.push_recovery.runtime_adapter import CandidateRuntimeAdapter

            runtime = CandidateRuntimeAdapter.from_candidate_dir(spec.candidate_artifact.path,
                num_envs=4, planned_training_steps=1, diagnostic_interval_steps=1)
            inputs = self.foundation.composition.task_plugin.component_smoke_adapter().build_component_smoke_fixture().reward_inputs
            inputs = replace(inputs, base_height=torch.tensor([0.34, 0.39, 0.42, 0.36]),
                roll=torch.tensor([0.02, -0.04, 0.08, -0.03]))
            # Execute the sealed callable through the real adapter on fixed inputs.
            with patch.object(runtime, "_build_inputs", return_value=inputs):
                runtime.apply_step_rewards(env=None, actions=None, rewards=torch.zeros(4))
            summary["runtime_activation_summary"] = runtime.runtime_summary()
        # Match the production subprocess JSON boundary (for example tensor shapes).
        return json.loads(json.dumps(summary))

    def _run_phase_once(self, spec, *, context):
        import copy

        binding, physical_uuid = self._binding(context)
        self._claim_actual("eval_phase", spec.job_id)
        config = self.foundation.composition.task_plugin.config
        seed = {"status": "pass", "aggregate": {"success_rate": 0.625,
            "episodes": config.eval_num_envs, "failure_breakdown": {"recovered": 0.625, "fall": 0.375}},
            "secondary_metrics": {"mean_action_l2": 0.1, "mean_action_rate_l2": 0.1,
                "mean_min_base_height_m": 0.7, "mean_max_abs_pitch_rad": 0.1,
                "mean_max_abs_roll_rad": 0.1, "mean_final_xy_drift_m": 0.1,
                "mean_final_planar_speed_mps": 0.1, "recovery_window_success_rate": 0.625},
            "execution_cost": {"evaluation_env_count": config.eval_num_envs,
                "vector_env_step_calls": 200, "closed_first_episode_control_steps": config.eval_num_envs * 200}}
        if spec.subject.kind == "candidate":
            if self.scenario in {"accepted", "sealed_rejected"}:
                rate = 0.875 if self.scenario == "accepted" or spec.phase.value != "sealed_validation" else 0.0
                seed["aggregate"].update(success_rate=rate, failure_breakdown={"recovered": rate, "fall": 1-rate})
                seed["secondary_metrics"]["recovery_window_success_rate"] = rate
            elif self.scenario == "no_selection":
                seed["secondary_metrics"]["mean_min_base_height_m"] = 0.1
        seed_count = self.foundation.phased_adapter.seed_count(spec.phase)
        summary = build_push_phase_summary(phase=spec.phase, evaluation_plan_hash=spec.evaluation_plan_hash,
            seed_summaries=[copy.deepcopy(seed) for _ in range(seed_count)],
            expected_seed_count=seed_count, wall_time_ns=10_000_000,
            identity=spec.job_identity().to_public_dict(), execution_device=spec.execution_device,
            checkpoint_safety_report=_checkpoint_report(spec.execution_checkpoint().checksum),
            runtime_dynamics_hash=canonical_json_hash(config.eval_runtime_dynamics()))
        summary["backend_execution_environment_verification"] = TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
        summary.update(_gpu_summary_fields(spec.job_id, binding, physical_uuid))
        self._publish_summary(spec.metadata, summary)
        result = self.delegate._phase_result_from_summary(summary, eval_spec=spec,
            expected_checkpoint_hash=spec.execution_checkpoint().checksum, actual_gpu_process_binding=binding)
        self._publish_output("eval_phase", spec.job_id, result.to_private_dict())
        return result


class PushCandidateProvider(_FormalFixtureProvider):
    def next_authoring_response(self, *, prompt_payload):
        if self.generation_index != 1:
            return super().next_authoring_response(prompt_payload=prompt_payload)
        candidate_id = f"fixture-g1-{self.lane_id}"
        candidate_dir = self.output_dir / candidate_id
        expression = "inputs.base_height" if self.lane_id == "lane-a" else "inputs.roll * inputs.roll"
        name = "fixture_base_height" if self.lane_id == "lane-a" else "fixture_roll_square"
        source = ("from true_rl_agent.candidates.components import RewardTerm, TrainingComponents\n\n"
            f"def {name}(inputs):\n    return {expression}\n\n"
            f"def build_training_components():\n    return TrainingComponents(name='{name}', "
            f"reward_terms=[RewardTerm(name='{name}', weight=0.1, fn={name})])\n")
        _write_candidate(candidate_dir, candidate_id=candidate_id, source=source)
        (candidate_dir / "rationale.md").write_text("Exercise one Push reward callable in the shared host.\n")
        return CandidateDraftResponse(candidate=CandidatePackageRef(candidate_id=candidate_id,
            path=str(candidate_dir), change_set=CandidateChangeSet(frozenset({"reward_callable"})),
            metadata={}), provider_usage=_provider_usage())


def build_fixture_foundation(root, *, provider, cell, resume=False, pause=None, audit=False):
    context = TaskCompositionContext(project_root=ROOT, output_dir=root / "session", read_only=resume)
    values = {**provider.cli_defaults(), "cell_variant_id": cell}
    composition = resolve_task_composition(provider=provider, values=values, context=context)
    final_plan = composition.task_plugin.prepare_final_confirmation(
        output_dir=context.output_dir, session_id=FORMAL_SESSION_ID,
        required_closed_generations=15, resume_existing=resume)
    values.update(mode="formal_generation_session", session_id=FORMAL_SESSION_ID,
        max_generations=15, final_confirmation_plan=context.output_dir / "final_confirmation" / "plan.json")
    composition = resolve_task_composition(provider=provider, values=values, context=context)
    work = composition.task_plugin.phased_evaluation_adapter().generation_workload(
        cell_manifest=composition.resolved_cell_manifest)
    cost = fixture_cost(work)
    cost = replace(cost, operation_reserves=tuple(replace(item,
        source_allocated_gpu_capacity_ns=50_000_000_000_000,
        source_output_bytes=100_000_000,
        allocated_gpu_capacity_ns_per_attempt=62_500_000_000_000,
        output_bytes_per_attempt=125_000_000) for item in cost.operation_reserves),
        finalization_output_reserve_bytes=10_000_000)
    executable = root / _nvidia_smi_name()
    if not executable.exists():
        _write_fake_nvidia_smi(executable)
    launch = GenerationLaunchConfig(project_root=ROOT, output_dir=context.output_dir,
        lease_root=root / "leases", provider_env_file=root / "provider.env",
        session_id=FORMAL_SESSION_ID, cell_variant_id=cell, machine_id="fixture",
        gpu0_uuid=GPU_UUIDS[0], gpu1_uuid=GPU_UUIDS[1],
        max_allocated_active_gpu_hours=cost.max_allocated_active_gpu_hours,
        max_simulated_control_steps=cost.max_simulated_control_steps, max_output_bytes=cost.max_output_bytes,
        nvidia_smi_executable=str(executable), resume_existing=resume,
        pause_after_generation=pause, terminal_audit_only=audit)
    return prepare_generation_foundation(launch, composition=composition, source_commit="a" * 40,
        inventory=GPU_UUIDS, cost=cost, final_plan=final_plan,
        provider_policy={"max_attempts_per_lane": 3, "max_output_tokens": 2000,
            "input_units_per_attempt_reserve": 10000, "api_key_env": "FIXTURE_KEY",
            "base_url": "https://fixture.invalid", "endpoint": "/chat/completions",
            "model": "fixture", "timeout_s": 30, "temperature": 0.6,
            "reasoning_mode": "adaptive", "reasoning_effort": "high",
            "empty_final_reasoning_retry": True})


class SharedGenerationSessionTests(unittest.TestCase):
    def test_push_runtime_and_infrastructure_failures_keep_distinct_evidence(self):
        self._failure_sessions(PushRecoveryCompositionProvider(), "push.g1_29dof.reference_v1",
                               SharedPushRunner, PushCandidateProvider)

    @unittest.skipUnless(_isaac_assets_source_available(), "Stair runtime source closure requires target Isaac installation")
    def test_stair_runtime_and_infrastructure_failures_keep_distinct_evidence(self):
        self._failure_sessions(StairClimbingCompositionProvider(), "stair.g1_37dof.unified_v3",
                               SharedStairRunner, _FormalFixtureProvider)

    def _failure_sessions(self, provider, cell, runner_type, candidate_provider):
        for scenario, reason in (("runtime_failure", "candidate_runtime_failed"),
                                 ("infrastructure_failure", "train_incomplete")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
                foundation = build_fixture_foundation(root, provider=provider, cell=cell, pause=3)
                runner = runner_type(foundation=foundation)
                runner.scenario = scenario
                foundation = replace(foundation, runner=runner)
                source = foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
                with patch.object(candidate_provider, "reference_source", source), patch.object(
                        generation_runtime, "OpenAICompatibleCandidateProvider", candidate_provider):
                    result = run_generation_session(foundation)
                self.assertEqual(result["status"], "paused")
                first = load_ledger(foundation.config.output_dir / "ledger.jsonl")[0]
                self.assertEqual(first["generation_outcome"], "candidate_execution_incomplete")
                self.assertEqual(first["trials"][0]["status_reason"], reason)
                self.assertIsNone(first["early_rejection"])
                self.assertIsNone(first["formal"])
                self.assertEqual(validate_generation_ledger_record(first).to_public_dict(), first["parent_current_best"])

    @unittest.skipUnless(_isaac_assets_source_available(), "Stair runtime source closure requires target Isaac installation")
    def test_stair_early_rejection_sealed_rejection_and_no_selection(self):
        cases = (("early_reject", "formal_rejected"),
                 ("sealed_rejected", "formal_rejected"), ("no_selection", "no_selection"))
        for scenario, outcome in cases:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
                foundation = build_fixture_foundation(root, provider=StairClimbingCompositionProvider(),
                    cell="stair.g1_37dof.unified_v3", pause=3)
                runner = SharedStairRunner(foundation=foundation)
                runner.scenario = scenario
                foundation = replace(foundation, runner=runner)
                source = foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
                with patch.object(_FormalFixtureProvider, "reference_source", source), patch.object(
                        generation_runtime, "OpenAICompatibleCandidateProvider", _FormalFixtureProvider):
                    result = run_generation_session(foundation)
                self.assertEqual(result["status"], "paused")
                first = load_ledger(foundation.config.output_dir / "ledger.jsonl")[0]
                self.assertEqual(first["generation_outcome"], outcome,
                                 [trial["status_reason"] for trial in first["trials"]])
                self.assertEqual(first["early_rejection"] is not None, scenario == "early_reject")
                self.assertEqual(first["formal"] is not None, scenario == "sealed_rejected")
                self.assertEqual(validate_generation_ledger_record(first).to_public_dict(), first["parent_current_best"])

    @unittest.skipUnless(_isaac_assets_source_available(), "Stair runtime source closure requires target Isaac installation")
    def test_stair_accepted_head_resumes_and_closes_final(self):
        self._accepted_head_session(StairClimbingCompositionProvider(), "stair.g1_37dof.unified_v3",
                                    SharedStairRunner, _FormalFixtureProvider)

    def test_push_keeps_sealed_dependent_rejection_and_no_selection_distinct(self):
        for scenario, outcome in (("sealed_rejected", "formal_rejected"), ("no_selection", "no_selection")):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
                root = Path(directory).resolve()
                (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
                foundation = build_fixture_foundation(root, provider=PushRecoveryCompositionProvider(),
                    cell="push.g1_29dof.reference_v1", pause=3)
                runner = SharedPushRunner(foundation=foundation)
                runner.scenario = scenario
                foundation = replace(foundation, runner=runner)
                with patch.object(generation_runtime, "OpenAICompatibleCandidateProvider", PushCandidateProvider):
                    result = run_generation_session(foundation)
                self.assertEqual(result["status"], "paused")
                first = load_ledger(foundation.config.output_dir / "ledger.jsonl")[0]
                self.assertEqual(first["generation_outcome"], outcome)
                self.assertIsNone(first["early_rejection"])
                self.assertEqual(first["formal"] is not None, scenario == "sealed_rejected")
                self.assertEqual(validate_generation_ledger_record(first).to_public_dict(), first["parent_current_best"])

    def test_push_accepted_head_resumes_and_closes_final(self):
        self._accepted_head_session(PushRecoveryCompositionProvider(), "push.g1_29dof.reference_v1",
                                    SharedPushRunner, PushCandidateProvider)

    def _accepted_head_session(self, provider, cell, runner_type, candidate_provider):
        with tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
            root = Path(directory).resolve()
            (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
            foundation = build_fixture_foundation(root, provider=provider, cell=cell, pause=3)
            runner = runner_type(foundation=foundation)
            runner.scenario = "accepted"
            foundation = replace(foundation, runner=runner)
            source = foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
            with patch.object(candidate_provider, "reference_source", source), patch.object(
                    generation_runtime, "OpenAICompatibleCandidateProvider", candidate_provider):
                paused = run_generation_session(foundation)
                self.assertEqual(paused["status"], "paused")
                first = load_ledger(foundation.config.output_dir / "ledger.jsonl")[0]
                self.assertEqual(first["generation_outcome"], "accepted",
                    [trial["status_reason"] for trial in first["trials"]])
                self.assertIsNone(first["early_rejection"])
                resumed = build_fixture_foundation(root, provider=provider, cell=cell, resume=True)
                resumed = replace(resumed, runner=runner)
                report = run_generation_session(resumed)
            self.assertEqual(report["checks"]["status"], "PASS")
            self.assertEqual(report["current_best"]["anchor_kind"], "candidate")
            self.assertEqual(report["closed_generation_count"], 15)
            self.assertEqual(report["generation_outcomes"], {"accepted": 1, "paired_authoring_incomplete": 14})
            audit = replace(resumed, config=replace(resumed.config, terminal_audit_only=True))
            self.assertEqual(run_generation_session(audit), report)

    def test_push_early_rejection_closes_without_sealed_or_a_full_result(self):
        with tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
            root = Path(directory).resolve()
            (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
            foundation = build_fixture_foundation(root, provider=PushRecoveryCompositionProvider(),
                cell="push.g1_29dof.reference_v1", pause=3)
            runner = SharedPushRunner(foundation=foundation)
            foundation = replace(foundation, runner=runner)
            with patch.object(generation_runtime, "OpenAICompatibleCandidateProvider", PushCandidateProvider):
                result = run_generation_session(foundation)
            self.assertEqual(result["status"], "paused")
            ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
            first = ledger[0]
            self.assertEqual(first["generation_outcome"], "formal_rejected")
            self.assertIsNone(first["formal"])
            self.assertEqual(first["early_rejection"]["sealed_validation"], "not_run_due_to_early_reject")
            self.assertFalse(any("sealed_validation_phase_receipt" in item["kind"] for item in first["private_artifacts"]))
            self.assertEqual(validate_generation_ledger_record(first).to_public_dict(), first["parent_current_best"])

    @unittest.skipUnless(_isaac_assets_source_available(), "Stair runtime source closure requires target Isaac installation")
    def test_stair_reference_head_uses_shared_preparation_session_final_and_replay(self):
        self._reference_head_session(StairClimbingCompositionProvider(), "stair.g1_37dof.unified_v3", SharedStairRunner)

    def test_push_reference_head_uses_shared_preparation_session_final_and_replay(self):
        self._reference_head_session(PushRecoveryCompositionProvider(), "push.g1_29dof.reference_v1", SharedPushRunner)

    def _reference_head_session(self, provider, cell, runner_type):
        with tempfile.TemporaryDirectory() as directory, use_test_isaac_lab_execution_environment():
            root = Path(directory).resolve()
            (root / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
            foundation = build_fixture_foundation(root, provider=provider, cell=cell)
            runner = runner_type(foundation=foundation)
            foundation = replace(foundation, runner=runner)
            with patch.object(_FormalFixtureProvider, "reference_only", True), patch.object(
                    generation_runtime, "OpenAICompatibleCandidateProvider", _FormalFixtureProvider), patch.object(
                    generation_application, "build_generation_terminal_binding",
                    wraps=generation_application.build_generation_terminal_binding) as terminal_replay:
                report = run_generation_session(foundation)
            terminal_replay.assert_called_once()
            self.assertEqual(report["checks"]["status"], "PASS")
            self.assertEqual(report["closed_generation_count"], 15)
            self.assertEqual(report["current_best"]["anchor_kind"], "reference")
            ledger_path = foundation.config.output_dir / "ledger.jsonl"
            before = ledger_path.read_bytes()
            audit = replace(foundation, config=replace(foundation.config, resume_existing=True,
                                                       terminal_audit_only=True))
            with patch.object(generation_application, "build_generation_terminal_binding",
                              wraps=generation_application.build_generation_terminal_binding) as terminal_replay:
                self.assertEqual(run_generation_session(audit), report)
            terminal_replay.assert_called_once()
            self.assertEqual(ledger_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
