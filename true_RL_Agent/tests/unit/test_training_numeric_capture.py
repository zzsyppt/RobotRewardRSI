from __future__ import annotations

from contextlib import ExitStack
from functools import reduce
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents
from true_rl_agent.candidates.mechanism import MechanismEvidenceGate, RuntimeActivationEvidence
from true_rl_agent.candidates.schema import CandidateSurfaceAssessment
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.runner.training_numeric_capture import (
    TrainingNumericCapture, TrainingNumericAttachment, finish_training_numeric_capture,
)
from true_rl_agent.tasks.push_recovery.runtime_adapter import CandidateRuntimeAdapter
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.reference_program import resolve_official_stair_reference_program
from true_rl_agent.tasks.stair_climbing.reward_inputs import StairRewardInputsV2
from true_rl_agent.tasks.stair_climbing.reward_program import StairCompleteRewardDispatcher

ROOT = Path(__file__).resolve().parents[2]


class TrainingNumericCaptureTests(unittest.TestCase):
    def make_capture(self, path, **overrides):
        return TrainingNumericCapture(path, **{
            "identity": {"source": "test"}, "semantics": {}, "planned_control_steps": 24,
            "num_envs": 2, "steps_per_iteration": 8, "source_step_offset": 0, **overrides,
        })

    def assert_runtime_unchanged(self, baseline, observed):
        self.assertEqual(baseline, observed)
        gate = MechanismEvidenceGate()
        surface = CandidateSurfaceAssessment("code_level", True, "fixture")
        self.assertEqual(gate.evaluate(surface=surface, evidence=RuntimeActivationEvidence.from_mapping(baseline)),
                         gate.evaluate(surface=surface, evidence=RuntimeActivationEvidence.from_mapping(observed)))

    def test_fixed_windows_offsets_subset_and_original_dtypes_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.make_capture(Path(directory) / "archive", planned_control_steps=96,
                                        num_envs=20, source_step_offset=7, initial_learning_iteration=5,
                                        reported_iteration=lambda: 5)
            expected_steps = list(range(2, 95, 4))
            expected_steps[0] = 1
            for step in range(1, 97):
                raw = np.arange(20, dtype=np.float32) + step
                values = {"raw": raw, "weighted": raw.astype(np.float64) * 0.3}
                capture.observe(step + 7, values if capture.wants_step(step + 7) else None)
                raw[:] = -1000  # Saved chunks must not retain aliases to live values.
            result = capture.finish(training_status="pass")
            self.assertEqual(capture.finish(training_status="pass"), result)
            coverage = load_numeric_evidence_manifest(result["reference"])["metadata"]
            self.assertEqual([w["status"] for w in coverage["windows"]], ["sampled"] * 3)
            self.assertEqual([w["sampled_env_transitions"] for w in coverage["windows"]], [128] * 3)
            self.assertEqual(coverage["source_step_offset_excluded"], 7)
            chunks = list(iter_numeric_evidence(result["reference"]))
            samples = [(meta, arrays) for meta, arrays in chunks if meta["kind"] == "training_sample"]
            self.assertEqual([m["training_control_step"] for m, _ in samples], expected_steps)
            for meta, arrays in samples:
                step = meta["training_control_step"]
                self.assertEqual(meta["source_control_step"], step + 7)
                self.assertEqual(meta["learning_iteration_from_control_steps"], 5 + (step - 1) // 8)
                self.assertEqual(meta["step_in_rollout"], (step - 1) % 8 + 1)
                self.assertEqual(arrays["raw"].dtype, np.dtype("float32"))
                self.assertEqual(arrays["weighted"].dtype, np.dtype("float64"))
                np.testing.assert_array_equal(arrays["raw"], np.arange(16, dtype=np.float32) + step)
                np.testing.assert_array_equal(arrays["env_id"], np.arange(16))

    def test_partial_not_reached_and_reached_but_not_sampled_are_distinct(self):
        for include_arrays in (False, True):
            with self.subTest(include_arrays=include_arrays), tempfile.TemporaryDirectory() as directory:
                capture = self.make_capture(Path(directory) / "archive")
                for step in range(1, 5):
                    capture.observe(step, {"value": np.zeros(2)} if include_arrays else None)
                result = capture.finish(training_status="fail")
                self.assertEqual(result["status"], "complete")  # Archive completeness, not training success.
                coverage = result["coverage"]
                self.assertEqual(coverage["training_status"], "fail")
                self.assertEqual([w["status"] for w in coverage["windows"]],
                                 ["partial" if include_arrays else "not_sampled", "not_reached", "not_reached"])
                self.assertEqual(sum(w["sampled_control_steps"] for w in coverage["windows"]), 4 if include_arrays else 0)
                self.assertTrue(list(iter_numeric_evidence(result["reference"])))

    def test_long_plan_spreads_fixed_budget_across_all_training_thirds(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.make_capture(Path(directory) / "archive", planned_control_steps=48000,
                                        num_envs=4096, steps_per_iteration=24, source_step_offset=7)
            planned = [list(range(1000 + 16000 * part, 16000 * (part + 1), 2000)) for part in range(3)]
            planned[0][0] = 1
            self.assertEqual([w["planned_steps"] for w in capture.windows], planned)
            values = {"value": np.arange(4096, dtype=np.float32)}
            for step in range(1, 48001):
                capture.observe(step + 7, values if capture.wants_step(step + 7) else None)
            result = capture.finish(training_status="pass")
            coverage = result["coverage"]
            self.assertEqual(coverage["schema_version"], "true_rl_agent.training_numeric_capture.v2")
            self.assertEqual([(w["start_step"], w["end_step"]) for w in coverage["windows"]],
                             [(1, 16000), (16001, 32000), (32001, 48000)])
            self.assertEqual([w["sampled_steps"] for w in coverage["windows"]], planned)
            self.assertEqual(sum(w["sampled_env_transitions"] for w in coverage["windows"]), 384)
            self.assertEqual([steps[-1] - steps[0] for steps in planned], [14999, 14000, 14000])
            self.assertTrue(all(len({(step - 1) // 24 for step in steps}) == 8 for steps in planned))

    def test_long_plan_not_reached_uses_first_scheduled_point_and_partial_keeps_indices(self):
        for stop, expected in ((0, ["not_reached"] * 3),
                               (100, ["partial", "not_reached", "not_reached"]),
                               (999, ["partial", "not_reached", "not_reached"]),
                               (1000, ["partial", "not_reached", "not_reached"]),
                               (17000, ["sampled", "partial", "not_reached"])):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as directory:
                capture = self.make_capture(Path(directory) / "archive", planned_control_steps=48000)
                for step in range(1, stop + 1):
                    capture.observe(step, {"value": np.ones(2)} if capture.wants_step(step) else None)
                evidence = capture.finish(training_status="fail")
                coverage = evidence["coverage"]
                self.assertEqual([w["status"] for w in coverage["windows"]], expected)
                for window in coverage["windows"]:
                    self.assertEqual(window["sampled_steps"], [s for s in window["planned_steps"] if s <= stop])
                if stop == 100:
                    samples = [(meta, arrays) for meta, arrays in iter_numeric_evidence(evidence["reference"])
                               if meta["kind"] == "training_sample"]
                    self.assertEqual(len(samples), 1)
                    self.assertEqual(samples[0][0]["training_control_step"], 1)
                    np.testing.assert_array_equal(samples[0][1]["value"], np.ones(2))
                    self.assertEqual(coverage["observed_control_steps"], 100)
                    self.assertEqual(coverage["windows"][0]["sampled_env_transitions"], 2)

    def test_short_run_and_disabled_or_not_started_do_not_invent_samples(self):
        self.assertEqual(finish_training_numeric_capture(None, training_status="pass"), {"status": "not_enabled"})
        self.assertEqual(finish_training_numeric_capture(None, training_status="fail", requested=True), {"status": "not_started"})
        with tempfile.TemporaryDirectory() as directory:
            capture = self.make_capture(Path(directory) / "archive", planned_control_steps=1)
            capture.observe(1, {"value": np.ones(2)})
            result = capture.finish(training_status="pass")
            self.assertEqual([w["status"] for w in result["coverage"]["windows"]],
                             ["not_scheduled", "not_scheduled", "sampled"])

    def test_preflight_and_duplicate_steps_cannot_enter_training_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = self.make_capture(Path(directory) / "archive", source_step_offset=1)
            with self.assertRaisesRegex(ValueError, "nonconsecutive"):
                capture.observe(1, {"value": np.ones(2)})
            capture.observe(2, {"value": np.ones(2)})
            with self.assertRaisesRegex(ValueError, "nonconsecutive"):
                capture.observe(2, {"value": np.ones(2)})

    def test_push_hook_keeps_rewards_and_records_actual_clipped_weighted_values(self):
        for physical in (False, True):
            with self.subTest(physical=physical), tempfile.TemporaryDirectory() as directory:
                components = TrainingComponents(name="sample-probe", reward_terms=[
                    RewardTerm(name="benefit", weight=2.0, clip_max=3.0,
                               fn=lambda _: torch.tensor([1.0, 5.0], dtype=torch.float32)),
                    RewardTerm(name="penalty", weight=-3.0,
                               fn=lambda _: torch.tensor([2.0, 4.0], dtype=torch.float32)),
                ])
                adapters = [CandidateRuntimeAdapter(manifest=None, components=components,
                            num_envs=2, planned_training_steps=25) for _ in range(2)]
                kwargs = {"env": SimpleNamespace(), "actions": torch.zeros((2, 1)),
                          "rewards": torch.ones(2)}
                for adapter in adapters:
                    if physical:
                        adapter.enable_physical_program()
                    adapter.apply_step_rewards(**kwargs)
                capture = TrainingNumericAttachment(lambda: self.make_capture(Path(directory) / "archive", source_step_offset=1,
                    semantics=adapters[1].training_numeric_semantics(control_dt=0.02)),
                    bind=adapters[1].bind_training_numeric_capture)
                rng_before = torch.get_rng_state().clone()
                for _ in range(24):
                    self.assertTrue(torch.equal(adapters[0].apply_step_rewards(**kwargs), adapters[1].apply_step_rewards(**kwargs)))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                self.assert_runtime_unchanged(adapters[0].runtime_summary(), adapters[1].runtime_summary())
                result = capture.finish(training_status="pass")
                for meta, arrays in iter_numeric_evidence(result["reference"]):
                    if meta["kind"] != "training_sample":
                        continue
                    np.testing.assert_array_equal(arrays["raw_term_000"], [1.0, 3.0])
                    np.testing.assert_array_equal(arrays["weighted_term_000"], [2.0, 6.0])
                    np.testing.assert_array_equal(arrays["weighted_term_001"], [-6.0, -12.0])
                    np.testing.assert_array_equal(arrays["candidate_total"], [-4.0, -6.0])
                    np.testing.assert_array_equal(arrays["returned_reward"], [-3.0, -5.0])
                    self.assertEqual(arrays["candidate_total"].dtype, np.dtype("float32"))
                    self.assertEqual(meta["task"]["reward_weight_scale"], 1.0)

    def test_stair_hook_excludes_preflight_and_keeps_qualified_reward_output(self):
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_protocol_v2.json")
        program = resolve_official_stair_reference_program(config, project_root=ROOT).reward_program
        values = {field.name: torch.zeros((2, *field.trailing_shape),
                  dtype=torch.bool if field.dtype == "bool" else torch.float32)
                  for field in program.contract.input_fields}
        inputs = StairRewardInputsV2._validated(contract=program.contract, values=values)
        dispatchers = [StairCompleteRewardDispatcher(program, num_envs=2) for _ in range(2)]
        for dispatcher in dispatchers:
            dispatcher.compute(inputs, common_step=1)
        with tempfile.TemporaryDirectory() as directory:
            capture = TrainingNumericAttachment(lambda: self.make_capture(Path(directory) / "archive", planned_control_steps=8, source_step_offset=1,
                semantics=dispatchers[1].training_numeric_semantics(control_dt=0.02)),
                bind=dispatchers[1].bind_training_numeric_capture)
            returned = []
            for common_step in range(2, 10):
                baseline = dispatchers[0].compute(inputs, common_step=common_step)
                observed = dispatchers[1].compute(inputs, common_step=common_step)
                self.assertTrue(torch.equal(baseline, observed))
                returned.append(observed.numpy().copy())
            self.assert_runtime_unchanged(dispatchers[0].runtime_summary(), dispatchers[1].runtime_summary())
            result = capture.finish(training_status="pass")
            self.assertEqual(result["coverage"]["observed_control_steps"], 8)
            self.assertEqual([w["sampled_control_steps"] for w in result["coverage"]["windows"]], [2, 3, 3])
            self.assertEqual(result["coverage"]["semantics"]["program_source_checksum"], program.program_source_checksum)
            for meta, arrays in iter_numeric_evidence(result["reference"]):
                if meta["kind"] != "training_sample":
                    continue
                np.testing.assert_array_equal(arrays["total_rate"], returned[meta["training_control_step"] - 1])
                weighted = [arrays[f"weighted_term_{index:03d}"] for index in range(len(program.term_names))]
                np.testing.assert_array_equal(reduce(np.add, weighted), arrays["total_rate"])
                self.assertEqual(meta["task"]["common_step"], meta["training_control_step"] + 1)

    def test_stair_training_entrypoints_bind_actual_source_preflight_to_numeric_identity(self):
        from true_rl_agent.tasks.stair_climbing import isaaclab_execution as execution
        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_protocol_v2.json")
        actual = {"status": "pass", "source_manifest_hash": "a" * 64}
        preflight = SimpleNamespace(to_summary_dict=lambda: dict(actual))
        for entrypoint in (execution.execute_stair_training, execution.execute_stair_training_smoke):
            with patch.object(execution, "_validate_source_manifest_preflight"), \
                    patch.object(execution, "_execute_stair_training", return_value="observed") as run:
                self.assertEqual(entrypoint(config=config, source_manifest_preflight=preflight,
                    agent_configuration=config.training_agent_configuration(), reward_program=None,
                    device="cuda:0", run_dir=Path("not-created"), checkpoint_output=Path("not-created.pt"),
                    run_name="identity-test", capture_training_numeric=True,
                    numeric_evidence_identity={"job": {"id": "preserved"}, "environment_source_preflight": {"unverified": True}}),
                    "observed")
                identity = run.call_args.kwargs["numeric_evidence_identity"]
                self.assertEqual(identity["environment_source_preflight"], actual)
                self.assertEqual(identity["job"], {"id": "preserved"})

    def test_native_stair_basic_reward_capture_without_extra_input_columns(self):
        import hashlib
        from true_rl_agent.candidates.verifier import StaticCandidateVerifierPolicy
        from true_rl_agent.evaluation.binding import canonical_json_hash
        from true_rl_agent.evaluation.subject import ExperimentSubject
        from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
        from true_rl_agent.tasks.stair_climbing.reward_inputs import StairRewardInputsVNext
        from true_rl_agent.tasks.stair_climbing.reward_program import qualify_stair_diagnostic_parent
        from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
        from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest
        from true_rl_agent.tasks.stair_climbing.vnext_reward_contract import build_stair_vnext_reward_runtime_contract
        from scripts.training_numeric_smoke import check_sampled_rewards

        config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        manifest = resolve_stair_cell_manifest(cell_variant_id=config.cell_variant_id, config=config,
                                              plugin=StairClimbingTaskPlugin(config))
        projection = stair_runtime_projection_from_manifest(manifest)
        contract = build_stair_vnext_reward_runtime_contract(projection=projection, authority=config.vnext_reward_authority)
        source = '''from true_rl_agent.candidates.components import RewardTerm, TrainingComponents
def action_rate_l2(inputs):
    return inputs.action_delta.square().sum(dim=1)
def dof_torques_l2(inputs):
    return inputs.joint_torque.square().sum(dim=1)
def build_training_components():
    return TrainingComponents(name="fixture_program", reward_terms=[
        RewardTerm(name="action_rate_l2", weight=-0.1, clip_max=5.0, fn=action_rate_l2),
        RewardTerm(name="dof_torques_l2", weight=-0.2, clip_max=10.0, fn=dof_torques_l2),
    ])
'''
        program = qualify_stair_diagnostic_parent(contract=contract,
            subject=ExperimentSubject.candidate("sampled-input-fixture"),
            subject_manifest_hash=canonical_json_hash({"fixture": "sampled-inputs"}), source_bytes=source.encode(),
            expected_source_checksum="sha256:" + hashlib.sha256(source.encode()).hexdigest(),
            source_name="sampled_input_fixture.py", static_verifier_policy=StaticCandidateVerifierPolicy(),
            trusted_reference_source_text=source)
        values = {field.name: torch.zeros((2, *field.trailing_shape),
                  dtype=torch.bool if field.dtype == "bool" else torch.float32) for field in contract.input_fields}
        values["last_action"][:, :2] = torch.tensor([[1., -2.], [2., 1.]])
        values["prev_action"][:, :2] = torch.tensor([[0., -1.], [-1., 1.]])
        values["action_delta"] = values["last_action"] - values["prev_action"]
        values["joint_vel"][:, :2] = torch.tensor([[0.25, -0.5], [1.5, -2.]])
        values["joint_torque"][:, :2] = torch.tensor([[3., 4.], [1., 2.]])
        inputs = StairRewardInputsVNext._validated(projection=projection, values=values)
        dispatcher = StairCompleteRewardDispatcher(program, num_envs=2)
        semantics = dispatcher.training_numeric_semantics(control_dt=0.02)
        self.assertNotIn("sampled_inputs", semantics)
        with tempfile.TemporaryDirectory() as directory:
            capture = TrainingNumericAttachment(
                lambda: self.make_capture(Path(directory) / "archive", planned_control_steps=1, semantics=semantics),
                bind=dispatcher.bind_training_numeric_capture)
            dispatcher.compute(inputs, common_step=1)
            result = capture.finish(training_status="pass")
            check_sampled_rewards(result, task="stair")
            sample = next(arrays for meta, arrays in iter_numeric_evidence(result["reference"])
                          if meta["kind"] == "training_sample")
        # Independently calculated from two nonzero dimensions: action 2/9,
        # torque 25/5. Raw archives are already clipped, before weighting.
        np.testing.assert_array_equal(sample["raw_term_000"], [2., 5.])
        np.testing.assert_array_equal(sample["raw_term_001"], [10., 5.])
        np.testing.assert_allclose(sample["weighted_term_000"], [-0.2, -0.5])
        np.testing.assert_allclose(sample["weighted_term_001"], [-2., -1.])
        np.testing.assert_allclose(sample["total_rate"], [-2.2, -1.5])
        self.assertEqual(set(sample), {"env_id", "raw_term_000", "raw_term_001",
                                      "weighted_term_000", "weighted_term_001", "total_rate"})

    def test_attachment_failure_is_terminal_and_preserves_bounded_root_error(self):
        def fail(*args, **kwargs):
            try:
                raise OSError("fixture write failure " + "x" * 600)
            except OSError as exc:
                raise RuntimeError("capture wrapper") from exc

        # Every optional entry shares the same boundary, including lazy inputs.
        for failure in ("create", "bind", "wants_step", "sample", "observe", "finish"):
            with self.subTest(failure=failure):
                recorder = Mock()
                recorder.wants_step.return_value = True
                recorder.finish.return_value = {"status": "complete"}
                create, bind, sample = Mock(return_value=recorder), Mock(), Mock(return_value=({"reward": np.ones(2)}, {}))
                target = {"create": create, "bind": bind, "sample": sample}.get(failure)
                (target if target is not None else getattr(recorder, failure)).side_effect = fail
                with self.assertLogs("true_rl_agent.runner.training_numeric_capture", level="WARNING") as logs:
                    attachment = TrainingNumericAttachment(create, bind)
                    attachment.observe(1, sample)
                    result = attachment.finish(training_status="pass")
                    calls = (recorder.mock_calls.copy(), sample.mock_calls.copy())
                    attachment.observe(2, sample)
                    self.assertEqual(attachment.finish(training_status="fail"), result)
                self.assertEqual((recorder.mock_calls, sample.mock_calls), calls)
                self.assertFalse(attachment.active)
                self.assertEqual(result["status"], "incomplete")
                self.assertEqual(result["error_type"], "OSError")
                self.assertEqual(result["error_message"], ("fixture write failure " + "x" * 600)[:512])
                self.assertEqual(len(logs.output), 1)
                if failure != "finish":
                    recorder.finish.assert_not_called()

    def test_training_entry_keeps_checkpoint_after_attachment_failure_but_not_training_failure(self):
        """Real entry/result code with CPU environment/learner substitutes, never Isaac."""
        from true_rl_agent.tasks.stair_climbing import isaaclab_execution as execution

        task_config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_protocol_v2.json")
        program = resolve_official_stair_reference_program(task_config, project_root=ROOT).reward_program
        inputs = StairRewardInputsV2._validated(contract=program.contract, values={
            field.name: torch.zeros((2, *field.trailing_shape), dtype=torch.bool if field.dtype == "bool" else torch.float32)
            for field in program.contract.input_fields})
        execution_contract = {"torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32, "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark, "init_at_random_episode_length": False,
            "checkpoint_selector": "final_iteration"}
        config = SimpleNamespace(is_unified_protocol=False, train_seed=1, task_protocol_hash=task_config.task_protocol_hash,
            protocol_v1=SimpleNamespace(to_dict=lambda: {}),
            protocol_v2=SimpleNamespace(runner_owned_training_distribution={}),
            training_execution_contract=lambda: execution_contract,
            training_agent_configuration=lambda: SimpleNamespace(verify_runner=lambda runner: {}))
        workload = SimpleNamespace(kind="phase_d_smoke", num_envs=2, num_steps_per_env=1,
            max_iterations=1, budget_steps=2, checkpoint_save_interval=1)

        for failure in ("semantics", "observe", "finish", "learn", "checkpoint"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                root = Path(directory)
                dispatcher = StairCompleteRewardDispatcher(program, num_envs=2)
                term = SimpleNamespace(dispatcher=dispatcher,
                    runtime_summary=lambda: {**dispatcher.runtime_summary(), "dispatcher_calls": dispatcher.step_calls})
                env = SimpleNamespace(device="cpu", step_dt=0.02, reset=Mock(), close=Mock(),
                    reward_manager=SimpleNamespace(active_terms=["complete_reward"], get_term_cfg=lambda name: SimpleNamespace(weight=1.0)))
                env.unwrapped = env
                env_cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=2), recorders=None)
                agent_cfg = SimpleNamespace(device="cpu", clip_actions=None, max_iterations=1, to_dict=lambda: {})

                def learner(*args, log_dir, **kwargs):
                    def learn(**kwargs):
                        dispatcher.compute(inputs, common_step=1)
                        if failure == "learn":
                            raise RuntimeError("fixture learner failure")
                        if failure != "checkpoint":
                            Path(log_dir).mkdir(parents=True)
                            (Path(log_dir) / "model_0.pt").write_bytes(b"CPU fixture checkpoint, not policy weights")
                    return SimpleNamespace(current_learning_iteration=0, add_git_repo_to_log=Mock(), learn=learn)

                stack.enter_context(patch.dict(sys.modules, {
                    "gymnasium": SimpleNamespace(make=lambda *a, **kw: env),
                    "isaaclab_rl.rsl_rl": SimpleNamespace(RslRlVecEnvWrapper=lambda env, **kw: env),
                    "rsl_rl.runners": SimpleNamespace(OnPolicyRunner=learner),
                    "true_rl_agent.tasks.stair_climbing.isaaclab_reward_runtime": SimpleNamespace(
                        get_stair_complete_reward_term=lambda env: term, run_stair_reward_activation_preflight=lambda *a, **kw: {}),
                    "true_rl_agent.tasks.stair_climbing.isaaclab_runtime_env": SimpleNamespace(
                        build_stair_train_env_cfg=lambda *a, **kw: env_cfg, register_production_stair_envs=lambda: None,
                        stair_training_distribution_runtime_evidence=lambda *a, **kw: {}),
                }))
                for name, value in {
                    "_validate_training_inputs": None,
                    "_stair_runtime_projection_for_execution": SimpleNamespace(runtime_adapter_id="cpu-fixture"),
                    "isaaclab_locomotion_runtime_spec": SimpleNamespace(instantiate_runner_cfg=lambda: agent_cfg),
                    "configure_stair_rsl_rl_agent": {}, "verify_stair_termination_manager_resolution": {},
                    "verify_stair_rsl_rl_observation_groups": None, "initial_model_state_hash": "a" * 64,
                }.items():
                    stack.enter_context(patch.object(execution, name, return_value=value))
                # The learn failure occurs after an attachment failure: it must still win.
                if failure == "semantics":
                    stack.enter_context(patch.object(dispatcher, "training_numeric_semantics", side_effect=OSError("fixture semantics failure")))
                elif failure != "checkpoint":
                    stack.enter_context(patch.object(TrainingNumericCapture, "finish" if failure == "finish" else "observe",
                                                     side_effect=OSError("fixture attachment failure")))
                result = execution._execute_stair_training(config=config, agent_configuration=object(), reward_program=program,
                    device="cpu", run_dir=root / "run", checkpoint_output=root / "checkpoint.pt", run_name="cpu-boundary",
                    expected_initial_policy_state_hash=None, resolved_cell_manifest=None, workload=workload,
                    capture_training_numeric=True)
                self.assertEqual(result.status, "fail" if failure in {"learn", "checkpoint"} else "pass")
                self.assertTrue(result.training_executed)
                self.assertGreater(result.runtime_activation_summary["dispatcher_calls"], 0)
                if failure in {"learn", "checkpoint"}:
                    self.assertIsNone(result.checkpoint)
                    self.assertIn("learner failure" if failure == "learn" else "no checkpoint", result.errors[0]["message"])
                else:
                    self.assertEqual(Path(result.checkpoint["uri"]).read_bytes(), b"CPU fixture checkpoint, not policy weights")
                    self.assertEqual(result.errors, ())
                if failure != "checkpoint":
                    self.assertEqual(result.training["numeric_evidence"]["status"], "incomplete")
                    self.assertEqual(result.training["numeric_evidence"]["error_type"], "OSError")
                env.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
