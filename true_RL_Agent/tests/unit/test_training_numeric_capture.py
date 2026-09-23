from __future__ import annotations

from functools import reduce
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from true_rl_agent.candidates.components import RewardTerm, TrainingComponents
from true_rl_agent.runner.numeric_evidence import iter_numeric_evidence, load_numeric_evidence_manifest
from true_rl_agent.runner.training_numeric_capture import TrainingNumericCapture, finish_training_numeric_capture
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
                capture = self.make_capture(Path(directory) / "archive", source_step_offset=1,
                    semantics=adapters[1].training_numeric_semantics(control_dt=0.02))
                adapters[1].bind_training_numeric_capture(capture)
                rng_before = torch.get_rng_state().clone()
                for _ in range(24):
                    self.assertTrue(torch.equal(adapters[0].apply_step_rewards(**kwargs), adapters[1].apply_step_rewards(**kwargs)))
                self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))
                self.assertEqual(adapters[0].runtime_summary(), adapters[1].runtime_summary())
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
            capture = self.make_capture(Path(directory) / "archive", planned_control_steps=8, source_step_offset=1,
                semantics=dispatchers[1].training_numeric_semantics(control_dt=0.02))
            dispatchers[1].bind_training_numeric_capture(capture)
            returned = []
            for common_step in range(2, 10):
                baseline = dispatchers[0].compute(inputs, common_step=common_step)
                observed = dispatchers[1].compute(inputs, common_step=common_step)
                self.assertTrue(torch.equal(baseline, observed))
                returned.append(observed.numpy().copy())
            self.assertEqual(dispatchers[0].runtime_summary(), dispatchers[1].runtime_summary())
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


if __name__ == "__main__":
    unittest.main()
