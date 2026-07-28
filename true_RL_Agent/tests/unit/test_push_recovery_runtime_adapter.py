from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.components import TrainingComponents  # noqa: E402
from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence  # noqa: E402
from true_rl_agent.tasks.push_recovery import PushRecoveryTaskPlugin  # noqa: E402
from true_rl_agent.tasks.push_recovery.runtime_adapter import (  # noqa: E402
    CandidateRewardVecEnvWrapper,
    CandidateRuntimeAdapter,
    RuntimeComponentError,
    guard_vector,
    runtime_contract,
)


class _FakeVecEnv:
    def __init__(self, *, extras: dict | None = None) -> None:
        self.unwrapped = self
        self.step_calls = 0
        self.extras = extras or {}

    def step(self, actions):
        self.step_calls += 1
        return ({"obs": self.step_calls}, [1.0, 1.0], [False, False], self.extras)


class _VelocityDeltaVecEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.step_calls = 0
        self.robot_data = type("RobotData", (), {})()
        self.robot = type("Robot", (), {"data": self.robot_data})()
        self.scene = {"robot": self.robot}
        self.velocities = [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.22, 0.0, 0.0], [0.03, 0.0, 0.0]],
        ]

    def step(self, actions):
        index = min(self.step_calls, len(self.velocities) - 1)
        self.robot_data.root_lin_vel_w = self.velocities[index]
        self.step_calls += 1
        return ({"obs": self.step_calls}, [1.0, 1.0], [False, False], {})


class PushRecoveryRuntimeAdapterTests(unittest.TestCase):
    def test_runtime_adapter_rejects_noncanonical_integer_inputs(self) -> None:
        components = TrainingComponents(name="strict-runtime-inputs")
        valid = {
            "manifest": None,
            "components": components,
            "num_envs": 2,
            "planned_training_steps": 8,
            "diagnostic_interval_steps": 1,
        }
        for field_name, invalid in (
            ("num_envs", True),
            ("num_envs", 2.5),
            ("planned_training_steps", "8"),
            ("planned_training_steps", 0),
            ("diagnostic_interval_steps", False),
            ("diagnostic_interval_steps", -1),
        ):
            kwargs = {**valid, field_name: invalid}
            with self.subTest(field_name=field_name, invalid=invalid):
                with self.assertRaisesRegex(RuntimeComponentError, field_name):
                    CandidateRuntimeAdapter(**kwargs)

    def _write_candidate(self, root: Path) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: runtime_reward
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            textwrap.dedent(
                """
                from true_rl_agent.candidates.components import DiagnosticTerm, RewardTerm, ScheduleTerm, TrainingComponents


                def reward(inputs):
                    return [-0.25, 0.25]


                def diagnostic(inputs):
                    return {"progress": inputs.progress_fraction}


                def build_training_components():
                    return TrainingComponents(
                        name="runtime_reward",
                        reward_terms=[RewardTerm(name="recovery_bonus", weight=2.0, fn=reward)],
                        diagnostics=[DiagnosticTerm(name="progress_probe", fn=diagnostic)],
                        schedules=[ScheduleTerm(name="scale", key="reward_weight_scale", value=0.5)],
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def _write_push_signal_candidate(self, root: Path) -> Path:
        candidate = root / "candidate"
        (candidate / "training_code").mkdir(parents=True)
        (candidate / "candidate_manifest.yaml").write_text(
            textwrap.dedent(
                """
                schema_version: true_rl_agent.candidate.v1
                candidate_id: push_signal_reward
                entrypoint: training_code.components:build_training_components
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        (candidate / "training_code" / "components.py").write_text(
            textwrap.dedent(
                """
                from true_rl_agent.candidates.components import RewardTerm, TrainingComponents


                def reward(inputs):
                    return [
                        float(active) + float(vec[0])
                        for active, vec in zip(inputs.push_active, inputs.push_vec_public)
                    ]


                def build_training_components():
                    return TrainingComponents(
                        name="push_signal_reward",
                        reward_terms=[RewardTerm(name="push_signal_reward", weight=1.0, fn=reward)],
                    )
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        return candidate

    def test_candidate_reward_adapter_adds_guarded_rewards_and_records_activation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
                diagnostic_interval_steps=1,
            )
            wrapper = CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter)

            result = wrapper.step([[0.0], [0.0]])

        self.assertEqual(result[1], [0.75, 1.25])
        runtime = adapter.runtime_summary()
        self.assertEqual(runtime["reward_term_calls"], 1)
        self.assertEqual(runtime["diagnostic_calls"], 1)
        self.assertTrue(runtime["nonzero_reward_contribution"])
        self.assertEqual(runtime["last_candidate_reward_mean"], 0.0)
        self.assertGreater(runtime["max_abs_candidate_reward"], 0.0)
        self.assertEqual(
            runtime["component_activation_records"],
            [
                {
                    "name": "recovery_bonus",
                    "observation_count": 1,
                    "nonzero_observation_count": 1,
                    "max_abs_weighted_contribution": 0.25,
                    "last_weighted_mean": 0.0,
                    "effective_weight": 1.0,
                    "sampling_interval_steps": 1,
                    "component_kind": "reward_term",
                    "nonzero_observation_fraction": 1.0,
                }
            ],
        )
        self.assertIn("reward_callable", runtime["capability_activation_records"])
        self.assertIn("reward_weight_scale", runtime["capability_activation_records"])
        evidence = RuntimeActivationEvidence.from_mapping(runtime)
        self.assertEqual(
            evidence.to_public_dict()["component_activation_records"],
            runtime["component_activation_records"],
        )
        self.assertIn("progress_probe", evidence.diagnostic_outputs)

    def test_candidate_runtime_adapter_binds_expected_id_to_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            candidate_dir = self._write_candidate(Path(tmp))

            with self.assertRaisesRegex(ValueError, "candidate manifest identity mismatch"):
                CandidateRuntimeAdapter.from_candidate_dir(
                    candidate_dir,
                    num_envs=2,
                    planned_training_steps=8,
                    expected_candidate_id="different_candidate",
                )

    def test_guard_vector_rejects_wrong_length_and_nonfinite_outputs(self) -> None:
        with self.assertRaisesRegex(RuntimeComponentError, "must return 2 values"):
            guard_vector([1.0], name="bad", kind="reward", num_envs=2)
        with self.assertRaisesRegex(RuntimeComponentError, "non-finite"):
            guard_vector([1.0, float("nan")], name="bad", kind="reward", num_envs=2)

    def test_runtime_contract_reserved_reward_inputs_match_prompt_contract(self) -> None:
        runtime = runtime_contract()
        prompt = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[],
            public_context={"previous_rounds": 0, "public_only": True},
        )["reward_inputs_contract"]

        self.assertEqual(runtime["reserved_zero_fallback_inputs"], list(prompt["reserved_fields"]))
        self.assertEqual(runtime["candidate_receives_prompt_effective"], prompt["field_order"])

    def test_runtime_contract_exposes_public_push_signal_as_effective_inputs(self) -> None:
        runtime = runtime_contract()
        prompt = PushRecoveryTaskPlugin().prompt_adapter().build_prompt_sections(
            public_records=[],
            public_context={"previous_rounds": 0, "public_only": True},
        )["reward_inputs_contract"]

        self.assertIn("push_active", runtime["candidate_receives_prompt_effective"])
        self.assertIn("push_vec_public", runtime["candidate_receives_prompt_effective"])
        self.assertEqual(runtime["public_push_signal_inputs"], prompt["public_push_signal_fields"])
        self.assertIn("push_active", prompt["fields"])
        self.assertIn("push_vec_public", prompt["fields"])
        self.assertNotIn("push_active", prompt["reserved_fields"])
        self.assertNotIn("push_vec_public", prompt["reserved_fields"])

    def test_runtime_uses_explicit_public_push_signal_from_step_extras(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_push_signal_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
            )
            wrapper = CandidateRewardVecEnvWrapper(
                _FakeVecEnv(
                    extras={
                        "push_active": [0.0, 1.0],
                        "push_vec_public": [[0.0, 0.0], [0.25, -0.1]],
                    }
                ),
                adapter,
            )

            result = wrapper.step([[0.0], [0.0]])

        self.assertEqual(result[1], [1.0, 2.25])
        provenance = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(provenance["push_active"], "observed")
        self.assertEqual(provenance["push_vec_public"], "observed")

    def test_runtime_derives_public_push_signal_from_observed_velocity_delta(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_push_signal_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
                diagnostic_interval_steps=1,
            )
            wrapper = CandidateRewardVecEnvWrapper(_VelocityDeltaVecEnv(), adapter)

            first = wrapper.step([[0.0], [0.0]])
            second = wrapper.step([[0.0], [0.0]])

        self.assertEqual(first[1], [1.0, 1.0])
        self.assertAlmostEqual(second[1][0], 2.22)
        self.assertEqual(second[1][1], 1.0)
        provenance = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(provenance["push_active"], "observed")
        self.assertEqual(provenance["push_vec_public"], "observed")
        activation = adapter.runtime_summary()["component_activation_records"][0]
        self.assertEqual(activation["observation_count"], 2)
        self.assertEqual(activation["nonzero_observation_count"], 1)
        self.assertEqual(activation["nonzero_observation_fraction"], 0.5)

    def test_runtime_keeps_public_push_signal_zero_when_unobserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            adapter = CandidateRuntimeAdapter.from_candidate_dir(
                self._write_push_signal_candidate(Path(tmp)),
                num_envs=2,
                planned_training_steps=8,
            )
            wrapper = CandidateRewardVecEnvWrapper(_FakeVecEnv(), adapter)

            result = wrapper.step([[0.0], [0.0]])

        self.assertEqual(result[1], [1.0, 1.0])
        provenance = adapter.runtime_summary()["input_provenance"]
        self.assertEqual(provenance["push_active"], "zero_fallback")
        self.assertEqual(provenance["push_vec_public"], "zero_fallback")


if __name__ == "__main__":
    unittest.main()
