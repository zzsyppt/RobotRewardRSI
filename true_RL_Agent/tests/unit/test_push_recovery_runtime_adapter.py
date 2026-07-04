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

from true_rl_agent.candidates.mechanism import RuntimeActivationEvidence  # noqa: E402
from true_rl_agent.tasks.push_recovery.runtime_adapter import (  # noqa: E402
    CandidateRewardVecEnvWrapper,
    CandidateRuntimeAdapter,
    RuntimeComponentError,
    guard_vector,
)


class _FakeVecEnv:
    def __init__(self) -> None:
        self.unwrapped = self
        self.step_calls = 0

    def step(self, actions):
        self.step_calls += 1
        return ({"obs": self.step_calls}, [1.0, 1.0], [False, False], {})


class PushRecoveryRuntimeAdapterTests(unittest.TestCase):
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
        self.assertIn("reward_callable", runtime["capability_activation_records"])
        self.assertIn("reward_weight_scale", runtime["capability_activation_records"])
        evidence = RuntimeActivationEvidence.from_mapping(runtime)
        self.assertIn("progress_probe", evidence.diagnostic_outputs)

    def test_guard_vector_rejects_wrong_length_and_nonfinite_outputs(self) -> None:
        with self.assertRaisesRegex(RuntimeComponentError, "must return 2 values"):
            guard_vector([1.0], name="bad", kind="reward", num_envs=2)
        with self.assertRaisesRegex(RuntimeComponentError, "non-finite"):
            guard_vector([1.0, float("nan")], name="bad", kind="reward", num_envs=2)


if __name__ == "__main__":
    unittest.main()
