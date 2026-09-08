"""Checkpoint ABI tests; CPU tensor coverage is required in the daily suite."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from true_rl_agent.backends.rsl_rl_checkpoint import (
    initial_model_state_hash,
    legacy_model_states,
    load_inference_checkpoint,
    load_training_checkpoint,
)

import torch


class RslCheckpointMappingTests(unittest.TestCase):
    def test_legacy_mapping_keeps_models_normalizers_and_distribution_separate(self):
        names = {
            "actor.0.weight": object(),
            "critic.0.weight": object(),
            "actor_obs_normalizer._mean": object(),
            "critic_obs_normalizer._std": object(),
            "std": object(),
        }
        result = legacy_model_states({"model_state_dict": names})
        self.assertEqual(
            result,
            {
                "actor_state_dict": {
                    "mlp.0.weight": names["actor.0.weight"],
                    "obs_normalizer._mean": names["actor_obs_normalizer._mean"],
                    "distribution.std_param": names["std"],
                },
                "critic_state_dict": {
                    "mlp.0.weight": names["critic.0.weight"],
                    "obs_normalizer._std": names["critic_obs_normalizer._std"],
                },
            },
        )


class RslCheckpointTensorTests(unittest.TestCase):
    def test_scalar_normalizer_state_participates_in_initial_identity(self):
        state = {"count": torch.tensor(0), "weight": torch.ones(2, 3)}
        actor = SimpleNamespace(state_dict=lambda: state)
        runner = SimpleNamespace(alg=SimpleNamespace(actor=actor, critic=actor))
        first = initial_model_state_hash(runner)
        self.assertEqual(first, initial_model_state_hash(runner))
        state["count"] += 1
        self.assertNotEqual(first, initial_model_state_hash(runner))
        state.clear()
        with self.assertRaisesRegex(ValueError, "empty"):
            initial_model_state_hash(runner)

    def test_inference_never_loads_optimizer_and_legacy_cannot_resume(self):
        calls = []
        runner = SimpleNamespace(
            device="cpu",
            current_learning_iteration=0,
            alg=SimpleNamespace(
                load=lambda *args, **kwargs: calls.append((args, kwargs))
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint.pt"
            torch.save(
                {"actor_state_dict": {}, "critic_state_dict": {}, "iter": 7}, path
            )
            load_training_checkpoint(runner, path)
            self.assertEqual(runner.current_learning_iteration, 7)
            self.assertIsNone(calls[-1][0][1])
            loading = load_inference_checkpoint(runner, path)
            self.assertEqual(loading["checkpoint_format"], "rsl_rl_5")
            self.assertFalse(loading["optimizer_loaded"])
            self.assertEqual(calls[-1][0][1], {"actor": True, "critic": True})
            torch.save({"model_state_dict": {}}, path)
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                load_training_checkpoint(runner, path)

    def test_legacy_log_standard_deviation_keeps_log_parameter_name(self):
        value = torch.tensor([-0.3, 0.2])
        state = legacy_model_states({"model_state_dict": {"log_std": value}})
        self.assertIs(state["actor_state_dict"]["distribution.log_std_param"], value)
