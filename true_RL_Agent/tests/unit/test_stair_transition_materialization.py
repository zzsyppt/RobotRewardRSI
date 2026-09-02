from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.evaluation_runtime import (  # noqa: E402
    _invoke_inference_policy,
)
from true_rl_agent.tasks.stair_climbing.transition_decoder import (  # noqa: E402
    STAIR_TRANSITION_HISTORY_CHUNK_STEPS,
    _stack_to_host,
)


class _Value:
    def __init__(self, value: int) -> None:
        self.value = value

    def detach(self) -> _Value:
        return self


class _HostTensor:
    def __init__(self, values: tuple[int, ...]) -> None:
        self.values = values

    def cpu(self) -> _HostTensor:
        return self


class _FakeTorch:
    def __init__(self) -> None:
        self.stack_sizes: list[int] = []
        self.cat_calls = 0

    def stack(self, values: tuple[_Value, ...], *, dim: int) -> _HostTensor:
        self.assert_zero_dim(dim)
        self.stack_sizes.append(len(values))
        return _HostTensor(tuple(value.value for value in values))

    def cat(self, tensors: tuple[_HostTensor, ...], *, dim: int) -> _HostTensor:
        self.assert_zero_dim(dim)
        self.cat_calls += 1
        return _HostTensor(
            tuple(value for tensor in tensors for value in tensor.values)
        )

    @staticmethod
    def assert_zero_dim(dim: int) -> None:
        if dim != 0:
            raise AssertionError("history materialization must stack on the step axis")


class StairTransitionMaterializationTests(unittest.TestCase):
    def test_gpu_staging_is_fixed_size_and_preserves_step_order(self) -> None:
        torch = _FakeTorch()
        value_count = STAIR_TRANSITION_HISTORY_CHUNK_STEPS * 2 + 1
        values = tuple(_Value(index) for index in range(value_count))

        materialized = _stack_to_host(
            torch,
            values,
            field_name="test_history",
        )

        self.assertEqual(
            torch.stack_sizes,
            [STAIR_TRANSITION_HISTORY_CHUNK_STEPS] * 2 + [1],
        )
        self.assertEqual(torch.cat_calls, 1)
        self.assertEqual(materialized.values, tuple(range(value_count)))

    def test_policy_call_runs_under_inference_mode(self) -> None:
        calls = []

        class _InferenceMode:
            def __enter__(self) -> None:
                calls.append("enter")

            def __exit__(self, *_args: object) -> None:
                calls.append("exit")

        fake_torch = SimpleNamespace(inference_mode=_InferenceMode)

        with patch.dict(sys.modules, {"torch": fake_torch}):
            result = _invoke_inference_policy(
                lambda observation: calls.append(observation) or "actions",
                "observation",
            )

        self.assertEqual(result, "actions")
        self.assertEqual(calls, ["enter", "observation", "exit"])


if __name__ == "__main__":
    unittest.main()
