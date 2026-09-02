from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.candidates.components import ScheduleTerm, TrainingComponents
from true_rl_agent.candidates.prompt_projection import build_prompt_payload
from true_rl_agent.candidates.schedule_contract import (
    REWARD_WEIGHT_CHANGE_COMPONENT,
    REWARD_WEIGHT_SCALE_KEY,
    allowed_schedule_keys,
    effective_schedule_keys,
    schedule_key_bounds,
)
from true_rl_agent.candidates.verifier import StaticCandidateVerifier
from true_rl_agent.tasks.push_recovery.runtime_adapter import runtime_contract


class CandidateScheduleContractTests(unittest.TestCase):
    def test_schedule_contract_is_shared_by_prompt_verifier_components_and_runtime(self) -> None:
        self.assertEqual(allowed_schedule_keys(), (REWARD_WEIGHT_SCALE_KEY,))
        self.assertEqual(effective_schedule_keys(), (REWARD_WEIGHT_SCALE_KEY,))
        self.assertEqual(schedule_key_bounds(), {REWARD_WEIGHT_SCALE_KEY: [0.0, 10.0]})

        payload = build_prompt_payload(public_context={"public_only": True})
        self.assertEqual(
            payload["required_candidate_package"]["component_contract"]["allowed_schedule_keys"],
            list(allowed_schedule_keys()),
        )
        self.assertEqual(
            payload["control_effectiveness_contract"]["schedule_controls"]["effective_keys"],
            list(effective_schedule_keys()),
        )

        self.assertEqual(
            StaticCandidateVerifier().policy.allowed_schedule_keys,
            frozenset(allowed_schedule_keys()),
        )
        components = TrainingComponents(
            name="schedule_contract",
            schedules=[ScheduleTerm(name="scale", key=REWARD_WEIGHT_SCALE_KEY, value=1.0)],
        )
        self.assertEqual(components.to_change_set().changed_components, frozenset({REWARD_WEIGHT_CHANGE_COMPONENT}))

        runtime = runtime_contract()
        self.assertEqual(runtime["schedule_controls"]["effective_keys"], list(effective_schedule_keys()))
        self.assertEqual(runtime["schedule_controls"]["bounds"], schedule_key_bounds())


if __name__ == "__main__":
    unittest.main()
