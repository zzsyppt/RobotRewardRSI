from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.json_contract import canonical_json_bytes  # noqa: E402
from true_rl_agent.tasks.stair_climbing.phase_d_prompt_preflight import (  # noqa: E402
    EXPECTED_STAIR_PHASE_D_PROMPT_PREFLIGHT_REPORT_HASH,
    load_stair_phase_d_prompt_preflight_report,
    validate_stair_phase_d_prompt_preflight_report,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (  # noqa: E402
    PREVIOUS_STAIR_PROTOCOL_V2_HASH,
)

REPORT_PATH = ROOT / "PLAN" / "stair" / "phase_d_prompt_preflight_report.json"


class StairPhaseDPromptPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.historical = load_stair_phase_d_prompt_preflight_report(REPORT_PATH)

    def test_committed_report_remains_read_only_historical_evidence(self) -> None:
        self.assertEqual(
            self.historical["protocol_binding"]["task_protocol_hash"],
            PREVIOUS_STAIR_PROTOCOL_V2_HASH,
        )
        self.assertEqual(self.historical["status"], "pass")
        self.assertEqual(
            self.historical["report_hash"],
            EXPECTED_STAIR_PHASE_D_PROMPT_PREFLIGHT_REPORT_HASH,
        )
        self.assertEqual(
            self.historical["capability_exposure"],
            {
                "complete_reward_program_prompt_visible": False,
                "complete_reward_program_candidate_writable": False,
                "complete_reward_program_effective_now": False,
                "provider_declared_component_count": 0,
            },
        )

    def test_report_rejects_tampering_and_private_payload_fields(self) -> None:
        tampered = copy.deepcopy(self.historical)
        tampered["prompt_payload"]["utf8_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "report hash mismatch"):
            validate_stair_phase_d_prompt_preflight_report(tampered)

        private = copy.deepcopy(self.historical)
        private["pilot_raw_seed"] = 123
        private["report_hash"] = canonical_json_hash(
            {key: item for key, item in private.items() if key != "report_hash"}
        )
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            validate_stair_phase_d_prompt_preflight_report(private)

        rehashed = copy.deepcopy(self.historical)
        rehashed["reference_source"]["reference_id"] = "forged-reference"
        rehashed["report_hash"] = canonical_json_hash(
            {key: item for key, item in rehashed.items() if key != "report_hash"}
        )
        with self.assertRaisesRegex(ValueError, "frozen report identity mismatch"):
            validate_stair_phase_d_prompt_preflight_report(rehashed)

    def test_loader_requires_exact_canonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_bytes(
                canonical_json_bytes(self.historical, trailing_newline=False)
            )
            with self.assertRaisesRegex(ValueError, "not canonical JSON"):
                load_stair_phase_d_prompt_preflight_report(path)


if __name__ == "__main__":
    unittest.main()
