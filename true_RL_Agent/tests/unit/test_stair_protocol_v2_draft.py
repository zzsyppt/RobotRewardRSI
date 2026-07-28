from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH,
    StairProtocolV2Draft,
)

V1_PATH = PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v1.json"
V2_DRAFT_PATH = (
    PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
)


def _write_rehashed_draft(
    output_dir: Path,
    path: tuple[str | int, ...],
    value: Any,
) -> Path:
    payload = json.loads(V2_DRAFT_PATH.read_text(encoding="utf-8"))
    cursor: Any = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    payload["draft_protocol_hash"] = canonical_json_hash(
        {
            key: item
            for key, item in payload.items()
            if key != "draft_protocol_hash"
        }
    )
    output = output_dir / ("-".join(str(item) for item in path) + ".json")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


class StairProtocolV2DraftTests(unittest.TestCase):
    def test_stable_overlay_over_immutable_v1(self) -> None:
        v1_bytes_before = V1_PATH.read_bytes()
        v1 = StairProtocolV1.load(V1_PATH)
        draft = StairProtocolV2Draft.load(V2_DRAFT_PATH)
        payload = draft.to_public_dict()

        self.assertEqual(
            payload["base_protocol"]["task_protocol_hash"],
            v1.protocol_hash,
        )
        self.assertEqual(V1_PATH.read_bytes(), v1_bytes_before)
        self.assertEqual(draft.protocol_hash, EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH)
        self.assertEqual(
            draft.allowed_pilot_budget_steps,
            tuple(payload["pilot_training"]["allowed_budget_steps"]),
        )
        self.assertEqual(
            draft.capability_profile.capability_ids,
            ("complete_reward_program",),
        )
        self.assertEqual(
            StairProtocolV2Draft.load(V2_DRAFT_PATH).protocol_hash,
            draft.protocol_hash,
        )

    def test_rejects_rehashed_semantic_weakening(self) -> None:
        mutations = (
            (("base_protocol", "task_protocol_hash"), "0" * 64),
            (("genesis_anchor_policy", "random"), "bootstrap"),
            (("reference_program", "entrypoint"), "training_code.other:build"),
            (("pilot_training", "allowed_budget_steps", 0), 1),
            (("pilot_training", "maximum_budget_steps"), 999_999_999),
            (("reference_capability_gate", "low_step_bin_ids", 0), "stair_999"),
            (
                (
                    "reference_capability_gate",
                    "minimum_macro_mean_success_rate",
                ),
                0.0,
            ),
            (
                (
                    "reference_capability_gate",
                    "threshold_relaxation_after_pilot_allowed",
                ),
                0,
            ),
            (("inheritance", "height_bins"), "copied_values"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            for path, value in mutations:
                with self.subTest(path=path):
                    tampered = _write_rehashed_draft(output_dir, path, value)
                    with self.assertRaisesRegex(ValueError, "Stair V2 draft"):
                        StairProtocolV2Draft.load(
                            tampered,
                            base_v1_path=V1_PATH,
                        )

    def test_semantic_names_remain_inside_task_owned_code(self) -> None:
        task_dir = (
            PROJECT_ROOT
            / "src"
            / "true_rl_agent"
            / "tasks"
            / "stair_climbing"
        )
        payload = json.loads(V2_DRAFT_PATH.read_text(encoding="utf-8"))
        forbidden = (
            payload["reference_capability_gate"]["low_step_bin_ids"][0],
            "minimum_macro_mean_success_rate",
            "minimum_macro_mean_max_progress_m",
            payload["protocol_id"],
        )

        leaks: list[str] = []
        for source_path in (PROJECT_ROOT / "src" / "true_rl_agent").rglob("*.py"):
            if source_path.is_relative_to(task_dir):
                continue
            text = source_path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                leaks.append(str(source_path.relative_to(PROJECT_ROOT)))
        self.assertEqual(leaks, [])


if __name__ == "__main__":
    unittest.main()
