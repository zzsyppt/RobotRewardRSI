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
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.phase0_contract import StairProtocolV1
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH,
    EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH,
    StairProtocolV2Draft,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (
    resolved_stair_training_distribution,
    validate_stair_training_distribution_contract,
)

V1_PATH = PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v1.json"
V2_DRAFT_PATH = (
    PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft.json"
)
V2_DRAFT_R2_PATH = (
    PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2_draft_r2.json"
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
    def test_legacy_draft_is_parser_only_not_executable_task_config(self) -> None:
        self.assertEqual(
            StairProtocolV2Draft.load(V2_DRAFT_PATH).protocol_hash,
            EXPECTED_STAIR_PROTOCOL_V2_DRAFT_HASH,
        )
        with self.assertRaises(ValueError):
            StairClimbingTaskConfig.defaults(V2_DRAFT_PATH)

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
            tuple(payload["pilot_training"]["allowed_budget_steps"]),
            (49_152_000, 98_304_000, 147_456_000),
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

    def test_r2_binds_runner_owned_bootstrap_without_changing_v1(self) -> None:
        v1 = StairProtocolV1.load(V1_PATH)
        draft = StairProtocolV2Draft.load(V2_DRAFT_R2_PATH)
        self.assertEqual(
            draft.protocol_hash,
            EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH,
        )
        contract = draft.runner_owned_training_distribution
        self.assertIsNotNone(contract)
        assert contract is not None
        resolved = resolved_stair_training_distribution(
            contract,
            base_protocol=v1.to_dict(),
        )
        self.assertTrue(resolved["terrain"]["generator_curriculum"])
        self.assertEqual(
            resolved["terrain"]["step_height_range_m"],
            [0.05, 0.2],
        )
        self.assertEqual(
            resolved["command"]["lin_vel_x_range_mps"],
            [0.0, 0.8],
        )
        self.assertFalse(contract["candidate_writable"])
        self.assertFalse(contract["prompt_visible"])
        self.assertEqual(
            contract["selection_provenance"],
            {
                "kind": "upstream_alignment_not_failure_result",
                "invalid_pre_capture_pilot_results_used": False,
            },
        )
        self.assertEqual(draft.pilot_training_seed_count, 1)
        self.assertEqual(draft.pilot_max_iterations, 2000)
        self.assertEqual(
            draft.pilot_training_contract["budget_policy"],
            "operator_selected_fixed_budget_without_optimality_or_convergence_claim",
        )
        source_identities = [
            (item["import_package"], item["package_relative_path"])
            for item in contract["upstream_reference"]["source_files"]
        ]
        self.assertIn(("isaaclab", "terrains/terrain_importer.py"), source_identities)
        self.assertIn(
            (
                "isaaclab_tasks",
                "manager_based/locomotion/velocity/mdp/curriculums.py",
            ),
            source_identities,
        )
        source_by_path = {
            item["package_relative_path"]: item
            for item in contract["upstream_reference"]["source_files"]
        }
        self.assertEqual(
            source_by_path["terrains/terrain_importer.py"]["covered_symbols"],
            ("TerrainImporter.update_env_origins",),
        )
        self.assertEqual(
            source_by_path[
                "manager_based/locomotion/velocity/mdp/curriculums.py"
            ]["covered_symbols"],
            ("terrain_levels_vel",),
        )

        weakened = draft.to_public_dict()["runner_owned_training_distribution"]
        weakened["candidate_writable"] = True
        with self.assertRaisesRegex(ValueError, "candidate_writable"):
            validate_stair_training_distribution_contract(
                weakened,
                base_protocol=v1.to_dict(),
            )

    def test_fixed_budget_value_is_owned_only_by_the_trusted_draft(self) -> None:
        payload = json.loads(V2_DRAFT_R2_PATH.read_text(encoding="utf-8"))
        payload["pilot_training"]["fixed_max_iterations"] = 1999
        payload["draft_protocol_hash"] = canonical_json_hash(
            {
                key: item
                for key, item in payload.items()
                if key != "draft_protocol_hash"
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "forged-fixed-budget.json"
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "trusted hash anchor"):
                StairProtocolV2Draft.load(path, base_v1_path=V1_PATH)

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
