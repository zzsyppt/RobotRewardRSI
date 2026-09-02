from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.json_contract import load_strict_json_object  # noqa: E402
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    STAIR_PROTOCOL_V2_RELATIVE_PATH,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.pilot_artifact_store import (  # noqa: E402
    validate_stair_phase_d_pilot_artifact_manifest,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (  # noqa: E402
    EXPECTED_STAIR_PROTOCOL_V2_HASH,
    EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH,
    STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH,
    StairProtocolV2,
    validate_stair_protocol_v2_freeze_candidate,
)
from true_rl_agent.tasks.stair_climbing.training_distribution import (  # noqa: E402
    resolved_stair_training_distribution,
)

FINAL_PATH = ROOT / "configs" / "tasks" / "stair_protocol_v2.json"
MANIFEST_PATH = ROOT / STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH


class StairProtocolV2Tests(unittest.TestCase):
    def test_final_protocol_is_the_only_production_default(self) -> None:
        protocol = StairProtocolV2.load(FINAL_PATH)
        config = StairClimbingTaskConfig.defaults(FINAL_PATH)

        self.assertEqual(STAIR_PROTOCOL_V2_RELATIVE_PATH, "configs/tasks/stair_protocol_v2.json")
        self.assertEqual(protocol.protocol_hash, EXPECTED_STAIR_PROTOCOL_V2_HASH)
        self.assertEqual(
            protocol.seed_anchor_hash,
            EXPECTED_STAIR_PROTOCOL_V2_PILOT_DRAFT_HASH,
        )
        self.assertEqual(protocol.fixed_budget_steps, 196_608_000)
        self.assertEqual(config.budget_steps, 196_608_000)
        self.assertEqual(config.train_max_iterations, 2_000)
        self.assertEqual(config.train_profile, "stair_v2_train")
        self.assertEqual(config.eval_profile, "stair_v2_eval")

    def test_final_lineage_is_bound_to_full_repository_manifest(self) -> None:
        final = StairProtocolV2.load(FINAL_PATH).to_public_dict()
        manifest = validate_stair_phase_d_pilot_artifact_manifest(
            manifest=load_strict_json_object(
                MANIFEST_PATH,
                context="test Stair V2 pilot artifact manifest",
                require_canonical=True,
                trailing_newline=True,
            )
        )
        lineage = final["pilot_lineage"]
        self.assertEqual(
            lineage["pilot_artifact_manifest_repository_path"],
            STAIR_PROTOCOL_V2_PILOT_ARTIFACT_MANIFEST_RELATIVE_PATH,
        )
        self.assertEqual(
            lineage["draft_protocol_hash"],
            manifest["draft_protocol_hash"],
        )
        self.assertEqual(
            lineage["phase_d_report_hash"],
            manifest["phase_d_report_hash"],
        )
        self.assertEqual(lineage["pilot_artifact_uri"], manifest["artifact_uri"])
        self.assertEqual(
            lineage["pilot_artifact_content_hash"], manifest["content_hash"]
        )
        self.assertEqual(
            lineage["pilot_artifact_manifest_hash"], manifest["manifest_hash"]
        )

    def test_final_training_distribution_covers_all_but_the_ood_frontier(self) -> None:
        config = StairClimbingTaskConfig.defaults(FINAL_PATH)
        contract = config.protocol_v2.runner_owned_training_distribution
        resolved = resolved_stair_training_distribution(
            contract,
            base_protocol=config.protocol_v1.to_dict(),
        )

        self.assertEqual(
            resolved["terrain"]["step_height_range_m"],
            [0.05, 0.35],
        )
        self.assertTrue(contract["formal_reference_and_candidates_identical"])
        self.assertFalse(contract["historical_pilot_distribution_identical"])
        self.assertFalse(contract["candidate_writable"])
        self.assertFalse(contract["prompt_visible"])
        self.assertEqual(
            [
                float(item["height_m"])
                for item in config.protocol_v1.eval_bins
                if float(item["height_m"])
                > resolved["terrain"]["step_height_range_m"][1]
            ],
            [0.4],
        )

    def test_freeze_candidate_rejects_rehashed_manifest_lineage_forgery(self) -> None:
        forged = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
        forged["pilot_lineage"]["pilot_artifact_manifest_hash"] = "1" * 64
        forged["protocol_hash"] = canonical_json_hash(
            {key: value for key, value in forged.items() if key != "protocol_hash"}
        )

        with self.assertRaisesRegex(ValueError, "artifact manifest lineage mismatch"):
            with self._temporary_protocol(forged) as path:
                validate_stair_protocol_v2_freeze_candidate(
                    path,
                    base_v1_path=FINAL_PATH.with_name("stair_protocol_v1.json"),
                )

    def test_rehashed_semantic_mutations_are_rejected_by_trust_anchor(self) -> None:
        source = json.loads(FINAL_PATH.read_text(encoding="utf-8"))
        mutations = (
            lambda value: value["production_training"].__setitem__(
                "budget_steps", 98_304_000
            ),
            lambda value: value["production_training"].__setitem__(
                "train_profile", "stair_v2_weakened"
            ),
            lambda value: value["formal_protocol"].__setitem__(
                "acceptance_protocol_hash", "1" * 64
            ),
            lambda value: value["reference_capability_gate"].__setitem__(
                "minimum_macro_mean_success_rate", 0.001
            ),
            lambda value: value["complete_reward_runtime"]["output_contract"].__setitem__(
                "maximum_absolute_total_rate", 1_000_000.0
            ),
        )
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                forged = copy.deepcopy(source)
                mutate(forged)
                forged["protocol_hash"] = canonical_json_hash(
                    {key: value for key, value in forged.items() if key != "protocol_hash"}
                )
                with self.assertRaisesRegex(ValueError, "trusted hash anchor"):
                    with self._temporary_protocol(forged) as path:
                        StairProtocolV2.load(
                            path,
                            base_v1_path=FINAL_PATH.with_name("stair_protocol_v1.json"),
                        )

    def _temporary_protocol(self, payload):
        import tempfile
        from contextlib import contextmanager

        @contextmanager
        def managed():
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "stair_protocol_v2.json"
                path.write_text(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                yield path

        return managed()


if __name__ == "__main__":
    unittest.main()
