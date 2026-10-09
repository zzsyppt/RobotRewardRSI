"""Semantic swaps must fail even after claims and record IDs are regenerated."""
from copy import deepcopy
import unittest

from tests.unit.test_push_trajectory import episode
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.tasks.push_recovery.trajectory_card import (
    _render_claims, build_push_card, project_push_card,
)
from true_rl_agent.tasks.push_recovery.trajectory_measurement import CONFIG, measure_push_episode


def resign(card):
    card["claims"] = _render_claims(card["domain"], card["cohorts"], card["examples"],
                                   card["selection"], card["parameters"], card["timing"])
    core = {k: v for k, v in card.items() if k not in {"card_id", "public_report", "projection_audit"}}
    card["card_id"] = stable_id("push-card", core)


def reidentify(record):
    record["record_id"] = stable_id("push-evidence", record["value"]["episode_id"],
                                    record["measurement_config"], record["value"])


def swap(mapping, left, right):
    mapping[left], mapping[right] = mapping[right], mapping[left]


class PushTrajectorySemanticTests(unittest.TestCase):
    def setUp(self):
        self.episodes = [episode(env=i) for i in range(4)]
        self.records = [measure_push_episode(e) for e in self.episodes]
        self.card = build_push_card(self.records, "fixture", episodes=self.episodes)

    def project(self, card, episodes=None):
        return project_push_card(card, episodes=self.episodes if episodes is None else episodes,
                                 source_id="fixture")

    def test_resigned_card_operand_swaps_are_rejected(self):
        for case in ("metric", "unit", "window", "direction_source", "protocol", "source_id"):
            with self.subTest(case=case):
                card = deepcopy(self.card)
                event = card["examples"][0]["value"]["events"][0]
                if case == "metric":
                    swap(card["cohorts"]["all"]["pose"], "height_m", "planar_speed_mps")
                elif case == "unit":
                    event["vector_unit"] = "N"
                elif case == "window":
                    swap(event["windows"], "pre", "post")
                elif case == "direction_source":
                    event["root_along_disturbance"]["evidence_ref"]["chunks"][0]["env_index"] = 99
                elif case == "protocol":
                    card["domain"]["resolved_protocol_hash"] = "other-protocol"
                else:
                    card["domain"]["source_id"] = "other-source"
                resign(card)
                if case == "unit":
                    text = next(c["text"] for c in card["claims"] if c["key"].startswith("event:"))
                    self.assertIn("世界向量[1,0]m/s", text)
                with self.assertRaisesRegex(ValueError, "semantic operands"):
                    self.project(card)

    def test_reidentified_measurement_swaps_are_rejected_before_aggregation(self):
        for case in ("metric", "window", "direction_source"):
            with self.subTest(case=case):
                records = deepcopy(self.records)
                v = records[0]["value"]
                if case == "metric":
                    swap(v["pose"], "height_m", "planar_speed_mps")
                elif case == "window":
                    swap(v["events"][0]["windows"], "pre", "post")
                else:
                    v["events"][0]["root_along_disturbance"]["evidence_ref"]["chunks"][0]["env_index"] = 99
                reidentify(records[0])
                with self.assertRaisesRegex(ValueError, "measurement operands"):
                    build_push_card(records, "fixture", episodes=self.episodes)

    def test_unselected_episode_is_still_bound(self):
        selected = {r["record_id"] for r in self.card["examples"]}
        records = deepcopy(self.records)
        record = next(r for r in records if r["record_id"] not in selected)
        record["value"]["pose"]["height_m"]["mean"] = 123
        reidentify(record)
        with self.assertRaisesRegex(ValueError, "measurement operands"):
            build_push_card(records, "fixture", episodes=self.episodes)

    def test_complete_episode_set_is_required(self):
        for records in (self.records[:-1], self.records + [self.records[0]],
                        [self.records[0]] * len(self.records)):
            with self.subTest(count=len(records)), self.assertRaises(ValueError):
                build_push_card(records, "fixture", episodes=self.episodes)

    def test_card_mutations_do_not_modify_raw_context_or_global_config(self):
        original_task = deepcopy(self.episodes[0].task)
        original_source = deepcopy(self.episodes[0].source)
        original_config = dict(CONFIG)
        for key in ("parameters", "timing", "config", "source"):
            records = [measure_push_episode(e) for e in self.episodes]
            card = build_push_card(records, "fixture", episodes=self.episodes)
            if key == "parameters":
                card[key]["recovery_height_m"] = 99
            elif key == "timing":
                card[key]["required_dwell_steps"] = 99
            elif key == "config":
                card[key]["pre_window_s"] = 99
            else:
                card["examples"][0]["value"]["source"]["capabilities"]["root_signed_xy"] = "not_recorded"
            resign(card)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "semantic operands"):
                self.project(card)
            self.assertEqual(self.episodes[0].task, original_task)
            self.assertEqual(self.episodes[0].source, original_source)
            self.assertEqual(CONFIG, original_config)
        self.records[0]["measurement_config"]["pre_window_s"] = 99
        self.assertEqual(CONFIG, original_config)

    def test_legal_signed_axis_change_requires_matching_raw_operands(self):
        changed = deepcopy(self.episodes)
        for e in changed:
            e.arrays["planar_motion"] = e.arrays["planar_motion"][:, [1, 0, 3, 2]]
        with self.assertRaisesRegex(ValueError, "semantic operands"):
            self.project(self.card, changed)
        records = [measure_push_episode(e) for e in changed]
        card = build_push_card(records, "fixture", episodes=changed)
        self.assertEqual(self.project(card, changed), card["public_report"])
        self.assertEqual(card["cohorts"]["all"]["pose"], self.card["cohorts"]["all"]["pose"])
        self.assertEqual(card["examples"][0]["value"]["events"][0]["root_along_disturbance"]["mean_velocity_mps"], 0)

    def test_v1_cannot_gain_signed_direction_by_reidentification(self):
        e = episode(signed=False)
        record = measure_push_episode(e)
        card = build_push_card([record], "fixture", episodes=[e])
        card["domain"]["root_signed_xy"] = "recorded_world_xy"
        resign(card)
        with self.assertRaisesRegex(ValueError, "semantic operands"):
            self.project(card, [e])


if __name__ == "__main__":
    unittest.main()
