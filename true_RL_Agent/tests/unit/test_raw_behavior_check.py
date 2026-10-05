"""Hand-calculated expectations for the independent raw checker, without rendering."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.check_stair_raw_behavior import check_author, circular, episode_facts, read_raw


class RawBehaviorCheckTests(unittest.TestCase):
    def archive(self, root, *, unknown_positions=False, unknown_joint=False):
        # Two unequal episodes: translating with fixed body-relative feet, and stationary.
        p = np.zeros((3, 2, 3), np.float32)
        p[:, 0, 0] = [0, .1, .2]; p[:, 0, 1] = [0, .2, .4]
        q = np.zeros((3, 2, 4), np.float32); q[..., 0] = 1
        support = p[:, :, None]+np.array([[0, .1, -.2], [0, -.1, -.2]], np.float32)
        force = np.zeros_like(support)
        force[[0, 2], 0, 0, 2] = 2; force[:, 1, 1, 2] = 2
        valid = np.array([[1, 1], [1, 1], [1, 0]], bool)
        joint = np.full((3, 2, 1), .5, np.float32); joint[0, 0, 0] = 1.0856
        if unknown_positions:
            support.fill(np.nan)
        if unknown_joint:
            joint[1, 0, 0] = np.nan
        arrays = dict(base_pos_w=p, base_quat_w=q, support_pos_w=support,
                      support_contact_forces_w=force, valid=valid, joint_pos=joint,
                      episode_length=np.array([[1, 1], [2, 2], [3, 0]]), terrain_env_origins_w=np.zeros((2, 3)))
        chunk = root/"chunk.npz"
        np.savez(chunk, **arrays)
        manifest = {"identity": {"splits": ["public_dev"]}, "metadata": {
            "coordinates": {"base_quat_w": "wxyz", "positions": "world_m"},
            "runtime_projection": {"resolved_roles": {"support_effector": ["L_foot", "R_foot"]}},
            "semantics": {"episode_outcomes": {"signals": {"slip": {"contact_force_threshold": 1}}}},
            "measurement_contract": {"joint_position": {"joint_names": ["hip_joint"]}},
            "runtime_joint_position_limits": {"joint_names": ["hip_joint"], "values": {
                "joint_pos_limits": [[[0, 1]], [[0, 1]]], "soft_joint_pos_limits": [[[.1, .9]], [[.1, .9]]]}},
            "seeds": [{"run_index": 1, "step_dt": .25, "episode_steps": [3, 2], "outcomes": [
                {"primary_reason": "off_course", "evaluation_cell_id": "wide"},
                {"primary_reason": "success", "evaluation_cell_id": "wide"}]}]},
            "chunks": [{"file": chunk.name, "sha256": hashlib.sha256(chunk.read_bytes()).hexdigest(),
                        "metadata": {"run_index": 1, "step_start": 1}}]}
        path = root/"manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, arrays

    def test_hand_computed_episode_weighting_clock_and_real_joint_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.archive(Path(directory))
            raw = read_raw(path)
        group = raw["groups"]["all"]
        f = group["facts"]["all"]
        self.assertAlmostEqual(f["net"]["distribution"]["median"], .1)
        self.assertAlmostEqual(f["net_lateral"]["distribution"]["median"], .2)
        self.assertAlmostEqual(f["speed"]["distribution"]["median"], np.sqrt(.8)/2)
        self.assertAlmostEqual(f["robot_0.support.L_foot.relative"]["distribution"]["median"], 0)
        self.assertAlmostEqual(f["contact.2"]["distribution"]["mean"], 50)  # Not pooled 2/5.
        end = raw["groups"]["outcome:success"]["facts"]["terminal"]["contact.2"]
        self.assertEqual((end["valid"], end["unknown"], end["not_applicable"]), (0, 0, 2))
        self.assertEqual(group["low_progress_facet"]["not_applicable"], 4)
        runtime = raw["runtime_joint"]["all"]["facts"]["hard"]["hip_joint"]
        self.assertEqual((runtime["valid"], runtime["upper_outside"]), (5, 1))
        self.assertAlmostEqual(runtime["minimum_margin"], -.0856, places=6)
        self.assertEqual(raw["cells"]["wide"]["episodes"], 2)

    def test_gap_unknown_direction_degeneracy_and_first_observation(self):
        with tempfile.TemporaryDirectory() as directory:
            _, arrays = self.archive(Path(directory))
            a = {k: v[[0, 2], 0] for k, v in arrays.items() if k != "terrain_env_origins_w"}
        facts = episode_facts(a, np.array([1, 3]), .25, 3, [0, 0, 0], ["L_foot", "R_foot"], 1)
        self.assertIsNone(facts["presence"])
        self.assertIsNone(facts["facts"]["terminal"]["forward_speed"]["value"])
        self.assertEqual(facts["low_progress_facet"]["not_applicable"], 1)
        self.assertEqual(facts["low_progress_facet"]["unknown"], 1)
        a["base_quat_w"][1] = [0, 1, 0, 0]
        down = episode_facts(a, np.array([1, 3]), .25, 3, [0, 0, 0], ["L_foot", "R_foot"], 1)
        self.assertEqual(down["first_down_step"], 3)
        self.assertEqual(down["facts"]["all"]["first_down_time"]["value"], .75)
        self.assertEqual(down["facts"]["terminal"]["up_direction_end"]["not_applicable"], 1)
        self.assertIsNone(circular([90, -90])["circular_mean"])
        self.assertAlmostEqual(abs(circular([179, -179])["circular_mean"]), 180)

    def author_payload(self):
        # Literal values from the arithmetic above, not a production renderer.
        texts = {"scope": "本队列2回合，存档结局：偏离路线标签1、成功1。",
            "motion": "净正向位移0.1m；净横向位移0.2m；根部位置差分水平速度0.447m/s；反向位移累计0m；低进度且足相对身体变化占比0%（有效1/2回合，缺失0读数）；身体前轴投影朝向0°(R=1)；上轴向下0/2回合。",
            "outcome.off_course": "偏离路线标签1回合。", "outcome.success": "成功1回合。",
            "foot.0": "L：接触占比33.3%、世界点速0.447m/s、相对点速0m/s；相对(x,y,z)均值=0/0.1/-0.2m；末窗相对点速0m/s。",
            "foot.1": "R：接触占比50%、世界点速0.447m/s、相对点速0m/s；相对(x,y,z)均值=0/-0.1/-0.2m；末窗相对点速0m/s。",
            "contacts": "R50%；L33.3%；全部未触发16.7%。",
            "runtime_joint.hip_joint": "hip界外下/上：有效5点，hard 0/20%、soft 0/20%；最小余量(hard/soft)-0.0856/-0.1856rad。",
            "definitions": "力模>1N", "gaps": "初始不适用4、内部未知0点"}
        return {"public_feedback_summary": {"behavior": {"profile": {"report": {"process": [
            {"id": "claim:"+k, "text": v} for k, v in texts.items()]}}}}}

    def test_missing_and_swapped_author_quantities_are_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.archive(Path(directory))
            raw = read_raw(path)
        payload = self.author_payload()
        self.assertEqual(check_author(raw, payload), [])
        rows = payload["public_feedback_summary"]["behavior"]["profile"]["report"]["process"]
        wrong = deepcopy(payload)
        wrong_row = wrong["public_feedback_summary"]["behavior"]["profile"]["report"]["process"][1]
        wrong_row["text"] = wrong_row["text"].replace("净横向位移0.2m", "净横向位移0.447m")
        self.assertTrue(any("净横向位移" in e for e in check_author(raw, wrong)))
        rows[:] = [r for r in rows if r["id"] != "claim:outcome.success"]
        self.assertTrue(any("outcome.success: missing" in e for e in check_author(raw, payload)))

    def test_unknown_xyz_matches_each_column(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.archive(Path(directory), unknown_positions=True)
            raw = read_raw(path)
        payload = self.author_payload()
        feet = [r for r in payload["public_feedback_summary"]["behavior"]["profile"]["report"]["process"] if r["id"].startswith("claim:foot.")]
        for row in feet:
            row["text"] = row["text"].replace("世界点速0.447m/s", "世界点速未知m/s（有效0/2回合，缺失3读数）").replace(
                "、相对点速0m/s", "、相对点速未知m/s（有效0/2回合，缺失3读数）").replace("末窗相对点速0m/s", "末窗相对点速未知m/s")
            start, end = row["text"].split("相对(x,y,z)均值=")
            row["text"] = start+"相对(x,y,z)均值=未知/未知/未知m"+end.split("m", 1)[1]
        self.assertEqual(check_author(raw, payload), [])
        # One known operand among unknowns exercises column-specific matching.
        raw["groups"]["all"]["facts"]["all"]["robot_0.support.L_foot.y"]["distribution"]["median"] = .1
        feet[0]["text"] = feet[0]["text"].replace("未知/未知/未知m", "未知/0.1/未知m")
        self.assertEqual(check_author(raw, payload), [])
        feet[0]["text"] = feet[0]["text"].replace("未知/0.1/未知m", "0.1/未知/未知m")
        self.assertTrue(any("column" in e for e in check_author(raw, payload)))

    def test_runtime_missing_points_belong_to_selected_joint(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _ = self.archive(Path(directory), unknown_joint=True)
            raw = read_raw(path)
        self.assertEqual(raw["runtime_joint"]["all"]["facts"]["hard"]["hip_joint"]["unknown"], 1)
        payload = self.author_payload()
        rows = payload["public_feedback_summary"]["behavior"]["profile"]["report"]["process"]
        joint = next(r for r in rows if r["id"] == "claim:runtime_joint.hip_joint")
        good = joint["text"].replace("有效5点", "有效4点").replace("0/20%", "0/25%")+"缺测1点。"
        joint["text"] = good
        self.assertEqual(check_author(raw, payload), [])
        for replacement in ("缺测0点。", ""):
            joint["text"] = good.replace("缺测1点。", replacement)
            self.assertTrue(any("unknown q-point count" in e for e in check_author(raw, payload)))
        rows[0]["text"] += "缺测1点。"
        self.assertTrue(any("unknown q-point count" in e for e in check_author(raw, payload)))


if __name__ == "__main__":
    unittest.main()
