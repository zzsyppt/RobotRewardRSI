"""Named, asset-bound robot-link angles from synthetic native snapshot positions."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import struct
import unittest

import numpy as np

from true_rl_agent.evaluation.visual_observer import MediaImage, MediaWindow
from true_rl_agent.robots.base import RobotProfile
from true_rl_agent.tasks.push_recovery.body_geometry import (
    bind_push_body_anchors, push_body_geometry_definitions, push_body_geometry_facts, push_body_geometry_s0,
    push_body_image_points,
)

ROOT = Path(__file__).resolve().parents[2]


class PushBodyGeometryTests(unittest.TestCase):
    def setUp(self):
        path = ROOT / "configs/robots/unitree_g1_37dof_isaaclab_v2.json"
        self.profile = json.loads(path.read_text())
        self.spec = json.loads(path.with_suffix(".body_anchors.json").read_text())
        profile = RobotProfile.from_mapping(self.profile)
        self.names = profile.runtime_facts["body_names"]
        self.runtime = {"robot_profile_id": profile.robot_profile_id,
                        "robot_profile_hash": profile.profile_hash, "body_names": self.names}
        self.binding = bind_push_body_anchors(self.spec, runtime_projection=self.runtime, robot_profile=self.profile)
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 1, 1)
        self.window = MediaWindow("e1", "camera", "layout", "roi", tuple(self.names),
            (MediaImage(png, 1, 1, (1, 3), "v0"),),
            identity={"robot_id": profile.robot_profile_id, "env_index": 2, "runner_seed": 123})

    def snapshot(self):
        positions = np.zeros((len(self.names), 3))
        # Left H-K-A is straight at K, right is a right angle. Different sides
        # deliberately have different geometry regardless of joint coordinates.
        points = {"left": ([0, 0, 1], [0, 0, 0], [0, 0, -1]),
                  "right": ([0, 1, 1], [0, 1, 0], [1, 1, 0])}
        for side, values in points.items():
            for point, xyz in zip(("H", "K", "A"), values):
                positions[self.names.index(self.binding["legs"][side][point]["body_name"])] = xyz
        state = {"body_link_pos_w": positions.tolist(), "joint_pos": [999.] * 37,
                 "articulation_body_names": list(self.names)}
        return {"schema_version": "native_rgb_snapshots.v4", "env_index": 2, "seed": 123,
                "articulation_body_names": list(self.names), "state_conventions": {
                    "body_link_pos_w": "link origins in world xyz, metres, in articulation_body_names order"},
                "samples": [{"control_step": step, "state_before": deepcopy(state), "state_after": deepcopy(state)}
                            for step in (1, 2, 3)]}

    def measure(self, snapshot):
        return push_body_geometry_s0(self.window, snapshot, binding=self.binding)

    def test_geometry_uses_only_displayed_positions_and_keeps_sides_independent(self):
        scalars = self.measure(self.snapshot())
        self.assertEqual({(row["name"], row["step"]): row["value"] for row in scalars},
            {(side + "_hka_geometry_deg", step): angle
             for side, angle in (("left", 180.), ("right", 90.)) for step in (1, 3)})
        media = replace(self.window, s0=scalars, s0_definitions=push_body_geometry_definitions(self.binding))
        facts = push_body_geometry_facts(media, binding=self.binding)
        self.assertEqual([f["body"] for f in facts], ["left_knee_link", "right_knee_link"])
        self.assertIn("1: 180.0; 3: 180.0", facts[0]["text"])
        self.assertIn("not anatomical", facts[0]["text"])
        self.assertEqual([ref["step"] for ref in facts[1]["scalar_refs"]], [1, 3])

    def test_name_order_and_world_rigid_transform_do_not_change_angles(self):
        snapshot = self.snapshot()
        snapshot["articulation_body_names"].reverse()
        rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
        for sample in snapshot["samples"]:
            for key in ("state_before", "state_after"):
                points = np.asarray(sample[key]["body_link_pos_w"])[::-1]
                sample[key]["body_link_pos_w"] = (points @ rotation.T + [12, -7, 4]).tolist()
                sample[key]["articulation_body_names"].reverse()
        self.assertEqual(self.measure(snapshot), self.measure(self.snapshot()))

    def test_missing_old_snapshot_or_unbound_spec_does_not_invent_evidence(self):
        old = self.snapshot()
        old["schema_version"] = "native_rgb_snapshots.v3"
        for sample in old["samples"]:
            for key in ("state_before", "state_after"):
                sample[key].pop("body_link_pos_w")
        self.assertEqual(self.measure(old), ())
        self.assertEqual(push_body_geometry_s0(self.window, self.snapshot(), binding=None), ())
        self.assertEqual(push_body_geometry_definitions(None), ())
        self.assertEqual(push_body_geometry_facts(self.window, binding=self.binding), ())

    def test_asset_profile_runtime_and_nonzero_offset_mismatches_are_rejected(self):
        for mutation in ("asset", "profile", "offset", "runtime_names"):
            spec, runtime = deepcopy(self.spec), deepcopy(self.runtime)
            if mutation == "asset":
                spec["root_asset"]["sha256"] = "0" * 64
            elif mutation == "profile":
                runtime["robot_profile_hash"] = "0" * 64
            elif mutation == "offset":
                spec["legs"]["left"]["K"]["child_local_position_m"] = [0, 0, .1]
            else:
                runtime["body_names"].pop()
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bind_push_body_anchors(spec, runtime_projection=runtime, robot_profile=self.profile)

    def test_zero_length_changed_state_or_ambiguous_names_do_not_produce_angles(self):
        for mutation in ("zero_length", "changed_state", "duplicate_name", "state_name_order", "wrong_episode"):
            snapshot = self.snapshot()
            if mutation == "zero_length":
                for key in ("state_before", "state_after"):
                    snapshot["samples"][0][key]["body_link_pos_w"][self.names.index("left_hip_yaw_link")] = [0, 0, 0]
            elif mutation == "changed_state":
                snapshot["samples"][0]["state_after"]["body_link_pos_w"][0][0] = 1
            elif mutation == "duplicate_name":
                snapshot["articulation_body_names"][0] = snapshot["articulation_body_names"][1]
            elif mutation == "state_name_order":
                snapshot["samples"][0]["state_before"]["articulation_body_names"].reverse()
            else:
                snapshot["env_index"] = 3
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.measure(snapshot)

    def test_palm_origin_is_bound_for_localization_without_centre_or_angle_claim(self):
        self.assertEqual(self.binding["palm_origins"]["left"]["body_name"], "left_palm_link")
        self.assertEqual(self.binding["palm_origins"]["right"]["child_local_position_m"], [0, 0, 0])
        self.assertTrue(all("palm" not in row["name"] for row in self.measure(self.snapshot())))

    def test_image_points_select_six_named_anchors_only_for_displayed_pairs(self):
        snapshot = self.snapshot()
        snapshot["resolution"] = [1, 1]
        for sample in snapshot["samples"]:
            for key in ("state_before", "state_after"):
                sample[key]["sha256"] = "synthetic-identical-state"
            sample["views"] = [{"view_id": "v0", "camera_params": {
                "cameraModel": "pinhole", "renderProductResolution": [1, 1], "metersPerSceneUnit": 1,
                "cameraViewTransform": np.eye(4).tolist(), "cameraProjection": np.eye(4).tolist()}}]
        points = push_body_image_points(self.window, snapshot, binding=self.binding)
        self.assertEqual(len(points), 12)
        self.assertEqual({(row["control_step"], row["view_id"]) for row in points}, {(1, "v0"), (3, "v0")})
        self.assertEqual({row["body_name"] for row in points}, {
            f"{side}_{part}_link" for side in ("left", "right") for part in ("hip_yaw", "knee", "palm")})
        self.assertEqual(points[0]["image_xy_px"], [.5, .5])
        self.assertEqual(push_body_image_points(self.window, snapshot, binding=None), ())
        old = {**snapshot, "schema_version": "native_rgb_snapshots.v3"}
        self.assertEqual(push_body_image_points(self.window, old, binding=self.binding), ())
        with self.assertRaises(ValueError):
            push_body_image_points(self.window, {**snapshot, "env_index": 3}, binding=self.binding)


if __name__ == "__main__":
    unittest.main()
