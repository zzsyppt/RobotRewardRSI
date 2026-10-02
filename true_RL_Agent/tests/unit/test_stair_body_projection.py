"""Asset-owned image identity; projection is not a visibility judgment."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import struct
import unittest

from test_body_projection import snapshot_fixture
from true_rl_agent.evaluation.visual_observer import MediaImage, MediaWindow, prepare_observer_request
from true_rl_agent.tasks.stair_climbing.visual_observation import bind_stair_root_frame, stair_body_image_points


class StairBodyProjectionTests(unittest.TestCase):
    def setUp(self):
        configs = Path(__file__).resolve().parents[2] / "configs/robots"
        self.spec = json.loads((configs / "unitree_go2_12dof_isaaclab_v1.body_frame.json").read_bytes())
        self.profile = json.loads((configs / "unitree_go2_12dof_isaaclab_v1.json").read_bytes())
        self.projection = {key: self.spec[key] for key in (
            "robot_profile_id", "robot_profile_hash", "kinematic_base_source")}
        self.projection["resolved_roles"] = {"base": ["base"]}
        self.binding = self.bind(self.spec)
        names = self.profile["runtime_facts"]["body_names"]
        self.snapshot = snapshot_fixture()
        self.snapshot.update(articulation_body_names=names, env_index=2, seed=17)
        for key in ("state_before", "state_after"):
            self.snapshot["samples"][0][key].update(
                articulation_body_names=names, body_link_pos_w=[[i * .05, 0, -2] for i in range(len(names))])
        identity = {key: key for key in ("session_id", "task_id", "protocol_id", "source_id",
                                        "checkpoint_id", "episode_id", "run_id")}
        identity.update(robot_id=self.spec["robot_profile_id"], env_index=2, runner_seed=17)
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + struct.pack(">II", 640, 480)
        self.window = MediaWindow("e1", "camera", "layout", "roi", tuple(names),
            tuple(MediaImage(png, 640, 480, (17,), view) for view in ("v0", "v1")), identity=identity)

    def bind(self, spec):
        return bind_stair_root_frame(spec, runtime_projection=self.projection, robot_profile=self.profile)

    def test_named_identity_survives_order_and_public_serialization(self):
        points = stair_body_image_points(self.window, self.snapshot, root_frame=self.binding)
        self.assertEqual(len(points), 8)
        self.assertEqual([p["body_name"] for p in points[:4]], ["FR_calf", "FL_calf", "RR_calf", "RL_calf"])
        reordered = deepcopy(self.snapshot)
        reordered["articulation_body_names"] = list(reversed(reordered["articulation_body_names"]))
        for key in ("state_before", "state_after"):
            state = reordered["samples"][0][key]
            state["articulation_body_names"] = list(reversed(state["articulation_body_names"]))
            state["body_link_pos_w"].reverse()
        self.assertEqual(points, stair_body_image_points(self.window, reordered, root_frame=self.binding))
        request = prepare_observer_request([replace(self.window, body_image_points=points)], model="offline-fixture")
        public = json.loads(request["payload"]["messages"][1]["content"][0]["text"])
        self.assertEqual(len(public["body_image_points"]), 8)
        self.assertEqual(public["body_image_points"][0]["body"], "FR_calf")
        self.assertNotIn("visible", json.dumps(points))
        self.assertTrue(request["audit"]["passed"])

    def test_missing_old_evidence_stays_empty_and_wrong_binding_is_rejected(self):
        self.assertEqual(stair_body_image_points(self.window, self.snapshot, root_frame=None), ())
        old_spec = {k: v for k, v in self.spec.items() if k != "image_body_names"}
        self.assertEqual(stair_body_image_points(self.window, self.snapshot, root_frame=self.bind(old_spec)), ())
        self.assertEqual(stair_body_image_points(self.window, {"schema_version": "native_rgb_snapshots.v3"},
                                                root_frame=self.binding), ())
        with self.assertRaisesRegex(ValueError, "bound window"):
            stair_body_image_points(self.window, {**self.snapshot, "env_index": 3}, root_frame=self.binding)
        with self.assertRaisesRegex(ValueError, "bound asset"):
            self.bind({**self.spec, "image_body_names": ["absent_link"]})


if __name__ == "__main__":
    unittest.main()
