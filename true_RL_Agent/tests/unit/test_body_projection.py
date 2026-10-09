"""CPU checks of captured-matrix transport, not real rendered-image accuracy."""

import copy
import json
import unittest

import numpy as np

from true_rl_agent.evaluation.body_projection import project_snapshot_body_links


def snapshot_fixture(*, metres_per_unit=1.0):
    names = ["root", "part_a", "part_b", "behind", "plane", "outside"]
    # Matrices use row vectors; the camera looks along -Z. x_ndc=x/-z,
    # y_ndc=y/-z for this 90-degree pinhole fixture.
    projection = [[1., 0., 0., 0.], [0., 1., 0., 0.],
                  [0., 0., -1.02, -1.], [0., 0., -.202, 0.]]
    state = {"articulation_body_names": names,
             "body_link_pos_w": [[0., 0., -2.], [.5, .5, -2.], [-.5, -.5, -2.],
                                 [0., 0., 2.], [0., 0., 0.], [3., 0., -2.]],
             "sha256": "same-captured-state"}
    params = {"cameraModel": "pinhole", "renderProductResolution": [640, 480],
              "metersPerSceneUnit": metres_per_unit,
              "cameraViewTransform": np.eye(4).tolist(), "cameraProjection": projection}
    side_params = copy.deepcopy(params)
    # Actual camera translated +1 scene unit in world X -> world-to-view -1.
    side_params["cameraViewTransform"][3][0] = -1.
    return {"schema_version": "native_rgb_snapshots.v4", "resolution": [640, 480],
            "articulation_body_names": names,
            "state_conventions": {"body_link_pos_w": "link origins in world xyz, metres, in articulation_body_names order"},
            "samples": [{"control_step": 17, "state_before": state, "state_after": copy.deepcopy(state),
                         "views": [{"view_id": "v0", "camera_params": params},
                                   {"view_id": "v1", "camera_params": side_params}]}]}


class BodyProjectionTests(unittest.TestCase):
    def project(self, snapshot, names, view="v0"):
        return project_snapshot_body_links(snapshot, control_step=17, view_id=view, body_names=names)

    def test_coordinates_name_order_and_two_independent_views(self):
        snapshot = snapshot_fixture()
        original = copy.deepcopy(snapshot)
        first = self.project(snapshot, ["part_b", "root", "part_a"])
        self.assertEqual([row["body_name"] for row in first], ["part_b", "root", "part_a"])
        self.assertEqual([row["image_xy_px"] for row in first], [[240., 300.], [320., 240.], [400., 180.]])
        self.assertTrue(all(type(value) is float for row in first for value in row["image_xy_px"]))
        self.assertTrue(all(row["projection_status"] == "in_frame" for row in first))
        second = self.project(snapshot, ["root"], "v1")
        self.assertEqual(second[0], {"body_name": "root", "control_step": 17, "view_id": "v1",
                                     "image_xy_px": [160., 240.], "projection_status": "in_frame"})
        self.assertEqual(snapshot, original)
        self.assertNotIn("visible", json.dumps(first))

    def test_world_metres_convert_to_camera_scene_units(self):
        metres = snapshot_fixture()
        centimetres = snapshot_fixture(metres_per_unit=.01)
        # The same +1-metre camera translation is +100 in a centimetre stage.
        centimetres["samples"][0]["views"][1]["camera_params"]["cameraViewTransform"][3][0] = -100.
        self.assertEqual(self.project(metres, ["root", "part_a"], "v1"),
                         self.project(centimetres, ["root", "part_a"], "v1"))

    def test_behind_plane_and_outside_are_not_visible_claims(self):
        rows = self.project(snapshot_fixture(), ["behind", "plane", "outside"])
        self.assertEqual([row["projection_status"] for row in rows],
                         ["behind_camera", "on_camera_plane", "outside_frame"])
        self.assertEqual([row["image_xy_px"] for row in rows], [None, None, [800., 240.]])

    def test_old_missing_capture_returns_empty_but_v4_missing_data_fails(self):
        self.assertEqual(self.project({"schema_version": "native_rgb_snapshots.v3"}, ["root"]), [])
        snapshot = snapshot_fixture()
        del snapshot["samples"][0]["views"][0]["camera_params"]
        with self.assertRaisesRegex(ValueError, "captured pinhole"):
            self.project(snapshot, ["root"])

    def test_rejects_inconsistent_names_state_resolution_and_units(self):
        mutations = [
            (lambda s: s["samples"][0]["state_before"]["articulation_body_names"].reverse(), "link order"),
            (lambda s: s["samples"][0]["state_after"]["body_link_pos_w"][0].__setitem__(0, .01), "unchanged"),
            (lambda s: s["samples"][0]["views"][0]["camera_params"].__setitem__("metersPerSceneUnit", 0), "positive"),
            (lambda s: s["samples"][0]["views"][0]["camera_params"].__setitem__("renderProductResolution", [320, 240]), "resolution"),
            (lambda s: s["samples"][0]["views"][0]["camera_params"].__setitem__("cameraViewTransform", [1., 2.]), "cameraViewTransform"),
        ]
        for mutate, error in mutations:
            with self.subTest(error=error):
                snapshot = snapshot_fixture()
                mutate(snapshot)
                with self.assertRaisesRegex(ValueError, error):
                    self.project(snapshot, ["root"])
        with self.assertRaisesRegex(ValueError, "requested names"):
            self.project(snapshot_fixture(), ["unknown_link"])


if __name__ == "__main__":
    unittest.main()
