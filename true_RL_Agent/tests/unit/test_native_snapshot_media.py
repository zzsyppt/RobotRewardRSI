"""Native capture bytes and step/view labels must survive observer transport."""

import base64
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from PLAN.feedback_sys.P3NEW import extract_dual_view_evidence as exporter
from true_rl_agent.evaluation.visual_observer import (
    IDENTITY_FIELDS, MediaWindow, prepare_observer_request, read_native_snapshot_media,
)
from true_rl_agent.runner.native_video import NativeVideoOptions, NativeVideoRecorder
from tests.unit.test_native_video import camera_params_fixture


class NativeSnapshotMediaTests(unittest.TestCase):
    def test_capture_manifest_to_actual_request_keeps_bytes_and_only_recorded_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            recorder = object.__new__(NativeVideoRecorder)
            recorder.options = NativeVideoOptions(output, resolution=(4, 2), snapshot_interval_steps=1,
                secondary_eye=(1., 1., 1.), secondary_lookat=(0., 0., 0.))
            recorder._name, recorder._seed = "private-run", 123
            recorder._recording, recorder._control_step = True, 0
            recorder._frames, recorder.records = [], []
            recorder._imageables = ()
            recorder.control_dt_s = .02
            recorder._camera_render_product = SimpleNamespace(path="/Render/ActualRgb")
            camera_params = camera_params_fixture()
            camera_params["renderProductResolution"] = [4, 2]
            recorder._camera_params_annotator = SimpleNamespace(get_data=lambda: camera_params)
            recorder.env = SimpleNamespace(unwrapped=SimpleNamespace(scene={
                "robot": SimpleNamespace(joint_names=["joint_b", "joint_a"], body_names=["root", "foot"])},
                video_recorder=SimpleNamespace(_capture=SimpleNamespace(_render_product=recorder._camera_render_product)),
                viewport_camera_controller=SimpleNamespace(update_view_to_asset_root=lambda _: None,
                    set_view_env_index=lambda _: None,
                    update_view_location=lambda **_: None)),
                render=lambda: np.full((2, 4, 3), recorder._control_step, dtype=np.uint8))
            state = {"root_pos_w": [0., 0., 1.], "joint_pos": [.2, .1], "physics_step_count": 10,
                     "articulation_body_names": ["root", "foot"]}
            recorder._snapshots = []
            with patch("true_rl_agent.runner.native_video._selected_state", return_value=state):
                for _ in range(3):
                    recorder.record_step()
            record = recorder.finish(frame_count=2)
            path = output / record["file"]
            manifest, images = read_native_snapshot_media(path, manifest_sha256=record["sha256"])
            self.assertEqual(manifest["articulation_joint_names"], ["joint_b", "joint_a"])
            self.assertEqual(manifest["snapshot_interval_steps"], 1)
            self.assertEqual([(image.displayed_steps, image.view_id) for image in images],
                             [((1,), "v0"), ((1,), "v1"), ((2,), "v0"), ((2,), "v1")])
            full_manifest, selected = read_native_snapshot_media(
                path, manifest_sha256=record["sha256"], selected_steps=(2,))
            self.assertEqual(full_manifest, manifest)
            self.assertEqual(selected, images[2:])
            with self.assertRaisesRegex(ValueError, "not captured"):
                read_native_snapshot_media(path, manifest_sha256=record["sha256"], selected_steps=(3,))
            window = MediaWindow("e1", "asset_root_tracking.v1", "native_frame.v1", "full_frame.v1",
                                 ("root",), images, identity={key: "private-id" for key in IDENTITY_FIELDS})
            prepared = prepare_observer_request((window,), model="test-model")
            content = prepared["payload"]["messages"][1]["content"]
            transported = [base64.b64decode(item["image_url"]["url"].split(",", 1)[1])
                           for item in content if item["type"] == "image_url"]
            self.assertEqual(transported, [image.data for image in images])
            labels = [json.loads(item["text"]) for item in content
                      if item["type"] == "text" and "view_id" in json.loads(item["text"])]
            self.assertEqual(labels, [
                {"alias": "e1", "view_id": "v0", "displayed_steps": [1], "image_size_px": [4, 2]},
                {"alias": "e1", "view_id": "v1", "displayed_steps": [1], "image_size_px": [4, 2]},
                {"alias": "e1", "view_id": "v0", "displayed_steps": [2], "image_size_px": [4, 2]},
                {"alias": "e1", "view_id": "v1", "displayed_steps": [2], "image_size_px": [4, 2]},
            ])
            for label, png in zip(labels, transported):
                self.assertEqual(label["image_size_px"], list(struct.unpack(">II", png[16:24])))
            # Both export modes preserve the existing manifest-relative reader
            # and the actual observer payload; only PNG inode ownership differs.
            for storage in ("copy", "hardlink"):
                exported = output / storage
                manifest_copy = exported / path.name
                exporter.copy_bytes(path, manifest_copy, record["sha256"])
                self.assertFalse(path.samefile(manifest_copy))
                for sample in manifest["samples"]:
                    for view in sample["views"]:
                        source, destination = output / view["file"], exported / view["file"]
                        options = {} if storage == "copy" else {"storage": storage}
                        receipt = exporter.copy_png(source, destination, view["sha256"], **options)
                        self.assertEqual(receipt["sha256"], exporter.sha(source.read_bytes()))
                        self.assertEqual(destination.read_bytes(), source.read_bytes())
                        self.assertEqual(source.samefile(destination), storage == "hardlink")
                        self.assertTrue(destination.resolve().is_relative_to(exported.resolve()))
                        if storage == "hardlink":
                            self.assertTrue(receipt["shared_inode"])
                            self.assertFalse(receipt["independent_backup"])
                        else:
                            self.assertNotIn("storage", receipt)
                exported_manifest, exported_images = read_native_snapshot_media(
                    manifest_copy, manifest_sha256=record["sha256"])
                self.assertEqual((exported_manifest, exported_images), (manifest, images))
                exported_window = MediaWindow("e1", "asset_root_tracking.v1", "native_frame.v1", "full_frame.v1",
                    ("root",), exported_images, identity={key: "private-id" for key in IDENTITY_FIELDS})
                self.assertEqual(prepare_observer_request((exported_window,), model="test-model")["payload"],
                                 prepared["payload"])
            view = manifest["samples"][0]["views"][0]
            failed_link = output / "no-fallback" / view["file"]
            with patch.object(exporter.os, "link", side_effect=OSError("different filesystem")):
                with self.assertRaisesRegex(OSError, "different filesystem"):
                    exporter.copy_png(output / view["file"], failed_link, view["sha256"], storage="hardlink")
            self.assertFalse(failed_link.exists())
            self.assertNotIn("private-run", json.dumps(prepared["payload"]))
            self.assertNotIn("private-id", json.dumps(prepared["payload"]))
            image_path = output / manifest["samples"][0]["views"][0]["file"]
            image_path.write_bytes(image_path.read_bytes() + b"changed")
            # A selected read checks only selected image bytes, not the whole archive.
            self.assertEqual(read_native_snapshot_media(
                path, manifest_sha256=record["sha256"], selected_steps=(2,))[1], selected)
            with self.assertRaisesRegex(ValueError, "image hash differs"):
                read_native_snapshot_media(path, manifest_sha256=record["sha256"])


if __name__ == "__main__":
    unittest.main()
