"""Native image order must survive real encoding, including awkward durations."""

import subprocess
import argparse
import json
import zlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from true_rl_agent.runner.native_video import (
    NativeVideoOptions, NativeVideoRecorder, _camera_params_data, _encode_frames, _raw_rgb_png, _selected_state,
    add_native_video_arguments, native_video_argv_from_args, native_video_options_from_args,
)


def camera_params_fixture():
    import numpy as np

    return {"cameraModel": "pinhole", "renderProductResolution": np.array([2, 2]),
            "cameraViewTransform": np.eye(4).reshape(-1), "cameraProjection": np.eye(4),
            "cameraAperture": np.array([20.95, 15.29]), "cameraApertureOffset": np.zeros(2),
            "cameraNearFar": np.array([0.1, 1000.]), "cameraFocalLength": 24.,
            "metersPerSceneUnit": 1.}


class NativeVideoEncodingTests(unittest.TestCase):
    def test_camera_cli_preserves_defaults_and_tracks_asset_root(self) -> None:
        parser = argparse.ArgumentParser()
        add_native_video_arguments(parser)
        defaults = parser.parse_args(["--video-dir", "frames"])
        old = native_video_options_from_args(defaults)
        self.assertEqual(old.eye, (4.5, -4.5, 3.0))
        self.assertEqual(old.lookat, (0.8, 0.0, 0.0))
        self.assertEqual(old.resolution, (1280, 720))
        args = parser.parse_args(["--video-dir", "frames", "--video-env-index", "3",
            "--video-camera-eye", "2", "-2", "1.5", "--video-camera-lookat", "0", "0", "0.5",
            "--video-resolution", "640", "480"])
        options = native_video_options_from_args(args)
        self.assertEqual((options.eye, options.lookat, options.resolution),
                         ((2.0, -2.0, 1.5), (0.0, 0.0, 0.5), (640, 480)))
        cfg = SimpleNamespace(scene=SimpleNamespace(num_envs=4), viewer=SimpleNamespace(),
                              video_recorder=SimpleNamespace())
        options.configure_viewer(cfg)
        self.assertEqual((cfg.viewer.origin_type, cfg.viewer.asset_name, cfg.viewer.env_index),
                         ("asset_root", "robot", 3))
        self.assertEqual(cfg.viewer.resolution, (640, 480))
        self.assertEqual((cfg.video_recorder.window_width, cfg.video_recorder.window_height), (640, 480))
        hd = NativeVideoOptions(Path("."), env_index=3, resolution=(1920, 1080))
        hd.configure_viewer(cfg)
        self.assertEqual(cfg.viewer.resolution, (1920, 1080))
        self.assertEqual((cfg.video_recorder.window_width, cfg.video_recorder.window_height), (1920, 1080))
        with self.assertRaisesRegex(ValueError, "requires cfg.video_recorder"):
            options.configure_viewer(SimpleNamespace(scene=SimpleNamespace(num_envs=4),
                                                     viewer=SimpleNamespace()))
        self.assertEqual(native_video_argv_from_args(args)[-3:], ["--video-resolution", "640", "480"])
        with self.assertRaisesRegex(ValueError, "positive even"):
            native_video_options_from_args(parser.parse_args(["--video-dir", "frames", "--video-resolution", "641", "480"]))
        snapshots = parser.parse_args(["--video-dir", "frames", "--video-snapshot-steps", "1", "9",
            "--video-secondary-camera-eye", "-1", "-2", "1",
            "--video-secondary-camera-lookat", "0", "0", "-0.1"])
        self.assertEqual(native_video_options_from_args(snapshots).snapshot_steps, (1, 9))
        self.assertIn("--video-secondary-camera-eye", native_video_argv_from_args(snapshots))
        periodic = parser.parse_args(["--video-dir", "frames", "--video-snapshot-interval-steps", "1",
            "--video-secondary-camera-eye", "-1", "-2", "1",
            "--video-secondary-camera-lookat", "0", "0", "0"])
        self.assertEqual(native_video_options_from_args(periodic).snapshot_interval_steps, 1)
        self.assertIn("--video-snapshot-interval-steps", native_video_argv_from_args(periodic))
        pair = parser.parse_args(["--video-dir", "frames", "--video-env-index", "1",
            "--video-secondary-env-index", "3", "--video-snapshot-interval-steps", "1"])
        self.assertEqual(native_video_options_from_args(pair).secondary_env_index, 3)
        forwarded = parser.parse_args(native_video_argv_from_args(pair))
        self.assertEqual(native_video_options_from_args(forwarded), native_video_options_from_args(pair))
        for flags, error in ((["--video-secondary-env-index", "1"], "distinct"),
                             (["--video-secondary-env-index", "-1"], "nonnegative")):
            with self.assertRaisesRegex(ValueError, error):
                native_video_options_from_args(parser.parse_args(["--video-dir", "frames",
                    "--video-env-index", "1", "--video-snapshot-steps", "1", *flags]))
        with self.assertRaisesRegex(ValueError, "requires snapshot"):
            native_video_options_from_args(parser.parse_args(["--video-dir", "frames",
                "--video-secondary-env-index", "1"]))
        with self.assertRaisesRegex(ValueError, "outside"):
            native_video_options_from_args(pair).configure_viewer(SimpleNamespace(scene=SimpleNamespace(num_envs=3)))
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            native_video_options_from_args(parser.parse_args(["--video-dir", "frames",
                "--video-snapshot-steps", "1", "--video-snapshot-interval-steps", "1"]))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            native_video_options_from_args(parser.parse_args(["--video-dir", "frames",
                "--video-snapshot-interval-steps", "0"]))
        with self.assertRaisesRegex(ValueError, "distinct increasing"):
            native_video_options_from_args(parser.parse_args(["--video-dir", "frames", "--video-snapshot-steps", "2", "1"]))
        with self.assertRaisesRegex(ValueError, "provided together"):
            native_video_options_from_args(parser.parse_args(["--video-dir", "frames", "--video-snapshot-steps", "1",
                "--video-secondary-camera-eye", "1", "2", "3"]))

    def test_recording_metadata_names_tracking_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            recorder = object.__new__(NativeVideoRecorder)
            recorder.options = NativeVideoOptions(Path(tmp), env_index=2)
            recorder._recording, recorder._frames = True, []
            recorder._control_step = 0
            recorder._name, recorder._seed = "empty", 7
            recorder.control_dt_s, recorder.fps, recorder.records = .02, 50., []
            row = recorder.finish(frame_count=0)
            self.assertEqual(row["camera_origin_type"], "asset_root")
            self.assertEqual(row["camera_tracking_version"], "asset_root_tracking.v1")
            self.assertEqual(row["camera_asset_name"], "robot")

    def test_manual_samples_copy_reused_render_buffers_and_default_start_resets_mode(self) -> None:
        import numpy as np

        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        recorder = object.__new__(NativeVideoRecorder)
        recorder.options = NativeVideoOptions(Path("."), resolution=(2, 2))
        recorder.env = SimpleNamespace(render=lambda: pixels)
        recorder._recording, recorder._frames = False, []
        recorder.start("manual", 42, task_records_steps=True)
        pixels[:] = 7
        recorder.record_step()
        pixels[:] = 99
        self.assertTrue((recorder._frames[0] == 7).all())
        recorder._recording = False
        recorder.record_step()
        self.assertEqual(len(recorder._frames), 1)
        recorder.start("ordinary", 43)
        self.assertFalse(recorder._task_records_steps)

    def test_selected_dual_views_keep_state_and_trim_after_episode_end(self) -> None:
        import numpy as np

        with self.assertRaisesRegex(ValueError, r"actual=\(2, 2, 3\)/uint8, expected=\(2, 4, 3\)/uint8"):
            _raw_rgb_png(np.zeros((2, 2, 3), dtype=np.uint8), (4, 2))

        class Tensor:
            def __init__(self, array): self.torch = array

        class Array:
            def __init__(self, array): self.array = array
            def __getitem__(self, index): return Array(self.array[index])
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.array

        class Controller:
            def __init__(self): self.calls = []; self.eye = (1., 0., 0.)
            def set_view_env_index(self, index): self.calls.append(("env", index))
            def update_view_to_asset_root(self, name): self.calls.append(("root", name))
            def update_view_location(self, eye, lookat):
                self.eye = tuple(eye); self.calls.append(("view", self.eye))

        class Env:
            def __init__(self):
                self.renders = 0
                self.viewport_camera_controller = Controller()
                self.sim = SimpleNamespace(get_physics_step_count=lambda: 17,
                    physics_manager=SimpleNamespace(get_simulation_time=lambda: 0.34))
                data = SimpleNamespace(**{name: Tensor(Array(np.array([values], dtype=np.float32)))
                    for name, values in {"root_pos_w": [0, 0, 1],
                        "root_quat_w": [0, 0, 0, 1],
                        "root_lin_vel_w": [0.1, 0.2, 0.3],
                        "root_ang_vel_w": [0.4, 0.5, 0.6],
                        "joint_pos": [0, 1], "joint_vel": [0, 0],
                        "body_link_pos_w": [[0, 0, 1], [0, 0, 0.5]],
                        "body_link_quat_w": [[0, 0, 0, 1], [0, 0, 0, 1]]}.items()})
                self.scene = {"robot": SimpleNamespace(data=data,
                    joint_names=["hip", "knee"], body_names=["base", "leg"])}
                self.video_recorder = SimpleNamespace(_capture=SimpleNamespace(
                    _render_product=SimpleNamespace(path="/Render/ActualRgb")))

            def camera_params(self):
                result = camera_params_fixture()
                result["cameraViewTransform"][12] = -10 * self.viewport_camera_controller.eye[0]
                return result

            def render(self):
                self.renders += 1
                return np.full((2, 2, 3), 7 if self.viewport_camera_controller.eye[0] > 0 else 9,
                               dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmp:
            native = Env()
            recorder = object.__new__(NativeVideoRecorder)
            recorder.options = NativeVideoOptions(Path(tmp), resolution=(2, 2),
                eye=(1., 0., 0.), lookat=(0., 0., 0.), snapshot_steps=(1, 3),
                secondary_eye=(-1., 0., 0.), secondary_lookat=(0., 0., 0.))
            recorder.env = SimpleNamespace(unwrapped=native, render=native.render)
            recorder._imageables = ()
            recorder._recording = False
            recorder._camera_render_product = native.video_recorder._capture._render_product
            recorder._camera_params_annotator = SimpleNamespace(get_data=native.camera_params)
            recorder._attach_camera_params = lambda: None
            recorder.control_dt_s, recorder.fps, recorder.records = .02, 50., []
            recorder.start("episode", 3, task_records_steps=True)
            for _ in range(3): recorder.record_step()
            self.assertEqual(len(recorder._snapshots), 2)
            self.assertEqual(len(list(Path(tmp).glob("episode.*.png"))), 4)
            json.dumps(recorder._snapshots)  # Only metadata remains after each selected step.
            self.assertEqual(native.renders, 16)  # 8 warmup; 2 steps x 2 views x 2 renders.
            self.assertEqual(native.viewport_camera_controller.eye, (1., 0., 0.))
            row = recorder.finish(frame_count=2)
            manifest = json.loads((Path(tmp) / row["file"]).read_text())
            self.assertEqual(manifest["schema_version"], "native_rgb_snapshots.v4")
            self.assertEqual(manifest["sampling_mode"], "fixed_steps")
            self.assertEqual(manifest["articulation_joint_names"], ["hip", "knee"])
            self.assertEqual(manifest["articulation_body_names"], ["base", "leg"])
            self.assertEqual((row["frame_count"], row["snapshot_count"]), (2, 2))
            self.assertEqual(len(manifest["samples"]), 1)
            sample = manifest["samples"][0]
            self.assertEqual(sample["control_step"], 1)
            self.assertEqual(sample["state_before"], sample["state_after"])
            self.assertEqual(sample["state_before"]["physics_step_count"], 17)
            self.assertEqual(sample["state_before"]["root_pos_w"], [0, 0, 1])
            self.assertEqual(sample["state_before"]["root_quat_w"], [0, 0, 0, 1])
            self.assertEqual(set(sample["state_before"]), {"root_pos_w", "root_quat_w",
                "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel",
                "body_link_pos_w", "body_link_quat_w", "articulation_body_names",
                "physics_step_count", "simulation_time_s", "sha256"})
            self.assertEqual(sample["state_before"]["body_link_pos_w"], [[0, 0, 1], [0, 0, .5]])
            self.assertEqual(sample["state_before"]["articulation_body_names"], ["base", "leg"])
            self.assertEqual([v["camera_params"]["cameraViewTransform"][3][0]
                              for v in sample["views"]], [-10., 10.])
            self.assertEqual(sample["views"][0]["camera_render_product_path"], "/Render/ActualRgb")
            self.assertIn("xyzw", manifest["state_conventions"]["root_quat_w"])
            self.assertEqual([v["view_id"] for v in sample["views"]], ["v0", "v1"])
            self.assertEqual(len(list(Path(tmp).glob("*.png"))), 2)
            self.assertFalse(list(Path(tmp).glob("*.step000003.*")))
            def first_pixel(path):
                png = path.read_bytes()
                self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
                size = int.from_bytes(png[33:37], "big")
                self.assertEqual(png[37:41], b"IDAT")
                return zlib.decompress(png[41:41 + size])[1]
            self.assertEqual([first_pixel(Path(tmp) / view["file"])
                              for view in sample["views"]], [7, 9])
            self.assertEqual(manifest["renders_per_view"], 2)
            self.assertEqual(native.viewport_camera_controller.calls[-1], ("view", (1., 0., 0.)))

            recorder.options = NativeVideoOptions(Path(tmp), resolution=(2, 2),
                eye=(1., 0., 0.), lookat=(0., 0., 0.), snapshot_interval_steps=1,
                secondary_eye=(-1., 0., 0.), secondary_lookat=(0., 0., 0.))
            recorder.start("dense", 4, task_records_steps=True)
            for _ in range(4): recorder.record_step()
            self.assertEqual(len(list(Path(tmp).glob("dense.*.png"))), 8)
            json.dumps(recorder._snapshots)
            native.scene["robot"].body_names.reverse()
            with self.assertRaisesRegex(RuntimeError, "body names do not match"):
                recorder.finish(frame_count=3)
            native.scene["robot"].body_names.reverse()
            recorder._recording = True
            dense = recorder.finish(frame_count=3)
            dense_manifest = json.loads((Path(tmp) / dense["file"]).read_text())
            self.assertEqual(dense_manifest["schema_version"], "native_rgb_snapshots.v4")
            self.assertEqual(dense_manifest["sampling_mode"], "interval_from_step_1")
            self.assertEqual(dense_manifest["snapshot_interval_steps"], 1)
            self.assertEqual([s["control_step"] for s in dense_manifest["samples"]], [1, 2, 3])
            self.assertEqual((dense["frame_count"], dense["snapshot_count"]), (3, 6))
            self.assertEqual(len(list(Path(tmp).glob("dense.*.png"))), 6)
            self.assertFalse(list(Path(tmp).glob("dense.step000004.*")))

    def test_camera_params_uses_existing_rgb_product_and_copies_vendor_values(self) -> None:
        from unittest.mock import Mock

        product = SimpleNamespace(path="/Render/RealRgb")
        annotator = Mock()
        registry = SimpleNamespace(get_annotator=Mock(return_value=annotator))
        rep = SimpleNamespace(AnnotatorRegistry=registry)
        modules = {"omni": SimpleNamespace(replicator=SimpleNamespace(core=rep)),
                   "omni.replicator": SimpleNamespace(core=rep), "omni.replicator.core": rep}
        recorder = object.__new__(NativeVideoRecorder)
        recorder.env = SimpleNamespace(unwrapped=SimpleNamespace(video_recorder=SimpleNamespace(
            _capture=SimpleNamespace(_render_product=product))))
        recorder._camera_params_annotator = recorder._camera_render_product = None
        with patch.dict("sys.modules", modules):
            recorder._attach_camera_params()
            recorder._attach_camera_params()
        registry.get_annotator.assert_called_once_with("camera_params")
        annotator.attach.assert_called_once_with([product])
        self.assertIs(recorder._camera_render_product, product)
        data = camera_params_fixture()
        data["cameraViewTransform"][12] = -4.5
        copied = _camera_params_data(data, (2, 2))
        data["cameraViewTransform"][12] = 999
        self.assertEqual(copied["cameraViewTransform"][3][0], -4.5)
        self.assertEqual(copied["cameraProjection"], data["cameraProjection"].tolist())
        json.dumps(copied)
        with self.assertRaisesRegex(RuntimeError, "resolution"):
            _camera_params_data(data, (4, 2))
        data["cameraProjection"][0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "cameraProjection"):
            _camera_params_data(data, (2, 2))

    def test_named_link_poses_are_bound_into_state_hash(self) -> None:
        import numpy as np

        class Array:
            def __init__(self, array): self.array = array
            def __getitem__(self, index): return Array(self.array[index])
            def detach(self): return self
            def cpu(self): return self
            def numpy(self): return self.array

        values = {"root_pos_w": [[0., 0., 1.]], "root_quat_w": [[0., 0., 0., 1.]],
                  "root_lin_vel_w": [[0., 0., 0.]], "root_ang_vel_w": [[0., 0., 0.]],
                  "joint_pos": [[0.]], "joint_vel": [[0.]],
                  "body_link_pos_w": [[[0., 0., 1.], [0., 0., .5]]],
                  "body_link_quat_w": [[[0., 0., 0., 1.], [0., 0., 0., 1.]]]}
        data = SimpleNamespace(**{name: SimpleNamespace(torch=Array(np.array(value)))
                                  for name, value in values.items()})
        robot = SimpleNamespace(data=data, body_names=["base", "leg"])
        env = SimpleNamespace(scene={"robot": robot}, sim=SimpleNamespace(
            get_physics_step_count=lambda: 17,
            physics_manager=SimpleNamespace(get_simulation_time=lambda: .34)))
        before = _selected_state(env, 0)
        data.body_link_pos_w.torch.array[0, 1, 0] = .01
        moved = _selected_state(env, 0)
        self.assertNotEqual(before["sha256"], moved["sha256"])
        for key in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w", "joint_pos", "joint_vel"):
            self.assertEqual(before[key], moved[key])
        robot.body_names.reverse()
        self.assertNotEqual(moved["sha256"], _selected_state(env, 0)["sha256"])
        robot.body_names.pop()
        with self.assertRaisesRegex(RuntimeError, "named link poses"):
            _selected_state(env, 0)

    def test_snapshot_rejects_physics_advance_during_render(self) -> None:
        import numpy as np
        from unittest.mock import patch

        clock = {"step": 5}
        controller = SimpleNamespace(update_view_to_asset_root=lambda _: None,
                                     set_view_env_index=lambda _: None,
                                     update_view_location=lambda **_: None)
        scene_env = SimpleNamespace(viewport_camera_controller=controller)
        recorder = object.__new__(NativeVideoRecorder)
        recorder.options = NativeVideoOptions(Path("."), resolution=(2, 2), snapshot_steps=(1,))
        recorder.env = SimpleNamespace(unwrapped=scene_env,
            render=lambda: (clock.__setitem__("step", clock["step"] + 1) or
                            np.zeros((2, 2, 3), dtype=np.uint8)))
        recorder._recording, recorder._control_step, recorder._snapshots = True, 0, []
        recorder._imageables = ()
        with patch("true_rl_agent.runner.native_video._selected_state",
                   side_effect=lambda *_: {"physics_step_count": clock["step"]}):
            with self.assertRaisesRegex(RuntimeError, "physics state changed"):
                recorder.record_step()
        self.assertEqual(recorder._snapshots, [])

    def test_every_control_frame_is_encoded_once_in_order(self) -> None:
        import imageio_ffmpeg
        import numpy as np

        # Each image carries its index as ten wide binary bars. Reading those
        # bars after lossy H264 encoding detects both dropped and repeated frames.
        indices = np.arange(469)
        bits = (indices[:, None] >> np.arange(10)) & 1
        bars = np.repeat(bits * 255, 8, axis=1)
        # Isaac returns RGB sliced from RGBA storage, so the real input is strided.
        frames = [np.tile(row[None, :, None], (16, 1, 4)).astype(np.uint8)[:, :, :3] for row in bars]
        self.assertFalse(frames[0].flags.c_contiguous)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "control_frames.mp4"
            _encode_frames(path, frames, resolution=(80, 16), fps=50.0)
            decoded = subprocess.run(
                [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-i", str(path),
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
                capture_output=True, check=True,
            )
            observed = []
            for image in np.frombuffer(decoded.stdout, dtype=np.uint8).reshape(-1, 16, 80, 3):
                values = image[8, np.arange(10) * 8 + 4, 0] > 127
                observed.append(sum(int(bit) << index for index, bit in enumerate(values)))
            self.assertEqual(observed, indices.tolist())


if __name__ == "__main__":
    unittest.main()
