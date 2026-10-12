"""Native RGB recording attached to an existing Isaac evaluation environment.

This module owns presentation and encoding only. Task executors own resets,
actions, episode outcomes, seeds and evidence identities.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import zlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeVideoOptions:
    output_dir: Path
    env_index: int = 0
    resolution: tuple[int, int] = (1280, 720)
    eye: tuple[float, float, float] = (4.5, -4.5, 3.0)
    lookat: tuple[float, float, float] = (0.8, 0.0, 0.0)
    snapshot_steps: tuple[int, ...] = ()
    snapshot_interval_steps: int | None = None
    secondary_eye: tuple[float, float, float] | None = None
    secondary_lookat: tuple[float, float, float] | None = None
    secondary_env_index: int | None = None

    def __post_init__(self) -> None:
        if (len(self.resolution) != 2 or any(type(value) is not int or value <= 0 or value % 2
                                              for value in self.resolution)):
            raise ValueError("video resolution must be two positive even integers")
        for name, value in (("eye", self.eye), ("lookat", self.lookat),
                            ("secondary_eye", self.secondary_eye),
                            ("secondary_lookat", self.secondary_lookat)):
            if value is None:
                continue
            if len(value) != 3 or any(not math.isfinite(float(axis)) for axis in value):
                raise ValueError(f"video {name} must have three finite coordinates")
        if (len(self.snapshot_steps) > 12 or
            any(type(step) is not int or step <= 0 for step in self.snapshot_steps) or
            tuple(sorted(set(self.snapshot_steps))) != self.snapshot_steps):
            raise ValueError("video snapshot steps must be 1..12 distinct increasing positive integers")
        if self.snapshot_interval_steps is not None and (
            type(self.snapshot_interval_steps) is not int or self.snapshot_interval_steps <= 0
        ):
            raise ValueError("video snapshot interval must be a positive integer")
        if self.snapshot_steps and self.snapshot_interval_steps is not None:
            raise ValueError("video fixed snapshot steps and interval are mutually exclusive")
        if (self.secondary_eye is None) != (self.secondary_lookat is None):
            raise ValueError("video secondary camera eye/lookat must be provided together")
        if self.secondary_eye is not None and not self.snapshot_mode:
            raise ValueError("video secondary camera requires snapshot mode")
        if self.secondary_env_index is not None:
            if (type(self.secondary_env_index) is not int or self.secondary_env_index < 0
                    or self.secondary_env_index == self.env_index):
                raise ValueError("video secondary env must be a distinct nonnegative index")
            if not self.snapshot_mode:
                raise ValueError("video secondary env requires snapshot mode")

    @property
    def snapshot_mode(self) -> bool:
        return bool(self.snapshot_steps) or self.snapshot_interval_steps is not None

    def configure_viewer(self, cfg: Any) -> None:
        if not 0 <= self.env_index < cfg.scene.num_envs:
            raise ValueError("video env_index is outside the evaluation cohort")
        if self.secondary_env_index is not None and self.secondary_env_index >= cfg.scene.num_envs:
            raise ValueError("video secondary env_index is outside the evaluation cohort")
        video_recorder = getattr(cfg, "video_recorder", None)
        if video_recorder is None:
            raise ValueError("native RGB recording requires cfg.video_recorder")
        cfg.viewer.resolution = self.resolution
        video_recorder.window_width, video_recorder.window_height = self.resolution
        cfg.viewer.origin_type = "asset_root"
        cfg.viewer.asset_name = "robot"
        cfg.viewer.env_index = self.env_index
        cfg.viewer.eye = self.eye
        cfg.viewer.lookat = self.lookat


class NativeVideoRecorder:
    def __init__(self, env: Any, options: NativeVideoOptions) -> None:
        from gymnasium import Wrapper
        from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController

        self.options = options
        # Lab creates this controller for GUI/visualizer sessions only. Native
        # headless RGB capture needs the same controller to track the asset.
        if env.unwrapped.viewport_camera_controller is None:
            env.unwrapped.viewport_camera_controller = ViewportCameraController(
                env.unwrapped, env.unwrapped.cfg.viewer
            )
        # Evaluation clones may share a terrain origin. Hide other clones in
        # the image only; retain their physics, actions and evaluation records.
        from pxr import UsdGeom

        stage = env.unwrapped.sim.stage
        self._imageables = tuple(UsdGeom.Imageable(stage.GetPrimAtPath(path))
                                for path in env.unwrapped.scene.env_prim_paths)
        for index, imageable in enumerate(self._imageables):
            if index == options.env_index:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        self.control_dt_s = float(env.unwrapped.step_dt)
        self.fps = 1.0 / self.control_dt_s
        self._recording = False
        self._task_records_steps = False
        self._frames: list[Any] = []
        self._snapshots: list[dict[str, Any]] = []
        self._control_step = 0
        self._camera_params_annotator = None
        self._camera_render_product = None
        owner = self

        class RecordingEnv(Wrapper):
            def step(self, action: Any) -> Any:
                transition = self.env.step(action)
                if not owner._task_records_steps:
                    owner.record_step()
                return transition

        self.env = RecordingEnv(env)
        options.output_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._name = ""
        self._seed = 0

    def start(self, name: str, seed: int, *, task_records_steps: bool = False) -> None:
        self._name, self._seed = name, seed
        self._task_records_steps = task_records_steps
        if self.options.snapshot_mode:
            self._activate_environment()
        # Kit creates the RGB annotator lazily and initially returns a blank
        # frame. Prime rendering without stepping physics, then capture the
        # first native frame after the first control step. Lab's
        # num_rerenders_on_reset only runs for scene RTX sensors; viewport RGB
        # has no such sensor and creates its annotator inside render().
        for warmup in range(8):
            self.env.render()
            if warmup == 0 and self.options.snapshot_mode:
                self._attach_camera_params()
        self._frames = []
        self._snapshots = []
        self._control_step = 0
        self._recording = True

    def _attach_camera_params(self) -> None:
        # This is the installed IsaacLab Kit RGB path. Attach to its existing
        # product; a second camera/product could disagree with the saved RGB.
        import omni.replicator.core as rep

        product = self.env.unwrapped.video_recorder._capture._render_product
        if product is None:
            raise RuntimeError("native RGB render product was not initialized")
        if product is self._camera_render_product:
            return
        if self._camera_params_annotator is not None:
            self._camera_params_annotator.detach([self._camera_render_product])
        self._camera_params_annotator = rep.AnnotatorRegistry.get_annotator("camera_params")
        self._camera_params_annotator.attach([product])
        self._camera_render_product = product

    def _activate_environment(self) -> None:
        # Recorders share one viewport. Construction of another recorder may
        # have hidden this clone; select both visibility and camera ownership.
        for index, imageable in enumerate(self._imageables):
            if index == self.options.env_index:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        controller = self.env.unwrapped.viewport_camera_controller
        controller.set_view_env_index(self.options.env_index)
        controller.update_view_to_asset_root("robot")

    def record_step(self) -> None:
        """Capture one step; the task's call site determines reset/event timing."""
        if self._recording:
            self._control_step += 1
            if self.options.snapshot_mode:
                if (self._control_step in self.options.snapshot_steps or
                    (self.options.snapshot_interval_steps is not None and
                     (self._control_step - 1) % self.options.snapshot_interval_steps == 0)):
                    self._capture_snapshot_pair()
            else:
                self._frames.append(self.env.render().copy())

    def _capture_snapshot_pair(self) -> None:
        """Capture selected views while checking that rendering did not step physics."""
        scene_env = self.env.unwrapped
        controller = scene_env.viewport_camera_controller
        before = _selected_state(scene_env, self.options.env_index)
        self._activate_environment()
        views = [("v0", self.options.eye, self.options.lookat)]
        if self.options.secondary_eye is not None:
            views.append(("v1", self.options.secondary_eye, self.options.secondary_lookat))
        captures = []
        try:
            for view_id, eye, lookat in views:
                controller.update_view_to_asset_root("robot")
                controller.update_view_location(eye=eye, lookat=lookat)
                # Kit's annotator may return the previous camera on the first
                # app update. Discard one render after every camera move.
                self.env.render()
                frame = self.env.render().copy()
                if _selected_state(scene_env, self.options.env_index) != before:
                    raise RuntimeError("physics state changed during camera-only snapshot")
                if scene_env.video_recorder._capture._render_product is not self._camera_render_product:
                    raise RuntimeError("native RGB render product changed during snapshot")
                camera_params = _camera_params_data(
                    self._camera_params_annotator.get_data(), self.options.resolution)
                captures.append((view_id, eye, lookat, frame, camera_params))
        finally:
            controller.update_view_to_asset_root("robot")
            controller.update_view_location(eye=self.options.eye, lookat=self.options.lookat)
        after = _selected_state(scene_env, self.options.env_index)
        if before != after:
            raise RuntimeError("physics state changed during camera-only snapshot")
        root = before["root_pos_w"]
        view_rows = []
        written = []
        try:
            for view_id, eye, lookat, frame, camera_params in captures:
                path = self.options.output_dir / f"{self._name}.step{self._control_step:06d}.{view_id}.png"
                if path.exists():
                    raise FileExistsError(f"native snapshot already exists: {path}")
                payload = _raw_rgb_png(frame, self.options.resolution)
                written.append(path)
                path.write_bytes(payload)
                view_rows.append({"file": path.name, "sha256": hashlib.sha256(payload).hexdigest(),
                                  "view_id": view_id, "camera_eye_offset": list(eye),
                                  "camera_lookat_offset": list(lookat),
                                  "camera_eye_w": [root[i] + eye[i] for i in range(3)],
                                  "camera_lookat_w": [root[i] + lookat[i] for i in range(3)],
                                  "camera_render_product_path": str(self._camera_render_product.path),
                                  "camera_params": camera_params})
        except Exception:
            for path in written:
                path.unlink(missing_ok=True)
            raise
        self._snapshots.append({"control_step": self._control_step, "state_before": before,
                                "state_after": after, "views": view_rows})

    def finish(self, *, frame_count: int) -> dict[str, Any]:
        if not self._recording or not 0 <= frame_count <= self._control_step or (
            not self.options.snapshot_mode and frame_count > len(self._frames)
        ):
            raise RuntimeError("native recording does not cover the requested interval")
        # Keep actual native frames for the task-selected interval. Other
        # cohort members may require a longer rollout after this one closes.
        self._recording = False
        if self.options.snapshot_mode:
            return self._finish_snapshots(frame_count)
        frames, self._frames = self._frames[:frame_count], []
        path = self.options.output_dir / (self._name + ".mp4")
        if frames:
            _encode_frames(path, frames, resolution=self.options.resolution, fps=self.fps)
        self.records.append(
            {
                "file": path.name if frame_count else None,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                if frame_count
                else None,
                "recording_status": "recorded" if frame_count else "no_pre_reset_frame",
                "seed": self._seed,
                "env_index": self.options.env_index,
                "resolution": list(self.options.resolution),
                "camera_eye_offset": list(self.options.eye),
                "camera_lookat_offset": list(self.options.lookat),
                "camera_origin_type": "asset_root",
                "camera_tracking_version": "asset_root_tracking.v1",
                "camera_asset_name": "robot",
                "other_envs_visually_hidden": True,
                "frame_count": frame_count,
                "fps": self.fps,
                "encoder": "imageio_ffmpeg_raw_rgb",
                "control_dt_s": self.control_dt_s,
                "starts_at_reset": False,
                "start_control_step": 1,
                "render_warmup_frames": 8,
                "simulated_duration_s": frame_count * self.control_dt_s,
            }
        )
        return self.records[-1]

    def _finish_snapshots(self, frame_count: int) -> dict[str, Any]:
        articulation = self.env.unwrapped.scene["robot"]
        joint_names = list(articulation.joint_names)
        body_names = list(articulation.body_names)
        groups = []
        for sample in self._snapshots:
            step = sample["control_step"]
            if len(joint_names) != len(sample["state_before"]["joint_pos"]):
                raise RuntimeError("snapshot joint names do not match sampled articulation order")
            if body_names != sample["state_before"]["articulation_body_names"]:
                raise RuntimeError("snapshot body names do not match sampled articulation order")
            if step > frame_count:
                for view in sample["views"]:
                    (self.options.output_dir / view["file"]).unlink()
            else:
                groups.append(sample)
        manifest = {"schema_version": "native_rgb_snapshots.v4", "name": self._name,
                    "seed": self._seed, "env_index": self.options.env_index,
                    "articulation_joint_names": joint_names,
                    "articulation_body_names": body_names,
                    "resolution": list(self.options.resolution),
                    "camera_origin_type": "asset_root", "camera_asset_name": "robot",
                    "camera_tracking_version": "asset_root_tracking.v1",
                    "state_conventions": {
                        "root_pos_w": "root link position in world xyz, metres",
                        "root_quat_w": "root link orientation in world xyzw, unit quaternion",
                        "root_lin_vel_w": "root centre-of-mass velocity in world xyz, metres/second",
                        "root_ang_vel_w": "root centre-of-mass angular velocity in world xyz, radians/second",
                        "joint_pos": "articulation coordinates, radians (revolute) or metres (prismatic)",
                        "joint_vel": "articulation coordinate rates, radians/second or metres/second",
                        "body_link_pos_w": "link origins in world xyz, metres, in articulation_body_names order",
                        "body_link_quat_w": "link orientations in world xyzw, unit quaternions, in articulation_body_names order",
                        "camera_eye_w_and_lookat_w": "world xyz, metres",
                        "camera_eye_offset_and_lookat_offset": "root-link-position-relative xyz, metres",
                        "camera_params": "unmodified Replicator camera_params values from the RGB render product; matrices reshaped to 4x4 in returned element order",
                    },
                    "other_envs_visually_hidden": True,
                    "episode_control_step_count": frame_count,
                    "selected_control_steps": list(self.options.snapshot_steps),
                    "snapshot_interval_steps": self.options.snapshot_interval_steps,
                    "sampling_mode": "interval_from_step_1" if self.options.snapshot_interval_steps is not None else "fixed_steps",
                    "captured_step_count": len(groups),
                    "snapshot_count": sum(len(group["views"]) for group in groups),
                    "renders_per_view": 2, "render_warmup_frames": 8,
                    "samples": groups, "control_dt_s": self.control_dt_s,
                    "starts_at_reset": False, "start_control_step": 1,
                    "simulated_duration_s": frame_count * self.control_dt_s}
        path = self.options.output_dir / (self._name + ".snapshots.json")
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self._frames, self._snapshots = [], []
        self.records.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "recording_status": "recorded" if groups else "no_selected_pre_reset_frame",
                             "seed": self._seed, "env_index": self.options.env_index,
                             "frame_count": frame_count, "snapshot_count": manifest["snapshot_count"],
                             "capture_mode": "selected_rgb_snapshots", "manifest": manifest})
        return self.records[-1]


def native_video_recorders(env: Any, options: NativeVideoOptions) -> tuple[NativeVideoRecorder, ...]:
    """Independent streams on one physical environment; tasks own their clocks."""
    indices = (options.env_index,) if options.secondary_env_index is None else (
        options.env_index, options.secondary_env_index)
    return tuple(NativeVideoRecorder(env, replace(options, env_index=index, secondary_env_index=None))
                 for index in indices)


def _selected_state(env: Any, env_index: int) -> dict[str, Any]:
    import numpy as np

    articulation = env.scene["robot"]
    robot = articulation.data
    body_names = list(articulation.body_names)
    if not body_names or len(set(body_names)) != len(body_names):
        raise RuntimeError("snapshot requires distinct articulation body names")
    arrays = {}
    digest = hashlib.sha256()
    for name in ("root_pos_w", "root_quat_w", "root_lin_vel_w", "root_ang_vel_w",
                 "joint_pos", "joint_vel", "body_link_pos_w", "body_link_quat_w"):
        value = getattr(robot, name).torch[env_index].detach().cpu().numpy()
        if name in ("body_link_pos_w", "body_link_quat_w"):
            width = 3 if name == "body_link_pos_w" else 4
            if value.shape != (len(body_names), width) or not np.isfinite(value).all():
                raise RuntimeError(f"snapshot {name} does not match finite named link poses")
        arrays[name] = value.tolist()
        digest.update(name.encode("ascii") + str(value.shape).encode("ascii") + value.tobytes())
    digest.update(json.dumps(body_names, ensure_ascii=True).encode("ascii"))
    sim = env.sim
    clock = {"physics_step_count": int(sim.get_physics_step_count()),
             "simulation_time_s": float(sim.physics_manager.get_simulation_time())}
    digest.update(json.dumps(clock, sort_keys=True).encode("ascii"))
    return {**arrays, "articulation_body_names": body_names, **clock, "sha256": digest.hexdigest()}


def _camera_params_data(data: dict[str, Any], resolution: tuple[int, int]) -> dict[str, Any]:
    """Copy the actual RGB product's calibration, without deriving it from look-at."""
    import numpy as np

    if data.get("cameraModel") != "pinhole":
        raise RuntimeError("native snapshot requires pinhole camera_params")
    if list(data.get("renderProductResolution", ())) != list(resolution):
        raise RuntimeError("camera_params resolution does not match native RGB")
    result = {"cameraModel": data["cameraModel"], "renderProductResolution": list(resolution)}
    for name, shape in (("cameraViewTransform", (4, 4)), ("cameraProjection", (4, 4)),
                        ("cameraAperture", (2,)), ("cameraApertureOffset", (2,)),
                        ("cameraNearFar", (2,)), ("cameraFocalLength", ()),
                        ("metersPerSceneUnit", ())):
        value = np.asarray(data[name])
        if value.size != math.prod(shape) or not np.isfinite(value).all():
            raise RuntimeError(f"invalid native camera_params {name}")
        result[name] = value.reshape(shape).tolist()
    return result


def _raw_rgb_png(frame: Any, resolution: tuple[int, int]) -> bytes:
    import numpy as np

    width, height = resolution
    rgb = np.asarray(frame)
    if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
        raise ValueError(
            f"native snapshot RGB shape/dtype mismatch: actual={rgb.shape}/{rgb.dtype}, "
            f"expected={(height, width, 3)}/uint8"
        )
    scanlines = b"".join(b"\0" + rgb[row].tobytes(order="C") for row in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))


def _encode_frames(
    path: Path, frames: list[Any], *, resolution: tuple[int, int], fps: float,
) -> None:
    import imageio_ffmpeg

    # Feed each native image exactly once. A timestamp-based image sequence
    # can repeat or skip frames through float rounding.
    writer = imageio_ffmpeg.write_frames(
        str(path), size=resolution, fps=fps, codec="libx264",
        pix_fmt_in="rgb24", pix_fmt_out="yuv420p", quality=7, macro_block_size=1,
    )
    writer.send(None)
    try:
        for frame in frames:
            # Isaac's RGB view can retain the stride of an RGBA buffer.
            writer.send(frame.tobytes(order="C"))
    finally:
        writer.close()


def add_native_video_arguments(parser: Any) -> None:
    parser.add_argument("--video-dir", type=Path, default=None)
    parser.add_argument("--video-env-index", type=int, default=0)
    parser.add_argument("--video-secondary-env-index", type=int, default=None)
    parser.add_argument("--video-camera-eye", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--video-camera-lookat", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--video-resolution", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--video-snapshot-steps", nargs="+", type=int, default=None, metavar="STEP")
    parser.add_argument("--video-snapshot-interval-steps", type=int, default=None, metavar="N")
    parser.add_argument("--video-secondary-camera-eye", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--video-secondary-camera-lookat", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"))


def native_video_options_from_args(args: Any) -> NativeVideoOptions | None:
    if args.video_dir is None:
        return None
    args.enable_cameras = True
    defaults = NativeVideoOptions(args.video_dir.resolve(), env_index=args.video_env_index)
    return NativeVideoOptions(
        defaults.output_dir, env_index=defaults.env_index,
        eye=tuple(args.video_camera_eye) if args.video_camera_eye is not None else defaults.eye,
        lookat=tuple(args.video_camera_lookat) if args.video_camera_lookat is not None else defaults.lookat,
        resolution=tuple(args.video_resolution) if args.video_resolution is not None else defaults.resolution,
        snapshot_steps=tuple(args.video_snapshot_steps) if args.video_snapshot_steps is not None else (),
        snapshot_interval_steps=args.video_snapshot_interval_steps,
        secondary_eye=tuple(args.video_secondary_camera_eye) if args.video_secondary_camera_eye is not None else None,
        secondary_lookat=tuple(args.video_secondary_camera_lookat) if args.video_secondary_camera_lookat is not None else None,
        secondary_env_index=args.video_secondary_env_index,
    )


def native_video_argv_from_args(args: Any) -> list[str]:
    """Forward presentation flags from a replay driver to its eval subprocess."""
    overrides = (args.video_camera_eye, args.video_camera_lookat, args.video_resolution,
                  args.video_snapshot_steps, args.video_snapshot_interval_steps, args.video_secondary_camera_eye,
                  args.video_secondary_camera_lookat, args.video_secondary_env_index)
    if args.video_dir is None:
        if any(value is not None for value in overrides) or args.video_env_index != 0:
            raise ValueError("video presentation options require --video-dir")
        return []
    native_video_options_from_args(args)  # Validate dimensions and finite offsets.
    argv = ["--video-dir", str(args.video_dir), "--video-env-index", str(args.video_env_index)]
    for name, values in (("--video-camera-eye", args.video_camera_eye),
                         ("--video-camera-lookat", args.video_camera_lookat),
                         ("--video-resolution", args.video_resolution),
                         ("--video-snapshot-steps", args.video_snapshot_steps),
                         ("--video-secondary-camera-eye", args.video_secondary_camera_eye),
                         ("--video-secondary-camera-lookat", args.video_secondary_camera_lookat)):
        if values is not None:
            argv.extend((name, *(str(value) for value in values)))
    if args.video_snapshot_interval_steps is not None:
        argv.extend(("--video-snapshot-interval-steps", str(args.video_snapshot_interval_steps)))
    if args.video_secondary_env_index is not None:
        argv.extend(("--video-secondary-env-index", str(args.video_secondary_env_index)))
    return argv
