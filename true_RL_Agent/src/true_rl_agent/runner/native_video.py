"""Native RGB recording attached to an existing Isaac evaluation environment.

This module owns presentation and encoding only. Task executors own resets,
actions, episode outcomes, seeds and evidence identities.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeVideoOptions:
    output_dir: Path
    env_index: int = 0
    resolution: tuple[int, int] = (1280, 720)
    eye: tuple[float, float, float] = (4.5, -4.5, 3.0)
    lookat: tuple[float, float, float] = (0.8, 0.0, 0.0)

    def configure_viewer(self, cfg: Any) -> None:
        if not 0 <= self.env_index < cfg.scene.num_envs:
            raise ValueError("video env_index is outside the evaluation cohort")
        cfg.viewer.resolution = self.resolution
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
        for index, prim_path in enumerate(env.unwrapped.scene.env_prim_paths):
            imageable = UsdGeom.Imageable(stage.GetPrimAtPath(prim_path))
            if index == options.env_index:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
        self.control_dt_s = float(env.unwrapped.step_dt)
        self.fps = 1.0 / self.control_dt_s
        self._recording = False
        self._task_records_steps = False
        self._frames: list[Any] = []
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
        # Kit creates the RGB annotator lazily and initially returns a blank
        # frame. Prime rendering without stepping physics, then capture the
        # first native frame after the first control step. Lab's
        # num_rerenders_on_reset only runs for scene RTX sensors; viewport RGB
        # has no such sensor and creates its annotator inside render().
        for _ in range(8):
            self.env.render()
        self._frames = []
        self._recording = True

    def record_step(self) -> None:
        """Capture the state selected by the task, before any task-owned reset."""
        if self._recording:
            self._frames.append(self.env.render().copy())

    def finish(self, *, frame_count: int) -> dict[str, Any]:
        if not self._recording or not 0 <= frame_count <= len(self._frames):
            raise RuntimeError("native recording does not cover the requested interval")
        # Keep actual native frames for the task-selected interval. Other
        # cohort members may require a longer rollout after this one closes.
        self._recording = False
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


def native_video_options_from_args(args: Any) -> NativeVideoOptions | None:
    if args.video_dir is None:
        return None
    args.enable_cameras = True
    return NativeVideoOptions(args.video_dir.resolve(), env_index=args.video_env_index)
