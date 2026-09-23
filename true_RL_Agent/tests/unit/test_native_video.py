"""Native image order must survive real encoding, including awkward durations."""

import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from true_rl_agent.runner.native_video import NativeVideoRecorder, _encode_frames

class NativeVideoEncodingTests(unittest.TestCase):
    def test_manual_samples_copy_reused_render_buffers_and_default_start_resets_mode(self) -> None:
        import numpy as np

        pixels = np.zeros((2, 2, 3), dtype=np.uint8)
        recorder = object.__new__(NativeVideoRecorder)
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
