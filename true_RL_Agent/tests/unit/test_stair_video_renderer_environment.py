from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from true_rl_agent.tasks.stair_climbing.video_renderer_environment import (  # noqa: E402
    RENDERER_ENVIRONMENT_VARIABLE,
    build_renderer_environment,
    ensure_renderer_environment,
    renderer_worker_command,
    renderer_worker_environment,
)


class StairVideoRendererEnvironmentTests(unittest.TestCase):
    def _project_root(self, temporary_dir: str) -> Path:
        root = Path(temporary_dir)
        requirements = root / "requirements"
        requirements.mkdir()
        requirements.joinpath("stair_checkpoint_video_renderer.txt").write_text(
            (ROOT / "requirements" / "stair_checkpoint_video_renderer.txt").read_text(
                encoding="utf-8"
            ),
            encoding="utf-8",
        )
        return root

    def test_plan_is_content_addressed_inside_project_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._project_root(temporary_dir)

            first = build_renderer_environment(root)
            second = build_renderer_environment(root)

            self.assertEqual(first.environment_id, second.environment_id)
            self.assertEqual(first.target_dir, second.target_dir)
            self.assertIn(root.resolve(), first.target_dir.parents)
            self.assertEqual(first.package_versions["pybullet"], "3.2.7")
            self.assertEqual(len(first.package_versions), 6)

    def test_provisioning_uses_project_venv_and_never_global_or_user_install(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = self._project_root(temporary_dir)
            plan = build_renderer_environment(root)

            def fake_run(command, **kwargs):
                if command[:3] == [sys.executable, "-m", "venv"]:
                    target = Path(command[3])
                    self.assertEqual(target, plan.target_dir)
                    plan.python_executable.parent.mkdir(parents=True)
                    plan.python_executable.write_bytes(b"python")
                    plan.target_dir.joinpath("pyvenv.cfg").write_text(
                        "include-system-site-packages = false\n",
                        encoding="utf-8",
                    )
                else:
                    self.assertEqual(
                        command[:4],
                        [str(plan.python_executable), "-m", "pip", "install"],
                    )
                    self.assertIn("--isolated", command)
                    self.assertIn("--only-binary=:all:", command)
                    self.assertNotIn("--target", command)
                    self.assertNotIn("--user", command)
                    self.assertNotIn("--prefix", command)
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with (
                patch(
                    "true_rl_agent.tasks.stair_climbing."
                    "video_renderer_environment.subprocess.run",
                    side_effect=fake_run,
                ) as run,
                patch(
                    "true_rl_agent.tasks.stair_climbing."
                    "video_renderer_environment._probe_environment",
                    return_value=None,
                ),
            ):
                environment = ensure_renderer_environment(root)

            self.assertEqual(run.call_count, 2)
            self.assertTrue(environment.marker_path.is_file())
            worker_env = renderer_worker_environment(environment)
            self.assertNotIn("PYTHONPATH", worker_env)
            self.assertEqual(worker_env["PYTHONNOUSERSITE"], "1")
            self.assertEqual(
                worker_env[RENDERER_ENVIRONMENT_VARIABLE],
                str(environment.marker_path),
            )
            command = renderer_worker_command(
                environment,
                ROOT / "scripts" / "isaaclab_stair_checkpoint_video.py",
                ["--render-worker"],
            )
            self.assertEqual(command[0], str(environment.python_executable))
            self.assertEqual(command[1], "-I")
            manifest = environment.to_manifest_dict()
            self.assertEqual(
                manifest["isolation"]["kind"],
                "project_owned_content_addressed_venv",
            )
            self.assertFalse(manifest["isolation"]["global_environment_modified"])


if __name__ == "__main__":
    unittest.main()
