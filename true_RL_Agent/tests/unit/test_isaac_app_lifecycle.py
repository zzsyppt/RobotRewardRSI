from __future__ import annotations

import unittest

from true_rl_agent.runner.isaac_app_lifecycle import close_isaac_app_on_success


class _FakeSimulationApp:
    def __init__(self, *, close_error: Exception | None = None) -> None:
        self.close_calls = 0
        self.close_error = close_error

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


class IsaacAppLifecycleTests(unittest.TestCase):
    def test_failed_business_status_skips_close_and_preserves_code(self) -> None:
        app = _FakeSimulationApp()

        code = close_isaac_app_on_success(app, exit_code=7)

        self.assertEqual(code, 7)
        self.assertEqual(app.close_calls, 0)

    def test_success_closes_app(self) -> None:
        app = _FakeSimulationApp()

        code = close_isaac_app_on_success(app, exit_code=0)

        self.assertEqual(code, 0)
        self.assertEqual(app.close_calls, 1)

    def test_close_failure_becomes_nonzero(self) -> None:
        app = _FakeSimulationApp(close_error=RuntimeError("close failed"))

        code = close_isaac_app_on_success(app, exit_code=0)

        self.assertEqual(code, 1)
        self.assertEqual(app.close_calls, 1)

    def test_exit_code_is_strict(self) -> None:
        app = _FakeSimulationApp()
        for invalid in (True, 1.0, "1", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    close_isaac_app_on_success(  # type: ignore[arg-type]
                        app,
                        exit_code=invalid,
                    )
        for invalid in (-1, 256):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    close_isaac_app_on_success(app, exit_code=invalid)


if __name__ == "__main__":
    unittest.main()
