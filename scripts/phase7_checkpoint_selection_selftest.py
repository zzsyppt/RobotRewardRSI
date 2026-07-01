#!/usr/bin/env python3
"""Self-test for immutable runner checkpoint selection.

This test intentionally uses only the Python standard library and project code,
so it can run locally without Isaac Lab.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from humanoid_agent.immutable_runner.checkpoints import (  # noqa: E402
    checkpoint_iteration,
    find_latest_checkpoint,
    find_latest_checkpoint_from_summary,
    list_model_checkpoints,
)


def _touch(path: Path) -> None:
    path.write_bytes(b"checkpoint")


def _assert_equal(actual: object, expected: object, message: str) -> None:
    if actual != expected:
        raise AssertionError(f"{message}: expected={expected!r}, actual={actual!r}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="phase7_checkpoint_selection_") as tmp:
        root = Path(tmp)
        for name in [
            "model_0.pt",
            "model_100.pt",
            "model_900.pt",
            "model_1000.pt",
            "model_1999.pt",
            "model_3999.pt",
            "model_final.pt",
            "model_abc.pt",
            "not_model_9999.pt",
            "model_4000.tmp",
        ]:
            _touch(root / name)

        latest = find_latest_checkpoint(root)
        _assert_equal(latest, root / "model_3999.pt", "numeric latest checkpoint selection")
        _assert_equal(checkpoint_iteration(root / "model_1999.pt"), 1999, "iteration parsing")
        _assert_equal(checkpoint_iteration(root / "model_final.pt"), None, "non-standard name ignored")
        _assert_equal(
            [item.iteration for item in list_model_checkpoints(root)],
            [0, 100, 900, 1000, 1999, 3999],
            "valid checkpoints sorted numerically",
        )

        two_file_case = root / "two_file_case"
        two_file_case.mkdir()
        _touch(two_file_case / "model_900.pt")
        _touch(two_file_case / "model_1999.pt")
        _assert_equal(
            find_latest_checkpoint(two_file_case),
            two_file_case / "model_1999.pt",
            "model_900 vs model_1999 selects 1999",
        )

        polluted_summary = {
            "training": {
                "log_dir": str(root),
                "latest_checkpoint": str(root / "model_900.pt"),
            }
        }
        _assert_equal(
            find_latest_checkpoint_from_summary(polluted_summary),
            root / "model_3999.pt",
            "summary log_dir overrides polluted latest_checkpoint",
        )

        legacy_summary = {
            "training": {
                "latest_checkpoint": str(root / "model_1999.pt"),
            }
        }
        _assert_equal(
            find_latest_checkpoint_from_summary(legacy_summary),
            None,
            "summary latest_checkpoint fallback is disabled by default",
        )
        _assert_equal(
            find_latest_checkpoint_from_summary(legacy_summary, allow_legacy_latest_checkpoint=True),
            root / "model_1999.pt",
            "summary latest_checkpoint fallback requires explicit legacy opt-in",
        )

        invalid_only = root / "invalid_only"
        invalid_only.mkdir()
        _touch(invalid_only / "model_final.pt")
        _assert_equal(find_latest_checkpoint(invalid_only), None, "invalid-only directory fails closed")
        _assert_equal(find_latest_checkpoint(root / "missing"), None, "missing directory fails closed")

    print(json.dumps({"status": "pass", "test": "phase7_checkpoint_selection"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
