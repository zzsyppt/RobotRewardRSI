from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.audit import run_loop_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one true_rl_agent dry-run audit round.")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    output_dir = args.output_dir or _default_output_dir("round")
    summary = run_loop_audit(output_dir=output_dir, rounds=1, session_id=args.session_id)
    print(json.dumps(summary["checks"], indent=2, sort_keys=True))
    return 0 if all(summary["checks"].values()) else 1


def _default_output_dir(prefix: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return PROJECT_ROOT / "runs" / f"{prefix}_{stamp}"


if __name__ == "__main__":
    raise SystemExit(main())
