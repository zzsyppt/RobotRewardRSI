"""Compatibility filename for the shared formal CLI; use explicit task and cell."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.run_generation_session import main


if __name__ == "__main__":
    raise SystemExit(main())
