"""Phase 0 bounded CPU-only verification of existing Push contracts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "unit"))
MODULES = (
    "test_push_physical_protocol",
    "test_push_force_protocol",
    "test_push_force_runtime",
    "test_push_force_consumers",
    "test_push_feedback_projection",
    "test_push_recovery_runtime_adapter",
    "test_push_recovery_v2_prompt_smoke",
    "test_push_phased_evaluation",
)

def main() -> int:
    import torch
    torch.set_num_threads(1)
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    with (OUT / "push_cpu_regression.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    receipt = {
        "schema_version": "feedback_p0.push_cpu_regression.v1",
        "scope": "existing synthetic CPU unit tests; no simulator/provider/training acceptance",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(),
        "modules": list(MODULES),
        "tests_run": result.testsRun,
        "failures": [{"test": str(test), "traceback": trace} for test, trace in result.failures],
        "errors": [{"test": str(test), "traceback": trace} for test, trace in result.errors],
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "passed_without_skip": result.wasSuccessful() and not result.skipped,
        "elapsed_seconds": time.time() - started,
        "test_source_sha256": {
            name: hashlib.sha256((ROOT / "tests" / "unit" / (name + ".py")).read_bytes()).hexdigest()
            for name in MODULES
        },
    }
    (OUT / "push_cpu_regression.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("tests_run", "passed_without_skip", "elapsed_seconds", "cuda_available")}, ensure_ascii=False))
    return 0 if receipt["passed_without_skip"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
