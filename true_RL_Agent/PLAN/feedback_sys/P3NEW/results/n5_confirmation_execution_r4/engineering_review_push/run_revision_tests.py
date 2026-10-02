"""CPU-only review entrypoint: revised Push renderer over frozen dependencies."""
import importlib.util
import sys
import unittest
from pathlib import Path

frozen = Path(sys.argv[1]).resolve()
revision = Path(sys.argv[2]).resolve()
sys.path[:0] = [str(frozen / "src"), str(frozen)]
import true_rl_agent.tasks.push_recovery as push_package

push_package.__path__.insert(0, str(revision))
spec = importlib.util.spec_from_file_location("push_revision_tests", revision / "test_push_process_report.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
suite = unittest.defaultTestLoader.loadTestsFromModule(module)
result = unittest.TextTestRunner(verbosity=2).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
