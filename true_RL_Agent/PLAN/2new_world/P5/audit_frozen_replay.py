"""Exercise the production historical CLI against the original P4 fixture."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PROJECT / "src"), str(PROJECT)]
from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
from true_rl_agent.tasks.stair_climbing import generation_history


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    fixture, receipt = map(Path, sys.argv[1:])
    before = snapshot(fixture)
    output = fixture / "formal-output"
    frozen_hash = before["formal-output/generation_session_report.json"][2]
    assert frozen_hash == "cec022d738bae577a715a1bfa41b839f50514f5f8b5255e65f3a6dd04437941a"
    old = json.loads((PROJECT / "PLAN/2new_world/P4/audit_historical_cli_v2.json").read_text())
    argv = old["argv"]
    argv[argv.index("--project-root") + 1] = str(PROJECT)
    argv[argv.index("--output-dir") + 1] = str(output)
    argv[argv.index("--lease-root") + 1] = str(fixture / "formal-leases")
    spec = importlib.util.spec_from_file_location("historical_cli", PROJECT / "scripts/audit_stair_generation_history.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    stdout = io.StringIO()
    result = {"status": "RUNNING", "argv": argv, "fixture_backend_and_checkout_override": True}
    start = time.monotonic()
    try:
        with use_test_isaac_lab_execution_environment(), patch.object(
            generation_history, "_verify_source_checkout", return_value="a" * 40,
        ), contextlib.redirect_stdout(stdout):
            assert cli.main(argv) == 0
        after = snapshot(fixture)
        assert after == before, "historical replay changed fixture files"
        result.update(status="PASS", files_unchanged=len(before), comparison="bytes+mtime_ns+sha256",
                      report_sha256=frozen_hash, cli_result=json.loads(stdout.getvalue()))
    except Exception as exc:
        result.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                      files_unchanged=snapshot(fixture) == before)
        raise
    finally:
        result["elapsed_s"] = time.monotonic() - start
        receipt.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
