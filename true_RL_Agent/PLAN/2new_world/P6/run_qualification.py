"""Bound one complete P6 reference qualification; the child CLI owns its GPU lease."""
from __future__ import annotations

import argparse
import json
import hashlib
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.runner.process_handle import OwnedProcessTimeout, run_owned_process

WALL_TIMEOUT_S = 14400
TERMINATION_GRACE_S = 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory for process receipt and stdout/stderr logs")
    parser.add_argument("--prior-process-receipt", type=Path, help="Original completed attempt receipt; its elapsed time remains charged")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply the complete qualification CLI command after --")
    prior = None
    charged_s = 0.0
    if args.prior_process_receipt is not None:
        prior_bytes = args.prior_process_receipt.read_bytes()
        prior = json.loads(prior_bytes)
        charged_s = float(prior["elapsed_wall_s"])
        if not math.isfinite(charged_s) or charged_s < 0 or "ended_at_unix" not in prior:
            parser.error("prior attempt must have a finite completed wall time")
    remaining_s = math.floor(WALL_TIMEOUT_S - charged_s)
    if remaining_s <= 0:
        parser.error("original qualification wall budget is exhausted")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema_version": "true_rl_agent.p6.qualification_process.v1",
        "purpose": "P6 formal reference qualification: one cell, one full training and three evaluation phases",
        "qualification_verdict_owner": "qualification_cli_and_genesis_receipt",
        "gpu_lease_owner": "qualification_cli", "wrapper_acquires_gpu_lease": False,
        "wall_timeout_s": remaining_s, "termination_grace_s": TERMINATION_GRACE_S,
        "original_wall_budget_s": WALL_TIMEOUT_S, "charged_prior_wall_s": charged_s,
        "prior_process_receipt": (None if prior is None else {
            "path": str(args.prior_process_receipt.resolve()),
            "sha256": hashlib.sha256(prior_bytes).hexdigest()}),
        "wrapper_retries": 0, "argv": command, "argv_hash": canonical_json_hash(command),
        "cwd": str(ROOT), "started_at_unix": time.time(), "status": "starting",
        "stdout_log": str(output / "stdout.log"), "stderr_log": str(output / "stderr.log"),
    }
    started = time.monotonic()

    def persist() -> None:
        (output / "process.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def on_started(pid: int) -> None:
        report.update(child_pid=pid, status="running")
        persist()

    persist()
    try:
        result = run_owned_process(command, cwd=ROOT, log_dir=output,
            timeout_s=remaining_s, termination_grace_s=TERMINATION_GRACE_S, on_started=on_started)
        report.update(returncode=result.returncode, child_pid=result.pid,
            status="completed" if result.returncode == 0 else "nonzero_exit")
    except OwnedProcessTimeout as exc:
        report.update(status="timeout", returncode=124, error_type=type(exc).__name__, error=str(exc))
    except KeyboardInterrupt:
        report.update(status="interrupted", returncode=130)
    except Exception as exc:
        report.update(status="error", returncode=1, error_type=type(exc).__name__, error=str(exc))
    finally:
        report.update(ended_at_unix=time.time(), elapsed_wall_s=time.monotonic() - started)
        persist()
    print(json.dumps({"process_receipt": str(output / "process.json"), "returncode": report["returncode"]}), flush=True)
    return report["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
