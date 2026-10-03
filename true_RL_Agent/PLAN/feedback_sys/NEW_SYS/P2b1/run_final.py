"""Capture final local CPU commands and Windows peak working-set evidence."""
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "outputs/p2b"


class MemoryCounters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
        *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
          "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]


def run(name, arguments):
    metrics = OUT / (name + ".memory.json")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--child", str(metrics), *arguments]
    started = time.perf_counter()
    with (OUT / (name + ".stdout.txt")).open("wb") as stdout, (OUT / (name + ".stderr.txt")).open("wb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr)
        child.wait()
    receipt = {"name": name, "argv": command, "exit_code": child.returncode,
        "logical_argv": [sys.executable, "-B", *arguments],
        "wall_seconds": time.perf_counter() - started,
        "memory": json.loads(metrics.read_bytes()),
        "stdout_sha256": hashlib.sha256((OUT / (name + ".stdout.txt")).read_bytes()).hexdigest(),
        "stderr_sha256": hashlib.sha256((OUT / (name + ".stderr.txt")).read_bytes()).hexdigest()}
    if child.returncode:
        print((OUT / (name + ".stderr.txt")).read_text(encoding="utf-8"))
    return receipt


def main():
    receipts = []
    for group in ("k4_force_v1", "k4_velocity_v2"):
        receipts.append(run("build_" + group, ["scripts/build_push_behavior_card.py", "--source-lock",
            "PLAN/feedback_sys/NEW_SYS/P2b1/source_lock.json", "--group", group, "--output", "outputs/p2b/" + group]))
    receipts.append(run("oracle", ["PLAN/feedback_sys/NEW_SYS/P2b1/audit_replay.py"]))
    receipts.append(run("new_tests", ["-m", "unittest", "tests.unit.test_push_trajectory", "-v"]))
    (OUT / "command_receipts.json").write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipts, indent=2))
    raise SystemExit(int(any(r["exit_code"] for r in receipts)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        metrics = Path(sys.argv[2])
        arguments = sys.argv[3:]
        sys.path.insert(0, str(ROOT))  # Preserve python -m's working-directory import path.
        try:
            if arguments[0] == "-m":
                sys.argv = arguments[1:]
                runpy.run_module(arguments[1], run_name="__main__", alter_sys=True)
            else:
                sys.argv = arguments
                runpy.run_path(arguments[0], run_name="__main__")
        finally:
            counters = MemoryCounters()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(-1), ctypes.byref(counters), counters.cb)
            metrics.write_text(json.dumps({"scope": "actual_python_process_not_windows_venv_launcher",
                "peak_working_set_bytes": counters.PeakWorkingSetSize if ok else None}), encoding="utf-8")
    else:
        main()
