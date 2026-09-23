"""Execute explicitly registered acceptance cases; each driver owns its GPU lease."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--case", action="append", required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    for name in args.case:
        argv = [part.replace("{source}", str(args.source)).replace("{run_root}", str(args.run_root)) for part in cases[name]]
        record = {"case": name, "argv": [sys.executable, *argv], "source": str(args.source),
                  "started_unix": time.time()}
        receipt = args.run_root / (name + ".launch.json")
        with receipt.open("x") as stream:
            json.dump(record, stream, indent=2)
        with (args.run_root / (name + ".log")).open("x") as log:
            result = subprocess.run(["/usr/bin/time", "-f", "peak_rss_kib=%M elapsed_s=%e exit=%x",
                "-o", str(args.run_root / (name + ".time")), sys.executable, *argv],
                cwd=args.source, stdout=log, stderr=subprocess.STDOUT)
        record.update(ended_unix=time.time(), returncode=result.returncode)
        receipt.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({"case": name, "returncode": result.returncode}), flush=True)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
