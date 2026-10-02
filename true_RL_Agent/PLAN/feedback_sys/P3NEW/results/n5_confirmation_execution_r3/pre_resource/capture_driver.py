"""Run exactly one frozen capture command; record process/exit without retries."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    task = sys.argv[1]
    assert task in ("stair", "push")
    card = json.loads((HERE / "capture_card.json").read_bytes())
    assert card["status"] == "FROZEN"
    assert digest(HERE / "capture_card.json") == "85a4bf2146578340a849ab4948a082cd439b9a5ec2dbf3ac1b2f2ab876108218"
    manifest = json.loads((HERE / "runtime_source_manifest.json").read_bytes())
    assert digest(HERE / "runtime_source_manifest.json") == card["source"]["manifest"]["sha256"]
    source = Path(card["source"]["remote_root"])
    changed = [name for name, record in manifest.items() if digest(source / name) != record["sha256"] or (source / name).stat().st_size != record["size_bytes"]]
    assert not changed, changed
    command = card["commands"][task]
    argv = command["argv"]
    assert argv[argv.index("--mode") + 1] == "run"
    output = Path(argv[argv.index("--output") + 1])
    assert not output.exists(), "Preserve any previous run; never restart automatically"
    if task == "stair":
        assert not Path(card["output_root"]).exists()
    else:
        previous = json.loads((HERE / "stair.exit.json").read_bytes())
        assert previous["returncode"] == 0, "Stair did not complete successfully"
    disk = shutil.disk_usage(HERE).free
    required = card["resources"]["minimum_disk_before_task_bytes"][task]
    assert disk >= required, (disk, required)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    gpu_rows = [[int(x.strip()) for x in line.split(",")] for line in gpu.stdout.splitlines()]
    assert any(total >= 32000 and used < 1024 for _, used, total in gpu_rows), "No idle 32GB GPU; production lease still decides placement"
    checkpoint = card["tasks"][task]["checkpoint"]
    assert digest(Path(checkpoint["path"])) == checkpoint["sha256"]
    env = os.environ.copy()
    env.update(command["environment_set"])
    for name in command["environment_unset_process_local"]:
        env.pop(name, None)
    with (HERE / f"{task}.stdout.log").open("xb") as stdout, (HERE / f"{task}.stderr.log").open("xb") as stderr:
        process = subprocess.Popen(argv, cwd=command["cwd"], env=env, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr)
        save(HERE / f"{task}.started.json", {"started_at_utc": now(), "driver_pid": os.getpid(), "child_pid": process.pid, "task": task, "argv": argv, "cwd": command["cwd"], "environment_set": command["environment_set"], "environment_unset_process_local": command["environment_unset_process_local"], "source_files_verified": len(manifest), "checkpoint_bytes_verified": True, "card_sha256": digest(HERE / "capture_card.json"), "driver_sha256": digest(Path(__file__)), "gpu_before": gpu_rows, "disk_free_before": disk, "automatic_retries": 0})
        code = process.wait()
    save(HERE / f"{task}.exit.json", {"finished_at_utc": now(), "task": task, "returncode": code, "driver_pid": os.getpid(), "child_pid": process.pid, "disk_free_after": shutil.disk_usage(HERE).free})
    return code


if __name__ == "__main__":
    raise SystemExit(main())
