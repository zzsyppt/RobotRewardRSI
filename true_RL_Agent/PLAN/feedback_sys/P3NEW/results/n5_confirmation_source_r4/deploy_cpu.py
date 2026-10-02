"""Deploy the new R4 snapshot and run only remote_cpu.py; preserve all failures."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
BASE = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
REMOTE = BASE + "/n5_confirmation_source_r4_final_control"
SOURCE = BASE + "/n5_confirmation_source_r4_final"
SSH = ["ssh", "-o", "BatchMode=yes", "humanoid-server"]


def run(name, argv, payload=None, timeout=120):
    with (HERE / (name + ".stdout.txt")).open("xb") as stdout, (HERE / (name + ".stderr.txt")).open("xb") as stderr:
        result = subprocess.run(argv, input=payload, stdout=stdout, stderr=stderr, timeout=timeout)
    (HERE / (name + ".command.json")).write_text(json.dumps({"argv": argv, "returncode": result.returncode}, indent=2)+"\n")
    print(json.dumps({"step": name, "returncode": result.returncode}), flush=True)
    if result.returncode:
        print((HERE / (name + ".stderr.txt")).read_text(errors="replace")[-5000:], flush=True)
        raise SystemExit(result.returncode)


def main():
    run("history_scan", SSH + ["/usr/bin/python3", "-"], (HERE / "history_scan.py").read_bytes())
    history = json.loads((HERE / "history_scan.stdout.txt").read_bytes())
    assert not history["errors"]
    (HERE / "history_cohorts.before.json").write_bytes((HERE / "history_scan.stdout.txt").read_bytes())
    roots = f"from pathlib import Path\ns=Path('{SOURCE}');c=Path('{REMOTE}');p=Path('{BASE}/n5_confirmation_preflight_r4_final')\nassert not s.exists() and not c.exists() and not p.exists()\ns.mkdir();c.mkdir()\nprint('fresh R4 roots created; no capture or models')\n"
    run("remote_roots", SSH + ["/usr/bin/python3", "-"], roots.encode())
    shutil.copyfile(HERE.parent / "n5_confirmation_execution_r3/capture_card.json", HERE / "r3_capture_card.readonly.json")
    names = ("source.tar.gz", "source_manifest.json", "source_preparation.json", "source_delta.json", "commands.candidate.json",
             "history_cohorts.before.json", "r3_capture_card.readonly.json", "remote_cpu.py")
    run("upload", ["scp", *[str(HERE / name) for name in names], "humanoid-server:" + REMOTE + "/"], timeout=180)
    unpack = f"""import hashlib,json,tarfile
from pathlib import Path
s=Path('{SOURCE}');c=Path('{REMOTE}')
assert not list(s.iterdir())
with tarfile.open(c/'source.tar.gz','r:gz') as archive:
 for item in archive.getmembers():
  assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
 archive.extractall(s)
expected=json.loads((c/'source_manifest.json').read_bytes())
actual={{str(p.relative_to(s)):{{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}} for p in s.rglob('*') if p.is_file()}}
assert actual==expected
print(json.dumps({{'verified_source_files':len(actual)}}))
"""
    run("deployment", SSH + ["/usr/bin/python3", "-"], unpack.encode())
    run("cpu_validation", SSH + ["/data/yuyangliu/envs/isaaclab6_validation/bin/python", "-B", REMOTE + "/remote_cpu.py"], timeout=1800)
    run("download", ["scp", "humanoid-server:" + REMOTE + "/verification_results.tar", str(HERE / "verification_results.tar")], timeout=180)
    last = json.loads((HERE / "cpu_validation.stdout.txt").read_text().strip().splitlines()[-1])
    archive_path = HERE / "verification_results.tar"
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == last["download_archive_sha256"]
    target = HERE / "downloaded"
    target.mkdir(exist_ok=False)
    with tarfile.open(archive_path) as archive:
        for item in archive.getmembers():
            assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
        for item in archive.getmembers():
            payload = archive.extractfile(item).read()
            destination = Path("\\\\?\\" + str((target / item.name).absolute()))
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream: stream.write(payload)
            assert hashlib.sha256(destination.read_bytes()).digest() == hashlib.sha256(payload).digest()
    verification = json.loads((target / "control/verification.json").read_bytes())
    receipt = {"status": "SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE",
               "source_manifest_sha256": verification["source_manifest_sha256"], "source_files": verification["source_files"],
               "local_source": str(HERE / "source"), "remote_source": SOURCE,
               "verification_sha256": hashlib.sha256((target / "control/verification.json").read_bytes()).hexdigest(),
               "verification_archive_sha256": last["download_archive_sha256"], "tests": verification["test_count"],
               "preflight_exit_codes": {task: 0 for task in ("stair", "push")},
               "actual_model_calls": 0, "actual_gpu_runs": 0, "actual_capture_runs": 0,
               "source_unchanged_after_execution": verification["source_unchanged_after_tests_and_preflights"],
               "run_contract_status": "NOT_RELEASED", "final_capture_card_created": False,
               "known_limits": verification["limitations"]}
    with (HERE / "source_freeze_receipt.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
