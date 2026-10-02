"""Preserve initial source/test failure; add its one omitted existing CPU test."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = "/data/yuyangliu/AGENT/feedback_p3new_20260929"
SOURCE = BASE + "/n5_confirmation_source_r3_final"
CONTROL = BASE + "/n5_confirmation_source_r3_final_control"
PREFLIGHT = BASE + "/n5_confirmation_preflight_r3_final"
SSH = ["ssh", "-o", "BatchMode=yes", "humanoid-server"]


def read(path): return json.loads(path.read_bytes())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def record(path): return {"sha256": digest(path), "size_bytes": path.stat().st_size}


def save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def run(name, argv, payload=None, timeout=180):
    with (HERE / (name + ".stdout.txt")).open("xb") as stdout, (HERE / (name + ".stderr.txt")).open("xb") as stderr:
        result = subprocess.run(argv, input=payload, stdout=stdout, stderr=stderr, timeout=timeout)
    save(HERE / (name + ".command.json"), {"argv": argv, "returncode": result.returncode})
    print(json.dumps({"step": name, "returncode": result.returncode}), flush=True)
    if result.returncode:
        print((HERE / (name + ".stdout.txt")).read_text(errors="replace")[-6000:])
        print((HERE / (name + ".stderr.txt")).read_text(errors="replace")[-3000:])
        raise SystemExit(result.returncode)


def main():
    initial = read(HERE / "source_manifest.json")
    assert all(record(HERE / "source" / name) == item for name, item in initial.items())
    final = deepcopy(initial)
    missing = "tests/unit/test_confirmation_event_table.py"
    assert missing not in final
    final[missing] = record(ROOT / missing)
    destination = HERE / "source_final"
    destination.mkdir(exist_ok=False)
    for name in sorted(final):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name if name == missing else HERE / "source" / name, target)
        assert record(target) == final[name]
    save(HERE / "source_manifest.final.json", dict(sorted(final.items())))
    save(HERE / "source_correction.json", {"reason": "Initial CPU test registration included an existing test module absent from the inherited R2 source manifest",
         "preserved_initial_source_manifest_sha256": digest(HERE / "source_manifest.json"), "added_file": missing,
         "added_file_record": final[missing], "all_other_484_source_files_identical": True,
         "initial_test_result": "23 actual tests passed; one missing-module loader error; task preflight not reached",
         "changes_to_runtime_or_evaluation_semantics": False})
    with tarfile.open(HERE / "source.final.tar.gz", "x:gz") as archive:
        for name in sorted(final): archive.add(destination / name, arcname=name, recursive=False)
    commands = read(HERE / "commands.candidate.json")
    def rewrite(value):
        if isinstance(value, str):
            return value.replace("n5_confirmation_source_r3", "n5_confirmation_source_r3_final").replace(
                "n5_confirmation_preflight_r3", "n5_confirmation_preflight_r3_final")
        if isinstance(value, list): return [rewrite(item) for item in value]
        if isinstance(value, dict): return {key: rewrite(item) for key, item in value.items()}
        return value
    commands = rewrite(commands)
    save(HERE / "commands.candidate.final.json", commands)
    preparation = read(HERE / "source_preparation.json")
    preparation.update(status="CORRECTED_CPU_SOURCE_PREFLIGHT_PENDING_NOT_RELEASED", source_files=len(final),
        source_manifest_sha256=digest(HERE / "source_manifest.final.json"), bundle=record(HERE / "source.final.tar.gz"))
    preparation["added"].append(missing)
    save(HERE / "source_preparation.final.json", preparation)
    remote_script = (HERE / "remote_cpu.py").read_text().replace("n5_confirmation_source_r3", "n5_confirmation_source_r3_final").replace(
        "n5_confirmation_preflight_r3", "n5_confirmation_preflight_r3_final")
    (HERE / "remote_cpu.final.py").write_text(remote_script)
    staging = HERE / "final_control_inputs"
    staging.mkdir(exist_ok=False)
    names = {"source.tar.gz": "source.final.tar.gz", "source_manifest.json": "source_manifest.final.json",
             "source_preparation.json": "source_preparation.final.json", "source_delta.json": "source_delta.json",
             "source_correction.json": "source_correction.json", "commands.candidate.json": "commands.candidate.final.json",
             "history_cohorts.before.json": "history_cohorts.complete.json", "r2_capture_card.readonly.json": "r2_capture_card.readonly.json",
             "remote_cpu.py": "remote_cpu.final.py"}
    for target, original in names.items(): shutil.copyfile(HERE / original, staging / target)
    failed = HERE / "failed_cpu_initial"
    failed.mkdir(exist_ok=False)
    run("failed_cpu_download", ["scp", *["humanoid-server:"+BASE+"/n5_confirmation_source_r3_control/"+name
        for name in ("unit_tests.stdout.txt", "unit_tests.stderr.txt", "unit_tests.execution.json")], str(failed)])
    roots = f"from pathlib import Path\ns=Path('{SOURCE}');c=Path('{CONTROL}');p=Path('{PREFLIGHT}')\nassert not s.exists() and not c.exists() and not p.exists()\ns.mkdir();c.mkdir()\nprint('fresh final R3 CPU roots created')\n"
    run("final_remote_roots", SSH+["/usr/bin/python3", "-"], roots.encode())
    run("final_upload", ["scp", *[str(path) for path in staging.iterdir()], "humanoid-server:"+CONTROL+"/"])
    unpack = f"""import hashlib,json,tarfile
from pathlib import Path
s=Path('{SOURCE}');c=Path('{CONTROL}')
assert not list(s.iterdir())
with tarfile.open(c/'source.tar.gz') as archive:
 for item in archive.getmembers():
  assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
 archive.extractall(s)
actual={{str(p.relative_to(s)):{{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'size_bytes':p.stat().st_size}} for p in s.rglob('*') if p.is_file()}}
assert actual==json.loads((c/'source_manifest.json').read_bytes())
print(json.dumps({{'source_files':len(actual)}}))
"""
    run("final_deployment", SSH+["/usr/bin/python3", "-"], unpack.encode())
    run("final_cpu_validation", SSH+["/data/yuyangliu/envs/isaaclab6_validation/bin/python", "-B", CONTROL+"/remote_cpu.py"], timeout=1800)
    run("final_download", ["scp", "humanoid-server:"+CONTROL+"/verification_results.tar", str(HERE / "verification_results.final.tar")])
    last = json.loads((HERE / "final_cpu_validation.stdout.txt").read_text().strip().splitlines()[-1])
    assert digest(HERE / "verification_results.final.tar") == last["download_archive_sha256"]
    downloaded = HERE / "downloaded_final"
    downloaded.mkdir(exist_ok=False)
    with tarfile.open(HERE / "verification_results.final.tar") as archive:
        for item in archive.getmembers():
            assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
        archive.extractall(downloaded, filter="data")
    verification = read(downloaded / "control/verification.json")
    save(HERE / "source_freeze_receipt.json", {"status": "SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE",
         "source_manifest": str(HERE / "source_manifest.final.json"), "source_manifest_sha256": digest(HERE / "source_manifest.final.json"),
         "source_files": len(final), "local_source": str(destination), "remote_source": SOURCE,
         "commands": str(HERE / "commands.candidate.final.json"), "verification": str(downloaded / "control/verification.json"),
         "verification_sha256": digest(downloaded / "control/verification.json"),
         "verification_archive_sha256": last["download_archive_sha256"], "tests": verification["test_count"],
         "preflight_exit_codes": {"stair": 0, "push": 0}, "actual_model_calls": 0, "actual_gpu_runs": 0, "actual_capture_runs": 0,
         "source_unchanged_after_execution": verification["source_unchanged_after_tests_and_preflights"],
         "prior_artifacts_unchanged": all(record(HERE / "source" / name) == item for name,item in initial.items()),
         "run_contract_status": "NOT_RELEASED", "final_capture_card_created": False,
         "new_model_calibration": "PENDING; old GPT6-SOL calibration does not certify GPT-6.1-SOL",
         "known_limits": verification["limitations"]})
    print(json.dumps({"status": "SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE", "source_files": len(final),
                      "source_manifest_sha256": digest(HERE / "source_manifest.final.json"), "tests": verification["test_count"]}), flush=True)


if __name__ == "__main__":
    main()
