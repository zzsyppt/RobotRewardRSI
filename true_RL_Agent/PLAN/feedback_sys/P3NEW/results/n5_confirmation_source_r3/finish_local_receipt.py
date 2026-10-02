"""Resume only verified local evidence extraction after Windows long-path failure."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from complete_cpu_source import HERE, SOURCE, digest, read, record, save


def main():
    last = json.loads((HERE / "final_cpu_validation.stdout.txt").read_text().strip().splitlines()[-1])
    bundle = HERE / "verification_results.final.tar"
    assert digest(bundle) == last["download_archive_sha256"]
    downloaded = HERE / "downloaded_final"
    with tarfile.open(bundle) as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            assert member.isfile() and not relative.is_absolute() and ".." not in relative.parts
            payload = archive.extractfile(member).read()
            target = Path("\\\\?\\" + str((downloaded / Path(*relative.parts)).absolute()))
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert target.read_bytes() == payload
            else:
                with target.open("xb") as stream:
                    stream.write(payload)
            assert hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(payload).hexdigest()
    verification = read(downloaded / "control/verification.json")
    initial = read(HERE / "source_manifest.json")
    final = read(HERE / "source_manifest.final.json")
    assert all(record(HERE / "source_final" / name) == item for name, item in final.items())
    save(HERE / "source_freeze_receipt.json", {
        "status": "SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE",
        "source_manifest": str(HERE / "source_manifest.final.json"),
        "source_manifest_sha256": digest(HERE / "source_manifest.final.json"),
        "source_files": len(final), "local_source": str(HERE / "source_final"), "remote_source": SOURCE,
        "commands": str(HERE / "commands.candidate.final.json"),
        "commands_sha256": digest(HERE / "commands.candidate.final.json"),
        "verification": str(downloaded / "control/verification.json"),
        "verification_sha256": digest(downloaded / "control/verification.json"),
        "verification_archive_sha256": last["download_archive_sha256"],
        "tests": verification["test_count"], "preflight_exit_codes": {"stair": 0, "push": 0},
        "actual_model_calls": 0, "actual_gpu_runs": 0, "actual_capture_runs": 0,
        "source_unchanged_after_execution": verification["source_unchanged_after_tests_and_preflights"],
        "prior_artifacts_unchanged": all(record(HERE / "source" / name) == item for name, item in initial.items()),
        "run_contract_status": "NOT_RELEASED", "final_capture_card_created": False,
        "new_model_calibration": "PENDING; old GPT6-SOL calibration does not certify GPT-6.1-SOL",
        "known_limits": verification["limitations"],
        "local_resume": "Remote CPU validation and archive download succeeded. Resume verified local archive extraction with Windows extended paths; no remote commands rerun."
    })
    print(json.dumps({"status": "SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE",
                      "source_files": len(final), "source_manifest_sha256": digest(HERE / "source_manifest.final.json"),
                      "tests": verification["test_count"]}))


if __name__ == "__main__":
    main()
