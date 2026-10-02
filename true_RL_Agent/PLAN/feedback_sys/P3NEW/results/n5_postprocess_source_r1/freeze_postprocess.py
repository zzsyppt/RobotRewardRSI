"""Copy existing frozen bytes into an independent CPU postprocess source tree."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[4]
RESULTS = HERE.parent
RUNTIME = RESULTS / "n5_source_preflight_r4"
FINAL = RESULTS / "n5_final_freeze_r1"
SOURCE = HERE / "source"
DEPS = ("PLAN/feedback_sys/P3R/confirmation_context.py",
        "PLAN/feedback_sys/P3R/author_ceiling_probe.py",
        "PLAN/feedback_sys/P3/author_experiment.py")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), size_bytes=path.stat().st_size)


def save(name, value):
    with (HERE / name).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    runtime_path = RUNTIME / "source_manifest.json"
    final_path = FINAL / "experiment_files.final.json"
    runtime = json.loads(runtime_path.read_bytes())
    final = json.loads(final_path.read_bytes())
    remote = json.loads((HERE / "remote_dependency_check.stdout.json").read_bytes())
    assert len(runtime["files"]) == 400 and len(final["files"]) == 46
    assert not remote["target_exists"]
    rows = {}
    for relative, digest in runtime["files"].items():
        origin = RUNTIME / "source" / relative
        assert sha(origin) == digest, relative
        rows[relative] = dict(sha256=digest, size_bytes=origin.stat().st_size,
                              origin=ref(origin), source_groups=["r4_runtime"])
    overlaps = []
    for row in final["files"]:
        relative = row["repository_relative_path"]
        assert row["snapshot"].startswith("frozen_files_final/")
        origin = FINAL / row["snapshot"]
        assert sha(origin) == row["sha256"] and origin.stat().st_size == row["bytes"]
        if relative in rows:
            assert rows[relative]["sha256"] == row["sha256"], relative
            rows[relative]["source_groups"].append("final_frozen_experiment")
            overlaps.append(relative)
        else:
            rows[relative] = dict(sha256=row["sha256"], size_bytes=row["bytes"],
                                  origin=ref(origin), source_groups=["final_frozen_experiment"])
    remote_deps = {row["repository_relative_path"]: row for row in remote["files"]}
    assert set(remote_deps) == set(DEPS)
    for relative in DEPS:
        origin = PROJECT / relative
        assert relative not in rows
        assert sha(origin) == remote_deps[relative]["sha256"]
        assert origin.stat().st_size == remote_deps[relative]["size_bytes"]
        rows[relative] = dict(sha256=sha(origin), size_bytes=origin.stat().st_size,
            origin=ref(origin), source_groups=["existing_dependency_omission_supplement"],
            original_remote_reference=remote_deps[relative])
    assert len(rows) == 445 and len(overlaps) == 4
    SOURCE.mkdir(exist_ok=False)
    for relative, row in rows.items():
        dest = SOURCE / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["origin"]["path"], dest)
        assert sha(dest) == row["sha256"]
    assert all(sha(row["origin"]["path"]) == row["sha256"] for row in rows.values())
    save("source_manifest.json", dict(schema_version="p3new.cpu_postprocess_source.v1",
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        purpose="Independent CPU postprocess source; existing-byte copy, no capture or model release",
        runtime_source_manifest=ref(runtime_path), final_experiment_manifest=ref(final_path),
        remote_source_root="/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_postprocess_source_r1",
        files={relative: rows[relative] for relative in sorted(rows)}, file_count=len(rows),
        overlaps=overlaps, production_changes=0, dependency_code_changes=0,
        provider_requests=0, gpu_jobs=0, new_n5_results_read=False))
    bundle = HERE / "source.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for relative in sorted(rows):
            archive.add(SOURCE / relative, arcname=relative, recursive=False)
    with tarfile.open(bundle, "r:gz") as archive:
        assert {m.name for m in archive.getmembers()} == set(rows)
        assert all(hashlib.sha256(archive.extractfile(p).read()).hexdigest() == row["sha256"]
                   for p, row in rows.items())
    save("local_freeze_receipt.json", dict(status="EXISTING_BYTES_COPIED_CPU_ONLY",
        source_manifest=ref(HERE / "source_manifest.json"), bundle=ref(bundle),
        file_count=len(rows), runtime_files=400, final_experiment_files=46, overlap_files=4,
        supplemented_dependency_files=3, all_origin_and_copy_hashes_match=True,
        source_frozen_runtime_not_modified=True, script=ref(Path(__file__))))
    print(json.dumps(dict(source_manifest=ref(HERE / "source_manifest.json"),
                          bundle=ref(bundle), files=len(rows))))


if __name__ == "__main__":
    main()
