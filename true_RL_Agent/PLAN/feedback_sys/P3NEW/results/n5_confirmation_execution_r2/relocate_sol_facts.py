"""Move one actual R2 facts packet without changing its model-visible content.

export runs on the server; prepare runs on Windows. Both are offline. A ZIP
contains only the explicit materials, original manifest/card and transfer map.
Report manifests are built later by the frozen prepare_report_manifest helper
from the local facts manifest; this program deliberately does not rebind them.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return {"file": str(path), "sha256": digest(raw), "size_bytes": len(raw)}


def export_packet(manifest_path, prepared, archive):
    manifest_path, prepared, archive = map(lambda p: Path(p).resolve(), (manifest_path, prepared, archive))
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    card_raw = (prepared / "prepare.json").read_bytes()
    card = json.loads(card_raw)
    require(manifest["stage"] == card["stage"] == "facts", "Only original facts packets may be relocated")
    require(card["judge_id"] == "gpt6-sol" and card["packet_id"] == manifest["packet_id"], "Not the corresponding SOL preparation")
    require(card["status"] == "PREPARED_NOT_RELEASED", "Not a genuine prepared facts packet")
    require(card["manifest"]["sha256"] == digest(manifest_raw), "Remote manifest differs from its preparation")
    require(card["request_sha256"] == digest((prepared / "request.http_body.json").read_bytes()), "Remote request differs from its preparation")
    require(card["own_facts_seal"] is None and card["capacity_facts_fixture"] is None, "Facts must not contain prior answers")
    require(len(manifest["materials"]) == len(card["materials"]), "Prepared material count differs")
    rows, members = [], {}
    for index, (item, prepared_item) in enumerate(zip(manifest["materials"], card["materials"])):
        path = Path(item["file"])
        path = (manifest_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        raw = path.read_bytes()
        require(digest(raw) == item["sha256"] == prepared_item["sha256"], "Material hash differs")
        require(item["id"] == prepared_item["id"] and item["role"] == prepared_item["role"], "Prepared material order differs")
        suffix = {"application/json": ".json", "image/png": ".png"}[item["mime_type"]]
        member = "materials/" + item["sha256"] + suffix
        members[member] = raw
        rows.append({"index": index, "id": item["id"], "source_file": str(path),
                     "archive_member": member, "sha256": item["sha256"], "size_bytes": len(raw)})
    transfer = {"schema_version": "p3new.sol_facts_transfer.v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                "packet_id": manifest["packet_id"], "remote_manifest": reference(manifest_path),
                "remote_prepared": reference(prepared / "prepare.json"),
                "request_sha256": card["request_sha256"], "materials": rows,
                "export_script": reference(__file__), "model_calls": 0}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        bundle.writestr("remote.manifest.json", manifest_raw)
        bundle.writestr("remote.prepare.json", card_raw)
        bundle.writestr("transfer.json", encoded(transfer))
        for name, raw in members.items():
            bundle.writestr(name, raw)
    receipt = {"status": "EXPORTED_NO_MODEL_CALL", "archive": reference(archive), "packet_id": manifest["packet_id"],
               "request_sha256": card["request_sha256"], "material_count": len(rows), "model_calls": 0}
    Path(str(archive) + ".receipt.json").write_bytes(encoded(receipt))
    return receipt


def prepare_local(archive, source_root, output):
    archive, source_root, output = map(lambda p: Path(p).resolve(), (archive, source_root, output))
    require(not output.exists(), "Local packet output must be new; failed attempts are retained")
    runner = source_root / "PLAN/feedback_sys/P3NEW/run_confirmation_review.py"
    with zipfile.ZipFile(archive, "r") as bundle:
        remote_raw = bundle.read("remote.manifest.json")
        remote = json.loads(remote_raw)
        remote_card_raw = bundle.read("remote.prepare.json")
        remote_card = json.loads(remote_card_raw)
        transfer_raw = bundle.read("transfer.json")
        transfer = json.loads(transfer_raw)
        require(remote["stage"] == remote_card["stage"] == "facts" and remote_card["judge_id"] == "gpt6-sol", "Only SOL facts are allowed")
        require(digest(remote_raw) == transfer["remote_manifest"]["sha256"] == remote_card["manifest"]["sha256"], "Original manifest hash differs")
        require(digest(remote_card_raw) == transfer["remote_prepared"]["sha256"], "Original prepare card hash differs")
        require(digest(runner.read_bytes()) == remote_card["runner"]["sha256"], "Local frozen runner differs from remote")
        require(len(remote["materials"]) == len(transfer["materials"]), "Transfer material count differs")
        output.mkdir(parents=True)
        (output / "materials").mkdir()
        (output / "remote.manifest.json").write_bytes(remote_raw)
        (output / "remote.prepare.json").write_bytes(remote_card_raw)
        (output / "transfer.json").write_bytes(transfer_raw)
        local = deepcopy(remote)
        for index, (item, row) in enumerate(zip(local["materials"], transfer["materials"])):
            suffix = {"application/json": ".json", "image/png": ".png"}[item["mime_type"]]
            member = "materials/" + item["sha256"] + suffix
            require(row["index"] == index and row["id"] == item["id"] and row["archive_member"] == member, "Transfer order or member differs")
            raw = bundle.read(member)
            require(digest(raw) == item["sha256"] == row["sha256"] and len(raw) == row["size_bytes"], "Original material bytes differ")
            target = output / member
            if not target.exists():
                target.write_bytes(raw)
            item["file"] = member
        restored = deepcopy(local)
        for item, original in zip(restored["materials"], remote["materials"]):
            item["file"] = original["file"]
        require(restored == remote, "A field other than materials[].file changed")
    manifest = output / "facts.manifest.json"
    manifest.write_bytes(encoded(local))
    argv = [sys.executable, "-B", str(runner), "prepare", "--manifest", str(manifest),
            "--judge-id", "gpt6-sol", "--output", str(output / "prepared")]
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(source_root / "src"), str(source_root), str(runner.parent)))
    result = subprocess.run(argv, cwd=source_root, env=environment, capture_output=True)
    (output / "prepare.stdout.log").write_bytes(result.stdout)
    (output / "prepare.stderr.log").write_bytes(result.stderr)
    receipt = {"schema_version": "p3new.sol_facts_relocation.v1", "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
               "packet_id": remote["packet_id"], "archive": reference(archive), "script": reference(__file__),
               "remote_manifest": transfer["remote_manifest"], "local_manifest": reference(manifest),
               "only_manifest_material_file_fields_changed": True, "material_bytes_and_order_unchanged": True,
               "prepare_argv": argv, "prepare_exit_code": result.returncode, "model_calls": 0,
               "remote_request_sha256": remote_card["request_sha256"], "status": "PREPARE_FAILED"}
    if result.returncode == 0:
        local_card = json.loads((output / "prepared/prepare.json").read_bytes())
        actual_sha = digest((output / "prepared/request.http_body.json").read_bytes())
        equal = actual_sha == local_card["request_sha256"] == remote_card["request_sha256"] == transfer["request_sha256"]
        modules_equal = all(local_card[key]["sha256"] == remote_card[key]["sha256"] for key in ("runner", "transport", "coverage_helper"))
        receipt.update(local_request_sha256=actual_sha, request_sha256_equal=equal,
                       frozen_prepare_modules_equal=modules_equal,
                       status="PREPARED_IDENTICAL_NO_MODEL_CALL" if equal and modules_equal else "PREPARATION_MISMATCH")
    (output / "relocation_receipt.json").write_bytes(encoded(receipt))
    require(receipt["status"] == "PREPARED_IDENTICAL_NO_MODEL_CALL", "Local request preparation failed or differs; inspect retained receipt")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Server: archive only one explicit facts material closure")
    export.add_argument("--manifest", required=True, type=Path)
    export.add_argument("--prepared", required=True, type=Path)
    export.add_argument("--archive", required=True, type=Path)
    prepare = commands.add_parser("prepare", help="Windows: relocate paths and run frozen offline prepare")
    prepare.add_argument("--archive", required=True, type=Path)
    prepare.add_argument("--source-root", required=True, type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = (export_packet(args.manifest, args.prepared, args.archive) if args.command == "export"
              else prepare_local(args.archive, args.source_root, args.output))
    print(json.dumps({key: result[key] for key in ("status", "packet_id", "model_calls")}))


if __name__ == "__main__":
    main()
