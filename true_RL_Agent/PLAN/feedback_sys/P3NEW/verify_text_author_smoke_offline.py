"""Reconstruct four frozen author requests on the deployment host; no network."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch


BASE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("text_smoke_preparation", BASE / "prepare_text_author_smoke.py")
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def verify(stage_path, receipt_path):
    from true_rl_agent.runner.source_snapshot import project_source_snapshot

    stage = prep.D.read(stage_path)
    assert stage["status"] == "PREPARED_NOT_RELEASED", "this check cannot release or run a card"
    for reference in stage["frozen_inputs"]:
        prep.check(reference)
    snapshot = project_source_snapshot(prep.ROOT)
    assert snapshot == prep.D.read(prep.check(stage["production_source_snapshot"]))
    prep.verify_current_push_materials()
    secret = Path("/data/yuyangliu/secrets/deepseek.env")
    assert sys.platform == "linux" and sys.version_info[:2] == (3, 12)
    assert str(Path(sys.executable).absolute()) == stage["execution"]["python"]
    assert secret.is_file() and os.access(secret, os.R_OK), "registered secret file is not readable"
    rows = []
    # The provider serializer's fake opener still runs; real socket I/O cannot.
    with patch.object(socket.socket, "connect", side_effect=AssertionError("network forbidden in offline verification")):
        for row in stage["episodes"]:
            comp, builder, _ = prep.context_for(row["case_id"], receipt_path.parent / "offline_contexts" / row["case_id"])
            payload = prep.R.first_prompt(comp, builder)
            body, user, delivered = prep.D.capture_request(payload, receipt_path.parent)
            assert prep.D.digest(body) == row["initial_http_body"]["sha256"], row["case_id"]
            assert prep.D.digest(user) == row["initial_payload"]["sha256"], row["case_id"]
            assert prep.D.behavior_in(delivered, row["task"]) == prep.D.read(prep.check(row["report_input"]))
            rows.append({"case_id": row["case_id"], "http_sha256": prep.D.digest(body),
                         "http_bytes": len(body), "payload_sha256": prep.D.digest(user), "identical_to_local": True})
    assert project_source_snapshot(prep.ROOT) == snapshot, "source drift during offline verification"
    receipt = {"status": "PASS_REMOTE_OFFLINE_PREPARATION", "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(prep.ROOT), "host": socket.gethostname(), "python": sys.executable, "python_version": sys.version,
        "stage_ref": prep.artifact(stage_path), "source_snapshot_ref": stage["production_source_snapshot"],
        "frozen_files_verified": len(stage["frozen_inputs"]), "production_source_files_verified": len(snapshot["files"]),
        "current_push_full_phase_and_projector_verified": True,
        "current_push_phase_ref": prep.artifact(prep.material_dir("push") / "public_dev_result.json"),
        "current_push_full_ref": prep.artifact(prep.material_dir("push") / "behavior_full.json"),
        "credential_check": "registered file is regular and readable; contents not opened", "secret_contents_read": False,
        "stage_status": stage["status"], "real_network_socket_connect_blocked": True,
        "api_calls": 0, "gpu_work": "NOT_RUN", "training": "NOT_RUN", "episodes": rows}
    prep.D.write(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.receipt.exists():
        raise ValueError("receipt already exists; inspect it instead of overwriting evidence")
    result = verify(args.stage.resolve(), args.receipt.resolve())
    print(json.dumps(result, ensure_ascii=False))
