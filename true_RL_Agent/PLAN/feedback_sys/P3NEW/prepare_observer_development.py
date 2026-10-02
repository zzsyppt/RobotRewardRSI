"""Prepare two old-media observer requests offline; never send them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from true_rl_agent.evaluation.visual_observer import (
    MediaImage, MediaWindow, ObserverBudget, call_observer,
    prepare_observer_request,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "observer_development_r1"
MODEL = "deepseek-flash"
FINGERPRINT = "aeb56401ca74e127821c4f9126dcb669"
PER_CALL_RESERVATION_USD = 0.0169152


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stair_windows():
    review = read_json(ROOT / "PLAN/feedback_sys/P3R/audit_response_20260929/visual_probe_model_review.json")
    windows, provenance = [], []
    for ordinal, episode in enumerate(review["episodes"], 1):
        image = episode["viewed_image"]
        image_path = ROOT / "PLAN/feedback_sys" / image["path"]
        data = image_path.read_bytes()
        if sha(data) != image["sha256"]:
            raise ValueError(f"Stair image changed: {image_path}")
        recording_path = ROOT / "PLAN/feedback_sys" / episode["source_manifest"]["path"]
        recording_bytes = recording_path.read_bytes()
        if sha(recording_bytes) != episode["source_manifest"]["sha256"]:
            raise ValueError(f"Stair recording metadata changed: {recording_path}")
        recording = json.loads(recording_bytes)["recordings"][0]
        metadata = read_json(recording_path)
        run = 1  # The four reviewed images and source recordings are explicitly r1.
        env = recording["env_index"]
        identity = {
            "session_id": "p3-review-video-20260924", "task_id": metadata["resolved_cell_manifest"]["task_id"],
            "robot_id": metadata["resolved_cell_manifest"]["robot_profile"]["robot_profile_id"],
            "protocol_id": metadata["execution"]["environment_source_preflight"]["task_protocol_hash"],
            "source_id": sha(recording_bytes),  # Replay recordings.json, never P3NEW numeric manifest.
            "checkpoint_id": metadata["execution"]["checkpoint_hash"],
            "episode_id": f"r{run}.e{env}", "run_id": str(run), "is_replay": True,
        }
        steps = tuple(image["readable_step_labels"])
        asset = MediaImage(data, 2560, 1080, steps)
        windows.append(MediaWindow(f"e{ordinal}", "fixed_camera.v1", "old_contact_sheet_4x3.v1",
                                   "old_contact_sheet.v1", ("root",), (asset,), identity=identity))
        provenance.append({"alias": f"e{ordinal}", "image_path": image_path.relative_to(ROOT).as_posix(),
                           "image_sha256": sha(data), "recordings_path": recording_path.relative_to(ROOT).as_posix(),
                           "recordings_sha256": sha(recording_bytes), "displayed_steps": steps,
                           "identity": identity, "media_kind": "old_replay_contact_sheet"})
    return windows, provenance


def push_windows():
    video_dir = ROOT / "PLAN/feedback_sys/P3R/results/checkpoint_evidence_r1/video_medium19_r2/video"
    image_path = video_dir / "medium19_run1_grid.png"
    recording_path = video_dir / "recordings.json"
    data, recording_bytes = image_path.read_bytes(), recording_path.read_bytes()
    if sha(data) != "1316752ae53f5416eed45a3945c1b048ec08fc39569da67cc91780cbb1114bf1":
        raise ValueError("Push grid changed")
    metadata = json.loads(recording_bytes)
    recording = next(row for row in metadata["recordings"] if row["run_index"] == 1)
    manifest = read_json(video_dir.parent / "composition" / "resolved_cell_manifest.json")
    identity = {
        "session_id": "p3r-video-medium19-r2", "task_id": manifest["task_id"],
        "robot_id": manifest["robot_profile"]["robot_profile_id"],
        "protocol_id": recording["resolved_protocol_hash"],
        "source_id": sha(recording_bytes),  # Replay recordings.json, never P3NEW numeric manifest.
        "checkpoint_id": metadata["checkpoint"]["checkpoint_hash"],
        "episode_id": f"r{recording['run_index']}.e{recording['env_index']}",
        "run_id": str(recording["run_index"]), "is_replay": True,
    }
    steps = (1, 68, 70, 71, 95, 109, 122, 179, 250, 350, 400, 411)
    asset = MediaImage(data, 2560, 1080, steps)
    window = MediaWindow("e1", "fixed_camera.v1", "old_contact_sheet_4x3.v1",
                         "old_contact_sheet.v1", ("root",), (asset,), identity=identity)
    provenance = {"alias": "e1", "image_path": image_path.relative_to(ROOT).as_posix(), "image_sha256": sha(data),
                  "recordings_path": recording_path.relative_to(ROOT).as_posix(), "recordings_sha256": sha(recording_bytes),
                  "displayed_steps": steps, "identity": identity, "media_kind": "old_replay_contact_sheet"}
    return [window], [provenance]


def write_package(name: str, windows, provenance):
    target = OUT / name
    target.mkdir(parents=True, exist_ok=True)
    prepared = prepare_observer_request(
        windows, model=MODEL, max_tokens=4096,
        generation_options={"thinking": {"type": "disabled"}})
    request, audit = prepared["request_bytes"], prepared["audit"]
    if not audit["passed"]:
        raise ValueError(f"{name} blind request audit failed")
    budget = ObserverBudget(max_pixels=12_000_000, max_image_bytes=4_000_000,
                            max_cost_usd=0.05, max_calls=1)
    reservation = budget.reserve(windows, PER_CALL_RESERVATION_USD)

    class OfflineTransport:
        calls = 0

        def post_json(self, *, url, payload, api_key, timeout_s):
            self.calls += 1
            if url != "offline://fake-transport" or payload != prepared["payload"]:
                raise ValueError("offline fake received different request")
            return {"model": MODEL, "system_fingerprint": "offline-fake", "usage": None}

    fake = OfflineTransport()
    fake_receipt = call_observer(
        prepared, windows, budget=ObserverBudget(max_pixels=12_000_000,
        max_image_bytes=4_000_000, max_cost_usd=0.05, max_calls=1),
        reserved_cost_usd=PER_CALL_RESERVATION_USD, url="offline://fake-transport",
        api_key="offline", receipt_sink=lambda _: None, transport=fake)
    if fake.calls != 1 or fake_receipt["request_sha256"] != sha(request):
        raise ValueError("offline call_observer rebuild check failed")
    (target / "request.json").write_bytes(request)
    (target / "private_sidecar.json").write_text(json.dumps({
        "provenance": provenance, "prepare_private_sidecar": prepared["private_sidecar"],
        "request_sha256": sha(request), "expected_fingerprint": FINGERPRINT,
        "same_run_numeric_claims": False, "replay_only": True,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "card.json").write_text(json.dumps({
        "status": "PREPARED_NOT_SENT", "model": MODEL, "thinking": "disabled",
        "max_tokens": 4096, "expected_fingerprint": FINGERPRINT,
        "request_sha256": sha(request), "request_bytes": len(request),
        "driver_sha256": sha(Path(__file__).read_bytes()),
        "visual_observer_sha256": sha((ROOT / "src/true_rl_agent/evaluation/visual_observer.py").read_bytes()),
        "source_dependencies": [{"path": row[key], "sha256": row[digest]}
                                for row in provenance for key, digest in
                                (("image_path", "image_sha256"), ("recordings_path", "recordings_sha256"))],
        "blind_text_audit": audit, "reservation": reservation,
        "media_source": "old_replay_contact_sheets", "new_renderer_used": False,
        "s0_count": 0, "measurement_claim_count": 0, "numeric_checker_used": False,
        "manual_pixel_review": "Only control-step labels were seen; no outcome text was seen in five original PNGs.",
        "offline_call_observer_rebuild": "PASS_FAKE_TRANSPORT_NO_NETWORK",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"name": name, "status": "PREPARED_NOT_SENT", "request_sha256": sha(request),
            "reservation": reservation}


def main():
    stair = write_package("stair", *stair_windows())
    push = write_package("push", *push_windows())
    (OUT / "card.json").write_text(json.dumps({
        "status": "PREPARED_NOT_SENT", "requests": [stair, push],
        "driver_sha256": sha(Path(__file__).read_bytes()),
        "visual_observer_sha256": sha((ROOT / "src/true_rl_agent/evaluation/visual_observer.py").read_bytes()),
        "total_calls": 2, "max_calls_per_task": 1,
        "reserved_cost_usd": 2 * PER_CALL_RESERVATION_USD,
        "total_cost_cap_usd": 0.05,
        "pricing_assumption_usd_per_million": {"input": 0.3, "output": 1.2},
        "token_assumption_per_call": {"input_at_most": 40000, "output_at_most": 4096},
        "actual_input_tokens": None, "sent": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
