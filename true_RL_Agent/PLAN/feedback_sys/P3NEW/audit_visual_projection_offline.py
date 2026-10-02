"""Synthetic capacity/binding regression, never visual or semantic qualification."""
from copy import deepcopy
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.evaluation.visual_observer import (
    LAYOUT_VERSION, SCHEMA_VERSION, MAX_OBSERVATION_CHARS, MAX_UNCERTAINTY_CHARS,
    MediaImage, MediaWindow, audit_observer_output,
)
from true_rl_agent.tasks.stair_climbing.process_report import derive_stair_process_behavior, project_stair_process_report
from true_rl_agent.tasks.push_recovery.process_report import build_push_process_behavior, project_push_process_behavior
from true_rl_agent.tasks.stair_climbing.visual_observation import attach_stair_visual_observations
from true_rl_agent.tasks.push_recovery.visual_observation import attach_push_visual_observations


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def baseline_full(task):
    p3r = ROOT / "PLAN/feedback_sys/P3R/results"
    if task == "stair":
        return derive_stair_process_behavior(read(p3r / "stair_motion_v2_full.json"))
    phase = read(p3r / "context_dev_originals_r1/push/public_dev_result.json")
    summaries = phase["task_evidence"]["seed_summaries"]
    for i, summary in enumerate(summaries, 1):
        summary["numeric_evidence_ref"]["manifest_path"] = str(p3r / f"push_numeric_dev_r1/numeric/public_dev-{i:02}/manifest.json")
    return build_push_process_behavior(summaries, legacy_full=read(p3r / "push_behavior_r1_full.json"))


def synthetic_audit(task, full):
    if task == "stair":
        episode = full["episodes"][0]
        identity = dict(source_id=full["domain"]["archive_hash"], checkpoint_id=full["domain"]["checkpoint_hash"],
                        protocol_id=full["domain"]["task_protocol_hash"], run_id=str(episode["run"]), episode_id=episode["episode_id"])
        claim = "root_small_x_endpoint_delta"
    else:
        source, episode = full["source_bindings"][0], full["transcripts"][0]
        identity = dict(source_id=source["manifest_sha256"], checkpoint_id=full["binding"]["checkpoint_hash"],
                        protocol_id=full["binding"]["resolved_protocol_hash"], run_id=str(source["actual_run_index"]),
                        episode_id=f"r{source['actual_run_index']}.e{episode['env_id']}")
        claim = "root_small_xy_endpoint_delta"
    identity.update(session_id="SYNTHETIC_CAPACITY_ONLY", task_id=task, robot_id="SYNTHETIC_ROBOT")
    identity.update({"numeric_" + key: value for key, value in list(identity.items())})
    last = episode["steps"] if task == "stair" else episode["final_control_step"]
    steps = tuple(1 + (last - 1) * index // 5 for index in range(6))
    refs = [{"step": step, "view_id": view} for step in steps for view in ("v0", "v1")]
    observations = []
    # Maximum-length UTF-8 prose tests byte pressure, not useful semantics.
    for i in range(3):
        observations.append({"body": "root", "spatial_visibility": "visible", "relation": "state",
            "evidence_refs": deepcopy(refs),
            "text": "形" * MAX_OBSERVATION_CHARS, "uncertainty": "限" * MAX_UNCERTAINTY_CHARS})
    observations[0]["numeric_claim"] = {"id": claim, "asserted": True}
    # Twelve individual full-resolution PNGs exercise the native reference shape.
    # Solid synthetic pixels do not test semantics or provider image-byte limits.
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    images = []
    for index, ref in enumerate(refs):
        scanlines = (b"\0" + bytes((20 + index, 50, 100)) * 1920) * 1080
        png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 1920, 1080, 8, 2, 0, 0, 0))
               + chunk(b"IDAT", zlib.compress(scanlines)) + chunk(b"IEND", b""))
        images.append(MediaImage(png, 1920, 1080, (ref["step"],), ref["view_id"]))
    media = MediaWindow("e1", "synthetic_camera", LAYOUT_VERSION, "synthetic_full_frame", ("root",),
        tuple(images), identity=identity, measurement_claims=({"id": claim, "body": "root",
            "definition": "Synthetic polarity/rendering fixture only; not a measured predicate."},))
    raw = {"episodes": [{"alias": "e1", "observations": observations, "not_visible": []}]}
    audit = audit_observer_output(raw, [media], numeric_check=lambda observation, window: {
        "status": "conflict", "numeric_evidence": {"synthetic_only": True, "not_a_measurement": True}})
    assert audit["passed"] and len(audit["observations"]) == 3
    assert all(len(row["image_evidence"]) == 12 for row in audit["observations"])
    audit["synthetic_only"] = True
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True, help="New result directory under P3NEW.")
    output = parser.parse_args().output.resolve()
    assert output.is_relative_to(Path(__file__).parent.resolve()) and not output.exists(), "use a new P3NEW result directory"
    output.mkdir(parents=True, exist_ok=False)
    path = Path(__file__).with_name("audit_delivery_offline.py")
    spec = importlib.util.spec_from_file_location("n4_synthetic_fixture", path)
    delivery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(delivery)
    results = {}
    for task, project, attach, limit in (
        ("stair", project_stair_process_report, attach_stair_visual_observations, 8192),
        ("push", project_push_process_behavior, attach_push_visual_observations, 12288),
    ):
        full = baseline_full(task)
        before = canonical_json_hash(full)
        public = project(full)
        audit = synthetic_audit(task, full)
        augmented = attach(full, audit, observer_version=SCHEMA_VERSION + ".synthetic_capacity")
        assert canonical_json_hash(full) == before
        assert augmented["episodes" if task == "stair" else "transcripts"] == full["episodes" if task == "stair" else "transcripts"]
        numeric = deepcopy(augmented)
        numeric.pop("observer_evidence")
        numeric.pop("observer_source_binding")
        if task == "stair":
            numeric["semantic"]["sentences"] = full["semantic"]["sentences"]
            numeric["semantic"]["sampling"]["visual_observed_episode_n"] = full["semantic"]["sampling"]["visual_observed_episode_n"]
        else:
            numeric["sentences"] = full["sentences"]
            numeric["evidence_catalog"] = full["evidence_catalog"]
        assert canonical_json_hash(numeric) == before, "observer changed L1/L2 numerical evidence"
        view = project(augmented)
        with_parent, difference = delivery.synthetic_s4(task, view)
        assert report_bytes(with_parent) <= limit
        assert with_parent["report"]["counterexample"] == public["report"]["counterexample"]
        assert with_parent["report"]["gaps"] == public["report"]["gaps"]
        assert len(augmented["observer_evidence"]["observations"]) == 3
        assert view["report"]["visual"][0]["text"].count("形") == MAX_OBSERVATION_CHARS
        assert view["report"]["visual"][0]["text"].count("限") == MAX_UNCERTAINTY_CHARS
        # Exercise the real delivery resolver: one parent reference retains the
        # complete raw observation and can resolve every original image child.
        parent = "observer_evidence/observations/0"
        resolved = delivery.resolve_refs(task, augmented, [parent, parent + "/image_evidence/*"])
        assert [item["resolved_count"] for item in resolved] == [1, 12]
        record = augmented["observer_evidence"]["observations"][0]
        assert record["raw_observation"] == audit["raw_output"]["episodes"][0]["observations"][0]
        assert resolved[1]["resolved"] == [
            {"pointer": f"/{parent}/image_evidence/{index}", "sha256": canonical_json_hash(image)}
            for index, image in enumerate(record["image_evidence"])]
        results[task] = {"status": "PASS", "numeric_evidence_hash_unchanged": before,
            "public_bytes": report_bytes(public), "with_visual_bytes": report_bytes(view),
            "with_visual_and_synthetic_parent_bytes": report_bytes(with_parent), "limit": limit,
            "all_observation_n": 3, "displayed_observation_n": 1,
            "native_shape": {"steps": 6, "views": 2, "images": 12, "width": 1920, "height": 1080},
            "image_references_per_observation": 12, "all_image_children_resolve_from_parent": True,
            "text_characters": MAX_OBSERVATION_CHARS, "uncertainty_characters": MAX_UNCERTAINTY_CHARS,
            "counterexample_and_residual_unchanged": True, "parent_pair_is_synthetic": True}
    sources = [ROOT / "src/true_rl_agent" / item for item in (
        "evaluation/visual_observer.py", "evaluation/process_report.py",
        "tasks/stair_climbing/visual_observation.py", "tasks/push_recovery/visual_observation.py",
        "tasks/stair_climbing/process_report.py", "tasks/push_recovery/process_report.py")]
    result = {"schema_version": "feedback.p3new.visual_capacity.v1", "synthetic_only": True,
              "observer_schema_version": SCHEMA_VERSION,
              "semantic_or_visual_qualification": False, "network_calls": 0, "new_simulations": 0,
              "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "results": results}
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
