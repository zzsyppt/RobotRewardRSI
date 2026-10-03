"""Replay the six parent-acceptance swaps with explicit real K4 raw contexts."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.tasks.push_recovery.trajectory_adapter import iter_push_episodes
from true_rl_agent.tasks.push_recovery.trajectory_card import (
    _render_claims, build_push_card, json_bytes, load_push_artifacts, project_push_card,
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resign(card):
    card["claims"] = _render_claims(card["domain"], card["cohorts"], card["examples"],
                                   card["selection"], card["parameters"], card["timing"])
    core = {k: v for k, v in card.items() if k not in {"card_id", "public_report", "projection_audit"}}
    card["card_id"] = stable_id("push-card", core)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lock = json.loads((ROOT / "PLAN/feedback_sys/NEW_SYS/P2b1/source_lock.json").read_bytes())
    contexts, cards, records, sources, raw_files = {}, {}, {}, {}, []
    for group in lock["groups"]:
        name = group["source_id"]
        contexts[name], sources[name] = [], []
        for source in group["sources"]:
            path = ROOT / source["manifest_path"]
            assert sha(path) == source["manifest_sha256"]
            manifest = json.loads(path.read_bytes())
            for file, expected in [(path, source["manifest_sha256"]),
                                   *((path.parent / c["file"], c["sha256"]) for c in manifest["chunks"])]:
                assert sha(file) == expected
                raw_files.append({"path": str(file.relative_to(ROOT)), "sha256": expected, "bytes": file.stat().st_size})
            # Reuse the externally locked identity, never infer it from a card.
            ref = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
                   "manifest_path": str(path), "manifest_sha256": source["manifest_sha256"]}
            contexts[name].extend(iter_push_episodes(ref, expected_identity=source["identity"]))
        out = ROOT / "outputs/p2b" / name
        cards[name] = json.loads((out / "policy_card.json").read_bytes())
        records[name] = [json.loads(line) for line in (out / "evidence.jsonl").read_text(encoding="utf-8").splitlines()]
        sources[name] = json.loads((out / "manifest.json").read_bytes())["sources"]

    results = []
    force, velocity = "k4_force_v1", "k4_velocity_v2"

    def check(name, card, source_id, key):
        resign(card)
        text = next(c["text"] for c in card["claims"] if c["key"] == key)
        try:
            project_push_card(card, episodes=contexts[source_id], source_id=source_id)
        except ValueError as exc:
            results.append({"case": name, "accepted": False, "exception_type": "ValueError",
                            "error": str(exc), "resigned": True, "real_raw_context": True,
                            "rerendered_text": text})
        else:
            raise AssertionError("semantic swap was accepted: " + name)

    card = deepcopy(cards[force])
    pose = card["cohorts"]["all"]["pose"]
    pose["planar_speed_mps"], pose["height_m"] = pose["height_m"], pose["planar_speed_mps"]
    check("swap_height_m_and_speed_mps_operands", card, force, "planar_speed_mps")
    key = next(c["key"] for c in cards[velocity]["claims"] if c["key"].startswith("event:") and "r1.e46" in c["text"])
    for case in ("unit", "pre_post", "direction_source"):
        card = deepcopy(cards[velocity])
        example = next(r for r in card["examples"] if (r["value"]["run_index"], r["value"]["env_index"]) == (1, 46))
        event = example["value"]["events"][0]
        if case == "unit":
            event["vector_unit"] = "N"
        elif case == "pre_post":
            windows = event["windows"]
            windows["pre"], windows["post"] = windows["post"], windows["pre"]
        else:
            event["root_along_disturbance"]["evidence_ref"]["chunks"][0]["env_index"] = 34
        check(case, card, velocity, key)
    card = deepcopy(cards[velocity])
    card["domain"]["resolved_protocol_hash"] = cards[force]["domain"]["resolved_protocol_hash"]
    check("velocity_card_bound_to_force_protocol", card, velocity, "scope")
    swapped = deepcopy(records[force])
    for record in swapped:
        pose = record["value"]["pose"]
        pose["height_m"], pose["planar_speed_mps"] = pose["planar_speed_mps"], pose["height_m"]
        record["record_id"] = stable_id("push-evidence", record["value"]["episode_id"], record["measurement_config"], record["value"])
    try:
        build_push_card(swapped, force, episodes=contexts[force])
    except ValueError as exc:
        results.append({"case": "swap_record_measurements_reid_then_build_card", "accepted": False,
                        "exception_type": "ValueError", "error": str(exc), "resigned": True, "real_raw_context": True})
    else:
        raise AssertionError("swapped evidence was accepted")
    originals = []
    for name in (force, velocity):
        card = build_push_card(records[name], name, episodes=contexts[name])
        assert card == cards[name]
        assert project_push_card(card, episodes=contexts[name], source_id=name) == card["public_report"]
        manifest = ROOT / "outputs/p2b" / name / "manifest.json"
        assert load_push_artifacts(manifest, manifest_sha256=sha(manifest), expected_sources=sources[name]) == card
        originals.append({"group": name, "episodes": len(contexts[name]), "card_id": card["card_id"],
                          "claims": len(card["claims"]), "public_bytes": len(json_bytes(card["public_report"])), "accepted": True})
    result = {"status": "PASS", "scope": "production semantic binding, not independent numerical oracle",
              "counterexamples": results, "originals": originals, "unchanged_raw_files": raw_files}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(json_bytes(result) + b"\n")
    print(json.dumps({"status": "PASS", "rejected_semantic_swaps": len(results), "originals": originals,
                      "raw_files_verified": len(raw_files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
