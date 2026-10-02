"""CPU mechanical fixtures using old data; not N5 capture or model review."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1]
ROOT = TOOLS.parents[2]
sys.path.insert(0, str(TOOLS))
import export_confirmation_raw as raw
import prepare_confirmation_media as media
import run_confirmation_review as runner

read, save, ref = media.read, media.save, runner.ref
checks = []


def passed(name, **details):
    checks.append({"name": name, "passed": True, **details})


def rejects(name, action):
    try:
        action()
    except (ValueError, KeyError) as error:
        passed(name, error=str(error))
    else:
        raise AssertionError(name)


def main():
    paths = [ROOT / "PLAN/feedback_sys/P3R/results/stair_numeric_dev_r1/numeric_evidence/manifest.json"] + [
        ROOT / f"PLAN/feedback_sys/P3R/results/push_numeric_dev_r1/numeric/public_dev-{i:02d}/manifest.json" for i in range(1, 4)]
    manifests = [raw.read_manifest(p) for p in paths]
    protocol = TOOLS / "N5_FACT_REVIEW_PROTOCOL_ZH.md"
    card = {"status": "FROZEN", "synthetic_only": True, "not_N5": True,
            "review_protocol": ref(protocol), "tasks": {}}
    for task, manifest, envs in zip(("stair", "push"), manifests[:2], (288, 64)):
        card["tasks"][task] = {"checkpoint": {"sha256": manifest["identity"]["checkpoint_hash"].removeprefix("sha256:")},
            "current_evaluation_binding": manifest["identity"]["evaluation_binding"],
            "numeric_eval_envs": envs, "numeric_public_dev_seed_runs": 3,
            "cohort": {"namespace": manifest["identity"].get("seed_namespace")}}
    raw.verify_confirmation_sources(manifests, card)
    passed("old_raw_metadata_with_explicit_synthetic_frozen_card")
    for label, mutate in (
        ("wrong_checkpoint", lambda m: m[0]["identity"].__setitem__("checkpoint_hash", "sha256:" + "0"*64)),
        ("old_cohort_under_new_binding", lambda m: m[1]["identity"]["evaluation_binding"].__setitem__("seed_cohort_hash", "0"*64)),
        ("wrong_push_run_order", lambda m: m.__setitem__(slice(1, 3), [m[2], m[1]])),
        ("wrong_push_common_source", lambda m: m[2]["identity"].__setitem__("evaluation_plan_hash", "0"*64)),
        ("incomplete_stair_envs", lambda m: m[0]["metadata"]["seeds"][0]["outcomes"].pop()),
        ("misordered_push_outcomes", lambda m: m[1]["metadata"]["runtime_episode_records"].reverse()),
    ):
        changed = deepcopy(manifests)
        mutate(changed)
        rejects(label, lambda: raw.verify_confirmation_sources(changed, card))

    # Generated data stays under this new development directory. Temp cleanup is
    # restricted to its verified child; no frozen source/artifact is modified.
    with tempfile.TemporaryDirectory(prefix="fixture_", dir=HERE) as name:
        temp = Path(name).resolve()
        assert temp.parent == HERE
        card_path = temp / "synthetic_capture_card.json"
        save(card_path, card)
        args = SimpleNamespace(purpose="confirmation", capture_card=card_path, review_protocol=protocol)
        bad_card = deepcopy(card)
        bad_card["status"] = "DRAFT_NOT_RELEASED"
        save(card_path, bad_card)
        rejects("raw_export_rejects_unfrozen_card", lambda: raw.export(args))
        bad_card["status"] = "FROZEN"
        bad_card["review_protocol"]["sha256"] = "0" * 64
        save(card_path, bad_card)
        rejects("raw_export_rejects_changed_protocol", lambda: raw.export(args))
        save(card_path, card)
        out = temp / "raw_export"
        command = [sys.executable, "-B", str(TOOLS / "export_confirmation_raw.py"), "--stair", str(paths[0]),
            "--push", *map(str, paths[1:]), "--output", str(out), "--purpose", "confirmation", "--profile", "related_v1",
            "--capture-card", str(card_path), "--review-protocol", str(protocol)]
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
        (HERE / "raw_export.stdout.txt").write_text(result.stdout, encoding="utf-8")
        exported = read(out / "manifest.json")
        assert exported["selected_count"] == 24 and exported["focus_count"] == 12
        assert exported["pool_counts"] == {"stair": 864, "push": 192}
        assert exported["frozen_inputs"]["capture_card"]["sha256"] == raw.sha(card_path)
        save(HERE / "synthetic_raw_export_manifest.json", exported)
        passed("real_old_npz_full_export_synthetic_formal", selected=24, focuses=12,
               original_values_checked=exported["selected_raw_values_roundtrip_checked"], not_N5=True)

        # Test just the newly added formal orchestration with the existing real
        # serialized development packet. build_unit is stubbed: this is not a
        # second dense-media/source or simulation validation.
        old = TOOLS / "results/confirmation_media_rootreview_r1/materials/media-03"
        export = temp / "media_source"
        export.mkdir()
        episode = {"run_index": 1, "env_index": 19,
            "episode_outcome": manifests[1]["metadata"]["runtime_episode_records"][19],
            "numeric_reference": {"manifest_path": str(paths[1]), "manifest_sha256": raw.sha(paths[1])}}
        save(export / "extraction.json", {"task": "push", "episodes": [episode]})
        full = old / "full_episode.json"
        steps = read(old / "observer_selection.json")["episodes"][0]["selected_steps"]
        selections = {}
        for task in ("stair", "push"):
            p = temp / (task + ".selection.json")
            save(p, {"scope": "n5_confirmation", "source": {"capture_card_sha256": raw.sha(card_path),
                "extraction_sha256": raw.sha(export / "extraction.json"), "full_sha256": raw.sha(full)},
                "episodes": [] if task == "stair" else [{"alias": "e1" if episode["episode_outcome"]["success"] else "e2",
                    "run_index": 1, "env_index": 19, "selected_steps": steps}]})
            selections[task] = ref(p)
        config = {"purpose": "confirmation", "capture_card": ref(card_path), "selections": selections,
            "units": [{"id": "synthetic-media-01", "export": str(export), "run_index": 1, "env_index": 19, "full": ref(full)}]}
        inputs = temp / "inputs.json"

        def old_packet_builder(unit, output, protocol_material, *, purpose):
            assert purpose == "confirmation"
            dest = output / unit["id"]
            dest.mkdir()
            materials = [protocol_material]
            for m in read(old / "facts.manifest.json")["materials"][1:]:
                suffix = m["file"].split("/media-03/", 1)[1]
                materials.append({**m, **ref(old / suffix)})
            facts = {"schema_version": "p3new.confirmation_review_materials.v1", "packet_id": unit["id"],
                "stage": "facts", "purpose": purpose, "max_output_tokens": 32768, "materials": materials}
            save(dest / "facts.manifest.json", facts)
            return dest, "push", episode, {}, {}, {"selected_steps": steps}, facts

        save(inputs, config)
        with patch.object(media, "build_unit", old_packet_builder):
            receipt = media.prepare(inputs, temp / "formal_media", protocol)
        assert receipt["status"] == "CONFIRMATION_FACTS_PREPARED_NOT_RELEASED" and receipt["actual_calls"] == 0
        assert receipt["missing_outcome_aliases"]["stair"] == ["e1", "e2"]
        assert receipt["units"][0]["packet_alias"] == "e1"
        prepared = temp / "formal_media/synthetic-media-01/prepared_facts"
        bodies = [read(prepared / judge / "request.http_body.json") for judge in ("deepseek", "gpt6-sol")]
        assert bodies[0]["messages"][1]["content"] == bodies[1]["messages"][1]["content"]
        assert all("report_text_inventory" not in json.dumps(b) for b in bodies)
        save(HERE / "synthetic_media_receipt.json", receipt)
        passed("formal_media_orchestration_old_packet_builder_stub", shared_original_pngs=12,
               identical_judge_evidence=True, facts_only=True, no_calls=True, not_dense_capture_validation=True)
        bad = deepcopy(config)
        bad["units"][0]["observer_directory"] = "historical_answer"
        save(inputs, bad)
        rejects("formal_media_rejects_historical_answers", lambda: media.prepare(inputs, temp / "bad_old_answer", protocol))
        bad = deepcopy(config)
        p = Path(bad["selections"]["push"]["file"])
        value = read(p)
        value["episodes"][0]["alias"] = "e2" if value["episodes"][0]["alias"] == "e1" else "e1"
        save(p, value)
        bad["selections"]["push"] = ref(p)
        save(inputs, bad)
        rejects("formal_media_rejects_wrong_outcome_alias", lambda: media.prepare(inputs, temp / "bad_alias", protocol))
    save(HERE / "verification.json", {"status": "MECHANICAL_DEVELOPMENT_PASS_NOT_N5", "checks": checks,
        "api_calls": 0, "gpu_calls": 0, "original_inputs": [ref(p) for p in paths],
        "sources": [ref(TOOLS / f) for f in ("export_confirmation_raw.py", "prepare_confirmation_media.py")],
        "limits": ["Synthetic frozen cards never qualify N5", "No semantic/model review", "Media builder stubbed; formal orchestration only",
                   "Push runner_seed absent from raw manifest; complete evaluation/seed-cohort binding checked instead"]})
    print(json.dumps({"status": "PASS_NOT_N5", "checks": len(checks)}))


if __name__ == "__main__":
    main()
