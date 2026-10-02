"""Pack frozen confirmation facts or old media capacity fixtures; never send or release.

--inputs JSON --output NEW_DIR --protocol FILE. Inputs contain development units
{id, export, run_index, env_index, full:{file,sha256}, duplicate_of?}. Optional
observer_directory plus report_projection:{file,sha256} preserve historical
same-episode answers explicitly separately from the new selected-frame fixture.
This media axis does not change the 24 numeric episodes or 12 numeric focuses.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import html
from pathlib import Path
import re
import shutil
import sys

import numpy as np
import true_rl_agent
import export_confirmation_raw as raw_export
import prepare_confirmation_materials as packing
import prepare_native_observer as native
import run_confirmation_review as runner

SOURCE_ROOT = Path(true_rl_agent.__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from true_rl_agent.evaluation.visual_observer import _public_window, audit_observer_output
from true_rl_agent.tasks.stair_climbing.visual_observation import select_stair_media_steps, attach_stair_visual_observations
from true_rl_agent.tasks.push_recovery.visual_observation import select_push_media_steps, attach_push_visual_observations
from true_rl_agent.tasks.stair_climbing.process_report import project_stair_process_report
from true_rl_agent.tasks.push_recovery.process_report import project_push_process_behavior

read, save, digest = packing.read, packing.save, packing.digest
PURPOSE = "old_development_numeric_packing_offline_only"


def checked(ref):
    path = Path(ref["file"]).resolve()
    native.require(digest(path) == ref["sha256"], "input reference hash differs")
    return path


def material(path, role, name, mime="application/json"):
    return {"id": name, "role": role, "mime_type": mime, **runner.ref(path)}


def numeric_episode(dest, task, episode, alias):
    ref = episode["numeric_reference"]
    path = checked({"file": ref["manifest_path"], "sha256": ref["manifest_sha256"]})
    manifest = raw_export.read_manifest(path)
    meta, run, env = manifest["metadata"], episode["run_index"], episode["env_index"]
    arrays, static, terms, missing, chunks = raw_export.load_run(path, manifest, task, run if task == "stair" else None)
    record, = [r for r in raw_export.source_rows(task, manifest, "media_source") if r["run"] == run and r["env"] == env]
    native.require(record["terminal"] == episode["episode_outcome"], "raw outcome differs from captured episode")
    raw, positions = raw_export.episode_arrays(arrays, env, "episode_length" if task == "stair" else "control_step", record["steps"])
    per_static = {k: a[env].copy() for k, a in static.items()}
    if task == "stair":
        from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
        from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
        from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell
        cell = StairUnifiedEvaluationSuite.from_mapping(meta["evaluation_suite"], training_profile=
            StairUnifiedTrainingProfile.from_mapping(meta["training_profile"])).private_cell_assignment("public_dev")[env]
        native.require(cell.cell_id == record["cell"], "numeric cell differs")
        origin = per_static["terrain_env_origins_w"]
        native.require(np.allclose(origin, cell.expected_origin_m, rtol=0, atol=1e-5), "terrain origin differs")
        dt = next(s["step_dt"] for s in meta["seeds"] if s["run_index"] == run)
        curves, measurements = raw_export.stair_measurements(raw, origin, stair_surface_query_for_ordered_cell(cell.spec),
            meta["semantics"]["episode_outcomes"]["signals"], dt, terms)
        definitions = raw_export.stair_definitions(meta, cell, terms)
    else:
        dt = meta["time"]["step_dt_s"]
        curves, measurements = raw_export.push_measurements(raw, record["terminal"], meta, include_window_peak=True)
        definitions = raw_export.push_definitions(meta, include_window_peak=True)
    row = {"episode_id": alias, "task": task, "run_group": "media_run", "scope": record["scope"],
        "cell": record["cell"], "steps": record["steps"], "dt_s": dt, "original_success": record["success"],
        "original_outcome": record["outcome"], "original_terminal_record": record["terminal"],
        "missing_fields": missing, "direct_measurements": measurements}
    receipt = raw_export.export_episode(dest, row, raw, per_static, curves, definitions, positions, 64)
    folder = dest / "episodes" / alias
    header = read(folder / "episode.json")
    related = packing.related_episode(header, [read(folder / p["file"]) for p in header["parts"]], folder / "original_values.npz")
    save(dest / "related_numeric.json", related)
    save(dest / "source_numeric.json", {"manifest": runner.ref(path), "identity": manifest["identity"],
        "chunks": chunks, "export_check": receipt, "full_valid_steps": record["steps"]})
    return meta, row


def build_unit(unit, output, protocol_material, *, purpose=PURPOSE,
               confirmation_selection=None, episode_alias=None):
    dest = output / unit["id"]
    dest.mkdir(parents=True)
    export = Path(unit["export"]).resolve()
    extraction = read(export / "extraction.json")
    task, run, env = extraction["task"], unit["run_index"], unit["env_index"]
    episode, = [e for e in extraction["episodes"] if (e["run_index"], e["env_index"]) == (run, env)]
    full_path = checked(unit["full"])
    full = read(full_path)
    native.require(read(export / "source_summary.json")["task_evidence"]["behavior_full"]["sha256"] == digest(full_path), "full and media differ")
    snapshot = read(export / episode["snapshot_manifest"])
    available = [s["control_step"] for s in snapshot["samples"]]
    formal = purpose == "confirmation"
    alias = episode_alias if formal else "e1"
    native.require(not formal or (confirmation_selection is not None and alias in {"e1", "e2"}),
                   "confirmation media requires the original selection and explicit alias")
    meta, raw_row = numeric_episode(dest / "raw", task, episode, alias)
    if task == "stair":
        process, = [e for e in full["episodes"] if e["episode_id"] == f"r{run}.e{env}"]
        selection = select_stair_media_steps(process, available_steps=available)
    else:
        source_index, = [i for i, s in enumerate(full["source_bindings"]) if s["actual_run_index"] == run]
        process, = [e for e in full["transcripts"] if e["source"] == source_index and e["env_id"] == env]
        selection = select_push_media_steps(process, runtime_episode_record=episode["episode_outcome"],
            available_steps=available, step_dt_s=meta["time"]["step_dt_s"])
    steps = selection["selected_steps"]
    native.require(len(steps) <= 6 and available[-1] in steps, "media selection lost terminal or exceeds six steps")
    selection_path = dest / "observer_selection.json"
    if formal:
        # Consume the same complete v2 selection as the actual observer. Keeping
        # both aliases preserves its contract; this call addresses only one.
        selection_path.write_bytes(Path(confirmation_selection).read_bytes())
        selected, selected_alias, _ = native.selection_episode(read(selection_path), alias)
        native.require((selected["run_index"], selected["env_index"], selected["selected_steps"])
                       == (run, env, steps), "media unit differs from the original selected episode")
        built = native.build(export, selection_path, episode_alias=selected_alias)
    else:
        save(selection_path, {"schema_version": "p3new.native_observer_selection.v1", "scope": "old_development_not_n5",
            "purpose": "old same-episode media capacity check; not an observation request or confirmation",
            "source": {"extraction_sha256": digest(export / "extraction.json"), "description": "sealed old development replay"},
            "episodes": [{"run_index": run, "env_index": env, "selected_steps": steps}]})
        built = native.build(export, selection_path)
    window, = built["windows"]
    native.require(window.alias == alias, "media numeric and visual aliases differ")
    save(dest / "alignment.json", built["alignment"])
    save(dest / "observer_blind_input.json", _public_window(window))
    image_materials, image_rows = [], []
    for index, image in enumerate(window.images):
        step, = image.displayed_steps
        filename = f"{alias}-step{step:06d}-{image.view_id}.png"
        path = dest / filename
        path.write_bytes(image.data)
        sample, = [s for s in snapshot["samples"] if s["control_step"] == step]
        view, = [v for v in sample["views"] if v["view_id"] == image.view_id]
        native.require(digest(path) == view["sha256"], "PNG source bytes changed")
        image_rows.append({"image_id": f"image-{index:02d}", "alias": alias, "step": step, "view_id": image.view_id,
            "sha256": digest(path), "width": image.width, "height": image.height,
            "camera": {k: v for k, v in view.items() if k not in {"file", "sha256", "view_id"}}})
        image_materials.append(material(path, "raw_image", f"image-{index:02d}", "image/png"))
    save(dest / "media.json", {"schema_version": "p3new.confirmation_media.v1", "alias": alias, "images": image_rows,
        "observer_blind_input": _public_window(window), "state_conventions": snapshot["state_conventions"],
        "camera_origin_type": snapshot["camera_origin_type"], "camera_coordinate_unit": "metre in recorded world frame",
        "coverage": {"displayed_steps": steps, "numeric_first_step": 1, "numeric_last_step": raw_row["steps"],
            "omitted_steps_are_not_visible": True}, "alignment_status": built["alignment"]["status"],
        "alignment_not_checked": built["alignment"]["not_checked"]})
    save(dest / "questions.json", {"axis": "media", "alias": alias, "questions": packing.QUESTIONS,
        "scope": "One media episode: distinguish spatial visibility and cited samples, verify S0/body axes; complete 1..T related numerics are supplied. Do not infer group frequency or unseen motion.",
        "duplicate_capacity_only": bool(unit.get("duplicate_of")), "independent_episode": not bool(unit.get("duplicate_of")),
        "no_success_claim_for_duplicate": bool(unit.get("duplicate_of"))})
    materials = [protocol_material, material(dest / "questions.json", "focus_questions", "questions"),
        material(dest / "raw/related_numeric.json", "raw_numeric", "related-numeric"),
        material(dest / "media.json", "raw_media_metadata", "media-metadata"), *image_materials]
    facts = {"schema_version": "p3new.confirmation_review_materials.v1", "packet_id": unit["id"], "stage": "facts",
        "purpose": purpose, "max_output_tokens": 32768, "materials": materials}
    save(dest / "facts.manifest.json", facts)
    save(dest / "full_episode.json", {"source": runner.ref(full_path), "episode": process,
        "scope": "Unchanged selected episode from original full derivation; report-stage data, absent from facts manifest"})
    return dest, task, episode, full, built, selection, facts


def prepare_confirmation(config, inputs, output, protocol):
    """Prepare facts only from the frozen outcome/hash media selections.

    No observer answer, synthetic report, release or historical replacement is
    created here. Missing success/failure classes remain explicit in the receipt.
    """
    capture_path = checked(config["capture_card"])
    input_refs = [runner.ref(inputs), runner.ref(capture_path), runner.ref(protocol)]
    capture = read(capture_path)
    native.require(capture.get("status") == "FROZEN" and
        capture["review_protocol"]["sha256"] == digest(protocol), "confirmation freeze differs")
    native.require(not output.exists(), "confirmation output must be new")
    selections, expected, missing = {}, set(), {}
    for task in ("stair", "push"):
        path = checked(config["selections"][task])
        value = read(path)
        input_refs.append(runner.ref(path))
        native.require(value["scope"] == "n5_confirmation" and
            value["source"]["capture_card_sha256"] == digest(capture_path), "selection capture differs")
        aliases = [e["alias"] for e in value["episodes"]]
        identities = [(e["run_index"], e["env_index"]) for e in value["episodes"]]
        native.require(len(set(aliases)) == len(aliases) and set(aliases) <= {"e1", "e2"}
            and len(set(identities)) == len(identities), "confirmation selection aliases/episodes differ")
        selections[task] = value
        expected.update((task, e["run_index"], e["env_index"]) for e in value["episodes"])
        missing[task] = sorted({"e1", "e2"} - {e["alias"] for e in value["episodes"]})
    actual = [(read(Path(u["export"]) / "extraction.json")["task"], u["run_index"], u["env_index"]) for u in config["units"]]
    native.require(len(set(actual)) == len(actual) and set(actual) == expected, "media units differ from frozen selection")
    native.require(len({u["id"] for u in config["units"]}) == len(config["units"]), "media packet ids must be unique")
    native.require(not any(u.get("duplicate_of") or u.get("observer_directory") for u in config["units"]), "confirmation cannot reuse historical answers or duplicate capacity units")
    output.mkdir(parents=True)
    save(output / "protocol.json", {"status": "FROZEN_FOR_CONFIRMATION", **runner.packet_protocol(protocol),
        "modality": "media with same-episode full-time related numerics"})
    rows = []
    for unit, key in zip(config["units"], actual, strict=True):
        task = key[0]
        frozen, = [e for e in selections[task]["episodes"] if (e["run_index"], e["env_index"]) == key[1:]]
        export = Path(unit["export"])
        checked(unit["full"])
        native.require(digest(export / "extraction.json") == selections[task]["source"]["extraction_sha256"]
            and unit["full"]["sha256"] == selections[task]["source"]["full_sha256"], "selected source drifted")
        episode, = [e for e in read(export / "extraction.json")["episodes"] if (e["run_index"], e["env_index"]) == key[1:]]
        native.require(episode["episode_outcome"]["success"] == (frozen["alias"] == "e1"), "selection outcome alias differs")
        reference = episode["numeric_reference"]
        manifest_path = checked({"file": reference["manifest_path"], "sha256": reference["manifest_sha256"]})
        raw_export.verify_confirmation_source(task, raw_export.read_manifest(manifest_path), capture["tasks"][task], run_index=key[1])
        input_refs.extend([runner.ref(export / "extraction.json"), runner.ref(manifest_path), unit["full"]])
        dest, task, episode, full, built, selection, facts = build_unit(unit, output,
            material(output / "protocol.json", "review_protocol", "protocol"), purpose="confirmation",
            confirmation_selection=checked(config["selections"][task]), episode_alias=frozen["alias"])
        native.require(selection["selected_steps"] == frozen["selected_steps"], "selected frames drifted")
        common = None
        variants = []
        for judge in ("deepseek", "gpt6-sol"):
            prepared = dest / "prepared_facts" / judge
            card = runner.prepare(dest / "facts.manifest.json", judge, prepared)
            parts = json.loads((prepared / "request.http_body.json").read_bytes())["messages"][1]["content"]
            native.require(common is None or parts == common, "reviewers received different media")
            common = parts
            variants.append({"judge": judge, "request_sha256": card["request_sha256"],
                "request_bytes": card["request_bytes"], "png_count": card["png_count"], "actual_calls": 0})
        rows.append({"unit_id": unit["id"], "task": task, "alias": frozen["alias"], "packet_alias": built["windows"][0].alias,
            "source_run": key[1], "source_env": key[2], "original_success": episode["episode_outcome"]["success"],
            "selected_steps": selection["selected_steps"], "facts_manifest": runner.ref(dest / "facts.manifest.json"),
            "variants": variants})
    for reference in input_refs:
        checked(reference)
    receipt = {"status": "CONFIRMATION_FACTS_PREPARED_NOT_RELEASED", "actual_calls": 0,
        "capture_card": runner.ref(capture_path), "review_protocol": runner.ref(protocol), "inputs": runner.ref(inputs),
        "units": rows, "missing_outcome_aliases": missing, "semantic_review_performed": False,
        "report_stage": "not prepared; both judges must seal all facts first",
        "independent_media_episodes": len(rows), "numeric_axes_unchanged": "24 episodes + 12 focus units",
        "limits": "Preparation is not evidence of actual model reading, capacity qualification or semantic support"}
    save(output / "confirmation_media_receipt.json", receipt)
    return receipt


def historical_material(dest, unit):
    if not unit.get("observer_directory"):
        return None
    original = Path(unit["observer_directory"])
    private = read(original / "private_sidecar.json")
    choice, = private["selection"]["episodes"]
    native.require((choice["run_index"], choice["env_index"]) == (unit["run_index"], unit["env_index"])
        and Path(private["source_export"]).resolve() == Path(unit["export"]).resolve(), "historical observer belongs to another episode")
    export = Path(unit["export"])
    snapshot = read(export / private["snapshot_manifest"])
    old_images = []
    actual = dest / "r8_actual"
    actual.mkdir()
    sidecar, = private["prepare_private_sidecar"]
    for image in sidecar["images"]:
        step, = image["displayed_steps"]
        sample, = [s for s in snapshot["samples"] if s["control_step"] == step]
        view, = [v for v in sample["views"] if v["view_id"] == image["view_id"]]
        path = export / Path(private["snapshot_manifest"]).parent / view["file"]
        native.require(digest(path) == image["sha256"] == view["sha256"], "actual observer PNG differs")
        dest_image = actual / f"e1-step{step:06d}-{image['view_id']}.png"
        shutil.copyfile(path, dest_image)
        old_images.append({"step": step, "view_id": image["view_id"], "file": str(dest_image.relative_to(dest)), "sha256": digest(dest_image)})
    for name in ("raw_answer.txt", "raw_response.json", "mechanical_audit.json", "result.json"):
        shutil.copyfile(original / "response" / name, actual / name)
    shutil.copyfile(checked(unit["report_projection"]), actual / "public_report.json")
    historical = {"status": "REAL_R8_SAME_EPISODE_NOT_RESPONSE_TO_NEW_SELECTED_FRAMES",
        "original_selected_steps": choice["selected_steps"], "original_images": old_images,
        "raw_answer_utf8": (actual / "raw_answer.txt").read_text(encoding="utf-8"),
        "raw_answer_source": runner.ref(original / "response/raw_answer.txt"),
        "mechanical_audit": read(actual / "mechanical_audit.json"), "report_projection": read(actual / "public_report.json")}
    save(dest / "historical_observer.json", historical)
    return historical


def report_capacity(dest, unit, full, built, task, facts):
    window, = built["windows"]
    refs = [{"step": image.displayed_steps[0], "view_id": image.view_id} for image in window.images]
    raw = {"episodes": [{"alias": window.alias, "not_visible": [], "observations": [
        {"body": "root", "evidence_refs": refs, "text": "形" * 200, "uncertainty": "限" * 120,
         "spatial_visibility": "visible", "relation": "temporal_change"} for _ in range(3)]}]}
    audit = audit_observer_output(raw, built["windows"], numeric_check=built["numeric_check"])
    attach, project = ((attach_stair_visual_observations, project_stair_process_report) if task == "stair" else
                       (attach_push_visual_observations, project_push_process_behavior))
    augmented = attach(full, audit, observer_version="synthetic_capacity_only_not_observer_response")
    save(dest / "synthetic_observer.json", {"synthetic_only": True, "not_a_response_or_semantic_score": True,
        "raw_output": raw, "mechanical_audit": audit, "public_report_projection": project(augmented)})
    materials = facts["materials"] + [material(dest / "synthetic_observer.json", "observer_output", "synthetic-observer-capacity")]
    if (dest / "historical_observer.json").exists():
        materials.append(material(dest / "historical_observer.json", "observer_output", "historical-same-episode-observer"))
    report = {**facts, "stage": "report", "materials": materials,
        "facts_manifest_sha256": digest(dest / "facts.manifest.json"), "facts_manifest": runner.ref(dest / "facts.manifest.json"),
        "capacity_only": True, "not_executable": "No genuine facts answers; synthetic observer is not a real response"}
    save(dest / "report.capacity.manifest.json", report)


def gallery(items, old=False):
    figures = []
    for item in items:
        path = item["file"] if old else f"e1-step{item['step']:06d}-{item['view_id']}.png"
        figures.append(f'<figure><img style="width:46vw" src="{path}"><figcaption>step {item["step"]}, {item["view_id"]}</figcaption></figure>')
    return "".join(figures)


def prepare(inputs, output, protocol, *, root_review_only=False):
    config = read(inputs)
    if config.get("purpose") == "confirmation":
        native.require(not root_review_only, "confirmation facts are not a developer review gallery")
        return prepare_confirmation(config, inputs, output, protocol)
    native.require(config["purpose"] == "development" and not output.exists(), "development only; output must be new")
    output.mkdir(parents=True)
    save(output / "protocol.json", {"status": "CAPACITY_ONLY_NOT_FROZEN", **runner.packet_protocol(protocol),
        "modality": "media with same-episode full-time related numerics"})
    rows, seen = [], {}
    for unit in config["units"]:
        native.require(unit["id"] not in seen, "duplicate unit id")
        if unit.get("duplicate_of"):
            prior = seen[unit["duplicate_of"]]
            native.require(all(unit[k] == prior[k] for k in ("export", "run_index", "env_index", "full")), "capacity duplicate changed source")
        dest, task, episode, full, built, selection, facts = build_unit(unit, output, material(output / "protocol.json", "review_protocol", "protocol"))
        historical = historical_material(dest, unit)
        if root_review_only:
            images = read(dest / "media.json")["images"]
            page = ('<!doctype html><meta charset="utf-8"><title>Old development root review</title>'
                '<h1>Developer root review — old development, not independent N5 judging</h1>'
                '<p><a href="raw/related_numeric.json">Complete 1..T related curves</a> | <a href="media.json">Camera and S0 definitions</a> | <a href="full_episode.json">Original episode derivation</a></p>'
                '<h2>New selector frames — no corresponding observer call</h2>' + gallery(images))
            page += ('<h2>Actual r8 observer frames and unchanged answer</h2><pre style="white-space:pre-wrap">'
                + html.escape(historical["raw_answer_utf8"]) + '</pre><p><a href="r8_actual/mechanical_audit.json">Original mechanical audit</a> | <a href="r8_actual/public_report.json">Report projection</a></p>'
                + gallery(historical["original_images"], True) if historical else '<h2>No actual observer response exists for this episode</h2>')
            (dest / "review.html").write_text(page, encoding="utf-8")
            rows.append({"unit_id": unit["id"], "task": task, "independent_episode": not bool(unit.get("duplicate_of")),
                "original_success": episode["episode_outcome"]["success"], "selection": selection,
                "actual_r8_response_present": historical is not None, "stage": "development_root_review"})
            seen[unit["id"]] = unit
            print(json.dumps({"prepared_root_review_unit": unit["id"], "selected_steps": selection["selected_steps"]}), flush=True)
            continue
        report_capacity(dest, unit, full, built, task, facts)
        variants = []
        for stage in ("facts", "report"):
            previous_parts = None
            for judge in ("deepseek", "gpt6-sol"):
                fixture = None
                if stage == "report":
                    fixture = dest / f"{judge}.synthetic_own_facts.json"
                    save(fixture, {"synthetic_only": True, "judge_id": judge, "packet_id": unit["id"], "stage": "facts", "complete": False,
                        "findings": {"not_a_fact_or_score": "CAPACITY_ONLY", "padding": "x " * 32768}})
                manifest = dest / ("facts.manifest.json" if stage == "facts" else "report.capacity.manifest.json")
                body, _, card = runner.build(manifest, judge, capacity_facts_fixture=fixture)
                payload = json.loads(body)
                parts = payload["messages"][1]["content"]
                common = parts if stage == "facts" else parts[:-1]
                native.require(previous_parts is None or common == previous_parts, "reviewer source content/order differs")
                previous_parts = common
                (dest / f"{stage}.{judge}.capacity.http_body.json").write_bytes(body)
                text_bytes = len(payload["messages"][0]["content"].encode()) + sum(len(p["text"].encode()) for p in parts if p["type"] == "text")
                proxy = text_bytes + 1024 + 32768 + 8192
                variants.append({"stage": stage, "judge": judge, "http_body_bytes": len(body), "http_sha256": card["request_sha256"],
                    "png_count": card["png_count"], "body_limit_bytes": runner.MAX_BODY, "text_utf8_bytes": text_bytes,
                    "text_proxy_with_output_margin": proxy, "text_proxy_le_180000": proxy <= 180000,
                    "fresh_sol_plus_bootstrap32768": proxy + 32768, "image_input_tokens": None,
                    "context_status": "UNVERIFIED_VISUAL_TOKENS; byte proxy is not a tokenizer count", "actual_calls": 0})
        rows.append({"unit_id": unit["id"], "task": task, "duplicate_of": unit.get("duplicate_of"),
            "independent_episode": not bool(unit.get("duplicate_of")), "original_success": episode["episode_outcome"]["success"],
            "selection": selection, "variants": variants})
        seen[unit["id"]] = unit
    receipt = {"status": "DEVELOPMENT_ROOT_REVIEW_NOT_INDEPENDENT_JUDGING" if root_review_only else "DEVELOPMENT_MEDIA_CAPACITY_ONLY_NOT_EXECUTABLE", "actual_calls": 0, "semantic_scores": None,
        "units": rows, "independent_media_episodes": sum(r["independent_episode"] for r in rows),
        "formal_success_failure_coverage": "NOT_ESTABLISHED; old r3 Push has no success media",
        "unchanged_numeric_axes": "24 episodes + 12 focuses; this tool creates only separate media units",
        "source_root": str(SOURCE_ROOT), "inputs": runner.ref(inputs),
        "dependencies": {str(Path(m.__file__).resolve()): digest(Path(m.__file__)) for m in (native, raw_export, packing, runner)},
        "no_release_files": True, "limits": "Synthetic report fixture and duplicate do not establish observation, independent evidence or semantic PASS"}
    save(output / "capacity_receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--root-review-only", action="store_true", help="Deliver readable real inputs first; omit synthetic report/capacity bodies")
    args = parser.parse_args()
    result = prepare(args.inputs.resolve(), args.output.resolve(), args.protocol.resolve(), root_review_only=args.root_review_only)
    print(json.dumps({k: result[k] for k in ("status", "actual_calls", "independent_media_episodes")}))
