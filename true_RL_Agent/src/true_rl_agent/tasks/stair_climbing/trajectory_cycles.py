"""Offline v3 cycle supplement; no reward, policy, or normal delivery changes."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import report_bytes
from true_rl_agent.evaluation.trajectory_translation.cohort import cohort_binding, distribution
from true_rl_agent.evaluation.trajectory_translation.cycles import (
    BASE_CATALOG_VERSION, CATALOG_VERSION, CYCLE_CONFIG, EXTRACTOR_VERSION, measure_contact_cycles,
)
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from .trajectory_adapter import _load_contract, iter_stair_episodes
from .trajectory_artifacts import peak_memory_bytes
from .trajectory_report import validate_stair_trajectory_report


def summarize_cycles(episodes: list[dict]) -> dict:
    """Episode-equal period/coverage estimates and pooled edge counts stay apart."""
    groups = {}
    records = [r for e in episodes for r in e["records"]]
    for key, entities in sorted({(r["key"], tuple(r["entity_ids"])) for r in records}):
        members = [r for r in records if r["key"] == key and tuple(r["entity_ids"]) == entities]
        if key == "contact_cycles":
            values = [r["value"] for r in members]
            totals = {name: sum(v["counts"][name] for v in values) for name in values[0]["counts"]}
            periods = [float(np.median([v["complete_cycles"][i]["raw_inter_onset_interval_s"]
                                       for i in v["qualified_cycle_indices"]]))
                       for v in values if v["qualified_cycle_indices"]]
            fractions = [v["qualified_span_fraction"] for v in values if v["qualified_span_fraction"] is not None]
            reasons = Counter()
            for v in values:
                reasons.update(v["rejected_window_reasons"])
            data = {"counts": totals, "episodes_with_qualified_windows": len(periods),
                    "episodes_with_known_rows": sum(v["counts"]["known_rows"] > 0 for v in values),
                    "episodes_with_known_adjacent_edges": sum(v["counts"]["known_adjacent_edges"] > 0 for v in values),
                    "episodes_with_candidate_windows": sum(v["counts"]["candidate_windows"] > 0 for v in values),
                    "unknown_rhythm_episodes": len(values) - len(periods),
                    "episode_median_qualified_cycle_period_s": distribution(periods),
                    "episode_equal_qualified_span_fraction": distribution(fractions),
                    "pooled_qualified_span_fraction": totals["covered_unique_edges"] / totals["known_adjacent_edges"]
                    if totals["known_adjacent_edges"] else None,
                    "rejected_window_reasons": dict(reasons)}
        else:
            data = {"candidate_windows": sum(r["value"]["candidate_windows"] for r in members),
                    "qualified_windows": sum(r["value"]["qualified_windows"] for r in members),
                    "episodes_with_qualified_phase": sum(r["value"]["qualified_windows"] > 0 for r in members)}
        groups[key + ":" + ",".join(entities)] = {"entity_ids": list(entities), "episodes": len(members), **data}
    return {"episode_count": len(episodes), "groups": groups,
            "weighting": "episode-equal summaries; pooled unique valid readout edges separately; no window-weighted global period",
            "scope": "whole recorded public-dev cohort; local qualification is not whole-episode periodicity"}


def _cycle_statements(binding: dict, summary: dict) -> list[dict]:
    statements = []
    for key, group in summary["groups"].items():
        if key.startswith("contact_cycles:"):
            c = group["counts"]
            period = group["episode_median_qualified_cycle_period_s"]["median"]
            text = (f"{group['entity_ids'][0]}: {c['qualified_windows']}/{c['candidate_windows']} four-cycle readout windows qualified; "
                    f"{c['qualified_unique_cycles']}/{c['complete_cycles']} unique complete cycles; "
                    f"{group['episodes_with_qualified_windows']}/{group['episodes']} episodes contain a qualified local window. "
                    + (f"Median of per-episode median qualified periods {period:.6g}s. " if period is not None else "Period unknown. ")
                    + "Current >1N marker, no physical contact/gait claim; uncovered spans remain unqualified.")
            statements.append({"id": "cycle:" + canonical_json_hash([binding, CYCLE_CONFIG, key, group]), "text": text})
    return statements


def _validate_supplement(supplement: dict, episodes: list[dict]) -> tuple[list[dict], dict]:
    """Bind adapter-produced operands to their actual manifest, then render them.

    This checks finite report semantics, not numeric extraction again. Records
    must come from the audited adapter/build artifact; IDs are not signatures.
    Fixed unit/frame validation also applies to existing v1 record IDs.
    """
    schema = {"schema_version": "stair.cycle_supplement.v1", "catalog_version": CATALOG_VERSION,
              "base_catalog_version": BASE_CATALOG_VERSION, "measurement_config": CYCLE_CONFIG}
    if any(supplement.get(k) != v for k, v in schema.items()):
        raise ValueError("cycle supplement schema/config mismatch")
    source = supplement["source"]
    archive = _load_contract({"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA,
        "manifest_path": source["manifest_path"], "manifest_sha256": source["manifest_sha256"]})
    identity, context = archive.manifest["identity"], archive.manifest["metadata"]
    expected_source = {"manifest_sha256": source["manifest_sha256"], "trial_id": identity["job_id"],
        **{k: identity[k] for k in ("checkpoint_hash", "task_id", "seed_namespace", "evaluation_binding")},
        "profile": context["profile"], "semantics_hash": context["semantics"]["protocol_hash"],
        "runtime_projection_hash": archive.projection.projection_hash, "acceptance_split": "public_dev"}
    if any(source.get(k) != v for k, v in expected_source.items()):
        raise ValueError("cycle supplement source differs from numeric manifest")
    entities = {"root": ["robot_0.root"],
        "support": [f"robot_0.support.{name}" for name in archive.projection.support_body_names],
        "forbidden": [f"robot_0.body.{name}" for name in archive.projection.forbidden_contact_body_names],
        "joints": list(archive.projection.joint_names)}
    binding = cohort_binding({"source": expected_source, "entities": entities}, CATALOG_VERSION)
    if canonical_json_hash(supplement["binding"]) != canonical_json_hash(binding):
        raise ValueError("cycle supplement source/entity binding mismatch")
    slots = [(run, env) for run, lengths in archive.lengths.items() for env in range(len(lengths))]
    if [(e["run_index"], e["env_index"]) for e in episodes] != slots:
        raise ValueError("cycle supplement episode coverage/order mismatch")
    support = entities["support"]
    metrics = [("contact_cycles", [entity]) for entity in support]
    metrics += [("contact_phase", [a, b]) for i, a in enumerate(support) for b in support[i+1:]]
    metadata = {"record_type": "measurement", "catalog_version": CATALOG_VERSION,
        "base_catalog_version": BASE_CATALOG_VERSION, "extractor_version": EXTRACTOR_VERSION,
        "measurement_config": CYCLE_CONFIG, "unit": "s; phase turns", "frame": "recorder_readout"}
    for episode in episodes:
        run, env = episode["run_index"], episode["env_index"]
        seed = context["seeds"][run-1]
        length = int(archive.lengths[run][env])
        eid = stable_id("episode", source["manifest_sha256"], identity["job_id"], run, env)
        if (episode["episode_id"] != eid or episode["runner_seed"] != seed["runner_seed"]
                or episode["dt_s"] != seed["step_dt"]):
            raise ValueError("cycle supplement episode identity/clock mismatch")
        chunks = [{"file": c["file"], "sha256": c["sha256"], "env_index": env,
                   "rows_half_open": [0, min(c["metadata"]["steps"], length-c["metadata"]["step_start"]+1)]}
                  for c in archive.manifest["chunks"]
                  if c["metadata"]["run_index"] == run and c["metadata"]["step_start"] <= length]
        ref = {"manifest_sha256": source["manifest_sha256"], "run_index": run,
               "channels": ["support_contact_forces_w"], "steps_half_open": [1, length+1], "chunks": chunks}
        records = episode["records"]
        if [(r["key"], r["entity_ids"]) for r in records] != metrics:
            raise ValueError("cycle supplement metric/entity order mismatch")
        foot_ids = {r["entity_ids"][0]: r["record_id"] for r in records[:len(support)]}
        for record in records:
            if any(record.get(k) != v for k, v in metadata.items()):
                raise ValueError("cycle measurement metadata unit/frame/schema mismatch")
            if record["episode_id"] != eid or record["evidence_ref"] != ref:
                raise ValueError("cycle measurement source locator mismatch")
            if record["record_id"] != stable_id("cycle-evidence", eid, CYCLE_CONFIG,
                                                record["key"], record["entity_ids"], record["value"]):
                raise ValueError("cycle measurement operand ID mismatch")
            if record["key"] == "contact_phase":
                a, b = record["entity_ids"]
                if (record["value"]["reference_record_id"] != foot_ids[a]
                        or record["value"]["target_record_id"] != foot_ids[b]):
                    raise ValueError("cycle phase operand entity mismatch")
    runs = [{"run_index": run, "runner_seed": context["seeds"][run-1]["runner_seed"], "episodes": len(lengths)}
            for run, lengths in archive.lengths.items()]
    summary = summarize_cycles(episodes)
    if supplement["runs"] != runs or supplement["summary"] != summary:
        raise ValueError("cycle supplement summary differs from bound measurement operands")
    statements = _cycle_statements(binding, summary)
    if supplement["optional_statements"] != statements:
        raise ValueError("cycle supplement sentence differs from bound measurement operands")
    domain = {"evaluation_job_id": identity["job_id"], "archive_hash": source["manifest_sha256"],
        **{k: identity[k] for k in ("checkpoint_hash", "task_id", "seed_namespace")},
        "projection_hash": archive.projection.projection_hash, "acceptance_split": "public_dev",
        "catalog_version": BASE_CATALOG_VERSION,
        **{k: identity["evaluation_binding"].get(k) for k in ("task_protocol_hash", "evaluation_binding_hash")}}
    if "evaluation_plan_hash" in identity:
        domain["evaluation_plan_hash"] = identity["evaluation_plan_hash"]
    return statements, domain


def preview_with_cycles(base_public: dict, supplement: dict, episodes: list[dict], *,
                        max_bytes: int = 8192) -> tuple[dict, dict]:
    """Keep base verbatim, try whole optional sentences within ONE shared budget.

    This is an offline candidate envelope, not a new production public schema.
    If none fits, return the original public object with no envelope overhead.
    """
    if not 0 < max_bytes <= 8192:
        raise ValueError("cycle preview shares the existing Stair 8192 byte cap")
    base = validate_stair_trajectory_report(base_public, max_bytes=max_bytes)
    statements, domain = _validate_supplement(supplement, episodes)
    if any(base["domain"].get(k) != v for k, v in domain.items()):
        raise ValueError("cycle preview source differs from base public report")
    included, omitted = [], []
    result = base
    for statement in statements:
        candidate = {"schema_version": "stair.cycle_offline_preview.v1", "status": "offline_not_production_delivery",
                     "base_report": base, "optional_cycle_statements": [*included, statement]}
        if report_bytes(candidate) <= max_bytes:
            included.append(statement)
            result = candidate
        else:
            omitted.append({"id": statement["id"], "reason": "optional_whole_sentence_combined_budget"})
    return result, {"status": "offline_preview_not_production_delivery", "base_canonical_content_unchanged": True,
                    "serialization": "canonical_compact_sorted_utf8_no_trailing_newline",
                    "base_content_hash": base["report_projection"]["content_hash"],
                    "base_required_claim_ids_hash": base["report_projection"]["required_claim_ids_hash"],
                    "included_ids": [s["id"] for s in included], "omitted": omitted,
                    "combined_bytes": report_bytes(result), "max_bytes": max_bytes}


def build_cycle_supplement(reference: dict, output: Path, *, source_id: str,
                           base_public: dict | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started, cpu_started = time.perf_counter(), time.process_time()
    episodes, binding, source, runs = [], None, None, {}
    evidence_path = output / "cycles.jsonl"
    with evidence_path.open("x", encoding="utf-8", newline="\n") as stream:
        for episode in iter_stair_episodes(reference, source_id=source_id):
            current = cohort_binding({"source": episode.source, "entities": episode.entities}, CATALOG_VERSION)
            if binding is not None and binding != current:
                raise ValueError("cycle supplement cannot mix source or entity bindings")
            binding = current
            source = {k: v for k, v in episode.source.items() if k != "runner_seed"}
            run = runs.setdefault(episode.run_index, {"run_index": episode.run_index,
                                  "runner_seed": episode.source["runner_seed"], "episodes": 0})
            if run["runner_seed"] != episode.source["runner_seed"]:
                raise ValueError("cycle supplement run seed changed")
            run["episodes"] += 1
            records = measure_contact_cycles(episode)
            row = {"run_index": episode.run_index, "env_index": episode.env_index,
                   "runner_seed": episode.source["runner_seed"],
                   "episode_id": episode.episode_id, "dt_s": episode.dt_s, "records": records}
            episodes.append(row)
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    if not episodes:
        raise ValueError("empty cycle supplement source")
    summary = summarize_cycles(episodes)
    statements = _cycle_statements(binding, summary)
    report = {"schema_version": "stair.cycle_supplement.v1", "catalog_version": CATALOG_VERSION,
              "base_catalog_version": BASE_CATALOG_VERSION, "measurement_config": CYCLE_CONFIG,
              "binding": binding, "source": source, "runs": list(runs.values()),
              "summary": summary, "optional_statements": statements,
              "limitations": ["Local current-marker readout rhythm, not physical contact or a named gait.",
                  "History slots are not used. Sensor update delay is not known.",
                  "No outcome/rule/parent/reward/action change; this is known development evidence, not a holdout.",
                  "Half-open phase bins can reject near-zero jitter despite regular per-foot cycles.",
                  "Nominal limits and terrain point-height qualification remain P2c2/P2c3 obligations."]}
    outputs = {"report.json": report}
    if base_public is not None:
        preview, receipt = preview_with_cycles(base_public, report, episodes)
        outputs.update({"preview.json": preview, "preview_receipt.json": receipt})
    else:
        _validate_supplement(report, episodes)
    for name, value in outputs.items():
        if name == "preview.json":
            encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        else:
            encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        (output / name).write_bytes(encoded.encode("utf-8"))
    package = Path(__file__).resolve().parents[2]
    dependencies = [Path(__file__), *sorted((package / "evaluation/trajectory_translation").glob("*.py")),
                    *(Path(__file__).with_name(name) for name in ("trajectory_adapter.py", "trajectory_artifacts.py",
                                                               "trajectory_report.py", "runtime_projection.py")),
                    package / "runner/numeric_evidence.py"]
    manifest = {"schema_version": "stair.cycle_supplement_manifest.v1", "complete": True,
                "catalog_version": CATALOG_VERSION, "base_catalog_version": BASE_CATALOG_VERSION,
                "source_manifest_ref": reference, "binding": binding, "measurement_config": CYCLE_CONFIG,
                "artifacts": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                              for name in ["cycles.jsonl", *outputs]},
                "code_sha256": {p.relative_to(package).as_posix(): file_sha256(p) for p in dependencies},
                "episode_count": len(episodes), "wall_seconds": time.perf_counter()-started,
                "cpu_seconds": time.process_time()-cpu_started, "peak_working_set_bytes": peak_memory_bytes(),
                "external_model_calls": 0, "gpu_runs": 0, "training_runs": 0,
                "status": "offline_qualification_pending_independent_gate"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
