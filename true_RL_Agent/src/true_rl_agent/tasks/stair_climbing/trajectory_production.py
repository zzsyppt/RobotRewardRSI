"""Unified base/cycle/nominal selection for the versioned Stair production view."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes, validate_public_report
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.trajectory_translation.contracts import stable_id
from true_rl_agent.evaluation.trajectory_translation import cycles as cycle_operator, joint_limits as limit_operator
from true_rl_agent.evaluation.trajectory_translation.cycles import measure_contact_cycles
from true_rl_agent.evaluation.trajectory_translation.joint_limits import measure_joint_limits
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256, validate_claims, select_policy_summary
from .trajectory_adapter import _load_contract, iter_stair_episodes
from .trajectory_capabilities import resolve_capabilities
from .trajectory_cycles import summarize_cycles, _cycle_statements
from . import trajectory_joint_limits as limits
from . import trajectory_terrain as terrain

PROFILE = "trajectory_numeric_v2"
CATALOG_VERSION = "stair.trajectory.production_catalog.v1"
PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v6"
FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v5"
SELECTION_VERSION = "stair.trajectory.unified_selection.v1"
LEXICON_VERSION = "stair.trajectory_terms.v2"
MAX_PUBLIC_BYTES = 12288
DEFAULT_PUBLIC_BYTES = 10240  # Smallest predeclared budget covering K1/K2/K3 required facts.
VERSIONS = {"catalog": CATALOG_VERSION, "public": PUBLIC_SCHEMA,
            "selection": SELECTION_VERSION, "lexicon": LEXICON_VERSION}


def _write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False), encoding="utf-8")


def build_measurements(reference: dict, output: Path, *, source_id: str) -> dict:
    """One complete adapter pass for the three accepted supplementary operators."""
    archive = _load_contract(reference)
    capabilities, bounds = resolve_capabilities(archive, reference)
    output.mkdir(parents=True, exist_ok=False)
    cycle_rows, joint_rows, terrain_rows = [], [], []
    with (output / "measurements.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for ep in iter_stair_episodes(reference, source_id=source_id):
            identity = {"episode_id": ep.episode_id, "run_index": ep.run_index, "env_index": ep.env_index,
                        "runner_seed": ep.source["runner_seed"], "dt_s": ep.dt_s}
            cycle = {**identity, "records": measure_contact_cycles(ep)}
            cycle_rows.append(cycle)
            joint = None
            if bounds is not None:
                joint = {**identity, "joint_names": list(ep.entities["joints"]),
                         "records": measure_joint_limits(ep, bounds, q_unit="rad", asset_id=limits.ASSET_SHA)}
                joint_rows.append(joint)
            surface = None
            if capabilities["nominal_terrain"]["status"] == "available":
                cell = archive.cells[ep.env_index]
                records, _ = terrain.measure_terrain(ep, cell.spec.geometry_payload(),
                    geometry_digest=cell.geometry_digest, terrain_type=cell.terrain_type_index, regime=cell.terrain_regime_id)
                surface = {**identity, "records": records}
                terrain_rows.append(surface)
            stream.write(json.dumps({"cycle": cycle, "joint": joint, "terrain": surface},
                                    separators=(",", ":"), allow_nan=False) + "\n")
    summary = {"catalog_version": CATALOG_VERSION, "archive_sha256": reference["manifest_sha256"],
               "capabilities": capabilities, "cycle": summarize_cycles(cycle_rows),
               "joint": limits.summarize(joint_rows) if joint_rows else None,
               "terrain": terrain.summarize(terrain_rows) if terrain_rows else None}
    _write(output / "summary.json", summary)
    manifest = {"schema_version": "stair.production_measurements.v1", "complete": True,
                "catalog_version": CATALOG_VERSION, "source": reference,
                "artifacts": {name: {"sha256": file_sha256(output / name), "bytes": (output / name).stat().st_size}
                              for name in ("measurements.jsonl", "summary.json")}}
    from .trajectory_artifacts import _write_json
    _write_json(output / "manifest.json", manifest)
    return {"manifest_path": str(output / "manifest.json"), "manifest_sha256": file_sha256(output / "manifest.json")}


def load_measurements(reference: dict, *, verify_records: bool = False) -> dict:
    path = Path(reference["manifest_path"])
    if file_sha256(path) != reference["manifest_sha256"]:
        raise ValueError("production measurement manifest hash mismatch")
    manifest = json.loads(path.read_bytes())
    if (manifest["schema_version"] != "stair.production_measurements.v1"
            or manifest["catalog_version"] != CATALOG_VERSION or manifest["complete"] is not True
            or set(manifest["artifacts"]) != {"summary.json", "measurements.jsonl"}):
        raise ValueError("production measurement version mismatch")
    for name, item in manifest["artifacts"].items():
        item_path = path.parent / name
        if item_path.stat().st_size != item["bytes"] or file_sha256(item_path) != item["sha256"]:
            raise ValueError("production measurement artifact bytes/hash mismatch")
    summary = json.loads((path.parent / "summary.json").read_bytes())
    if summary["archive_sha256"] != manifest["source"]["manifest_sha256"] or summary["catalog_version"] != CATALOG_VERSION:
        raise ValueError("production measurement source mismatch")
    if verify_records:
        rows = [json.loads(line) for line in (path.parent / "measurements.jsonl").read_text(encoding="utf-8").splitlines()]
        archive = _load_contract(manifest["source"])
        capabilities, bounds = resolve_capabilities(archive, manifest["source"])
        if capabilities != summary["capabilities"]:
            raise ValueError("production capability/source binding mismatch")
        for ep, row in zip(iter_stair_episodes(manifest["source"], source_id="production_replay"), rows, strict=True):
            _validate_measurement_row(ep, row, bounds, archive.cells[ep.env_index], capabilities)
        for key, aggregate in (("cycle", summarize_cycles), ("joint", limits.summarize), ("terrain", terrain.summarize)):
            items = [row[key] for row in rows if row[key] is not None]
            if summary[key] != (aggregate(items) if items else None):
                raise ValueError("production summary differs from measurement operands")
    return summary


def _validate_measurement_row(ep, row, bounds, cell, capabilities):
    """Retain operator metadata and raw locators without re-running extraction."""
    identity = {"episode_id": ep.episode_id, "run_index": ep.run_index, "env_index": ep.env_index,
                "runner_seed": ep.source["runner_seed"], "dt_s": ep.dt_s}
    support = ep.entities["support"]
    cycle_metrics = [("contact_cycles", [s]) for s in support]
    cycle_metrics += [("contact_phase", [a, b]) for i, a in enumerate(support) for b in support[i+1:]]
    for key in ("cycle", "joint", "terrain"):
        item = row[key]
        enabled = key == "cycle" or (bounds is not None if key == "joint" else capabilities["nominal_terrain"]["status"] == "available")
        if not enabled:
            if item is not None:
                raise ValueError("unqualified production measurement present")
            continue
        if item is None or any(item.get(k) != v for k, v in identity.items()):
            raise ValueError("production measurement episode/source/clock mismatch")
        if key == "joint" and item.get("joint_names") != list(ep.entities["joints"]):
            raise ValueError("production joint summary name/order mismatch")
        expected_entities = ([entities for _, entities in cycle_metrics] if key == "cycle" else
                             [[j] for j in ep.entities["joints"]] if key == "joint" else
                             [[e] for e in (*ep.entities["root"], *support)])
        if [r["entity_ids"] for r in item["records"]] != expected_entities:
            raise ValueError("production measurement entity order/coverage mismatch")
        for column, r in enumerate(item["records"]):
            value, entities = r["value"], tuple(r["entity_ids"])
            if key == "cycle":
                metric = cycle_metrics[column][0]
                if metric == "contact_phase":
                    feet = {r["entity_ids"][0]: r["record_id"] for r in item["records"][:len(support)]}
                    if (value["reference_record_id"] != feet[entities[0]]
                            or value["target_record_id"] != feet[entities[1]]):
                        raise ValueError("production cycle phase operand binding mismatch")
                expected = {"record_type": "measurement", "key": metric, "entity_ids": list(entities),
                    "record_id": stable_id("cycle-evidence", ep.episode_id, cycle_operator.CYCLE_CONFIG, metric, entities, value),
                    "episode_id": ep.episode_id, "catalog_version": cycle_operator.CATALOG_VERSION,
                    "base_catalog_version": cycle_operator.BASE_CATALOG_VERSION, "extractor_version": cycle_operator.EXTRACTOR_VERSION,
                    "measurement_config": cycle_operator.CYCLE_CONFIG, "unit": "s; phase turns", "frame": "recorder_readout",
                    "evidence_ref": ep.reference(("support_contact_forces_w",)), "value": value}
            elif key == "joint":
                expected = ep.record("nominal_joint_limits", "measurement", entities, ("joint_pos",), value,
                    unit="rad; fraction; sampled s", frame="nominal_usd_joint_coordinate", config=limit_operator.CONFIG)
                expected.update(catalog_version=limit_operator.CATALOG_VERSION, base_catalog_version=limit_operator.BASE_CATALOG_VERSION,
                                extractor_version="trajectory.nominal_joint_limits.v1", asset_id=limits.ASSET_SHA)
                expected.pop("record_id")
                expected["record_id"] = stable_id("nominal-limit-evidence", expected)
                bound = bounds[entities[0]]
                if value["nominal_bounds_rad"] != [bound["lower"], bound["upper"]]:
                    raise ValueError("production nominal bound mismatch")
                for witness in value["witnesses"].values():
                    index = witness["step"] - 1
                    if witness["joint_column"] != column or witness["source"] != ep.reference(("joint_pos",), index, index+1):
                        raise ValueError("production joint witness locator mismatch")
            else:
                channel = "base_pos_w" if column == 0 else "support_pos_w"
                expected = ep.record("M10.nominal_reference_height", "measurement", entities, (channel,), value,
                    unit="m", frame=terrain.FRAME, config=terrain.CONFIG)
                expected["catalog_version"] = terrain.CATALOG_VERSION
                expected.pop("record_id")
                expected["record_id"] = stable_id("nominal-terrain", expected)
                geometry = {"cell_id": ep.task["cell_id"], "terrain_type": cell.terrain_type_index,
                    "geometry_digest": cell.geometry_digest, "geometry": cell.spec.geometry_payload(), "regime": cell.terrain_regime_id,
                    "origin_world_m": list(ep.task["origin_world"]),
                    "static_refs": {**ep.task["static_refs"], "terrain_types": {**ep.task["static_refs"]["terrain_env_origins_w"], "array": "terrain_types"}}}
                if value["geometry_binding"] != geometry:
                    raise ValueError("production terrain geometry/source binding mismatch")
                for witness in [*value["witnesses"].values(), *value["first_by_status"].values()]:
                    if witness is None:
                        continue
                    index = witness["step"] - 1
                    if witness["column"] != (None if column == 0 else column-1) or witness["source_ref"] != ep.reference((channel,), index, index+1):
                        raise ValueError("production terrain witness locator mismatch")
            if r != expected:
                raise ValueError("production measurement unit/frame/source/record identity mismatch")


# Each old mandatory definition has an explicit replacement, not a budget deletion.
DEFINITIONS = {
    "contact_definition": "Current marker: recorded world net normal-contact-force norm >1N; no physical-contact continuity, contact-object or load-share inference. An off marker does not prove no contact.",
    "activity_definition": "Low progress: maximum additional progress in a complete >=0.5s window <0.02m; shorter initial windows unknown. Foot XY speed threshold 0.05m/s; same-readout conjunction only, not gait, refusal or whole-body rest.",
    "coverage": "Episode-equal distributions; pooled samples/time separately, each quantity has its own valid denominator. N*dt is sample-cell time, (N-1)*dt observed span. Left/right-censored runs excluded from complete-duration distributions; missing/unknown are not negative.",
    "comparison_scope": "Whole cohort then outcome/stratum/cell facets. Endpoint lengths and success composition affect whole-episode statistics. Same cell does not pair actual commands/randomization; no significance, improvement or causal claim.",
    "parent": "Parent numeric comparison unavailable: no explicitly bound parent archive. Separate-source examples are not parents and never enter the primary denominator.",
    "missing": "Actions, torque/saturation, reward terms/gates/air-time, actual mass/full-body momentum, tangential force/contact objects and other nonfoot forces missing. Recorded nonfoot scope is bound base only; exact reward response/physical balance unavailable.",
    "deferred": "Enabled: base and local readout cycles; nominal limits/terrain only when qualified below. Terrain uses each point's XY, world-z and m; not sole clearance or contact distance. Effective runtime limits and runtime collision surface unknown. Historical link-position/COM-velocity equivalence, default-pose/high-position thresholds unverified; no raised-leg/dragging labels.",
    "clock_history": "Pre-reset control-step readout; current sensor delay and history per-slot time/valid unknown. History ANY is separate from current, not physics substeps or true impulse.",
}


def _extension_claims(summary):
    binding = {"archive_sha256": summary["archive_sha256"], "catalog_version": CATALOG_VERSION,
               "capabilities": summary["capabilities"]}
    rows, trace, required = {}, {}, []

    def add(key, group_key, row, operands, reason):
        cid = stable_id("claim", SELECTION_VERSION, binding, key, group_key, row["text"])
        rows[cid] = ("process", {"id": cid, "text": row["text"]})
        trace[cid] = {"measurement": key, "group": group_key, "operands": operands,
                      "source_manifest_sha256": summary["archive_sha256"], "selection_reason": reason}
        return cid

    groups = summary["cycle"]["groups"]
    ordered = sorted((k for k in groups if k.startswith("contact_cycles:")),
                     key=lambda k: (-groups[k]["counts"]["qualified_unique_cycles"], groups[k]["entity_ids"][0]))
    statements = _cycle_statements(binding, summary["cycle"])
    by_entity = {g["entity_ids"][0]: s for g, s in zip(
        (g for k, g in groups.items() if k.startswith("contact_cycles:")), statements, strict=True)}
    for index, key in enumerate(ordered):
        group = groups[key]
        row = dict(by_entity[group["entity_ids"][0]])
        c = group["counts"]
        row["text"] += (f" Unique qualified readout edges {c['covered_unique_edges']}/{c['known_adjacent_edges']}; "
                        f"unknown rhythm {group['unknown_rhythm_episodes']}/{group['episodes']} episodes.")
        cid = add("cycle", key, row, group, "most_qualified_unique_cycles_then_entity")
        if index == 0:
            required.append(cid)
    for key, renderer in (("joint", limits.statements), ("terrain", terrain.statements)):
        data = summary[key]
        if data is None:
            capability = summary["capabilities"]["nominal_joint_limits" if key == "joint" else "nominal_terrain"]
            cid = add(key, "unavailable", {"text": f"Nominal {key} unavailable: {capability['reason']}. "
                      "This is missing qualification, not zero observations; effective limits/runtime surface remain unknown."},
                      capability, "explicit_missing_qualification")
            required.append(cid)
            continue
        rendered = dict(zip(data["groups"], renderer(binding, data), strict=True))
        if key == "joint":
            def rank(name):
                value = data["groups"][name]["episode_equal"]["minimum_signed_margin_rad"]["minimum"]
                return (value is None, value if value is not None else 0, name)
            selected = [min(data["groups"], key=rank)]
            reason = "lowest_minimum_signed_margin_then_joint_name"
        else:
            selected = [min((k for k, g in data["groups"].items() if g["regime"] == regime),
                            key=lambda k: (data["groups"][k]["pooled_known_fraction"], data["groups"][k]["entity_id"]))
                        for regime in sorted({g["regime"] for g in data["groups"].values()})]
            reason = "each_regime_lowest_known_fraction_then_entity"
        for group_key, group in data["groups"].items():
            row = dict(rendered[group_key])
            if key == "terrain":
                c, median = group["counts"], group["episode_equal"]["mean"]["median"]
                row["text"] = (f"{group['regime']} / {group['entity_id']}: nominal terrain known {c['known']}/{c['total']} "
                    f"reference-point samples; {c['unknown']} unknown. "
                    + (f"Episode-equal median mean (point z - nominal height) {median:.6g}m." if median is not None else "Vertical difference unknown."))
            row["text"] += f" Valid episodes {group['known_episodes']}/{group['episodes']}."
            cid = add(key, group_key, row, group, reason)
            if group_key in selected:
                required.append(cid)
    return rows, trace, required


def base_operands(inputs: dict) -> dict:
    """Internal validated operands have no legacy author-view byte constraint."""
    claims, records = inputs["card"]["claims"], inputs["records"]
    needed = {c["scope_record_id"] for c in claims} | {o["record_id"] for c in claims for o in c["operands"].values()}
    return {"schema_version": "stair.production_base_operands.v1", "claims": claims,
            "records_json": json.dumps({rid: records[rid] for rid in sorted(needed)}, ensure_ascii=False, separators=(",", ":")),
            "domain": inputs["domain"], "card_id": "card:" + inputs["card_hash"], "source_audit": inputs["source_audit"]}


def project_production_report(base_full: dict, summary: dict, *, max_bytes: int) -> dict:
    """Select all required semantics together; optional claims never crowd them out."""
    from .trajectory_report import _ACTIONABILITY
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError("production trajectory report budget must be in 1..12288")
    if base_full["schema_version"] == "true_rl_agent.stair.behavior.full.v4":
        # Explicit historical replay input; production finalization calls the
        # verified operand loader directly, never builds an 8KB legacy view.
        from .trajectory_report import project_stair_trajectory_report
        project_stair_trajectory_report(base_full)
        old = base_full["audit"]
        base_full = {"schema_version": "stair.production_base_operands.v1",
            "claims": [old["claim_trace"][cid]["claim"] for cid in old["claim_order"]],
            "records_json": old["derivation_records_json"], "domain": base_full["public_report"]["domain"],
            "card_id": base_full["public_report"]["card_id"],
            "source_audit": {k: old[k] for k in ("source_bindings", "source_files", "evidence_links", "card_reference") if k in old}}
    if base_full["schema_version"] != "stair.production_base_operands.v1":
        raise ValueError("production base operands version mismatch")
    base = base_full
    from .trajectory_report import require_trajectory_source_domain, _load_pinned
    source_audit = base["source_audit"]
    require_trajectory_source_domain(base["domain"], source_audit["source_bindings"])
    card_ref = source_audit["card_reference"]
    card_manifest = _load_pinned(Path(card_ref["manifest_path"]), card_ref["manifest_sha256"])
    if (base["card_id"] != "card:" + card_manifest["artifacts"]["policy_card.json"]["sha256"]
            or card_manifest["primary_binding"] != source_audit["source_bindings"][0]):
        raise ValueError("production base card/source binding mismatch")
    records = json.loads(base["records_json"])
    claims = base["claims"]
    validate_claims(claims, records)
    mandatory = select_policy_summary(claims, records)
    for key in ("clock_history", "coverage", "comparison_scope"):
        mandatory.append(next(c["claim_id"] for c in claims if c["template_id"] == "definition"
                              and c["operands"]["text"]["path"] == ["text", key]))
    claim_trace = {c["claim_id"]: {"claim": c, "scope_record_id": c["scope_record_id"],
        "source_manifest_sha256": records[c["scope_record_id"]].get("source_manifest_sha256", base["domain"]["archive_hash"])} for c in claims}
    audit = {"claim_order": [c["claim_id"] for c in claims], "required_claim_ids": mandatory, "claim_trace": claim_trace}
    if base["domain"]["archive_hash"] != summary["archive_sha256"]:
        raise ValueError("production base/measurement source mismatch")
    rows, trace, mapping, required = {}, {}, {}, []
    for old_id in audit["claim_order"]:
        claim = audit["claim_trace"][old_id]["claim"]
        if claim["template_id"] == "definition":
            key = claim["operands"]["text"]["path"][-1]
            if old_id not in audit["required_claim_ids"]:
                continue
            text = DEFINITIONS.get(key, claim["text"])
            if key == "clock_history" and "history未采集" in claim["text"]:
                text = "History not captured: valid denominator 0, all unknown; no inferred history events. Current update delay unknown; manager endpoint is independent of force thresholds."
            cid = stable_id("claim", SELECTION_VERSION, old_id, text)
            section = "parent_diff" if key == "parent" else "gaps"
        else:
            cid, text = old_id, claim["scope"] + ": " + claim["text"]
            scope = records[claim["scope_record_id"]]
            if scope["record_type"] == "episode_summary":
                text = claim["scope"] + " | " + scope["task"]["cell_id"] + " | " + scope["task"]["outcome"]["primary_reason"] + ": " + claim["text"]
                if scope["source_manifest_sha256"] != base["domain"]["archive_hash"]:
                    text = "Separate-source example, outside primary denominator: " + text
            section = ("results" if claim["template_id"] == "cohort_scope" else
                       "counterexample" if claim["template_id"] in {"inverted_before", "successful_return"} else "process")
        rows[cid] = (section, {"id": cid, "text": text})
        trace[cid] = {"base_claim_id": old_id, "base_trace": audit["claim_trace"][old_id]}
        if old_id in audit["required_claim_ids"]:
            mapping[old_id] = cid
    if len(mapping) != 18:
        raise ValueError("production profile requires all 18 base semantic slots")
    required = [mapping[cid] for cid in audit["required_claim_ids"]]
    extra_rows, extra_trace, extra_required = _extension_claims(summary)
    rows.update(extra_rows); trace.update(extra_trace); required.extend(extra_required)
    optional = sorted(set(rows) - set(required))
    domain = {**base["domain"], "catalog_version": CATALOG_VERSION}
    card_id = "card:" + canonical_json_hash([base["card_id"], summary, VERSIONS])

    def render(included):
        report = {s: [rows[cid][1] for cid in included if rows[cid][0] == s] for s in SENTENCE_SECTIONS}
        report["visual"] = [{"id": "trajectory.visual", "text": "Numeric only; media uninspected. No causal, author-understanding or training-benefit claim."}]
        if not report["counterexample"]:
            report["counterexample"] = [{"id": "trajectory.counterexample_unavailable", "text": "Supported counterexample unavailable; this does not establish a failure mechanism."}]
        report["gaps"].append({"id": "trajectory.omissions", "text":
            f"Selected {len(included)}/{len(rows)} candidate claims; full operands, original short events and omitted IDs/reasons remain in the internal audit. IDs enable tracing, not automatic author access."})
        report["actionability"] = deepcopy(_ACTIONABILITY)
        receipt = {"selection_version": SELECTION_VERSION, "claim_count": len(rows),
                   "included_claim_count": len(included), "omitted_claim_count": len(rows)-len(included),
                   "omitted_claim_ids_hash": canonical_json_hash([cid for cid in rows if cid not in included]),
                   "required_claim_ids_hash": canonical_json_hash(required)}
        value = {"schema_version": PUBLIC_SCHEMA, "domain": domain, "derivation_version": SELECTION_VERSION,
                 "lexicon_version": LEXICON_VERSION, "card_id": card_id, "report": report, "report_projection": receipt}
        receipt["content_hash"] = canonical_json_hash(value)
        return value

    included = list(required)
    public = render(included)
    required_bytes = report_bytes(public)
    if required_bytes > max_bytes:
        raise ValueError(f"production mandatory facts exceed byte budget: {required_bytes} > {max_bytes}")
    for cid in optional:
        candidate = render([*included, cid])
        if report_bytes(candidate) <= max_bytes:
            included.append(cid); public = candidate
    validate_production_report(public, max_bytes=max_bytes)
    omitted = [{"claim_id": cid, "reason": "optional_whole_claim_budget", "trace": trace[cid]}
               for cid in rows if cid not in included]
    superseded = [{"claim_id": cid, "reason": "superseded_definition_or_offline_status"}
                  for cid in audit["claim_order"] if cid not in mapping and cid not in rows]
    return {"schema_version": FULL_SCHEMA, "public_report": public,
            "audit": {"schema_version": "stair.production_delivery_audit.v1", "base_operands": base_full,
                      "measurement_summary": summary, "base_semantic_mapping": mapping,
                      "required_claim_ids": required, "included_claim_ids": included,
                      "omitted_claims": omitted, "superseded_base_claims": superseded,
                      "claim_trace": trace, "max_report_bytes": max_bytes,
                      "report_bytes": report_bytes(public), "required_report_bytes": required_bytes,
                      "tokens": "unknown", "runtime_validation": "not_performed_by_CPU_finalizer"}}


def validate_production_report(value, *, max_bytes=MAX_PUBLIC_BYTES):
    from .trajectory_report import _validate_trajectory_public
    return _validate_trajectory_public(value, max_bytes=max_bytes, public_schema=PUBLIC_SCHEMA,
        derivation_version=SELECTION_VERSION, lexicon_version=LEXICON_VERSION,
        catalog_version=CATALOG_VERSION, max_public_bytes=MAX_PUBLIC_BYTES)


def replay_production_report(value):
    if value["schema_version"] == PUBLIC_SCHEMA:
        return validate_production_report(value)
    audit = value["audit"]
    if "measurements_reference" in audit:
        if load_measurements(audit["measurements_reference"], verify_records=True) != audit["measurement_summary"]:
            raise ValueError("production full measurement binding mismatch")
    expected = project_production_report(audit["base_operands"], audit["measurement_summary"], max_bytes=audit["max_report_bytes"])
    if value["public_report"] != expected["public_report"] or any(audit.get(k) != v for k, v in expected["audit"].items()):
        raise ValueError("production full report differs from operands/selection")
    if "trajectory_build" in audit:
        from .trajectory_pipeline import validate_build_receipt
        validate_build_receipt(audit["trajectory_build"], value["public_report"])
    return validate_production_report(value["public_report"])
