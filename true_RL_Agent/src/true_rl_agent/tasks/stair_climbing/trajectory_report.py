"""Opt-in delivery of verified Stair offline cards; old reports remain replayable.

The public view keeps original claim ids/text. The full artifact owns omitted
claims and raw locators; an opaque id does not give the author file access.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re
from pathlib import Path
from typing import Any, Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes, validate_public_report
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from true_rl_agent.evaluation.trajectory_translation.contracts import CATALOG_VERSION
from true_rl_agent.evaluation.trajectory_translation.cohort import cohort_binding
from true_rl_agent.evaluation.trajectory_translation.report import (
    CARD_VERSION, TEMPLATE_VERSION, SELECTION_VERSION, file_sha256, read_link,
    select_policy_summary, validate_claims,
)
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest
from .trajectory_adapter import iter_stair_episodes

FULL_SCHEMA = "true_rl_agent.stair.behavior.full.v4"
PUBLIC_SCHEMA = "true_rl_agent.stair.behavior.public.v5"
DERIVATION_VERSION = "stair.trajectory_report.v1"
LEXICON_VERSION = "stair.trajectory_terms.v1"
MAX_PUBLIC_BYTES = 8192
_DOMAIN_KEYS = {"evaluation_job_id", "evaluation_plan_hash", "checkpoint_hash", "task_id",
                "task_protocol_hash", "evaluation_binding_hash", "seed_namespace",
                "archive_hash", "projection_hash", "catalog_version", "acceptance_split"}
_ACTIONABILITY = {
    "available": ["existing task-approved reward inputs and components only"],
    "diagnostic_only": ["trajectory windows, readout history and cohort statistics do not extend reward ABI"],
    "authority": "No new inputs, controller state, outcome rules or training authority. Observations are not causes.",
}


def _load_pinned(path: Path, sha: str) -> dict:
    if file_sha256(path) != sha:
        raise ValueError("trajectory artifact hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def build_stair_trajectory_report(card_reference: Mapping[str, Any], *, numeric_reference: dict,
                                  expected_identity: Mapping[str, Any], max_bytes: int = MAX_PUBLIC_BYTES) -> dict:
    inputs = load_stair_trajectory_inputs(card_reference, numeric_reference=numeric_reference, expected_identity=expected_identity)
    full = project_trajectory_card(inputs["card"], inputs["records"], domain=inputs["domain"],
                                  card_hash=inputs["card_hash"], max_bytes=max_bytes)
    full["audit"].update(inputs["source_audit"])
    return full


def load_stair_trajectory_inputs(card_reference: Mapping[str, Any], *, numeric_reference: dict,
                                 expected_identity: Mapping[str, Any]) -> dict:
    """Revalidate pinned P1b bytes and source bindings before publishing a view.

    expected_identity comes from the evaluation job, never from the card itself.
    P1b's historical `trial_id` names an evaluation job, not a generation lane.
    """
    archive = load_numeric_evidence_manifest(numeric_reference)
    required = {"job_id", "task_id", "checkpoint_hash", "evaluation_plan_hash", "evaluation_binding", "seed_namespace", "splits"}
    if (not required <= expected_identity.keys() or expected_identity["splits"] != ["public_dev"]
            or any(archive["identity"].get(k) != v for k, v in expected_identity.items())):
        raise ValueError("trajectory evaluation/source identity mismatch")
    manifest_path = Path(card_reference["manifest_path"])
    manifest = _load_pinned(manifest_path, card_reference["manifest_sha256"])
    if (manifest["schema_version"] != "trajectory.card_manifest.v1" or manifest["complete"] is not True
            or manifest["acceptance_split"] != "public_dev" or manifest["card_version"] != CARD_VERSION
            or manifest["catalog_version"] != CATALOG_VERSION or manifest["template_version"] != TEMPLATE_VERSION
            or manifest["selection_version"] != SELECTION_VERSION):
        raise ValueError("trajectory card version or public scope mismatch")
    for name in ("policy_card.json", "evidence.jsonl"):
        path, info = manifest_path.parent / name, manifest["artifacts"][name]
        if path.stat().st_size != info["bytes"] or file_sha256(path) != info["sha256"]:
            raise ValueError("trajectory card content bytes/hash mismatch")
    card = json.loads((manifest_path.parent / "policy_card.json").read_text(encoding="utf-8"))
    if (card["schema_version"] != CARD_VERSION or card["catalog_version"] != CATALOG_VERSION
            or card["template_version"] != TEMPLATE_VERSION or card["selection_version"] != SELECTION_VERSION
            or card["cohort_binding"] != manifest["primary_binding"]):
        raise ValueError("trajectory card/manifest binding mismatch")
    # This adapter checks every chunk before yielding its first episode.
    first = next(iter_stair_episodes(numeric_reference, source_id="delivery"))
    expected_binding = cohort_binding({"source": first.source, "entities": {k: list(v) for k, v in first.entities.items()}}, CATALOG_VERSION)
    if card["cohort_binding"] != expected_binding:
        raise ValueError("trajectory card source/trial/checkpoint/task/catalog mismatch")
    bindings = [manifest["primary_binding"], *manifest["example_source_bindings"]]
    if any(b["acceptance_split"] != "public_dev" or b["task_id"] != expected_binding["task_id"]
           or b["catalog_version"] != CATALOG_VERSION for b in bindings):
        raise ValueError("trajectory example source is not public Stair catalog evidence")
    records, links, recorded_claims = {}, {}, []
    files = manifest["source_files"]
    # Old raw locators stay internal. Verify whole P1a files before seeking them.
    source_bindings = []
    for file_id, ref in files.items():
        p1a = _load_pinned(Path(ref["manifest_path"]), ref["manifest_sha256"])
        if (p1a["acceptance_split"] != "public_dev" or p1a["catalog_version"] != CATALOG_VERSION
                or p1a["evidence"]["sha256"] != file_id or ref["sha256"] != file_id
                or file_sha256(Path(ref["evidence_path"])) != file_id):
            raise ValueError("trajectory evidence source/version/hash mismatch")
        if p1a["source_manifest_ref"] == numeric_reference:
            source_bindings.append(expected_binding)
        else:
            raw_first = next(iter_stair_episodes(p1a["source_manifest_ref"], source_id=p1a["source_id"]))
            source_bindings.append(cohort_binding({"source": raw_first.source,
                "entities": {k: list(v) for k, v in raw_first.entities.items()}}, CATALOG_VERSION))
    if sorted(source_bindings, key=lambda b: b["manifest_sha256"]) != sorted(bindings, key=lambda b: b["manifest_sha256"]):
        raise ValueError("trajectory audit source bindings mismatch")
    with (manifest_path.parent / "evidence.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record["record_type"] == "claim":
                recorded_claims.append(record)
            elif record["record_type"] == "evidence_link":
                links[record["record_id"]] = record
                records[record["record_id"]] = read_link(record, files)
            elif "record_id" in record:
                records[record["record_id"]] = record
                if record["record_type"] == "episode_summary":
                    records[record["activity"]["record_id"]] = record["activity"]
    if recorded_claims != card["claims"] or len(recorded_claims) != manifest["claim_count"]:
        raise ValueError("trajectory claims differ from evidence ledger")
    validate_claims(card["claims"], records)  # Includes P1b metric/label/entity/unit binding.
    if card["policy_summary_claim_ids"] != select_policy_summary(card["claims"], records):
        raise ValueError("trajectory mandatory summary selection mismatch")
    primary = [r for r in records.values() if r.get("record_type") == "episode_summary"
               and r["source_manifest_sha256"] == expected_binding["manifest_sha256"]]
    all_group = next(r for r in records.values() if r.get("record_type") == "cohort_summary" and r["group"] == "all")
    if (set(all_group["member_ids"]) != {r["record_id"] for r in primary}
            or len(primary) != manifest["primary_episodes"]
            or sum(r["coverage"]["valid_rows"] for r in primary) != manifest["primary_valid_rows"]):
        raise ValueError("trajectory primary denominator/source mismatch")
    identity = archive["identity"]
    binding = identity["evaluation_binding"]
    domain = {"evaluation_job_id": identity["job_id"], "evaluation_plan_hash": identity["evaluation_plan_hash"],
              "checkpoint_hash": identity["checkpoint_hash"], "task_id": identity["task_id"],
              "task_protocol_hash": binding["task_protocol_hash"],
              "evaluation_binding_hash": binding["evaluation_binding_hash"], "seed_namespace": identity["seed_namespace"],
              "archive_hash": numeric_reference["manifest_sha256"], "projection_hash": expected_binding["robot_projection_hash"],
              "catalog_version": CATALOG_VERSION, "acceptance_split": "public_dev"}
    return {"card": card, "records": records, "domain": domain,
            "card_hash": manifest["artifacts"]["policy_card.json"]["sha256"],
            "source_audit": dict(card_reference=dict(card_reference), source_bindings=bindings, source_files=files,
                                 evidence_links=links, card_evidence=manifest["artifacts"]["evidence.jsonl"])}


def project_trajectory_card(card: dict, records: dict, *, domain: dict, card_hash: str,
                            max_bytes: int = MAX_PUBLIC_BYTES) -> dict:
    """Atomic deterministic selection; no truncation, translation or free labels."""
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_PUBLIC_BYTES:
        raise ValueError("trajectory report byte budget must be in 1..8192")
    validate_claims(card["claims"], records)
    claims = {c["claim_id"]: c for c in card["claims"]}
    mandatory = select_policy_summary(card["claims"], records)
    for key in ("clock_history", "coverage", "comparison_scope"):
        mandatory.append(next(c["claim_id"] for c in claims.values() if c["template_id"] == "definition"
                              and c["operands"]["text"]["path"] == ["text", key]))
    mandatory = list(dict.fromkeys(mandatory))
    rows, trace = {}, {}
    for cid, claim in claims.items():
        scope = records[claim["scope_record_id"]]
        label = claim["scope"]
        if scope["record_type"] == "episode_summary":
            label += " | " + scope["task"]["cell_id"] + " | " + scope["task"]["outcome"]["primary_reason"]
            if scope["source_manifest_sha256"] != domain["archive_hash"]:
                label += " | separate-source development example; outside primary denominator"
        template = claim["template_id"]
        section = ("results" if template == "cohort_scope" else "counterexample" if template in {"inverted_before", "successful_return"}
                   else "parent_diff" if template == "definition" and claim["operands"]["text"]["path"] == ["text", "parent"]
                   else "gaps" if template == "definition" else "process")
        rows[cid] = (section, {"id": cid, "text": label + ": " + claim["text"]})
        trace[cid] = {"claim": claim, "scope_record_id": claim["scope_record_id"],
                      "source_manifest_sha256": scope.get("source_manifest_sha256", domain["archive_hash"])}
    # The offline delivery-status sentence describes P1b and is not current feedback.
    eligible = [c for c in claims.values() if not (c["template_id"] == "definition"
                and c["operands"]["text"]["path"] == ["text", "delivery"])]
    priority = {"episode_base_runs": 0, "episode_activity": 1, "episode_low_speed": 2,
                "successful_return": 3, "inverted_before": 3, "cell_comparison": 4}
    optional = sorted((c["claim_id"] for c in eligible if c["claim_id"] not in mandatory),
                      key=lambda cid: (priority.get(claims[cid]["template_id"], 5), cid))

    def render(included):
        omitted = [cid for cid in claims if cid not in included]
        report = {s: [rows[cid][1] for cid in included if rows[cid][0] == s] for s in SENTENCE_SECTIONS}
        report["visual"] = [{"id": "trajectory.visual", "text": "Numeric-only observation; media not inspected. No visual or author-model understanding claim."}]
        if not report["counterexample"]:
            report["counterexample"] = [{"id": "trajectory.counterexample_unavailable", "text": "No selected supported counterexample; absence of an example does not establish a failure mechanism."}]
        report["gaps"].append({"id": "trajectory.omissions", "text":
            f"Author view includes {len(included)}/{len(claims)} claims; {len(omitted)} omitted by whole-claim priority/budget or stale offline delivery status. "
            "Full claims, omitted IDs/reasons and raw locators remain in the internal audit. IDs are traceable, not automatic author retrieval."})
        report["actionability"] = deepcopy(_ACTIONABILITY)
        projection = {"selection_version": DERIVATION_VERSION, "claim_count": len(claims),
                      "included_claim_count": len(included), "omitted_claim_count": len(omitted),
                      "omitted_claim_ids_hash": canonical_json_hash(omitted),
                      "required_claim_ids_hash": canonical_json_hash(mandatory)}
        value = {"schema_version": PUBLIC_SCHEMA, "domain": deepcopy(domain),
                 "derivation_version": DERIVATION_VERSION, "lexicon_version": LEXICON_VERSION,
                 "card_id": "card:" + card_hash, "report": report, "report_projection": projection}
        projection["content_hash"] = canonical_json_hash(value)
        return value

    included = list(mandatory)
    public = render(included)
    required_bytes = report_bytes(public)
    if required_bytes > max_bytes:
        raise ValueError(f"trajectory mandatory facts exceed byte budget: {required_bytes} > {max_bytes}")
    for cid in optional:
        candidate = render([*included, cid])
        if report_bytes(candidate) <= max_bytes:
            included.append(cid)
            public = candidate
    validate_stair_trajectory_report(public, max_bytes=max_bytes)
    omitted = [{"claim_id": cid, "reason": "stale_offline_delivery_status" if cid not in {c['claim_id'] for c in eligible}
                else "optional_whole_claim_budget", "claim": claims[cid]} for cid in claims if cid not in included]
    return {"schema_version": FULL_SCHEMA, "public_report": public,
            "audit": {"schema_version": "stair.trajectory_delivery_audit.v1", "claim_order": list(claims), "required_claim_ids": mandatory,
                      "source_domain": deepcopy(domain), "card_id": "card:" + card_hash,
                      "included_claim_ids": included, "omitted_claims": omitted, "claim_trace": trace,
                      # P1b renders a few operand mappings in insertion order.
                      # Retain those exact input bytes through sorted full JSON.
                      "derivation_records_json": json.dumps({rid: records[rid] for rid in sorted({c["scope_record_id"] for c in claims.values()}
                          | {o["record_id"] for c in claims.values() for o in c["operands"].values()})}, ensure_ascii=False, separators=(",", ":")),
                      "report_bytes": report_bytes(public), "max_report_bytes": max_bytes,
                      "required_report_bytes": required_bytes,
                      "source_identity_note": "P1b cohort_binding.trial_id is archive evaluation job_id, not generation trial_id",
                      "author_understanding": "not_tested; P3 boundary", "external_model_calls": 0}}


def validate_stair_trajectory_report(value: Mapping[str, Any], *, max_bytes: int | None = None) -> dict:
    if value.get("schema_version") == "true_rl_agent.stair.behavior.public.v6":
        from .trajectory_production import validate_production_report, MAX_PUBLIC_BYTES as production_limit
        return validate_production_report(value, max_bytes=production_limit if max_bytes is None else max_bytes)
    return _validate_trajectory_public(value, max_bytes=MAX_PUBLIC_BYTES if max_bytes is None else max_bytes,
        public_schema=PUBLIC_SCHEMA, derivation_version=DERIVATION_VERSION, lexicon_version=LEXICON_VERSION,
        catalog_version=CATALOG_VERSION, max_public_bytes=MAX_PUBLIC_BYTES)


def _validate_trajectory_public(value, *, max_bytes, public_schema, derivation_version,
                                lexicon_version, catalog_version, max_public_bytes):
    expected = {"schema_version", "domain", "derivation_version", "lexicon_version", "card_id", "report", "report_projection"}
    if (set(value) != expected or value.get("schema_version") != public_schema
            or value["derivation_version"] != derivation_version or value["lexicon_version"] != lexicon_version):
        raise ValueError("Stair trajectory public version/fields mismatch")
    domain = value["domain"]
    if (set(domain) != _DOMAIN_KEYS or domain["task_id"] != "unitree_stair_climbing"
            or domain["acceptance_split"] != "public_dev" or domain["catalog_version"] != catalog_version
            or any(not isinstance(v, str) or not v for v in domain.values())):
        raise ValueError("Stair trajectory public domain mismatch")
    result = deepcopy(dict(value))
    projection = result["report_projection"]
    if set(projection) != {"selection_version", "claim_count", "included_claim_count", "omitted_claim_count",
                           "omitted_claim_ids_hash", "required_claim_ids_hash", "content_hash"}:
        raise ValueError("Stair trajectory projection receipt mismatch")
    if (any(type(projection[k]) is not int or projection[k] < 0 for k in
            ("claim_count", "included_claim_count", "omitted_claim_count"))
            or any(not isinstance(projection[k], str) or not re.fullmatch(r"[0-9a-f]{64}", projection[k]) for k in
                   ("content_hash", "omitted_claim_ids_hash", "required_claim_ids_hash"))
            or not isinstance(result["card_id"], str) or not re.fullmatch(r"card:[0-9a-f]{64}", result["card_id"])):
        raise ValueError("Stair trajectory projection receipt counts/digests invalid")
    digest = projection.pop("content_hash")
    if digest != canonical_json_hash(result):
        raise ValueError("Stair trajectory public content hash mismatch")
    projection["content_hash"] = digest
    count = sum(row["id"].startswith("claim:") for section in SENTENCE_SECTIONS for row in result["report"][section])
    if (count != projection["included_claim_count"] or projection["claim_count"] != count + projection["omitted_claim_count"]
            or projection["selection_version"] != derivation_version):
        raise ValueError("Stair trajectory claim counts mismatch")
    validate_public_report(result, max_bytes=min(max_bytes, max_public_bytes))
    assert_public_projection_safe(result)
    if "sealed_validation" in json.dumps(result, ensure_ascii=False).casefold():
        raise ValueError("Stair trajectory public report contains non-public split content")
    return result


def project_stair_trajectory_report(value: Mapping[str, Any]) -> dict:
    if value.get("schema_version") in {"true_rl_agent.stair.behavior.full.v5", "true_rl_agent.stair.behavior.public.v6"}:
        from .trajectory_production import replay_production_report
        return replay_production_report(value)
    if value.get("schema_version") == FULL_SCHEMA:
        public = validate_stair_trajectory_report(value["public_report"])
        audit = value["audit"]
        if public["domain"] != audit["source_domain"] or public["card_id"] != audit["card_id"]:
            raise ValueError("Stair trajectory full source/card binding mismatch")
        if "source_bindings" in audit:
            require_trajectory_source_domain(public["domain"], audit["source_bindings"])
        # Full replay re-derives from claim operands, including the P1b semantic
        # checks. A self-hash alone cannot authenticate an edited sentence.
        claims = [audit["claim_trace"][cid]["claim"] for cid in audit["claim_order"]]
        expected = project_trajectory_card({"claims": claims}, json.loads(audit["derivation_records_json"]),
            domain=public["domain"], card_hash=public["card_id"][5:], max_bytes=audit["max_report_bytes"])
        if public != expected["public_report"] or any(audit.get(k) != v for k, v in expected["audit"].items()):
            raise ValueError("Stair trajectory full audit/projection mismatch")
        if "trajectory_build" in audit:
            from .trajectory_pipeline import validate_build_receipt
            validate_build_receipt(audit["trajectory_build"], public)
            if audit["trajectory_build"]["projection"]["max_report_bytes"] != audit["max_report_bytes"]:
                raise ValueError("Stair trajectory full build/projection budget mismatch")
        return public
    return validate_stair_trajectory_report(value)


def require_trajectory_source_domain(domain: Mapping[str, Any], bindings: list) -> None:
    primary = bindings[0]
    expected_source = {"evaluation_job_id": primary["trial_id"], "checkpoint_hash": primary["checkpoint_hash"],
        "task_id": primary["task_id"], "archive_hash": primary["manifest_sha256"],
        "seed_namespace": primary["seed_namespace"], "projection_hash": primary["robot_projection_hash"],
        "catalog_version": primary["catalog_version"], "acceptance_split": primary["acceptance_split"],
        **{k: primary["evaluation_binding"][k] for k in ("task_protocol_hash", "evaluation_binding_hash")}}
    if any(domain[k] != v for k, v in expected_source.items()):
        raise ValueError("Stair trajectory full audit source binding mismatch")


def require_trajectory_phase_binding(report: Mapping[str, Any], *, job_id: str, checkpoint_hash: str,
                                     evaluation_plan_hash: str, evaluation_binding_hash: str,
                                     archive_hash: str, seed_namespace: str) -> None:
    domain = validate_stair_trajectory_report(report)["domain"]
    expected = {"evaluation_job_id": job_id, "checkpoint_hash": checkpoint_hash,
                "evaluation_plan_hash": evaluation_plan_hash, "evaluation_binding_hash": evaluation_binding_hash,
                "archive_hash": archive_hash, "seed_namespace": seed_namespace}
    if any(domain[k] != v for k, v in expected.items()):
        raise ValueError("Stair trajectory report/phase source binding mismatch")


def require_trajectory_trial_binding(report: Mapping[str, Any], trial: Mapping[str, Any]) -> None:
    """Bridge a generation trial to its production public-dev job and plan."""
    domain = validate_stair_trajectory_report(report)["domain"]
    phase = trial["public_dev"]
    expected = {"evaluation_job_id": f"{trial['trial_id']}-eval-public-dev",
                "checkpoint_hash": phase.get("checkpoint_hash"),
                "evaluation_plan_hash": phase.get("evaluation_plan_hash")}
    if any(domain[k] != v for k, v in expected.items()):
        raise ValueError("Stair trajectory sibling/trial/phase binding mismatch")
