"""Bound internal Push facts and select prose against the actual author view."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, is_sha256_digest
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes, validate_public_report, select_whole_claims
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from .trajectory_card import MAX_PUBLIC_BYTES, TEMPLATE_VERSION

FULL_SCHEMA = "true_rl_agent.push_recovery.behavior.full.v5"
PUBLIC_SCHEMA = "true_rl_agent.push_recovery.behavior.public.v5"
VERSION = "push.trajectory_delivery.v3"
IDENTITY_FIELDS = ("job_id", "task_id", "subject_kind", "subject_id", "subject_manifest_hash",
                   "program_source_checksum", "training_binding_hash",
                   "checkpoint_hash", "evaluation_plan_hash")
DOMAIN_FIELDS = {*IDENTITY_FIELDS, "evaluation_binding_hash", "source_id", "acceptance_split",
                 "resolved_protocol_hash", "mechanism", "root_signed_xy", "manifest_sha256"}


def phase_binding(identity: Mapping, sources: list[dict]) -> dict:
    return {**{k: identity[k] for k in IDENTITY_FIELDS},
            "evaluation_binding_hash": identity["evaluation_binding"]["evaluation_binding_hash"],
            "manifest_sha256": sorted(s["reference"]["manifest_sha256"] for s in sources)}


def render_verified_report(full: dict) -> dict:
    """Internal delivery step, called only after raw-episode semantic validation."""
    card, budget = full["card"], full["max_report_bytes"]
    domain = {**card["domain"], **phase_binding(full["identity"], full["sources"])}
    claims = card["claims"]
    mandatory = [c["id"] for c in claims if c["mandatory"]]
    if not mandatory or len(set(mandatory)) != len(mandatory):
        raise ValueError("Push delivery requires unique nonempty required claims")
    sections = {}
    for c in claims:
        key = c["key"]
        sections[c["id"]] = ("results" if key == "scope" or key.startswith("outcome:") or key.startswith("stratum:")
            else "counterexample" if key.startswith("witness:") else "parent_diff" if key == "reward_parent"
            else "gaps" if key == "capability" else "process")

    def render(included):
        selected = set(included)
        report = {s: [{"id": c["id"], "text": c["text"]} for c in claims
                       if c["id"] in selected and sections[c["id"]] == s] for s in SENTENCE_SECTIONS}
        if not report["counterexample"]:
            report["counterexample"] = [{"id":"push.delivery.counterexample", "text":"本批未找到已定义恢复状态的合格见证。"}]
        report["visual"] = [{"id": "push.delivery.visual", "text": "本报告仅依据数值轨迹；未提供视觉判读。"}]
        omitted = [c["id"] for c in claims if c["id"] not in selected]
        report["gaps"].append({"id": "push.delivery.omissions", "text":
            f"保留{len(included)}/{len(claims)}条原卡事实；省略{len(omitted)}条可选整句。完整证据及省略ID在私有收据中，作者无原档路径。"})
        report["actionability"] = {"authority": "仅现有任务允许的奖励输入；无新增观测、控制器、协议或训练权限。",
                                    "limits": "根部测量不推断足接触、动作、奖励或因果；父代卡独立归属，不作未验证配对比较。"}
        value = {"schema_version": PUBLIC_SCHEMA, "domain": domain, "card_id": card["card_id"],
                 "template_version": TEMPLATE_VERSION, "report": report,
                 "full_derivation_sha256": canonical_json_hash(full),
                 "report_projection": {"version": VERSION, "claim_count": len(claims),
                     "included_claim_count": len(included), "omitted_claim_count": len(omitted),
                      "required_claim_ids": mandatory, "max_bytes": budget,
                     "required_claim_ids_hash": canonical_json_hash(mandatory),
                     "omitted_claim_ids_hash": canonical_json_hash(omitted)}}
        value["report_projection"]["content_hash"] = canonical_json_hash(value)
        return value

    # Keep every candidate fact internally. Only the final author view spends
    # the configured bytes, after selected/parent/sibling source roles are known.
    public = validate_push_trajectory_report(render([c["id"] for c in claims]), max_bytes=budget)
    author_behavior_payload(public)
    return public


def validate_push_trajectory_report(value: Mapping, *, max_bytes: int = MAX_PUBLIC_BYTES) -> dict:
    if (set(value) != {"schema_version", "domain", "card_id", "template_version", "report",
                       "full_derivation_sha256", "report_projection"}
            or value["schema_version"] != PUBLIC_SCHEMA or value["template_version"] != TEMPLATE_VERSION):
        raise ValueError("Push trajectory public version/fields mismatch")
    domain = value["domain"]
    if (set(domain) != DOMAIN_FIELDS or domain["task_id"] != "unitree_push_recovery"
            or domain["acceptance_split"] != "public_dev" or domain["subject_kind"] not in {"reference", "candidate"}
            or domain["root_signed_xy"] not in {"not_recorded", "recorded_world_xy"}
            or domain["mechanism"] not in {"world_planar_external_force", "world_planar_velocity_increment"}
            or any(not isinstance(v, str) or not v for k, v in domain.items() if k != "manifest_sha256")
            or not isinstance(domain["manifest_sha256"], list) or not domain["manifest_sha256"]
            or len(set(domain["manifest_sha256"])) != len(domain["manifest_sha256"])
            or any(not is_sha256_digest(v) for v in domain["manifest_sha256"])):
        raise ValueError("Push trajectory public domain mismatch")
    result = deepcopy(dict(value))
    projection = result["report_projection"]
    if set(projection) != {"version", "claim_count", "included_claim_count", "omitted_claim_count",
                            "required_claim_ids_hash", "omitted_claim_ids_hash", "content_hash", "required_claim_ids", "max_bytes"}:
        raise ValueError("Push trajectory projection receipt fields mismatch")
    digest = projection.pop("content_hash")
    if digest != canonical_json_hash(result):
        raise ValueError("Push trajectory public content hash mismatch")
    projection["content_hash"] = digest
    count = sum(row["id"].startswith("push-claim:") for s in SENTENCE_SECTIONS for row in result["report"][s])
    if (projection["version"] != VERSION or count < 1 or count != projection["included_claim_count"]
            or any(type(projection[k]) is not int or projection[k] < 0 for k in
                   ("claim_count", "included_claim_count", "omitted_claim_count"))
            or projection["claim_count"] != count + projection["omitted_claim_count"]
            or any(not is_sha256_digest(projection[k]) for k in
                   ("content_hash", "required_claim_ids_hash", "omitted_claim_ids_hash"))
            or not is_sha256_digest(result["full_derivation_sha256"])
            or not isinstance(result["card_id"], str) or not result["card_id"].startswith("push-card:")
            or not is_sha256_digest(result["card_id"][10:])):
        raise ValueError("Push trajectory projection counts/digests mismatch")
    required = projection["required_claim_ids"]
    ids = {row["id"] for section in SENTENCE_SECTIONS for row in result["report"][section]}
    if (not isinstance(required, list) or not required or len(set(required)) != len(required)
            or not set(required) <= ids or canonical_json_hash(required) != projection["required_claim_ids_hash"]
            or type(projection["max_bytes"]) is not int or not 0 < projection["max_bytes"] <= min(max_bytes, MAX_PUBLIC_BYTES)):
        raise ValueError("Push trajectory required facts/budget mismatch")
    # Structural replay of the internal envelope; its hashes are not author prose.
    validate_public_report(result, max_bytes=report_bytes(result))
    assert_public_projection_safe(result)
    return result


def _author_source(source: Mapping) -> dict:
    if "card_id" not in source:
        return {k: source[k] for k in ("status", "reason") if k in source}
    return {"card_id": "card:" + source["card_id"].split(":", 1)[1][:12],
            **{k: source[k] for k in ("subject_kind", "subject_id") if k in source}}


def author_behavior_payload(public: Mapping, *, sources: Mapping | None = None) -> dict:
    """Called after identity validation; compact IDs never replace internal binding."""
    projection = public["report_projection"]
    required = projection["required_claim_ids"]
    report = public["report"]
    rows = [row for section in SENTENCE_SECTIONS for row in report[section]
            if row["id"].startswith("push-claim:")]
    short_ids = {row["id"]: f"p{i+1}" for i,row in enumerate(rows)}
    supplied = sources or {}
    visible_sources = {"selected_feedback": _author_source({"card_id": public["card_id"], **public["domain"]})}
    if supplied.get("generation_id") is not None:
        visible_sources["generation_id"] = supplied["generation_id"]
    for role in ("parent", "current_best"):
        if role in supplied:
            visible_sources[role] = _author_source(supplied[role])
    if supplied.get("siblings"):
        visible_sources["siblings"] = [{**{k:item[k] for k in ("trial_id","lane_id","trial_status") if k in item},
            "source":_author_source(item["source"])} for item in supplied["siblings"]]
    def render(included):
        selected = set(included)
        visible = {}
        for section in SENTENCE_SECTIONS:
            visible[section] = [{"id":short_ids.get(row["id"], row["id"]), "text":row["text"]}
                for row in report[section] if row["id"] in selected or row["id"] not in short_ids]
        omitted = len(rows)-len(included)
        visible["gaps"] = [r for r in visible["gaps"] if r["id"] != "push.delivery.omissions"]
        if omitted:
            visible["gaps"].append({"id":"omissions", "text":f"容量内省略{omitted}条可选整句；必需事实保留。"})
        visible["actionability"] = deepcopy(report["actionability"])
        return {"kind":"numeric_behavior", "sources":visible_sources, "report":visible}
    # Optional rows retain task order, never hash order or fixed slot counts.
    value, _, _ = select_whole_claims(required, [r["id"] for r in rows if r["id"] not in required],
        render, max_bytes=projection["max_bytes"])
    return value


def require_phase_binding(report: Mapping, expected: Mapping) -> None:
    domain = validate_push_trajectory_report(report)["domain"]
    required = {*IDENTITY_FIELDS, "evaluation_binding_hash", "manifest_sha256"}
    if set(expected) != required or any(domain[k] != expected[k] for k in required):
        raise ValueError("Push trajectory phase/source binding mismatch")


def require_trial_binding(report: Mapping, trial: Mapping) -> None:
    domain = validate_push_trajectory_report(report)["domain"]
    phase, candidate = trial["public_dev"], trial["candidate"]
    expected = {"job_id": f"{trial['trial_id']}-eval-public-dev", "subject_kind": "candidate",
                "subject_id": candidate["candidate_id"], "checkpoint_hash": phase.get("checkpoint_hash"),
                "evaluation_plan_hash": phase.get("evaluation_plan_hash"),
                "program_source_checksum": candidate.get("program_source_checksum")}
    if any(domain[k] != v for k, v in expected.items()):
        raise ValueError("Push trajectory sibling/trial/role binding mismatch")
