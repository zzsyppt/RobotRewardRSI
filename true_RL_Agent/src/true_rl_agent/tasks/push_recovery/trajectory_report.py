"""Bounded delivery of the frozen Push root card. Public replay performs no IO."""
from __future__ import annotations

from copy import deepcopy
from typing import Mapping

from true_rl_agent.evaluation.binding import canonical_json_hash, is_sha256_digest
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, report_bytes, validate_public_report
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from .trajectory_card import MAX_PUBLIC_BYTES, TEMPLATE_VERSION

FULL_SCHEMA = "true_rl_agent.push_recovery.behavior.full.v4"
PUBLIC_SCHEMA = "true_rl_agent.push_recovery.behavior.public.v4"
VERSION = "push.trajectory_delivery.v1"
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
    if len(mandatory) != 20 or len(set(mandatory)) != 20:
        raise ValueError("Push delivery requires all 20 frozen mandatory slots")
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
        report["visual"] = [{"id": "push.delivery.visual", "text": "本报告仅依据数值轨迹；未提供视觉判读。"}]
        report["gaps"].append({"id": "push.delivery.scope", "text":
            "本报告已由正常CPU finalizer接入作者公开上下文。冻结原卡中“尚未接入”描述P2b1历史交付状态；原句和ID保持不变。"})
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
                     "required_claim_ids_hash": canonical_json_hash(mandatory),
                     "omitted_claim_ids_hash": canonical_json_hash(omitted)}}
        value["report_projection"]["content_hash"] = canonical_json_hash(value)
        return value

    included = list(mandatory)
    public = render(included)
    if report_bytes(public) > budget:
        raise ValueError(f"mandatory Push delivery claims exceed public budget: {report_bytes(public)} > {budget}")
    for claim in claims:
        if claim["mandatory"]:
            continue
        proposed = render([*included, claim["id"]])
        if report_bytes(proposed) <= budget:
            included.append(claim["id"])
            public = proposed
    return validate_push_trajectory_report(public, max_bytes=budget)


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
                            "required_claim_ids_hash", "omitted_claim_ids_hash", "content_hash"}:
        raise ValueError("Push trajectory projection receipt fields mismatch")
    digest = projection.pop("content_hash")
    if digest != canonical_json_hash(result):
        raise ValueError("Push trajectory public content hash mismatch")
    projection["content_hash"] = digest
    count = sum(row["id"].startswith("push-claim:") for s in SENTENCE_SECTIONS for row in result["report"][s])
    if (projection["version"] != VERSION or count < 20 or count != projection["included_claim_count"]
            or any(type(projection[k]) is not int or projection[k] < 0 for k in
                   ("claim_count", "included_claim_count", "omitted_claim_count"))
            or projection["claim_count"] != count + projection["omitted_claim_count"]
            or any(not is_sha256_digest(projection[k]) for k in
                   ("content_hash", "required_claim_ids_hash", "omitted_claim_ids_hash"))
            or not is_sha256_digest(result["full_derivation_sha256"])
            or not isinstance(result["card_id"], str) or not result["card_id"].startswith("push-card:")
            or not is_sha256_digest(result["card_id"][10:])):
        raise ValueError("Push trajectory projection counts/digests mismatch")
    validate_public_report(result, max_bytes=min(max_bytes, MAX_PUBLIC_BYTES))
    assert_public_projection_safe(result)
    return result


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
