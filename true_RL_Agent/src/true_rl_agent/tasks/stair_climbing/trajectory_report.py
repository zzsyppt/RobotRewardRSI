"""Current Stair behavior delivery and source identity boundaries."""
from copy import deepcopy
import json
import re
from typing import Any, Mapping
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.process_report import SENTENCE_SECTIONS, validate_public_report
from true_rl_agent.evaluation.public_projection import assert_public_projection_safe
from .trajectory_production import PUBLIC_SCHEMA, FULL_SCHEMA, MAX_PUBLIC_BYTES, validate_production_report, replay_production_report
_DOMAIN_KEYS = {"evaluation_job_id", "evaluation_plan_hash", "checkpoint_hash", "task_id",
    "task_protocol_hash", "evaluation_binding_hash", "seed_namespace", "archive_hash",
    "projection_hash", "catalog_version", "acceptance_split"}
validate_stair_trajectory_report = validate_production_report
project_stair_trajectory_report = replay_production_report

def _validate_trajectory_public(value, *, public_schema, derivation_version,
                                lexicon_version, catalog_version, max_public_bytes):
    expected = {"schema_version", "domain", "derivation_version", "lexicon_version", "card_id", "report", "report_projection"}
    if (set(value) != expected or value.get("schema_version") != public_schema
            or value["derivation_version"] != derivation_version or value["lexicon_version"] != lexicon_version):
        raise ValueError("Stair trajectory public version/fields mismatch")
    domain = value["domain"]
    validate_trajectory_domain(domain, catalog_version=catalog_version)
    result = deepcopy(dict(value))
    projection = result["report_projection"]
    if set(projection) != {"max_bytes", "optional_claim_ids", "required_claim_ids", "content_hash"}:
        raise ValueError("Stair trajectory projection receipt mismatch")
    if (type(projection["max_bytes"]) is not int or not 0 < projection["max_bytes"] <= max_public_bytes
            or not isinstance(projection["content_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", projection["content_hash"])
            or not isinstance(result["card_id"], str) or not re.fullmatch(r"card:[0-9a-f]{64}", result["card_id"])):
        raise ValueError("Stair trajectory projection budget/digests invalid")
    digest = projection.pop("content_hash")
    if digest != canonical_json_hash(result):
        raise ValueError("Stair trajectory public content hash mismatch")
    projection["content_hash"] = digest
    ids = [row["id"] for section in SENTENCE_SECTIONS for row in result["report"][section] if row["id"].startswith("claim:")]
    ordered = projection["required_claim_ids"] + projection["optional_claim_ids"]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(ids):
        raise ValueError("Stair trajectory claim selection membership mismatch")
    # This is the unselected storage form; only the final author object has a capacity.
    validate_public_report({"kind":"numeric_behavior", "report":result["report"]}, max_bytes=None)
    assert_public_projection_safe(result)
    if "sealed_validation" in json.dumps(result, ensure_ascii=False).casefold():
        raise ValueError("Stair trajectory public report contains non-public split content")
    return result


def validate_trajectory_domain(domain, *, catalog_version):
    """Validate source identity independently of newly rendered report content."""
    if (set(domain) != _DOMAIN_KEYS or domain["task_id"] != "unitree_stair_climbing"
            or domain["acceptance_split"] != "public_dev" or domain["catalog_version"] != catalog_version
            or any(not isinstance(v, str) or not v for v in domain.values())):
        raise ValueError("Stair trajectory public domain mismatch")


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
