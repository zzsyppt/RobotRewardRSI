"""One explicit semantic review of immutable visual evidence; no model calls.

Supported means every proposition in the original text AND its uncertainty is
supported. Review reasons may explain individual clauses; neither prose nor a
numeric predicate is rewritten or automatically endorsed by this boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

SCHEMA_VERSION = "true_rl_agent.visual_review.v1"
REVIEW_CRITERION = "all_text_and_uncertainty_propositions"
VERDICTS = frozenset(("supported", "contradicted", "insufficient"))


def visual_review_binding(audited: Mapping[str, Any]) -> str:
    """Bind raw output, identity, images and measurements, not local rejection flags.

    Task attachment may add ``source_binding`` violations to a private copy.
    Those rejection flags cannot admit an observation and do not change what
    the reviewer saw. All other record fields remain part of the binding.
    """
    source = {"schema_version": audited["schema_version"], "raw_output": audited["raw_output"],
              "observations": [{key: value for key, value in record.items() if key != "violations"}
                               for record in audited["observations"]]}
    encoded = json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_review(audited: Mapping[str, Any], review: Mapping[str, Any]) -> list[str]:
    fields = {"schema_version", "source_sha256", "criterion", "reviewer", "decisions"}
    if (not isinstance(review, Mapping) or set(review) != fields
            or review["schema_version"] != SCHEMA_VERSION or review["criterion"] != REVIEW_CRITERION):
        raise ValueError("visual review schema or whole-observation criterion is invalid")
    if review["source_sha256"] != visual_review_binding(audited):
        raise ValueError("visual review source binding changed")
    reviewer = review["reviewer"]
    if (not isinstance(reviewer, Mapping) or set(reviewer) != {"id", "kind"}
            or reviewer["kind"] not in ("dev", "model", "human")
            or not isinstance(reviewer["id"], str) or not reviewer["id"].strip()):
        raise ValueError("visual review requires reviewer identity and provenance")
    decisions = review["decisions"]
    count = len(audited["observations"])
    if not isinstance(decisions, list) or len(decisions) != count:
        raise ValueError("visual review requires one decision for every observation")
    by_index = {}
    for decision in decisions:
        if (not isinstance(decision, Mapping)
                or set(decision) != {"observation_index", "verdict", "reason"}
                or type(decision["observation_index"]) is not int
                or not 0 <= decision["observation_index"] < count
                or decision["observation_index"] in by_index
                or not isinstance(decision["verdict"], str) or decision["verdict"] not in VERDICTS
                or not isinstance(decision["reason"], str) or not decision["reason"].strip()):
            raise ValueError("visual review has an invalid, duplicate or unexplained decision")
        by_index[decision["observation_index"]] = decision["verdict"]
    return [by_index[index] for index in range(count)]


def attach_visual_review(audited: Mapping[str, Any], review: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy with a complete review; preserve the original mechanical audit."""
    if "semantic_review" in audited:
        raise ValueError("visual review already attached; use the original audit for another review")
    _validate_review(audited, review)
    result = deepcopy(dict(audited))
    result["semantic_review"] = deepcopy(dict(review))
    return result


def visual_review_verdicts(audited: Mapping[str, Any]) -> list[str]:
    """Recheck the binding at projection, including edits made after attachment."""
    if "semantic_review" not in audited:
        return ["not_evaluated"] * len(audited["observations"])
    return _validate_review(audited, audited["semantic_review"])
