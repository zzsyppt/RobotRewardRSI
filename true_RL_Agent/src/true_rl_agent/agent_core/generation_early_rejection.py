"""Reject from a bound public-accept phase without inventing sealed evidence."""

from true_rl_agent.agent_core.decision import compare_public_accept_scores
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.public_accept_scores import PublicAcceptScores

SEALED_NOT_RUN = "not_run_due_to_early_reject"


def score_rejection_reason(scores, *, parent, protocol):
    if not parent.has_full_eval_stats(protocol):
        raise ValueError("early rejection requires an evaluated comparison head")
    scores = PublicAcceptScores.from_mapping(scores, protocol=protocol,
        eval_protocol_hash=parent.eval_protocol_hash)
    reason = compare_public_accept_scores(acceptance_protocol=protocol,
        acceptance_validation=scores, current_best=parent)
    if reason is not None and reason.endswith("stats_missing"):
        raise ValueError("early rejection comparison statistics are incomplete")
    return reason


def build_early_rejection(*, manifest_hash, trial_id, phase_receipt_checksum,
                        scores, parent, protocol):
    reason = score_rejection_reason(scores, parent=parent, protocol=protocol)
    if reason is None:
        return None
    return {"schema_version": "true_rl_agent.generation_early_rejection.v1",
        "manifest_hash": manifest_hash, "selected_trial_id": trial_id,
        "public_accept_phase_receipt_checksum": phase_receipt_checksum,
        "parent_current_best_hash": canonical_json_hash(parent.to_public_dict()),
        "acceptance_protocol_hash": protocol.protocol_hash,
        "acceptance_protocol": protocol.to_public_dict(), "scores": scores,
        "reason": reason, "sealed_validation": SEALED_NOT_RUN,
        "acceptance_authority": False}


def validate_early_rejection(evidence, *, manifest_hash, trial, parent, protocol):
    if protocol.protocol_hash != parent.acceptance_protocol_hash:
        raise ValueError("early rejection acceptance protocol differs from its parent")
    expected = build_early_rejection(manifest_hash=manifest_hash, trial_id=trial["trial_id"],
        phase_receipt_checksum=trial["public_accept"]["phase_receipt_checksum"],
        scores=evidence["scores"], parent=parent, protocol=protocol)
    if expected is None or evidence != expected:
        raise ValueError("early rejection differs from its bound score comparison")
    return expected["reason"]
