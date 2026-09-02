"""Prompt construction for Phase 5 LLM-agent proposals."""

from __future__ import annotations

import json
from typing import Any

FORBIDDEN_PROMPT_TOKENS = (
    "iid_hidden",
    "iid-hidden",
    "phase2_iid_hidden",
    "iid_hidden_success_rate",
    "iid_hidden_eval",
    "heldout_mean",
    "accept_reject",
)


def build_phase5_messages(
    *,
    summary: dict[str, Any],
    phase1_config: dict[str, Any],
    phase2_config: dict[str, Any],
    phase5_config: dict[str, Any],
) -> list[dict[str, str]]:
    """Build OpenAI-compatible chat messages for one proposal."""

    system = (
        "You are a constrained robotics RL configuration agent. "
        "You must output only a JSON object. You may not request secrets, hidden "
        "evaluation results, extra budget, metric changes, reward-code changes, "
        "or verifier changes. You must propose exactly one config-level patch."
    )
    prompt_cfg = phase5_config.get("prompt", {})
    contract = phase5_config["patch_contract"]
    public_summary = build_public_prompt_summary(summary)
    summary_text = _clip_json(public_summary, int(prompt_cfg.get("max_summary_chars", 24000)))
    user = {
        "task": prompt_cfg.get("instruction"),
        "phase_contract": {
            "allowed_documents": contract.get("documents", {}),
            "allowed_paths": contract.get("allowed_paths", {}),
            "max_operations": contract.get("max_operations", 16),
            "forbidden": [
                "Do not change evaluation profiles, eval seeds, eval max_steps, training seeds, num_envs, max_iterations, timeout budget, metric code, reward implementation code, failure classifier, or verifier.",
                "Do not use held-out evaluation results as proposal input.",
                "Do not include API keys, credentials, or secret placeholders.",
            ],
        },
        "required_response_schema": _response_schema(),
        "public_summary_json": json.loads(summary_text),
        "current_config_context": {
            "phase1_train_push": phase1_config.get("profiles", {}).get("phase2_train", {}).get("push", {}),
            "phase2_reward_weights": phase2_config.get("reward_weights", {}),
            "phase2_reward_params": phase2_config.get("reward_params", {}),
            "phase2_ppo": phase2_config.get("ppo", {}),
            "phase2_baseline": phase2_config.get("baseline", {}),
        },
        "selection_guidance": [
            "Use several statistics together, not only the dominant failure label.",
            "Prefer a small, interpretable intervention that can pass the verifier.",
            "If multiple failure labels are coupled, state that risk explicitly.",
            "Keep evaluation difficulty fixed and do not request held-out evaluation information.",
        ],
    }
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2, sort_keys=True)},
    ]
    assert_prompt_text_safe(messages_to_text(messages), context="phase5_messages")
    return messages


def messages_to_text(messages: list[dict[str, str]]) -> str:
    parts = []
    for message in messages:
        parts.append(f"## {message['role']}\n\n{message['content']}")
    return "\n\n".join(parts) + "\n"


def build_public_prompt_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return the Phase 5 prompt-safe public-only summary projection.

    The raw Phase 2 summary contains IID-hidden results and exact eval seed
    configuration.  Those are valid for offline audit and final reporting, but
    they must never be proposal input to an LLM agent.  This projection is
    deliberately allowlisted instead of redacting by substring.
    """

    public_eval = summary.get("public_eval", {})
    decision = summary.get("decision", {})
    seed_results = []
    for index, item in enumerate(summary.get("seed_results", []), start=1):
        seed_results.append(
            {
                "run_index": index,
                "public_success_rate": item.get("public_success_rate"),
                "public_failure_breakdown": item.get("public_failure_breakdown"),
                "public_condition_flag_breakdown": item.get("public_condition_flag_breakdown"),
            }
        )
    public_summary = {
        "schema_version": summary.get("schema_version"),
        "status": summary.get("status"),
        "public_eval": {
            "success_rate_stats": public_eval.get("success_rate_stats"),
            "target_success_rate_range": public_eval.get("target_success_rate_range"),
            "target_range_met": public_eval.get("target_range_met"),
            "condition_flag_breakdown": public_eval.get("condition_flag_breakdown"),
        },
        "seed_results": seed_results,
        "compute": {
            "gpu_hours_estimate_total": summary.get("compute", {}).get("gpu_hours_estimate_total"),
            "gpu_hours_estimate_per_seed_mean": summary.get("compute", {}).get("gpu_hours_estimate_per_seed_mean"),
            "accounting_method": summary.get("compute", {}).get("accounting_method"),
        },
        "decision": {
            "phase2_workflow_complete": decision.get("phase2_workflow_complete"),
            "difficulty_calibration_needed": decision.get("difficulty_calibration_needed"),
        },
        "prompt_redaction": {
            "public_only": True,
            "heldout_results_removed": True,
            "eval_seed_config_removed": True,
            "raw_config_evaluation_removed": True,
        },
    }
    if "phase6_feedback" in summary:
        phase6_feedback = summary["phase6_feedback"]
        _assert_prompt_safe(phase6_feedback, context="phase6_feedback")
        public_summary["phase6_feedback"] = phase6_feedback
    return public_summary


def _clip_json(data: dict[str, Any], max_chars: int) -> str:
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True)
    if len(text) <= max_chars:
        return text
    clipped = {
        "schema_version": data.get("schema_version"),
        "status": data.get("status"),
        "run_dir": data.get("run_dir"),
        "public_eval": data.get("public_eval"),
        "compute": data.get("compute"),
        "decision": data.get("decision"),
        "phase6_feedback": data.get("phase6_feedback"),
        "seed_results": [
            {
                "run_index": item.get("run_index"),
                "public_success_rate": item.get("public_success_rate"),
                "public_failure_breakdown": item.get("public_failure_breakdown"),
                "public_condition_flag_breakdown": item.get("public_condition_flag_breakdown"),
            }
            for item in data.get("seed_results", [])
        ],
        "clipped": True,
    }
    return json.dumps(clipped, ensure_ascii=False, indent=2, sort_keys=True)


def _assert_prompt_safe(value: Any, *, context: str) -> None:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True).lower()
    assert_prompt_text_safe(text, context=context)


def assert_prompt_text_safe(text: str, *, context: str) -> None:
    text = text.lower()
    hits = [token for token in FORBIDDEN_PROMPT_TOKENS if token in text]
    if hits:
        raise ValueError(f"Prompt contains forbidden tokens in {context}: {hits}")


def _response_schema() -> dict[str, Any]:
    return {
        "analysis": {
            "hypothesis": "string",
            "evidence": "object with cited public metrics",
            "patch_summary": ["string"],
            "expected_effect": "string",
            "risk": "string",
            "rollback_condition": "string",
        },
        "patch": {
            "operations": [
                {
                    "op": "set",
                    "document": "phase1 or phase2",
                    "path": "one of the allowed paths",
                    "value": "JSON scalar/object/list",
                    "reason": "string",
                }
            ]
        },
    }
