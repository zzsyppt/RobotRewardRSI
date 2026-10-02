"""R.0-b four-view author probe; reuses P3 context restoration and production loop.

Facts live in external, hash-bound JSON files. This script never invents facts,
trains candidates, or grades semantics. preflight/export do not call an API.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
VARIANTS = ("A", "T0", "Tc-num", "Tc-vis")
CASE_IDS = {"stair-01", "stair-03", "push-02", "push-04"}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_p3(path: Path):
    sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("p3_author_context", path / "author_experiment.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_ref(ref: dict[str, Any], base: Path) -> Path:
    path = Path(ref["path"])
    if not path.is_absolute():
        path = base / path
    if sha(path) != ref["sha256"].removeprefix("sha256:"):
        raise ValueError(f"input changed: {path.name}")
    return path


def source_card(path: Path, exp):
    data = read(path)
    old_path = resolve_ref(data["p3_cases"], path.parent)
    old = {c["case_id"]: c for c in read(old_path)["cases"]}
    if {c["case_id"] for c in data["cases"]} != CASE_IDS or len(data["cases"]) != 4:
        raise ValueError("R.0-b requires the four preregistered cases")
    for case in data["cases"]:
        if set(case["variants"]) != set(VARIANTS):
            raise ValueError("each case requires all four evidence views")
        for variant, view in case["variants"].items():
            expected_base = "T" if variant == "T0" else "A"
            if view["base"] != expected_base:
                raise ValueError("variant source base differs from the protocol")
            if ("behavior" in view) != (variant in {"Tc-num", "Tc-vis"}):
                raise ValueError("only curated views may replace the behavior subtree")
    return data, old, old_path.parent


def feedback_view(original, view, *, data_base, old_base, exp):
    feedback = copy.deepcopy(exp.bound_read(original[f"feedback_{view['base']}"], old_base))
    if "behavior" in view:
        behavior = read(resolve_ref(view["behavior"], data_base))
        if not isinstance(behavior, dict):
            raise ValueError("curated behavior must be an explicit facts object")
        # Experiment-only evidence, not a forged production behavior schema.
        owner = feedback if original["task"] == "stair" else feedback["public_dev"]
        owner["behavior"] = behavior
    return feedback


def build_context(case, variant, old, *, data_base, old_base, root, output, exp):
    from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, primary_ledger_epoch_hash
    from true_rl_agent.memory.context_builder import PublicContextBuilder
    original = old[case["case_id"]]
    spec = exp.bound_read(original["author_context"], old_base)
    comp, best, evidence, source, genesis_hash, initial_hash = exp.reference_context(spec, root=root, base=old_base, output=output)
    if best.checkpoint_hash != original["checkpoint_hash"] or spec["diagnostic_as_of"] != original["as_of_cutoff"]:
        raise ValueError("case parent or as-of changed")
    if exp.phase_source_bindings(spec, base=old_base) != original["source_bindings"]:
        raise ValueError("source archive binding differs from the original P3 case")
    feedback = feedback_view(original, case["variants"][variant], data_base=data_base, old_base=old_base, exp=exp)
    adapter = exp.FrozenFeedbackAdapter(comp.task_plugin.prompt_adapter(), feedback, original["common_focus"])
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter,
        task_prompt_adapter=adapter, phased_evaluation_adapter=comp.task_plugin.phased_evaluation_adapter(),
        acceptance_protocol=comp.task_plugin.task_spec.acceptance_protocol, initial_current_best=best,
        session_genesis_hash=genesis_hash, candidate_capability_profile=comp.task_plugin.task_spec.candidate_capability_profile,
        initial_reference_program_source=source, initial_anchor_public_evidence=evidence,
        expected_initial_policy_state_hash=initial_hash, ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(genesis_hash) if genesis_hash else None)
    return comp, builder, feedback


def first_prompt(comp, builder):
    return builder.build_prompt([], authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
        candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules()).provider_payload


def prompt_hash(value):
    from true_rl_agent.prompt_serialization import canonical_prompt_json
    return hashlib.sha256(canonical_prompt_json(value).encode()).hexdigest()


def schedule():
    rng = random.Random(2026092902)
    cases = sorted(CASE_IDS)
    rng.shuffle(cases)
    rows = []
    for case_id in cases:
        order = list(VARIANTS)
        rng.shuffle(order)
        for repetition in (1, 2):
            for variant in order if repetition == 1 else list(reversed(order)):
                index = len(rows) + 1
                rows.append({"case_id": case_id, "variant": variant, "repetition": repetition,
                    "generation_id": f"p3r-ceiling-20260929-generation-{(index + 1) // 2:03d}",
                    "lane_id": "lane-a" if index % 2 else "lane-b"})
    return rows


def usage_report(output, stage, exp):
    """Reuse old receipt parsing; reserve unknown P3-R calls at this card's cap.

    Historical P3 accounting remains immutable in its original tool. Its
    unknown-slot defaults are not the cap of a newly selected P3-R setting.
    """
    budget = stage["budget"]
    input_reserve, output_cap = budget["input_reserve"], budget["max_output_tokens"]
    reader_stage = copy.deepcopy(stage)
    reader_stage["provider"]["max_output_tokens"] = output_cap
    report = exp.usage_report(output, reader_stage)
    prices = budget["prices_usd_per_million"]
    for row in report["attempts_detail"]:
        if row.get("usage_known") is False:
            row.update(input_tokens=input_reserve, output_tokens=output_cap,
                usd_peak=(input_reserve * prices["cache_miss"] + output_cap * prices["output"]) / 1e6)
    total = sum(row["usd_peak"] for row in report["attempts_detail"])
    # These two checks depend on the new card, not P3's 40k/32768 defaults.
    violations = set(report["violations"]) - {"stage_usd_exceeded", "actual_input_exceeded_reserved_units"}
    if total > budget["stage_usd"]:
        violations.add("stage_usd_exceeded")
    if any(row.get("usage_known") and row["input_tokens"] > input_reserve for row in report["attempts_detail"]):
        violations.add("actual_input_exceeded_reserved_units")
    report.update(usd_peak_actual_or_reserved=total, violations=sorted(violations),
        accounting_profile="p3r_selected_runtime_cap.v1",
        selected_input_reserve=input_reserve, selected_output_cap=output_cap,
        history_note="P3 receipts are parsed by the historical reader; only this new P3-R run's unknown reservations use its selected runtime cap.")
    return report


def preflight(args, exp):
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from true_rl_agent.prompt_serialization import canonical_prompt_json_bytes
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    if (args.output / "stage.json").exists():
        raise ValueError("use a new preparation directory, not a prior stage")
    data, old, old_base = source_card(args.data, exp)
    runtime = read(args.runtime_stage)
    budget = read(args.budget)
    p = copy.deepcopy(runtime["provider"])
    output_cap = runtime["budget"]["max_output_tokens"]
    input_reserve = runtime["budget"]["input_reserve"]
    p["max_output_tokens"] = output_cap
    prices = budget["provider"]["prices_usd_per_million"]
    worst_request_usd = (input_reserve * prices["cache_miss"] + output_cap * prices["output"]) / 1e6
    stage = {"schema_version": "feedback_p3r.ceiling_probe.v1", "status": "PREPARED_NOT_RELEASED",
        "data_ref": {"path": str(args.data.resolve()), "sha256": sha(args.data)},
        "budget_ref": {"path": str(args.budget.resolve()), "sha256": sha(args.budget)},
        "runtime_stage_ref": {"path": str(args.runtime_stage.resolve()), "sha256": sha(args.runtime_stage)},
        "runtime_summary_ref": None, "provider": p,
        "budget": {"stage_usd": budget["substage_limits_usd"]["r0b"],
                   "prices_usd_per_million": budget["provider"]["prices_usd_per_million"],
                   "max_attempts_per_author_episode": 5, "max_source_requests_per_author_episode": 2,
                   "input_reserve": input_reserve, "max_output_tokens": output_cap,
                   "worst_usd_per_request": worst_request_usd,
                   "planned_max_requests": 160, "default_episodes": 32,
                   "worst_total_usd": worst_request_usd * 160,
                   "allocation_sufficient_for_planned_maximum": worst_request_usd * 160 <= budget["substage_limits_usd"]["r0b"]},
        "review_profile": "model_review", "schedule": schedule(), "first_prompts": [],
        "production_source_snapshot": project_source_snapshot(args.root),
        "harness_sha256": {"author_ceiling_probe.py": sha(Path(__file__)), "author_experiment.py": sha(args.p3_tools / "author_experiment.py")},
        "source_groups": {"observation_packages": 2, "policy_clusters": 2, "independence_claim": False},
        "decision_rule": {"effective_variant": "At least two distinct cases each have >=1 of their two outputs with action relevance 2, mechanism yes and machine checks zero errors.",
                          "runtime_oracle_only": True, "value_gate": "NOT_EVALUATED", "training": "NOT_RUN"}}
    cfg = OpenAICompatibleProviderConfig(model=p["requested_model"], output_dir=args.output / "unused_candidates",
        base_url=p["base_url"], endpoint=p["endpoint"], max_tokens=output_cap,
        reasoning_mode=p["reasoning_mode"], reasoning_effort=p["reasoning_effort"],
        empty_final_reasoning_retry=p.get("empty_final_reasoning_retry", False), temperature=p["temperature"])
    provider = OpenAICompatibleCandidateProvider(cfg)
    for case in data["cases"]:
        payloads = {}
        for variant in VARIANTS:
            comp, builder, feedback = build_context(case, variant, old, data_base=args.data.parent, old_base=old_base,
                                                   root=args.root, output=args.output / "contexts" / case["case_id"] / variant, exp=exp)
            payload = first_prompt(comp, builder)
            if payload.get("public_feedback_summary") != feedback:
                raise ValueError(f"frozen feedback was changed or truncated in {case['case_id']}/{variant}")
            if old[case["case_id"]]["common_focus"] not in payload.get("selection_guidance", []):
                raise ValueError("registered case focus is missing from the actual first prompt")
            payloads[variant] = payload
            source, source_hash, role = select_authoring_anchor_source(payload)
            if not source or role != "current_best":
                raise ValueError("preflight lacks the genuine parent source")
            request = provider._chat_completion_request(payload)
            path = args.output / "prompts" / f"{case['case_id']}_{variant}.json"
            write(path, payload)
            stage["first_prompts"].append({"case_id": case["case_id"], "variant": variant,
                "canonical_prompt_sha256": prompt_hash(payload), "request_sha256": prompt_hash(request),
                "system_sha256": hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest(),
                "prompt_bytes": canonical_prompt_json_bytes(payload), "feedback_bytes": canonical_prompt_json_bytes(feedback),
                "exact_frozen_feedback_retained": True,
                "provider_task_sections": payload["authoring_context_frame"]["budget_metadata"]["provider_task_sections"],
                "request_bytes": canonical_prompt_json_bytes(request), "parent_source_sha256": source_hash,
                "path": str(path.relative_to(args.output)).replace("\\", "/")})
        for variant in VARIANTS[1:]:
            # Existing P3 fairness check permits only behavior and its bound
            # source-public-result hash, plus derived frame accounting.
            exp.assert_arm_fairness(old[case["case_id"]]["task"], payloads["A"], payloads[variant],
                projection_hashes={arm: payloads[variant if arm == "T" else "A"]["public_feedback_summary"]["feedback_entries"][0]["source_public_result_hash"] for arm in ("A", "T")}
                if old[case["case_id"]]["task"] == "stair" else None)
    write(args.output / "stage.json", stage)
    return {"status": stage["status"], "first_prompts": 16, "author_episodes": 32,
            "worst_usd": stage["budget"]["worst_total_usd"], "allocated_usd": stage["budget"]["stage_usd"],
            "budget_admission": "SUFFICIENT" if stage["budget"]["allocation_sufficient_for_planned_maximum"] else "INSUFFICIENT_NO_API"}


def run(args, exp):
    from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
    from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
    from true_rl_agent.agent_core.provider_errors import provider_failure_requires_service_recovery
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    stage = read(args.stage)
    if stage["status"] != "FROZEN_API_RELEASED":
        raise ValueError("R.0-b needs reviewed frozen release")
    budget = stage["budget"]
    worst_total = budget["worst_usd_per_request"] * budget["planned_max_requests"]
    if worst_total > budget["stage_usd"]:
        raise ValueError(f"allocated USD cannot cover the frozen request maximum: {worst_total} > {budget['stage_usd']}")
    runtime_summary = read(resolve_ref(stage["runtime_summary_ref"], args.stage.parent))
    if runtime_summary["runtime_gate"] != "PASS":
        raise ValueError("R.0-a runtime gate is not passed")
    runtime = read(resolve_ref(stage["runtime_stage_ref"], args.stage.parent))
    if any(budget[key] != runtime["budget"][key] for key in ("input_reserve", "max_output_tokens")):
        raise ValueError("request caps differ from the selected R.0-a runtime card")
    if stage["production_source_snapshot"] != project_source_snapshot(args.root):
        raise ValueError("production source changed since first-prompt preflight")
    if stage["harness_sha256"] != {"author_ceiling_probe.py": sha(Path(__file__)), "author_experiment.py": sha(args.p3_tools / "author_experiment.py")}:
        raise ValueError("experiment harness changed since preflight")
    data_path = resolve_ref(stage["data_ref"], args.stage.parent)
    resolve_ref(stage["budget_ref"], args.stage.parent)
    data, old, old_base = source_card(data_path, exp)
    cases = {c["case_id"]: c for c in data["cases"]}
    expected = {(r["case_id"], r["variant"]): r for r in stage["first_prompts"]}
    p = stage["provider"]
    caps = {"attempts_per_lane_per_generation": budget["max_attempts_per_author_episode"],
            "input_units_per_attempt_reserve": budget["input_reserve"], "output_units_per_attempt_cap": budget["max_output_tokens"]}
    with acquire_process_file_lock(args.output / ".ceiling_probe.lock", blocking=False):
        for row in stage["schedule"]:
            generation, lane = row["generation_id"], row["lane_id"]
            round_id = f"{generation}-{lane}"
            result_path = args.output / "results" / f"{round_id}.json"
            if result_path.exists():
                continue
            usage = usage_report(args.output, stage, exp)
            if usage["violations"] or usage["usd_peak_actual_or_reserved"] + stage["budget"]["worst_usd_per_request"] > stage["budget"]["stage_usd"]:
                raise RuntimeError("stage cannot admit another attempt")
            comp, builder, _ = build_context(cases[row["case_id"]], row["variant"], old, data_base=data_path.parent,
                                            old_base=old_base, root=args.root, output=args.output / "contexts" / round_id, exp=exp)
            provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model=p["requested_model"],
                output_dir=args.output / "provider_candidates" / generation / lane, base_url=p["base_url"], endpoint=p["endpoint"],
                api_key_env=p["api_key_env"], env_file=Path(p["env_file"]), max_tokens=budget["max_output_tokens"],
                timeout_s=p["timeout_s"], max_response_bytes=p["max_response_bytes"], reasoning_mode=p["reasoning_mode"],
                reasoning_effort=p["reasoning_effort"], empty_final_reasoning_retry=p.get("empty_final_reasoning_retry", False), temperature=p["temperature"]))
            initial = first_prompt(comp, builder)
            if prompt_hash(provider._chat_completion_request(initial)) != expected[(row["case_id"], row["variant"])]["request_sha256"]:
                raise ValueError("actual initial request differs from freeze")
            provider.validate_credential()
            guarded = CostGuardedProvider(delegate=provider, output_dir=args.output, generation_id=generation,
                                          lane_id=lane, provider_caps=caps, max_generations=16)
            result = CandidateAuthoringLoop(context_builder=builder, provider=guarded,
                candidate_admission_policy=comp.candidate_admission_policy, round_dir=args.output / "authoring" / generation / lane,
                max_attempts=budget["max_attempts_per_author_episode"], max_source_requests_per_round=budget["max_source_requests_per_author_episode"], component_smoke_adapter=comp.task_plugin.component_smoke_adapter(),
                authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping()).run_round(round_id=round_id, records=[])
            write(result_path, {**row, "stop_reason": result.stop_reason,
                "candidate_private_path": result.candidate.path if result.candidate else None,
                "candidate": result.candidate.to_public_dict() if result.candidate else None,
                "authoring_checks": result.attempt_records, "prompt_payload_ref": result.prompt_payload_ref.to_public_dict(),
                "authoring_evidence_ref": result.authoring_evidence_ref.to_private_dict(),
                "static_failure": result.static_failure, "provider_failure": result.provider_failure,
                "train_status": "not_run", "review_profile": "model_review", "review_status": "PENDING"})
            usage = usage_report(args.output, stage, exp)
            write(args.output / "usage.json", usage)
            print(json.dumps({"case_id": row["case_id"], "variant": row["variant"], "repetition": row["repetition"],
                              "stop_reason": result.stop_reason, "usd_peak": usage["usd_peak_actual_or_reserved"]}), flush=True)
            if usage["violations"]:
                raise RuntimeError("receipt violates the frozen experiment; result preserved")
            if any(not all(isinstance(record.get("provider_usage", {}).get(key), int)
                           for key in ("request_units", "response_units", "total_units"))
                   and not provider_failure_requires_service_recovery(record.get("provider_failure"))
                   for record in result.attempt_records):
                raise RuntimeError("provider usage unavailable; episode preserved, later episodes not started")
    return {"episodes": len(list((args.output / "results").glob("*.json"))), "usage": usage_report(args.output, stage, exp)}


def export(args, exp):
    from true_rl_agent.memory import PromptPayloadRef
    from true_rl_agent.prompt_artifacts import verify_prompt_payload
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    from blind_export import rationale, reference_locations, reward_checks, IDENTITY_PATTERN
    stage = read(args.stage)
    # Assign IDs using the entire frozen order before selecting a case. Sorting
    # matches the original full export's result-file order, including failures.
    ordered = sorted(stage["schedule"], key=lambda r: (r["generation_id"], r["lane_id"]))
    random.Random(2026092903).shuffle(ordered)
    selected = [(f"proposal-{index:03d}", spec) for index, spec in enumerate(ordered, 1)
                if args.case_id is None or spec["case_id"] == args.case_id]
    count = 32 if args.case_id is None else 8
    if len(ordered) != 32 or len(selected) != count:
        raise ValueError("export needs the frozen 32-slot schedule and all 8 slots of a selected case")
    rows = []
    identity_keys = ("case_id", "variant", "repetition", "generation_id", "lane_id")
    for ident, spec in selected:
        path = args.results / "results" / f"{spec['generation_id']}-{spec['lane_id']}.json"
        if not path.is_file():
            raise ValueError(f"blind export requires all {count} selected terminals including failures")
        row = read(path)
        if any(row[key] != spec[key] for key in identity_keys):
            raise ValueError("terminal identity differs from the frozen schedule")
        rows.append((ident, row))
    if args.output.exists():
        raise ValueError("blind output must be new")
    data_path = resolve_ref(stage["data_ref"], args.stage.parent)
    data, old, old_base = source_card(data_path, exp)
    review_reference = {c["case_id"]: c.get("review_reference", {}) for c in data["cases"]
                        if args.case_id is None or c["case_id"] == args.case_id}
    mapping, sheet = [], []
    reviewer = args.output / "reviewer"
    for ident, row in rows:
        seen = verify_prompt_payload(PromptPayloadRef.from_mapping(row["prompt_payload_ref"]))
        code = (Path(row["candidate_private_path"]) / "training_code" / "components.py").read_text(encoding="utf-8") if row["candidate_private_path"] else ""
        reason = rationale(code) if code else ""
        original = old[row["case_id"]]
        truth = exp.bound_read(original["blind_truth"], old_base)
        initial_ref = next(r for r in stage["first_prompts"] if (r["case_id"], r["variant"]) == (row["case_id"], row["variant"]))
        initial = read(args.stage.parent / initial_ref["path"])
        parent_source, _, _ = select_authoring_anchor_source(initial)
        write(reviewer / f"{ident}-seen.json", seen)
        write(reviewer / f"{ident}-initial.json", initial)
        (reviewer / f"{ident}.py").write_text(code, encoding="utf-8")
        (reviewer / f"{ident}-parent.py").write_text(parent_source or "", encoding="utf-8")
        write(reviewer / f"truth-{row['case_id']}.json", truth)
        aids = {"reference_locations": reference_locations(reason, seen, truth),
                "reward_checks": reward_checks(code, initial, row["authoring_checks"]),
                "semantic_fact_check": "PENDING_MODEL_REVIEW; matching a number or identifier is not proof of correctness",
                "explicit_identity_warnings": sorted(set(IDENTITY_PATTERN.findall(code + json.dumps(seen, ensure_ascii=False))))}
        write(reviewer / f"{ident}-machine.json", aids)
        sheet.append({"proposal_id": ident, "case_id": row["case_id"], "focus": original["common_focus"],
            "terminal": row["stop_reason"], "rationale": reason, "machine_errors": None, "machine_claim_checks": [],
            "mechanism_matches_registered_failure": None, "action_relevance_0_2": None,
            "reviewer": None, "evidence_references": [], "reviewer_notes": ""})
        mapping.append({"proposal_id": ident, **{k: row[k] for k in ("case_id", "variant", "repetition", "generation_id", "lane_id")}})
    scope = {"case_id": args.case_id, "exported_terminals": count, "planned_terminals": 32,
             "materials_scope": "FULL_32" if args.case_id is None else "CASE_8_ONLY_FULL_32_NOT_EXPORTED",
             "global_conclusion": "NOT_EVALUATED"}
    write(reviewer / "review_sheet.json", {"profile": "model_review", "status": "PENDING", "scope": scope, "items": sheet})
    write(reviewer / "case_review_reference.json", review_reference)
    write(reviewer / "rubric.json", {"mechanism": "yes / no / unjudgeable: whether the proposed mechanism addresses this case's registered failure shape; no unrun causal verdict",
        "action_relevance": {"0": "not actionable or disconnected", "1": "plausible but weak link", "2": "explicit process evidence supports the bounded modification"},
        "machine_errors": "Only directly established numeric/source/code contradictions; hypothesis and future target are not observations. Unresolved checks cannot become zero errors.",
        "blindness_limit": "Variant labels hidden; input format can expose the condition. This is model_review, not human adjudication."})
    write(args.output / "withhold_mapping.json", mapping)
    result = {"status": "BLIND_MATERIALS_READY" if args.case_id is None else "CASE_BLIND_MATERIALS_READY",
              "proposals": count, "profile": "model_review", "judgments": "PENDING", "scope": scope}
    write(args.output / "export_summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run", "export", "inspect"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--p3-tools", type=Path, default=Path(__file__).resolve().parent.parent / "P3")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--budget", type=Path, default=Path(__file__).with_name("stage_budget.json"))
    parser.add_argument("--runtime-stage", type=Path)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--case-id", choices=sorted(CASE_IDS), help="export only; requires all eight terminals for this case")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.case_id is not None and args.mode != "export":
        parser.error("--case-id is only available for export")
    sys.path.insert(0, str(args.root / "src"))
    exp = load_p3(args.p3_tools)
    if args.mode == "preflight":
        result = preflight(args, exp)
    elif args.mode == "run":
        try:
            result = run(args, exp)
        finally:
            # Production service-recovery exceptions stop the invocation, but
            # their journal reservations must still appear in the stage total.
            write(args.output / "usage.json", usage_report(args.output, read(args.stage), exp))
    elif args.mode == "export":
        result = export(args, exp)
    else:
        result = usage_report(args.output, read(args.stage), exp)
        write(args.output / "usage.json", result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
