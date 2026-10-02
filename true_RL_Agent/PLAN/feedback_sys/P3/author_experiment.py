"""P3-only paired author experiment; production builders/journals, no RL jobs.

``preflight`` never creates a provider or reads credentials. ``run`` requires
the frozen case card and uses the original bounded author/repair/source loop.
Pair slots use generation-budget directory names solely to reuse that budget
guard; they are independent author episodes, not RL generations or a search.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bound_read(ref: dict[str, Any], base: Path) -> Any:
    path = Path(ref["path"])
    if not path.is_absolute():
        path = base / path
    if digest(path) != ref["sha256"].removeprefix("sha256:"):
        raise ValueError(f"frozen source changed: {path}")
    value = read(path)
    for token in ref.get("pointer", "").split("/")[1:]:
        token = token.replace("~1", "/").replace("~0", "~")
        value = value[int(token)] if isinstance(value, list) else value[token]
    return value


class FrozenFeedbackAdapter:
    """Experiment view only; production adapter owns all non-feedback fields."""
    def __init__(self, delegate: Any, feedback: dict[str, Any], focus: str):
        self.delegate, self.feedback, self.focus = delegate, feedback, focus

    def __getattr__(self, name: str) -> Any:
        return getattr(self.delegate, name)

    def build_prompt_sections(self, **kwargs: Any) -> dict[str, Any]:
        from true_rl_agent.tasks.prompt_adapter import sanitize_task_prompt_sections
        sections = copy.deepcopy(self.delegate.build_prompt_sections(**kwargs))
        sections["public_feedback_summary"] = copy.deepcopy(self.feedback)
        sections["selection_guidance"] = [*sections["selection_guidance"], self.focus]
        return sanitize_task_prompt_sections(sections)


def reference_context(spec: dict[str, Any], *, root: Path, base: Path, output: Path):
    """Restore existing evidence. Never promotes a new public-dev-only result."""
    from true_rl_agent.composition.base import TaskCompositionContext
    from true_rl_agent.evaluation.binding import EvaluationBindingV1, canonical_json_hash
    from true_rl_agent.evaluation.current_best import CurrentBestEvalStats
    if spec["task"] == "stair":
        from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider as Provider
    elif spec["task"] == "push":
        from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider as Provider
    else:
        raise ValueError("unknown task")
    provider = Provider()
    comp = provider.resolve(values={**provider.cli_defaults(), **spec["composition_values"]},
        context=TaskCompositionContext(project_root=root, output_dir=output / "composition"))
    protocol = comp.task_plugin.task_spec.acceptance_protocol
    source = comp.reference_program_artifact.source_text
    if "sha256:" + hashlib.sha256(source.encode()).hexdigest() != spec["reference_program_source_checksum"]:
        raise ValueError("current reference source differs from the actual qualified source")
    genesis_hash = None
    initial_policy_hash = None
    if spec["mode"] == "existing_reference_genesis":
        from true_rl_agent.agent_core.session_genesis import SessionGenesisArtifact
        raw = bound_read(spec["genesis"], base)
        genesis = SessionGenesisArtifact.from_mapping(raw)
        if genesis.task_id != comp.task_id or genesis.genesis_kind != "reference_program":
            raise ValueError("historical Genesis is not this task's reference")
        anchor = raw["anchor"]
        reference = anchor["reference_manifest"]
        public = anchor["public_result"]
        binding = EvaluationBindingV1.from_mapping(raw["evaluation_binding"])
        identity = {"reference_id": reference["reference_id"], "reference_manifest_hash": reference["manifest_hash"],
            "program_source_checksum": reference["source_checksum"], "training_binding_hash": anchor["training_binding"]["binding_hash"],
            "capability_profile_hash": reference["capability_profile_hash"], "checkpoint_hash": anchor["checkpoint_hash"]}
        genesis_hash = genesis.session_genesis_hash
        initial_policy_hash = anchor["initial_policy_state_hash"]
    elif spec["mode"] == "existing_stair_qualification":
        # Restore the saved assembly with the runner's public projection and the
        # task's qualification reader. The source files remain historical facts.
        if spec["task"] != "stair":
            raise ValueError("this saved-qualification reader is Stair-owned")
        saved = bound_read(spec["qualification_result"], base)
        assembly = bound_read(spec["qualification_assembly"], base)
        trained = bound_read(spec["training_summary"], base)
        if saved["status"] != "PASS" or saved["reference_gate"]["status"] != "pass":
            raise ValueError("saved reference is not qualified")
        summary = assembly["summary"]
        if canonical_json_hash(summary) != assembly["summary_hash"]:
            raise ValueError("saved assembly summary hash differs")
        phase = saved["phases"]["public_dev"]
        binding = EvaluationBindingV1.from_mapping(summary["evaluation_binding"])
        raw_public = comp.runner._public_result_from_summary(summary,
            job_id=summary["job_id"], expected_checkpoint_hash=saved["checkpoint"],
            expected_eval_protocol_hash=binding.eval_protocol_hash,
            expected_evaluation_binding_hash=binding.binding_hash,
            expected_sealed_seed_count=comp.task_plugin.config.sealed_validation_seed_count,
            evaluation_binding=binding, evaluation_subject=phase["subject"],
            subject_manifest_hash=phase["subject_manifest_hash"],
            program_source_checksum=phase["program_source_checksum"],
            training_binding_hash=phase["training_binding_hash"],
            capability_profile_hash=phase["capability_profile_hash"], expected_execution_device=summary["execution_device"])
        qualified = comp.task_plugin.qualify_reference_result(public_result=raw_public, train_summary=trained)
        if dict(qualified.gate) != saved["reference_gate"]:
            raise ValueError("current reader did not restore the original qualification gate")
        public = qualified.public_result.to_public_dict()
        identity = {"reference_id": phase["subject"]["id"], "reference_manifest_hash": phase["subject_manifest_hash"],
            "program_source_checksum": phase["program_source_checksum"], "training_binding_hash": phase["training_binding_hash"],
            "capability_profile_hash": phase["capability_profile_hash"], "checkpoint_hash": saved["checkpoint"]}
    else:
        raise ValueError("no supported, already qualified reference context")
    if identity["checkpoint_hash"] != spec["checkpoint_hash"] or identity["program_source_checksum"] != spec["reference_program_source_checksum"]:
        raise ValueError("case parent differs from its qualification evidence")
    if binding.task_protocol_hash != comp.task_plugin.task_spec.resolved_task_protocol_hash():
        raise ValueError("qualification and current task domains differ")
    best = CurrentBestEvalStats.from_reference_public_result_mapping(**identity, public_result=public,
        acceptance_protocol=protocol, expected_evaluation_binding=binding)
    if "diagnostic_public_dev_phase" in spec:
        # A new diagnostic view updates only public-dev observations. Original
        # qualification/acceptance evidence above is never overwritten.
        projected = diagnostic_phase(spec, comp, best, base=base)
        public = copy.deepcopy(public)
        public["metrics"]["public_dev"] = projected
    evidence = {"anchor_kind": "reference_program", "public_result": public, "public_result_hash": canonical_json_hash(public)}
    if genesis_hash is None:
        # Empty-history PublicContextBuilder requires a replay namespace even
        # for a pre-Genesis qualified reference. This is a NEW diagnostic
        # context identity, not an assertion that a historical Genesis exists.
        # No SessionGenesisArtifact or ledger is published by this experiment.
        genesis_hash = canonical_json_hash({"kind": "p3_empty_diagnostic_author_context",
            "qualified_reference": best.to_public_dict(),
            "qualification_result_sha256": spec["qualification_result"]["sha256"],
            "qualification_assembly_sha256": spec["qualification_assembly"]["sha256"]})
    if "diagnostic_public_dev_phase" in spec:
        genesis_hash = canonical_json_hash({"kind": "p3_empty_diagnostic_author_context",
            "qualified_reference_namespace": genesis_hash,
            "diagnostic_public_dev_phase_sha256": spec["diagnostic_public_dev_phase"]["sha256"],
            "diagnostic_as_of": spec["diagnostic_as_of"]})
    comp.task_plugin.bind_genesis_evidence(evidence)
    return comp, best, evidence, source, genesis_hash, initial_policy_hash


def diagnostic_phase(spec, comp, best, *, base, include_behavior=True):
    from true_rl_agent.evaluation.phases import EvaluationPhase, EvaluationPhaseResult
    phase = EvaluationPhaseResult.from_private_mapping(bound_read(spec["diagnostic_public_dev_phase"], base))
    if phase.phase is not EvaluationPhase.PUBLIC_DEV or phase.subject.kind != "reference":
        raise ValueError("diagnostic source must be a real reference public-dev phase")
    expected = {"checkpoint_hash": best.checkpoint_hash, "program_source_checksum": best.program_source_checksum,
        "training_binding_hash": best.training_binding_hash, "capability_profile_hash": best.capability_profile_hash,
        "eval_protocol_hash": best.eval_protocol_hash, "evaluation_binding_hash": best.evaluation_binding_hash,
        "subject_manifest_hash": best.subject_manifest_hash, "task_id": comp.task_id}
    if any(getattr(phase, key) != value for key, value in expected.items()):
        raise ValueError("fresh public-dev phase differs from the qualified reference/domain")
    projected = comp.task_plugin.phased_evaluation_adapter().project_public_phase_summary(phase_result=phase)
    behavior = projected.get("behavior")
    if not isinstance(behavior, dict):
        raise ValueError("fresh phase did not seal behavior")
    if spec["task"] == "stair":
        if behavior["domain"]["checkpoint_hash"] != best.checkpoint_hash:
            raise ValueError("Stair behavior parent differs")
        if behavior["domain"]["archive_hash"] != phase.task_evidence["numeric_evidence"]["manifest_sha256"]:
            raise ValueError("Stair behavior archive differs")
    else:
        binding = behavior["binding"]
        if (binding["checkpoint_hash"] != best.checkpoint_hash or binding["program_source_checksum"] != best.program_source_checksum
                or binding["evaluation_plan_hash"] != phase.evaluation_plan_hash):
            raise ValueError("Push behavior source differs from its phase")
        archived = [s["numeric_evidence_ref"]["manifest_sha256"] for s in phase.task_evidence["seed_summaries"]]
        if sorted(archived) != sorted(s["manifest_sha256"] for s in behavior["sources"]):
            raise ValueError("Push behavior archives differ from phase references")
    if not include_behavior:
        # Form A from the source receipt before either phase projection or
        # task feedback compression. Do not derive A by trimming rendered T.
        from dataclasses import replace
        evidence_without_behavior = copy.deepcopy(dict(phase.task_evidence))
        evidence_without_behavior.pop("behavior")
        projected = comp.task_plugin.phased_evaluation_adapter().project_public_phase_summary(
            phase_result=replace(phase, task_evidence=evidence_without_behavior))
    return projected


def feedback_anchors(spec, comp, best, evidence, *, base):
    """Derived public projections keep their own correct content hashes."""
    from true_rl_agent.evaluation.binding import canonical_json_hash
    anchors = {}
    for arm in ("A", "T"):
        anchor = copy.deepcopy(evidence)
        anchor["public_result"]["metrics"]["public_dev"] = diagnostic_phase(
            spec, comp, best, base=base, include_behavior=arm == "T")
        # The saved pre-P3 qualifications contain no top-level behavior. A
        # future input that does would be a different, unregistered experiment.
        if "behavior" in anchor["public_result"]["metrics"]:
            raise ValueError("qualified anchor unexpectedly contains another behavior observation")
        anchor["public_result_hash"] = canonical_json_hash(anchor["public_result"])
        anchors[arm] = anchor
    baseline, treatment = (copy.deepcopy(anchors[arm]["public_result"]) for arm in ("A", "T"))
    for source in (baseline, treatment):
        source["metrics"]["public_dev"].pop("behavior", None)
    if baseline != treatment:
        raise ValueError("A/T source public objects changed outside measured behavior")
    return anchors


def feedback_views(spec, comp, best, evidence, *, base):
    """Both arms use the real adapter and one measured public-dev phase."""
    views = {}
    for arm, anchor in feedback_anchors(spec, comp, best, evidence, base=base).items():
        views[arm] = comp.task_plugin.prompt_adapter().build_prompt_sections(
            public_records=(), public_context={"initial_anchor_public_evidence": anchor})["public_feedback_summary"]
    owners = {arm: (view if spec["task"] == "stair" else view["public_dev"]) for arm, view in views.items()}
    baseline_behavior = owners["A"].get("behavior")
    # Push intentionally renders an honest missing placeholder for old phases.
    # Keep that original A content; only measured behavior belongs in T.
    if baseline_behavior is not None and (not isinstance(baseline_behavior, dict)
            or baseline_behavior.get("status") not in {"missing", "not_available"}):
        raise ValueError("baseline feedback unexpectedly contains measured behavior")
    treatment_behavior = owners["T"].get("behavior")
    measurement_key = "profile" if spec["task"] == "stair" else "binding"
    if not isinstance(treatment_behavior, dict) or not isinstance(treatment_behavior.get(measurement_key), dict):
        raise ValueError("real task adapter did not expose behavior only in T")
    return views


def phase_source_bindings(spec, *, base):
    phase = bound_read(spec["diagnostic_public_dev_phase"], base)
    behavior = phase["task_evidence"]["behavior"]
    archives = ([behavior["domain"]["archive_hash"]] if spec["task"] == "stair" else
                sorted(s["manifest_sha256"] for s in behavior["sources"]))
    return {"phase_sha256": spec["diagnostic_public_dev_phase"]["sha256"],
        "checkpoint_hash": phase["checkpoint_hash"], "program_source_checksum": phase["program_source_checksum"],
        "training_binding_hash": phase["training_binding_hash"], "evaluation_plan_hash": phase["evaluation_plan_hash"],
        "evaluation_binding_hash": phase["evaluation_binding_hash"], "archive_manifest_sha256": archives,
        "availability": "new diagnostic observation; not the original qualification/author decision time",
        "as_of_cutoff": spec["diagnostic_as_of"]}


def build_case(case: dict[str, Any], arm: str, *, base: Path, root: Path, output: Path):
    from true_rl_agent.agent_core.ledger import PROVIDER_AUTHORING_MODE, primary_ledger_epoch_hash
    from true_rl_agent.memory.context_builder import PublicContextBuilder
    spec = bound_read(case["author_context"], base)
    if case.get("as_of_cutoff") != spec.get("diagnostic_as_of"):
        raise ValueError("case and diagnostic availability cutoffs differ")
    comp, best, evidence, source, genesis_hash, initial_hash = reference_context(spec, root=root, base=base, output=output)
    if case["checkpoint_hash"] != best.checkpoint_hash:
        raise ValueError("diagnosed checkpoint is not the authoring reference")
    feedback = bound_read(case[f"feedback_{arm}"], base)
    if feedback != feedback_views(spec, comp, best, evidence, base=base)[arm]:
        raise ValueError("frozen feedback differs from its source phase and genuine task adapter")
    if case["source_bindings"] != phase_source_bindings(spec, base=base):
        raise ValueError("case source bindings differ from the measured phase")
    adapter = FrozenFeedbackAdapter(comp.task_plugin.prompt_adapter(), feedback, case["common_focus"])
    builder = PublicContextBuilder(task_memory_adapter=comp.task_memory_adapter,
        task_prompt_adapter=adapter, phased_evaluation_adapter=comp.task_plugin.phased_evaluation_adapter(),
        acceptance_protocol=comp.task_plugin.task_spec.acceptance_protocol, initial_current_best=best,
        session_genesis_hash=genesis_hash, candidate_capability_profile=comp.task_plugin.task_spec.candidate_capability_profile,
        initial_reference_program_source=source, initial_anchor_public_evidence=evidence,
        expected_initial_policy_state_hash=initial_hash, ledger_authoring_mode=PROVIDER_AUTHORING_MODE,
        expected_ledger_epoch_hash=primary_ledger_epoch_hash(genesis_hash) if genesis_hash else None)
    hashes = {key: anchor["public_result_hash"] for key, anchor in
        feedback_anchors(spec, comp, best, evidence, base=base).items()}
    expected_hash_changes = ([{"path": "public_feedback_summary.feedback_entries[0].source_public_result_hash",
        "canonical_source_hash_by_arm": hashes, "source_objects_differ_only_by_behavior": True}]
        if spec["task"] == "stair" else [])
    if case.get("derived_feedback_hash_changes") != expected_hash_changes:
        raise ValueError("case card derived provenance differs from canonical source objects")
    return comp, builder, hashes


def schedule(stage: dict[str, Any], cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(stage["design"]["randomization_seed"])
    ordered = sorted(case["case_id"] for case in cases)
    rng.shuffle(ordered)
    blocks = []
    for case_id in ordered:
        first = rng.choice(["A", "T"])
        for repetition in range(1, 3):
            order = [first, "T" if first == "A" else "A"]
            if repetition == 2:
                order.reverse()
            index = len(blocks) + 1
            blocks.append({"slot": index, "case_id": case_id, "repetition": repetition,
                "generation_id": f"{stage['stage_id']}-generation-{index:03d}", "arm_order": order,
                "lane_by_arm": {"A": "lane-a", "T": "lane-b"}})
    return blocks


def usage_report(output: Path, stage: dict[str, Any]) -> dict[str, Any]:
    """Derive stage accounting from existing authoritative author journals."""
    from true_rl_agent.prompt_artifacts import RoundAuthoringJournal
    rows, violations, protocol_deviations = [], [], []
    price = stage["budget"]["prices_usd_per_million"]
    for lane in sorted((output / "authoring").glob("*/lane-*")):
        journal = RoundAuthoringJournal(lane, round_id=lane.parent.name + "-" + lane.name)
        for attempt in journal.snapshot().attempts:
            if attempt["status"] == "interrupted_before_prompt":
                rows.append({"round_id": journal.round_id, "sequence": attempt["sequence"], "usd_peak": 0, "status": attempt["status"]})
                continue
            provider = attempt.get("provider_attempt") or {}
            usage = provider.get("provider_usage") or {}
            known = all(isinstance(usage.get(k), int) for k in ("request_units", "response_units"))
            inputs, outputs = (usage["request_units"], usage["response_units"]) if known else (40000, 32768)
            hit, miss = usage.get("request_cache_hit_units"), usage.get("request_cache_miss_units")
            cache_known = isinstance(hit, int) and isinstance(miss, int) and hit + miss == inputs
            usd = (((hit * price["cache_hit"] + miss * price["cache_miss"]) if cache_known else inputs * price["cache_miss"])
                + outputs * price["output"]) / 1_000_000
            receipt = provider.get("provider_response_receipt") or {}
            rows.append({"round_id": journal.round_id, "sequence": attempt["sequence"], "status": attempt["status"],
                "usage_known": known, "cache_split_known": cache_known, "input_tokens": inputs, "output_tokens": outputs,
                "usd_peak": usd, "response_model": receipt.get("response_model"), "system_fingerprint": receipt.get("system_fingerprint")})
            if known and inputs > 40000:
                violations.append("actual_input_exceeded_reserved_units")
            if known and outputs > stage["provider"]["max_output_tokens"]:
                protocol_deviations.append({"kind": "provider_output_exceeded_requested_units",
                    "round_id": journal.round_id, "sequence": attempt["sequence"],
                    "requested_output_units": stage["provider"]["max_output_tokens"],
                    "actual_output_units": outputs,
                    "excess_output_units": outputs - stage["provider"]["max_output_tokens"],
                    "finish_reason": receipt.get("finish_reason"),
                    "response_model": receipt.get("response_model"),
                    "accounting": "full authoritative usage retained; never clipped to requested output"})
    total = sum(r["usd_peak"] for r in rows)
    if total > stage["budget"]["stage_usd"]:
        violations.append("stage_usd_exceeded")
    models = sorted({r["response_model"] for r in rows if r.get("response_model")})
    if len(models) > 1:
        violations.append("response_model_changed_within_stage")
    return {"attempts": len(rows), "usd_peak_actual_or_reserved": total, "price_basis": "peak; unknown usage fully reserved",
        "response_models": models, "violations": sorted(set(violations)),
        "protocol_deviations": protocol_deviations,
        "protocol_compliance": "HAS_RECORDED_DEVIATIONS" if protocol_deviations else "NO_RECORDED_DEVIATIONS",
        "attempts_detail": rows}


def assert_arm_fairness(task: str, a: dict[str, Any], t: dict[str, Any], *, projection_hashes=None) -> None:
    left, right = copy.deepcopy(a), copy.deepcopy(t)
    for arm, value in (("A", left), ("T", right)):
        feedback = value["public_feedback_summary"]
        owner = feedback if task == "stair" else feedback["public_dev"]
        owner.pop("behavior", None)
        if task == "stair" and "feedback_entries" in feedback:
            entries = feedback["feedback_entries"]
            if (len(entries) != 1 or projection_hashes is None
                    or entries[0].get("source_public_result_hash") != projection_hashes[arm]):
                raise ValueError("Stair public-result projection hash is not the recomputed canonical source")
            # A removes behavior before projection. Its content hash must
            # differ from T; retaining T's hash would mislabel A's source.
            entries[0].pop("source_public_result_hash")
        frame = value["authoring_context_frame"]
        budget = frame.pop("budget_metadata")
        # Byte/token estimates and task-section receipts are derived from the
        # differing representation. Their hard ceilings must remain identical.
        frame["frozen_hard_limits"] = budget["hard_limits"]
    if left != right:
        raise ValueError("A/T changed something outside task behavior and derived budget measurements")


def harness_hashes() -> dict[str, str]:
    return {name: digest(Path(__file__).with_name(name)) for name in
        ("author_experiment.py", "case_prepare.py", "case_freeze.py", "blind_export.py", "author_summary.py")}


def preflight(stage: dict[str, Any], cases: list[dict[str, Any]], *, base: Path, root: Path, output: Path) -> dict[str, Any]:
    from true_rl_agent.prompt_serialization import canonical_prompt_json, canonical_prompt_json_bytes
    from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
    rows, derived_hashes = [], []
    for case in cases:
        payloads = {}
        for arm in ("A", "T"):
            comp, builder, projection_hashes = build_case(case, arm, base=base, root=root, output=output / "contexts" / case["case_id"] / arm)
            payload = builder.build_prompt([], authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
                candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules()).provider_payload
            source, source_hash, role = select_authoring_anchor_source(payload)
            if source is None or role != "current_best":
                raise ValueError("preflight did not retain the genuine reference source")
            path = output / "prompts" / f"{case['case_id']}_{arm}.json"
            write(path, payload)
            payloads[arm] = payload
            rows.append({"case_id": case["case_id"], "arm": arm, "path": str(path), "sha256": digest(path),
                "canonical_prompt_sha256": hashlib.sha256(canonical_prompt_json(payload).encode()).hexdigest(),
                "prompt_bytes": canonical_prompt_json_bytes(payload), "parent_source_sha256": source_hash,
                "cluster_id": case["cluster_id"]})
        assert_arm_fairness(case["task"], payloads["A"], payloads["T"], projection_hashes=projection_hashes)
        if case["task"] == "stair":
            derived_hashes.append({"case_id": case["case_id"],
                "path": "public_feedback_summary.feedback_entries[0].source_public_result_hash",
                "canonical_source_hash_by_arm": projection_hashes,
                "reason": "same measured phase; A omits behavior before task projection, T retains it"})
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    return {"status": "PASS", "api_requests": 0, "gpu_jobs": 0, "prompts": rows,
        "unique_prompt_count": len({r["canonical_prompt_sha256"] for r in rows}), "schedule": schedule(stage, cases),
        "production_source_snapshot": project_source_snapshot(root), "harness_sha256": harness_hashes(),
        "checked_derived_projection_hash_changes": derived_hashes}


def run(stage, cases, *, base, root, output, freeze):
    from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
    from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
    from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig
    from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
    if freeze["status"] != "FROZEN" or freeze["root_api_authorized"] is not True:
        raise ValueError("case freeze and explicit root API release are required")
    from true_rl_agent.runner.source_snapshot import project_source_snapshot
    if project_source_snapshot(root) != freeze["production_source_snapshot"] or harness_hashes() != freeze["harness_sha256"]:
        raise ValueError("production verifier/loop/smoke or experiment code changed since freeze")
    caps = {"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 40000, "output_units_per_attempt_cap": 32768}
    p = stage["provider"]
    if stage["budget"]["worst_usd_240_requests"] > stage["budget"]["stage_usd"]:
        raise ValueError("frozen peak-price allocation exceeds the stage cap")
    by_id = {c["case_id"]: c for c in cases}
    from true_rl_agent.prompt_serialization import canonical_prompt_json
    expected_prompts = {(r["case_id"], r["arm"]): r["canonical_prompt_sha256"] for r in freeze["first_prompts"]}
    with acquire_process_file_lock(output / ".author_experiment.lock", blocking=False):
        for block in freeze["schedule"]:
            case = by_id[block["case_id"]]
            for arm in block["arm_order"]:
                lane = block["lane_by_arm"][arm]
                generation = block["generation_id"]
                round_id = generation + "-" + lane
                result_path = output / "results" / (round_id + ".json")
                if result_path.exists():
                    continue
                actuals = usage_report(output, stage)
                write(output / "usage.json", actuals)
                if actuals["violations"] or actuals["usd_peak_actual_or_reserved"] + stage["budget"]["worst_usd_per_request"] > stage["budget"]["stage_usd"]:
                    raise RuntimeError("stage usage cannot admit another author episode")
                comp, builder, _ = build_case(case, arm, base=base, root=root, output=output / "contexts" / round_id)
                current = builder.build_prompt([], authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping(),
                    candidate_verifier_prompt_rules=comp.candidate_admission_policy.authoring_prompt_rules()).provider_payload
                if hashlib.sha256(canonical_prompt_json(current).encode()).hexdigest() != expected_prompts[(case["case_id"], arm)]:
                    raise ValueError("actual first prompt differs from the frozen preflight")
                provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(model=p["requested_model"],
                    output_dir=output / "provider_candidates" / generation / lane, base_url=p["base_url"], endpoint=p["endpoint"],
                    api_key_env=p["api_key_env"], env_file=Path(p["env_file"]), max_tokens=p["max_output_tokens"],
                    timeout_s=p["timeout_s"], max_response_bytes=p["max_response_bytes"], reasoning_mode=p["reasoning_mode"],
                    reasoning_effort=p["reasoning_effort"], temperature=p["temperature"]))
                guarded = CostGuardedProvider(delegate=provider, output_dir=output, generation_id=generation,
                    lane_id=lane, provider_caps=caps, max_generations=24,
                    allow_conservative_interrupted_recovery=True)
                result = CandidateAuthoringLoop(context_builder=builder, provider=guarded,
                    candidate_admission_policy=comp.candidate_admission_policy,
                    round_dir=output / "authoring" / generation / lane, max_attempts=5,
                    max_source_requests_per_round=2, component_smoke_adapter=comp.task_plugin.component_smoke_adapter(),
                    authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping()).run_round(round_id=round_id, records=[])
                write(result_path, {"case_id": case["case_id"], "arm": arm, "repetition": block["repetition"],
                    "cluster_id": case["cluster_id"], "stop_reason": result.stop_reason,
                    "authoring_checks": [{key: row[key] for key in ("attempt", "provider_status", "provider_response_mode",
                        "verifier_status", "verifier_reason", "component_smoke") if key in row} for row in result.attempt_records],
                    "static_failure": result.static_failure, "provider_failure": result.provider_failure,
                    "candidate": result.candidate.to_public_dict() if result.candidate else None,
                    "candidate_private_path": result.candidate.path if result.candidate else None,
                    "prompt_payload_ref": result.prompt_payload_ref.to_public_dict(),
                    "authoring_evidence_ref": result.authoring_evidence_ref.to_private_dict(),
                    "parent_program_source_checksum": result.parent_program_source_checksum,
                    "train_status": "not_run", "human_review": "pending"})
                actuals = usage_report(output, stage)
                write(output / "usage.json", actuals)
                print(json.dumps({"case_id": case["case_id"], "arm": arm, "repetition": block["repetition"],
                    "stop_reason": result.stop_reason, "candidate_verified": result.stop_reason == "candidate_verified",
                    "stage_attempts": actuals["attempts"], "stage_usd_peak": actuals["usd_peak_actual_or_reserved"]}), flush=True)
                if actuals["violations"]:
                    raise RuntimeError("provider usage/model violates the frozen protocol; receipt preserved")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "run"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.root / "src"))
    stage, card = read(args.stage), read(args.cases)
    cases = card["cases"]
    if len(cases) != 12 or any(c.get("status") != "ready_for_freeze" for c in cases):
        raise ValueError("twelve populated cases are required; draft slots cannot run")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        report = preflight(stage, cases, base=args.cases.parent, root=args.root, output=args.output)
        write(args.output / "preflight.json", report)
        print(json.dumps({"status": report["status"], "api_requests": 0, "gpu_jobs": 0,
            "prompts": len(report["prompts"]), "unique_prompt_count": report["unique_prompt_count"],
            "max_prompt_bytes": max(r["prompt_bytes"] for r in report["prompts"]),
            "production_source_files": len(report["production_source_snapshot"]["files"])}))
    else:
        if args.freeze is None:
            raise ValueError("--freeze required")
        freeze = read(args.freeze)
        if freeze["stage_sha256"] != digest(args.stage) or freeze["cases_sha256"] != digest(args.cases):
            raise ValueError("stage/case card changed after freeze")
        try:
            run(stage, cases, base=args.cases.parent, root=args.root, output=args.output, freeze=freeze)
        finally:
            # Even a transport interruption must leave the stage projection of
            # its durable journal, including conservatively reserved charges.
            write(args.output / "usage.json", usage_report(args.output, stage))


if __name__ == "__main__":
    main()
