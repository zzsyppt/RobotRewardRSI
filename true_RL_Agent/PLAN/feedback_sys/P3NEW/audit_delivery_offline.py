"""N4 development audit: genuine context/provider/HTTP serialization, no network.

Historical qualified public-dev archives are re-derived on CPU. Their immutable
original phases are not changed. The phase full artifact is external; generation
S4 fixtures are explicitly synthetic and are never attributed to that artifact.
This proves delivery of the declared engineering core, not independent semantic
recall, new-cohort confirmation, an author smoke test, or a live generation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from true_rl_agent.evaluation.binding import canonical_json_bytes, canonical_json_hash
from true_rl_agent.agent_core.authoring_response import SourceRequestResponse
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig, UrllibJsonTransport,
)
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError, build_provider_repair_feedback
from true_rl_agent.agent_core.source_request_service import resolve_source_request
from true_rl_agent.prompting.source_catalog import (
    has_inline_current_authoring_failed_source, prompt_source_catalog_logical_ids,
)
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt

ORIGINALS = ROOT / "PLAN/feedback_sys/P3R/results/context_dev_originals_r1"
SECTIONS = ("results", "process", "counterexample", "visual", "gaps", "parent_diff")
FIVE_BUNDLES = ("denominators_and_outcomes", "common_and_uncommon_process",
    "compatible_success_counterexample", "localized_main_process_relation", "missing_observation_and_sampling")
BUNDLES = {"results": [FIVE_BUNDLES[0]], "process": [FIVE_BUNDLES[1], FIVE_BUNDLES[3]],
    "counterexample": [FIVE_BUNDLES[2]], "gaps": [FIVE_BUNDLES[4]], "visual": [FIVE_BUNDLES[4]],
    "parent_diff": ["additional_bound_parent_difference_or_missing"]}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))
    return ref(path)


def ref(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return {"path": str(path), "sha256": digest(data), "size_bytes": len(data)}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def bound_reader():
    index = read(ORIGINALS / "source_index.json")
    by_hash = {item["sha256"]: ORIGINALS / name for name, item in index.items()}
    for expected, path in by_hash.items():
        assert digest(path.read_bytes()) == expected, path

    def load(reference, base):
        path = by_hash.get(reference["sha256"], Path(reference["path"]))
        if not path.is_absolute():
            path = base / path
        assert digest(path.read_bytes()) == reference["sha256"], path
        value = read(path)
        pointer = reference.get("pointer", "")
        for part in pointer.strip("/").split("/") if pointer else ():
            value = value[int(part)] if isinstance(value, list) else value[part.replace("~1", "/").replace("~0", "~")]
        return value

    return load, [ref(path) for path in by_hash.values()]


def raw_receipt(reference):
    """Record the immutable manifest and the chunks actually hash-checked below."""
    path = Path(reference["manifest_path"])
    value = read(path)
    assert digest(path.read_bytes()) == reference["manifest_sha256"]
    chunks = []
    for item in value["chunks"]:
        p = path.parent / item["file"]
        received = ref(p)
        assert received["sha256"] == item["sha256"]
        assert received["size_bytes"] == item["stored_bytes"]
        chunks.append({**received, "metadata": item["metadata"]})
    return {"manifest": ref(path), "identity": value["identity"], "chunks": chunks,
            "hash_verified": True, "literal_report_sentence_in_raw": False,
            "scope": "L0 source arrays; task-owned deterministic reduction creates sentence text"}


def measured_cache(task, out):
    """Extract only unchanged v2 fields from our already hash-bound full copy."""
    full = read(out / "behavior_full.json")
    phase = read(out / "public_dev_result.json")
    bound = phase["task_evidence"]["behavior_full"]
    assert ref(out / "behavior_full.json")["sha256"] == bound["sha256"]
    if task == "stair":
        keys = ("schema_version", "domain", "semantics", "episodes", "summary_hash")
        measured = {key: deepcopy(full[key]) for key in keys}
        measured["schema_version"] = "true_rl_agent.stair.behavior.full.v2"
    else:
        keys = ("schema_version", "rule_version", "binding", "actionability", "parent_diff", "status",
                "comparison_domain", "sources", "definitions", "coverage", "profile", "transcripts")
        measured = {key: deepcopy(full[key]) for key in keys}
        measured.update(schema_version="true_rl_agent.push_recovery.behavior.v2", rule_version="push_process_v2")
        from true_rl_agent.tasks.push_recovery.behavior import _actionability
        from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs
        measured["actionability"] = _actionability({"reward_input_fields": PushRecoveryV2RewardInputs.field_names(),
            "reward_input_schema_hash": measured["binding"]["reward_input_schema_hash"]})
        assert canonical_json_hash(measured) == full["source_derivation_hash"], "Push cached v2 reconstruction drift"
    receipt = write(out / "measured_v2.json", measured)
    write(out / "measured_v2_provenance.json", {"measured": receipt,
        "source_full_sha256_before_current_rederivation": bound["sha256"],
        "method": "unchanged original v2 fields copied from prior raw-verified development full; all current v3 templates are re-derived",
        "raw_sources": ref(out / "raw_sources.json")})
    return measured, receipt


def rebuild_phase(task, out, *, reuse_measured=False):
    original = ORIGINALS / task / "public_dev_result.json"
    phase = read(original)
    evidence = phase["task_evidence"]
    cached, cache_ref = measured_cache(task, out) if reuse_measured else (None, None)
    if task == "stair":
        from true_rl_agent.tasks.stair_climbing.process_report import build_stair_process_behavior, derive_stair_process_behavior, project_stair_process_report
        numeric = evidence["numeric_evidence"]
        numeric["manifest_path"] = str(ROOT / "PLAN/feedback_sys/P3R/results/stair_numeric_dev_r1/numeric_evidence/manifest.json")
        sources = [numeric]
        full = derive_stair_process_behavior(cached) if cached is not None else build_stair_process_behavior(numeric)
        public = project_stair_process_report(full)
    else:
        from true_rl_agent.tasks.push_recovery.process_report import build_push_process_behavior, project_push_process_behavior
        from true_rl_agent.tasks.push_recovery.reward_inputs import PushRecoveryV2RewardInputs
        old = evidence["behavior"]
        context = {**old["binding"], **old["comparison_domain"],
                   "reward_input_fields": list(PushRecoveryV2RewardInputs.field_names())}
        sources = []
        for i, summary in enumerate(evidence["seed_summaries"], 1):
            numeric = summary["numeric_evidence_ref"]
            numeric["manifest_path"] = str(ROOT / f"PLAN/feedback_sys/P3R/results/push_numeric_dev_r1/numeric/public_dev-{i:02}/manifest.json")
            sources.append(numeric)
        full = build_push_process_behavior(evidence["seed_summaries"], context=context, legacy_full=cached)
        public = project_push_process_behavior(full)
    assert full.get("status", "available") == "available", full
    full_ref = write(out / "behavior_full.json", full)
    evidence["behavior"] = public
    evidence["behavior_full"] = {"schema_version": "true_rl_agent.process_full_ref.v1",
        "file": "behavior_full.json", "sha256": full_ref["sha256"], "size_bytes": full_ref["size_bytes"]}
    phase["task_evidence_hash"] = canonical_json_hash(evidence)
    phase_ref = write(out / "public_dev_result.json", phase)
    write(out / "public_report.json", public)
    receipts = [raw_receipt(s) for s in sources]
    write(out / "raw_sources.json", receipts)
    write(out / "phase_provenance.json", {"original_phase": ref(original), "derived_phase": phase_ref,
        "external_full": full_ref, "derivation": "current task-owned process builder from historical L0",
        "new_rollouts": 0, "original_phase_unchanged": True,
        "measurement_cache": cache_ref, "L0_to_v2_recomputed_this_run": cached is None,
        "phase_has_measured_parent_candidate_difference": False,
        "generation_S4_scope": "later trusted projection; synthetic boundary tested separately"})
    return full, public, phase_ref, receipts


def sentences(task, full):
    return full["semantic"]["sentences"] if task == "stair" else full["sentences"]


def resolve_stair(value, reference):
    values = [("", value)]
    for part in reference.split("/"):
        next_values = []
        for pointer, node in values:
            if isinstance(node, list):
                if part == "*":
                    chosen = list(enumerate(node))
                elif part.isdigit():
                    chosen = [(int(part), node[int(part)])]
                else:
                    chosen = [(i, row) for i, row in enumerate(node) if isinstance(row, dict) and row.get("episode_id") == part]
            elif isinstance(node, dict) and part in node:
                chosen = [(part, node[part])]
            else:
                chosen = []
            next_values.extend((pointer + "/" + str(k), row) for k, row in chosen)
        values = next_values
    return [{"pointer": p, "sha256": canonical_json_hash(v)} for p, v in values]


def resolve_refs(task, full, references):
    output = []
    for name in references:
        if task == "stair" or name.startswith("observer_evidence/") or name == "observer_evidence":
            resolved = resolve_stair(full, name)
        else:
            item = full["evidence_catalog"].get(name)
            resolved = []
            if item is not None:
                value = full
                for part in item["pointer"].strip("/").split("/") if item["pointer"] else ():
                    value = value[int(part)] if isinstance(value, list) else value[part]
                resolved = [{**item, "sha256": canonical_json_hash(value)}]
        output.append({"evidence_ref": name, "resolved": resolved, "resolved_count": len(resolved)})
    return output


def string_fragment(text):
    return json.dumps(text, ensure_ascii=False)[1:-1].encode("utf-8")


def byte_evidence(data, fragment):
    start = data.find(fragment)
    return {"present": start >= 0, "offset": start, "length": len(fragment),
            "fragment_sha256": digest(fragment), "container_sha256": digest(data)}


class CapturedRequest(Exception):
    pass


class NoNetworkOpener:
    """Stop only after production UrllibJsonTransport made the real HTTP bytes."""
    def open(self, request, *, timeout):
        self.body = request.data
        self.url = request.full_url
        self.timeout = timeout
        raise CapturedRequest()


def capture_request(payload, out):
    capture = NoNetworkOpener()
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model="deepseek-v4-flash", base_url="https://api.deepseek.com", api_key="offline-placeholder",
        output_dir=out / "no_candidates", max_tokens=65536, reasoning_mode="adaptive", reasoning_effort="high"),
        transport=UrllibJsonTransport())
    with patch("urllib.request.build_opener", return_value=capture):
        try:
            provider.next_authoring_response(prompt_payload=payload)
        except CapturedRequest:
            pass
        else:
            raise AssertionError("fake transport did not capture the production request")
    assert capture.body
    request = json.loads(capture.body)
    user = request["messages"][-1]["content"]
    delivered = json.loads(user)
    assert delivered == payload
    return capture.body, user.encode("utf-8"), delivered


def make_frames(builder, comp, out):
    draft = builder.build_prompt([], frame_kind="candidate_draft").provider_payload
    catalog = tuple(sorted(prompt_source_catalog_logical_ids(draft)))
    requestable = [row["logical_id"] for row in draft["source_catalog"]["items"] if row.get("requestable")]
    assert requestable, "source followup must exercise a genuine allowlisted request"
    resolution = resolve_source_request(prompt_payload=draft, response=SourceRequestResponse(
        logical_ids=(requestable[0],), rationale="Offline delivery regression fixture.", source_catalog_logical_ids=catalog),
        source_request_count=0, source_request_bytes_returned=0)
    assert resolution.source_payloads, resolution.status
    followup = builder.build_prompt([], frame_kind="source_followup", candidate_repair_feedback=resolution.feedback,
        source_payloads=resolution.source_payloads).provider_payload
    fixture = out / "synthetic_failed_package"
    (fixture / "training_code").mkdir(parents=True, exist_ok=True)
    (fixture / "candidate_manifest.yaml").write_text("schema_version: true_rl_agent.candidate.v1\ncandidate_id: offline-repair\nentrypoint: training_code.components:build_training_components\n", encoding="utf-8")
    (fixture / "training_code/components.py").write_text(comp.reference_program_artifact.source_text, encoding="utf-8")
    (fixture / "rationale.md").write_text("Synthetic repair-source fixture only; never executed.", encoding="utf-8")
    write(fixture / "operation_diff.json", {})
    write(fixture / "diagnostics.json", {})
    failed = SimpleNamespace(path=str(fixture), candidate_id="offline-repair")
    static = builder.build_prompt([], frame_kind="static_repair", latest_failed_candidate=failed,
        candidate_repair_feedback=build_provider_repair_feedback(ProviderCandidateError(
            "verifier_error", "Synthetic static repair fixture; no candidate was verified or executed."))).provider_payload
    write(out / "static_fixture_payload.json", static)
    assert has_inline_current_authoring_failed_source(static), "static fixture did not enter current-source minimal repair"
    exhausted = builder.build_prompt([], frame_kind="static_repair", candidate_repair_feedback=build_provider_repair_feedback(
        ProviderCandidateError("reasoning_output_exhausted", "Synthetic reasoning length fixture.",
            details={"finish_reason": "length", "failure_kind": "empty_final_content_with_reasoning"}))).provider_payload
    return {"draft": draft, "source_followup": followup, "static_repair": static, "reasoning_exhaustion_repair": exhausted}


def behavior_in(payload, task):
    feedback = payload["public_feedback_summary"]
    return feedback["behavior"]["profile"] if task == "stair" else feedback["public_dev"]["behavior"]


def audit_frames(task, full, public, builder, comp, out, *, generation=None, base_phase_full=None):
    out.mkdir(parents=True, exist_ok=True)
    full_bytes = canonical_json_bytes(full)
    phase_bytes = canonical_json_bytes(base_phase_full) if base_phase_full is not None else full_bytes
    full_rows = {row["id"]: row for row in sentences(task, full)}
    public_bytes = canonical_json_bytes(public)
    core = [{"section": section, **row} for section in SECTIONS for row in public["report"][section]]
    mandatory_ids = {row["id"] for row in full_rows.values() if row.get("required", True)}
    if generation:
        mandatory_ids -= {row["id"] for row in full_rows.values() if row["section"] == "parent_diff"}
        mandatory_ids |= {row["id"] for row in public["report"]["parent_diff"]}
    assert mandatory_ids <= {row["id"] for row in core}, "required full sentence absent from bounded report"
    core_ref = write(out / "engineering_core_before_requests.json", {
        "scope": "all declared required sentence slots, plus every emitted sentence and actionability",
        "not_N5_independent_necessary_fact_denominator": True,
        "five_bundle_checklist": list(FIVE_BUNDLES), "sentences": core,
        "required_ids": sorted(mandatory_ids), "actionability": public["report"]["actionability"]})
    ledger, frames = [], []
    for name, payload in make_frames(builder, comp, out).items():
        body, user_bytes, delivered = capture_request(payload, out)
        delivered_behavior = behavior_in(delivered, task)
        assert delivered_behavior == public, (task, name, "public report changed")
        frame_dir = out / name
        frame_dir.mkdir(parents=True, exist_ok=True)
        (frame_dir / "request.http_body.json").write_bytes(body)
        (frame_dir / "user_payload.json").write_bytes(user_bytes)
        frame = delivered["authoring_context_frame"]
        budget = frame["budget_metadata"]
        validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=delivered)
        assert budget["actual_prompt_payload_bytes"] == len(user_bytes)
        forbidden_full_keys = ("evidence_catalog", "semantic", "term_cards", "candidate_lexicon", "behavior_full", "observer_evidence")
        assert not any(key in delivered_behavior for key in forbidden_full_keys)
        frame_receipt = {"name": name, "frame_kind": frame.get("kind", frame.get("frame_kind")),
            "request": ref(frame_dir / "request.http_body.json"), "user_payload": ref(frame_dir / "user_payload.json"),
            "public_behavior_bytes": len(public_bytes), "public_behavior_max_bytes": 8192 if task == "stair" else 12288,
            "public_behavior_headroom_bytes": (8192 if task == "stair" else 12288) - len(public_bytes),
            "user_payload_headroom_bytes": budget["hard_limits"]["max_prompt_payload_bytes"] - len(user_bytes),
            "budget": budget, "actual_http_serialization": "production UrllibJsonTransport._request_json",
            "transport_boundary": "fake opener; no network", "network_calls": 0, "provider_usage": None,
            "request_bytes_are_not_provider_token_usage": True}
        frames.append(frame_receipt)
        for row in core:
            post_phase = generation is not None and row["section"] == "parent_diff"
            synthetic_visual = base_phase_full is not None and row["section"] == "visual"
            original = full_rows.get(row["id"])
            references = [] if post_phase else resolve_refs(task, full, original.get("evidence_refs", []) if original else [])
            fragment = string_fragment(row["text"])
            escaped = string_fragment(fragment.decode("utf-8"))
            fact = {"id": row["id"], "section": row["section"], "bundle": BUNDLES[row["section"]],
                "frame": name, "text": row["text"], "required": row["id"] in mandatory_ids,
                "origin": "synthetic_post_phase_generation_S4" if post_phase else "synthetic_visual_fixture" if synthetic_visual else "phase_derivation",
                "raw": {"status": "not_applicable_synthetic" if post_phase or synthetic_visual else "hash_bound_task_derivation_or_contract",
                    "archive_available": not (post_phase or synthetic_visual),
                    "manifest_sha256s": [] if post_phase or synthetic_visual else ([full["domain"]["archive_hash"]] if task == "stair"
                        else [source["manifest_sha256"] for source in full["sources"]]),
                    "literal_sentence_expected": False, "source_receipts": "../raw_sources.json",
                    "full_evidence_refs": references,
                    "semantic_support_independently_verified_here": False},
                "phase_full": byte_evidence(phase_bytes, fragment),
                "full_derivation": {**byte_evidence(full_bytes, fragment), "role": "synthetic_visual_full_fixture" if base_phase_full is not None else "phase_full"},
                "bounded_public": byte_evidence(public_bytes, fragment),
                "user_payload": byte_evidence(user_bytes, fragment), "actual_request": byte_evidence(body, escaped)}
            if post_phase:
                fact["generation_derivation"] = byte_evidence(canonical_json_bytes(generation), fragment)
                assert not fact["phase_full"]["present"], "synthetic S4 unexpectedly in immutable phase full"
                assert fact["generation_derivation"]["present"]
            else:
                assert original and original["text"] == row["text"], (task, row["id"], "full/public text drift")
                assert fact["full_derivation"]["present"]
                assert fact["phase_full"]["present"] is not synthetic_visual
                assert all(item["resolved_count"] for item in references), (task, row["id"], references)
            assert all(fact[layer]["present"] for layer in ("bounded_public", "user_payload", "actual_request")), (task, name, row["id"])
            ledger.append(fact)
        action = public["report"]["actionability"]
        actual_action = delivered_behavior["report"]["actionability"]
        full_action = full["semantic"]["actionability"] if task == "stair" else full["actionability"]
        assert action == actual_action == full_action
        frame_receipt["actionability_exact_object_equal"] = True
        action_bytes = canonical_json_bytes(action)
        action_wire_bytes = json.dumps(actual_action, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        action_fact = {"id": "actionability", "section": "actionability", "frame": name,
            "origin": "task_capability_contract", "raw": {"status": "not_a_measured_claim"},
            "phase_full": byte_evidence(phase_bytes, action_bytes),
            "bounded_public": byte_evidence(public_bytes, action_bytes),
            "user_payload": byte_evidence(user_bytes, action_wire_bytes),
            "actual_request": byte_evidence(body, string_fragment(action_wire_bytes.decode("utf-8")))}
        assert all(action_fact[layer]["present"] for layer in ("phase_full", "bounded_public", "user_payload", "actual_request"))
        ledger.append(action_fact)
    write(out / "sentence_delivery_ledger.json", ledger)
    result = {"status": "PASS_OFFLINE_ENGINEERING", "core_catalog": core_ref, "frames": frames,
        "sentence_checks": len(core) * len(frames), "actionability_checks": len(frames),
        "sentence_count_per_frame": len(core), "lost_required_sentences": 0,
        "seven_sections_exact": True, "synthetic_only": generation is not None,
        "synthetic_visual_fixture": base_phase_full is not None,
        "phase_full_external": True, "network_calls": 0, "author_episodes": 0,
        "limits": ["Historical development cohort, no new numerical collection.",
                   "Source/static/reasoning branches are forced fixtures, not model choices.",
                   "No independent semantic support or recall score is established.",
                   "Strict6 still requires its separately authorized author read-smoke and frozen N5 facts."]}
    write(out / "result.json", result)
    return result


def synthetic_s4(task, public):
    candidate, parent = deepcopy(public), deepcopy(public)
    checkpoint = "sha256:" + "1" * 64
    if task == "stair":
        from true_rl_agent.tasks.stair_climbing.process_report import compare_stair_process_reports, attach_stair_process_difference, PATTERN_TERMS
        parent["domain"]["checkpoint_hash"] = checkpoint
        row = next(v for v in parent["comparison_summary"]["strata"].values() if v["n"] > 0)
        term = PATTERN_TERMS[0]
        row[term] = row[term] - 1 if row[term] > 0 else 1
        diff = compare_stair_process_reports(parent, candidate, expected_parent_checkpoint_hash=checkpoint)
        assert diff["status"] == "available"
        projected = attach_stair_process_difference(candidate, diff)
    else:
        from true_rl_agent.tasks.push_recovery.process_report import compare_push_process_behavior, project_push_process_behavior
        parent["binding"]["checkpoint_hash"] = checkpoint
        counts = parent["comparison_basis"]["terminal_counts"]
        counts["success_recovered"] -= 1
        other = next(key for key in counts if key != "success_recovered")
        counts[other] += 1
        diff = compare_push_process_behavior(candidate, parent, expected_parent_checkpoint_hash=checkpoint)
        assert diff["status"] == "available"
        candidate["parent_diff"] = diff
        projected = project_push_process_behavior(candidate)
    return projected, {"synthetic_only": True, "not_measured_parent_candidate": True,
        "phase_full_mutated": False, "parent_fixture": parent, "candidate_fixture": candidate,
        "difference": diff, "projected_parent_section": projected["report"]["parent_diff"]}


def install_report(builder, feedback, task, report, generation):
    value = deepcopy(feedback)
    owner = value if task == "stair" else value["public_dev"]
    if task == "stair":
        owner["behavior"]["profile"] = report
        owner["behavior"]["diff"] = generation["difference"]
    else:
        owner["behavior"] = report
    builder.task_prompt_adapter.feedback = value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "PLAN/feedback_sys/P3NEW/results/delivery_offline_r1")
    parser.add_argument("--resume-derived", action="store_true", help="Reuse already sealed development phase while debugging only; final audit omits this flag.")
    parser.add_argument("--reuse-measured", action="store_true", help="Re-derive current v3 templates from unchanged v2 fields already verified against these archives.")
    args = parser.parse_args()
    output = args.output.resolve()
    assert output.is_relative_to(ROOT / "PLAN/feedback_sys/P3NEW"), "audit writes are limited to P3NEW"
    confirmation = module(ROOT / "PLAN/feedback_sys/P3R/confirmation_context.py", "n4_confirmation_context")
    visual_fixture = module(Path(__file__).with_name("audit_visual_projection_offline.py"), "n4_visual_fixture")
    exp = confirmation._p3(ROOT)
    exp.bound_read, originals = bound_reader()
    source_paths = [ROOT / "src/true_rl_agent" / item for item in (
        "tasks/stair_climbing/process_report.py", "tasks/stair_climbing/behavior.py",
        "tasks/push_recovery/process_report.py", "tasks/push_recovery/behavior.py",
        "evaluation/process_report.py", "memory/context_builder.py", "prompting/authoring_frame.py",
        "prompting/payload.py", "prompt_serialization.py", "agent_core/openai_compatible_provider.py",
        "evaluation/visual_observer.py", "tasks/stair_climbing/visual_observation.py",
        "tasks/push_recovery/visual_observation.py", "tasks/stair_climbing/prompt_adapter.py",
        "tasks/push_recovery/prompt_adapter.py", "tasks/stair_climbing/phased_adapter.py",
        "tasks/push_recovery/phased_adapter.py", "agent_core/source_request_service.py",
        "prompting/source_catalog.py")]
    source_paths.extend([Path(__file__), Path(__file__).with_name("audit_visual_projection_offline.py"),
        ROOT / "PLAN/feedback_sys/P3R/confirmation_context.py", ROOT / "PLAN/feedback_sys/P3/author_experiment.py"])
    source_receipts = [ref(path) for path in source_paths]
    results = {}
    for task in ("stair", "push"):
        out = output / task
        if args.resume_derived and (out / "public_dev_result.json").exists():
            phase = read(out / "public_dev_result.json")
            full = read(out / "behavior_full.json")
            bound = phase["task_evidence"]["behavior_full"]
            assert ref(out / "behavior_full.json")["sha256"] == bound["sha256"]
            assert ref(out / "behavior_full.json")["size_bytes"] == bound["size_bytes"]
            public, phase_ref, sources = phase["task_evidence"]["behavior"], ref(out / "public_dev_result.json"), read(out / "raw_sources.json")
        else:
            print(task, "re-deriving current templates from verified v2" if args.reuse_measured
                  else "rebuilding historical L0 into current phase full", flush=True)
            full, public, phase_ref, sources = rebuild_phase(task, out, reuse_measured=args.reuse_measured)
        spec = read(ROOT / f"PLAN/feedback_sys/P3/author_case_inputs_r5/author_context_{task}.json")
        case = {"task": task, "checkpoint_hash": spec["checkpoint_hash"],
            "common_focus": "Read denominators, common and uncommon processes, compatible success counterexamples, localized temporal relations, missing observations and sampling scope. Process observations do not authorize changes to multiple mechanisms."}
        kwargs = {}
        if task == "push":
            kwargs = {"diagnostic_eval_base_seed": spec["composition_values"]["eval_base_seed"],
                "source_manifest_ref": ref(ORIGINALS / "push/source_manifest.json"),
                "training_summary_ref": ref(ORIGINALS / "push/training_summary.json")}
        comp, builder, feedback = confirmation.build_context(case, "T", spec,
            diagnostic_phase_ref=phase_ref, diagnostic_plan_ref=ref(ORIGINALS / task / "public_dev_job.json"),
            allowed_seed_cohort_hash=spec["validated_evaluation_binding"]["seed_cohort_hash"],
            root=ROOT, base=ROOT, output=out / "context", development_fixture=True, exp=exp, **kwargs)
        owner = feedback if task == "stair" else feedback["public_dev"]
        write(out / "context_feedback.json", feedback)
        visible = owner["behavior"]["profile"] if task == "stair" else owner["behavior"]
        assert visible == public, "task adapter changed phase public report before frame construction"
        real = audit_frames(task, full, public, builder, comp, out / "phase_report")
        synthetic, generation = synthetic_s4(task, public)
        write(out / "synthetic_generation_S4.json", generation)
        install_report(builder, feedback, task, synthetic, generation)
        regression = audit_frames(task, full, synthetic, builder, comp, out / "synthetic_generation_S4", generation=generation)
        attach = visual_fixture.attach_stair_visual_observations if task == "stair" else visual_fixture.attach_push_visual_observations
        project = visual_fixture.project_stair_process_report if task == "stair" else visual_fixture.project_push_process_behavior
        original_hash = canonical_json_hash(full)
        augmented = attach(full, visual_fixture.synthetic_audit(task, full),
                           observer_version=visual_fixture.SCHEMA_VERSION + ".synthetic_capacity")
        assert canonical_json_hash(full) == original_hash
        numerical_key = "episodes" if task == "stair" else "transcripts"
        assert augmented[numerical_key] == full[numerical_key]
        with_visual, visual_generation = synthetic_s4(task, project(augmented))
        assert with_visual["report"]["counterexample"] == public["report"]["counterexample"]
        assert with_visual["report"]["gaps"] == public["report"]["gaps"]
        assert with_visual["report"]["visual"][0]["text"].count("形") == visual_fixture.MAX_OBSERVATION_CHARS
        assert with_visual["report"]["visual"][0]["text"].count("限") == visual_fixture.MAX_UNCERTAINTY_CHARS
        write(out / "synthetic_visual_full.json", augmented)
        write(out / "synthetic_visual_generation_S4.json", visual_generation)
        install_report(builder, feedback, task, with_visual, visual_generation)
        visual_regression = audit_frames(task, augmented, with_visual, builder, comp,
            out / "synthetic_visual_generation_S4", generation=visual_generation, base_phase_full=full)
        visual_regression["pressure_fixture"] = {"raw_observations": 3, "displayed_observations": 1,
            "observer_schema_version": visual_fixture.SCHEMA_VERSION,
            "displayed_text_characters": visual_fixture.MAX_OBSERVATION_CHARS,
            "uncertainty_characters_per_observation": visual_fixture.MAX_UNCERTAINTY_CHARS,
            "text": "maximum-length synthetic Chinese UTF-8 prose", "counterexample_and_gaps_unchanged": True,
            "numeric_episode_evidence_unchanged": True, "real_images_observed": 0}
        results[task] = {"phase_report": real, "synthetic_generation_S4": regression,
            "synthetic_visual_generation_S4": visual_regression,
            "archive_count": len(sources), "raw_chunks": sum(len(s["chunks"]) for s in sources)}
        print(task, "PASS", real["sentence_checks"] + regression["sentence_checks"] + visual_regression["sentence_checks"], "sentence checks", flush=True)
    assert all(ref(item["path"])["sha256"] == item["sha256"] for item in originals)
    source_stable = all(ref(item["path"])["sha256"] == item["sha256"] for item in source_receipts)
    result = {"schema_version": "p3new.offline_delivery_audit.v1", "status": "PASS_OFFLINE_ENGINEERING",
        "network_calls": 0, "author_episodes": 0, "new_rollouts": 0, "tasks": results,
        "resumed_previous_derivation": args.resume_derived,
        "rederived_current_templates_from_verified_v2": args.reuse_measured,
        "original_artifacts_unchanged": originals, "harness": ref(Path(__file__)),
        "production_sources": source_receipts, "production_sources_unchanged_during_run": source_stable,
        "strict6_complete": False, "N5_semantic_confirmation": "not_started"}
    if not source_stable:
        result["status"] = "RERUN_REQUIRED_SOURCE_CHANGED_DURING_AUDIT"
    write(output / "result.json", result)
    print(output / "result.json", flush=True)


if __name__ == "__main__":
    main()
