"""Send one frozen native-development author episode, with no retry or followup.

CLI: --stage NEW_DIR/stage.released.json --output NEW_DIR
     --run-case {stair-native-dev,push-native-dev} [--check-only]
The release is the unchanged prepare_native_author_smoke card plus status
FROZEN_API_RELEASED, pricing_status VERIFIED_FOR_AUTHOR_MODEL, current verified
budget.prices_usd_per_million, and execution_script={path,sha256,size_bytes}.
Freeze this driver in frozen_inputs too. --check-only accepts the draft card,
reads the original shared journal, reconstructs its request without credentials/
network, and creates CPU context files only. Run with PYTHONPATH pointing at frozen
production src before PLAN imports. Only 4 -> 6/gen003 and 6 -> 8/gen004 are allowed;
gen004 freezes budget.historical_usage_ref for the six actual prior calls.
The original shared authoring journal remains authoritative; new usage/terminals
are written here, never over historical text usage.json. Development smoke only.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from unittest.mock import patch

# Load the selected package before PLAN helpers prepend their historical src.
sys.path.append(str(Path(__file__).resolve().parents[3] / "src"))
import true_rl_agent
import prepare_native_author_final_slots as P
# multiprocessing spawn inherits sys.path, not the already imported package.
# Restore the selected source after all historical PLAN helper imports.
SOURCE_ROOT = Path(true_rl_agent.__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "src"))
from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig, UrllibJsonTransport,
    normalize_provider_usage,
)
from true_rl_agent.agent_core.process_lock import acquire_process_file_lock
from true_rl_agent.candidates.authoring_policy import select_authoring_anchor_source
from true_rl_agent.runner.source_snapshot import project_source_snapshot

D, R, C, require = P.D, P.R, P.C, P.N.require
SHARED = Path("/data/yuyangliu/AGENT/feedback_p3new_20260929/text_author_smoke_r1/PLAN/feedback_sys/P3NEW/results/text_author_smoke_r1")
PYTHON = "/data/yuyangliu/envs/isaaclab6_validation/bin/python"
GENERATION_PREFIX = "p3new-text-smoke-20260929-generation-"
CASE_IDS = ("stair-native-dev", "push-native-dev")
CAPS = {"attempts_per_lane_per_generation": 5, "input_units_per_attempt_reserve": 40000,
        "output_units_per_attempt_cap": 65536}


class RecordedTransport:
    """Observe production urllib's actual bytes and retain its no-retry policy.

    No headers/credentials are persisted. The response is the exact bounded read,
    including invalid JSON; production remains responsible for parsing/errors.
    An injected opener is exclusively an offline test seam.
    """
    def __init__(self, directory, expected_body, max_response_bytes, opener_factory=None):
        self.directory, self.expected_body = directory, expected_body
        self.delegate = UrllibJsonTransport(max_response_bytes=max_response_bytes)
        self.max_response_bytes = max_response_bytes
        self.transport_policy_id = self.delegate.transport_policy_id
        self.opener_factory = opener_factory
        self.calls = 0

    def post_json(self, **kwargs):
        import urllib.request
        require(self.calls == 0, "second transport call forbidden")
        require(json.dumps(kwargs["payload"], ensure_ascii=False).encode("utf-8") == self.expected_body,
                "actual provider payload differs from frozen first request")
        self.calls += 1
        directory, expected = self.directory, self.expected_body
        directory.mkdir(parents=True, exist_ok=False)
        factory = self.opener_factory or urllib.request.build_opener

        class Response:
            def __init__(self, response):
                self.response = response

            def __enter__(self):
                self.response.__enter__()
                return self

            def __exit__(self, *args):
                return self.response.__exit__(*args)

            def read(self, size):
                raw = self.response.read(size)
                with (directory / "response.http_body.bin").open("xb") as stream:
                    stream.write(raw)
                return raw

        class Opener:
            def __init__(self, opener):
                self.opener = opener

            def open(self, request, *, timeout):
                require(request.data == expected, "production HTTP bytes differ from prepared input")
                with (directory / "request.http_body.json").open("xb") as stream:
                    stream.write(request.data)
                user = json.loads(request.data)["messages"][-1]["content"].encode("utf-8")
                (directory / "user_payload.json").write_bytes(user)
                return Response(self.opener.open(request, timeout=timeout))

        try:
            with patch("urllib.request.build_opener", side_effect=lambda *a, **k: Opener(factory(*a, **k))):
                raw = self.delegate.post_json(**kwargs)
            D.write(directory / "raw_response.json", raw)
            D.write(directory / "response_receipt.json", {
                "usage": raw.get("usage"), "response_model": raw.get("model"),
                "system_fingerprint": raw.get("system_fingerprint"),
                "request": D.ref(directory / "request.http_body.json"),
                "raw_response": D.ref(directory / "response.http_body.bin"),
            })
            return raw
        except BaseException as exc:
            D.write(directory / "transport_terminal.json", {
                "status": "failed_or_interrupted", "exception_class": type(exc).__name__,
                "usage": None, "retry_allowed": False,
            })
            raise


def validate(stage, stage_path, output, *, released):
    require(stage["schema_version"] == "p3new.native_author_smoke.v1"
            and stage["scope"] == "old_development_not_n5", "wrong experiment scope")
    require(output == stage_path.parent and output.is_relative_to(P.BASE.resolve())
            and output != P.HISTORY.resolve(), "use the new prepared material directory")
    if released:
        require(stage["status"] == "FROZEN_API_RELEASED"
                and stage["pricing_status"] == "VERIFIED_FOR_AUTHOR_MODEL", "reviewed release required")
        require(P.checked(stage["execution_script"]) == Path(__file__).resolve(), "released driver differs")
        require(stage["execution_script"] in stage["frozen_inputs"], "driver must be frozen")
    else:
        require(stage["status"] in {"PREPARED_NOT_RELEASED", "FROZEN_API_RELEASED"}, "wrong card state")
    for reference in stage["frozen_inputs"]:
        P.checked(reference)
    require(SOURCE_ROOT == Path(stage["production_source_root"]).resolve(), "PYTHONPATH source root differs")
    require(project_source_snapshot(SOURCE_ROOT) == D.read(P.checked(stage["production_source_snapshot"])),
            "frozen production source drift")
    require(all(not name.startswith("true_rl_agent") or not getattr(module, "__file__", None)
                or Path(module.__file__).resolve().is_relative_to(SOURCE_ROOT / "src")
                for name, module in tuple(sys.modules.items())), "mixed production package roots")
    require(Path(stage["material_repository_root"]).resolve() == P.ROOT.resolve(), "material root differs")
    budget, provider = stage["budget"], stage["provider"]
    allocation = (budget["historical_actual_calls"], stage["episode_budget"]["total_if_run"],
                  budget["max_generations"])
    ceilings = {(4, 6, 3): 0.127089852, (6, 8, 4): 0.200609736}
    require(all(type(value) is int for value in allocation) and allocation in ceilings,
            "only the frozen 4 -> 6/gen003 or 6 -> 8/gen004 allocation is allowed")
    require(budget["stage_usd_cap"] == 4.0
            and abs(budget["historical_usd_ceiling"] - ceilings[allocation]) < 1e-12
            and budget["new_actual_calls_cap"] == 2 and budget["max_attempts_per_new_episode"] == 1
            and budget["no_retry_or_replacement_episode"] is True
            and budget["input_reserve"] == 40000 and budget["max_output_tokens"] == 65536
            and budget["production_provider_caps"] == CAPS, "frozen budget differs")
    if budget["max_generations"] == 4:
        require(budget["historical_usage_ref"] in stage["frozen_inputs"], "six-call usage must be frozen")
        historical_path = P.checked(budget["historical_usage_ref"])
        require(historical_path not in [output / "usage.json"]
                + [output / (case_id + ".usage.json") for case_id in CASE_IDS],
                "historical usage cannot be a mutable current-stage accounting output")
        require(not {"previous_release_ref", "prior_case_recovery"}.intersection(stage),
                "gen004 cannot use gen003 prior-case recovery")
    require(provider["reasoning_effort"] == "high" and provider["reasoning_mode"] == "adaptive"
            and provider["max_output_tokens"] == 65536 and provider["max_input_tokens"] == 40000
            and provider["empty_final_reasoning_retry"] is False, "provider settings differ")
    expected = [("stair-native-dev", "stair", "lane-a"), ("push-native-dev", "push", "lane-b")]
    require([(r["case_id"], r["task"], r["lane_id"]) for r in stage["episodes"]] == expected,
            "only the two ordered native cases are allowed")
    generation = GENERATION_PREFIX + f"{budget['max_generations']:03d}"
    require(all(r["generation_id"] == generation and r["round_id"] == generation + "-" + r["lane_id"]
                for r in stage["episodes"]), "frozen generation slots differ")


def construct(stage, row, output, shared):
    material = P.checked(row["material"]).parent
    comp, builder, _ = P.context_for(row["task"], material, D.read(material / "provenance.json"),
                                     output / "execution_contexts" / row["case_id"])
    body = P.checked(row["initial_http_body"]).read_bytes()
    p = stage["provider"]
    transport = RecordedTransport(output / "transport" / row["case_id"], body, p["max_response_bytes"])
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model=p["requested_model"], base_url=p["base_url"], endpoint=p["endpoint"],
        api_key_env=p["api_key_env"], env_file=Path(p["env_file"]),
        output_dir=output / "provider_candidates" / row["case_id"], max_tokens=65536,
        reasoning_mode="adaptive", reasoning_effort="high", temperature=p["temperature"],
        timeout_s=p["timeout_s"], max_response_bytes=p["max_response_bytes"], empty_final_reasoning_retry=False),
        transport=transport)
    first = R.first_prompt(comp, builder)
    require(json.dumps(provider._chat_completion_request(first), ensure_ascii=False).encode("utf-8") == body,
            "reconstructed first HTTP body differs")
    require(json.loads(body)["messages"][-1]["content"].encode("utf-8")
            == P.checked(row["initial_payload"]).read_bytes(), "frozen user payload differs")
    _, source_hash, role = select_authoring_anchor_source(first)
    require(role == "current_best" and source_hash == row["qualified_parent_source_sha256"], "parent source differs")
    guarded = CostGuardedProvider(delegate=provider, output_dir=shared, generation_id=row["generation_id"],
        lane_id=row["lane_id"], provider_caps=CAPS, max_generations=stage["budget"]["max_generations"])
    loop = CandidateAuthoringLoop(context_builder=builder, provider=guarded,
        candidate_admission_policy=comp.candidate_admission_policy,
        round_dir=shared / "authoring" / row["generation_id"] / row["lane_id"], max_attempts=1,
        max_source_requests_per_round=0, component_smoke_adapter=comp.task_plugin.component_smoke_adapter(),
        authoring_capability_readiness=comp.authoring_capability_readiness.to_prompt_mapping())
    return loop, transport


def accounting(stage, shared):
    card = deepcopy(stage)
    card["budget"]["stage_usd"] = 4.0
    require(card["budget"]["prices_usd_per_million"] == card["budget"]["historical_price_basis"],
            "price changes need separately preserved historical accounting")
    return R.usage_report(shared, card, C._p3(P.ROOT))


def check_usage(usage, stage, *, expected_count):
    historical_count = stage["budget"]["historical_actual_calls"]
    require(not usage["violations"] and usage["attempts"] == expected_count
            and historical_count <= expected_count <= stage["episode_budget"]["total_if_run"],
            "journal count/budget violation")
    rows = usage["attempts_detail"]
    require(len(rows) == expected_count, "journal detail count differs")
    require(all(r.get("usage_known") is True and r["status"] == "completed" for r in rows),
            "unknown/interrupted prior usage; retain evidence and stop")
    p = stage["provider"]
    require(usage["response_models"] == [p["verified_response_model"]]
            and all(r["system_fingerprint"] == p["verified_system_fingerprint"] for r in rows),
            "response model/fingerprint drift")
    archived = D.read(P.checked(stage["budget"]["historical_usage_ref"]) if historical_count == 6
                      else P.HISTORY / "usage.json")
    historical = {(r["round_id"], r["sequence"]): r for r in archived["attempts_detail"]}
    current = {(r["round_id"], r["sequence"]): r for r in rows}
    require(archived["attempts"] == len(archived["attempts_detail"]) == len(historical) == historical_count
            and len(current) == expected_count
            and abs(archived["usd_peak_actual_or_reserved"] - stage["budget"]["historical_usd_ceiling"]) < 1e-12
            and all(current.get(key) == value for key, value in historical.items()),
            "historical usage changed")
    new = [r for key, r in current.items() if key not in historical]
    allowed = [r["round_id"] for r in stage["episodes"][:expected_count - historical_count]]
    require(sorted(r["round_id"] for r in new) == sorted(allowed)
            and all(r["sequence"] == 1 for r in new), "unexpected extra/replacement attempt")


def validate_prior_recovery(stage, stage_path, output, shared, case_id):
    """Only continue the untouched Push slot after the frozen Stair interruption."""
    recovery_keys = {"previous_release_ref", "prior_case_recovery"}
    if not recovery_keys.intersection(stage):
        return False
    require(recovery_keys.issubset(stage) and case_id == "push-native-dev",
            "recovery release may only continue the original Push case")
    previous_ref = stage["previous_release_ref"]
    previous_path = P.checked(previous_ref)
    require(previous_path == output / "stage.released.json" and previous_path != stage_path,
            "recovery must retain the original release")
    previous = D.read(previous_path)
    require(not recovery_keys.intersection(previous), "recovery chaining is not supported")
    mutable = recovery_keys | {"execution_script", "frozen_inputs"}
    require({k: v for k, v in stage.items() if k not in mutable}
            == {k: v for k, v in previous.items() if k not in mutable},
            "recovery changed episodes/provider/budget/source or original release content")
    require(stage["frozen_inputs"][:len(previous["frozen_inputs"])] == previous["frozen_inputs"],
            "original frozen inputs must remain unchanged")
    recovery = D.read(P.checked(stage["prior_case_recovery"]))
    require(recovery["status"] == "RECOVERED_TERMINAL_VERIFIER_ERROR_NO_RESUBMISSION"
            and recovery["stage_ref"] == previous_ref and recovery["api_calls"] == 0
            and recovery["credential_contents_read"] is False and recovery["gpu_jobs"] == 0
            and recovery["original_delivery_status"] == "FAILED"
            and recovery["continuation_stop_reason"] == "static_failed"
            and recovery["failure_category"] == "verifier_error",
            "prior recovery is not a closed, zero-call verifier interruption")
    prior = stage["episodes"][0]
    terminal_path = P.checked(recovery["original_terminal_ref"])
    require(terminal_path == output / "results" / (prior["round_id"] + ".json"),
            "recovery names another prior terminal")
    terminal = D.read(terminal_path)
    require(terminal["stage_ref"] == previous_ref and terminal["delivery_status"] == "FAILED"
            and terminal["exception_class"] == "IsolatedWorkerProtocolError"
            and terminal["transport_invocations"] == 1, "original failed call differs")
    receipt_path = P.checked(recovery["transport_receipt_ref"])
    require(receipt_path == output / "transport" / prior["case_id"] / "response_receipt.json",
            "recovery names another provider receipt")
    receipt = D.read(receipt_path)
    raw = D.read(P.checked(receipt["raw_response"]))
    require(P.checked(receipt["request"]).read_bytes() == P.checked(prior["initial_http_body"]).read_bytes()
            and raw["usage"] == receipt["usage"] == recovery["raw_provider_usage"]
            and raw["model"] == receipt["response_model"], "received provider evidence differs")
    attempt_path = P.checked(recovery["appended_attempt_ref"])
    prior_dir = shared / "authoring" / prior["generation_id"] / prior["lane_id"]
    require(attempt_path == prior_dir / "authoring_attempts/attempt_000001/attempt_result.json"
            and list(prior_dir.rglob("attempt_result.json")) == [attempt_path],
            "prior recovery must close exactly the original attempt")
    attempt = D.read(attempt_path)
    provider, continuation = attempt["provider_attempt"], attempt["continuation_state"]
    require(attempt["round_id"] == prior["round_id"] and attempt["sequence"] == 1
            and provider["attempt"] == 1 and provider["provider_status"] == "pass"
            and provider["verifier_status"] == "fail"
            and provider["provider_usage"] == recovery["normalized_usage"] == normalize_provider_usage(raw["usage"])
            and provider["provider_response_receipt"] == recovery["provider_response_receipt"]
            and continuation["disposition"] == "terminal" and continuation["stop_reason"] == "static_failed"
            and continuation["static_failure"]["failure_category"] == "verifier_error",
            "prior journal is not settled with the true usage and verifier failure")
    for reference in (previous_ref, stage["prior_case_recovery"], stage["execution_script"],
                      recovery["original_terminal_ref"], recovery["transport_receipt_ref"],
                      recovery["appended_attempt_ref"]):
        require(reference in stage["frozen_inputs"], "recovery evidence must be frozen")
    return True


def run(stage_path, output, case_id, *, check_only=False):
    stage = D.read(stage_path)
    validate(stage, stage_path, output, released=not check_only)
    row = next(r for r in stage["episodes"] if r["case_id"] == case_id)
    shared = Path(stage["budget"]["shared_journal_directory"]).resolve()
    historical_count = stage["budget"]["historical_actual_calls"]
    require(shared == SHARED == P.HISTORY.resolve(), "original remote shared journal required")
    recovered_prior = validate_prior_recovery(stage, stage_path, output, shared, case_id)
    if check_only:
        require(not (shared / "authoring" / row["generation_id"] / row["lane_id"]).exists()
                and not (output / "transport" / case_id).exists(), "target slot already occupied")
        usage = accounting(stage, shared)
        require(usage["attempts"] <= historical_count + stage["episodes"].index(row),
                "journal advanced beyond the unoccupied target slot")
        check_usage(usage, stage, expected_count=historical_count + 1 if recovered_prior else usage["attempts"])
        loop, _ = construct(stage, row, output, shared)
        return {"case_id": case_id, "status": "PASS_OFFLINE_REQUEST_RECONSTRUCTION", "api_calls": 0,
                "credential_contents_read": False, "max_attempts": loop.max_attempts,
                "max_source_requests_per_round": loop.max_source_requests_per_round,
                "production_source_root": str(SOURCE_ROOT), "http_body": row["initial_http_body"]}
    require(sys.platform == "linux" and str(Path(sys.executable).absolute()) == PYTHON,
            "run only on the registered execution host/interpreter")
    require(stage["provider"]["env_file"] == "/data/yuyangliu/secrets/deepseek.env", "registered credential path differs")
    terminal_path = output / "results" / (row["round_id"] + ".json")
    with acquire_process_file_lock(shared / ".text_author_smoke.lock", blocking=False):
        if terminal_path.exists():
            terminal = D.read(terminal_path)
            require(terminal["stage_ref"] == D.ref(stage_path), "terminal belongs to a different release")
            require(terminal.get("delivery_status") == "PASS",
                    "saved delivery failed or lacks postcall validation; no retry permitted")
            return terminal
        index = stage["episodes"].index(row)
        for prior in stage["episodes"][:index]:
            prior_path = output / "results" / (prior["round_id"] + ".json")
            require(prior_path.exists(), "preserve frozen Stair then Push order")
            prior_terminal = D.read(prior_path)
            require(prior_terminal["stage_ref"] == stage.get("previous_release_ref", D.ref(stage_path))
                    and (prior_terminal.get("delivery_status") == "PASS" or recovered_prior),
                    "prior delivery failed or lacks postcall validation; stop later episodes")
        round_dir = shared / "authoring" / row["generation_id"] / row["lane_id"]
        require(not round_dir.exists() and not (output / "transport" / case_id).exists(),
                "occupied/interrupted slot cannot be retried or replaced")
        before = accounting(stage, shared)
        check_usage(before, stage, expected_count=historical_count + index)
        prices = stage["budget"]["prices_usd_per_million"]
        worst = (40000 * prices["cache_miss"] + 65536 * prices["output"]) / 1e6
        require(before["usd_peak_actual_or_reserved"] + worst <= 4.0, "remaining dollar cap cannot reserve next call")
        loop, transport = construct(stage, row, output, shared)
        # This is the first credential read; every release/input/source check is complete.
        loop.provider.delegate.validate_credential()
        terminal = {**row, "stage_ref": D.ref(stage_path), "driver_ref": D.ref(Path(__file__)),
            "scope": "old_development_not_n5", "train_status": "not_run", "review_status": "PENDING",
            "AT_value_gate": "NOT_APPLICABLE", "semantic_acceptance": "NOT_ESTABLISHED",
            "candidate_verified_is_not_RL_effect": True, "retry_or_replacement_allowed": False}
        failure = None
        try:
            result = loop.run_round(round_id=row["round_id"], records=[])
            terminal.update(stop_reason=result.stop_reason,
                candidate_private_path=result.candidate.path if result.candidate else None,
                candidate=result.candidate.to_public_dict() if result.candidate else None,
                authoring_checks=result.attempt_records,
                prompt_payload_ref=result.prompt_payload_ref.to_public_dict(),
                authoring_evidence_ref=result.authoring_evidence_ref.to_private_dict(),
                static_failure=result.static_failure, provider_failure=result.provider_failure)
        except BaseException as exc:
            failure = exc
            terminal.update(stop_reason="execution_failed_or_interrupted", exception_class=type(exc).__name__)
        finally:
            terminal["transport_invocations"] = transport.calls
            terminal["postcall_validation"] = {}
            accounting_phase = "journal_accounting"
            try:
                usage = accounting(stage, shared)
                terminal["usage_ref"] = D.write(output / (case_id + ".usage.json"), usage)
                D.write(output / "usage.json", usage)
                accounting_phase = "postcall_usage_checks"
                check_usage(usage, stage, expected_count=historical_count + 1 + index)
                terminal["postcall_validation"]["accounting"] = {"status": "PASS"}
            except Exception as exc:
                terminal["accounting_error_class"] = type(exc).__name__
                terminal["postcall_validation"]["accounting"] = {
                    "status": "FAILED", "phase": accounting_phase, "exception_class": type(exc).__name__,
                    "reason": str(exc) if accounting_phase == "postcall_usage_checks" else None}
                if failure is None:
                    failure = exc
            try:
                require(transport.calls == 1, "expected exactly one actual provider invocation")
                terminal["postcall_validation"]["transport"] = {"status": "PASS"}
            except Exception as exc:
                terminal["postcall_validation"]["transport"] = {
                    "status": "FAILED", "exception_class": type(exc).__name__,
                    "reason": "expected exactly one actual provider invocation"}
                if failure is None:
                    failure = exc
            material = P.checked(row["material"]).parent
            terminal["four_layer_refs"] = {
                "source": D.read(material / "provenance.json"),
                "full": D.ref(material / "behavior_full.json"),
                "public": D.ref(material / "public_report.json"),
                "actual_author_delivery": [D.ref(p) for p in sorted(transport.directory.rglob("*")) if p.is_file()],
            }
            terminal["journal_refs"] = [D.ref(p) for p in sorted(round_dir.rglob("*")) if p.is_file()]
            terminal["candidate_refs"] = [D.ref(p) for p in sorted((output / "provider_candidates" / case_id).rglob("*")) if p.is_file()]
            terminal["delivery_status"] = "PASS" if failure is None else "FAILED"
            D.write(terminal_path, terminal)
        if failure is not None:
            raise failure
        return terminal


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-case", choices=CASE_IDS, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    result = run(args.stage.resolve(), args.output.resolve(), args.run_case, check_only=args.check_only)
    print(json.dumps({key: result[key] for key in ("case_id", "status", "stop_reason", "delivery_status", "api_calls", "train_status") if key in result}))
