"""CPU-only normal Push lifecycle with explicit K4 archive identity rebinding.

Original arrays are copied unchanged. Fixture checkpoint/training/nonpublic
outcomes/provider responses are not scientific evidence of a new policy.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from tests.fakes import generation_runner as fake
from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment, TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION as ENV
from tests.integration.test_generation_session import build_fixture_foundation, SharedPushRunner, PushCandidateProvider
from tests.unit.test_push_trajectory_pipeline import rebind_archives, seed_from_manifest
from true_rl_agent.agent_core import generation_runtime
from true_rl_agent.agent_core.generation_application import run_generation_session
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.public_safety import assert_public_only_prompt_payload
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.evaluation.trajectory_translation.report import file_sha256
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.memory.context_builder import PublicContextBuilder
from true_rl_agent.prompting.task_sections import validate_provider_task_prompt_receipt
from true_rl_agent.runner.subprocess_runner import SubprocessRunner, _SummaryJobCompletion
from true_rl_agent.runner.gpu_lease import PreGenesisGpuLeaseHolder, GpuLeaseUnavailableError
from true_rl_agent.tasks.push_recovery.composition import PushRecoveryCompositionProvider
from true_rl_agent.tasks.push_recovery.behavior import push_behavior_context
from true_rl_agent.tasks.push_recovery.phased_evaluation import build_push_phase_summary
from true_rl_agent.tasks.push_recovery.trajectory_pipeline import build_phase_trajectory


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StopAfterNextRequest(BaseException):
    pass


class ProfileProvider(PushRecoveryCompositionProvider):
    def cli_defaults(self):
        return {**super().cli_defaults(), "behavior_feedback_profile": "trajectory_numeric_v1",
                "trajectory_report_max_bytes": 12288}


class ArchiveRunner(SharedPushRunner):
    def __init__(self, *, foundation, sources):
        super().__init__(foundation=foundation)
        self.production, self.sources = replace(foundation.runner, eval_phase_summary_finalizer=self.finalize), sources
        self.specs, self.mappings, self.replayed = {}, [], []
        self.lease_checks = []

    def probe_lease(self, spec, *, available):
        device = spec.gpu_process_binding.host_execution_device
        holder = PreGenesisGpuLeaseHolder(session_id="cpu-lease-check", genesis_identity_hash="a" * 64,
            session_contract_hash="b" * 64, lane_id=device, invocation_segment_id="probe", logical_job_id="probe")
        try:
            with self.foundation.gpu_lease_manager.acquire(device, holder=holder):
                assert available, "rollout lost its lease"
        except GpuLeaseUnavailableError:
            assert not available, "CPU finalizer retained the GPU lease"
        self.lease_checks.append({"job_id": spec.job_id, "available": available})

    def finalize(self, spec, summary, *, output_dir):
        self.probe_lease(spec, available=True)
        return self.foundation.runner.eval_phase_summary_finalizer(spec, summary, output_dir=output_dir)

    def _train_summary(self, spec, *, binding, physical_uuid):
        from true_rl_agent.runner.budget import build_budget_accounting
        config = self.foundation.composition.task_plugin.config
        summary = {"schema_version": "true_rl_agent.fixture.push_train.v1",
            **spec.job_identity().to_public_dict(), "status": "pass", "errors": [],
            "execution_device": binding.process_execution_device, "formal_training_executed": True,
            "training": {"initial_policy_state_hash": spec.expected_initial_policy_state_hash or "a" * 64,
                "budget_accounting": build_budget_accounting(budget_steps=config.budget_steps,
                    num_envs=config.train_num_envs, num_steps_per_env=config.train_steps_per_iteration,
                    max_iterations=config.train_max_iterations)},
            **fake._gpu_summary_fields(spec.job_id, binding, physical_uuid)}
        if spec.candidate_artifact is not None:
            import torch
            from true_rl_agent.tasks.push_recovery.runtime_adapter import CandidateRuntimeAdapter
            runtime = CandidateRuntimeAdapter.from_candidate_dir(spec.candidate_artifact.path,
                num_envs=4, planned_training_steps=1, diagnostic_interval_steps=1)
            inputs = self.foundation.composition.task_plugin.component_smoke_adapter().build_component_smoke_fixture().reward_inputs
            with patch.object(runtime, "_build_inputs", return_value=inputs):
                runtime.apply_step_rewards(env=None, actions=None, rewards=torch.zeros(4))
            summary["runtime_activation_summary"] = runtime.runtime_summary()
        return json.loads(json.dumps(summary))

    def run_eval_phase(self, spec):
        self.specs[spec.job_id] = spec
        result = self.production.run_eval_phase(spec)
        directory = Path(spec.metadata["summary_path"]).parent
        before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in directory.rglob("*") if p.is_file()}
        # Completed attempt recovery must not reload numeric evidence.
        with patch("true_rl_agent.runner.numeric_evidence.load_numeric_evidence_manifest", side_effect=AssertionError("completed replay read raw")):
            again = self.production.run_eval_phase(spec)
        assert result.to_private_dict() == again.to_private_dict()
        assert before == {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in directory.rglob("*") if p.is_file()}
        self.replayed.append(spec.job_id)
        return result

    def boundary(self, runner, metadata, **kwargs):
        job_id = kwargs["identity"].job_id
        spec, execution = self.specs[job_id], kwargs["execution_context"]
        binding, path = execution.gpu_process_binding, Path(metadata["summary_path"])
        if kwargs.get("reuse_existing_summary"):
            return _SummaryJobCompletion(read(path), "pass", 0, binding)
        self._claim_actual("eval_phase", job_id)
        self.probe_lease(spec, available=False)
        config = self.foundation.composition.task_plugin.config
        context = push_behavior_context(self.foundation.composition.resolved_cell_manifest,
            evaluation_plan_hash=spec.evaluation_plan_hash, checkpoint_hash=spec.execution_checkpoint().checksum,
            program_source_checksum=spec.program_source_checksum,
            runtime_dynamics_hash=canonical_json_hash(config.eval_runtime_dynamics()))
        identity = {**spec.job_identity().to_public_dict(),
            "evaluation_binding": spec.evaluation_binding.to_public_dict(),
            "evaluation_plan_hash": spec.evaluation_plan_hash, "checkpoint_hash": spec.execution_checkpoint().checksum,
            "program_source_checksum": spec.program_source_checksum, "training_binding_hash": spec.training_binding_hash,
            "runtime_dynamics_hash": context["runtime_dynamics_hash"], "backend_execution_environment_verification": ENV}
        count = self.foundation.phased_adapter.seed_count(spec.phase)
        if spec.phase is EvaluationPhase.PUBLIC_DEV:
            assert len(self.sources) == count == 3
            seeds, mapping = rebind_archives(self.sources, Path(metadata["numeric_evidence_dir"]), identity, context)
            self.mappings.append({"job_id": job_id, "summary_path": str(path), "sources": mapping,
                                  "spec": spec.to_public_dict()})
        else:
            # Nonpublic outcomes are explicitly synthetic fixture observations.
            # They are never archived or used to build the root behavior card.
            seeds = [seed_from_manifest(read(self.sources[i % 3]), split=spec.phase.value) for i in range(count)]
        summary = build_push_phase_summary(phase=spec.phase, evaluation_plan_hash=spec.evaluation_plan_hash,
            seed_summaries=seeds, expected_seed_count=count, wall_time_ns=10_000_000,
            identity=identity, execution_device=binding.process_execution_device,
            checkpoint_safety_report=fake._checkpoint_report(spec.execution_checkpoint().checksum),
            runtime_dynamics_hash=context["runtime_dynamics_hash"],
            behavior_context=context if spec.phase is EvaluationPhase.PUBLIC_DEV else None)
        summary.update(fake._gpu_summary_fields(job_id, binding, execution.physical_id))
        summary["evaluation_protocol"] = spec.evaluation_protocol().to_public_dict()
        write(path, summary)
        write(path.with_name("fixture_handoff.json"), summary)
        print("CPU handoff", job_id, flush=True)
        return _SummaryJobCompletion(summary, summary["status"], 0, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    lock = read(ROOT / "PLAN/feedback_sys/NEW_SYS/P2b1/source_lock.json")
    group = next(g for g in lock["groups"] if g["source_id"] == "k4_force_v1")
    sources = [ROOT / s["manifest_path"] for s in group["sources"]]
    for path, expected in zip(sources, group["sources"]):
        assert file_sha256(path) == expected["manifest_sha256"]
    capture = load_module("p2b2_http", ROOT / "PLAN/feedback_sys/NEW_SYS/P1c/capture_delivery.py").capture
    requests, config_rejections, frames = [], [], []
    last_prompt = {}
    original_build = PublicContextBuilder.build_prompt
    def trace_prompt(builder, records, **kwargs):
        result = original_build(builder, records, **kwargs)
        last_prompt.update(builder=builder, records=records, kwargs=kwargs)
        return result

    class Provider(PushCandidateProvider):
        def next_authoring_response(self, *, prompt_payload):
            name = f"g{self.generation_index}-{self.lane_id}"
            attempt = sum(r["name"].startswith(name) for r in requests)
            if attempt:
                name += f"-attempt-{attempt + 1}"
            body, user = capture(prompt_payload, output)
            target = output / "requests" / name
            target.mkdir(parents=True)
            (target / "request.http_body.json").write_bytes(body)
            (target / "user_payload.json").write_bytes(user)
            payload = json.loads(user)
            assert_public_only_prompt_payload(payload)
            for marker in ("manifest_path", "numeric_evidence_dir", "trajectory_build", "evidence.jsonl"):
                assert marker not in user.decode("utf-8"), marker
            behavior = payload["public_feedback_summary"]["public_dev"]["behavior"]
            requests.append({"name": name, "http_sha256": file_sha256(target / "request.http_body.json"),
                "user_sha256": file_sha256(target / "user_payload.json"), "http_bytes": len(body), "user_bytes": len(user),
                "feedback": payload["public_feedback_summary"]})
            print("HTTP captured", name, behavior["card_id"], flush=True)
            if self.generation_index == 2:
                maker = load_module("p2b2_frames", ROOT / "PLAN/feedback_sys/P3NEW/audit_delivery_offline.py")
                builder = SimpleNamespace(build_prompt=lambda _records, **kwargs: original_build(
                    last_prompt["builder"], last_prompt["records"], **{**last_prompt["kwargs"], **kwargs}))
                for frame_name, value in maker.make_frames(builder, foundation.composition, output / "frames").items():
                    http, user_frame = capture(value, output)
                    delivered = json.loads(user_frame)
                    assert delivered["public_feedback_summary"]["public_dev"]["behavior"] == behavior
                    assert delivered["public_feedback_summary"]["trajectory_sources"] == payload["public_feedback_summary"]["trajectory_sources"]
                    budget = delivered["authoring_context_frame"]["budget_metadata"]
                    validate_provider_task_prompt_receipt(budget["provider_task_sections"], prompt_payload=delivered)
                    assert budget["actual_prompt_payload_bytes"] == len(user_frame)
                    directory = output / "frames" / frame_name
                    directory.mkdir(parents=True)
                    (directory / "request.http_body.json").write_bytes(http)
                    (directory / "user_payload.json").write_bytes(user_frame)
                    claims = []
                    for section, rows in behavior["report"].items():
                        if section == "actionability":
                            continue
                        for row in rows:
                            fragment = json.dumps(row["text"], ensure_ascii=False)[1:-1]
                            escaped = json.dumps(fragment, ensure_ascii=False)[1:-1].encode("utf-8")
                            assert fragment.encode("utf-8") in user_frame and escaped in http
                            claims.append({"id": row["id"], "section": section,
                                "user_offset": user_frame.index(fragment.encode("utf-8")), "http_offset": http.index(escaped)})
                    frames.append({"name": frame_name, "http_bytes": len(http), "user_bytes": len(user_frame),
                        "http_sha256": file_sha256(directory / "request.http_body.json"),
                        "user_sha256": file_sha256(directory / "user_payload.json"), "claims": claims,
                        "card_id": behavior["card_id"], "budget": budget})
                raise StopAfterNextRequest()
            from true_rl_agent.agent_core.authoring_response import CandidateDraftResponse
            from true_rl_agent.candidates.schema import CandidatePackageRef, CandidateChangeSet
            candidate_id = f"fixture-g1-{self.lane_id}"
            candidate_dir = self.output_dir / candidate_id
            term = "fixture_height" if self.lane_id == "lane-a" else "fixture_roll"
            expr = "inputs.base_height_relative" if self.lane_id == "lane-a" else "inputs.roll * inputs.roll"
            source = foundation.composition.reference_program_artifact.source_text
            assert source.count("def build_training_components():") == 1 and source.count("        ],\n    )") == 1
            source = source.replace("def build_training_components():",
                f"def {term}(inputs):\n    return {expr}\n\ndef build_training_components():")
            source = source.replace("        ],\n    )",
                f"            RewardTerm(name='{term}', weight=0.1, fn={term}),\n        ],\n    )")
            fake._write_candidate(candidate_dir, candidate_id=candidate_id, source=source)
            return CandidateDraftResponse(candidate=CandidatePackageRef(candidate_id=candidate_id,
                path=str(candidate_dir), change_set=CandidateChangeSet(frozenset({"reward_callable"})), metadata={}),
                provider_usage=fake._provider_usage())

    (output / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
    with use_test_isaac_lab_execution_environment():
        foundation = build_fixture_foundation(output, provider=ProfileProvider(), cell="push.go2_12dof.recovery_v3", pause=1)
        runner = ArchiveRunner(foundation=foundation, sources=sources)
        def boundary(actual, metadata, **kwargs):
            return runner.boundary(actual, metadata, **kwargs)
        with patch.object(generation_runtime, "OpenAICompatibleCandidateProvider", Provider), \
                patch.object(PublicContextBuilder, "build_prompt", new=trace_prompt), \
                patch.object(SubprocessRunner, "_execute_summary_job", new=boundary):
            paused = run_generation_session(replace(foundation, runner=runner))
            write(output / "paused.json", paused)
            ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
            assert len(ledger) == 1
            resumed = build_fixture_foundation(output, provider=ProfileProvider(), cell="push.go2_12dof.recovery_v3", resume=True, pause=1)
            try:
                run_generation_session(replace(resumed, runner=runner))
            except ValueError as error:
                assert str(error) == "requested generation boundary is already closed", str(error)
            else:
                raise AssertionError("closed generation boundary repeated")
            class SmallerBudget(ProfileProvider):
                def cli_defaults(self):
                    return {**super().cli_defaults(), "trajectory_report_max_bytes": 12287}
            for provider in (PushRecoveryCompositionProvider(), SmallerBudget()):
                changed = build_fixture_foundation(output, provider=provider, cell="push.go2_12dof.recovery_v3", resume=True, pause=1)
                previous = len(runner.mappings), len(requests)
                try:
                    generation_runtime.prepare_generation_genesis(replace(changed, runner=runner), require_existing=True)
                except ValueError as error:
                    assert "session Genesis identity drift" in str(error), str(error)
                    config_rejections.append(str(error))
                else:
                    raise AssertionError("configuration drift accepted")
                assert previous == (len(runner.mappings), len(requests))
            continued = replace(resumed, config=replace(resumed.config, pause_after_generation=None))
            try:
                run_generation_session(replace(continued, runner=runner))
            except StopAfterNextRequest:
                pass
            else:
                raise AssertionError("next request not captured")
    for item in runner.mappings:
        item["phase"] = read(item["summary_path"])
        item["full_sha256"] = file_sha256(Path(item["summary_path"]).with_name("behavior_full.json"))
        assert item["phase"]["task_evidence"]["trajectory_build"]["episodes"] == 192
    assert len(runner.mappings) == 3
    assert len({m["phase"]["task_evidence"]["behavior"]["card_id"] for m in runner.mappings}) == 3
    assert requests[-1]["name"] == "g2-lane-a"
    write(output / "verification.json", {"status": "PASS_CPU_NORMAL_PUSH_GENERATION_HTTP",
        "requests": requests, "frames": frames, "source_mapping": runner.mappings, "completed_runner_replays": runner.replayed,
        "configuration_drift_rejections": config_rejections, "lease_checks": runner.lease_checks,
        "generation_outcome": ledger[0]["generation_outcome"],
        "wall_seconds": time.perf_counter() - start, "new_policy_scientific_evidence": False,
        "gpu_runs": 0, "model_calls": 0, "network_calls": 0,
        "scope": "production composition/runner/finalizer/receipts/generation/provider HTTP serialization; fixture process/training/nonpublic outcomes"})
    print("PASS_CPU_NORMAL_PUSH_GENERATION_HTTP", flush=True)


if __name__ == "__main__":
    main()
