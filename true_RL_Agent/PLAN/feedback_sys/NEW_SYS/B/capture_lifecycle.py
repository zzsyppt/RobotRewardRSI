"""Historical R4/K1 lifecycle replay; main is not a current validation entry.

Only process/training/provider boundaries are fixtures. Numeric arrays are original
R4/K1 bytes; fixture manifest identity is explicitly rebound, never new-policy data.
The offline capture() helper still serves provider tests and the P2b2 replay.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.fakes import generation_runner as fake
from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment, TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION
from tests.integration.test_generation_session import build_fixture_foundation, SharedStairRunner
from tests.integration import test_generation_session as session_fixture
from true_rl_agent.agent_core import generation_runtime
from true_rl_agent.agent_core.generation_application import run_generation_session
from true_rl_agent.agent_core.ledger import load_ledger
from true_rl_agent.agent_core.public_safety import assert_public_only_prompt_payload
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.evaluation.phases import EvaluationPhase
from true_rl_agent.runner.numeric_evidence import file_sha256
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.runner.numeric_evidence import NUMERIC_EVIDENCE_REF_SCHEMA
from true_rl_agent.runner.subprocess_runner import SubprocessRunner, _SummaryJobCompletion
from true_rl_agent.tasks.stair_climbing.composition import StairClimbingCompositionProvider
from true_rl_agent.tasks.stair_climbing.episode_evidence import StairEpisodeOutcome
from true_rl_agent.tasks.stair_climbing.isaaclab_execution import StairEvaluationExecutionResult
from true_rl_agent.tasks.stair_climbing.phased_evaluation import build_stair_eval_phase_summary
from true_rl_agent.tasks.stair_climbing.seed_policy import STAIR_FORMAL_EVAL_NAMESPACE
from true_rl_agent.agent_core.openai_compatible_provider import OpenAICompatibleCandidateProvider, OpenAICompatibleProviderConfig, UrllibJsonTransport
from true_rl_agent.tasks.stair_climbing.trajectory_adapter import _load_contract
from true_rl_agent.tasks.stair_climbing.trajectory_capabilities import CAPTURE_PROFILE
from tests.unit.test_stair_trajectory_production import cpu_contract


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


class RequestCaptured(Exception):
    pass


class CaptureOpener:
    def open(self, request, *, timeout):
        self.body = request.data
        raise RequestCaptured()


def capture(payload, output):
    opener = CaptureOpener()
    provider = OpenAICompatibleCandidateProvider(OpenAICompatibleProviderConfig(
        model="offline-request-capture", base_url="https://request-capture.invalid",
        api_key="offline-placeholder", output_dir=output / "unused", max_tokens=32768,
        reasoning_mode="disabled"), transport=UrllibJsonTransport())
    with patch("urllib.request.build_opener", return_value=opener):
        try:
            provider.next_authoring_response(prompt_payload=payload)
        except RequestCaptured:
            pass
        else:
            raise AssertionError("production transport did not reach capture boundary")
    request = json.loads(opener.body)
    user = request["messages"][-1]["content"].encode("utf-8")
    if json.loads(user) != payload:
        raise AssertionError("serialized provider payload differs")
    return opener.body, user



class StopAfterNextRequest(BaseException):
    pass


ProfileProvider = StairClimbingCompositionProvider


class ArchiveRunner(SharedStairRunner):
    def __init__(self, *, foundation, sources):
        super().__init__(foundation=foundation)
        self.production = foundation.runner
        self.sources = sources
        self.specs = {}
        self.mappings = []
        self.replayed = []

    def run_eval_phase(self, spec):
        if spec.phase is not EvaluationPhase.PUBLIC_DEV:
            return super().run_eval_phase(spec)
        self.specs[spec.job_id] = spec
        result = self.production.run_eval_phase(spec)
        # Exercise the actual runner attempt-completion recovery, not just the
        # outer trial journal's shortcut. It must leave every artifact untouched.
        before = {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in Path(spec.metadata["summary_path"]).parent.rglob("*") if p.is_file()}
        again = self.production.run_eval_phase(spec)
        assert again.to_private_dict() == result.to_private_dict()
        assert before == {str(p): (p.stat().st_mtime_ns, p.stat().st_size) for p in Path(spec.metadata["summary_path"]).parent.rglob("*") if p.is_file()}
        self.replayed.append(spec.job_id)
        return result

    def boundary(self, runner, metadata, **kwargs):
        job_id = kwargs["identity"].job_id
        spec = self.specs[job_id]
        context = kwargs.get("execution_context")
        binding = context.gpu_process_binding if context else spec.gpu_process_binding
        path = Path(metadata["summary_path"])
        if kwargs.get("reuse_existing_summary"):
            return _SummaryJobCompletion(read(path), "pass", 0, binding)
        self._claim_actual("eval_phase", job_id)
        source_name = "r4" if spec.subject.kind == "candidate" and "lane-b" in job_id else "k1"
        source = self.sources[source_name]
        original = read(source)
        archive = deepcopy(original)
        config = self.foundation.composition.task_plugin.config
        directory = Path(metadata["numeric_evidence_dir"])
        directory.mkdir(parents=True, exist_ok=False)
        for chunk in archive["chunks"]:
            shutil.copyfile(source.parent / chunk["file"], directory / chunk["file"])
            assert file_sha256(directory / chunk["file"]) == chunk["sha256"]
        identity = {**original["identity"], **spec.job_identity().to_public_dict(),
            "checkpoint_hash": spec.execution_checkpoint().checksum,
            "evaluation_plan_hash": spec.evaluation_plan_hash,
            "evaluation_binding": spec.evaluation_binding.to_public_dict(),
            "seed_namespace": STAIR_FORMAL_EVAL_NAMESPACE,
            "environment_source_preflight": read(self.sources["k1"])["identity"]["environment_source_preflight"],
            "backend_execution_environment_verification": TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION}
        archive["identity"] = identity
        # Training profile mapping order is part of the archive contract.
        (directory / "manifest.json").write_text(json.dumps(archive, ensure_ascii=False) + "\n", encoding="utf-8")
        reference = {"schema_version": NUMERIC_EVIDENCE_REF_SCHEMA, "manifest_path": str(directory / "manifest.json"),
                     "manifest_sha256": file_sha256(directory / "manifest.json")}
        qualified = _load_contract(reference)
        archive["metadata"].update(profile=CAPTURE_PROFILE,
            measurement_contract=cpu_contract(qualified.projection, qualified.cells))
        (directory / "manifest.json").write_text(json.dumps(archive, ensure_ascii=False)+"\n", encoding="utf-8")
        reference["manifest_sha256"] = file_sha256(directory / "manifest.json")
        outcomes = tuple(tuple(StairEpisodeOutcome(**{**o, "env_index": index,
            "diagnostic_flags": tuple(o["diagnostic_flags"]), "active_terminal_terms": ()}) for index, o in enumerate(seed["outcomes"]))
            for seed in archive["metadata"]["seeds"])
        execution = StairEvaluationExecutionResult(summaries={"public_dev": archive["metadata"]["summary"]},
            per_seed_outcomes={"public_dev": outcomes},
            split_vector_env_step_calls={"public_dev": sum(seed["observed_control_steps"] for seed in archive["metadata"]["seeds"])},
            split_wall_times_s={"public_dev": 0.0}, checkpoint_hash=identity["checkpoint_hash"],
            seed_namespace=STAIR_FORMAL_EVAL_NAMESPACE, seed_cohort_hash=spec.evaluation_binding.seed_cohort_hash,
            source_manifest_preflight=identity["environment_source_preflight"],
            backend_execution_environment_verification=TEST_ISAAC_LAB_EXECUTION_ENVIRONMENT_VERIFICATION,
            execution_device=spec.execution_device, numeric_evidence=reference)
        handoff = build_stair_eval_phase_summary(config=config, identity=spec.job_identity(),
            evaluation_plan_hash=spec.evaluation_plan_hash, phase=spec.phase, execution=execution,
            evaluation_binding=spec.evaluation_binding,
            checkpoint_safety_report=fake._checkpoint_report(spec.execution_checkpoint().checksum))
        handoff.update(fake._gpu_summary_fields(job_id, binding, context.physical_id))
        write(path, handoff)
        write(path.with_name("injected_handoff.json"), handoff)
        self.mappings.append({"source": source_name, "original_manifest_path": str(source),
            "original_manifest_sha256": file_sha256(source), "original_identity": original["identity"],
            "fixture_identity": identity, "fixture_manifest": reference,
            "changed_identity_fields": sorted(k for k in identity if identity[k] != original["identity"].get(k)),
            "unchanged_npz_sha256": [c["sha256"] for c in archive["chunks"]],
            "spec": spec.to_public_dict(), "summary_path": str(path)})
        print("CPU handoff", source_name, job_id, flush=True)
        return _SummaryJobCompletion(handoff, "postprocessing", 0, binding)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--r4", type=Path, required=True)
    parser.add_argument("--k1", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {"r4": args.r4.resolve(), "k1": args.k1.resolve()}
    assert file_sha256(sources["r4"]) == "8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83"
    assert file_sha256(sources["k1"]) == "0389a10d7cb9dbe14fff063f43c56482fcc6fbd918281a37beacb59c043db8d3"
    requests = []
    config_rejections = []

    class Provider(fake._FormalFixtureProvider):
        def next_authoring_response(self, *, prompt_payload):
            name = f"g{self.generation_index}-{self.lane_id}"
            body, user = capture(prompt_payload, output)
            target = output / "requests" / name
            target.mkdir(parents=True)
            (target / "request.http_body.json").write_bytes(body)
            (target / "user_payload.json").write_bytes(user)
            feedback = json.loads(user)["public_feedback_summary"]["behavior"]
            assert_public_only_prompt_payload(json.loads(user))
            for private_marker in ("manifest_path", "numeric_evidence_dir", "trajectory_build", "evidence.jsonl", "unchanged_npz_sha256"):
                assert private_marker not in user.decode("utf-8"), private_marker
            requests.append({"name": name, "http_sha256": file_sha256(target / "request.http_body.json"),
                "user_sha256": file_sha256(target / "user_payload.json"), "http_bytes": len(body), "user_bytes": len(user),
                "behavior": feedback})
            print("HTTP captured", name, feedback.get("source_evaluation_job_id", "description_unavailable"), flush=True)
            if self.generation_index == 2:
                raise StopAfterNextRequest()
            return super().next_authoring_response(prompt_payload=prompt_payload)

    (output / "provider.env").write_text("FIXTURE_KEY=fixture-only\n", encoding="utf-8")
    source_preflight = read(sources["k1"])["identity"]["environment_source_preflight"]
    original_prepare = session_fixture.prepare_generation_foundation
    def prepare_with_archive_storage(config, **kwargs):
        # Only this CPU fixture's storage reserve changes. The real numeric
        # evidence is much larger than the original synthetic phase fixture.
        cost = replace(kwargs["cost"], session_output_reserve_bytes=6 * 1024**3)
        kwargs.update(cost=cost)
        return original_prepare(replace(config, max_output_bytes=cost.max_output_bytes), **kwargs)
    with use_test_isaac_lab_execution_environment(), patch.object(fake, "_source_preflight", return_value=source_preflight), \
            patch.object(session_fixture, "prepare_generation_foundation", side_effect=prepare_with_archive_storage):
        foundation = build_fixture_foundation(output, provider=ProfileProvider(), cell="stair.go2_12dof.unified_v3", pause=1)
        runner = ArchiveRunner(foundation=foundation, sources=sources)
        source = foundation.composition.reference_program_artifact.source_bytes.decode("utf-8")
        def boundary(actual, metadata, **kwargs):
            return runner.boundary(actual, metadata, **kwargs)
        with patch.object(Provider, "reference_source", source), patch.object(generation_runtime, "OpenAICompatibleCandidateProvider", Provider), patch.object(SubprocessRunner, "_execute_summary_job", new=boundary):
            paused = run_generation_session(replace(foundation, runner=runner))
            write(output / "paused.json", paused)
            write(output / "source_mapping.json", runner.mappings)
            ledger = load_ledger(foundation.config.output_dir / "ledger.jsonl")
            assert len(ledger) == 1
            resumed = build_fixture_foundation(output, provider=ProfileProvider(), cell="stair.go2_12dof.unified_v3", resume=True, pause=1)
            try:
                run_generation_session(replace(resumed, runner=runner))
            except ValueError as error:
                assert str(error) == "requested generation boundary is already closed", str(error)
            else:
                raise AssertionError("closed generation boundary was executed twice")
            class SmallerBudget(ProfileProvider):
                def cli_defaults(self):
                    return {**super().cli_defaults(), "trajectory_report_max_bytes": 8191}
            for provider in (SmallerBudget(),):
                previous_calls = (len(runner.mappings), len(requests))
                changed = build_fixture_foundation(output, provider=provider, cell="stair.go2_12dof.unified_v3", resume=True, pause=1)
                try:
                    generation_runtime.prepare_generation_genesis(replace(changed, runner=runner), require_existing=True)
                except ValueError as error:
                    assert "session Genesis identity drift" in str(error), str(error)
                    config_rejections.append(str(error))
                else:
                    raise AssertionError("changed feedback configuration resumed an existing session")
                assert previous_calls == (len(runner.mappings), len(requests))
            continued = replace(resumed, config=replace(resumed.config, pause_after_generation=None))
            try:
                run_generation_session(replace(continued, runner=runner))
            except StopAfterNextRequest:
                pass
            else:
                raise AssertionError("next real author request was not captured")
    for item in runner.mappings:
        phase = read(item["summary_path"])
        item["phase"] = phase
        item["full_sha256"] = file_sha256(Path(item["summary_path"]).with_name("behavior_full.json"))
    assert len(runner.mappings) == 3
    assert len({i["phase"]["task_evidence"]["numeric_evidence"]["manifest_sha256"] for i in runner.mappings}) == 3
    assert len({i["phase"]["task_evidence"]["behavior"]["card_id"] for i in runner.mappings}) == 3
    next_request = requests[-1]
    assert next_request["name"] == "g2-lane-a"
    sibling = next(i for i in runner.mappings if "lane-b" in i["fixture_identity"]["job_id"])
    assert next_request["behavior"]["profile"] == {"kind":"numeric_behavior", "report":sibling["phase"]["task_evidence"]["behavior"]["report"]}
    assert next_request["behavior"]["diff"]["status"] == "available"
    assert "behavior_comparison" not in (output/"requests/g2-lane-a/user_payload.json").read_text(encoding="utf-8")
    assert all(not (Path(i["summary_path"]).parent/"trajectory").exists() for i in runner.mappings)
    artifact_bytes = sum(Path(i["summary_path"]).with_name("behavior_full.json").stat().st_size for i in runner.mappings)
    receipt = {"status": "PASS_CPU_COMPOSITION_ARCHIVE_GENERATION_HTTP", "requests": requests,
        "sources": runner.mappings, "runner_completed_replays": runner.replayed,
        "changed_configuration_resume_rejections": config_rejections,
        "wall_seconds": time.perf_counter() - started, "description_artifact_bytes":artifact_bytes,
        "generation_outcome": ledger[0]["generation_outcome"],
        "boundary": "normal composition runner; archived phase injection replaces process/GPU boundary; fixture training/nonpublic outcomes/provider drafts",
        "new_policy_evidence": False, "gpu_runs": 0, "external_model_calls": 0, "http_network_calls": 0,
        "not_verified": ["actual checkpoint loading", "Isaac rollout", "GPU qualification", "model comprehension", "P2/E2 transfer"]}
    write(output / "verification.json", receipt)
    print(json.dumps({k: receipt[k] for k in ["status", "generation_outcome", "runner_completed_replays"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
