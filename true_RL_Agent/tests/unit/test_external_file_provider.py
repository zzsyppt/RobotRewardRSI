"""CPU file handoffs through the existing author loop, journal and cost gate."""
import hashlib
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tests.fakes.decision_records import TEST_ACCEPTANCE_PROTOCOL
from tests.fakes.fake_components import FakeTaskPlugin, FakeVerifier
from tests.unit.test_candidate_authoring_loop import PublicContextBuilder, _accepted_record, _write_candidate_package
from true_rl_agent.agent_core.authoring_loop import CandidateAuthoringLoop
from true_rl_agent.agent_core.external_file_provider import ExternalAuthorWaiting, ExternalFileCandidateProvider, ExternalFileProviderConfig
from true_rl_agent.agent_core.generation_provider_budget import CostGuardedProvider, collect_provider_actuals
from true_rl_agent.agent_core.generation_runtime import generation_provider_config
from true_rl_agent.prompting import serialize_prompt_payload
from true_rl_agent.prompt_artifacts import RoundAuthoringJournal

SOURCE = '''from true_rl_agent.candidates.components import RewardTerm, TrainingComponents
def current_best_reward(inputs):
    return inputs.base_height
def build_training_components():
    return TrainingComponents(name="external_fixture", reward_terms=[
        RewardTerm(name="current_best_reward", weight=1.0, fn=current_best_reward)],
        rationale="CPU external author fixture.")
'''


class ExternalFileProviderTests(unittest.TestCase):
    def test_materialized_candidate_without_journal_completion_is_not_reissued(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = ExternalFileProviderConfig(output_dir=root / "candidates", model="fixture",
                max_tokens=2000, timeout_s=1, resume_pending=True)
            def loop():
                return CandidateAuthoringLoop(context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(), acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL),
                    provider=ExternalFileCandidateProvider(config), candidate_admission_policy=FakeVerifier(),
                    round_dir=root / "authoring", max_attempts=1)
            with patch("true_rl_agent.agent_core.external_file_provider.time.monotonic", side_effect=[0, 2]):
                with self.assertRaises(ExternalAuthorWaiting):
                    loop().run_round(round_id="pending", records=[])
            journal = RoundAuthoringJournal(root / "authoring", round_id="pending")
            before = journal.snapshot()
            request = root / "candidates/requests/001/request.json"
            request_bytes = request.read_bytes()
            (root / "candidates/001_already_materialized").mkdir()
            with self.assertRaisesRegex(RuntimeError, "already materialized.*resume unsupported"):
                loop().run_round(round_id="pending", records=[])
            self.assertEqual(journal.snapshot(), before)
            self.assertEqual(request.read_bytes(), request_bytes)
            self.assertEqual(len(list(request.parent.parent.iterdir())), 1)

    def test_async_timeout_pauses_session_and_resumes_one_reserved_attempt(self):
        from dataclasses import replace
        from tests.fakes.backend_environment import use_test_isaac_lab_execution_environment
        from tests.integration.test_generation_session import (
            build_fixture_foundation, SharedPushRunner, PushCandidateProvider,
            PushRecoveryCompositionProvider, run_generation_session)
        from true_rl_agent.agent_core import generation_runtime
        from true_rl_agent.agent_core.generation_cost_journal import GenerationActualCostJournal

        with tempfile.TemporaryDirectory() as tmp, use_test_isaac_lab_execution_environment():
            root = Path(tmp)
            policy = {"backend":"external_file", "model":"fixture", "usage_policy":"unknown_full_reserve",
                "max_attempts_per_lane":1, "input_units_per_attempt_reserve":10000,
                "max_output_tokens":2000, "timeout_s":1}
            kwargs = dict(provider=PushRecoveryCompositionProvider(), cell="push.g1_29dof.reference_v1",
                          max_generations=2, pause=1, provider_policy=policy)
            foundation = build_fixture_foundation(root, **kwargs)
            runner = SharedPushRunner(foundation=foundation)
            foundation = replace(foundation, runner=runner)
            release_response = threading.Event()
            delivered = threading.Event()
            requests, threads, rounds = [], [], {}
            answer = {"candidate_id":"external-delayed", "declared_changed_components":["reward_callable"],
                "components_py":SOURCE.replace("current_best_reward", "fixture_roll_square").replace(
                    "return inputs.base_height", "return inputs.roll * inputs.roll")}

            def exchange(directory):
                requests.append(directory)
                request = json.loads((directory / "request.json").read_text())
                attempt = request["authoring_attempt"]
                attempt_dir = Path(attempt["prompt_payload"]["path"]).parent
                binding = json.loads((attempt_dir / "external_request_binding.json").read_text())
                self.assertEqual(binding["request"], request)  # Bound before publication/callback.
                def deliver():
                    if release_response.wait(20):
                        temporary = directory / "response.pending"
                        temporary.write_text(json.dumps(answer), encoding="utf-8")
                        temporary.replace(directory / "response.txt")
                        delivered.set()
                thread = threading.Thread(target=deliver, daemon=True)
                threads.append(thread)
                thread.start()

            def provider_factory(config):
                if config.output_dir.name == "lane-a":
                    return PushCandidateProvider(config)
                return ExternalFileCandidateProvider(config, exchange=exchange)

            original_run = CandidateAuthoringLoop.run_round
            def remember(loop, **arguments):
                rounds[arguments["round_id"]] = (loop, arguments)
                return original_run(loop, **arguments)

            try:
                with patch.object(generation_runtime, "ExternalFileCandidateProvider", side_effect=provider_factory), \
                        patch.object(CandidateAuthoringLoop, "run_round", remember), \
                        patch("true_rl_agent.agent_core.openai_compatible_provider.UrllibJsonTransport.post_json",
                              side_effect=AssertionError("CPU fixture cannot call a model")):
                    paused = run_generation_session(foundation)
                    self.assertEqual((paused["status"], paused["reason"]), ("paused", "external_author_waiting"))
                    self.assertFalse(delivered.is_set())
                    self.assertEqual(len(requests), 1)
                    generation = f"{foundation.config.session_id}-generation-001"
                    lane_root = foundation.config.output_dir / "authoring" / generation
                    journal = RoundAuthoringJournal(lane_root / "lane-b", round_id=f"{generation}-lane-b")
                    pending = journal.snapshot()
                    self.assertEqual(pending.reserved_attempt_count, 1)
                    self.assertEqual(pending.attempts[0]["status"], "interrupted_after_prompt")
                    original_ref = pending.prompt_payload_refs[0]
                    prior_path = lane_root / "lane-a/authoring_attempts/attempt_000001/attempt_result.json"
                    prior_bytes = prior_path.read_bytes()
                    prior = json.loads(prior_bytes)["provider_attempt"]["provider_usage"]
                    costs = GenerationActualCostJournal(foundation.config.output_dir, contract=foundation.cost_contract)
                    paused_cost = costs.load_chain()[-1]
                    paused_cost_bytes = Path(paused_cost.path).read_bytes()
                    before = collect_provider_actuals(foundation, ledger=[])
                    self.assertEqual(before["total_units"], prior["total_units"] + 12000)
                    self.assertEqual(before["conservative_attempts"], 1)
                    release_response.set()
                    self.assertTrue(delivered.wait(5))
                    rebuilt = build_fixture_foundation(root, resume=True, **kwargs)
                    rebuilt = replace(rebuilt, runner=runner)
                    # Both lanes already have their prompt; the pending one must
                    # use its immutable payload even at a frozen attempt limit of 1.
                    with patch("true_rl_agent.agent_core.authoring_loop.build_authoring_prompt",
                               side_effect=AssertionError("resume rebuilt an author prompt")):
                        completed = run_generation_session(rebuilt)
                        self.assertEqual(completed["closed_generation_count"], 1)
                        done = journal.snapshot()
                        self.assertEqual(done.reserved_attempt_count, 1)
                        self.assertEqual(done.attempts[0]["status"], "completed")
                        self.assertEqual(done.prompt_payload_refs, (original_ref,))
                        loop, arguments = rounds[f"{generation}-lane-b"]
                        with patch.object(loop.provider.delegate._materializer, "parse",
                                          side_effect=AssertionError("completed response parsed twice")):
                            replayed = original_run(loop, **arguments)
                    self.assertEqual(replayed.stop_reason, "candidate_verified")
                    self.assertEqual(Path(replayed.candidate.path).name, "001_external-delayed")
                    self.assertEqual(len(requests), 1)
                    self.assertEqual(len(list(requests[0].parent.iterdir())), 1)
                    self.assertEqual(prior_path.read_bytes(), prior_bytes)
                    self.assertEqual(Path(paused_cost.path).read_bytes(), paused_cost_bytes)
                    after = collect_provider_actuals(rebuilt, ledger=[])
                    self.assertEqual(after, before)
                    receipt = done.completed_attempt_records[0]["provider_response_receipt"]
                    self.assertEqual(receipt["usage_status"], "unknown_full_reserve")
                    self.assertEqual(receipt["response_text_sha256"], "sha256:" + hashlib.sha256(
                        (requests[0] / "response.txt").read_bytes()).hexdigest())
                    self.handoff_evidence = {"pause":paused, "max_attempts":1,
                        "request_count":len(requests), "journal_attempt_count":done.reserved_attempt_count,
                        "prompt_payload_ref":original_ref.to_public_dict(),
                        "candidate_directory":Path(replayed.candidate.path).name,
                        "prior_completed_usage":prior, "pending_usage":before, "completed_usage":after,
                        "response_receipt":receipt, "paused_cost_receipt":dict(paused_cost.payload),
                        "completed_cost_receipt":dict(costs.load_chain()[-1].payload)}
            finally:
                release_response.set()
                for thread in threads:
                    thread.join(5)

    def test_draft_source_followup_and_parse_repair_use_existing_loop_and_full_reserve(self):
        for first_mode in ("draft", "source_request", "bad_json"):
            with self.subTest(first_mode=first_mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generation = "fixture-generation-001"
                caps = {"attempts_per_lane_per_generation": 3,
                        "input_units_per_attempt_reserve": 300000, "output_units_per_attempt_cap": 32768}
                received = []

                def exchange(directory):
                    payload = json.loads((directory / "user.txt").read_text(encoding="utf-8"))
                    received.append(payload)
                    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
                    self.assertEqual(request["backend"], "external_file")
                    self.assertEqual(request["requested_model"], "gpt-6.1-sol")
                    self.assertEqual((directory / "user.txt").read_bytes(), serialize_prompt_payload(payload).encode())
                    self.assertIn("external candidate-package author", (directory / "system.txt").read_text())
                    if len(received) == 1 and first_mode == "source_request":
                        answer = {"response_mode": "source_request",
                                  "logical_ids": ["current_best/training_code/components.py"],
                                  "rationale": "Read the current reward before proposing a change."}
                    else:
                        answer = {"candidate_id": "external-fixture", "components_py": SOURCE,
                                  "declared_changed_components": ["reward_callable"]}
                    raw = b"incomplete JSON" if len(received) == 1 and first_mode == "bad_json" else json.dumps(answer).encode()
                    pending = directory / "response.pending"
                    pending.write_bytes(raw)
                    pending.replace(directory / "response.txt")

                config = generation_provider_config(output_dir=root / "candidates", env_file=None, policy={
                    "backend": "external_file", "model": "gpt-6.1-sol", "usage_policy": "unknown_full_reserve",
                    "max_output_tokens": 32768, "timeout_s": 5})
                self.assertIsInstance(config, ExternalFileProviderConfig)
                delegate = ExternalFileCandidateProvider(config, exchange=exchange)
                provider = CostGuardedProvider(delegate=delegate, output_dir=root, generation_id=generation,
                    lane_id="lane-a", provider_caps=caps, max_generations=2,
                    external_unknown_usage_model="gpt-6.1-sol")
                parent_source = SOURCE.replace("return inputs.base_height", "return inputs.base_height.square()")
                accepted = _write_candidate_package(root / "accepted", "accepted-best", parent_source)
                records = [_accepted_record("previous-round", accepted, 0.31)] if first_mode == "source_request" else []
                loop = CandidateAuthoringLoop(context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(), acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL),
                    provider=provider, candidate_admission_policy=FakeVerifier(),
                    round_dir=root / "authoring" / generation / "lane-a", max_attempts=3)
                # A real API invocation or credentials lookup is never involved.
                with patch("true_rl_agent.agent_core.openai_compatible_provider.UrllibJsonTransport.post_json",
                           side_effect=AssertionError("CPU fixture cannot call a model")):
                    result = loop.run_round(round_id=f"{generation}-lane-a", records=records)
                self.assertEqual(result.stop_reason, "candidate_verified", result.attempt_records)
                self.assertEqual(Path(result.candidate.path, "training_code/components.py").read_bytes(), SOURCE.encode())
                diagnostics = json.loads(Path(result.candidate.path, "diagnostics.json").read_text())
                self.assertEqual(diagnostics["provider_backend"], "external_file")
                count = 1 if first_mode == "draft" else 2
                self.assertEqual(len(result.attempt_records), count)
                if first_mode != "draft":
                    self.assertEqual(received[1]["authoring_context_frame"]["frame_kind"],
                                     "source_followup" if first_mode == "source_request" else "static_repair")
                if first_mode == "source_request":
                    self.assertEqual(result.attempt_records[0]["source_request_result"]["status"], "source_payload_returned")
                    self.assertEqual(received[1]["source_payloads"][0]["content"],
                                     Path(accepted.path, "training_code/components.py").read_bytes().decode())
                for index, record in enumerate(result.attempt_records, 1):
                    self.assertEqual(record.get("provider_usage", {}), {})
                    receipt = record["provider_response_receipt"]
                    self.assertEqual(receipt["usage_status"], "unknown_full_reserve")
                    raw = (root / "candidates" / "requests" / f"{index:03d}" / "response.txt").read_bytes()
                    self.assertEqual(receipt["response_text_sha256"], "sha256:" + hashlib.sha256(raw).hexdigest())
                foundation = SimpleNamespace(config=SimpleNamespace(output_dir=root, session_id="fixture"),
                    session_contract=SimpleNamespace(max_generations=2),
                    cost_contract=SimpleNamespace(to_dict=lambda: {"provider": caps}))
                totals = collect_provider_actuals(foundation, ledger=[])
                self.assertEqual(totals["conservative_attempts"], count)
                self.assertEqual(totals["total_units"], count * 332768)


if __name__ == "__main__":
    unittest.main()
