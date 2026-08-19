from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests.fakes.decision_records import (
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
)
from tests.fakes.fake_components import FakeTaskPlugin, FakeVerifier
from true_rl_agent.agent_core.authoring_loop import (
    CandidateAuthoringLoop,
    build_authoring_prompt,
)
from true_rl_agent.agent_core.openai_compatible_provider import (
    OpenAICompatibleCandidateProvider,
)
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.memory import PublicContextBuilder
from true_rl_agent.prompt_artifacts import verify_prompt_payload
from true_rl_agent.tasks.stair_climbing.paired_ready_probe import (
    STAIR_PAIRED_READY_PROBE_MONEY_GOVERNOR,
    STAIR_PAIRED_READY_PROBE_TOTAL_OUTPUT_TOKENS,
    build_stair_paired_ready_probe_plan,
    create_stair_paired_ready_probe_plan,
    load_stair_paired_ready_probe_plan,
    stair_paired_ready_probe_plan_hash,
    validate_stair_paired_ready_probe_plan,
)
from true_rl_agent.tasks.stair_climbing.paired_ready_probe_runtime import (
    RecordingProbeTransport,
    STAIR_PAIRED_READY_PROBE_CASE_RESULT_SCHEMA_VERSION,
    STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS,
    _create_or_load_runtime_correction,
    _case_result as _runtime_case_result,
    _project_existing_terminal_authoring_result,
    _project_runtime_correction_trigger_result,
    _verify_existing_case_result,
    build_stair_paired_ready_probe_result,
    fixed_stair_paired_ready_provider_config,
)


_SOURCE_IDENTITY = {
    "source_commit": "a" * 40,
    "source_archive_checksum": "sha256:" + "b" * 64,
    "source_ledger_checksum": "sha256:" + "c" * 64,
    "source_session_genesis_hash": "d" * 64,
}


def _checksum(label: str) -> str:
    return "sha256:" + hashlib.sha256(label.encode("utf-8")).hexdigest()


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _exclusion(round_index: int) -> dict[str, object]:
    projection: dict[str, object] = {
        "schema_version": "true_rl_agent.single_component_semantic_exclusion.v1",
        "operation": "add",
        "canonical_mechanism_delta": f"RewardTerm:mechanism_{round_index:02d}",
        "component_kind": "RewardTerm",
        "public_input_dependencies": [f"input_{round_index:02d}"],
        "parameter_normalized_semantic_signature": _digest(
            f"semantic-{round_index}"
        ),
        "instruction": (
            "Propose one component mechanism whose normalized callable, input, "
            "and clip structure is substantively different from this exclusion; "
            "renaming or changing only numeric parameters is not distinct."
        ),
    }
    projection["exclusion_hash"] = canonical_json_hash(projection)
    return projection


def _request(round_index: int, exclusion: dict[str, object]) -> dict[str, object]:
    prompt = {
        "schema_version": "test.production.prompt.v1",
        "round_index": round_index,
        "candidate_capabilities": {
            "verifier_prompt_rules": {
                "single_component_semantic_exclusion": exclusion,
            }
        },
    }
    return {
        "model": "deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": "Return one candidate JSON object."},
            {
                "role": "user",
                "content": json.dumps(prompt, ensure_ascii=False, indent=2),
            },
        ],
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": 32_768,
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }


def _cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for round_index in range(1, 16):
        exclusion = _exclusion(round_index)
        cases.append(
            {
                "round_index": round_index,
                "source_round_id": f"formal-round-{round_index:03d}",
                "prefix_ledger_hash": _digest(f"prefix-{round_index}"),
                "prefix_ledger_count": round_index - 1,
                "source_prompt_checksum": _checksum(f"prompt-{round_index}"),
                "a_candidate_artifact_checksum": _checksum(
                    f"artifact-{round_index}"
                ),
                "a_candidate_source_checksum": _checksum(f"source-{round_index}"),
                "a_candidate_anchor_checksum": _checksum(f"anchor-{round_index}"),
                "production_exclusion_projection": exclusion,
                "provider_request": _request(round_index, exclusion),
            }
        )
    return cases


def _build(cases: list[dict[str, object]]) -> dict[str, object]:
    return build_stair_paired_ready_probe_plan(cases, **_SOURCE_IDENTITY)


def _create(path: Path, cases: list[dict[str, object]]) -> dict[str, object]:
    return create_stair_paired_ready_probe_plan(
        path,
        cases=cases,
        **_SOURCE_IDENTITY,
    )


def _case_result(
    plan: dict[str, object],
    round_index: int,
    classification: str,
) -> dict[str, object]:
    planned = plan["cases"][round_index - 1]
    candidate = (
        {
            "candidate_id": f"candidate-{round_index}",
            "relative_path": f"candidate-{round_index}",
            "program_source_checksum": _checksum(f"candidate-{round_index}"),
            "program_source_size_bytes": 100,
        }
        if classification
        in {"valid_structurally_distinct", "equivalent_leak"}
        else None
    )
    sibling = (
        {
            "schema_version": (
                "true_rl_agent.sibling_semantic_exclusion_assessment.v1"
            ),
            "status": "pass",
            "blocking": False,
            "reason": "sibling_semantic_delta_distinct",
            "equivalence": None,
            "exclusion_hash": planned["production_exclusion_hash"],
        }
        if classification
        in {"valid_structurally_distinct", "equivalent_leak"}
        else None
    )
    body: dict[str, object] = {
        "schema_version": STAIR_PAIRED_READY_PROBE_CASE_RESULT_SCHEMA_VERSION,
        "plan_hash": plan["plan_hash"],
        "case_id": planned["case_id"],
        "round_index": round_index,
        "source_round_id": planned["source_round_id"],
        "production_exclusion_hash": planned["production_exclusion_hash"],
        "provider_request_count": 1,
        "provider_request_hash": planned["expected_provider_request_hash"],
        "provider_response_checksum": (
            _checksum(f"response-{round_index}")
            if classification
            in {
                "valid_structurally_distinct",
                "equivalent_leak",
                "provider_failed",
            }
            else None
        ),
        "prompt_payload_ref": {"path": "prompt.json", "checksum": _checksum("prompt")},
        "authoring_evidence_ref": {
            "kind": "provider_authoring_evidence",
            "uri": "authoring.json",
            "checksum": _checksum("authoring"),
            "size_bytes": 100,
            "reason": "provider_authoring_audit",
        },
        "stop_reason": (
            "candidate_verified"
            if classification
            in {"valid_structurally_distinct", "equivalent_leak"}
            else (
                "provider_failed"
                if classification == "provider_failed"
                else "static_failed"
            )
        ),
        "classification": classification,
        "candidate": candidate,
        "sibling_semantic_exclusion_assessment": sibling,
        "independent_equivalence": (
            "parameter_only_delta" if classification == "equivalent_leak" else None
        ),
        "independent_comparison_error": None,
        "provider_usage": None,
        "provider_response_receipt": (
            {"schema_version": "true_rl_agent.provider_response_receipt.v3"}
            if classification
            in {"valid_structurally_distinct", "equivalent_leak"}
            else None
        ),
        "authority": {
            "train_jobs": 0,
            "eval_jobs": 0,
            "sealed_queries": 0,
            "primary_ledger_writes": 0,
            "current_best_transitions": 0,
            "gpu_jobs": 0,
        },
    }
    return {**body, "case_result_hash": canonical_json_hash(body)}


class _FakeTransport:
    transport_policy_id = "test.transport.v1"
    max_response_bytes = 512 * 1024

    def __init__(self) -> None:
        self.calls = 0

    def post_json(self, **kwargs):
        self.calls += 1
        return {"id": "response-1", "choices": []}


class _ReasoningOnlyTransport:
    transport_policy_id = "test.transport.v1"
    max_response_bytes = 512 * 1024

    def __init__(self) -> None:
        self.calls = 0

    def post_json(self, **kwargs):
        self.calls += 1
        return {
            "id": "reasoning-only-response",
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "reasoning_content": "private draft only",
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 10,
                "completion_tokens_details": {"reasoning_tokens": 10},
                "total_tokens": 110,
            },
        }


class _FailIfCalledTransport(_ReasoningOnlyTransport):
    def post_json(self, **kwargs):
        self.calls += 1
        raise AssertionError("terminal replay attempted a second provider request")


class _SingleCandidateProvider:
    def __init__(self, candidate: CandidatePackageRef) -> None:
        self.candidate = candidate

    def next_candidate(self, *, prompt_payload):
        return self.candidate


class _ExclusionFakeVerifier(FakeVerifier):
    def __init__(self, round_index: int = 1) -> None:
        self.round_index = round_index

    def authoring_prompt_rules(self):
        rules = dict(super().authoring_prompt_rules())
        rules["single_component_semantic_exclusion"] = _exclusion(
            self.round_index
        )
        return rules


class StairPairedReadyProbeTests(unittest.TestCase):
    def test_create_once_plan_round_trips_with_frozen_provider_and_zero_authority(
        self,
    ) -> None:
        cases = _cases()
        plan = _build(cases)
        self.assertEqual(plan["case_count"], 15)
        self.assertEqual(
            [case["round_index"] for case in plan["cases"]],
            list(range(1, 16)),
        )
        self.assertEqual(plan["provider"]["timeout_seconds"], 300)
        self.assertEqual(plan["provider"]["reasoning_effort"], "max")
        self.assertEqual(plan["provider"]["temperature_policy"], "omitted")
        self.assertEqual(plan["budget"]["reserved_output_tokens"], 491_520)
        self.assertEqual(
            plan["budget"]["reserved_output_tokens"],
            STAIR_PAIRED_READY_PROBE_TOTAL_OUTPUT_TOKENS,
        )
        self.assertEqual(
            plan["money_governor"],
            STAIR_PAIRED_READY_PROBE_MONEY_GOVERNOR,
        )
        self.assertEqual(set(plan["authority"].values()), {0})
        for field_name, expected in _SOURCE_IDENTITY.items():
            self.assertEqual(plan[field_name], expected)
        self.assertEqual(
            stair_paired_ready_probe_plan_hash(plan),
            plan["plan_hash"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "paired_ready_probe_plan.json"
            first = _create(path, cases)
            second = _create(path, cases)
            self.assertEqual(first, plan)
            self.assertEqual(second, plan)
            self.assertEqual(load_stair_paired_ready_probe_plan(path), plan)
            self.assertEqual(
                path.read_bytes(),
                canonical_json_bytes(plan, trailing_newline=True),
            )

            drifted_cases = _cases()
            drifted_cases[0]["source_round_id"] = "different-round-id"
            with self.assertRaises(FileExistsError):
                _create(path, drifted_cases)

    def test_plan_rejects_missing_duplicate_or_non_unique_requests(self) -> None:
        cases = _cases()
        with self.assertRaisesRegex(ValueError, "exactly 15"):
            _build(cases[:-1])

        duplicate_round = _cases()
        duplicate_round[14]["round_index"] = 14
        duplicate_round[14]["prefix_ledger_count"] = 13
        with self.assertRaisesRegex(ValueError, "ordered rounds 1..15"):
            _build(duplicate_round)

        duplicate_request = _cases()
        duplicate_request[14]["provider_request"] = copy.deepcopy(
            duplicate_request[13]["provider_request"]
        )
        duplicate_request[14]["production_exclusion_projection"] = copy.deepcopy(
            duplicate_request[13]["production_exclusion_projection"]
        )
        with self.assertRaisesRegex(ValueError, "requests must be unique"):
            _build(duplicate_request)

    def test_plan_rejects_tampering_and_non_production_request_fields(self) -> None:
        plan = _build(_cases())
        tampered = copy.deepcopy(plan)
        tampered["cases"][0]["provider_request"]["max_tokens"] = 16_384
        with self.assertRaisesRegex(ValueError, "max_tokens mismatch"):
            validate_stair_paired_ready_probe_plan(tampered)

        hash_tampered = copy.deepcopy(plan)
        hash_tampered["plan_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "plan hash mismatch"):
            validate_stair_paired_ready_probe_plan(hash_tampered)

        identity_replacements = {
            "source_commit": "e" * 40,
            "source_archive_checksum": "sha256:" + "f" * 64,
            "source_ledger_checksum": "sha256:" + "1" * 64,
            "source_session_genesis_hash": "2" * 64,
        }
        for field_name, replacement in identity_replacements.items():
            with self.subTest(identity_field=field_name):
                identity_tampered = copy.deepcopy(plan)
                identity_tampered[field_name] = replacement
                with self.assertRaisesRegex(ValueError, "plan hash mismatch"):
                    validate_stair_paired_ready_probe_plan(identity_tampered)

        temperature_request = _cases()
        temperature_request[0]["provider_request"]["temperature"] = 0.2
        with self.assertRaisesRegex(ValueError, "request fields mismatch"):
            _build(temperature_request)

    def test_plan_enforces_request_byte_preflight_and_exact_exclusion_binding(
        self,
    ) -> None:
        oversized = _cases()
        oversized[0]["provider_request"]["messages"][0]["content"] = "x" * 300_000
        with self.assertRaisesRegex(ValueError, "exceeds 300000 bytes"):
            _build(oversized)

        wrong_exclusion = _cases()
        wrong_exclusion[0]["production_exclusion_projection"] = _exclusion(15)
        with self.assertRaisesRegex(ValueError, "exclusion exactly once"):
            _build(wrong_exclusion)

    def test_runtime_provider_request_and_transport_match_frozen_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = fixed_stair_paired_ready_provider_config(
                output_dir=root / "candidates",
                env_file=None,
            )
            self.assertEqual(config.model, "deepseek-v4-flash")
            self.assertEqual(config.reasoning_mode, "enabled")
            self.assertEqual(config.reasoning_effort, "max")
            self.assertEqual(config.max_tokens, 32_768)
            self.assertEqual(config.timeout_s, 300)

            expected = _request(1, _exclusion(1))
            delegate = _FakeTransport()
            transport = RecordingProbeTransport(
                delegate=delegate,
                request_path=root / "request.json",
                response_path=root / "response.json",
                expected_request=expected,
            )
            response = transport.post_json(
                url="https://api.deepseek.com/chat/completions",
                payload=expected,
                api_key="not-recorded",
                timeout_s=300,
            )
            self.assertEqual(response["id"], "response-1")
            self.assertEqual(delegate.calls, 1)
            self.assertNotIn("temperature", expected)
            with self.assertRaisesRegex(RuntimeError, "second provider request"):
                transport.post_json(
                    url="https://api.deepseek.com/chat/completions",
                    payload=expected,
                    api_key="not-recorded",
                    timeout_s=300,
                )
            replay_delegate = _FakeTransport()
            replay = RecordingProbeTransport(
                delegate=replay_delegate,
                request_path=root / "request.json",
                response_path=root / "response.json",
                expected_request=expected,
            )
            self.assertEqual(replay.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, "second provider request"):
                replay.post_json(
                    url="https://api.deepseek.com/chat/completions",
                    payload=expected,
                    api_key="not-recorded",
                    timeout_s=300,
                )
            self.assertEqual(replay_delegate.calls, 0)

    def test_terminal_result_uses_private_authoring_evidence_projection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            case_dir = output_dir / "cases" / "round-001"
            case_dir.mkdir(parents=True)
            env_file = Path(tmp) / "provider.env"
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            verifier = _ExclusionFakeVerifier()
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                initial_current_best=make_bootstrap_current_best(),
                session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            )
            prompt_build = build_authoring_prompt(
                context_builder=context_builder,
                records=[],
                repair_feedback=None,
                latest_failed_candidate=None,
                candidate_verifier_prompt_rules=verifier.authoring_prompt_rules(),
            )
            config = fixed_stair_paired_ready_provider_config(
                output_dir=case_dir / "materialized_candidates",
                env_file=env_file,
            )
            expected_request = OpenAICompatibleCandidateProvider(
                config
            )._chat_completion_request(prompt_build.provider_payload)
            cases = _cases()
            cases[0]["provider_request"] = expected_request
            plan_path = output_dir / "probe_plan.json"
            plan = _create(plan_path, cases)

            first_delegate = _ReasoningOnlyTransport()
            first_transport = RecordingProbeTransport(
                delegate=first_delegate,
                request_path=case_dir / "provider_request.json",
                response_path=case_dir / "provider_response.json",
                expected_request=expected_request,
            )
            first = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=OpenAICompatibleCandidateProvider(
                    config,
                    transport=first_transport,
                ),
                candidate_admission_policy=verifier,
                round_dir=case_dir / "authoring",
                max_attempts=1,
                max_source_requests_per_round=0,
            ).run_round(
                round_id="p6a2-round-001",
                records=[],
            )
            self.assertEqual(first.stop_reason, "provider_failed")
            self.assertEqual(first.provider_failure["category"], "json_parse_error")
            self.assertEqual(first_delegate.calls, 1)
            self.assertEqual(first_transport.call_count, 1)
            self.assertFalse(
                hasattr(first.authoring_evidence_ref, "to_public_dict")
            )
            request_bytes = (case_dir / "provider_request.json").read_bytes()
            response_bytes = (case_dir / "provider_response.json").read_bytes()
            (output_dir / "run.log").write_text(
                "AttributeError: 'AuthoringEvidenceRef' object has no attribute "
                "'to_public_dict'\n",
                encoding="utf-8",
            )
            (output_dir / "run.exitcode").write_text("1\n", encoding="utf-8")
            correction_identity = {
                "plan_source_commit": plan["source_commit"],
                "runtime_source_commit": "e" * 40,
                "runtime_parent_commit": plan["source_commit"],
                "changed_paths": list(
                    STAIR_PAIRED_READY_PROBE_RUNTIME_CORRECTION_CHANGED_PATHS
                ),
            }
            correction = _create_or_load_runtime_correction(
                output_dir=output_dir,
                plan_path=plan_path,
                plan=plan,
                runtime_correction_identity=correction_identity,
            )

            replay = _project_runtime_correction_trigger_result(
                case_dir=case_dir,
                planned=plan["cases"][0],
                correction=correction,
            )
            self.assertEqual(replay.stop_reason, "provider_failed")
            self.assertEqual(
                replay.authoring_evidence_ref.checksum,
                first.authoring_evidence_ref.checksum,
            )
            self.assertEqual(
                len(
                    list(
                        (case_dir / "authoring" / "authoring_evidence").glob(
                            "*.json"
                        )
                    )
                ),
                1,
            )
            self.assertEqual(
                (case_dir / "provider_request.json").read_bytes(),
                request_bytes,
            )
            self.assertEqual(
                (case_dir / "provider_response.json").read_bytes(),
                response_bytes,
            )
            receipt = _runtime_case_result(
                plan_hash=plan["plan_hash"],
                planned=plan["cases"][0],
                prepared=SimpleNamespace(exclusion=None),
                authoring_result=replay,
                case_dir=case_dir,
            )
            self.assertEqual(
                receipt["authoring_evidence_ref"],
                replay.authoring_evidence_ref.to_private_dict(),
            )
            self.assertEqual(receipt["classification"], "provider_failed")
            with self.assertRaises((FileNotFoundError, ValueError)):
                _project_existing_terminal_authoring_result(
                    case_dir=case_dir,
                    planned=plan["cases"][0],
                    authoring_evidence_ref={
                        **receipt["authoring_evidence_ref"],
                        "uri": str(case_dir / "missing_authoring_evidence.json"),
                    },
                )
            self.assertEqual(
                _create_or_load_runtime_correction(
                    output_dir=output_dir,
                    plan_path=plan_path,
                    plan=plan,
                    runtime_correction_identity=correction_identity,
                ),
                correction,
            )
            corrected_results = [receipt] + [
                _case_result(
                    plan,
                    round_index,
                    (
                        "valid_structurally_distinct"
                        if round_index <= 11
                        else "authoring_invalid"
                    ),
                )
                for round_index in range(2, 16)
            ]
            aggregate = build_stair_paired_ready_probe_result(
                plan=plan,
                case_results=corrected_results,
                runtime_correction=correction,
            )
            self.assertIs(aggregate["go"], True)
            self.assertEqual(
                aggregate["provider_execution_commit_counts"],
                {plan["source_commit"]: 1, "e" * 40: 14},
            )
            self.assertEqual(
                sum(aggregate["provider_execution_commit_counts"].values()),
                15,
            )
            with self.assertRaisesRegex(TypeError, "verified correction"):
                build_stair_paired_ready_probe_result(
                    plan=plan,
                    case_results=corrected_results,
                    runtime_correction=correction.payload,
                )
            evidence_path = Path(replay.authoring_evidence_ref.path)
            tampered_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            selected_attempt = next(
                item
                for item in tampered_evidence["attempts"]
                if item["attempt_id"] == tampered_evidence["selected_attempt_id"]
            )
            selected_attempt["provider_attempt"]["candidate_id"] = "forged-candidate"
            selected_attempt["provider_attempt"]["candidate_changed_components"] = [
                "forged_component"
            ]
            tampered_bytes = canonical_json_bytes(
                tampered_evidence,
                trailing_newline=True,
            )
            evidence_path.write_bytes(tampered_bytes)
            tampered_ref = {
                **replay.authoring_evidence_ref.to_private_dict(),
                "checksum": "sha256:" + hashlib.sha256(tampered_bytes).hexdigest(),
                "size_bytes": len(tampered_bytes),
            }
            with self.assertRaisesRegex(
                ValueError,
                "selected candidate identity mismatch",
            ):
                _project_existing_terminal_authoring_result(
                    case_dir=case_dir,
                    planned=plan["cases"][0],
                    authoring_evidence_ref=tampered_ref,
                )

    def test_terminal_nontrigger_case_replays_without_second_provider_request(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp) / "output"
            case_dir = output_dir / "cases" / "round-002"
            case_dir.mkdir(parents=True)
            env_file = Path(tmp) / "provider.env"
            env_file.write_text("DEEPSEEK_API_KEY=test-key\n", encoding="utf-8")
            verifier = _ExclusionFakeVerifier(round_index=2)
            context_builder = PublicContextBuilder(
                task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                initial_current_best=make_bootstrap_current_best(),
                session_genesis_hash=TEST_SESSION_GENESIS_HASH,
            )
            prompt_build = build_authoring_prompt(
                context_builder=context_builder,
                records=[],
                repair_feedback=None,
                latest_failed_candidate=None,
                candidate_verifier_prompt_rules=verifier.authoring_prompt_rules(),
            )
            config = fixed_stair_paired_ready_provider_config(
                output_dir=case_dir / "materialized_candidates",
                env_file=env_file,
            )
            request = OpenAICompatibleCandidateProvider(
                config
            )._chat_completion_request(prompt_build.provider_payload)
            cases = _cases()
            cases[1]["provider_request"] = request
            plan = _build(cases)
            planned = plan["cases"][1]

            first_delegate = _ReasoningOnlyTransport()
            first_transport = RecordingProbeTransport(
                delegate=first_delegate,
                request_path=case_dir / "provider_request.json",
                response_path=case_dir / "provider_response.json",
                expected_request=request,
            )
            first = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=OpenAICompatibleCandidateProvider(
                    config,
                    transport=first_transport,
                ),
                candidate_admission_policy=verifier,
                round_dir=case_dir / "authoring",
                max_attempts=1,
                max_source_requests_per_round=0,
            ).run_round(round_id="p6a2-round-002", records=[])
            self.assertEqual(first.stop_reason, "provider_failed")
            self.assertEqual(first_delegate.calls, 1)
            request_bytes = (case_dir / "provider_request.json").read_bytes()
            response_bytes = (case_dir / "provider_response.json").read_bytes()

            replay_delegate = _FailIfCalledTransport()
            replay_transport = RecordingProbeTransport(
                delegate=replay_delegate,
                request_path=case_dir / "provider_request.json",
                response_path=case_dir / "provider_response.json",
                expected_request=request,
            )
            replay = CandidateAuthoringLoop(
                context_builder=context_builder,
                provider=OpenAICompatibleCandidateProvider(
                    config,
                    transport=replay_transport,
                ),
                candidate_admission_policy=verifier,
                round_dir=case_dir / "authoring",
                max_attempts=1,
                max_source_requests_per_round=0,
            ).run_round(round_id="p6a2-round-002", records=[])
            self.assertEqual(replay.stop_reason, "provider_failed")
            self.assertEqual(replay_delegate.calls, 0)
            self.assertEqual(replay_transport.call_count, 1)
            replay = _project_existing_terminal_authoring_result(
                case_dir=case_dir,
                planned=planned,
                authoring_evidence_ref=replay.authoring_evidence_ref.to_private_dict(),
            )
            receipt = _runtime_case_result(
                plan_hash=plan["plan_hash"],
                planned=planned,
                prepared=SimpleNamespace(exclusion=None),
                authoring_result=replay,
                case_dir=case_dir,
            )
            result_path = case_dir / "case_result.json"
            result_path.write_bytes(
                canonical_json_bytes(receipt, trailing_newline=True)
            )
            self.assertEqual(
                _verify_existing_case_result(
                    result_path=result_path,
                    plan_hash=plan["plan_hash"],
                    planned=planned,
                    prepared=SimpleNamespace(exclusion=None),
                    case_dir=case_dir,
                ),
                receipt,
            )
            self.assertEqual(
                (case_dir / "provider_request.json").read_bytes(),
                request_bytes,
            )
            self.assertEqual(
                (case_dir / "provider_response.json").read_bytes(),
                response_bytes,
            )

    def test_existing_verified_candidate_binds_source_bytes_to_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "cases" / "round-003"
            candidate_dir = case_dir / "candidate-003"
            source_path = candidate_dir / "training_code" / "components.py"
            source_path.parent.mkdir(parents=True)
            (candidate_dir / "candidate_manifest.yaml").write_text(
                "\n".join(
                    (
                        "schema_version: true_rl_agent.candidate.v1",
                        "candidate_id: candidate-003",
                        "entrypoint: training_code.components:build_training_components",
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            source_path.write_text(
                "def build_training_components():\n    return None\n",
                encoding="utf-8",
            )
            for name, content in (
                ("rationale.md", "Candidate rationale.\n"),
                ("operation_diff.json", "{}\n"),
                ("diagnostics.json", "{}\n"),
            ):
                (candidate_dir / name).write_text(content, encoding="utf-8")
            candidate = CandidatePackageRef(
                candidate_id="candidate-003",
                path=str(candidate_dir),
                change_set=CandidateChangeSet(frozenset({"reward_callable"})),
                metadata={"fake_public_score": 0.3, "verifier_status": "pass"},
            )
            verifier = _ExclusionFakeVerifier(round_index=3)
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    initial_current_best=make_bootstrap_current_best(),
                    session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                ),
                provider=_SingleCandidateProvider(candidate),
                candidate_admission_policy=verifier,
                round_dir=case_dir / "authoring",
                max_attempts=1,
                max_source_requests_per_round=0,
            ).run_round(round_id="p6a2-round-003", records=[])
            self.assertEqual(result.stop_reason, "candidate_verified")
            planned = {
                "case_id": "round-003",
                "expected_provider_prompt_hash": canonical_json_hash(
                    verify_prompt_payload(result.prompt_payload_ref)
                ),
            }
            self.assertEqual(
                _project_existing_terminal_authoring_result(
                    case_dir=case_dir,
                    planned=planned,
                    authoring_evidence_ref=result.authoring_evidence_ref.to_private_dict(),
                ).candidate,
                candidate,
            )
            source_path.write_text(
                "def build_training_components():\n    return 1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "selected candidate identity mismatch",
            ):
                _project_existing_terminal_authoring_result(
                    case_dir=case_dir,
                    planned=planned,
                    authoring_evidence_ref=result.authoring_evidence_ref.to_private_dict(),
                )

    def test_aggregate_gate_requires_ten_distinct_zero_leaks_and_plan_binding(
        self,
    ) -> None:
        plan = _build(_cases())
        passing = [
            _case_result(
                plan,
                round_index,
                (
                    "valid_structurally_distinct"
                    if round_index <= 10
                    else "authoring_invalid"
                ),
            )
            for round_index in range(1, 16)
        ]
        aggregate = build_stair_paired_ready_probe_result(
            plan=plan,
            case_results=passing,
        )
        self.assertIs(aggregate["go"], True)
        self.assertEqual(aggregate["provider_request_count"], 15)

        leaking = copy.deepcopy(passing)
        leaking[0] = _case_result(plan, 1, "equivalent_leak")
        self.assertIs(
            build_stair_paired_ready_probe_result(
                plan=plan,
                case_results=leaking,
            )["go"],
            False,
        )

        drifted = copy.deepcopy(passing)
        drifted[0]["source_round_id"] = "wrong-source-round"
        drifted_body = dict(drifted[0])
        drifted_body.pop("case_result_hash")
        drifted[0]["case_result_hash"] = canonical_json_hash(drifted_body)
        with self.assertRaisesRegex(ValueError, "differs from its frozen plan"):
            build_stair_paired_ready_probe_result(
                plan=plan,
                case_results=drifted,
            )

        impossible = copy.deepcopy(passing)
        impossible[0]["candidate"] = None
        impossible_body = dict(impossible[0])
        impossible_body.pop("case_result_hash")
        impossible[0]["case_result_hash"] = canonical_json_hash(impossible_body)
        with self.assertRaisesRegex(ValueError, "disagrees with its evidence"):
            build_stair_paired_ready_probe_result(
                plan=plan,
                case_results=impossible,
            )

if __name__ == "__main__":
    unittest.main()
