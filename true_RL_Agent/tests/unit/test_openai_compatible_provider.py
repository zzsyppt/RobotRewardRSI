from __future__ import annotations

import json
import io
import socket
import ssl
import sys
import tempfile
import threading
import unittest
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.authoring_response import (  # noqa: E402
    CandidateDraftResponse,
    SourceRequestResponse,
)
from true_rl_agent.agent_core.authoring_loop import (  # noqa: E402
    CandidateAuthoringLoop,
)
from true_rl_agent.agent_core.openai_compatible_provider import (  # noqa: E402
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
    UrllibJsonTransport,
    _safe_error,
    load_env_file,
)
from true_rl_agent.agent_core.provider_contract import (  # noqa: E402
    PROMPT_ADVERTISED_DECLARED_COMPONENTS,
)
from true_rl_agent.agent_core.provider_errors import (
    ProviderCandidateError,  # noqa: E402
    build_provider_repair_feedback,  # noqa: E402
    normalize_provider_receipt,  # noqa: E402
)
from true_rl_agent.agent_core.provider_limits import (  # noqa: E402
    PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND,
)
from tests.fakes.decision_records import (  # noqa: E402
    TEST_ACCEPTANCE_PROTOCOL,
    TEST_SESSION_GENESIS_HASH,
    make_bootstrap_current_best,
)
from tests.fakes.fake_components import FakeTaskPlugin, FakeVerifier  # noqa: E402
from true_rl_agent.candidates.package_contract import (  # noqa: E402
    CANDIDATE_PACKAGE_FILE_MAX_BYTES,
)
from true_rl_agent.candidates.prompt_projection import (
    build_prompt_payload,  # noqa: E402
)
from true_rl_agent.candidates.verifier import StaticCandidateVerifier  # noqa: E402
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402


class _FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(self, *, url: str, payload: dict[str, Any], api_key: str, timeout_s: int) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "api_key": api_key, "timeout_s": timeout_s})
        return self.response


class _SequenceTransport:
    transport_policy_id = "test.transport.v1"
    max_response_bytes = 512 * 1024

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post_json(
        self,
        *,
        url: str,
        payload: dict[str, Any],
        api_key: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "url": url,
                "payload": payload,
                "api_key": api_key,
                "timeout_s": timeout_s,
            }
        )
        return self.responses[len(self.calls) - 1]


class _FakeHTTPResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> _FakeHTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        return self.payload if size < 0 else self.payload[:size]


class _FakeOpener:
    def __init__(
        self,
        *,
        response: _FakeHTTPResponse | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.calls = 0
        self.requests: list[Any] = []

    def open(self, *_args: object, **_kwargs: object) -> _FakeHTTPResponse:
        self.calls += 1
        self.requests.append(_args[0])
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class OpenAICompatibleProviderTests(unittest.TestCase):
    def _response(self) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "candidate_id": "upright_reward_v1",
                                "declared_changed_components": ["reward_callable"],
                                "components_py": "\n".join(
                                    [
                                        "from true_rl_agent.candidates.components import RewardTerm, TrainingComponents",
                                        "",
                                        "",
                                        "def upright_reward(inputs):",
                                        "    return inputs.base_height - 0.25 * (inputs.roll * inputs.roll + inputs.pitch * inputs.pitch)",
                                        "",
                                        "",
                                        "def build_training_components():",
                                        "    return TrainingComponents(",
                                        "        name='upright_reward_v1',",
                                        "        reward_terms=[RewardTerm(name='upright_reward', weight=1.0, fn=upright_reward)],",
                                        "        rationale='Reward upright height using public inputs.',",
                                        "    )",
                                    ]
                                ),
                            }
                        )
                    }
                }
            ]
        }

    def test_provider_config_preserves_legacy_positional_field_order(self) -> None:
        config = OpenAICompatibleProviderConfig(
            "deepseek-v4-pro",
            Path("candidate-output"),
            "https://api.deepseek.com/v1",
            "/chat/completions",
            "DEEPSEEK_API_KEY",
            "test-key",
            Path("provider.env"),
            8192,
            240,
            0.1,
        )

        self.assertEqual(config.max_tokens, 8192)
        self.assertEqual(config.timeout_s, 240)
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.reasoning_mode, "provider_default")
        self.assertFalse(config.empty_final_reasoning_retry)

    def test_empty_final_reasoning_retry_config_is_explicit_and_adaptive_only(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "must be boolean"):
            OpenAICompatibleProviderConfig(
                model="deepseek-v4-flash",
                output_dir=Path("candidate-output"),
                empty_final_reasoning_retry="enabled",  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "requires adaptive reasoning"):
            OpenAICompatibleProviderConfig(
                model="deepseek-v4-flash",
                output_dir=Path("candidate-output"),
                reasoning_mode="enabled",
                empty_final_reasoning_retry=True,
            )

    def test_provider_materializes_candidate_package_from_json_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    base_url="https://api.deepseek.com",
                    api_key="test-key",
                ),
                transport=transport,
            )

            candidate = provider.next_candidate(
                prompt_payload=build_prompt_payload(public_context={"previous_rounds": 0, "public_only": True})
            )
            verification = StaticCandidateVerifier().verify(candidate)
            operation_diff = json.loads((Path(candidate.path) / "operation_diff.json").read_text(encoding="utf-8"))
            diagnostics = json.loads((Path(candidate.path) / "diagnostics.json").read_text(encoding="utf-8"))
            rationale = (Path(candidate.path) / "rationale.md").read_text(
                encoding="utf-8"
            )

        self.assertEqual(candidate.candidate_id, "upright_reward_v1")
        self.assertEqual(candidate.change_set.changed_components, frozenset({"reward_callable"}))
        self.assertEqual(verification.status, "pass")
        self.assertEqual(operation_diff["materialized_by"], "openai_compatible_candidate_materializer")
        self.assertEqual(operation_diff["declared_changed_components"], ["reward_callable"])
        self.assertEqual(diagnostics["status"], "provider_materialized")
        self.assertEqual(diagnostics["materialized_by"], "openai_compatible_candidate_materializer")
        self.assertEqual(diagnostics["diagnostic_items"], [])
        self.assertEqual(
            rationale,
            "Reward upright height using public inputs.",
        )
        self.assertEqual(verification.details["derived_change_set"], ["reward_callable"])
        self.assertEqual(transport.calls[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(transport.calls[0]["payload"]["response_format"], {"type": "json_object"})
        self.assertNotIn("thinking", transport.calls[0]["payload"])
        request_text = json.dumps(transport.calls[0]["payload"], sort_keys=True)
        self.assertIn("JSON", request_text)
        self.assertIn("component_contract", request_text)
        self.assertIn("def build_training_components() -> TrainingComponents:", request_text)
        self.assertIn("RewardTerm(name: str, weight: float", request_text)
        self.assertIn("task_owned_contract_not_provided", request_text)
        self.assertNotIn("base_pos_xy", request_text)
        self.assertIn("candidate_package_only", request_text)
        self.assertIn("no_prose_outside_json", request_text)
        self.assertIn("diagnostics.json", request_text)
        self.assertIn("operation_diff.json", request_text)
        self.assertIn("runner_patch", request_text)
        self.assertIn("one_of", request_text)
        self.assertIn("source_request", request_text)
        self.assertNotIn("test-key", request_text)

    def test_http_transport_rejects_oversized_response_before_json_parse(self) -> None:
        transport = UrllibJsonTransport(max_response_bytes=32)
        opener = _FakeOpener(
            response=_FakeHTTPResponse(b"{" + b"x" * 64 + b"}")
        )
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(ProviderCandidateError) as caught:
                transport.post_json(
                    url="https://api.deepseek.com/chat/completions",
                    payload={"model": "deepseek-v4-pro"},
                    api_key="test-key",
                    timeout_s=1,
                )

        self.assertEqual(caught.exception.category, "runtime_error")
        self.assertEqual(caught.exception.details["failure_kind"], "response_too_large")
        self.assertFalse(caught.exception.details["retryable"])
        self.assertEqual(opener.calls, 1)

    def test_provider_url_policy_is_enforced_only_by_builtin_transport(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="test-model",
                    output_dir=Path(tmp),
                    base_url="memory://provider",
                    endpoint="candidate",
                    api_key="test-key",
                ),
                transport=transport,
            )
            provider.next_candidate(
                prompt_payload=build_prompt_payload(
                    public_context={"public_only": True}
                )
            )

        self.assertEqual(transport.calls[0]["url"], "memory://provider/candidate")

        rejected_by_builtin_transport = (
            "file:///tmp/provider.json",
            "data:application/json,{}",
            "https://user:pass@provider.example/v1/chat/completions",
            "https://provider.example/v1/chat/completions#fragment",
        )
        for url in rejected_by_builtin_transport:
            with self.subTest(url=url):
                with self.assertRaises(ProviderCandidateError) as caught:
                    UrllibJsonTransport().post_json(
                        url=url,
                        payload={"model": "test-model"},
                        api_key="dummy-test-key",
                        timeout_s=1,
                    )
                self.assertEqual(
                    caught.exception.details["failure_kind"],
                    "url_policy_error",
                )

        config = OpenAICompatibleProviderConfig(
            model="test-model",
            output_dir=Path("candidate-output"),
            base_url="http://provider.example/v1",
        )
        transport = UrllibJsonTransport()
        with self.assertRaises(ProviderCandidateError) as caught:
            transport.post_json(
                url=config.base_url + "/chat/completions",
                payload={"model": "test-model"},
                api_key="dummy-test-key",
                timeout_s=1,
            )
        self.assertEqual(caught.exception.details["failure_kind"], "url_policy_error")
        self.assertFalse(caught.exception.details["retryable"])

    def test_http_transport_preserves_azure_style_api_version_query(self) -> None:
        url = (
            "https://resource.openai.azure.com/openai/deployments/model/"
            "chat/completions?api-version=2024-02-01"
        )
        opener = _FakeOpener(response=_FakeHTTPResponse(b"{}"))

        with patch("urllib.request.build_opener", return_value=opener):
            result = UrllibJsonTransport().post_json(
                url=url,
                payload={"model": "test-model"},
                api_key="dummy-test-key",
                timeout_s=1,
            )

        self.assertEqual(result, {})
        self.assertEqual(opener.calls, 1)
        request = opener.requests[0]
        self.assertEqual(request.full_url, url)
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(
            json.loads(request.data.decode("utf-8")),
            {"model": "test-model"},
        )
        self.assertEqual(
            request.get_header("Authorization"),
            "Bearer dummy-test-key",
        )
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertIsNone(request.get_header("Accept"))

    def test_builtin_transport_policy_identity_cannot_be_overridden(self) -> None:
        with self.assertRaises(TypeError):
            UrllibJsonTransport(transport_policy_id="custom-policy")  # type: ignore[call-arg]

    def test_http_transport_normalizes_network_and_timeout_failures_once(self) -> None:
        transport = UrllibJsonTransport()
        cases = (
            (
                urllib.error.URLError(ConnectionRefusedError("private endpoint")),
                "network_error",
            ),
            (urllib.error.URLError(socket.timeout("private timeout")), "timeout"),
            (socket.timeout("private timeout"), "timeout"),
        )
        for error, expected_kind in cases:
            with self.subTest(expected_kind=expected_kind):
                opener = _FakeOpener(error=error)
                with patch("urllib.request.build_opener", return_value=opener):
                    with self.assertRaises(ProviderCandidateError) as caught:
                        transport.post_json(
                            url="https://api.deepseek.com/chat/completions",
                            payload={"model": "deepseek-v4-pro"},
                            api_key="dummy-test-key",
                            timeout_s=1,
                        )
                self.assertEqual(caught.exception.category, "runtime_error")
                self.assertEqual(caught.exception.details["failure_kind"], expected_kind)
                self.assertTrue(caught.exception.details["retryable"])
                self.assertNotIn("private", str(caught.exception))
                self.assertEqual(opener.calls, 1)

    def test_http_transport_rejects_unsafe_credential_before_header_encoding(self) -> None:
        secret = "sk-secret-line1\nsk-secret-line2"
        with self.assertRaises(ProviderCandidateError) as caught:
            UrllibJsonTransport().post_json(
                url="https://api.deepseek.com/chat/completions",
                payload={"model": "deepseek-v4-pro"},
                api_key=secret,
                timeout_s=1,
            )

        self.assertEqual(
            caught.exception.details["failure_kind"],
            "credential_format_error",
        )
        self.assertFalse(caught.exception.details["retryable"])
        self.assertNotIn("secret-line", str(caught.exception))

    def test_http_transport_rejects_del_and_c1_credentials_before_request(self) -> None:
        for codepoint in range(0x7F, 0xA0):
            control_character = chr(codepoint)
            secret = f"sk-credential-leak-marker{control_character}tail"
            with (
                self.subTest(codepoint=codepoint),
                patch("urllib.request.build_opener") as build_opener,
                self.assertRaises(ProviderCandidateError) as caught,
            ):
                UrllibJsonTransport().post_json(
                    url="https://api.deepseek.com/chat/completions",
                    payload={"model": "deepseek-v4-pro"},
                    api_key=secret,
                    timeout_s=1,
                )

            build_opener.assert_not_called()
            self.assertEqual(caught.exception.category, "runtime_error")
            self.assertEqual(
                caught.exception.details["failure_kind"],
                "credential_format_error",
            )
            self.assertFalse(caught.exception.details["retryable"])
            self.assertNotIn("credential-leak-marker", str(caught.exception))
            self.assertNotIn(
                "credential-leak-marker",
                str(caught.exception.to_public_dict()),
            )

    def test_http_transport_treats_tls_certificate_failure_as_nonretryable(self) -> None:
        opener = _FakeOpener(
            error=urllib.error.URLError(
                ssl.SSLCertVerificationError("private certificate detail")
            )
        )
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(ProviderCandidateError) as caught:
                UrllibJsonTransport().post_json(
                    url="https://api.deepseek.com/chat/completions",
                    payload={"model": "deepseek-v4-pro"},
                    api_key="dummy-test-key",
                    timeout_s=1,
                )

        self.assertEqual(
            caught.exception.details["failure_kind"],
            "tls_certificate_error",
        )
        self.assertFalse(caught.exception.details["retryable"])
        self.assertNotIn("private", str(caught.exception))

    def test_http_transport_rejects_redirect_without_exposing_error_body(self) -> None:
        transport = UrllibJsonTransport()
        error = urllib.error.HTTPError(
            "https://api.deepseek.com/chat/completions",
            302,
            "redirect",
            {"Location": "https://attacker.example/collect"},
            None,
        )
        opener = _FakeOpener(error=error)
        with patch("urllib.request.build_opener", return_value=opener):
            with self.assertRaises(ProviderCandidateError) as caught:
                transport.post_json(
                    url="https://api.deepseek.com/chat/completions",
                    payload={"model": "deepseek-v4-pro"},
                    api_key="dummy-test-key",
                    timeout_s=1,
                )

        self.assertEqual(caught.exception.details["failure_kind"], "redirect_rejected")
        self.assertEqual(caught.exception.details["http_status"], 302)
        self.assertFalse(caught.exception.details["retryable"])
        self.assertNotIn("attacker", str(caught.exception))
        self.assertEqual(opener.calls, 1)

    def test_http_transport_never_forwards_authorization_across_redirect(self) -> None:
        received_authorization: list[str | None] = []

        class DestinationHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                received_authorization.append(self.headers.get("Authorization"))
                self.send_response(200)
                self.end_headers()

            def log_message(self, *_args: object) -> None:
                return None

        destination = ThreadingHTTPServer(("127.0.0.1", 0), DestinationHandler)
        destination_thread = threading.Thread(
            target=destination.serve_forever,
            daemon=True,
        )
        destination_thread.start()

        destination_url = (
            f"http://127.0.0.1:{destination.server_address[1]}/collect"
        )

        class RedirectHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.send_response(302)
                self.send_header("Location", destination_url)
                self.end_headers()

            def log_message(self, *_args: object) -> None:
                return None

        origin = ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
        origin_thread = threading.Thread(target=origin.serve_forever, daemon=True)
        origin_thread.start()
        try:
            with self.assertRaises(ProviderCandidateError) as caught:
                UrllibJsonTransport().post_json(
                    url=(
                        f"http://127.0.0.1:{origin.server_address[1]}"
                        "/chat/completions"
                    ),
                    payload={"model": "test-model"},
                    api_key="dummy-never-forward",
                    timeout_s=2,
                )
        finally:
            origin.shutdown()
            origin.server_close()
            destination.shutdown()
            destination.server_close()
            origin_thread.join(timeout=2)
            destination_thread.join(timeout=2)

        self.assertEqual(caught.exception.details["failure_kind"], "redirect_rejected")
        self.assertEqual(received_authorization, [])

    def test_http_transport_classifies_http_status_without_body_content(self) -> None:
        transport = UrllibJsonTransport()
        for status, retryable in (
            (401, False),
            (402, False),
            (403, False),
            (429, True),
            (500, True),
        ):
            with self.subTest(status=status):
                error = urllib.error.HTTPError(
                    "https://api.deepseek.com/chat/completions",
                    status,
                    "sk-never-persist-this-body",
                    {},
                    None,
                )
                opener = _FakeOpener(error=error)
                with patch("urllib.request.build_opener", return_value=opener):
                    with self.assertRaises(ProviderCandidateError) as caught:
                        transport.post_json(
                            url="https://api.deepseek.com/chat/completions",
                            payload={"model": "deepseek-v4-pro"},
                            api_key="dummy-test-key",
                            timeout_s=1,
                        )
                self.assertEqual(caught.exception.details["failure_kind"], "http_error")
                self.assertEqual(caught.exception.details["http_status"], status)
                self.assertEqual(caught.exception.details["retryable"], retryable)
                self.assertNotIn("persist", str(caught.exception))
                self.assertEqual(opener.calls, 1)

    def test_adaptive_reasoning_keeps_thinking_for_new_candidate_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                ),
                transport=transport,
            )

            provider.next_candidate(
                prompt_payload=build_prompt_payload(
                    public_context={"previous_rounds": 0, "public_only": True}
                )
            )

        request = transport.calls[0]["payload"]
        self.assertEqual(request["thinking"], {"type": "enabled"})
        self.assertEqual(request["max_tokens"], 32_768)

    def test_adaptive_reasoning_retries_empty_final_channel_without_using_reasoning(
        self,
    ) -> None:
        reasoning_proposal = json.loads(
            self._response()["choices"][0]["message"]["content"]
        )
        reasoning_proposal["candidate_id"] = "reasoning_only_candidate_v1"
        reasoning_proposal["components_py"] = reasoning_proposal[
            "components_py"
        ].replace("upright_reward_v1", "reasoning_only_candidate_v1")
        private_reasoning = json.dumps(reasoning_proposal)
        first_response = {
            "model": "deepseek-v4-flash",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "reasoning_content": private_reasoning,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "completion_tokens_details": {"reasoning_tokens": 200},
            },
        }
        second_response = self._response()
        second_response["model"] = "deepseek-v4-flash"
        second_response["choices"][0]["finish_reason"] = "stop"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate_root = root / "candidates"
            transport = _SequenceTransport([first_response, second_response])
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-flash",
                    output_dir=candidate_root,
                    api_key="test-key",
                    reasoning_mode="adaptive",
                    reasoning_effort="max",
                    empty_final_reasoning_retry=True,
                ),
                transport=transport,
            )
            result = CandidateAuthoringLoop(
                context_builder=PublicContextBuilder(
                    task_prompt_adapter=FakeTaskPlugin().prompt_adapter(),
                    acceptance_protocol=TEST_ACCEPTANCE_PROTOCOL,
                    initial_current_best=make_bootstrap_current_best(),
                    session_genesis_hash=TEST_SESSION_GENESIS_HASH,
                ),
                provider=provider,
                candidate_admission_policy=FakeVerifier(),
                round_dir=root / "authoring",
                max_attempts=2,
            ).run_round(round_id="round-001", records=[])

            materialized_names = sorted(
                path.name for path in candidate_root.iterdir()
            )

        self.assertEqual(result.stop_reason, "candidate_verified")
        self.assertEqual(result.candidate.candidate_id, "upright_reward_v1")
        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(materialized_names, ["002_upright_reward_v1"])
        first_request = transport.calls[0]["payload"]
        second_request = transport.calls[1]["payload"]
        self.assertEqual(first_request["thinking"], {"type": "enabled"})
        self.assertEqual(first_request["reasoning_effort"], "max")
        self.assertNotIn("temperature", first_request)
        self.assertEqual(second_request["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", second_request)
        self.assertEqual(second_request["temperature"], 0.2)

        first_attempt = result.attempt_records[0]
        failure = first_attempt["provider_failure"]
        receipt = first_attempt["provider_response_receipt"]
        self.assertEqual(failure["category"], "json_parse_error")
        self.assertEqual(
            failure["details"]["failure_kind"],
            PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND,
        )
        self.assertEqual(receipt["response_text_utf8_bytes"], 0)
        self.assertTrue(receipt["reasoning_content_present"])
        self.assertEqual(
            receipt["reasoning_content_utf8_bytes"],
            len(private_reasoning.encode("utf-8")),
        )
        public_attempt_text = json.dumps(first_attempt, sort_keys=True)
        self.assertNotIn("reasoning_only_candidate_v1", public_attempt_text)
        self.assertNotIn(private_reasoning, public_attempt_text)

    def test_empty_final_retry_signal_requires_nonempty_reasoning_receipt(
        self,
    ) -> None:
        cases = (
            (
                "reasoning_absent",
                {"finish_reason": "stop", "message": {"content": ""}},
                "json_parse_error",
            ),
            (
                "reasoning_empty",
                {
                    "finish_reason": "stop",
                    "message": {"content": "", "reasoning_content": ""},
                },
                "json_parse_error",
            ),
            (
                "length_stopped",
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": '{"candidate_id":"private"}',
                    },
                },
                "reasoning_output_exhausted",
            ),
            (
                "nonempty_final",
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "not-json",
                        "reasoning_content": '{"candidate_id":"private"}',
                    },
                },
                "json_parse_error",
            ),
            (
                "null_final_length_stopped",
                {"finish_reason": "length", "message": {"content": None, "reasoning_content": "private"}},
                "reasoning_output_exhausted",
            ),
        )
        for name, choice, expected_category in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                provider = OpenAICompatibleCandidateProvider(
                    OpenAICompatibleProviderConfig(
                        model="deepseek-v4-flash",
                        output_dir=Path(tmp),
                        api_key="test-key",
                        reasoning_mode="adaptive",
                        reasoning_effort="max",
                        empty_final_reasoning_retry=True,
                    ),
                    transport=_FakeTransport({"choices": [choice]}),
                )
                with self.assertRaises(ProviderCandidateError) as caught:
                    provider.next_authoring_response(
                        prompt_payload=build_prompt_payload(
                            public_context={"public_only": True}
                        )
                    )

            self.assertEqual(caught.exception.category, expected_category)
            self.assertEqual("failure_kind" in caught.exception.to_public_dict()["details"],
                             expected_category == "reasoning_output_exhausted")

    def test_empty_final_retry_signal_requires_explicit_adaptive_opt_in(
        self,
    ) -> None:
        error = ProviderCandidateError(
            "json_parse_error",
            "provider final content channel was empty",
            details={
                "failure_kind": PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND,
                "finish_reason": "stop",
            },
        )
        prompt = build_prompt_payload(
            public_context={
                "public_only": True,
                "candidate_repair_feedback": build_provider_repair_feedback(
                    error
                ),
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            adaptive = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-flash",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                    reasoning_effort="max",
                    empty_final_reasoning_retry=True,
                ),
                transport=_FakeTransport({}),
            )._chat_completion_request(prompt)
            default_adaptive = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-flash",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                    reasoning_effort="max",
                ),
                transport=_FakeTransport({}),
            )._chat_completion_request(prompt)

        self.assertEqual(adaptive["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", adaptive)
        self.assertEqual(
            default_adaptive["thinking"], {"type": "enabled"}
        )
        self.assertEqual(default_adaptive["reasoning_effort"], "max")

    def test_empty_final_classification_does_not_depend_on_retry_permission(
        self,
    ) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": "",
                        "reasoning_content": '{"candidate_id":"private"}',
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-flash",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                    reasoning_effort="max",
                ),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_authoring_response(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

        failure = caught.exception.to_public_dict()
        self.assertEqual(failure["category"], "json_parse_error")
        self.assertEqual(failure["details"]["failure_kind"], PROVIDER_EMPTY_FINAL_REASONING_FAILURE_KIND)
        self.assertEqual(
            failure["reason"],
            "provider final content channel was empty",
        )

    def test_capacity_receipt_uses_provider_tokens_and_compact_sent_payload(self):
        from true_rl_agent.prompt_serialization import canonical_prompt_json
        context = {"context_window_tokens": 16384, "max_output_tokens": 4096,
                   "tokenizer_count_source": "provider_usage_prompt_tokens",
                   "specification_source": "fixture model contract"}
        response = self._response()
        response["usage"] = {"prompt_tokens": 987, "completion_tokens": 311}
        transport = _SequenceTransport([response])
        with tempfile.TemporaryDirectory() as tmp:
            config = OpenAICompatibleProviderConfig(model="fixture", output_dir=Path(tmp),
                api_key="test-key", max_tokens=2000, model_context=context)
            prompt = build_prompt_payload(public_context={"public_only": True})
            result = OpenAICompatibleCandidateProvider(config, transport=transport).next_authoring_response(prompt_payload=prompt)
        receipt = result.provider_receipt
        self.assertEqual(receipt["input_tokens"], 987)
        self.assertEqual(receipt["model_context"], context)
        self.assertEqual(receipt["joint_capacity_status"], "within_declared_window")
        self.assertEqual(normalize_provider_receipt(receipt), receipt)
        sent = transport.calls[0]["payload"]["messages"][-1]["content"]
        self.assertEqual(sent, canonical_prompt_json(prompt))
        self.assertEqual(json.loads(sent), prompt)
        self.assertNotIn('\n  "', sent)
        from true_rl_agent.prompt_serialization import sealed_prompt_json, canonical_prompt_json_bytes
        self.assertIn('\n  "', sealed_prompt_json({"schema_version": "true_rl_agent.prompt_payload.v4"}))
        section = {"schema_version": "anything", "text": "unicode: \u4e2d\u6587"}
        self.assertEqual(canonical_prompt_json_bytes(section), len(json.dumps(section, ensure_ascii=False, separators=(",", ":")).encode("utf-8")))
        with self.assertRaisesRegex(ValueError, "capacity"):
            OpenAICompatibleProviderConfig(model="fixture", output_dir=Path("unused"), max_tokens=5000, model_context=context)

    def test_http_400_is_nonretryable_without_guessing_provider_error_semantics(self):
        error = urllib.error.HTTPError("https://fixture.invalid", 400, "bad request", {},
            io.BytesIO(json.dumps({"error": {"code": "context_length_exceeded", "message": "private prompt"}}).encode()))
        with patch("urllib.request.build_opener", return_value=_FakeOpener(error=error)):
            with self.assertRaises(ProviderCandidateError) as caught:
                UrllibJsonTransport().post_json(url="https://fixture.invalid/chat/completions", payload={}, api_key="test-key", timeout_s=1)
        self.assertEqual(caught.exception.category, "runtime_error")
        self.assertFalse(caught.exception.details["retryable"])
        self.assertNotIn("private prompt", str(caught.exception.to_public_dict()))

    def test_max_reasoning_effort_is_sent_only_for_thinking_requests(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-flash",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="enabled",
                    reasoning_effort="max",
                ),
                transport=_FakeTransport({}),
            )
            enabled = provider._chat_completion_request(
                build_prompt_payload(
                    public_context={"previous_rounds": 0, "public_only": True}
                )
            )
            disabled_provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-flash",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="disabled",
                    reasoning_effort="max",
                ),
                transport=_FakeTransport({}),
            )
            disabled = disabled_provider._chat_completion_request(
                build_prompt_payload(
                    public_context={"previous_rounds": 0, "public_only": True}
                )
            )

        self.assertEqual(enabled["model"], "deepseek-v4-flash")
        self.assertEqual(enabled["reasoning_effort"], "max")
        self.assertNotIn("temperature", enabled)
        self.assertNotIn("reasoning_effort", disabled)
        self.assertEqual(disabled["temperature"], 0.2)

    def test_adaptive_reasoning_keeps_thinking_for_truncation_without_failed_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                ),
                transport=transport,
            )
            prompt = build_prompt_payload(
                public_context={
                    "previous_rounds": 0,
                    "public_only": True,
                    "candidate_repair_feedback": {
                        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
                        "failure_category": "response_truncated",
                    },
                }
            )

            provider.next_candidate(prompt_payload=prompt)

        request = transport.calls[0]["payload"]
        self.assertEqual(request["thinking"], {"type": "enabled"})

    def test_adaptive_reasoning_disables_thinking_for_inline_failed_source_repair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="test-model",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                ),
                transport=_FakeTransport({}),
            )
            prompt = build_prompt_payload(
                public_context={
                    "previous_rounds": 0,
                    "public_only": True,
                    "candidate_repair_feedback": {
                        "schema_version": "true_rl_agent.provider_repair_feedback.v1",
                        "failure_category": "response_truncated",
                    },
                }
            )
            prompt["source_payloads"] = [
                {
                    "logical_id": "attempt/latest_failed/training_code/components.py",
                    "delivery": "inline",
                    "content": "def build_training_components(): pass\n",
                }
            ]
            prompt["source_catalog"] = {
                "items": [
                    {
                        "logical_id": (
                            "attempt/latest_failed/training_code/components.py"
                        ),
                        "source_owner": "current_authoring_attempt",
                        "status": "inline",
                    }
                ]
            }

            request = provider._chat_completion_request(prompt)

        self.assertEqual(request["thinking"], {"type": "disabled"})

    def test_adaptive_reasoning_keeps_thinking_for_prior_ledger_failed_source(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="test-model",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                ),
                transport=_FakeTransport({}),
            )
            prompt = build_prompt_payload(
                public_context={
                    "previous_rounds": 1,
                    "public_only": True,
                    "candidate_repair_feedback": {
                        "schema_version": (
                            "true_rl_agent.provider_repair_feedback.v1"
                        ),
                        "failure_category": "response_truncated",
                    },
                }
            )
            prompt["source_payloads"] = [
                {
                    "logical_id": (
                        "attempt/latest_failed/training_code/components.py"
                    ),
                    "delivery": "inline",
                    "content": "def build_training_components(): pass\n",
                }
            ]
            prompt["source_catalog"] = {
                "items": [
                    {
                        "logical_id": (
                            "attempt/latest_failed/training_code/components.py"
                        ),
                        "source_owner": "prior_ledger_record",
                        "status": "inline",
                    }
                ]
            }

            request = provider._chat_completion_request(prompt)

        self.assertEqual(request["thinking"], {"type": "enabled"})

    def test_provider_resumes_after_existing_candidate_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "001_interrupted_candidate").mkdir()
            (root / "002_interrupted_candidate").mkdir()
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=root,
                    api_key="test-key",
                ),
                transport=transport,
            )

            candidate = provider.next_candidate(
                prompt_payload=build_prompt_payload(
                    public_context={"previous_rounds": 0, "public_only": True}
                )
            )

        self.assertEqual(Path(candidate.path).name, "003_upright_reward_v1")

    def test_provider_preserves_exact_components_source_bytes(self) -> None:
        response = self._response()
        proposal = json.loads(response["choices"][0]["message"]["content"])
        exact_source = str(proposal["components_py"]) + "  "
        proposal["components_py"] = exact_source
        response["choices"][0]["message"]["content"] = json.dumps(proposal)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )
            candidate = provider.next_candidate(
                prompt_payload=build_prompt_payload(
                    public_context={"previous_rounds": 0, "public_only": True}
                )
            )
            materialized = (
                Path(candidate.path) / "training_code" / "components.py"
            ).read_bytes()

        self.assertEqual(materialized, exact_source.encode("utf-8"))

    def test_provider_does_not_mistake_task_neutral_text_for_a_secret(self) -> None:
        response = self._response()
        proposal = json.loads(response["choices"][0]["message"]["content"])
        proposal["components_py"] = proposal["components_py"].replace(
            "Reward upright height using public inputs.",
            "Task-neutral and task-specific hypotheses are public-safe.",
        )
        response["choices"][0]["message"]["content"] = json.dumps(proposal)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )

            candidate = provider.next_candidate(
                prompt_payload=build_prompt_payload(
                    public_context={"public_only": True}
                )
            )

        self.assertEqual(candidate.candidate_id, "upright_reward_v1")

    def test_provider_error_redaction_uses_standalone_secret_boundary(self) -> None:
        ordinary = "Task-neutral task-specific risk-free disk-usage"

        safe_error = _safe_error(
            f"{ordinary} sk-lower123 SK-UPPER456"
        )
        failure = ProviderCandidateError(
            "runtime_error",
            f"{ordinary} sk-lower123 SK-UPPER456",
            details={
                "diagnostic": f"{ordinary} sk-detail123 SK-DETAIL456",
                "Authorization": "Bearer must-not-survive",
                "credential_diagnostic": "Bearer abc123",
                "embedded_header": "request failed: Authorization: Bearer abc123",
                "embedded_bearer": "upstream sent Bearer abc123",
                "short_alpha_bearer": "upstream sent Bearer abcdef",
                "quoted_header": 'request headers={"Authorization":"Basic abcdef"}',
                "upstream sent Bearer abcdef": "must-not-survive",
                'request headers={"Authorization":"Basic abcdef"}': (
                    "must-not-survive"
                ),
                r'request headers={\"Authorization\":\"Basic abcdef\"}': (
                    "must-not-survive"
                ),
            },
        ).to_public_dict()

        self.assertEqual(
            safe_error,
            f"{ordinary} [redacted] [redacted]",
        )
        self.assertEqual(
            failure["reason"],
            f"{ordinary} [redacted] [redacted]",
        )
        self.assertEqual(
            failure["details"]["diagnostic"],
            f"{ordinary} [redacted] [redacted]",
        )
        self.assertNotIn("Authorization", failure["details"])
        self.assertEqual(failure["details"]["credential_diagnostic"], "[redacted]")
        self.assertEqual(failure["details"]["embedded_header"], "[redacted]")
        self.assertEqual(failure["details"]["embedded_bearer"], "[redacted]")
        self.assertEqual(failure["details"]["short_alpha_bearer"], "[redacted]")
        self.assertEqual(failure["details"]["quoted_header"], "[redacted]")
        self.assertNotIn("must-not-survive", json.dumps(failure, sort_keys=True))

        repair = build_provider_repair_feedback(
            ProviderCandidateError(
                "runtime_error",
                "request failed: Authorization: Bearer abc123",
                details={
                    "message": "upstream sent Bearer abcdef",
                    "headers": 'request headers={"Authorization":"Basic abcdef"}',
                    "escaped_headers": (
                        r'request headers={\"Authorization\":\"Basic abcdef\"}'
                    ),
                },
            )
        )
        repair_text = json.dumps(repair, sort_keys=True)
        self.assertNotIn("Bearer abcdef", repair_text)
        self.assertNotIn("Basic abcdef", repair_text)
        self.assertNotIn("Authorization", repair_text)

    def test_provider_preserves_exact_embedded_hypothesis_text(self) -> None:
        response = self._response()
        proposal = json.loads(response["choices"][0]["message"]["content"])
        exact_hypothesis = "First line.\nSecond line with trailing spaces.  "
        proposal["components_py"] = proposal["components_py"].replace(
            "rationale='Reward upright height using public inputs.'",
            f"rationale={exact_hypothesis!r}",
        )
        response["choices"][0]["message"]["content"] = json.dumps(proposal)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )

            candidate = provider.next_candidate(
                prompt_payload=build_prompt_payload(
                    public_context={"public_only": True}
                )
            )
            materialized_hypothesis = (
                Path(candidate.path) / "rationale.md"
            ).read_bytes()

        self.assertEqual(materialized_hypothesis, exact_hypothesis.encode("utf-8"))

    def test_provider_rejects_oversized_components_before_ast_or_writes(
        self,
    ) -> None:
        response = self._response()
        proposal = json.loads(response["choices"][0]["message"]["content"])
        proposal["components_py"] = "x" * (
            CANDIDATE_PACKAGE_FILE_MAX_BYTES["training_code/components.py"] + 1
        )
        response["choices"][0]["message"]["content"] = json.dumps(proposal)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=root,
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )
            with patch(
                "true_rl_agent.agent_core.openai_compatible_provider."
                "extract_training_components_rationale"
            ) as extractor, self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

            extractor.assert_not_called()
            self.assertEqual(list(root.iterdir()), [])

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("training_code/components.py", caught.exception.reason)

    def test_provider_rejects_oversized_embedded_hypothesis_before_writes(
        self,
    ) -> None:
        response = self._response()
        proposal = json.loads(response["choices"][0]["message"]["content"])
        oversized_hypothesis = "h" * (
            CANDIDATE_PACKAGE_FILE_MAX_BYTES["rationale.md"] + 1
        )
        proposal["components_py"] = (
            "from true_rl_agent.candidates.components import TrainingComponents\n\n"
            "def build_training_components():\n"
            "    return TrainingComponents(\n"
            "        name='oversized_hypothesis',\n"
            f"        rationale={oversized_hypothesis!r},\n"
            "    )\n"
        )
        response["choices"][0]["message"]["content"] = json.dumps(proposal)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=root,
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )
            self.assertEqual(list(root.iterdir()), [])

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("rationale.md", caught.exception.reason)

    def test_provider_next_authoring_response_accepts_explicit_candidate_draft_mode(self) -> None:
        response = self._response()
        response["usage"] = {
            "prompt_tokens": 101,
            "completion_tokens": 23,
            "total_tokens": 124,
            "prompt_cache_hit_tokens": 80,
            "prompt_cache_miss_tokens": 21,
            "provider_private_counter": 999,
        }
        content = json.loads(response["choices"][0]["message"]["content"])
        content["response_mode"] = "candidate_draft"
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )

            authoring_response = provider.next_authoring_response(
                prompt_payload=build_prompt_payload(public_context={"public_only": True})
            )

        self.assertIsInstance(authoring_response, CandidateDraftResponse)
        self.assertEqual(authoring_response.response_mode, "candidate_draft")
        self.assertEqual(authoring_response.candidate.candidate_id, "upright_reward_v1")
        self.assertEqual(
            authoring_response.provider_usage,
            {
                "request_units": 101,
                "response_units": 23,
                "total_units": 124,
                "request_cache_hit_units": 80,
                "request_cache_miss_units": 21,
            },
        )
        self.assertNotIn("provider_usage", authoring_response.to_public_dict())

    def test_provider_classifies_nonstring_response_mode_as_repairable_schema_error(
        self,
    ) -> None:
        response = self._response()
        content = json.loads(response["choices"][0]["message"]["content"])
        content["response_mode"] = 7
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )

            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_authoring_response(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("response_mode must be a string", caught.exception.reason)

    def test_provider_parses_source_request_without_materializing_candidate(self) -> None:
        source_request = {
            "response_mode": "source_request",
            "logical_ids": ["current_best/training_code/components.py"],
            "rationale": "Need the current best source before drafting: D:/humanoid_agent/private/raw.txt",
        }
        response = {
            "choices": [{"message": {"content": json.dumps(source_request)}}],
            "usage": {
                "prompt_tokens": 81,
                "prompt_cache_hit_tokens": 60,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=root, api_key="test-key"),
                transport=_FakeTransport(response),
            )

            authoring_response = provider.next_authoring_response(
                prompt_payload=build_prompt_payload(public_context={"public_only": True})
            )

            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

            self.assertEqual(list(root.iterdir()), [])

        self.assertIsInstance(authoring_response, SourceRequestResponse)
        self.assertEqual(authoring_response.response_mode, "source_request")
        self.assertEqual(authoring_response.logical_ids, ("current_best/training_code/components.py",))
        self.assertEqual(
            authoring_response.provider_usage,
            {"request_units": 81, "request_cache_hit_units": 60},
        )
        self.assertNotIn("provider_usage", authoring_response.to_public_dict())
        self.assertNotIn("D:/humanoid_agent", json.dumps(authoring_response.to_public_dict(), sort_keys=True))
        self.assertEqual(caught.exception.category, "source_request_denied")
        self.assertEqual(
            caught.exception.provider_usage,
            {"request_units": 81, "request_cache_hit_units": 60},
        )
        self.assertNotIn("unsupported provider output key", caught.exception.to_public_dict()["reason"])

    def test_provider_rejects_source_request_logical_id_outside_prompt_catalog(self) -> None:
        source_request = {
            "response_mode": "source_request",
            "logical_ids": ["unknown/logical.py"],
            "rationale": "Need an undeclared source.",
        }
        response = {
            "model": "deepseek-v4-pro",
            "system_fingerprint": "fp-source-request",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(source_request)},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )

            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_authoring_response(
                    prompt_payload=build_prompt_payload(public_context={"public_only": True})
                )

        failure = caught.exception.to_public_dict()
        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("not declared in prompt source_catalog", failure["reason"])
        self.assertEqual(
            caught.exception.provider_receipt["response_model"],
            "deepseek-v4-pro",
        )
        self.assertEqual(
            caught.exception.provider_receipt["system_fingerprint"],
            "fp-source-request",
        )

    def test_source_request_public_dict_rejects_forbidden_public_values(self) -> None:
        from true_rl_agent.agent_core.authoring_response import (
            SourceRequestResponse,  # noqa: E402
        )
        from true_rl_agent.prompting.source_catalog import (
            prompt_source_catalog_logical_ids,  # noqa: E402
        )

        prompt_payload = build_prompt_payload(public_context={"public_only": True})
        source_catalog_logical_ids = tuple(sorted(prompt_source_catalog_logical_ids(prompt_payload)))

        with self.assertRaises(ValueError):
            SourceRequestResponse(
                logical_ids=("current_best/training_code/components.py",),
                rationale="Need raw_seed and sk-live-secret before drafting.",
                source_catalog_logical_ids=source_catalog_logical_ids,
            )
        with self.assertRaises(ValueError):
            SourceRequestResponse(
                logical_ids=("api_key",),
                rationale="Need source.",
                source_catalog_logical_ids=("api_key",),
            )

    def test_provider_uses_supplied_prompt_payload_without_rebuilding_or_sorting(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    base_url="https://api.deepseek.com",
                    api_key="test-key",
                ),
                transport=transport,
            )
            prompt_payload = {
                "schema_version": "true_rl_agent.prompt_payload.test",
                "z_extra_section": {"visible_to_provider": True},
                "public_context": {"previous_rounds": 0, "public_only": True},
            }

            candidate = provider.next_candidate(prompt_payload=prompt_payload)

        self.assertEqual(candidate.candidate_id, "upright_reward_v1")
        user_content = transport.calls[0]["payload"]["messages"][1]["content"]
        self.assertIn("z_extra_section", user_content)
        self.assertIn("visible_to_provider", user_content)
        self.assertLess(user_content.index("schema_version"), user_content.index("z_extra_section"))
        self.assertLess(user_content.index("z_extra_section"), user_content.index("public_context"))
        self.assertNotIn("candidate_component_contract", user_content)

    def test_provider_user_payload_matches_saved_prompt_artifact_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompt_payload = build_prompt_payload(
                public_context={
                    "public_only": True,
                    "task_prompt_sections": {
                        "task": {
                            "schema_version": "custom.task.v1",
                            "task_id": "stair_climbing",
                            "name": "阶梯任务",
                        },
                    },
                }
            )
            prompt_path = root / "round-001" / "prompt_payload.json"
            prompt_ref = PublicContextBuilder().save_prompt_payload(prompt_path, prompt_payload)
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=root / "candidates",
                    base_url="https://api.deepseek.com",
                    api_key="test-key",
                ),
                transport=transport,
            )

            provider.next_candidate(prompt_payload=prompt_payload)
            saved_text = prompt_path.read_text(encoding="utf-8")
            user_content = transport.calls[0]["payload"]["messages"][1]["content"]

        self.assertEqual(prompt_ref.to_public_dict()["path"], str(prompt_path))
        self.assertEqual(saved_text, user_content)
        self.assertEqual(json.loads(user_content), prompt_payload)
        self.assertIn("阶梯任务", saved_text)
        self.assertIn("阶梯任务", user_content)

    def test_system_prompt_is_generic_and_task_semantics_stay_in_user_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            transport = _FakeTransport(self._response())
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=transport,
            )
            prompt_payload = build_prompt_payload(
                public_context={
                    "public_only": True,
                    "task_prompt_sections": {
                        "task": {"schema_version": "custom.task.v1", "task_id": "stair_climbing"},
                    },
                }
            )

            provider.next_candidate(prompt_payload=prompt_payload)

        system_content = transport.calls[0]["payload"]["messages"][0]["content"]
        user_content = transport.calls[0]["payload"]["messages"][1]["content"]
        user_payload = json.loads(user_content)
        advertised_declared_values = user_payload["required_candidate_package"]["provider_json_output"][
            "declared_changed_components_allowed_values"
        ]
        system_lower = system_content.lower()

        self.assertNotIn("push recovery", system_content.lower())
        self.assertNotIn("stair_climbing", system_content)
        self.assertIn("stair_climbing", user_content)
        self.assertNotIn("budgets, success metrics, or task invariants", system_content)
        self.assertIn("user payload is the only source of task-specific context", system_lower)
        self.assertIn("read-only evidence", system_lower)
        self.assertIn("candidate_draft", system_content)
        self.assertIn("source_request", system_content)
        self.assertIn("match exactly one response shape", system_lower)
        self.assertNotIn("the JSON object must follow the provider_json_output contract in the user payload and include the candidate draft fields", system_content)
        self.assertNotIn("RewardInputs", system_content)
        self.assertNotIn("RewardTerm", system_content)
        self.assertNotIn("DiagnosticTerm", system_content)
        self.assertNotIn("ScheduleTerm", system_content)
        self.assertNotIn("TrainingComponents", system_content)
        self.assertIn("RewardInputs", user_content)
        self.assertIn("RewardTerm", user_content)
        for advertised_value in PROMPT_ADVERTISED_DECLARED_COMPONENTS:
            self.assertIn(advertised_value, advertised_declared_values)
            self.assertIn(advertised_value, user_content)
        for legacy_value in (
            "control_logic",
            "recovery_logic",
            "failure_classifier",
            "logging",
            "public_projection",
            "reward_weight_scale",
            "schedule",
        ):
            self.assertNotIn(legacy_value, system_content)
            self.assertNotIn(legacy_value, advertised_declared_values)

    def test_candidate_draft_rejects_standalone_rationale_field(self) -> None:
        response = self._response()
        content = json.loads(response["choices"][0]["message"]["content"])
        content["rationale"] = "This must not become a second hypothesis source."
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )

            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("unsupported provider output key", caught.exception.reason)

    def test_candidate_draft_requires_embedded_literal_hypothesis(self) -> None:
        for replacement in (
            "",
            "        rationale=build_hypothesis(),\n",
        ):
            with self.subTest(replacement=replacement):
                response = self._response()
                content = json.loads(
                    response["choices"][0]["message"]["content"]
                )
                content["components_py"] = content["components_py"].replace(
                    "        rationale='Reward upright height using public inputs.',\n",
                    replacement,
                )
                response["choices"][0]["message"]["content"] = json.dumps(
                    content
                )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    provider = OpenAICompatibleCandidateProvider(
                        OpenAICompatibleProviderConfig(
                            model="deepseek-v4-pro",
                            output_dir=root,
                            api_key="test-key",
                        ),
                        transport=_FakeTransport(response),
                    )

                    with self.assertRaises(ProviderCandidateError) as caught:
                        provider.next_candidate(
                            prompt_payload=build_prompt_payload(
                                public_context={"public_only": True}
                            )
                        )
                    self.assertEqual(list(root.iterdir()), [])

                self.assertEqual(caught.exception.category, "schema_error")
                self.assertIn("TrainingComponents.rationale", caught.exception.reason)

    def test_provider_rejects_legacy_declared_components_not_advertised_in_prompt_contract(self) -> None:
        response = self._response()
        content = json.loads(response["choices"][0]["message"]["content"])
        content["declared_changed_components"] = ["control_logic"]
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

        failure = caught.exception.to_public_dict()
        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("declared_changed_components contains unsupported values", failure["reason"])
        self.assertIn("control_logic", failure["reason"])

    def test_provider_rejects_path_like_candidate_ids_and_secret_text(self) -> None:
        bad_response = self._response()
        content = json.loads(bad_response["choices"][0]["message"]["content"])
        content["candidate_id"] = "../escape"
        content["components_py"] = "sk-leaked"
        bad_response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(bad_response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))
        self.assertEqual(caught.exception.category, "schema_error")

    def test_provider_rejects_candidate_id_over_shared_length_limit(self) -> None:
        bad_response = self._response()
        content = json.loads(bad_response["choices"][0]["message"]["content"])
        content["candidate_id"] = "a" * 81
        bad_response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(bad_response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("at most 80 characters", caught.exception.reason)

    def test_provider_rejects_nonportable_candidate_ids_before_materialization(
        self,
    ) -> None:
        for candidate_id in ("a.", "nul", "con.txt", "Candidate"):
            with self.subTest(candidate_id=candidate_id):
                bad_response = self._response()
                content = json.loads(
                    bad_response["choices"][0]["message"]["content"]
                )
                content["candidate_id"] = candidate_id
                bad_response["choices"][0]["message"]["content"] = json.dumps(
                    content
                )
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    provider = OpenAICompatibleCandidateProvider(
                        OpenAICompatibleProviderConfig(
                            model="deepseek-v4-pro",
                            output_dir=root,
                            api_key="test-key",
                        ),
                        transport=_FakeTransport(bad_response),
                    )
                    with self.assertRaises(ProviderCandidateError) as caught:
                        provider.next_candidate(
                            prompt_payload=build_prompt_payload(
                                public_context={"public_only": True}
                            )
                        )
                    self.assertEqual(list(root.iterdir()), [])

                self.assertEqual(caught.exception.category, "schema_error")

    def test_provider_rejects_unsupported_or_prohibited_output_keys(self) -> None:
        bad_response = self._response()
        content = json.loads(bad_response["choices"][0]["message"]["content"])
        content["runner_patch"] = {"budget_steps": 1}
        bad_response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(bad_response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

        failure = caught.exception.to_public_dict()
        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("unsupported provider output key", failure["reason"])
        self.assertNotIn("budget_steps", json.dumps(failure, sort_keys=True))

    def test_provider_wraps_malformed_candidate_json_as_public_repairable_error(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "not json with sk-live-secret and D:\\humanoid_agent\\private\\raw.txt"
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 77,
                "prompt_cache_miss_tokens": 77,
                "completion_tokens": -1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

        failure = caught.exception.to_public_dict()
        failure_text = json.dumps(failure, sort_keys=True)
        self.assertEqual(failure["category"], "json_parse_error")
        self.assertEqual(
            caught.exception.provider_usage,
            {"request_units": 77, "request_cache_miss_units": 77},
        )
        self.assertNotIn("sk-live-secret", failure_text)
        self.assertNotIn("D:\\humanoid_agent", failure_text)
        self.assertNotIn("not json", failure_text)

    def test_provider_classifies_length_stopped_json_and_requests_concise_repair(
        self,
    ) -> None:
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"candidate_id":"truncated"'},
                }
            ],
            "usage": {"completion_tokens": 8192},
        }
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                ),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_authoring_response(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

        failure = caught.exception.to_public_dict()
        feedback = build_provider_repair_feedback(caught.exception)
        self.assertEqual(caught.exception.category, "response_truncated")
        self.assertEqual(failure["details"], {"finish_reason": "length"})
        self.assertEqual(
            caught.exception.provider_usage,
            {"response_units": 8192},
        )
        instructions = " ".join(feedback["instructions"])
        self.assertIn("shorter response", instructions)
        self.assertIn("Keep components_py complete", instructions)
        self.assertNotIn("8192", json.dumps(feedback, sort_keys=True))

    def test_truncated_response_records_content_free_reasoning_receipt(self) -> None:
        response = {
            "model": "deepseek-v4-pro",
            "system_fingerprint": "fp-test",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": '{"candidate_id":"truncated"',
                        "reasoning_content": "private chain of thought",
                    },
                }
            ],
            "usage": {
                "completion_tokens": 8192,
                "completion_tokens_details": {"reasoning_tokens": 6144},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(
                    model="deepseek-v4-pro",
                    output_dir=Path(tmp),
                    api_key="test-key",
                    reasoning_mode="adaptive",
                ),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_authoring_response(
                    prompt_payload=build_prompt_payload(
                        public_context={"public_only": True}
                    )
                )

        receipt = caught.exception.provider_receipt
        receipt_text = json.dumps(receipt, sort_keys=True)
        self.assertEqual(
            caught.exception.provider_usage,
            {"response_units": 8192, "reasoning_units": 6144},
        )
        self.assertEqual(receipt["requested_thinking_mode"], "enabled")
        self.assertEqual(
            receipt["schema_version"],
            "true_rl_agent.provider_response_receipt.v1",
        )
        self.assertNotIn("requested_reasoning_effort", receipt)
        self.assertEqual(receipt["requested_output_units"], 32_768)
        self.assertEqual(receipt["response_model"], "deepseek-v4-pro")
        self.assertEqual(receipt["system_fingerprint"], "fp-test")
        self.assertEqual(receipt["finish_reason"], "length")
        self.assertTrue(receipt["response_text_present"])
        self.assertTrue(receipt["reasoning_content_present"])
        self.assertIn("sha256:", receipt["response_text_sha256"])
        self.assertIn("sha256:", receipt["reasoning_content_sha256"])
        self.assertNotIn("private chain of thought", receipt_text)
        self.assertNotIn('{"candidate_id":"truncated"', receipt_text)

    def test_provider_receipt_allowlist_drops_content_and_invalid_metadata(self) -> None:
        receipt = normalize_provider_receipt(
            {
                "schema_version": "true_rl_agent.provider_response_receipt.v1",
                "requested_model": "deepseek-v4-pro",
                "requested_thinking_mode": "enabled",
                "requested_output_units": 32_768,
                "finish_reason": "stop",
                "response_text_sha256": "not-a-hash",
                "raw_response": "private provider output",
                "reasoning_content": "private chain of thought",
            }
        )

        self.assertEqual(
            receipt["schema_version"],
            "true_rl_agent.provider_response_receipt.v1",
        )
        self.assertEqual(receipt["requested_model"], "deepseek-v4-pro")
        self.assertEqual(receipt["requested_thinking_mode"], "enabled")
        self.assertEqual(receipt["requested_output_units"], 32_768)
        self.assertEqual(receipt["finish_reason"], "stop")
        self.assertNotIn("raw_response", receipt)
        self.assertNotIn("reasoning_content", receipt)
        self.assertNotIn("response_text_sha256", receipt)

    def test_provider_receipt_v3_preserves_only_valid_reasoning_effort(self) -> None:
        base = {
            "schema_version": "true_rl_agent.provider_response_receipt.v3",
            "requested_model": "deepseek-v4-flash",
            "requested_thinking_mode": "enabled",
            "requested_output_units": 32_768,
            "transport_policy_id": (
                "true_rl_agent.openai_compatible_transport."
                "https_or_loopback_http_no_redirect.v1"
            ),
            "response_max_bytes": 512 * 1024,
            "finish_reason": "stop",
        }
        for effort in ("high", "max", "provider_default"):
            with self.subTest(effort=effort):
                receipt = normalize_provider_receipt(
                    {**base, "requested_reasoning_effort": effort}
                )
                self.assertEqual(receipt["requested_reasoning_effort"], effort)

        for unsupported in ("low", "medium", "xhigh", "unsupported"):
            with self.subTest(unsupported=unsupported):
                self.assertEqual(
                    normalize_provider_receipt(
                        {
                            **base,
                            "requested_reasoning_effort": unsupported,
                        }
                    ),
                    {},
                )

    def test_provider_rejects_duplicate_keys_and_non_finite_json(self) -> None:
        for content in (
            '{"response_mode":"candidate_draft","response_mode":"source_request"}',
            '{"response_mode":NaN}',
        ):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                response = {
                    "choices": [{"message": {"content": content}}]
                }
                provider = OpenAICompatibleCandidateProvider(
                    OpenAICompatibleProviderConfig(
                        model="deepseek-v4-pro",
                        output_dir=root,
                        api_key="test-key",
                    ),
                    transport=_FakeTransport(response),
                )

                with self.assertRaises(ProviderCandidateError) as caught:
                    provider.next_authoring_response(
                        prompt_payload=build_prompt_payload(
                            public_context={"public_only": True}
                        )
                    )

                self.assertEqual(caught.exception.category, "json_parse_error")
                self.assertEqual(list(root.iterdir()), [])

    def test_provider_repair_feedback_scrubs_windows_forward_slash_paths(self) -> None:
        error = ProviderCandidateError(
            "schema_error",
            "components_py failed at D:/humanoid_agent/private/raw.txt",
            details={"source_path": "D:/humanoid_agent/private/raw.txt"},
        )

        feedback = build_provider_repair_feedback(error)

        feedback_text = json.dumps(feedback, sort_keys=True)
        self.assertNotIn("D:/humanoid_agent", feedback_text)
        self.assertNotIn("private/raw.txt", feedback_text)
        self.assertIn("[path]", feedback_text)

    def test_provider_failure_public_dict_uses_shared_posix_path_scrub(self) -> None:
        error = ProviderCandidateError(
            "schema_error",
            "candidate failed from /opt/runner/site.py and /workspace/project/private/raw.txt",
            details={
                "traceback": "loaded /opt/runner/site.py from /workspace/project/private/raw.txt",
                "nested": {"path": "/opt/runner/site.py"},
            },
        )

        failure = error.to_public_dict()
        feedback = build_provider_repair_feedback(error)

        public_text = json.dumps({"failure": failure, "feedback": feedback}, sort_keys=True)
        self.assertNotIn("/opt/", public_text)
        self.assertNotIn("/workspace/", public_text)
        self.assertNotIn("private/raw.txt", public_text)
        self.assertIn("[path]", public_text)

    def test_provider_wraps_missing_components_as_schema_error(self) -> None:
        response = self._response()
        content = json.loads(response["choices"][0]["message"]["content"])
        del content["components_py"]
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("components_py", caught.exception.to_public_dict()["reason"])

    def test_provider_rejects_invalid_declared_components_as_schema_error(self) -> None:
        response = self._response()
        content = json.loads(response["choices"][0]["message"]["content"])
        content["declared_changed_components"] = "added reward terms"
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

        self.assertEqual(caught.exception.category, "schema_error")
        self.assertIn("declared_changed_components", caught.exception.to_public_dict()["reason"])

    def test_provider_wraps_materialize_failure_as_public_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(self._response()),
            )
            (Path(tmp) / "001_upright_reward_v1" / "training_code").mkdir(
                parents=True
            )
            with self.assertRaises(ProviderCandidateError) as caught:
                provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))

        failure = caught.exception.to_public_dict()
        failure_text = json.dumps(failure, sort_keys=True)
        self.assertEqual(failure["category"], "candidate_materialize_error")
        self.assertNotIn(str(Path(tmp)), failure_text)

    def test_provider_extracts_json_object_from_fenced_response_text(self) -> None:
        response = self._response()
        content = response["choices"][0]["message"]["content"]
        response["choices"][0]["message"]["content"] = f"```json\n{content}\n```"
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )
            candidate = provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))
            verification = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(candidate.candidate_id, "upright_reward_v1")
        self.assertEqual(verification.status, "pass")

    def test_load_env_file_reads_names_without_requiring_openai_variable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / "deepseek.env"
            env_file.write_text("DEEPSEEK_API_KEY=test-secret\n", encoding="utf-8")

            values = load_env_file(env_file)

        self.assertEqual(values["DEEPSEEK_API_KEY"], "test-secret")


if __name__ == "__main__":
    unittest.main()
