from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.openai_compatible_provider import (  # noqa: E402
    OpenAICompatibleCandidateProvider,
    OpenAICompatibleProviderConfig,
    load_env_file,
)
from true_rl_agent.agent_core.provider_contract import (  # noqa: E402
    LEGACY_PARSE_DECLARED_COMPONENTS,
    PROMPT_ADVERTISED_DECLARED_COMPONENTS,
)
from true_rl_agent.agent_core.provider_errors import ProviderCandidateError  # noqa: E402
from true_rl_agent.agent_core.provider_errors import build_provider_repair_feedback  # noqa: E402
from true_rl_agent.candidates.prompt_projection import build_prompt_payload  # noqa: E402
from true_rl_agent.candidates.verifier import StaticCandidateVerifier  # noqa: E402
from true_rl_agent.memory import PublicContextBuilder  # noqa: E402


class _FakeTransport:
    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def post_json(self, *, url: str, payload: dict[str, Any], api_key: str, timeout_s: int) -> dict[str, Any]:
        self.calls.append({"url": url, "payload": payload, "api_key": api_key, "timeout_s": timeout_s})
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
                                "rationale": "Add an upright height reward using public RewardInputs.",
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

        self.assertEqual(candidate.candidate_id, "upright_reward_v1")
        self.assertEqual(candidate.change_set.changed_components, frozenset({"reward_callable"}))
        self.assertEqual(verification.status, "pass")
        self.assertEqual(operation_diff["provider"], "openai_compatible_chat")
        self.assertEqual(operation_diff["declared_changed_components"], ["reward_callable"])
        self.assertEqual(diagnostics["status"], "provider_materialized")
        self.assertEqual(diagnostics["diagnostic_items"], [])
        self.assertEqual(verification.details["derived_change_set"], ["reward_callable"])
        self.assertEqual(transport.calls[0]["url"], "https://api.deepseek.com/chat/completions")
        self.assertEqual(transport.calls[0]["payload"]["response_format"], {"type": "json_object"})
        request_text = json.dumps(transport.calls[0]["payload"], sort_keys=True)
        self.assertIn("JSON", request_text)
        self.assertIn("candidate_component_contract", request_text)
        self.assertIn("def build_training_components() -> TrainingComponents:", request_text)
        self.assertIn("RewardTerm(name: str, weight: float", request_text)
        self.assertIn("task_owned_contract_not_provided", request_text)
        self.assertNotIn("base_pos_xy", request_text)
        self.assertIn("candidate_package_only", request_text)
        self.assertIn("no_prose_outside_json", request_text)
        self.assertIn("diagnostics.json", request_text)
        self.assertIn("operation_diff.json", request_text)
        self.assertIn("runner_patch", request_text)
        self.assertNotIn("test-key", request_text)

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
        self.assertEqual(saved_text, user_content + "\n")
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
        for legacy_value in LEGACY_PARSE_DECLARED_COMPONENTS:
            self.assertNotIn(legacy_value, system_content)
            self.assertNotIn(legacy_value, advertised_declared_values)

    def test_provider_materializer_tolerates_legacy_declared_components_without_advertising_them(self) -> None:
        response = self._response()
        content = json.loads(response["choices"][0]["message"]["content"])
        content["declared_changed_components"] = ["control_logic"]
        content["components_py"] = "\n".join(
            [
                "from true_rl_agent.candidates.components import DiagnosticTerm, TrainingComponents",
                "",
                "",
                "def public_diag(inputs):",
                "    return inputs.base_height",
                "",
                "",
                "def build_training_components():",
                "    return TrainingComponents(",
                "        name='legacy_declared_component_probe',",
                "        diagnostics=[DiagnosticTerm(name='public_diag', fn=public_diag)],",
                "    )",
            ]
        )
        response["choices"][0]["message"]["content"] = json.dumps(content)
        with tempfile.TemporaryDirectory() as tmp:
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(response),
            )
            candidate = provider.next_candidate(prompt_payload=build_prompt_payload(public_context={"public_only": True}))
            verification = StaticCandidateVerifier().verify(candidate)

        self.assertEqual(candidate.change_set.changed_components, frozenset({"control_logic"}))
        self.assertEqual(verification.status, "pass")
        self.assertEqual(verification.details["derived_change_set"], ["diagnostic"])

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
            ]
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
        self.assertNotIn("sk-live-secret", failure_text)
        self.assertNotIn("D:\\humanoid_agent", failure_text)
        self.assertNotIn("not json", failure_text)

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
            (Path(tmp) / "001_upright_reward_v1" / "training_code").mkdir(parents=True)
            provider = OpenAICompatibleCandidateProvider(
                OpenAICompatibleProviderConfig(model="deepseek-v4-pro", output_dir=Path(tmp), api_key="test-key"),
                transport=_FakeTransport(self._response()),
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
