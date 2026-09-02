from __future__ import annotations

import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

from true_rl_agent.tasks.stair_climbing.provider_readiness import (
    STAIR_PROVIDER_BALANCE_ORIGIN,
    STAIR_PROVIDER_READINESS_MAX_RESPONSE_BYTES,
    StairProviderReadinessError,
    require_stair_provider_readiness,
)


class _Response:
    def __init__(self, payload: dict[str, object]) -> None:
        self.raw = json.dumps(payload).encode("utf-8")
        self.read_limit: int | None = None

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        self.read_limit = limit
        return self.raw


class _Opener:
    def __init__(
        self,
        response: _Response | None = None,
        error: BaseException | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.request: urllib.request.Request | None = None
        self.timeout: int | None = None

    def open(
        self,
        request: urllib.request.Request,
        *,
        timeout: int,
    ) -> _Response:
        self.request = request
        self.timeout = timeout
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class StairProviderReadinessTests(unittest.TestCase):
    def _env_file(self, root: Path) -> Path:
        path = root / "provider.env"
        path.write_text("DEEPSEEK_API_KEY=test-secret\n", encoding="utf-8")
        return path

    def test_balance_gate_uses_get_and_projects_no_secret_or_amounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            response = _Response(
                {
                    "is_available": True,
                    "balance_infos": [
                        {
                            "currency": "USD",
                            "total_balance": "123.45",
                            "granted_balance": "23.45",
                            "topped_up_balance": "100.00",
                        }
                    ],
                }
            )
            opener = _Opener(response=response)
            with patch(
                "true_rl_agent.agent_core.openai_compatible_provider."
                "urllib.request.build_opener",
                return_value=opener,
            ):
                report = require_stair_provider_readiness(
                    base_url=STAIR_PROVIDER_BALANCE_ORIGIN,
                    api_key_env="DEEPSEEK_API_KEY",
                    env_file=self._env_file(Path(tmp)),
                )

        assert opener.request is not None
        self.assertEqual(opener.request.get_method(), "GET")
        self.assertEqual(
            opener.request.full_url,
            "https://api.deepseek.com/user/balance",
        )
        self.assertIsNone(opener.request.data)
        self.assertEqual(
            opener.request.get_header("Authorization"),
            "Bearer test-secret",
        )
        self.assertEqual(opener.request.get_header("Accept"), "application/json")
        self.assertIsNone(opener.request.get_header("Content-type"))
        self.assertEqual(opener.timeout, 30)
        self.assertEqual(
            response.read_limit,
            STAIR_PROVIDER_READINESS_MAX_RESPONSE_BYTES + 1,
        )
        self.assertTrue(all(report["checks"].values()))
        self.assertEqual(
            report["account_projection"],
            {"is_available": True, "currencies": ["USD"]},
        )
        self.assertEqual(
            report["request_projection"],
            {
                "method": "GET",
                "provider_balance_request_sent": True,
                "generation_request_sent": False,
            },
        )
        serialized = json.dumps(report, sort_keys=True)
        self.assertNotIn("test-secret", serialized)
        self.assertNotIn("123.45", serialized)
        self.assertFalse(any(report["side_effects"].values()))

    def test_balance_unavailable_fails_closed_without_persisting_amounts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            opener = _Opener(
                response=_Response(
                    {
                        "is_available": False,
                        "balance_infos": [
                            {
                                "currency": "CNY",
                                "total_balance": "0.00",
                                "granted_balance": "0.00",
                                "topped_up_balance": "0.00",
                            }
                        ],
                    }
                )
            )
            with (
                patch(
                    "true_rl_agent.agent_core.openai_compatible_provider."
                    "urllib.request.build_opener",
                    return_value=opener,
                ),
                self.assertRaises(StairProviderReadinessError) as caught,
            ):
                require_stair_provider_readiness(
                    base_url=STAIR_PROVIDER_BALANCE_ORIGIN,
                    api_key_env="DEEPSEEK_API_KEY",
                    env_file=self._env_file(Path(tmp)),
                )

        self.assertEqual(caught.exception.failure_kind, "insufficient_balance")
        self.assertIsNone(caught.exception.http_status)
        self.assertEqual(
            str(caught.exception),
            "provider readiness reports insufficient balance",
        )
        self.assertNotIn("0.00", str(caught.exception))

    def test_http_402_is_classified_as_insufficient_balance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            error = urllib.error.HTTPError(
                "https://api.deepseek.com/user/balance",
                402,
                "Payment Required",
                hdrs=None,
                fp=None,
            )
            opener = _Opener(error=error)
            with (
                patch(
                    "true_rl_agent.agent_core.openai_compatible_provider."
                    "urllib.request.build_opener",
                    return_value=opener,
                ),
                self.assertRaises(StairProviderReadinessError) as caught,
            ):
                require_stair_provider_readiness(
                    base_url=STAIR_PROVIDER_BALANCE_ORIGIN,
                    api_key_env="DEEPSEEK_API_KEY",
                    env_file=self._env_file(Path(tmp)),
                )

        self.assertEqual(caught.exception.failure_kind, "insufficient_balance")
        self.assertEqual(caught.exception.http_status, 402)
        self.assertEqual(
            str(caught.exception),
            "provider readiness reports insufficient balance",
        )

    def test_balance_schema_rejects_missing_or_non_boolean_availability(self) -> None:
        invalid_payloads = (
            {"balance_infos": []},
            {"is_available": "true", "balance_infos": []},
            {"is_available": True, "balance_infos": {}},
            {
                "is_available": True,
                "balance_infos": [
                    {
                        "currency": "USD",
                        "total_balance": "1.00",
                    }
                ],
            },
            {
                "is_available": True,
                "balance_infos": [
                    {
                        "currency": "DOGE",
                        "total_balance": "1.00",
                        "granted_balance": "0.00",
                        "topped_up_balance": "1.00",
                    }
                ],
            },
        )
        for payload in invalid_payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as tmp:
                opener = _Opener(response=_Response(payload))
                with (
                    patch(
                        "true_rl_agent.agent_core.openai_compatible_provider."
                        "urllib.request.build_opener",
                        return_value=opener,
                    ),
                    self.assertRaises(StairProviderReadinessError) as caught,
                ):
                    require_stair_provider_readiness(
                        base_url=STAIR_PROVIDER_BALANCE_ORIGIN,
                        api_key_env="DEEPSEEK_API_KEY",
                        env_file=self._env_file(Path(tmp)),
                    )
                self.assertEqual(caught.exception.failure_kind, "invalid_response")


if __name__ == "__main__":
    unittest.main()
