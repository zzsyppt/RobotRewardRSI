from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from true_rl_agent.agent_core.generation_contract import (
    GENERATION_SESSION_CONTRACT_FILENAME,
    GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
    GenerationSessionContract,
)
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing.generation_session import (
    STAIR_GENERATION_REPORT_FILENAME,
    STAIR_GENERATION_REPORT_SCHEMA_VERSION,
    StairGenerationPilotConfig,
    _publish_or_load_terminal_p6b_session_contract,
)
from true_rl_agent.tasks.stair_climbing.generation_session_cost import (
    DEVELOPMENT_GENERATION_PILOT_MODE,
)


_SOURCE_COMMIT = "a" * 40
_GPU_A = "GPU-11111111-1111-1111-1111-111111111111"
_GPU_B = "GPU-22222222-2222-2222-2222-222222222222"
_BOUND_FIELDS = (
    "authoring_policy",
    "continuation_policy",
    "evaluation_cost",
    "evaluation_protocol",
    "exploration_projector",
    "final_confirmation_plan",
    "provider_policy",
    "resource_plan",
    "retry_policy",
    "selector",
    "session_cost",
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _contract(*, authoring: str, provider: str | None = None) -> GenerationSessionContract:
    hashes = {field_name: _digest(field_name) for field_name in _BOUND_FIELDS}
    hashes["authoring_policy"] = _digest(authoring)
    if provider is not None:
        hashes["provider_policy"] = _digest(provider)
    return GenerationSessionContract(
        schema_version=GENERATION_SESSION_CONTRACT_V3_SCHEMA_VERSION,
        session_mode=DEVELOPMENT_GENERATION_PILOT_MODE,
        session_id="historical-p6b",
        task_id="stair_climbing",
        backend_id="isaac_lab_rsl_rl",
        max_generations=1,
        max_trials=2,
        max_train_jobs=2,
        bound_contract_hashes=hashes,
    )


def _config(root: Path) -> StairGenerationPilotConfig:
    return StairGenerationPilotConfig(
        project_root=Path(__file__).resolve().parents[2],
        output_dir=root,
        lease_root=root.parent / "leases",
        session_id="historical-p6b",
        expected_source_commit=_SOURCE_COMMIT,
        machine_id="test-machine",
        gpu0_uuid=_GPU_A,
        gpu1_uuid=_GPU_B,
        provider_env_file=root.parent / "provider-not-used.env",
        max_allocated_active_gpu_hours=12.0,
        max_simulated_control_steps=1_000_000_000,
        max_output_bytes=40 * 1024**3,
        resume_existing=True,
    )


def _publish_terminal_fixture(root: Path, contract: GenerationSessionContract) -> None:
    root.mkdir()
    (root / GENERATION_SESSION_CONTRACT_FILENAME).write_bytes(
        canonical_json_bytes(contract.to_dict(), trailing_newline=True)
    )
    report = {
        "schema_version": STAIR_GENERATION_REPORT_SCHEMA_VERSION,
        "session_id": "historical-p6b",
        "session_mode": DEVELOPMENT_GENERATION_PILOT_MODE,
        "output_dir": str(root.resolve()),
        "generation_session_contract_hash": contract.contract_hash,
        "terminal_binding": {
            "schema_version": "true_rl_agent.generation_terminal_binding.v1",
            "acceptance_authority": False,
            "closed_generation_count": 1,
            "required_generation_count": 1,
        },
        "checks": {
            "mode_is_development_generation_pilot": True,
            "one_generation_one_ledger_record": True,
            "lifecycle_terminal": True,
        },
        "acceptance_authority": False,
    }
    (root / STAIR_GENERATION_REPORT_FILENAME).write_bytes(
        canonical_json_bytes(report, trailing_newline=True)
    )


class StairP6BTerminalContractResumeTests(unittest.TestCase):
    def test_terminal_p6b_reuses_stored_contract_when_only_authoring_policy_drifted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "formal"
            stored = _contract(authoring="stored")
            current = _contract(authoring="current")
            _publish_terminal_fixture(root, stored)
            before = {
                path.name: path.read_bytes() for path in sorted(root.iterdir())
            }

            with patch(
                "true_rl_agent.agent_core.generation_contract.create_once_bytes",
                side_effect=AssertionError("existing terminal resume must not write"),
            ) as create_once:
                actual, ref = _publish_or_load_terminal_p6b_session_contract(
                    _config(root),
                    current,
                )

            self.assertEqual(actual, stored)
            self.assertEqual(actual.contract_hash, stored.contract_hash)
            payload = (root / GENERATION_SESSION_CONTRACT_FILENAME).read_bytes()
            self.assertEqual(ref.size_bytes, len(payload))
            self.assertEqual(
                ref.checksum,
                "sha256:" + hashlib.sha256(payload).hexdigest(),
            )
            create_once.assert_not_called()
            self.assertEqual(
                {path.name: path.read_bytes() for path in sorted(root.iterdir())},
                before,
            )

    def test_terminal_p6b_rejects_non_authoring_bound_hash_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve() / "formal"
            stored = _contract(authoring="stored")
            current = _contract(authoring="current", provider="drifted-provider")
            _publish_terminal_fixture(root, stored)

            with self.assertRaisesRegex(
                ValueError,
                "bound hashes drifted: provider_policy",
            ):
                _publish_or_load_terminal_p6b_session_contract(
                    _config(root),
                    current,
                )


if __name__ == "__main__":
    unittest.main()
