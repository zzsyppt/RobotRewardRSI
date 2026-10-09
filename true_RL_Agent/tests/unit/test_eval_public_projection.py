from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.evaluation.public_projection import (
    assert_public_projection_safe,
    project_public_runner_error,
    project_public_train_summary,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import CheckpointRef

_CHECKPOINT_CHECKSUM = "sha256:" + "c" * 64


class EvalPublicProjectionTests(unittest.TestCase):
    def test_public_projection_key_policy_matches_shared_boundaries(self) -> None:
        ordinary = {
            "tokenizer": "v1",
            "tokenizer_version": "v1",
            "token_count": 12,
            "secretary_score": 0.8,
            "privateer_progress": 0.4,
            "authorization_status": "succeeded",
            "proxyAuthorizationStatus": "not_configured",
            "x_authorization": "public-diagnostic-label",
            "auth_header": "omitted",
        }
        assert_public_projection_safe(ordinary)

        for key in (
            "api_key",
            "raw_seed_id",
            "trusted_eval",
            "private_seed",
            "secret_token",
            "APIKey",
            "APIKEY",
            "apikey",
            "SECRETKEY",
            "PRIVATESEED",
            "summary_path",
            "checkpoint_output_path",
            "host_name",
            "hostname",
            "hostname_validated",
            "authorization",
            "Authorization",
            "proxy_authorization",
            "Proxy-Authorization",
            "proxyauthorization",
            "PROXYAUTHORIZATION",
            "PROXYAuthorization",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                assert_public_projection_safe({key: "must-not-pass"})

    def test_checkpoint_public_dict_omits_metadata_by_default_with_explicit_opt_in(self) -> None:
        checkpoint = CheckpointRef(
            uri="artifact://round/checkpoints/best.pt",
            checksum=_CHECKPOINT_CHECKSUM,
            role="eval",
            sealed=True,
            metadata={
                "local_path": "D:/runner/round/checkpoints/best.pt",
                "source_latest_checkpoint": "/data/run/model_42.pt",
            },
        )

        public = checkpoint.to_public_dict()

        self.assertNotIn("metadata", public)
        self.assertEqual(
            checkpoint.to_public_dict(include_metadata=True)["metadata"]["local_path"],
            "D:/runner/round/checkpoints/best.pt",
        )

    def test_train_result_public_dict_keeps_budget_but_drops_raw_runner_paths(self) -> None:
        subject = ExperimentSubject.candidate("candidate-001")
        subject_manifest_hash = "d" * 64
        program_source_checksum = "sha256:" + "e" * 64
        training_binding_hash = "f" * 64
        capability_profile_hash = "1" * 64
        execution_device = "cuda:0"
        result = TrainJobResult(
            job_id="round-001-train",
            subject=subject,
            subject_manifest_hash=subject_manifest_hash,
            program_source_checksum=program_source_checksum,
            training_binding_hash=training_binding_hash,
            capability_profile_hash=capability_profile_hash,
            checkpoint=CheckpointRef(
                uri="artifact://round-001/checkpoints/best.pt",
                checksum=_CHECKPOINT_CHECKSUM,
                role="eval",
                sealed=True,
                metadata={
                    "local_path": "D:/runner/round-001/checkpoints/best.pt",
                    "source_latest_checkpoint": "/data/isaac/runs/model_42.pt",
                    "training_subject": subject.to_public_dict(),
                    "subject_manifest_hash": subject_manifest_hash,
                    "program_source_checksum": program_source_checksum,
                    "training_binding_hash": training_binding_hash,
                    "capability_profile_hash": capability_profile_hash,
                    "execution_device": execution_device,
                },
            ),
            execution_device=execution_device,
            summary={
                "formal_training_executed": True,
                "run": {"cwd": "D:/runner/round-001/train"},
                "training": {
                    "budget_accounting": {"status": "pass", "planned_env_steps": 49152000},
                    "log_dir": "/data/isaac/runs/round-001",
                    "checkpoint_output": "D:/runner/round-001/checkpoints/best.pt",
                },
                "configuration": {"config_path": "/data/configs/push_recovery.yaml"},
            },
        )

        public = result.to_public_dict()
        text = json.dumps(public, sort_keys=True)

        self.assertEqual(public["summary"]["training"]["budget_accounting"]["status"], "pass")
        for forbidden in (
            "metadata",
            "local_path",
            "source_latest_checkpoint",
            "D:/runner",
            "/data/",
            "cwd",
            "config_path",
            "log_dir",
            "checkpoint_output",
        ):
            self.assertNotIn(forbidden, text)

        with self.assertRaisesRegex(ValueError, "execution device mismatch"):
            replace(result, execution_device="cuda:1")

    def test_public_eval_result_rejects_paths_and_raw_seed_markers(self) -> None:
        unsafe_payloads = (
            {"summary_path": "D:/runner/round-001/eval/summary.json"},
            {"message": "raw_seed=123"},
            {"hidden_profile": "iid_hidden_eval"},
        )

        for metrics in unsafe_payloads:
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    PublicEvalResult(status="pass", public_score=1.0, metrics=metrics).to_public_dict()

    def test_public_projection_uses_shared_standalone_secret_boundary(self) -> None:
        ordinary = (
            "Task-neutral task-specific risk-free disk-usage "
            "draw_seed draw-seed notprivate_seeded notiid_hidden "
            "anti-iid-hidden "
            "counts_only_no_raw_seed_ids"
        )
        assert_public_projection_safe({"message": ordinary})

        for forbidden in (
            "sk-lower123",
            "SK-UPPER456",
            "iid_hidden_eval",
            "IID-HIDDEN cohort",
            "Bearer abc123",
            "Authorization: Bearer abc123",
            "Proxy-Authorization: Bearer abc123",
            "request failed: Authorization: Bearer abc123",
            'request headers={"Authorization":"Basic abcdef"}',
            r'request headers={\"Authorization\":\"Basic abcdef\"}',
        ):
            with self.subTest(forbidden=forbidden), self.assertRaises(ValueError):
                assert_public_projection_safe({"message": forbidden})

        for ordinary_bearer_text in (
            "bearer of good news",
            "upstream sent Bearer abc123",
            "upstream sent Bearer abcdef",
            "execution_authorization: state succeeded",
        ):
            with self.subTest(ordinary_bearer_text=ordinary_bearer_text):
                assert_public_projection_safe({"message": ordinary_bearer_text})

        projected_train = project_public_train_summary(
            {
                "schema_version": "bearer is ready",
                "status": "authorization succeeded",
                "errors": ["upstream sent Bearer abcdef"],
            }
        )
        self.assertEqual(projected_train["schema_version"], "bearer is ready")
        self.assertEqual(projected_train["status"], "authorization succeeded")
        self.assertEqual(projected_train["errors"], ["[redacted]"])

        runner_error = project_public_runner_error(
            RuntimeError(f"{ordinary} sk-lower123 SK-UPPER456")
        )

        self.assertEqual(
            runner_error["message"],
            f"{ordinary} [redacted] [redacted]",
        )
        assert_public_projection_safe(runner_error)

        credential_error = project_public_runner_error(
            RuntimeError("upstream sent Bearer abcdef")
        )
        self.assertEqual(credential_error["message"], "[redacted]")
        assert_public_projection_safe(credential_error)

        detailed_error = RuntimeError("ordinary runner failure")
        detailed_error.public_details = {
            "upstream sent Bearer abcdef": "must-not-survive",
            'request headers={"Authorization":"Basic abcdef"}': (
                "must-not-survive"
            ),
            r'request headers={\"Authorization\":\"Basic abcdef\"}': (
                "must-not-survive"
            ),
            "safe_detail": "upstream sent Bearer abcdef",
        }
        projected_details = project_public_runner_error(detailed_error)
        details_text = json.dumps(projected_details, sort_keys=True)
        self.assertNotIn("must-not-survive", details_text)
        self.assertNotIn("Bearer abcdef", details_text)
        self.assertEqual(projected_details["details"]["safe_detail"], "[redacted]")
        assert_public_projection_safe(projected_details)

    def test_public_path_detection_covers_common_container_roots_without_blocking_artifact_uri(self) -> None:
        for raw_path in (
            "/workspace/run/eval/summary.json",
            "/opt/isaac/run/eval/summary.json",
        ):
            with self.subTest(raw_path=raw_path):
                self.assertEqual(sanitize_prompt_public_value(raw_path), "[path]")
                with self.assertRaises(ValueError):
                    assert_public_projection_safe({"message": raw_path})
                runner_error = project_public_runner_error(RuntimeError(f"eval failed at {raw_path}"))
                text = json.dumps(runner_error, sort_keys=True)
                self.assertIn("[path]", runner_error["message"])
                self.assertNotIn(raw_path, text)

        artifact_uri = "artifact://round-001/checkpoints/best.pt"
        self.assertEqual(sanitize_prompt_public_value(artifact_uri), artifact_uri)
        assert_public_projection_safe({"checkpoint_uri": artifact_uri})


if __name__ == "__main__":
    unittest.main()
