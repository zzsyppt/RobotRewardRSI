from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.agent_core.public_safety import sanitize_prompt_public_value
from true_rl_agent.evaluation.public_projection import (
    assert_public_projection_safe,
    project_public_runner_error,
)
from true_rl_agent.evaluation.public_result import PublicEvalResult
from true_rl_agent.evaluation.subject import ExperimentSubject
from true_rl_agent.runner.base import TrainJobResult
from true_rl_agent.runner.specs import CheckpointRef

_CHECKPOINT_CHECKSUM = "sha256:" + "c" * 64


class EvalPublicProjectionTests(unittest.TestCase):
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
                },
            ),
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
