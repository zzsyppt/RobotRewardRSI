from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.binding import (
    canonical_json_hash,
    require_sha256_checksum,
    require_sha256_digest,
)
from true_rl_agent.tasks.stair_climbing.config import (
    STAIR_TASK_ID,
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.protocol_v2 import (
    PREVIOUS_STAIR_PROTOCOL_V2_HASH,
)
from true_rl_agent.tasks.stair_climbing.reference_program import (
    resolve_official_stair_reference_program,
)

MANIFEST_PATH = PROJECT_ROOT / "PLAN" / "stair" / "phase_e_completion_manifest.json"


class StairPhaseECompletionManifestTests(unittest.TestCase):
    def test_manifest_closes_formal_phase_e_evidence(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        declared_hash = require_sha256_digest(
            manifest.pop("manifest_hash"),
            "Phase E completion manifest hash",
        )
        self.assertEqual(canonical_json_hash(manifest), declared_hash)
        self.assertEqual(
            manifest["schema_version"],
            "true_rl_agent.stair.phase_e_completion.v1",
        )
        self.assertEqual(manifest["phase"], "E")
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["task_id"], STAIR_TASK_ID)

        session = manifest["session"]
        self.assertEqual(session["session_id"], "stair-r0-formal-aacaee2")
        self.assertEqual(session["ledger_state"], "absent")
        self.assertTrue(session["artifact_root_uri"].startswith("ssh://"))
        require_sha256_digest(
            session["embedded_session_genesis_hash"],
            "embedded session Genesis hash",
        )

        execution = manifest["execution"]
        self.assertEqual(execution["policy_initialization"], "random")
        self.assertEqual(execution["execution_device"], "cuda:1")
        self.assertEqual(execution["budget"]["max_iterations"], 2000)
        self.assertEqual(
            execution["budget"]["budget_steps"],
            execution["budget"]["num_envs"]
            * execution["budget"]["num_steps_per_env"]
            * execution["budget"]["max_iterations"],
        )
        for field_name in (
            "formal_training_code_commit",
            "load_only_verification_code_commit",
        ):
            commit_id = execution[field_name]
            self.assertEqual(len(commit_id), 40, field_name)
            self.assertEqual(commit_id, commit_id.lower(), field_name)
            self.assertGreaterEqual(int(commit_id, 16), 0, field_name)

        config = StairClimbingTaskConfig.defaults(
            PROJECT_ROOT / "configs" / "tasks" / "stair_protocol_v2.json",
        )
        official = resolve_official_stair_reference_program(
            config,
            project_root=PROJECT_ROOT,
        )
        lineage = manifest["lineage"]
        self.assertEqual(
            lineage["task_protocol_hash"],
            PREVIOUS_STAIR_PROTOCOL_V2_HASH,
        )
        self.assertEqual(
            config.protocol_v2.runner_owned_training_distribution[
                "selection_provenance"
            ]["previous_training_protocol_hash"],
            lineage["task_protocol_hash"],
        )
        self.assertEqual(
            lineage["capability_profile_hash"],
            config.protocol_v2.capability_profile.profile_hash,
        )
        self.assertEqual(
            lineage["reference_manifest_hash"],
            official.artifact.manifest.manifest_hash,
        )
        self.assertEqual(
            require_sha256_checksum(
                lineage["reward_source_checksum"],
                "reward source checksum",
            ),
            official.artifact.manifest.source_checksum,
        )
        self.assertEqual(
            manifest["artifacts"]["reference_reward_source"]["sha256"],
            official.artifact.manifest.source_checksum.removeprefix("sha256:"),
        )

        expected_artifact_roles = {
            "checkpoint",
            "evaluation_summary",
            "reference_artifact_manifest",
            "reference_private_evidence",
            "reference_reward_source",
            "session_genesis",
            "training_summary",
        }
        self.assertEqual(set(manifest["artifacts"]), expected_artifact_roles)
        for role, artifact in manifest["artifacts"].items():
            self.assertFalse(Path(artifact["relative_path"]).is_absolute(), role)
            self.assertNotIn("..", Path(artifact["relative_path"]).parts, role)
            require_sha256_digest(artifact["sha256"], f"{role} sha256")

        gate_statuses = manifest["evaluation"]["gate_statuses"]
        self.assertEqual(
            set(gate_statuses),
            {
                "accept_evidence_gate",
                "budget_accounting",
                "reference_capability_gate",
                "runtime_activation_gate",
                "sealed_validation_gate",
                "stair_behavior_gate",
            },
        )
        self.assertEqual(set(gate_statuses.values()), {"pass"})
        capability = manifest["evaluation"]["reference_capability"]
        self.assertGreaterEqual(
            capability["observed_macro_mean_success_rate"],
            capability["minimum_macro_mean_success_rate"],
        )
        self.assertGreaterEqual(
            capability["observed_macro_mean_max_progress_m"],
            capability["minimum_macro_mean_max_progress_m"],
        )

        load_only = manifest["load_only_replay"]
        self.assertEqual(load_only["status"], "pass")
        self.assertEqual(load_only["exit_code"], 0)
        self.assertTrue(load_only["evidence_root_uri"].startswith("ssh://"))
        self.assertTrue(load_only["genesis_bytes_unchanged"])
        self.assertTrue(load_only["genesis_mtime_unchanged"])
        self.assertEqual(load_only["ledger_state_before"], "absent")
        self.assertEqual(load_only["ledger_state_after"], "absent")
        self.assertEqual(load_only["ledger_record_count"], 0)
        self.assertEqual(load_only["runner_call_count"], 0)
        self.assertEqual(load_only["provider_call_count"], 0)
        self.assertEqual(load_only["matching_runtime_processes_before"], 0)
        self.assertEqual(load_only["matching_runtime_processes_after"], 0)
        self.assertEqual(load_only["gpu_processes"]["new_process_count"], 0)
        self.assertEqual(load_only["reference_source_status"], "pass")
        self.assertEqual(load_only["replay_status"], "pass")

        creation = manifest["formal_creation"]
        self.assertEqual(creation["status"], "pass")
        self.assertEqual(creation["exit_code"], 0)
        self.assertEqual(creation["launcher_process_after"], "absent")
        self.assertTrue(creation["control_root_uri"].startswith("ssh://"))


if __name__ == "__main__":
    unittest.main()
