from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.evaluation.score_stats import validate_score_stats
from true_rl_agent.json_contract import canonical_json_bytes
from true_rl_agent.tasks.stair_climbing import phase0_contract
from true_rl_agent.tasks.stair_climbing.evaluation_contract import (
    aggregate_stair_seed_scores,
    assert_primary_outcome_counts,
)
from true_rl_agent.tasks.stair_climbing.phase0_contract import (
    PHASE0_SOURCE_ARTIFACT_REFS,
    StairBaselineManifestV1,
    StairExtensionDescriptorV1,
    StairPhase0Bundle,
    StairPhase0ReportV1,
    StairProtocolV1,
    canonical_json_sha256,
    resolve_phase0_terminal_reason,
    verify_checkpoint_file,
)


def _rehash(payload: dict, key: str) -> None:
    payload[key] = canonical_json_sha256(
        {name: value for name, value in payload.items() if name != key}
    )


def _refresh_rollout_summary(payload: dict, protocol: dict) -> None:
    terminal_reasons = tuple(
        dict.fromkeys(
            protocol["phase0_probe"][
                "terminal_reason_by_manager_term"
            ].values()
        )
    )
    payload["rollout"]["summary_per_bin"] = {
        bin_id: {
            "episode_count": len(records),
            "terminal_reason_counts": {
                reason: sum(
                    record["terminal_reason"] == reason for record in records
                )
                for reason in terminal_reasons
            },
            "reached_first_step_count": sum(
                bool(record["reached_first_step"]) for record in records
            ),
            "fall_detected_count": sum(
                bool(record["fall_detected"]) for record in records
            ),
            "success_count": sum(bool(record["success"]) for record in records),
            "mean_max_progress_m": round(
                sum(float(record["max_progress_m"]) for record in records)
                / len(records),
                8,
            ),
        }
        for bin_id, records in payload["rollout"]["first_episodes"].items()
    }


def _write_bundle_resources(
    root: Path,
    *,
    descriptor: dict,
    protocol: dict,
    manifest: dict,
) -> None:
    descriptor_path = root / "configs" / "tasks" / "stair_extension_schema.json"
    protocol_path = root / descriptor["task_protocol"]["ref"]
    manifest_path = root / descriptor["baseline_manifest"]["ref"]
    for path, payload in (
        (descriptor_path, descriptor),
        (protocol_path, protocol),
        (manifest_path, manifest),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            canonical_json_bytes(payload, trailing_newline=True)
        )


class StairPhase0ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = StairPhase0Bundle.load(PROJECT_ROOT)
        self.protocol_path = PROJECT_ROOT / self.bundle.protocol_ref
        self.manifest_path = PROJECT_ROOT / self.bundle.baseline_manifest_ref
        self.descriptor_path = PROJECT_ROOT / self.bundle.descriptor_ref
        self.report_path = PROJECT_ROOT / self.bundle.phase0_report_ref
        self.protocol_payload = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.manifest_payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def test_immutable_protocol_rejects_duplicate_keys_and_noncanonical_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            duplicate_path = Path(tmp) / "duplicate.json"
            canonical = canonical_json_bytes(
                self.protocol_payload,
                trailing_newline=True,
            )
            duplicate_path.write_bytes(
                b'{"task_id":"forged-task",' + canonical[1:]
            )

            with self.assertRaisesRegex(ValueError, "invalid JSON resource"):
                StairProtocolV1.load(duplicate_path)

            noncanonical_path = Path(tmp) / "noncanonical.json"
            noncanonical_path.write_text(
                json.dumps(
                    self.protocol_payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid JSON resource"):
                StairProtocolV1.load(noncanonical_path)

    def test_frozen_bundle_and_admitted_report_load(self) -> None:
        protocol = self.bundle.protocol
        self.assertEqual(protocol.task_id, "unitree_stair_climbing")
        self.assertEqual(len(protocol.eval_bins), 7)
        self.assertEqual(protocol.evaluation_protocol.eval_num_envs, 56)
        self.assertEqual(protocol.evaluation_protocol.public_accept_seed_count, 5)
        self.assertEqual(protocol.acceptance_protocol.min_score_count, 5)
        self.assertEqual(self.bundle.baseline.checkpoint_size_bytes, 7_842_530)
        self.assertFalse(self.bundle.baseline.to_dict()["accepted_current_best"])

        report = self.bundle.load_report()
        payload = report.to_dict()
        self.assertEqual(payload["environment"]["policy_observation_shape"], [56, 310])
        self.assertTrue(payload["admission"]["warm_start_candidate_admitted"])
        self.assertFalse(payload["admission"]["accepted_current_best"])
        self.assertEqual(payload["rollout"]["control_steps_limit"], 600)

    def test_descriptor_is_the_only_resource_entrypoint(self) -> None:
        descriptor = StairExtensionDescriptorV1.load(self.descriptor_path).to_dict()
        self.assertEqual(
            set(descriptor),
            {
                "schema_version",
                "task_id",
                "task_protocol",
                "baseline_manifest",
                "phase0_report",
                "descriptor_hash",
            },
        )
        serialized = self.descriptor_path.read_text(encoding="utf-8")
        for duplicated_semantic_key in (
            "fixed_eval_bins",
            "future_failure_taxonomy",
            "runtime_plugin_enabled",
            "training_command_enabled",
            "stair_height_m",
        ):
            self.assertNotIn(duplicated_semantic_key, serialized)

    def test_descriptor_is_the_only_frozen_identity_anchor(self) -> None:
        frozen_names = sorted(
            name
            for name in dir(phase0_contract)
            if name.startswith("FROZEN_STAIR_")
        )
        self.assertEqual(
            frozen_names,
            ["FROZEN_STAIR_EXTENSION_DESCRIPTOR_V1_HASH"],
        )

        descriptor = json.loads(self.descriptor_path.read_text(encoding="utf-8"))
        descriptor["task_protocol"]["ref"] = "configs/tasks/other.json"
        _rehash(descriptor, "descriptor_hash")
        with self.assertRaisesRegex(ValueError, "frozen"):
            StairExtensionDescriptorV1.from_mapping(descriptor)

        protocol = copy.deepcopy(self.protocol_payload)
        protocol["protocol_id"] = "unitree_stair_climbing_v1_rehashed"
        _rehash(protocol, "task_protocol_hash")
        parsed_protocol = StairProtocolV1.from_mapping(protocol)
        self.assertEqual(parsed_protocol.protocol_id, protocol["protocol_id"])

        manifest = copy.deepcopy(self.manifest_payload)
        manifest["source"]["provider"] = "repacked mirror"
        _rehash(manifest, "manifest_hash")
        parsed_manifest = StairBaselineManifestV1.from_mapping(manifest)
        self.assertEqual(
            parsed_manifest.to_dict()["source"]["provider"],
            "repacked mirror",
        )

        canonical_descriptor = json.loads(
            self.descriptor_path.read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bundle_resources(
                root,
                descriptor=canonical_descriptor,
                protocol=protocol,
                manifest=self.manifest_payload,
            )
            with self.assertRaisesRegex(ValueError, "descriptor binding mismatch"):
                StairPhase0Bundle.load(root)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_bundle_resources(
                root,
                descriptor=canonical_descriptor,
                protocol=self.protocol_payload,
                manifest=manifest,
            )
            with self.assertRaisesRegex(ValueError, "descriptor binding mismatch"):
                StairPhase0Bundle.load(root)

    def test_protocol_rejects_rehashed_semantic_weakening(self) -> None:
        mutations = (
            (
                "required_pass_gates",
                lambda payload: payload["acceptance"].__setitem__(
                    "required_pass_gates", []
                ),
            ),
            (
                "top platform",
                lambda payload: payload["terrain"]["goal"].__setitem__(
                    "progress_goal_m_from_origin", 2.0
                ),
            ),
            (
                "spawn x range",
                lambda payload: payload["terrain"]["spawn"].__setitem__(
                    "x_range_m_from_origin", [-10.0, 10.0]
                ),
            ),
            (
                "timeout threshold",
                lambda payload: payload["episode_outcomes"]["signals"][
                    "timeout"
                ].__setitem__("threshold", 7.0),
            ),
            (
                "generation_mode",
                lambda payload: payload["terrain"]["training_distribution"].__setitem__(
                    "generation_mode", 123
                ),
            ),
            (
                "runtime_controls events",
                lambda payload: payload["runtime_controls"]["events"].__setitem__(
                    "push_robot_enabled", True
                ),
            ),
            (
                "termination schema",
                lambda payload: payload["phase0_probe"]["termination_schema"][
                    "terms"
                ][1].__setitem__("time_out", True),
            ),
            (
                "terminal reason mapping",
                lambda payload: payload["phase0_probe"][
                    "terminal_reason_by_manager_term"
                ].__setitem__("base_contact", "timeout"),
            ),
            (
                "evaluation lateral command",
                lambda payload: payload["evaluation"].__setitem__(
                    "lateral_command_mps", 0.5
                ),
            ),
        )
        for expected, mutate in mutations:
            with self.subTest(expected=expected):
                payload = copy.deepcopy(self.protocol_payload)
                mutate(payload)
                _rehash(payload, "task_protocol_hash")
                with self.assertRaisesRegex(ValueError, expected):
                    StairProtocolV1.from_mapping(payload)

    def test_protocol_explicitly_owns_runtime_controls(self) -> None:
        protocol = self.bundle.protocol.to_dict()
        self.assertEqual(
            set(protocol["runtime_controls"]),
            {"reset_velocity_ranges", "command", "events"},
        )
        self.assertEqual(protocol["evaluation"]["lateral_command_mps"], 0.0)
        self.assertEqual(protocol["evaluation"]["yaw_command_radps"], 0.0)
        self.assertEqual(
            protocol["runtime_controls"]["command"]["resampling"],
            "episode_length",
        )
        self.assertTrue(
            all(
                bounds == [0.0, 0.0]
                for bounds in protocol["runtime_controls"][
                    "reset_velocity_ranges"
                ].values()
            )
        )

    def test_phase0_terminal_reason_resolution_is_task_owned_and_priority_ordered(
        self,
    ) -> None:
        protocol = self.bundle.protocol.to_dict()
        self.assertEqual(
            resolve_phase0_terminal_reason(protocol, ["time_out"]),
            "timeout",
        )
        self.assertEqual(
            resolve_phase0_terminal_reason(
                protocol,
                ["time_out", "base_contact"],
            ),
            "base_contact",
        )
        with self.assertRaisesRegex(ValueError, "unmapped termination term"):
            resolve_phase0_terminal_reason(protocol, ["future_fall_term"])

    def test_runtime_policy_schema_names_real_manager_term_and_callable(self) -> None:
        policy = self.bundle.protocol.to_dict()["policy_compatibility"]
        terms = policy["policy_observation"]["terms"]
        self.assertEqual([item["manager_term"] for item in terms][6], "actions")
        self.assertEqual(
            terms[6]["callable"],
            "isaaclab.envs.mdp.observations.last_action",
        )
        self.assertEqual(
            policy["action"]["terms"][0]["resolved_joint_names"],
            policy["robot_schema"]["joint_names"],
        )

    def test_project_resource_ref_rejects_absolute_and_root_escape(self) -> None:
        with self.assertRaisesRegex(ValueError, "project-relative"):
            StairPhase0Bundle.load(
                PROJECT_ROOT,
                descriptor_ref=self.descriptor_path.resolve(),
            )
        with tempfile.TemporaryDirectory() as tmp:
            container = Path(tmp)
            root = container / "project"
            root.mkdir()
            (container / "outside.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "escapes project root"):
                StairPhase0Bundle.load(
                    root,
                    descriptor_ref="../outside.json",
                )

    def test_checkpoint_bytes_are_verified_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.pt"
            checkpoint.write_bytes(b"trusted-test-checkpoint")
            expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            manifest = SimpleNamespace(
                checkpoint_size_bytes=checkpoint.stat().st_size,
                checkpoint_sha256=expected_hash,
            )

            result = verify_checkpoint_file(checkpoint, manifest)
            self.assertEqual(result.sha256, expected_hash)
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint size mismatch"):
                verify_checkpoint_file(checkpoint, manifest)

    def test_probe_uses_shared_checkpoint_safety_and_never_unsafe_torch_load(self) -> None:
        probe_path = PROJECT_ROOT / "scripts" / "isaaclab_stair_phase0_probe.py"
        text = probe_path.read_text(encoding="utf-8")
        self.assertIn("build_checkpoint_safety_report", text)
        self.assertIn("validate_checkpoint_safety_report", text)
        self.assertIn("weights_only=True", text)
        self.assertNotIn("weights_only=False", text)

    def test_phase0_source_artifact_list_has_one_owner(self) -> None:
        probe_path = PROJECT_ROOT / "scripts" / "isaaclab_stair_phase0_probe.py"
        text = probe_path.read_text(encoding="utf-8")
        self.assertIn("PHASE0_SOURCE_ARTIFACT_REFS", text)
        self.assertNotIn("refs = (", text)
        for ref in PHASE0_SOURCE_ARTIFACT_REFS:
            self.assertTrue((PROJECT_ROOT / ref).is_file(), ref)

    def test_phase0_plans_do_not_copy_derived_artifact_hashes(self) -> None:
        hash_pattern = re.compile(r"\b[0-9a-f]{64}\b")
        for path in sorted((PROJECT_ROOT / "PLAN" / "stair").glob("*.md")):
            with self.subTest(path=path.name):
                self.assertIsNone(hash_pattern.search(path.read_text(encoding="utf-8")))

    def test_report_rejects_forged_runtime_identity_and_admission(self) -> None:
        payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        mutations = []

        fake_joints = copy.deepcopy(payload)
        robot_schema = fake_joints["environment"]["robot_schema"]
        robot_schema["joint_names"] = [f"fake_{index}" for index in range(37)]
        _rehash(robot_schema, "schema_hash")
        mutations.append((fake_joints, "robot_schema"))

        only_torso = copy.deepcopy(payload)
        robot_schema = only_torso["environment"]["robot_schema"]
        robot_schema["body_names"] = ["torso_link"]
        _rehash(robot_schema, "schema_hash")
        mutations.append((only_torso, "robot_schema"))

        bogus_origins = copy.deepcopy(payload)
        bogus_origins["environment"]["terrain_grid_origins_m"] = [
            [999.0, 999.0, 999.0]
        ] * 7
        mutations.append((bogus_origins, "terrain_grid_origins_m"))

        truncated_levels = copy.deepcopy(payload)
        truncated_levels["environment"]["terrain_levels"] = [0]
        mutations.append((truncated_levels, "terrain_levels"))

        wrong_term = copy.deepcopy(payload)
        observation_schema = wrong_term["environment"]["policy_observation_schema"]
        observation_schema["terms"][6]["manager_term"] = "last_action"
        _rehash(observation_schema, "schema_hash")
        mutations.append((wrong_term, "policy_observation_schema"))

        wrong_termination = copy.deepcopy(payload)
        termination_schema = wrong_termination["environment"]["termination_schema"]
        termination_schema["terms"][1]["parameters"]["threshold"] = 2.0
        _rehash(termination_schema, "schema_hash")
        mutations.append((wrong_termination, "termination_schema"))

        wrong_terminal_terms = copy.deepcopy(payload)
        first_record = next(
            iter(wrong_terminal_terms["rollout"]["first_episodes"].values())
        )[0]
        first_record["terminal_terms"] = ["time_out"]
        mutations.append((wrong_terminal_terms, "terminal reason mismatch"))

        all_early_done = copy.deepcopy(payload)
        for records in all_early_done["rollout"]["first_episodes"].values():
            for record in records:
                record.update(
                    {
                        "terminal_reason": "base_contact",
                        "terminal_terms": ["base_contact"],
                        "episode_steps": 1,
                        "max_progress_m": 0.0,
                        "reached_first_step": False,
                        "max_top_hold_steps": 0,
                        "success": False,
                        "max_fall_consecutive_steps": 0,
                        "fall_detected": False,
                    }
                )
        all_early_done["rollout"]["control_steps_executed"] = 1
        _refresh_rollout_summary(all_early_done, self.protocol_payload)
        mutations.append((all_early_done, "task viability gate"))

        impossible_counter = copy.deepcopy(payload)
        first_record = next(
            iter(impossible_counter["rollout"]["first_episodes"].values())
        )[0]
        first_record["max_top_hold_steps"] = first_record["episode_steps"] + 1
        mutations.append((impossible_counter, "top-hold steps exceed"))

        for mutated, expected in mutations:
            with self.subTest(expected=expected):
                _rehash(mutated, "report_hash")
                with self.assertRaisesRegex(ValueError, expected):
                    report = StairPhase0ReportV1.from_mapping(mutated)
                    report.assert_matches(self.bundle.protocol, self.bundle.baseline)

    def test_valid_failed_behavior_report_is_not_admission_evidence(self) -> None:
        payload = json.loads(self.report_path.read_text(encoding="utf-8"))
        for records in payload["rollout"]["first_episodes"].values():
            for record in records:
                record.update(
                    {
                        "terminal_reason": "base_contact",
                        "terminal_terms": ["base_contact"],
                        "episode_steps": 1,
                        "max_progress_m": 0.0,
                        "reached_first_step": False,
                        "max_top_hold_steps": 0,
                        "success": False,
                        "max_fall_consecutive_steps": 0,
                        "fall_detected": False,
                    }
                )
        payload["rollout"]["control_steps_executed"] = 1
        _refresh_rollout_summary(payload, self.protocol_payload)
        payload["status"] = "fail"
        payload["admission"]["task_viability_gate_passed"] = False
        payload["admission"]["warm_start_candidate_admitted"] = False
        _rehash(payload, "report_hash")

        report = StairPhase0ReportV1.from_mapping(payload)
        report.assert_matches(self.bundle.protocol, self.bundle.baseline)
        with self.assertRaisesRegex(ValueError, "not admitted"):
            report.assert_warm_start_admitted()

    def test_phase0_report_has_no_machine_local_checkpoint_path(self) -> None:
        text = self.report_path.read_text(encoding="utf-8")
        self.assertNotIn("/data/", text)
        self.assertNotIn("D:\\", text)

    def test_seven_bin_macro_statistics_use_one_value_per_seed(self) -> None:
        protocol = self.bundle.protocol
        bin_ids = [item["bin_id"] for item in protocol.eval_bins]
        seed_counts = [
            {
                bin_id: max(0, 8 - seed_index - bin_index)
                for bin_index, bin_id in enumerate(bin_ids)
            }
            for seed_index in range(5)
        ]
        aggregate = aggregate_stair_seed_scores(
            protocol,
            split="public_accept",
            per_seed_success_counts=seed_counts,
        )
        validation = validate_score_stats(
            aggregate["score_stats"],
            minimum_count=5,
            require_std=True,
            require_standard_error=True,
            require_protocol_hash=True,
        )
        self.assertEqual(len(aggregate["seed_macro_scores"]), 5)
        self.assertTrue(validation.valid, validation.reasons)

    def test_primary_outcome_count_contract_is_closed_and_warning_blocking(self) -> None:
        protocol = self.bundle.protocol
        reasons = protocol.to_dict()["episode_outcomes"]["primary_reasons"]
        counts = {reason: 0 for reason in reasons}
        counts["success"] = 3
        counts["base_contact"] = 2
        counts["timeout"] = 3
        assert_primary_outcome_counts(
            protocol,
            total_episode_count=8,
            primary_counts=counts,
        )
        with self.assertRaisesRegex(ValueError, "count invariant"):
            assert_primary_outcome_counts(
                protocol,
                total_episode_count=9,
                primary_counts=counts,
            )
        with self.assertRaisesRegex(ValueError, "warnings block"):
            assert_primary_outcome_counts(
                protocol,
                total_episode_count=8,
                primary_counts=counts,
                classification_warning_count=1,
            )

    def test_package_import_is_inert_without_isaac_or_gym_registration(self) -> None:
        code = (
            "import json,sys;"
            "import true_rl_agent.tasks.stair_climbing;"
            "print(json.dumps({'isaac': any(k == 'isaaclab' or k.startswith('isaaclab.') "
            "for k in sys.modules), 'gym': 'gymnasium' in sys.modules}))"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONPATH": str(SRC_ROOT)},
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            json.loads(completed.stdout),
            {"isaac": False, "gym": False},
        )

    def test_phase0_assets_have_no_null_or_unresolved_markers(self) -> None:
        for path in (self.descriptor_path, self.protocol_path, self.manifest_path):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("TODO", text.upper())
            self.assertNotIn("TBD", text.upper())
            self.assertNotIn("null", text)


if __name__ == "__main__":
    unittest.main()
