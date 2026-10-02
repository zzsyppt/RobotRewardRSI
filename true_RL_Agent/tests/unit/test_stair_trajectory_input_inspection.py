from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from true_rl_agent.runner.numeric_evidence import NumericEvidenceError
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("inspect_stair_trajectory_input", ROOT / "scripts/inspect_stair_trajectory_input.py")
inspection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspection)


class StairTrajectoryInputInspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = StairClimbingTaskConfig.defaults(ROOT / "configs/tasks/stair_go2_12dof_unified_protocol_v3.json")
        cls.resolved = resolve_stair_cell_manifest(cell_variant_id=cls.config.cell_variant_id,
            config=cls.config, plugin=StairClimbingTaskPlugin(cls.config))
        cls.projection = stair_runtime_projection_from_manifest(cls.resolved)
        cls.cells = cls.config.ordered_evaluation_suite.private_cell_assignment("public_dev")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive_count = 0

    def _archive(self, mutate=None):
        """Small synthetic first episodes, using real cell/robot contracts and writer."""
        self.archive_count += 1
        identity = {"job_id": "synthetic-input-audit", "subject_kind": "reference", "subject_id": "synthetic",
                    "task_id": "unitree_stair_climbing", "checkpoint_hash": "sha256:" + "a" * 64,
                    "seed_namespace": "synthetic_public_dev", "splits": ["public_dev"],
                    "resolved_cell_manifest_hash": self.resolved.manifest_hash,
                    "evaluation_binding": {"task_protocol_hash": self.resolved.resolved_cell_task_protocol_hash}}
        capture = StairNumericCapture(self.root / str(self.archive_count), identity=identity,
                                      config=self.config, resolved_cell_manifest=self.resolved)
        n, p = len(self.cells), self.projection
        context = copy.deepcopy(capture.context)
        history = {"schema_version": "true_rl_agent.stair.contact_force_history.v1",
                   "source": "contact_sensor.net_forces_w_history",
                   "force_semantics": "world_frame_net_normal_contact_force_N_not_contact_pair_or_surface",
                   "history_length": 3, "history_order": "newest_to_oldest",
                   "history_clock": "contact_sensor_buffer_update_events",
                   "configured_sensor_update_period_s": 0.005, "physics_dt_s": 0.005,
                   "control_dt_s": 0.02, "physics_steps_per_control": 4, "lazy_sensor_update": True,
                   "per_history_slot_timestamps": "not_recorded; configured_update_period_is_not_observed_sample_spacing",
                   "coverage": "existing_sensor_buffer_at_pre_reset_control_step_end; not_full_physics_trajectory",
                   "reset_history": "sensor_zeroes_history_on_reset; no_per_slot_validity_flags"}
        context["contact_force_history"] = {**history, "status": "recorded",
            "array_axes": ["control_step", "environment", "history", "bound_body", "xyz"]}
        lengths = np.full(n, 3, dtype=np.int64)
        lengths[:2] = [1, 2]
        outcomes = [{"bin_id": cell.bin_id, "evaluation_cell_id": cell.cell_id,
                     "episode_steps": int(length), "success": index > 1,
                     "primary_reason": ("forbidden_contact", "timeout")[index] if index < 2 else "success",
                     "diagnostic_flags": [], "max_progress_m": 0.1, "progress_fraction": 0.01}
                    for index, (length, cell) in enumerate(zip(lengths, self.cells, strict=True))]
        context["seeds"] = [{"run_index": 1, "runner_seed": 12, "split": "public_dev", "step_dt": 0.02,
                              "observed_control_steps": 3, "episode_steps": lengths.tolist(), "outcomes": outcomes}]
        context["summary"] = {"fixture": "synthetic; input integrity only, no physical outcome claim"}
        shape_tails = {"base_pos_w": (3,), "base_quat_w": (4,), "joint_pos": (p.joint_count,),
                       "support_pos_w": (p.support_count, 3), "support_lin_vel_w": (p.support_count, 3),
                       "support_contact_forces_w": (p.support_count, 3),
                       "forbidden_contact_forces_w": (p.forbidden_contact_count, 3),
                       "support_contact_forces_w_history": (3, p.support_count, 3),
                       "forbidden_contact_forces_w_history": (3, p.forbidden_contact_count, 3)}
        arrays = {key: np.zeros((3, n, *tail), dtype=np.float32) for key, tail in shape_tails.items()}
        arrays["base_quat_w"][:, :, 0] = 1
        arrays["joint_pos"][:] = 0.12
        arrays["base_pos_w"][:, :, 0] = np.arange(1, 4)[:, None] * 0.1
        arrays["support_contact_forces_w"][:, :, :, 2] = 10
        indices = np.arange(1, 4)[:, None]
        arrays["valid"] = indices <= lengths[None, :]
        arrays["episode_length"] = np.broadcast_to(indices, (3, n)).copy()
        arrays["done_mask"] = indices == lengths[None, :]
        # Success closes the scored episode without requiring a manager done.
        arrays["done_mask"][:, 2:] = False
        arrays["wrapper_done_mask"] = arrays["done_mask"].copy()
        arrays["termination_0"] = arrays["done_mask"].copy()
        arrays["termination_0"][:, 0] = False
        arrays["termination_1"] = np.zeros((3, n), dtype=np.bool_)
        arrays["termination_1"][0, 0] = True
        for value in arrays.values():
            if value.dtype.kind == "f":
                value[~arrays["valid"]] = 0
        chunks = []
        for start, stop in ((0, 2), (2, 3)):
            values = {key: value[start:stop].copy() for key, value in arrays.items()}
            if start == 0:
                values["terrain_env_origins_w"] = np.asarray([cell.expected_origin_m for cell in self.cells], dtype=np.float32)
                values["terrain_types"] = np.asarray([cell.terrain_type_index for cell in self.cells], dtype=np.int64)
            metadata = {"run_index": 1, "step_start": start + 1, "steps": stop - start, "static_present": start == 0,
                        "termination_term_order": ["time_out", "base_contact"],
                        "transition_header": {"schema_version": "true_rl_agent.task.stair_climbing.transition.normalized.v1",
                            "source_schema_version": "true_rl_agent.task.stair_climbing.transition.vnext.v1",
                            "runtime_projection_hash": p.projection_hash,
                            "reward_input_schema_hash_vnext": p.reward_input_schema_hash_vnext,
                            "support_count": p.support_count, "forbidden_contact_count": p.forbidden_contact_count,
                            "contact_force_history": copy.deepcopy(history)}}
            chunks.append((metadata, values))
        if mutate:
            mutate(capture.writer.identity, context, chunks)
        for metadata, values in chunks:
            capture.writer.write_chunk(values, metadata)
        return capture.writer.complete(context)

    def _inspect(self, reference):
        return inspection.inspect_stair_trajectory_input(Path(reference["manifest_path"]), reference["manifest_sha256"])

    def test_public_archive_counts_terminal_rows_and_excludes_padding(self):
        report = self._inspect(self._archive())
        self.assertEqual(report["coverage"]["valid_rows"], len(self.cells) * 3 - 3)
        self.assertEqual(report["coverage"]["padding_rows"], 3)
        self.assertEqual(report["episodes"][0]["last_sample_time_s"], 0.02)
        self.assertEqual(report["episodes"][1]["last_sample_time_s"], 0.04)
        self.assertEqual(report["runtime_projection"]["resolved_roles"]["support_effector"],
                         ["FL_foot", "FR_foot", "RL_foot", "RR_foot"])
        self.assertEqual(report["raw_anchors"][0]["values"]["support_contact_forces_w"][2], [0, 0, 10])
        self.assertIn("no_per_slot_validity_flags", report["clock"]["contact_force_history"]["reset_history"])
        self.assertIn("reward terms", report["capabilities"]["missing"])
        self.assertNotIn("behavior", report)

    def test_permissions_follow_identity_and_seed_not_incidental_contract_text(self):
        # The genuine suite contains sealed declarations; they do not grant access.
        self.assertEqual(self._inspect(self._archive())["status"], "pass")
        for location in ("identity", "seed"):
            with self.subTest(location=location):
                def mutate(identity, context, chunks):
                    if location == "identity":
                        identity["splits"] = ["public_dev", "sealed_validation"]
                    else:
                        context["seeds"][0]["split"] = "sealed_validation"
                with self.assertRaisesRegex(NumericEvidenceError, "public_dev"):
                    self._inspect(self._archive(mutate))

    def test_pinned_manifest_hash_rejects_modified_manifest(self):
        reference = self._archive()
        path = Path(reference["manifest_path"])
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(NumericEvidenceError, "manifest hash"):
            self._inspect(reference)

    def test_late_chunk_hash_is_actually_read(self):
        reference = self._archive()
        path = Path(reference["manifest_path"]).parent / "chunk_000001.npz"
        path.write_bytes(path.read_bytes() + b"corruption")
        with self.assertRaisesRegex(NumericEvidenceError, "chunk hash or size"):
            self._inspect(reference)

    def test_role_order_and_task_identity_cannot_drift(self):
        for field in ("roles", "task"):
            with self.subTest(field=field):
                def mutate(identity, context, chunks):
                    if field == "roles":
                        context["runtime_projection"]["resolved_roles"]["support_effector"].reverse()
                    else:
                        identity["task_id"] = "push_recovery"
                with self.assertRaisesRegex(NumericEvidenceError, "mismatch"):
                    self._inspect(self._archive(mutate))

    def test_gap_overlap_and_metadata_shape_are_rejected(self):
        for field, value in (("step_start", 4), ("step_start", 2), ("steps", 2)):
            with self.subTest(field=field, value=value):
                def mutate(identity, context, chunks):
                    chunks[1][0][field] = value
                with self.assertRaisesRegex(NumericEvidenceError, "gap or overlap|shape/dtype"):
                    self._inspect(self._archive(mutate))

    def test_reset_episode_must_not_reenter_valid_prefix(self):
        def mutate(identity, context, chunks):
            chunks[1][1]["valid"][0, 0] = True
        with self.assertRaisesRegex(NumericEvidenceError, "valid first-episode prefix"):
            self._inspect(self._archive(mutate))

    def test_termination_names_must_match_frozen_manager_mapping(self):
        for mutation in ("renamed", "missing"):
            with self.subTest(mutation=mutation):
                def mutate(identity, context, chunks):
                    for metadata, arrays in chunks:
                        if mutation == "renamed":
                            metadata["termination_term_order"][1] = "unbound_contact"
                        else:
                            metadata["termination_term_order"].pop()
                            del arrays["termination_1"]
                with self.assertRaisesRegex(NumericEvidenceError, "frozen task mapping"):
                    self._inspect(self._archive(mutate))

    def test_termination_mask_union_must_match_done_on_valid_rows(self):
        for mutation in ("nonterminal_mask", "done_without_term", "term_removed_at_done"):
            with self.subTest(mutation=mutation):
                def mutate(identity, context, chunks):
                    if mutation == "nonterminal_mask":
                        chunks[0][1]["termination_1"][0, 2] = True
                    elif mutation == "done_without_term":
                        chunks[1][1]["done_mask"][0, 2] = True
                        chunks[1][1]["wrapper_done_mask"][0, 2] = True
                    else:
                        chunks[0][1]["termination_1"][0, 0] = False
                with self.assertRaisesRegex(NumericEvidenceError, "done differs from OR"):
                    self._inspect(self._archive(mutate))
        # Invalid padding is not a terminal observation or a second episode.
        def padding(identity, context, chunks):
            chunks[1][1]["termination_1"][0, 0] = True
        self.assertEqual(self._inspect(self._archive(padding))["status"], "pass")

    def test_recorded_episode_clock_and_outcome_length_must_agree(self):
        for location in ("array", "outcome", "seed", "history"):
            with self.subTest(location=location):
                def mutate(identity, context, chunks):
                    if location == "array":
                        chunks[1][1]["episode_length"][0, 2] = 1
                    elif location == "outcome":
                        context["seeds"][0]["outcomes"][0]["episode_steps"] = 2
                    elif location == "seed":
                        context["seeds"][0]["step_dt"] = 0.03
                    else:
                        context["contact_force_history"]["physics_steps_per_control"] = 3
                with self.assertRaisesRegex(NumericEvidenceError, "episode|time mismatch|clock mismatch"):
                    self._inspect(self._archive(mutate))

    def test_nan_in_valid_rows_rejected_but_padding_is_not_an_observation(self):
        def bad(identity, context, chunks):
            chunks[1][1]["support_pos_w"][0, 2, 0, 0] = np.nan
        with self.assertRaisesRegex(NumericEvidenceError, "non-finite observed"):
            self._inspect(self._archive(bad))
        def padding(identity, context, chunks):
            chunks[1][1]["support_pos_w"][0, 0, :, :] = np.nan
        self.assertEqual(self._inspect(self._archive(padding))["coverage"]["padding_rows"], 3)

    def test_coordinated_history_and_seed_clock_change_cannot_override_frozen_binding(self):
        def mutate(identity, context, chunks):
            context["seeds"][0]["step_dt"] = 0.04
            context["contact_force_history"]["control_dt_s"] = 0.04
            context["contact_force_history"]["physics_steps_per_control"] = 8
            for metadata, arrays in chunks:
                metadata["transition_header"]["contact_force_history"]["control_dt_s"] = 0.04
                metadata["transition_header"]["contact_force_history"]["physics_steps_per_control"] = 8
        with self.assertRaisesRegex(NumericEvidenceError, "frozen robot binding or evaluation suite"):
            self._inspect(self._archive(mutate))

    def test_quaternion_and_missing_channel_are_rejected(self):
        for norm in (0, 1.0002):
            with self.subTest(norm=norm):
                def quaternion(identity, context, chunks):
                    chunks[0][1]["base_quat_w"][0, 0, 0] = norm
                with self.assertRaisesRegex(NumericEvidenceError, "quaternion"):
                    self._inspect(self._archive(quaternion))
        def missing(identity, context, chunks):
            del chunks[1][1]["joint_pos"]
        with self.assertRaisesRegex(NumericEvidenceError, "channel fields"):
            self._inspect(self._archive(missing))

    def test_outcome_cell_and_static_environment_mapping_are_checked(self):
        for location in ("outcome", "static"):
            with self.subTest(location=location):
                def mutate(identity, context, chunks):
                    if location == "outcome":
                        context["seeds"][0]["outcomes"][0]["evaluation_cell_id"] = "sealed_validation.fake"
                    else:
                        chunks[0][1]["terrain_types"][0] = 999
                with self.assertRaisesRegex(NumericEvidenceError, "assignment|environment order"):
                    self._inspect(self._archive(mutate))

    def test_cli_writes_only_after_success_and_does_not_overwrite(self):
        reference = self._archive()
        output = self.root / "audit.json"
        args = ["--manifest", reference["manifest_path"], "--manifest-sha256", reference["manifest_sha256"],
                "--output", str(output)]
        original_hash = hashlib.sha256(Path(reference["manifest_path"]).read_bytes()).hexdigest()
        with self.assertRaisesRegex(NumericEvidenceError, "manifest hash"):
            inspection.main([*args[:3], "0" * 64, *args[4:]])
        self.assertFalse(output.exists())
        self.assertEqual(inspection.main(args), 0)
        self.assertEqual(json.loads(output.read_text())["source_reference"], reference)
        with self.assertRaises(FileExistsError):
            inspection.main(args)
        self.assertEqual(hashlib.sha256(Path(reference["manifest_path"]).read_bytes()).hexdigest(), original_hash)


if __name__ == "__main__":
    unittest.main()
