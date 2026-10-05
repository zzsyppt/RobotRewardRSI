"""Synthetic Stair archive factory shared by current adapter/production tests."""
import copy
from pathlib import Path
import tempfile
import unittest
import numpy as np
from true_rl_agent.tasks.stair_climbing.config import StairClimbingTaskConfig
from true_rl_agent.tasks.stair_climbing.numeric_capture import StairNumericCapture
from true_rl_agent.tasks.stair_climbing.plugin import StairClimbingTaskPlugin
from true_rl_agent.tasks.stair_climbing.robot_binding import resolve_stair_cell_manifest
from true_rl_agent.tasks.stair_climbing.runtime_projection import stair_runtime_projection_from_manifest

ROOT = Path(__file__).resolve().parents[2]


class StairArchiveFixture(unittest.TestCase):
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
