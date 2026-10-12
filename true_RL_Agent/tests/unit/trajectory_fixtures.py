"""Small raw trajectories shared by behavior/operator tests."""
import numpy as np
from true_rl_agent.evaluation.trajectory_translation.contracts import NumericEpisode
FEET = tuple(f"robot_0.support.{name}" for name in ("FL", "FR", "RL", "RR"))
ROOT = "robot_0.root"
BASE = "robot_0.body.base"
JOINTS = tuple(f"{foot}_{part}_joint" for part in ("hip", "thigh", "calf")
               for foot in ("FL", "FR", "RL", "RR"))


def sample_episode(n=4, *, steps=None, dt=0.02, support=FEET, joints=JOINTS):
    steps = np.arange(1, n + 1) if steps is None else np.asarray(steps)
    n = len(steps)
    count = len(support)
    arrays = {
        "base_pos_w": np.zeros((n, 3)),
        "base_quat_w": np.tile([1.0, 0, 0, 0], (n, 1)),
        "support_pos_w": np.zeros((n, count, 3)),
        "support_lin_vel_w": np.zeros((n, count, 3)),
        "support_contact_forces_w": np.zeros((n, count, 3)),
        "support_contact_forces_w_history": np.zeros((n, 3, count, 3)),
        "forbidden_contact_forces_w": np.zeros((n, 1, 3)),
        "forbidden_contact_forces_w_history": np.zeros((n, 3, 1, 3)),
        "joint_pos": np.zeros((n, len(joints))),
        "done_mask": np.zeros(n, dtype=bool),
        "wrapper_done_mask": np.zeros(n, dtype=bool),
        "termination_0": np.zeros(n, dtype=bool),
        "termination_1": np.zeros(n, dtype=bool),
    }
    arrays["support_contact_forces_w"][:, :, 2] = 2.0
    return NumericEpisode(
        source={"acceptance_split": "public_dev", "manifest_sha256": "a" * 64,
                "trial_id": "synthetic-measurement", "checkpoint_hash": "b" * 64,
                "contact_force_history": {"history_length": 3}},
        run_index=1, env_index=0, steps=steps, dt_s=dt, arrays=arrays,
        entities={"root": (ROOT,), "support": tuple(support), "forbidden": (BASE,), "joints": tuple(joints)},
        source_chunks=({"file": "fixture.npz", "sha256": "c" * 64},),
        chunk_indices=np.zeros(n, dtype=np.int64), chunk_rows=steps - 1,
        task={"contact_threshold_n": 1.0, "origin_world": [0, 0, 0],
              "progress_direction_world": [1, 0, 0], "goal_progress_m": 4.0,
              "deadline_s": 20.0, "termination_term_order": ["time_out", "base_contact"],
              "outcome": {"success": True, "primary_reason": "success", "episode_steps": int(steps[-1])}},
    )
