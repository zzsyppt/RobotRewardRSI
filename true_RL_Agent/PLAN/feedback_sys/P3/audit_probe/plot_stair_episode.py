"""Derived measurement plot for one Stair episode from raw archive arrays (not video)."""
from __future__ import annotations
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, DEV_RESULT, RUN, ENV, OUT_PNG = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
sys.path.insert(0, SRC)
from true_rl_agent.runner.numeric_evidence import load_numeric_evidence_manifest, iter_numeric_evidence
from true_rl_agent.tasks.stair_climbing.unified_evaluation import StairUnifiedEvaluationSuite
from true_rl_agent.tasks.stair_climbing.unified_training import StairUnifiedTrainingProfile
from true_rl_agent.tasks.stair_climbing.terrain_evidence import stair_surface_query_for_ordered_cell

ref = json.load(open(DEV_RESULT))["task_evidence"]["numeric_evidence"]
manifest = load_numeric_evidence_manifest(ref); md = manifest["metadata"]
training = StairUnifiedTrainingProfile.from_mapping(md["training_profile"])
suite = StairUnifiedEvaluationSuite.from_mapping(md["evaluation_suite"], training_profile=training)
cell = suite.private_cell_assignment("public_dev")[ENV]
query = stair_surface_query_for_ordered_cell(cell.spec)
names = md["runtime_projection"]["resolved_roles"]["support_effector"]
chunks = sorted([(h["step_start"], a) for h, a in iter_numeric_evidence(ref) if h["run_index"] == RUN], key=lambda c: c[0])
cat = lambda k: np.concatenate([a[k] for _, a in chunks], axis=0)
valid = cat("valid")[:, ENV]; v = np.flatnonzero(valid)
origin = chunks[0][1]["terrain_env_origins_w"][ENV].astype(float)
base = cat("base_pos_w")[v, ENV].astype(float); spos = cat("support_pos_w")[v, ENV].astype(float)
svel = cat("support_lin_vel_w")[v, ENV].astype(float); sforce = cat("support_contact_forces_w")[v, ENV].astype(float)
t = np.arange(1, len(v) + 1) * cell.spec.control_dt_s
progress = base[:, 0] - origin[0]
spec = cell.spec
print("cell", cell.cell_id, "stratum", spec.stratum_id, "start_platform", spec.start_platform_length_m, "risers", spec.riser_heights_m, "treads", spec.tread_depths_m, "steps", len(v))
fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
axes[0].plot(t, progress, "k", label="root progress x (m)")
axes[0].axhline(spec.start_platform_length_m / 2, color="gray", ls=":", label="first riser plane")
cum = spec.start_platform_length_m / 2
for depth in spec.tread_depths_m:
    cum += depth; axes[0].axhline(cum, color="gray", ls=":", lw=0.5)
axes[0].plot(t, base[:, 2] - origin[2], "b", label="root height above origin (m)")
axes[0].legend(); axes[0].set_title(f"r{RUN}.e{ENV} {cell.cell_id}: root")
for j, name in enumerate(names):
    clear = np.array([spos[i, j, 2] - origin[2] - query.sample(progress_m=spos[i, j, 0] - origin[0], lateral_m=spos[i, j, 1] - origin[1]).height_m for i in range(len(v))])
    fn = np.linalg.norm(sforce[:, j], axis=-1)
    axes[1].plot(t, clear, label=f"{name} clearance above local surface (m)")
    axes[2].plot(t, fn, label=f"{name} |F| (N)")
    axes[3].plot(t, spos[:, j, 0] - origin[0], label=f"{name} x (m)")
axes[1].axhline(0.018, color="r", ls=":", label="0.018 m threshold"); axes[1].set_ylim(-0.05, 0.3); axes[1].legend(fontsize=7)
axes[2].set_yscale("symlog"); axes[2].axhline(1.0, color="r", ls=":"); axes[2].legend(fontsize=7)
axes[3].axhline(spec.start_platform_length_m / 2, color="gray", ls=":"); axes[3].legend(fontsize=7); axes[3].set_xlabel("episode time (s)")
plt.tight_layout(); plt.savefig(OUT_PNG, dpi=90)
# numeric summary of the last 8 s for the "hanging foot" question
last = slice(len(v) - 400, len(v))
for j, name in enumerate(names):
    fn = np.linalg.norm(sforce[last, j], axis=-1)
    clear = np.array([spos[i, j, 2] - origin[2] - query.sample(progress_m=spos[i, j, 0] - origin[0], lateral_m=spos[i, j, 1] - origin[1]).height_m for i in range(len(v) - 400, len(v))])
    print(name, "last8s |F| median", round(float(np.median(fn)), 3), "max", round(float(fn.max()), 3), "clearance median", round(float(np.median(clear)), 4), "min", round(float(clear.min()), 4),
          "x median", round(float(np.median(spos[last, j, 0] - origin[0])), 3), "vz max abs", round(float(np.abs(svel[last, j, 2]).max()), 4))
print("root last8s progress median", round(float(np.median(progress[last])), 3), "height median", round(float(np.median(base[last, 2] - origin[2])), 3))
