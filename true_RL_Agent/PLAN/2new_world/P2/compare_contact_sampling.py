"""Compare the retained fixed-state traces, including contact reward inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent / "results"
FILES = ("physics_facts", "trace", "reward_inputs", "reward_terms")


def compare(old, new):
    result = {"old": old, "new": new, "source_files": {}}
    for name in FILES:
        paths = [ROOT / folder / (name + ".npz") for folder in (old, new)]
        arrays = [np.load(path) for path in paths]
        assert arrays[0].files == arrays[1].files
        result[name] = {}
        for key in arrays[0].files:
            a, b = (array[key] for array in arrays)
            assert a.shape == b.shape
            result[name][key] = float(np.max(np.abs(a.astype(float) - b.astype(float))))
        for path in paths:
            result["source_files"][path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    a, b = (np.load(ROOT / folder / "trace.npz")["support_force"] for folder in (old, new))
    result["reset_contact"] = {
        "new_reset_force_max_abs_N": float(np.max(np.abs(b[0]))),
        "post_reset_force_max_abs_diff_N": float(np.max(np.abs(a[1:] - b[1:]))),
        "interpretation": "Timestamp-zero stale contacts are suppressed by the new upstream kernel; reset is reported separately.",
    }
    a, b = (np.load(ROOT / folder / "reward_inputs.npz") for folder in (old, new))
    # Probe arrays prepend the control-step axis to (env, history, body, xyz).
    masks = [np.linalg.norm(data["support_contact_force_history"], axis=-1).max(axis=-2) > 1.0 for data in (a, b)]
    result["history_contact_mask"] = {"different": int(np.count_nonzero(masks[0] != masks[1])), "count": int(masks[0].size)}
    return result


go2_before = compare("go2_physics_legacy05", "go2_physics_current05")
go2_after = compare("go2_physics_legacy05", "audit2_go2_contact")
g1_after = compare("g1_physics_legacy05", "audit_g1_contact_01")
contact_fields = (
    "support_air_time", "support_contact_time", "support_last_air_time",
    "support_first_contact", "support_contact_force_history",
    "support_contact_force", "support_contact",
)
assert all(go2_after["reward_inputs"][key] == 0 for key in contact_fields)
assert go2_after["reward_terms"]["feet_air_time"] == 0
assert go2_after["reset_contact"]["post_reset_force_max_abs_diff_N"] == 0
report = {
    "scope": "Nominal mass, fixed initial state and 32 control steps; noise and startup mass sampling disabled only for these diagnostics",
    "go2_contact_sampling": "PASS",
    "g1_numerical_equivalence": False,
    "go2_before": go2_before,
    "go2_after": go2_after,
    "g1_after": g1_after,
    "g1_interpretation": "Instantaneous physical forces and kinematics diverge; contact timing and history contact masks agree. Feet-slide value still differs with support velocity. No full physical equivalence claim.",
}
(ROOT / "audit_contact_sampling_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"go2_contact_sampling": report["go2_contact_sampling"], "g1_history_contact_mask": g1_after["history_contact_mask"], "g1_feet_air_time_diff": g1_after["reward_terms"]["feet_air_time"]}))
