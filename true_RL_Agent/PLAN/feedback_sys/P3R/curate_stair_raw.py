"""Independent NumPy extraction of declared development examples, no runtime imports."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

SLOTS = [48, 64, 69, 156, 158]
KEYS = ["base_pos_w", "base_quat_w", "support_pos_w", "support_lin_vel_w", "support_contact_forces_w", "episode_length"]


def longest(mask):
    starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
    ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
    return max(zip(starts, ends), key=lambda p: p[1] - p[0], default=None)


def main():
    path = Path(sys.argv[1]); raw = path.read_bytes(); manifest = json.loads(raw)
    assert hashlib.sha256(raw).hexdigest() == "22fb749620c4e614cbb4df4fbfdb82b70d76c8f1cd81a3eda50de465d9fb3e4a"
    buffers = {slot: {k: [] for k in KEYS} for slot in SLOTS}
    origins = None
    for chunk in manifest["chunks"]:
        if chunk["metadata"]["run_index"] != 1:
            continue
        file = path.parent / chunk["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == chunk["sha256"]
        with np.load(file, allow_pickle=False) as arrays:
            if "terrain_env_origins_w" in arrays: origins = arrays["terrain_env_origins_w"].copy()
            for slot in SLOTS:
                valid = arrays["valid"][:, slot]
                for key in KEYS: buffers[slot][key].append(arrays[key][:, slot][valid])
    threshold = manifest["metadata"]["semantics"]["episode_outcomes"]["signals"]["slip"]["contact_force_threshold"]
    result = {"manifest_sha256": hashlib.sha256(raw).hexdigest(), "identity": manifest["identity"],
              "definitions": {"coordinates": "world axes; base and foot positions below subtract environment origin, not terrain surface height",
                              "basis": "wxyz quaternion: forward_axis_world_z=2(xz-wy); up_axis_world_z=1-2(x*x+y*y); these remain valid at large roll",
                              "unloaded": f"instantaneous net contact force norm <= {threshold} N; no claim of physical gap",
                              "foot_selection": "longest unloaded interval per support, independent of velocity; report actual motion instead of classifying a freeze"},
              "episodes": []}
    for slot, buffer in buffers.items():
        a = {k: np.concatenate(v) for k, v in buffer.items()}
        pos = a["base_pos_w"].astype(float) - origins[slot]
        foot = a["support_pos_w"].astype(float) - origins[slot]
        quat = a["base_quat_w"].astype(float); w,x,y,z = quat.T
        roll = np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)); pitch = np.arcsin(np.clip(2*(w*y-z*x),-1,1))
        fz, uz = 2*(x*z-w*y), 1-2*(x*x+y*y)
        steps = a["episode_length"].astype(int)
        assert np.array_equal(steps,np.arange(1,len(steps)+1))
        selected = {1, len(steps), *range(50,len(steps),50)}
        selected |= ({58,346,360,369,385,394,414,426} if slot==64 else {63,170,185,205,225,232} if slot==156 else {175,190,200,300,400,500,588} if slot==48 else set())
        samples = [{"step": int(steps[i]), "base_position_env_relative_m": pos[i].round(6).tolist(),
                    "roll_rad": round(float(roll[i]),6), "pitch_rad": round(float(pitch[i]),6),
                    "body_forward_axis_world_z": round(float(fz[i]),6), "body_up_axis_world_z": round(float(uz[i]),6)}
                   for i in range(len(steps)) if int(steps[i]) in selected]
        unloaded = []
        forces = np.linalg.norm(a["support_contact_forces_w"].astype(float),axis=-1)
        for b,name in enumerate(["FL_foot","FR_foot","RL_foot","RR_foot"]):
            span = longest(forces[:,b]<=threshold)
            if span is None: continue
            start,end = span; section=slice(start,end+1)
            unloaded.append({"body":name,"start_step":int(steps[start]),"end_step":int(steps[end]),
                             "duration_s":round((end-start+1)*.02,4),
                             "point_position_axis_range_m":np.ptp(foot[section,b],axis=0).round(6).tolist(),
                             "point_max_3d_speed_mps":round(float(np.linalg.norm(a["support_lin_vel_w"][section,b],axis=-1).max()),6),
                             "root_forward_gain_m":round(float(pos[end,0]-pos[start,0]),6)})
        result["episodes"].append({"episode_id":f"r1.e{slot}","steps":len(steps),"samples":samples,
                                   "longest_unloaded_intervals":unloaded,
                                   "base_height_above_env_origin_m_range": [round(float(pos[:,2].min()),6),round(float(pos[:,2].max()),6)]})
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":main()
