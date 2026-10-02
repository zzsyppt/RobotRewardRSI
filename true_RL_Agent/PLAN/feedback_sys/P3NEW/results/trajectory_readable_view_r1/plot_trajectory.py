"""One old development episode, whole-time readable curves; no semantic classifier."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np

EXPECTED_SHA = "1eb9a45bd69bcea985074875b99fcf3d34d2673a6b569c600678958af79965f1"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decode(spec, count):
    shape, dtype = tuple(spec["component_shape"]), np.dtype(spec["dtype"])
    components = int(np.prod(shape)) if shape else 1
    if "by_c_order_component" in spec:
        columns = [json.loads("[" + text + "]") for text in spec["by_c_order_component"]]
        values = np.asarray(columns, dtype=dtype).T.copy()
        assert values.shape == (count, components)
    else:
        intervals = spec.get("inclusive_constant_intervals_by_component", [spec.get("inclusive_runs")])
        values, covered = np.full((count, components), np.nan), np.zeros((count, components), dtype=bool)
        assert len(intervals) == components
        for col, runs in enumerate(intervals):
            for first, last, value in runs:
                assert 1 <= first <= last <= count and not covered[first-1:last, col].any()
                values[first-1:last, col] = np.nan if value is None else value
                covered[first-1:last, col] = True
        assert covered.all()
        values = values.astype(dtype)
    return values.reshape((count, *shape)).copy(order="C")


def main(source, output):
    assert sha(source) == EXPECTED_SHA, "not the specified frozen episode-02 related-raw input"
    data = json.loads(source.read_bytes())
    count, dt = data["episode"]["steps"], data["episode"]["dt_s"]
    assert count == 525 and data["episode"]["episode_id"] == "stair-0061"
    assert data["clock"] == {"source_field": "episode_length", "first_step": 1, "last_step": count, "stride": 1}
    output.mkdir(parents=True, exist_ok=True)
    steps = np.arange(1, count + 1)
    arrays, decoded = {}, []
    for group in ("raw_columns", "raw_discrete", "direct_columns"):
        for field, spec in data[group].items():
            array = decode(spec, count)
            registry = data["source_direct_fields" if group == "direct_columns" else "source_fields"][field]
            assert array.shape == tuple(registry["shape"]) and array.dtype.str == registry["dtype"]
            assert hashlib.sha256(array.tobytes(order="C")).hexdigest() == registry["c_order_bytes_sha256"]
            arrays[field] = array
            decoded.append({"input_pointer": f"/{group}/{field}", "components": int(np.prod(array.shape[1:])) or 1,
                "steps": count, "c_order_bytes_sha256": registry["c_order_bytes_sha256"],
                "nonfinite_values": int((~np.isfinite(array)).sum())})
    origin = np.asarray(data["static_values"]["terrain_env_origins_w"], dtype=float)
    foot_names = data["definitions"]["bodies"]["support_effector"]
    assert foot_names == ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    base = arrays["base_pos_w"].astype(float) - origin
    feet = arrays["support_pos_w"].astype(float) - origin
    velocity = arrays["support_lin_vel_w"].astype(float)
    colors = ["#0072B2", "#D55E00", "#009E73", "#7C3C9B"]
    plots = []
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "path.simplify": False, "agg.path.chunksize": 0, "savefig.facecolor": "white"})

    def curve(label, values, field, component, *, operation="identity", color=None, style="-"):
        values = np.asarray(values, dtype=float)
        assert values.shape == steps.shape
        return {"label": label, "values": values, "source_field": field, "c_order_component": component,
            "operation": operation, "color": color, "style": style}

    def figure(filename, title, panels, note):
        fig, axes = plt.subplots(len(panels), 1, figsize=(20, 2.2*len(panels)+1.2), sharex=True,
            constrained_layout=True, squeeze=False)
        inventory = []
        for index, (ax, panel) in enumerate(zip(axes[:, 0], panels)):
            panel_title, unit, curves = panel
            records = []
            for i, line in enumerate(curves):
                values = line["values"]
                ax.plot(steps, np.ma.masked_invalid(values), color=line["color"] or colors[i % 4],
                    linestyle=line["style"], linewidth=1.05, label=line["label"])
                records.append({k: v for k, v in line.items() if k not in {"values", "color", "style"}} |
                    {"input_steps": count, "finite_steps": int(np.isfinite(values).sum()),
                     "undefined_steps": steps[~np.isfinite(values)].tolist()})
            ax.set_title(panel_title, loc="left", pad=5)
            ax.set_ylabel(unit)
            if unit == "0 / 1":
                ax.set_ylim(-.1, 1.1)
                ax.set_yticks([0, 1])
            ax.set_xlim(1, count)
            ax.xaxis.set_major_locator(MultipleLocator(50))
            ax.xaxis.set_minor_locator(MultipleLocator(10))
            ax.grid(axis="both", which="major", alpha=.28)
            ax.grid(axis="x", which="minor", alpha=.10)
            ax.legend(loc="upper left", ncols=min(len(curves), 4), fontsize=9, framealpha=.85)
            ax.tick_params(labelbottom=True)
            inventory.append({"panel_index": index, "title": panel_title, "unit": unit, "curves": records})
        axes[-1, 0].set_xlabel("Control step (all 1–525 retained; step interval 0.02 s)")
        fig.suptitle(title + "\n" + note, fontsize=12)
        path = output / filename
        fig.savefig(path, dpi=120)
        plt.close(fig)
        width, height = struct.unpack(">II", path.read_bytes()[16:24])
        plots.append({"file": filename, "sha256": sha(path), "size_bytes": path.stat().st_size,
            "image_size_px": [width, height], "panels": inventory})

    panels = []
    for axis, label in enumerate("xyz"):
        panels.append(("Root position — " + label + " in translated world axes", "m", [
            curve(label, base[:, axis], "base_pos_w", axis, operation=f"value - terrain_env_origins_w[{axis}]")]))
    for axis, label in enumerate("xyz"):
        difference = np.r_[np.nan, np.diff(arrays["base_pos_w"][:, axis].astype(float))/dt]
        panels.append(("Root sampled finite difference — " + label + " (not a recorded velocity)", "m/s", [
            curve("delta " + label + " / 0.02 s", difference, "base_pos_w", axis,
                  operation="(value[t]-value[t-1])/dt_s; first sample undefined")]))
    figure("01_root_motion.png", "stair-0061 / r1.e60 — whole root translation", panels,
        "Static terrain origin subtracted; no smoothing, window selection, event labels or causal claims.")

    panels = [("Root signed " + label, "rad", [curve(label, arrays["root_roll_pitch_rad"][:, i],
        "root_roll_pitch_rad", i)]) for i, label in enumerate(("roll", "pitch"))]
    panels += [("Root up-axis projection onto world z", "unitless", [
        curve("up z", arrays["root_up_axis_world_z"], "root_up_axis_world_z", 0)]),
        ("Original root quaternion (world orientation; wxyz)", "unitless", [
            curve(label, arrays["base_quat_w"][:, i], "base_quat_w", i) for i, label in enumerate("wxyz")])]
    figure("02_root_attitude.png", "stair-0061 / r1.e60 — whole root orientation", panels,
        "Signed scalar curves and original quaternion; no posture or failure-category labels.")

    figure("03_foot_reference_positions.png", "stair-0061 / r1.e60 — body reference-point positions", [
        (name, "m", [curve(label, feet[:, foot, axis], "support_pos_w", foot*3+axis,
            operation=f"value - terrain_env_origins_w[{axis}]") for axis, label in enumerate("xyz")])
        for foot, name in enumerate(foot_names)],
        "World axes translated by terrain origin. Reference points are not toe skin or verified contact points.")

    panels = []
    for foot, name in enumerate(foot_names):
        lines = [curve("v"+label, velocity[:, foot, axis], "support_lin_vel_w", foot*3+axis)
                 for axis, label in enumerate("xyz")]
        lines.append(curve("sqrt(vx²+vy²)", np.linalg.norm(velocity[:, foot, :2], axis=1),
            "support_lin_vel_w", [foot*3, foot*3+1], operation="sqrt(vx**2+vy**2)", color="#333333", style="--"))
        panels.append((name + " — signed world components and planar magnitude", "m/s", lines))
    figure("04_foot_velocities.png", "stair-0061 / r1.e60 — whole foot reference-point velocities", panels,
        "Recorded components plus explicit planar norm; no contact, slip or intended-action classification.")

    panels = []
    for foot, name in enumerate(foot_names):
        surface = arrays["support_surface_height_terrain_m"][:, foot]
        panels.append((name + " — reference z and projected surface height", "m", [
            curve("reference z", feet[:, foot, 2], "support_pos_w", foot*3+2,
                operation="value - terrain_env_origins_w[2]"),
            curve("surface z", surface, "support_surface_height_terrain_m", foot)]))
        panels.append((name + " — reference-point z minus projected surface", "m", [
            curve("reference clearance", feet[:, foot, 2]-surface,
                ["support_pos_w", "support_surface_height_terrain_m"], [foot*3+2, foot],
                operation="support_pos_w[:,foot,2] - terrain_env_origins_w[2] - support_surface_height_terrain_m[:,foot]")]))
    figure("05_foot_surface_relation.png", "stair-0061 / r1.e60 — geometric surface relation", panels,
        "Supplied analytic surface height at reference-point XY; not a contact-pair proof or anatomical clearance.")

    panels = [(name + " — current norm and sensor-buffer maximum", "N", [
        curve("current norm", arrays["support_force_norm_n"][:, foot], "support_force_norm_n", foot),
        curve("buffer maximum", arrays["support_buffer_max_force_norm_n"][:, foot], "support_buffer_max_force_norm_n", foot,
              style="--")]) for foot, name in enumerate(foot_names)]
    panels.append(("base — configured forbidden-contact role; supplied force norms", "N", [
        curve("current norm", arrays["forbidden_force_norm_n"][:, 0], "forbidden_force_norm_n", 0),
        curve("buffer maximum", arrays["forbidden_buffer_max_force_norm_n"][:, 0], "forbidden_buffer_max_force_norm_n", 0,
              style="--")]))
    figure("06_force_norms.png", "stair-0061 / r1.e60 — whole force-norm measurements", panels,
        "Sensor buffer lacks per-slot timestamps/validity; buffer maximum is not a full physics-time peak.")

    figure("07_recorded_flags.png", "stair-0061 / r1.e60 — all supplied discrete raw columns", [
        (field, "0 / 1", [curve(field, arrays[field], field, 0)]) for field in data["raw_discrete"]],
        "Original capture flags only; all-zero terminal flags are not reinterpreted as an outcome decision.")

    manifest = {"status": "DEVELOPMENT_READABILITY_VIEW_NOT_N5", "episode": "stair-0061 / r1.e60",
        "source_file": source.name, "source_sha256": EXPECTED_SHA, "script_sha256": sha(__file__),
        "clock": data["clock"], "dt_s": dt, "matplotlib_version": matplotlib.__version__, "backend": matplotlib.get_backend(),
        "decoded_source_columns": decoded, "all_supplied_fields_drawn": sorted(arrays), "figures": plots,
        "sampling": {"input_steps_per_column": count, "time_downsampling": False, "path_simplification": False,
                     "missing_or_undefined_values": "masked; no connecting across missing samples", "selected_positive_windows": False},
        "limits": ["PNG is a readable, lossy visual aid, not a lossless numeric replacement or independent truth",
            "Use original related-raw JSON/NPZ for exact values and independent arithmetic",
            "No production classifiers, generated semantic labels, thresholds, report text or reviewer answers were used",
            "Line segments connect sampled points only; no claim about unrecorded continuous trajectory",
            "Root velocity is explicitly finite difference, because no recorded root velocity column is supplied",
            "No evidence here that plots improve model accuracy or satisfy N5 capacity/semantic gates"],
        "model_calls": 0, "gpu_calls": 0}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "figures": len(plots), "panels": sum(len(p["panels"]) for p in plots),
        "decoded_fields": len(arrays), "source_sha256": EXPECTED_SHA}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.source, args.output)
