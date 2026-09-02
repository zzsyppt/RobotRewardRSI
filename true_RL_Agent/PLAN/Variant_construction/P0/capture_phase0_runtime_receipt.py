from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


RECEIPT_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase0_runtime_receipt.v3"
PROFILE_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase0_robot_runtime_facts.v3"
BINDING_SCHEMA_VERSION = "true_rl_agent.variant_construction.phase0_cell_binding_facts.v2"
FLOAT_DECIMAL_PLACES = 6


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _round_float(value: Any) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("runtime receipt cannot contain a non-finite number")
    rounded = round(number, FLOAT_DECIMAL_PLACES)
    return 0.0 if rounded == 0.0 else rounded


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _round_float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return _json_value(tensor.tolist())
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    selected: dict[str, Any] = {}
    for name in ("name", "body_names", "joint_names", "preserve_order"):
        if hasattr(value, name):
            selected[name] = _json_value(getattr(value, name))
    if selected:
        return selected
    return str(value)


def _callable_id(value: Any) -> str:
    module = getattr(value, "__module__", None)
    qualname = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if module and qualname:
        return f"{module}.{qualname}"
    return str(value)


def _config_terms(group: Any) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for name, value in vars(group).items():
        if name.startswith("_") or not hasattr(value, "func"):
            continue
        noise = getattr(value, "noise", None)
        noise_payload = None
        if noise is not None:
            noise_payload = {
                "class": f"{type(noise).__module__}.{type(noise).__qualname__}",
                "n_min": _json_value(getattr(noise, "n_min", None)),
                "n_max": _json_value(getattr(noise, "n_max", None)),
                "operation": _json_value(getattr(noise, "operation", None)),
            }
        terms.append(
            {
                "name": name,
                "callable": _callable_id(value.func),
                "params": _json_value(getattr(value, "params", {})),
                "scale": _json_value(getattr(value, "scale", None)),
                "clip": _json_value(getattr(value, "clip", None)),
                "noise": noise_payload,
            }
        )
    return terms


def _action_terms(actions: Any) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for name, value in vars(actions).items():
        if name.startswith("_") or value is None or not hasattr(value, "asset_name"):
            continue
        terms.append(
            {
                "name": name,
                "class": f"{type(value).__module__}.{type(value).__qualname__}",
                "asset_name": str(value.asset_name),
                "joint_names": _json_value(getattr(value, "joint_names", None)),
                "scale": _json_value(getattr(value, "scale", None)),
                "use_default_offset": _json_value(
                    getattr(value, "use_default_offset", None)
                ),
            }
        )
    return terms


def _height_scanner(scanner: Any) -> dict[str, Any] | None:
    if scanner is None:
        return None
    pattern = scanner.pattern_cfg
    return {
        "class": f"{type(scanner).__module__}.{type(scanner).__qualname__}",
        "prim_path": str(scanner.prim_path),
        "offset_pos_m": _json_value(scanner.offset.pos),
        "offset_rot_wxyz": _json_value(scanner.offset.rot),
        "ray_alignment": str(scanner.ray_alignment),
        "mesh_prim_paths": _json_value(scanner.mesh_prim_paths),
        "update_period_s": _round_float(scanner.update_period),
        "pattern": {
            "class": f"{type(pattern).__module__}.{type(pattern).__qualname__}",
            "resolution_m": _json_value(getattr(pattern, "resolution", None)),
            "size_m": _json_value(getattr(pattern, "size", None)),
            "direction": _json_value(getattr(pattern, "direction", None)),
            "ordering": _json_value(getattr(pattern, "ordering", None)),
        },
    }


def _contact_sensor(sensor: Any) -> dict[str, Any] | None:
    if sensor is None:
        return None
    return {
        "class": f"{type(sensor).__module__}.{type(sensor).__qualname__}",
        "prim_path": str(sensor.prim_path),
        "history_length": int(sensor.history_length),
        "track_air_time": bool(sensor.track_air_time),
        "update_period_s": _round_float(sensor.update_period),
    }


def _cell_binding(
    *,
    cell_key: str,
    source_config: str,
    cfg: Any,
    corruption_by_mode: Mapping[str, bool],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    policy = cfg.observations.policy
    actions = _action_terms(cfg.actions)
    observation = {
        "concatenate_terms": bool(policy.concatenate_terms),
        "history_length": _json_value(getattr(policy, "history_length", None)),
        "enable_corruption_by_mode": {
            str(key): bool(value) for key, value in corruption_by_mode.items()
        },
        "terms": _config_terms(policy),
    }
    sensors = {
        "height_scanner": _height_scanner(
            getattr(cfg.scene, "height_scanner", None)
        ),
        "contact_forces": _contact_sensor(
            getattr(cfg.scene, "contact_forces", None)
        ),
    }
    payload = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "cell_key": cell_key,
        "source_config": source_config,
        "source_hashes": dict(source_hashes),
        "control": {
            "simulation_dt_s": _round_float(cfg.sim.dt),
            "decimation": int(cfg.decimation),
            "control_dt_s": _round_float(cfg.sim.dt * cfg.decimation),
        },
        "actions": actions,
        "policy_observation": observation,
        "sensors": sensors,
        "policy_io_schema_hash": _canonical_hash(
            {"actions": actions, "policy_observation": observation}
        ),
        "sensor_schema_hash": _canonical_hash(sensors),
    }
    payload["binding_facts_hash"] = _canonical_hash(payload)
    return payload


def _read_uri_bytes(uri: str) -> tuple[bytes, str | None]:
    if uri.startswith(("http://", "https://", "omniverse://")):
        import omni.client

        result, version, content = omni.client.read_file(uri)
        if result != omni.client.Result.OK:
            raise RuntimeError(f"asset read failed: {uri}: {result}")
        return bytes(content), str(version)
    path = Path(uri)
    if uri.startswith("file:"):
        from urllib.parse import urlparse

        path = Path(urlparse(uri).path)
    return path.read_bytes(), None


def _usd_layer_closure(root_uri: str) -> dict[str, Any]:
    from pxr import Usd

    stage = Usd.Stage.Open(root_uri, load=Usd.Stage.LoadAll)
    if stage is None:
        raise RuntimeError(f"unable to open USD stage: {root_uri}")
    entries: list[dict[str, Any]] = []
    unresolved: list[str] = []
    for layer in stage.GetUsedLayers():
        identifier = str(layer.identifier)
        if identifier.startswith("anon:"):
            continue
        try:
            payload, version = _read_uri_bytes(identifier)
        except Exception:
            unresolved.append(identifier)
            continue
        entries.append(
            {
                "identity": identifier,
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "content_version": version,
            }
        )
    entries.sort(key=lambda item: item["identity"])
    closure = {
        "kind": "usd_used_layer_closure_v1",
        "root_identity": root_uri,
        "entries": entries,
        "unresolved": sorted(unresolved),
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
    }
    closure["closure_hash"] = _canonical_hash(closure)
    return closure


def _description_tree_closure(root_file: Path, description_root: Path) -> dict[str, Any]:
    entries = []
    for path in sorted(description_root.rglob("*")):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        entries.append(
            {
                "identity": path.relative_to(description_root).as_posix(),
                "size_bytes": len(payload),
                "sha256": _sha256_bytes(payload),
                "content_version": None,
            }
        )
    closure = {
        "kind": "description_tree_closure_v1",
        "root_identity": str(root_file),
        "entries": entries,
        "unresolved": [],
        "file_count": len(entries),
        "total_bytes": sum(int(item["size_bytes"]) for item in entries),
    }
    closure["closure_hash"] = _canonical_hash(closure)
    return closure


def _tensor_row(value: Any) -> list[float]:
    tensor = value.detach().cpu()
    if tensor.ndim > 1:
        tensor = tensor[0]
    return [_round_float(item) for item in tensor.tolist()]


def _actuator_facts(robot: Any) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    for name, actuator in robot.actuators.items():
        joint_names = list(actuator.joint_names)
        parameters: dict[str, Any] = {}
        for field_name in (
            "effort_limit",
            "velocity_limit",
            "stiffness",
            "damping",
            "armature",
            "friction",
            "dynamic_friction",
        ):
            value = getattr(actuator, field_name, None)
            if value is None:
                parameters[field_name] = None
                continue
            raw = _tensor_row(value) if hasattr(value, "detach") else _json_value(value)
            if isinstance(raw, list) and len(raw) == len(joint_names):
                parameters[field_name] = dict(zip(joint_names, raw, strict=True))
            else:
                parameters[field_name] = raw
        facts.append(
            {
                "name": str(name),
                "class": f"{type(actuator).__module__}.{type(actuator).__qualname__}",
                "joint_names": joint_names,
                "parameters_by_joint": parameters,
            }
        )
    return facts


def _robot_facts(
    *,
    profile_key: str,
    family_id: str,
    runtime_provider: str,
    upstream_revision: str,
    robot: Any,
    asset_closure: Mapping[str, Any],
    support_body_names: tuple[str, ...],
) -> dict[str, Any]:
    joint_names = list(robot.joint_names)
    body_names = list(robot.body_names)
    if any(name not in body_names for name in support_body_names):
        raise ValueError(f"support body missing from runtime: {profile_key}")
    default_joint_pos = _tensor_row(robot.data.default_joint_pos)
    default_root_state = _tensor_row(robot.data.default_root_state)
    body_pos = robot.data.body_pos_w.detach().cpu()[0]
    root_pos = robot.data.root_pos_w.detach().cpu()[0]
    relative_body_positions = {
        name: [_round_float(value) for value in (body_pos[index] - root_pos).tolist()]
        for index, name in enumerate(body_names)
    }
    support_offsets = [
        relative_body_positions[name][2] for name in support_body_names
    ]
    payload = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_key": profile_key,
        "robot_family_id": family_id,
        "runtime_provider": runtime_provider,
        "upstream_revision": upstream_revision,
        "asset_closure": dict(asset_closure),
        "joint_names": joint_names,
        "body_names": body_names,
        "support_body_names": list(support_body_names),
        "default_state": {
            "root_pos_m": default_root_state[0:3],
            "root_quat_wxyz": default_root_state[3:7],
            "root_lin_vel_mps": default_root_state[7:10],
            "root_ang_vel_radps": default_root_state[10:13],
            "joint_pos_rad_by_name": dict(
                zip(joint_names, default_joint_pos, strict=True)
            ),
        },
        "morphology": {
            "default_root_height_m": default_root_state[2],
            "default_body_pos_relative_to_root_m": relative_body_positions,
            "nominal_root_to_support_body_center_z_m": _round_float(
                -sum(support_offsets) / len(support_offsets)
            ),
        },
        "actuators": _actuator_facts(robot),
    }
    payload["runtime_facts_hash"] = _canonical_hash(payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    project_root = args.project_root.resolve(strict=True)
    sys.path.insert(0, str(project_root / "src"))

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True, device=args.device)
    simulation_app = app_launcher.app
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.utils import configclass
        from isaaclab_assets import G1_MINIMAL_CFG
        from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG
        from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
            UnitreeGo2RoughEnvCfg,
        )
        from unitree_rl_lab.assets.robots.unitree import (
            UNITREE_G1_29DOF_CFG,
            UNITREE_ROS_DIR,
        )
        from importlib import import_module
        from true_rl_agent.tasks.push_recovery.isaac_config import load_runtime_config
        from true_rl_agent.tasks.stair_climbing.isaaclab_env_cfg import (
            build_stair_env_cfg,
        )

        RobotEnvCfg = import_module(
            "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
        ).RobotEnvCfg

        @configclass
        class ProbeSceneCfg(InteractiveSceneCfg):
            g1_37 = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/G1_37")
            g1_29 = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/G1_29")
            go2 = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Go2")

        sim_cfg = sim_utils.SimulationCfg(
            dt=0.005,
            device=args.device,
            gravity=(0.0, 0.0, 0.0),
        )
        sim = sim_utils.SimulationContext(sim_cfg)
        scene = InteractiveScene(ProbeSceneCfg(num_envs=1, env_spacing=4.0))
        sim.reset()
        scene.reset()
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

        isaac_lab_root = Path("/data/yuyangliu/vendor/IsaacLab")
        unitree_root = Path("/data/yuyangliu/vendor/unitree_rl_lab")
        isaac_revision = (isaac_lab_root / ".source_commit").read_text().strip()
        unitree_revision = (unitree_root / ".source_commit").read_text().strip()
        g1_29_urdf = Path(UNITREE_G1_29DOF_CFG.spawn.asset_path)
        description_root = Path(UNITREE_ROS_DIR) / "robots/g1_description"

        profiles = [
            _robot_facts(
                profile_key="p0.g1_37_stair_execution",
                family_id="unitree_g1",
                runtime_provider="isaac_lab_2.3.0",
                upstream_revision=isaac_revision,
                robot=scene["g1_37"],
                asset_closure=_usd_layer_closure(G1_MINIMAL_CFG.spawn.usd_path),
                support_body_names=(
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ),
            ),
            _robot_facts(
                profile_key="p0.g1_29_push_execution",
                family_id="unitree_g1",
                runtime_provider="unitree_rl_lab_on_isaac_lab_2.3.0",
                upstream_revision=unitree_revision,
                robot=scene["g1_29"],
                asset_closure=_description_tree_closure(
                    g1_29_urdf,
                    description_root,
                ),
                support_body_names=(
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                ),
            ),
            _robot_facts(
                profile_key="p0.go2_upstream_execution_basis",
                family_id="unitree_go2",
                runtime_provider="isaac_lab_2.3.0",
                upstream_revision=isaac_revision,
                robot=scene["go2"],
                asset_closure=_usd_layer_closure(UNITREE_GO2_CFG.spawn.usd_path),
                support_body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
            ),
        ]

        protocol = json.loads(
            (project_root / "configs/tasks/stair_protocol_v1.json").read_text()
        )
        stair_train = build_stair_env_cfg(
            protocol,
            mode="train",
            seed=1701,
            device=args.device,
        )
        stair_eval = build_stair_env_cfg(
            protocol,
            mode="eval",
            seed=1701,
            device=args.device,
        )
        push_vendor = RobotEnvCfg()
        push_runtime = load_runtime_config(
            project_root / "configs/tasks/push_recovery.yaml"
        )
        go2_basis = UnitreeGo2RoughEnvCfg()
        source_hashes = {
            "stair_env": _sha256_bytes(
                (project_root / "src/true_rl_agent/tasks/stair_climbing/isaaclab_env_cfg.py").read_bytes()
            ),
            "stair_protocol": _sha256_bytes(
                (project_root / "configs/tasks/stair_protocol_v1.json").read_bytes()
            ),
            "push_runtime": _sha256_bytes(
                (project_root / "configs/tasks/push_recovery.yaml").read_bytes()
            ),
            "push_vendor_env": _sha256_bytes(
                (
                    unitree_root
                    / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py"
                ).read_bytes()
            ),
            "go2_rough_env": _sha256_bytes(
                (
                    isaac_lab_root
                    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py"
                ).read_bytes()
            ),
            "isaac_velocity_env": _sha256_bytes(
                (
                    isaac_lab_root
                    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py"
                ).read_bytes()
            ),
        }
        bindings = [
            _cell_binding(
                cell_key="stair.g1_37dof.clean_legacy_v1",
                source_config="true_rl_agent.tasks.stair_climbing.build_stair_env_cfg",
                cfg=stair_train,
                corruption_by_mode={
                    "train": bool(stair_train.observations.policy.enable_corruption),
                    "eval": bool(stair_eval.observations.policy.enable_corruption),
                },
                source_hashes={
                    "stair_env": source_hashes["stair_env"],
                    "stair_protocol": source_hashes["stair_protocol"],
                    "isaac_velocity_env": source_hashes["isaac_velocity_env"],
                },
            ),
            _cell_binding(
                cell_key="push.g1_29dof.legacy_v1",
                source_config="unitree_rl_lab RobotEnvCfg + project-local Push overlay",
                cfg=push_vendor,
                corruption_by_mode={
                    "vendor_default": bool(
                        push_vendor.observations.policy.enable_corruption
                    ),
                    "train": not bool(
                        push_runtime["environment"]["disable_observation_corruption"]
                    ),
                    "eval": not bool(
                        push_runtime["environment"]["disable_observation_corruption"]
                    ),
                },
                source_hashes={
                    "push_runtime": source_hashes["push_runtime"],
                    "push_vendor_env": source_hashes["push_vendor_env"],
                },
            ),
            _cell_binding(
                cell_key="go2.official_rough_upstream_basis",
                source_config="Isaac Lab UnitreeGo2RoughEnvCfg",
                cfg=go2_basis,
                corruption_by_mode={
                    "upstream_train": bool(
                        go2_basis.observations.policy.enable_corruption
                    )
                },
                source_hashes={
                    "go2_rough_env": source_hashes["go2_rough_env"],
                    "isaac_velocity_env": source_hashes["isaac_velocity_env"],
                },
            ),
        ]

        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "capture_kind": "live_headless_single_scene_plus_effective_config",
            "captured_repository_head": subprocess.run(
                ["git", "-C", str(project_root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "probe_script_sha256": _sha256_bytes(Path(__file__).read_bytes()),
            "float_quantization": {
                "decimal_places": FLOAT_DECIMAL_PLACES,
                "resolution": 10 ** (-FLOAT_DECIMAL_PLACES),
                "negative_zero_normalized": True,
            },
            "environment": {
                "python": ".".join(str(value) for value in sys.version_info[:3]),
                "device": str(args.device),
                "isaac_lab_release": (isaac_lab_root / "VERSION").read_text().strip(),
                "isaac_lab_source_commit": isaac_revision,
                "unitree_rl_lab_source_commit": unitree_revision,
            },
            "robot_runtime_facts": profiles,
            "legacy_cell_binding_facts": bindings,
        }
        receipt["receipt_hash"] = _canonical_hash(receipt)
        encoded_receipt = (
            json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        ).encode("utf-8")
        with args.output.open("wb") as output_handle:
            output_handle.write(encoded_receipt)
            output_handle.flush()
            os.fsync(output_handle.fileno())
        print(json.dumps({"status": "pass", "receipt_hash": receipt["receipt_hash"]}))
        sys.stdout.flush()
        # Isaac Sim 5.1 on the frozen headless host can block during Kit shutdown
        # after the receipt is already fsynced because its optional Iray library
        # lacks libGLU. This one-shot process owns no persistent state; exiting here
        # releases both CUDA and Kit resources without installing a host dependency.
        os._exit(0)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
