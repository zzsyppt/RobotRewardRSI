from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from true_rl_agent.backends.isaac_lab_rsl_rl import (  # noqa: E402
    IsaacLabRslRlBackendAdapter,
)
from true_rl_agent.composition.base import (  # noqa: E402
    TaskCompositionContext,
    resolve_task_composition,
)
from true_rl_agent.evaluation.binding import canonical_json_hash  # noqa: E402
from true_rl_agent.tasks.push_recovery.composition import (  # noqa: E402
    PushRecoveryCompositionProvider,
)
from true_rl_agent.tasks.push_recovery.config import (  # noqa: E402
    PushRecoveryTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.composition import (  # noqa: E402
    StairClimbingCompositionProvider,
)
from true_rl_agent.tasks.stair_climbing.config import (  # noqa: E402
    StairClimbingTaskConfig,
)
from true_rl_agent.tasks.stair_climbing.seed_policy import (  # noqa: E402
    STAIR_FORMAL_EVAL_NAMESPACE,
)


SNAPSHOT_PATH = Path(__file__).with_name("PHASE0_BASELINE_SNAPSHOT.json")
RUNTIME_RECEIPT_PATH = Path(__file__).with_name(
    "PHASE0_RUNTIME_PROBE_RECEIPT.json"
)
RUNTIME_PROBE_PATH = Path(__file__).with_name(
    "capture_phase0_runtime_receipt.py"
)

EXPECTED_CELLS = (
    "stair.g1_37dof",
    "push.g1_29dof",
    "stair.go2",
)
EXPECTED_VARIANTS = (
    "stair.g1_37dof.clean_legacy_v1",
    "stair.g1_37dof.irregular_v1",
    "stair.g1_37dof.rough_v1",
    "stair.go2.clean_v1",
    "stair.go2.irregular_v1",
    "stair.go2.rough_v1",
    "push.g1_29dof.legacy_v1",
)
EXPECTED_PROFILE_KEYS = (
    "p0.g1_37_stair_execution",
    "p0.g1_29_push_execution",
    "p0.go2_upstream_execution_basis",
)
EXPECTED_BINDING_KEYS = (
    "stair.g1_37dof.clean_legacy_v1",
    "push.g1_29dof.legacy_v1",
    "go2.official_rough_upstream_basis",
)
EXPECTED_BINDING_PROFILE_MAP = {
    "stair.g1_37dof.clean_legacy_v1": "p0.g1_37_stair_execution",
    "push.g1_29dof.legacy_v1": "p0.g1_29_push_execution",
    "go2.official_rough_upstream_basis": "p0.go2_upstream_execution_basis",
}

RECEIPT_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase0_runtime_receipt.v3"
)
PROFILE_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase0_robot_runtime_facts.v3"
)
BINDING_SCHEMA_VERSION = (
    "true_rl_agent.variant_construction.phase0_cell_binding_facts.v2"
)

PROFILE_FIELDS = {
    "schema_version",
    "profile_key",
    "robot_family_id",
    "runtime_provider",
    "upstream_revision",
    "asset_closure",
    "joint_names",
    "body_names",
    "support_body_names",
    "default_state",
    "morphology",
    "actuators",
    "runtime_facts_hash",
}
BINDING_FIELDS = {
    "schema_version",
    "cell_key",
    "source_config",
    "source_hashes",
    "control",
    "actions",
    "policy_observation",
    "sensors",
    "policy_io_schema_hash",
    "sensor_schema_hash",
    "binding_facts_hash",
}
ASSET_CLOSURE_FIELDS = {
    "kind",
    "root_identity",
    "entries",
    "unresolved",
    "file_count",
    "total_bytes",
    "closure_hash",
}
ACTUATOR_PARAMETER_FIELDS = {
    "effort_limit",
    "velocity_limit",
    "stiffness",
    "damping",
    "armature",
    "friction",
    "dynamic_friction",
}
LOCAL_BINDING_SOURCE_PATHS = {
    "stair_env": "src/true_rl_agent/tasks/stair_climbing/isaaclab_env_cfg.py",
    "stair_protocol": "configs/tasks/stair_protocol_v1.json",
    "push_runtime": "configs/tasks/push_recovery.yaml",
}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise AssertionError(f"{name} mismatch: {actual!r} != {expected!r}")


def _assert_sha256(value: Any, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AssertionError(f"{name} is not a lowercase SHA-256 digest")


def _assert_exact_keys(
    payload: Mapping[str, Any], expected: set[str], name: str
) -> None:
    _assert_equal(set(payload), expected, f"{name} fields")


def _without_hash(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != field}


def _resolved_contracts() -> dict[str, Any]:
    output_dir = PROJECT_ROOT / "__P0_SENTINEL_OUTPUT__"
    stair_config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs/tasks/stair_protocol_v2.json"
    )
    stair_provider = StairClimbingCompositionProvider()
    stair = resolve_task_composition(
        provider=stair_provider,
        values=stair_provider.cli_defaults(),
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )
    push_config = PushRecoveryTaskConfig()
    push_provider = PushRecoveryCompositionProvider()
    push = resolve_task_composition(
        provider=push_provider,
        values=push_provider.cli_defaults(),
        context=TaskCompositionContext(
            project_root=PROJECT_ROOT,
            output_dir=output_dir,
        ),
    )
    stair_backend = stair.backend_adapter.config
    push_backend = push.backend_adapter.config
    reward_runtime = stair_config.protocol_v2.complete_reward_runtime.to_public_dict()
    return {
        "stair": {
            "protocol_v1_hash": stair_config.protocol_v1.protocol_hash,
            "protocol_v2_hash": stair_config.protocol_v2.protocol_hash,
            "task_protocol_hash": stair_config.task_protocol_hash,
            "evaluation_profile_hash": stair_config.evaluation_profile.profile_hash,
            "evaluation_protocol_hash": stair_config.evaluation_protocol().protocol_hash,
            "acceptance_protocol_hash": stair_config.acceptance_protocol.protocol_hash,
            "seed_cohort_hash": stair_config.seed_resolver().namespace_cohort_hash(
                STAIR_FORMAL_EVAL_NAMESPACE
            ),
            "capability_profile_hash": (
                stair.task_plugin.task_spec.candidate_capability_profile.profile_hash
            ),
            "evaluation_binding_hash": stair_backend.evaluation_binding.binding_hash,
            "genesis_snapshot_hash": canonical_json_hash(
                dict(stair.genesis_identity_snapshot())
            ),
            "reward_abi_hash": canonical_json_hash(reward_runtime),
            "evaluation_cell_ids": [
                cell.cell_id for cell in stair_config.evaluation_profile.cells
            ],
            "robot": stair_backend.robot,
            "observation_schema_hash": stair_backend.observation_schema_hash,
            "action_schema_hash": stair_backend.action_schema_hash,
        },
        "push": {
            "task_protocol_hash": (
                push.task_plugin.task_spec.resolved_task_protocol_hash()
            ),
            "evaluation_protocol_hash": push_config.evaluation_protocol().protocol_hash,
            "acceptance_protocol_hash": push_config.acceptance_protocol().protocol_hash,
            "seed_cohort_hash": push_config.seed_cohort_hash(),
            "capability_profile_hash": (
                push.task_plugin.task_spec.candidate_capability_profile.profile_hash
            ),
            "evaluation_binding_hash": push_backend.evaluation_binding.binding_hash,
            "genesis_snapshot_hash": canonical_json_hash(
                dict(push.genesis_identity_snapshot())
            ),
            "robot": push_backend.robot,
            # These backend constants are compatibility labels, not content hashes.
            "observation_schema_identity": {
                "value": push_backend.observation_schema_hash,
                "kind": "legacy_label",
                "content_bound": False,
            },
            "action_schema_identity": {
                "value": push_backend.action_schema_hash,
                "kind": "legacy_label",
                "content_bound": False,
            },
        },
    }


def _runner_source_contract() -> dict[str, str]:
    return {
        "train_builder_source_sha256": hashlib.sha256(
            inspect.getsource(IsaacLabRslRlBackendAdapter._train_argv).encode()
        ).hexdigest(),
        "eval_builder_source_sha256": hashlib.sha256(
            inspect.getsource(IsaacLabRslRlBackendAdapter._eval_argv).encode()
        ).hexdigest(),
        "backend_source_sha256": _file_sha256(
            PROJECT_ROOT / "src/true_rl_agent/backends/isaac_lab_rsl_rl.py"
        ),
        "stair_train_entrypoint_sha256": _file_sha256(
            PROJECT_ROOT / "scripts/isaaclab_stair_rsl_rl_train.py"
        ),
        "stair_eval_entrypoint_sha256": _file_sha256(
            PROJECT_ROOT / "scripts/isaaclab_stair_rsl_rl_eval.py"
        ),
        "push_train_entrypoint_sha256": _file_sha256(
            PROJECT_ROOT / "scripts/isaaclab_rsl_rl_train.py"
        ),
        "push_eval_entrypoint_sha256": _file_sha256(
            PROJECT_ROOT / "scripts/isaaclab_rsl_rl_eval.py"
        ),
    }


def _source_file_contract(snapshot: Mapping[str, Any]) -> dict[str, str]:
    return {
        relative_path: _file_sha256(PROJECT_ROOT / relative_path)
        for relative_path in snapshot["source_file_hashes"]
    }


def _verify_reward_abi(snapshot: Mapping[str, Any]) -> None:
    stair_config = StairClimbingTaskConfig.defaults(
        PROJECT_ROOT / "configs/tasks/stair_protocol_v2.json"
    )
    reward_runtime = stair_config.protocol_v2.complete_reward_runtime.to_public_dict()
    declared = snapshot["stair_reward_abi"]
    fields = reward_runtime["input_schema"]["fields"]
    shape_by_name = {field["name"]: field["trailing_shape"] for field in fields}
    _assert_equal(canonical_json_hash(reward_runtime), declared["reward_abi_hash"], "Stair reward ABI hash")
    _assert_equal(reward_runtime["schema_version"], declared["schema_version"], "Stair reward runtime schema")
    _assert_equal(reward_runtime["input_schema"]["schema_version"], declared["input_schema_version"], "Stair reward input schema")
    _assert_equal(len(fields), declared["input_field_count"], "Stair reward fields")
    for field_name, expected_shape in declared["robot_specific_shapes"].items():
        _assert_equal(shape_by_name[field_name], expected_shape, f"Stair reward field shape: {field_name}")


def _normalized_reference_runner_argv() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        provider = StairClimbingCompositionProvider()
        values = provider.cli_defaults()
        values["execution_device"] = "cuda:0"
        composition = resolve_task_composition(
            provider=provider,
            values=values,
            context=TaskCompositionContext(
                project_root=PROJECT_ROOT,
                output_dir=root / "composition-output",
            ),
        )
        reference = composition.reference_program_artifact
        if reference is None:
            raise AssertionError("Stair composition did not resolve its reference program")
        binding = composition.backend_adapter.evaluation_binding_for(
            composition.task_plugin.task_spec
        )
        jobs = composition.backend_adapter.materialize_reference_jobs(
            session_id="phase0-product-reference",
            task_spec=composition.task_plugin.task_spec,
            reference=reference,
            artifact_root=root / "reference-artifacts",
            policy_initialization=composition.policy_initialization,
            evaluation_binding=binding,
        )

        def normalize(argv: Any) -> list[str]:
            normalized: list[str] = []
            previous: str | None = None
            for value in argv:
                text = (
                    str(value)
                    .replace(str(root), "<RUN_ROOT>")
                    .replace(str(PROJECT_ROOT), "<PROJECT_ROOT>")
                    .replace("\\", "/")
                )
                if previous is not None and previous.endswith("-json"):
                    text = "<CANONICAL_JSON_SHA256:" + hashlib.sha256(text.encode()).hexdigest() + ">"
                normalized.append(text)
                previous = str(value)
            return normalized

        payload = {
            "schema_version": "true_rl_agent.phase0.normalized_reference_runner_argv.v1",
            "train": normalize(jobs.train_spec.metadata["cli_argv"]),
            "eval": normalize(jobs.eval_spec.metadata["cli_argv"]),
        }
        payload["snapshot_hash"] = canonical_json_hash(payload)
        return payload


def _verify_asset_closure(closure: Mapping[str, Any], profile_key: str) -> None:
    _assert_exact_keys(closure, ASSET_CLOSURE_FIELDS, f"{profile_key} asset closure")
    _assert_equal(canonical_json_hash(_without_hash(closure, "closure_hash")), closure["closure_hash"], f"{profile_key} asset closure hash")
    _assert_equal(closure["unresolved"], [], f"{profile_key} unresolved assets")
    entries = closure["entries"]
    _assert_equal(len(entries), closure["file_count"], f"{profile_key} asset count")
    _assert_equal(sum(entry["size_bytes"] for entry in entries), closure["total_bytes"], f"{profile_key} asset bytes")
    if not entries or closure["total_bytes"] <= 0:
        raise AssertionError(f"{profile_key} asset closure is empty")
    identities: list[str] = []
    for index, entry in enumerate(entries):
        _assert_exact_keys(entry, {"identity", "size_bytes", "sha256", "content_version"}, f"{profile_key} asset entry {index}")
        if not isinstance(entry["identity"], str) or not entry["identity"]:
            raise AssertionError(f"{profile_key} asset identity is empty")
        if not isinstance(entry["size_bytes"], int) or entry["size_bytes"] <= 0:
            raise AssertionError(f"{profile_key} asset entry has no bytes")
        _assert_sha256(entry["sha256"], f"{profile_key} asset entry SHA")
        identities.append(entry["identity"])
    if len(identities) != len(set(identities)):
        raise AssertionError(f"{profile_key} asset identities are not unique")


def _verify_robot_profile(profile: Mapping[str, Any]) -> None:
    profile_key = profile["profile_key"]
    _assert_exact_keys(profile, PROFILE_FIELDS, profile_key)
    _assert_equal(profile["schema_version"], PROFILE_SCHEMA_VERSION, profile_key)
    _assert_equal(canonical_json_hash(_without_hash(profile, "runtime_facts_hash")), profile["runtime_facts_hash"], f"{profile_key} runtime facts hash")
    _verify_asset_closure(profile["asset_closure"], profile_key)

    joint_names = profile["joint_names"]
    body_names = profile["body_names"]
    support_names = profile["support_body_names"]
    for values, label in ((joint_names, "joint names"), (body_names, "body names"), (support_names, "support body names")):
        if not values or len(values) != len(set(values)):
            raise AssertionError(f"{profile_key} {label} are empty or duplicated")
    if not set(support_names).issubset(body_names):
        raise AssertionError(f"{profile_key} support bodies are not runtime bodies")

    default_state = profile["default_state"]
    _assert_exact_keys(default_state, {"root_pos_m", "root_quat_wxyz", "root_lin_vel_mps", "root_ang_vel_radps", "joint_pos_rad_by_name"}, f"{profile_key} default state")
    _assert_equal(len(default_state["root_pos_m"]), 3, f"{profile_key} root position")
    _assert_equal(len(default_state["root_quat_wxyz"]), 4, f"{profile_key} root quaternion")
    _assert_equal(len(default_state["root_lin_vel_mps"]), 3, f"{profile_key} root velocity")
    _assert_equal(len(default_state["root_ang_vel_radps"]), 3, f"{profile_key} root angular velocity")
    _assert_equal(set(default_state["joint_pos_rad_by_name"]), set(joint_names), f"{profile_key} default joint map")

    morphology = profile["morphology"]
    _assert_exact_keys(morphology, {"default_root_height_m", "default_body_pos_relative_to_root_m", "nominal_root_to_support_body_center_z_m"}, f"{profile_key} morphology")
    _assert_equal(morphology["default_root_height_m"], default_state["root_pos_m"][2], f"{profile_key} root height")
    if morphology["nominal_root_to_support_body_center_z_m"] <= 0:
        raise AssertionError(f"{profile_key} nominal support scale is not positive")
    body_positions = morphology["default_body_pos_relative_to_root_m"]
    _assert_equal(set(body_positions), set(body_names), f"{profile_key} body positions")
    if any(len(position) != 3 for position in body_positions.values()):
        raise AssertionError(f"{profile_key} body positions must be xyz triples")

    actuators = profile["actuators"]
    if not actuators:
        raise AssertionError(f"{profile_key} has no actuator facts")
    actuator_names: list[str] = []
    owned_joints: list[str] = []
    for actuator in actuators:
        _assert_exact_keys(actuator, {"name", "class", "joint_names", "parameters_by_joint"}, f"{profile_key} actuator")
        actuator_names.append(actuator["name"])
        actuator_joints = actuator["joint_names"]
        if not actuator_joints or len(actuator_joints) != len(set(actuator_joints)):
            raise AssertionError(f"{profile_key} actuator joint list is invalid")
        if not set(actuator_joints).issubset(joint_names):
            raise AssertionError(f"{profile_key} actuator owns an unknown joint")
        owned_joints.extend(actuator_joints)
        parameters = actuator["parameters_by_joint"]
        _assert_exact_keys(parameters, ACTUATOR_PARAMETER_FIELDS, f"{profile_key} actuator parameters")
        for field_name, values in parameters.items():
            if not isinstance(values, dict):
                raise AssertionError(f"{profile_key} actuator {field_name} is not resolved per joint")
            _assert_equal(set(values), set(actuator_joints), f"{profile_key} actuator {field_name} joints")
    if len(actuator_names) != len(set(actuator_names)):
        raise AssertionError(f"{profile_key} actuator names are duplicated")
    if len(owned_joints) != len(set(owned_joints)):
        raise AssertionError(f"{profile_key} joints belong to multiple actuator groups")
    _assert_equal(set(owned_joints), set(joint_names), f"{profile_key} actuator coverage")


def _verify_binding(binding: Mapping[str, Any], *, snapshot: Mapping[str, Any]) -> None:
    cell_key = binding["cell_key"]
    _assert_exact_keys(binding, BINDING_FIELDS, cell_key)
    _assert_equal(binding["schema_version"], BINDING_SCHEMA_VERSION, cell_key)
    _assert_equal(canonical_json_hash(_without_hash(binding, "binding_facts_hash")), binding["binding_facts_hash"], f"{cell_key} binding facts hash")
    _assert_equal(canonical_json_hash({"actions": binding["actions"], "policy_observation": binding["policy_observation"]}), binding["policy_io_schema_hash"], f"{cell_key} policy IO schema hash")
    _assert_equal(canonical_json_hash(binding["sensors"]), binding["sensor_schema_hash"], f"{cell_key} sensor schema hash")

    control = binding["control"]
    _assert_exact_keys(control, {"simulation_dt_s", "decimation", "control_dt_s"}, f"{cell_key} control")
    _assert_equal(round(control["simulation_dt_s"] * control["decimation"], 10), control["control_dt_s"], f"{cell_key} control timing")
    if not binding["actions"]:
        raise AssertionError(f"{cell_key} has no policy action")
    for action in binding["actions"]:
        _assert_exact_keys(action, {"name", "class", "asset_name", "joint_names", "scale", "use_default_offset"}, f"{cell_key} action")

    observation = binding["policy_observation"]
    _assert_exact_keys(observation, {"concatenate_terms", "history_length", "enable_corruption_by_mode", "terms"}, f"{cell_key} policy observation")
    if not observation["terms"]:
        raise AssertionError(f"{cell_key} has no policy observation terms")
    term_names: list[str] = []
    for term in observation["terms"]:
        _assert_exact_keys(term, {"name", "callable", "params", "noise", "clip", "scale"}, f"{cell_key} observation term")
        term_names.append(term["name"])
    if len(term_names) != len(set(term_names)):
        raise AssertionError(f"{cell_key} observation terms are duplicated")

    _assert_exact_keys(binding["sensors"], {"height_scanner", "contact_forces"}, f"{cell_key} sensors")
    if binding["sensors"]["height_scanner"] is None:
        raise AssertionError(f"{cell_key} height scanner was not resolved")
    if binding["sensors"]["contact_forces"] is None:
        raise AssertionError(f"{cell_key} contact sensor was not resolved")

    remote_hashes = set(snapshot["server_environment"]["remote_source_hashes"].values())
    for source_name, source_hash in binding["source_hashes"].items():
        _assert_sha256(source_hash, f"{cell_key} source {source_name}")
        local_path = LOCAL_BINDING_SOURCE_PATHS.get(source_name)
        if local_path is not None:
            _assert_equal(_file_sha256(PROJECT_ROOT / local_path), source_hash, f"{cell_key} local source {source_name}")
        elif source_hash not in remote_hashes:
            raise AssertionError(f"{cell_key} source {source_name} is not bound to server evidence")


def _normalized_receipt_term_params(term: Mapping[str, Any]) -> dict[str, Any]:
    parameters = dict(term["params"])
    sensor_cfg = parameters.get("sensor_cfg")
    if isinstance(sensor_cfg, Mapping):
        parameters.pop("sensor_cfg")
        parameters["sensor_cfg_name"] = sensor_cfg.get("name")
    return parameters


def _verify_stair_runtime_authority(*, snapshot: Mapping[str, Any], profiles: Mapping[str, Mapping[str, Any]], bindings: Mapping[str, Mapping[str, Any]]) -> None:
    protocol = json.loads((PROJECT_ROOT / "configs/tasks/stair_protocol_v1.json").read_text(encoding="utf-8"))["policy_compatibility"]
    profile = profiles["p0.g1_37_stair_execution"]
    binding = bindings["stair.g1_37dof.clean_legacy_v1"]
    _assert_equal(profile["joint_names"], protocol["robot_schema"]["joint_names"], "G1-37 runtime joints versus Stair protocol")
    _assert_equal(profile["body_names"], protocol["robot_schema"]["body_names"], "G1-37 runtime bodies versus Stair protocol")
    _assert_equal(binding["control"]["simulation_dt_s"], protocol["simulation_dt_s"], "G1-37 simulation dt")
    _assert_equal(binding["control"]["decimation"], protocol["decimation"], "G1-37 decimation")
    _assert_equal(binding["control"]["control_dt_s"], protocol["control_dt_s"], "G1-37 control dt")

    runtime_actions = binding["actions"]
    protocol_actions = protocol["action"]["terms"]
    _assert_equal(len(runtime_actions), len(protocol_actions), "G1-37 action term count")
    for runtime, declared in zip(runtime_actions, protocol_actions, strict=True):
        _assert_equal(runtime["name"], declared["manager_term"], "G1-37 action name")
        _assert_equal(runtime["asset_name"], declared["asset_name"], "G1-37 action asset")
        _assert_equal(runtime["joint_names"], declared["joint_selector"], "G1-37 action selector")
        _assert_equal(runtime["scale"], declared["scale"], "G1-37 action scale")
        _assert_equal(runtime["use_default_offset"], declared["use_default_offset"], "G1-37 action offset")
        _assert_equal(len(profile["joint_names"]), declared["dim"], "G1-37 action dimension")
    _assert_equal(len(profile["joint_names"]), protocol["action"]["total_dim"], "G1-37 total action dimension")

    runtime_observation = binding["policy_observation"]
    protocol_observation = protocol["policy_observation"]
    _assert_equal(runtime_observation["concatenate_terms"], protocol_observation["concatenate_terms"], "G1-37 observation concatenation")
    _assert_equal(len(runtime_observation["terms"]), len(protocol_observation["terms"]), "G1-37 observation term count")
    for runtime, declared in zip(runtime_observation["terms"], protocol_observation["terms"], strict=True):
        _assert_equal(runtime["name"], declared["manager_term"], "G1-37 observation name")
        _assert_equal(runtime["callable"], declared["callable"], "G1-37 observation callable")
        _assert_equal(_normalized_receipt_term_params(runtime), declared["parameters"], f"G1-37 observation parameters: {runtime['name']}")
    _assert_equal(sum(term["shape"][0] for term in protocol_observation["terms"]), protocol_observation["total_dim"], "G1-37 protocol observation dimension")
    for term_name in ("joint_pos", "joint_vel", "actions"):
        term = next(item for item in protocol_observation["terms"] if item["manager_term"] == term_name)
        _assert_equal(term["shape"], [len(profile["joint_names"])], f"G1-37 {term_name} dimension")

    stair_contract = snapshot["local_contracts"]["stair"]
    _assert_equal(stair_contract["observation_schema_hash"], protocol_observation["schema_hash"], "Stair observation schema authority")
    _assert_equal(stair_contract["action_schema_hash"], protocol["action"]["schema_hash"], "Stair action schema authority")


def _verify_runtime_receipt(*, snapshot: Mapping[str, Any], receipt_path: Path) -> tuple[int, int]:
    declared = snapshot["runtime_probe_receipt"]
    _assert_equal(declared["path"], "PHASE0_RUNTIME_PROBE_RECEIPT.json", "runtime receipt path")
    _assert_equal(_file_sha256(receipt_path), declared["file_sha256"], "runtime receipt file")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _assert_exact_keys(receipt, {"schema_version", "capture_kind", "captured_repository_head", "probe_script_sha256", "float_quantization", "environment", "robot_runtime_facts", "legacy_cell_binding_facts", "receipt_hash"}, "runtime receipt")
    _assert_equal(receipt["schema_version"], RECEIPT_SCHEMA_VERSION, "runtime receipt schema")
    _assert_equal(
        receipt["float_quantization"],
        {
            "decimal_places": 6,
            "resolution": 0.000001,
            "negative_zero_normalized": True,
        },
        "runtime receipt float quantization",
    )
    _assert_equal(canonical_json_hash(_without_hash(receipt, "receipt_hash")), receipt["receipt_hash"], "runtime receipt hash")
    _assert_equal(receipt["receipt_hash"], declared["receipt_hash"], "declared runtime receipt hash")
    _assert_equal(receipt["probe_script_sha256"], declared["probe_script_sha256"], "declared probe script")
    _assert_equal(_file_sha256(RUNTIME_PROBE_PATH), receipt["probe_script_sha256"], "current probe script")
    _assert_equal(receipt["captured_repository_head"], snapshot["repository"]["captured_local_head"], "runtime receipt captured repository HEAD")
    _assert_equal(receipt["environment"]["isaac_lab_source_commit"], snapshot["server_environment"]["isaac_lab_source_commit"], "runtime Isaac Lab revision")
    _assert_equal(receipt["environment"]["unitree_rl_lab_source_commit"], snapshot["server_environment"]["unitree_rl_lab_source_commit"], "runtime unitree_rl_lab revision")

    profiles: dict[str, Mapping[str, Any]] = {}
    for profile in receipt["robot_runtime_facts"]:
        _verify_robot_profile(profile)
        if profile["profile_key"] in profiles:
            raise AssertionError("runtime profile keys are duplicated")
        profiles[profile["profile_key"]] = profile
    _assert_equal(set(profiles), set(EXPECTED_PROFILE_KEYS), "runtime profile keys")
    _assert_equal({key: value["runtime_facts_hash"] for key, value in profiles.items()}, declared["profile_runtime_hashes"], "declared runtime profile hashes")
    _assert_equal({key: value["asset_closure"]["closure_hash"] for key, value in profiles.items()}, declared["asset_closure_hashes"], "declared asset closure hashes")
    _assert_equal(profiles["p0.g1_37_stair_execution"]["upstream_revision"], receipt["environment"]["isaac_lab_source_commit"], "G1-37 upstream revision")
    _assert_equal(profiles["p0.go2_upstream_execution_basis"]["upstream_revision"], receipt["environment"]["isaac_lab_source_commit"], "Go2 upstream revision")
    _assert_equal(profiles["p0.g1_29_push_execution"]["upstream_revision"], receipt["environment"]["unitree_rl_lab_source_commit"], "G1-29 upstream revision")

    bindings: dict[str, Mapping[str, Any]] = {}
    for binding in receipt["legacy_cell_binding_facts"]:
        _verify_binding(binding, snapshot=snapshot)
        if binding["cell_key"] in bindings:
            raise AssertionError("runtime binding keys are duplicated")
        bindings[binding["cell_key"]] = binding
    _assert_equal(set(bindings), set(EXPECTED_BINDING_KEYS), "runtime binding keys")
    _assert_equal({key: value["binding_facts_hash"] for key, value in bindings.items()}, declared["binding_facts_hashes"], "declared runtime binding hashes")
    _assert_equal({key: value["policy_io_schema_hash"] for key, value in bindings.items()}, declared["policy_io_schema_hashes"], "declared policy IO hashes")
    _assert_equal({key: value["sensor_schema_hash"] for key, value in bindings.items()}, declared["sensor_schema_hashes"], "declared sensor hashes")
    _assert_equal(
        declared["binding_profile_map"],
        EXPECTED_BINDING_PROFILE_MAP,
        "runtime binding-to-profile map",
    )
    for cell_key, profile_key in EXPECTED_BINDING_PROFILE_MAP.items():
        action = bindings[cell_key]["actions"]
        _assert_equal(len(action), 1, f"{cell_key} action count")
        _assert_equal(action[0]["joint_names"], [".*"], f"{cell_key} all-joint selector")
        if len(profiles[profile_key]["joint_names"]) <= 0:
            raise AssertionError(f"{cell_key} resolved action dimension is empty")
    _verify_stair_runtime_authority(snapshot=snapshot, profiles=profiles, bindings=bindings)
    return len(profiles), len(bindings)


def verify(*, snapshot_path: Path = SNAPSHOT_PATH, receipt_path: Path = RUNTIME_RECEIPT_PATH) -> dict[str, Any]:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    repository = snapshot["repository"]
    captured_heads = {
        repository["captured_local_head"],
        repository["captured_origin_head"],
        repository["captured_server_head"],
    }
    if len(captured_heads) != 1:
        raise AssertionError("Phase 0 capture provenance heads disagree")
    _assert_equal(snapshot["matrix_scope"]["cells"], list(EXPECTED_CELLS), "Core cells")
    _assert_equal(snapshot["matrix_scope"]["cell_variants"], list(EXPECTED_VARIANTS), "Core variants")
    _assert_equal(_resolved_contracts(), snapshot["local_contracts"], "resolved contracts")
    _assert_equal(_runner_source_contract(), snapshot["runner_contract"]["source_hashes"], "runner source contract")
    _assert_equal(_source_file_contract(snapshot), snapshot["source_file_hashes"], "source file hashes")
    _verify_reward_abi(snapshot)
    _assert_equal(_normalized_reference_runner_argv(), snapshot["runner_contract"]["normalized_reference_stair_argv"], "normalized reference Stair argv")
    profile_count, binding_count = _verify_runtime_receipt(snapshot=snapshot, receipt_path=receipt_path)
    _assert_equal(canonical_json_hash(_without_hash(snapshot, "snapshot_hash")), snapshot["snapshot_hash"], "snapshot hash")
    return {
        "status": "pass",
        "schema_version": "true_rl_agent.variant_construction.phase0.verify.v3",
        "captured_head": repository["captured_local_head"],
        "product_source_file_count": len(snapshot["source_file_hashes"]),
        "cell_count": len(EXPECTED_CELLS),
        "variant_count": len(EXPECTED_VARIANTS),
        "stair_evaluation_cell_count": len(snapshot["local_contracts"]["stair"]["evaluation_cell_ids"]),
        "runtime_profile_count": profile_count,
        "runtime_binding_count": binding_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT_PATH)
    parser.add_argument("--runtime-receipt", type=Path, default=RUNTIME_RECEIPT_PATH)
    parser.add_argument("--emit-runner-argv", action="store_true")
    args = parser.parse_args()
    payload = _normalized_reference_runner_argv() if args.emit_runner_argv else verify(snapshot_path=args.snapshot, receipt_path=args.runtime_receipt)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
