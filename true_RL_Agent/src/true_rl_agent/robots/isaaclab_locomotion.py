"""Robot-owned Isaac Lab locomotion execution specifications.

Task harnesses select one of these specs through the resolved runtime adapter.
No task semantics, terrain, reward, or evaluation policy belongs here.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.evaluation.binding import canonical_json_hash
from true_rl_agent.robots.defaults import (
    G1_37DOF_ISAACLAB_ADAPTER_ID,
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    GO2_12DOF_ISAACLAB_ADAPTER_ID,
)


@dataclass(frozen=True)
class IsaacLabSourceSpec:
    import_package: str
    package_relative_path: str
    covered_symbols: tuple[str, ...]


@dataclass(frozen=True)
class IsaacLabLocomotionRuntimeSpec:
    runtime_adapter_id: str
    environment_cfg_entrypoint: str
    runner_cfg_entrypoint: str
    agent_configuration: RslRlAgentConfiguration
    upstream_sources: tuple[IsaacLabSourceSpec, ...]

    def instantiate_environment_cfg(self) -> Any:
        return _load_entrypoint(self.environment_cfg_entrypoint)()

    def instantiate_runner_cfg(self) -> Any:
        return _load_entrypoint(self.runner_cfg_entrypoint)()

    def installed_source_evidence(self) -> dict[str, Any]:
        entries = []
        for source in self.upstream_sources:
            package = importlib.util.find_spec(source.import_package)
            if package is None or not package.submodule_search_locations:
                raise RuntimeError(
                    "Isaac Lab runtime package is unavailable: "
                    + source.import_package
                )
            roots = tuple(Path(item) for item in package.submodule_search_locations)
            matches = tuple(
                root / source.package_relative_path
                for root in roots
                if (root / source.package_relative_path).is_file()
            )
            if len(matches) != 1:
                raise RuntimeError(
                    "Isaac Lab runtime source does not resolve uniquely: "
                    + source.package_relative_path
                )
            data = matches[0].read_bytes()
            entries.append(
                {
                    "import_package": source.import_package,
                    "package_relative_path": source.package_relative_path,
                    "covered_symbols": list(source.covered_symbols),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                }
            )
        payload = {
            "schema_version": (
                "true_rl_agent.robot.isaaclab_locomotion_sources.v1"
            ),
            "runtime_adapter_id": self.runtime_adapter_id,
            "source_files": entries,
        }
        return {**payload, "source_manifest_hash": canonical_json_hash(payload)}


G1_ROUGH_AGENT_CONFIGURATION = RslRlAgentConfiguration(
    policy={
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    algorithm={
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.008,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
)

GO2_ROUGH_AGENT_CONFIGURATION = RslRlAgentConfiguration(
    policy={
        "class_name": "ActorCritic",
        "init_noise_std": 1.0,
        "noise_std_type": "scalar",
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [512, 256, 128],
        "critic_hidden_dims": [512, 256, 128],
        "activation": "elu",
    },
    algorithm={
        "class_name": "PPO",
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "learning_rate": 1.0e-3,
        "schedule": "adaptive",
        "gamma": 0.99,
        "lam": 0.95,
        "entropy_coef": 0.01,
        "desired_kl": 0.01,
        "max_grad_norm": 1.0,
        "value_loss_coef": 1.0,
        "use_clipped_value_loss": True,
        "clip_param": 0.2,
        "normalize_advantage_per_mini_batch": False,
        "rnd_cfg": None,
        "symmetry_cfg": None,
    },
)

_G1_SOURCES = (
    IsaacLabSourceSpec(
        "isaaclab_assets",
        "robots/unitree.py",
        ("G1_MINIMAL_CFG",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/velocity_env_cfg.py",
        ("LocomotionVelocityRoughEnvCfg",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/g1/rough_env_cfg.py",
        ("G1RoughEnvCfg",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/g1/agents/rsl_rl_ppo_cfg.py",
        ("G1RoughPPORunnerCfg",),
    ),
)

_GO2_SOURCES = (
    IsaacLabSourceSpec(
        "isaaclab_assets",
        "robots/unitree.py",
        ("UNITREE_GO2_CFG",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/velocity_env_cfg.py",
        ("LocomotionVelocityRoughEnvCfg",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/go2/rough_env_cfg.py",
        ("UnitreeGo2RoughEnvCfg",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/go2/flat_env_cfg.py",
        ("UnitreeGo2FlatEnvCfg",),
    ),
    IsaacLabSourceSpec(
        "isaaclab_tasks",
        "manager_based/locomotion/velocity/config/go2/agents/rsl_rl_ppo_cfg.py",
        ("UnitreeGo2RoughPPORunnerCfg",),
    ),
)

_SPECS = {
    G1_37DOF_ISAACLAB_ADAPTER_ID: IsaacLabLocomotionRuntimeSpec(
        runtime_adapter_id=G1_37DOF_ISAACLAB_ADAPTER_ID,
        environment_cfg_entrypoint=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1."
            "rough_env_cfg:G1RoughEnvCfg"
        ),
        runner_cfg_entrypoint=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents."
            "rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
        agent_configuration=G1_ROUGH_AGENT_CONFIGURATION,
        upstream_sources=_G1_SOURCES,
    ),
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID: IsaacLabLocomotionRuntimeSpec(
        runtime_adapter_id=G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
        environment_cfg_entrypoint=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1."
            "rough_env_cfg:G1RoughEnvCfg"
        ),
        runner_cfg_entrypoint=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents."
            "rsl_rl_ppo_cfg:G1RoughPPORunnerCfg"
        ),
        agent_configuration=G1_ROUGH_AGENT_CONFIGURATION,
        upstream_sources=_G1_SOURCES,
    ),
    GO2_12DOF_ISAACLAB_ADAPTER_ID: IsaacLabLocomotionRuntimeSpec(
        runtime_adapter_id=GO2_12DOF_ISAACLAB_ADAPTER_ID,
        environment_cfg_entrypoint=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go2."
            "rough_env_cfg:UnitreeGo2RoughEnvCfg"
        ),
        runner_cfg_entrypoint=(
            "isaaclab_tasks.manager_based.locomotion.velocity.config.go2.agents."
            "rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg"
        ),
        agent_configuration=GO2_ROUGH_AGENT_CONFIGURATION,
        upstream_sources=_GO2_SOURCES,
    ),
}


def isaaclab_locomotion_runtime_spec(
    runtime_adapter_id: str,
) -> IsaacLabLocomotionRuntimeSpec:
    try:
        return _SPECS[runtime_adapter_id]
    except KeyError as exc:
        raise KeyError(
            "runtime adapter has no Isaac Lab locomotion execution spec: "
            + str(runtime_adapter_id)
        ) from exc


def _load_entrypoint(value: str) -> Any:
    module_name, separator, object_name = value.partition(":")
    if not separator or not module_name or not object_name:
        raise RuntimeError("Isaac Lab runtime entrypoint is malformed")
    module = importlib.import_module(module_name)
    try:
        result = getattr(module, object_name)
    except AttributeError as exc:
        raise RuntimeError("Isaac Lab runtime entrypoint is unavailable: " + value) from exc
    if not callable(result):
        raise RuntimeError("Isaac Lab runtime entrypoint is not callable: " + value)
    return result


__all__ = [
    "G1_ROUGH_AGENT_CONFIGURATION",
    "GO2_ROUGH_AGENT_CONFIGURATION",
    "IsaacLabLocomotionRuntimeSpec",
    "IsaacLabSourceSpec",
    "isaaclab_locomotion_runtime_spec",
]
