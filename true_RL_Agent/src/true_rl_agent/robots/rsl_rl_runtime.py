"""Robot-owned RSL-RL policy and optimizer configurations.

Tasks own budgets, distributions, rewards, and evaluation.  The resolved
runtime adapter selects the policy/optimizer shape required by its robot row.
"""

from __future__ import annotations

from true_rl_agent.backends.rsl_rl_training_config import RslRlAgentConfiguration
from true_rl_agent.robots.defaults import (
    G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID,
    G1_37DOF_ISAACLAB_ADAPTER_ID,
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID,
    GO2_12DOF_ISAACLAB_ADAPTER_ID,
)
from true_rl_agent.robots.isaaclab_locomotion import (
    G1_ROUGH_AGENT_CONFIGURATION,
    GO2_ROUGH_AGENT_CONFIGURATION,
)


G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION = RslRlAgentConfiguration(
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


_AGENT_CONFIGURATIONS = {
    G1_29DOF_UNITREE_RL_LAB_ADAPTER_ID: (
        G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION
    ),
    G1_37DOF_ISAACLAB_ADAPTER_ID: G1_ROUGH_AGENT_CONFIGURATION,
    G1_37DOF_ISAACLAB_GROUPED_ADAPTER_ID: G1_ROUGH_AGENT_CONFIGURATION,
    GO2_12DOF_ISAACLAB_ADAPTER_ID: GO2_ROUGH_AGENT_CONFIGURATION,
}


def rsl_rl_agent_configuration(
    runtime_adapter_id: str,
) -> RslRlAgentConfiguration:
    try:
        return _AGENT_CONFIGURATIONS[runtime_adapter_id]
    except KeyError as exc:
        raise KeyError(
            "runtime adapter has no robot-owned RSL-RL configuration: "
            + str(runtime_adapter_id)
        ) from exc


__all__ = [
    "G1_29DOF_UNITREE_RL_LAB_AGENT_CONFIGURATION",
    "rsl_rl_agent_configuration",
]
