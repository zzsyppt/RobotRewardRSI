"""RSL-RL 5 model state and the minimal frozen RSL-RL 3 checkpoint reader.

Training always writes and resumes the native RSL-RL 5 container. Historical
MLP weights can be used for diagnostic inference; optimizer migration is not
implied by a successful policy conversion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

INITIAL_MODEL_STATE_HASH_SCHEMA_VERSION = "true_rl_agent.initial_policy_state.rsl_rl_5.v2"

def legacy_model_states(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Map model tensors by their declared names, never optimizer positions."""

    old = checkpoint["model_state_dict"]
    actor: dict[str, Any] = {}
    critic: dict[str, Any] = {}
    for name, value in old.items():
        if name.startswith("actor."):
            actor["mlp." + name.removeprefix("actor.")] = value
        elif name.startswith("critic."):
            critic["mlp." + name.removeprefix("critic.")] = value
        elif name.startswith("actor_obs_normalizer."):
            actor["obs_normalizer." + name.removeprefix("actor_obs_normalizer.")] = (
                value
            )
        elif name.startswith("critic_obs_normalizer."):
            critic["obs_normalizer." + name.removeprefix("critic_obs_normalizer.")] = (
                value
            )
        elif name == "std":
            actor["distribution.std_param"] = value
        elif name == "log_std":
            actor["distribution.log_std_param"] = value
        else:
            raise ValueError(f"unsupported historical RSL-RL model tensor: {name}")
    return {"actor_state_dict": actor, "critic_state_dict": critic}


def load_inference_checkpoint(runner: Any, path: str | Path) -> dict[str, Any]:
    """Load both bound models; inference never restores optimizer or iteration."""

    import torch

    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    legacy = "model_state_dict" in checkpoint
    state = legacy_model_states(checkpoint) if legacy else checkpoint
    runner.alg.load(state, {"actor": True, "critic": True}, strict=True)
    return {
        "checkpoint_format": "rsl_rl_3_model_state_dict" if legacy else "rsl_rl_5",
        "actor_loaded": True,
        "critic_loaded": True,
        "optimizer_loaded": False,
        "iteration_loaded": False,
    }


def load_training_checkpoint(runner: Any, path: str | Path) -> None:
    """Resume native state; frozen legacy checkpoints only support inference."""

    import torch

    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    if "model_state_dict" in checkpoint:
        raise ValueError("legacy RSL-RL checkpoint cannot resume native training")
    runner.alg.load(checkpoint, None, strict=True)
    runner.current_learning_iteration = int(checkpoint["iter"])


def initial_model_state_hash(runner: Any) -> str:
    """Bind actor, critic, distribution and normalization before PPO starts."""

    import torch

    state = {
        f"{role}.{name}": value
        for role in ("actor", "critic")
        for name, value in getattr(runner.alg, role).state_dict().items()
    }
    if not state:
        raise ValueError("initial model state is empty")
    digest = hashlib.sha256(INITIAL_MODEL_STATE_HASH_SCHEMA_VERSION.encode("ascii") + b"\x00")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"initial model tensor is non-finite: {name}")
        parts = (
            name.encode("utf-8"),
            str(value.dtype).encode("ascii"),
            ",".join(str(int(size)) for size in value.shape).encode("ascii"),
            value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"),
        )
        for part in parts:
            digest.update(len(part).to_bytes(8, byteorder="big", signed=False))
            digest.update(part)
    return digest.hexdigest()
