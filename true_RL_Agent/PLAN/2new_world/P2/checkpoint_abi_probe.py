"""Bounded migration comparison; run once with each installed interpreter.

This probe does not train or alter its frozen checkpoint. Production conversion
is imported from the backend; only the old-runtime side constructs ActorCritic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from tensordict import TensorDict

from true_rl_agent.backends.rsl_rl_checkpoint import legacy_model_states
from true_rl_agent.robots.isaaclab_locomotion import (
    G1_ROUGH_AGENT_CONFIGURATION,
    GO2_ROUGH_AGENT_CONFIGURATION,
)

parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=("g1", "go2"), required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--device", default="cpu")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7121)
torch.backends.cuda.matmul.allow_tf32 = False
checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
state = checkpoint["model_state_dict"]
configuration = (
    G1_ROUGH_AGENT_CONFIGURATION
    if args.robot == "g1"
    else GO2_ROUGH_AGENT_CONFIGURATION
)
actor_dim = state["actor.0.weight"].shape[1]
critic_dim = state["critic.0.weight"].shape[1]
num_actions = state["std"].numel()
# Numpy inputs are fixed independently of different Torch RNG implementations.
rng = np.random.default_rng(7121)
inputs = {
    "policy": rng.standard_normal((64, actor_dim)).astype("float32"),
    "critic": rng.standard_normal((64, critic_dim)).astype("float32"),
}
obs = TensorDict(
    {k: torch.from_numpy(v).to(args.device) for k, v in inputs.items()}, batch_size=[64]
)
version = importlib.metadata.version("rsl-rl-lib")
if version.startswith("3."):
    from rsl_rl.modules import ActorCritic

    model = ActorCritic(
        obs,
        {"policy": ["policy"], "critic": ["critic"]},
        num_actions,
        **configuration.to_dict()["policy"],
    ).to(args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.inference_mode():
        actor_output, critic_output = model.act_inference(obs), model.evaluate(obs)
    label = "old"
else:
    from rsl_rl.models import MLPModel

    mapped = legacy_model_states(checkpoint)
    sections = configuration.runtime_sections()
    models = {}
    for role, dimension in (("actor", num_actions), ("critic", 1)):
        cfg = dict(sections[role])
        cfg.pop("class_name")
        models[role] = MLPModel(
            obs, {"actor": ["policy"], "critic": ["critic"]}, role, dimension, **cfg
        ).to(args.device)
        models[role].load_state_dict(mapped[role + "_state_dict"], strict=True)
        models[role].eval()
    with torch.inference_mode():
        actor_output, critic_output = models["actor"](obs), models["critic"](obs)
    label = "new"
outputs = {"actor": actor_output.cpu().numpy(), "critic": critic_output.cpu().numpy()}
stem = args.robot + "_" + args.device.replace(":", "_")
np.savez(
    args.output / (stem + "_" + label + ".npz"),
    **outputs,
    **{"input_" + k: v for k, v in inputs.items()},
)
report = {
    "robot": args.robot,
    "device": args.device,
    "rsl_rl": version,
    "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    "actor_observation_dim": actor_dim,
    "critic_observation_dim": critic_dim,
    "num_actions": num_actions,
    "legacy_optimizer_migration": "unsupported",
}
if label == "new":
    old = np.load(args.output / (stem + "_old.npz"))
    report["comparison"] = {}
    for role, value in outputs.items():
        atol = 1e-6 if args.device == "cpu" else 2e-5
        np.testing.assert_allclose(value, old[role], atol=atol, rtol=1e-5)
        report["comparison"][role] = {
            "max_abs_error": float(np.max(np.abs(value - old[role]))),
            "atol": atol,
            "rtol": 1e-5,
        }
    report["status"] = "PASS"
(args.output / (stem + "_" + label + ".json")).write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report), flush=True)
