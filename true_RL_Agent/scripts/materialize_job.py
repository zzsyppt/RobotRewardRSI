from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from true_rl_agent.backends.isaac_lab_rsl_rl import IsaacLabRslRlBackendAdapter, IsaacLabRslRlBackendConfig
from true_rl_agent.candidates.schema import CandidateChangeSet, CandidatePackageRef
from true_rl_agent.policy import PolicyInitialization
from true_rl_agent.tasks.push_recovery import PushRecoveryTaskConfig, PushRecoveryTaskPlugin


def _runner_execution_payload(metadata: dict) -> dict:
    payload = {
        "cli_command": metadata["cli_command"],
        "cli_argv": list(metadata["cli_argv"]),
    }
    if "summary_path" in metadata:
        payload["summary_path"] = metadata["summary_path"]
    if "sealed_checkpoint_uri" in metadata:
        payload["sealed_checkpoint_uri"] = metadata["sealed_checkpoint_uri"]
    if "checkpoint_output_path" in metadata:
        payload["checkpoint_output_path"] = metadata["checkpoint_output_path"]
    if "work_dir" in metadata:
        payload["work_dir"] = metadata["work_dir"]
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize an IsaacLab/RSL-RL job bundle.")
    parser.add_argument("--round-id", default="manual-round-001")
    parser.add_argument("--candidate-id", default="manual-candidate")
    parser.add_argument("--component", action="append", default=None)
    parser.add_argument("--candidate-path", default=None)
    args = parser.parse_args(argv)

    components = args.component or ["reward_callable"]
    candidate_path = args.candidate_path or f"audit://{args.candidate_id}"
    candidate = CandidatePackageRef(
        candidate_id=args.candidate_id,
        path=candidate_path,
        change_set=CandidateChangeSet(frozenset(components)),
    )
    config = PushRecoveryTaskConfig()
    backend = IsaacLabRslRlBackendAdapter(
        IsaacLabRslRlBackendConfig(
            isaac_lab_python="/data/yuyangliu/envs/isaaclab/bin/python",
            train_entrypoint="scripts/isaaclab_rsl_rl_train.py",
            eval_entrypoint="scripts/isaaclab_rsl_rl_eval.py",
            default_budget_steps=config.budget_steps,
            train_num_envs=config.train_num_envs,
            train_max_iterations=config.train_max_iterations,
            train_steps_per_iteration=config.train_steps_per_iteration,
            eval_num_envs=config.eval_num_envs,
            public_seed_count=config.public_seed_count,
            public_accept_seed_count=config.public_accept_seed_count,
            trusted_seed_count=7,
            work_dir=str(PROJECT_ROOT),
        )
    )
    bundle = backend.materialize_jobs(
        round_id=args.round_id,
        task_spec=PushRecoveryTaskPlugin().task_spec,
        candidate=candidate,
        policy_initialization=PolicyInitialization.random(),
    )
    payload = {
        "schema_version": "true_rl_agent.scripts.materialized_job.v1",
        "candidate_changed_components": sorted(candidate.change_set.changed_components),
        "train": bundle.train_spec.to_public_dict(),
        "eval": bundle.eval_spec.to_public_dict(),
        "runner_execution": {
            "train": _runner_execution_payload(dict(bundle.train_spec.metadata)),
            "eval": _runner_execution_payload(dict(bundle.eval_spec.metadata)),
        },
        "task_invariants": dict(bundle.task_invariants.values) if bundle.task_invariants else {},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
