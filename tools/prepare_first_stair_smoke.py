"""为首次 Go2 爬楼梯 PPO 微型检查生成真实 reference 作业参数，不启动训练。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPOSITORY_ROOT / "true_RL_Agent"
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from true_rl_agent.backends.isaac_lab_execution_environment import IsaacLabExecutionEnvironmentManifest
from true_rl_agent.composition import TaskCompositionContext, resolve_task_composition
from true_rl_agent.composition.defaults import build_application_composition_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    allowed = (REPOSITORY_ROOT / "runs").resolve()
    if output == allowed or allowed not in output.parents:
        parser.error("准备目录必须是本仓库 runs/ 下的新子目录")
    output.mkdir(parents=True, exist_ok=False)
    provider = build_application_composition_registry().get("unitree_stair_climbing")
    values = {**provider.cli_defaults(), "cell_variant_id": "stair.go2_12dof.unified_v3",
              "isaac_python": sys.executable, "execution_device": "cuda:0", "mode": "diagnostic_prepare"}
    composition = resolve_task_composition(provider=provider, values=values,
        context=TaskCompositionContext(project_root=PROJECT_ROOT, output_dir=output))
    if composition.reference_program_artifact is None:
        raise RuntimeError("当前 composition 未提供受信任 reference reward")
    environment = composition.backend_adapter.resolve_session_execution_environment(
        session_id="first-go2-stair-smoke", session_dir=output, allow_create=False)
    manifest = IsaacLabExecutionEnvironmentManifest.from_mapping(environment)
    jobs = composition.backend_adapter.materialize_reference_jobs(
        session_id="first-go2-stair-smoke", task_spec=composition.task_plugin.task_spec,
        reference=composition.reference_program_artifact, artifact_root=output / "reference_artifact",
        policy_initialization=composition.policy_initialization,
        backend_execution_environment_manifest_hash=manifest.manifest_hash)
    train_argv = list(jobs.train_spec.metadata["cli_argv"])
    # 正式 runner 会设置 cwd；微型检查也可以从仓库根目录调用。
    entrypoint = Path(train_argv[1])
    if not entrypoint.is_absolute():
        train_argv[1] = str((PROJECT_ROOT / entrypoint).resolve())
    # 仅调整运行产物路径，固定科学协议和微型 PPO 预算由现有 executor 控制。
    train_argv.extend(("--kit_args", f"--portable-root {output / 'kit'}"))
    (output / "train_argv.json").write_text(json.dumps(train_argv, indent=2) + "\n", encoding="utf-8")
    (output / "environment_manifest.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    report = {"status": "PREPARED_NOT_EXECUTED", "cell_variant_id": values["cell_variant_id"],
              "train_argv": str(output / "train_argv.json"), "provider_requests": 0,
              "train_process_starts": 0, "formal_training_claim": False,
              "next_entrypoint": "true_RL_Agent/scripts/training_numeric_smoke.py",
              "diagnostic_workload": {"num_envs": 16, "steps_per_env": 8, "iterations": 1,
                                      "env_transitions": 128}}
    (output / "preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
