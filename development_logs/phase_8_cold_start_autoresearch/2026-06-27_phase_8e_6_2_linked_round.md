# Phase 8E-6-2：sealed checkpoint 到 trusted eval / decision 的链接层

## 目标

把已经由 runner/minimal-rootfs 训练侧产出的 sealed checkpoint 接到 Phase 8E 的公开评测与决策链路：

1. 从 runner-owned training summary 读取 `probes.train_start.latest_checkpoint`；
2. 只接受 `/runner_artifacts/train_start_smoke/sealed_checkpoints/model_<n>.pt` 形态的隔离命名空间路径；
3. 用 `train_start_paths.resolve_train_start_checkpoint_path()` 映射到 orchestrator 显式传入的 runner artifacts root；
4. 在 trusted parent 侧重新执行 `weights_only=True` checkpoint safe-load gate；
5. 调用 `phase8e_trusted_eval_probe.py` 产出 `public_result.json`；
6. 把这个 trusted public result 交给 `run_phase8e_round()` 做 verifier / decision / ledger / inheritance。

这一步不启动正式长训，不生成 LLM candidate，也不把 1-iteration smoke 伪装成 full-budget run。它只把后续 task2 编排需要的“训练产物 -> 公开评测 -> 决策”接口接牢。

## 设计判断

8E-6-2 采用薄编排层，而不是重写 `phase8e.py`：

- `phase8e.py` 继续负责 prompt、8D verifier、accept/reject、ledger 和 recipe inheritance；
- `phase8e_trusted_eval_probe.py` 继续负责 checkpoint-only trusted eval 边界和 public_result 写出；
- 新增 `phase8e_linked.py` 只负责把 runner 产物安全地喂给 trusted eval，并把结果传给现有决策层。

关键安全边界：

- checkpoint host path 不能来自候选目录、candidate output 或 summary 中的 host path；
- summary 里只把 isolated path 当作 untrusted data；
- host path 必须由 runner-owned `runner_artifacts_root` + 固定 sealed checkpoint 规则推导；
- trusted eval 前再次 safe-load checkpoint，避免训练进程内 advisory check 被候选污染；
- public_result 由 trusted eval probe 写出，不允许候选手工提供。

## 实现文件

- `src/humanoid_agent/cold_start_lab/phase8e_linked.py`
  - `resolve_train_start_checkpoint_for_eval()`
  - `run_phase8e_linked_round()`
  - `Phase8ELinkedError`
- `scripts/phase8e_linked_round.py`
  - task2 编排可调用的 CLI；
  - 显式要求 `--train-summary`、`--runner-artifacts-root`、`--public-dev-summary`。
- `scripts/phase8e_linked_round_selftest.py`
  - 路径契约回归；
  - 有 torch 时完整跑 checkpoint safe-load -> trusted eval -> public_result -> decision -> ledger。
- `src/humanoid_agent/cold_start_lab/__init__.py`
  - 导出链接层 API。

## 验收

本地执行：

```text
python scripts/phase8e_linked_round_selftest.py
python scripts/phase8e_selftest.py
python scripts/phase8e_trusted_public_eval_selftest.py
python -m py_compile src/humanoid_agent/cold_start_lab/phase8e_linked.py scripts/phase8e_linked_round.py scripts/phase8e_linked_round_selftest.py
```

本地无 torch，因此 `phase8e_linked_round_selftest.py` 的完整 trusted eval 分支会跳过，但路径契约与 forbidden path 回归会执行。服务器侧需要补跑完整分支。

## 后续接入点

8E task2 的正式编排应调用 `scripts/phase8e_linked_round.py`，并传入：

- full-budget isolated training 产出的 runner summary；
- runner-owned `runner_artifacts` 根目录；
- trusted public-dev / public-accept rollout summary；
- stable `lab_dir` 以保持 accepted recipe inheritance。

图中提醒的 checkpoint 路径问题已在接口上收紧：rollout/eval 消费的 checkpoint 必须是 runner 解析后的 sealed checkpoint host path，而不是候选上报路径。

## 2026-06-27 复查修正：budget gate 来源收紧

发现 `budget_accounting` 虽然是 required gate，但 8E-6-2 初版在未显式传入 budget summary 时会把 `train_summary_path` 直接传给 trusted eval。这样会让 child/training summary 里的 `status`、`num_envs`、`max_iterations` 等字段被原样投影到 `public_result.budget_accounting`，属于门完整性问题。

修正：

- 新增 `build_runner_owned_budget_summary()`；
- 从 `phase8e_loop.yaml -> runner.budget_config_path -> phase8b_budget.yaml` 读取 runner-owned immutable budget；
- 用 Phase8B 配置派生 `planned_env_steps`、`num_envs`、`max_iterations`、`num_steps_per_env`；
- child/training summary 只作为 evidence，必须与 runner-owned 预算字段一致；
- 缺少或不一致的 `num_envs`、`max_iterations`、`num_steps_per_env`、`config_derived_total_timesteps` 会让 `budget_accounting.status=fail`；
- `scripts/phase8e_linked_round.py` 不再接受 `--budget-summary` 直通参数，避免绕过 runner-owned budget summary 生成。

同步判断：

- `checkpoint_for_eval` 与已有 rollout summaries 的真实绑定仍是 task2 正式编排问题。8E-6-2 当前消费预先存在的 rollout summaries，因此 checkpoint safe-load gate 对 metric 只提供前置安全保证；后续 task2 必须让 rollout runner 在 `checkpoint_for_eval["host_path"]` 上真实产出 public-dev/public-accept summaries，才能闭合 metric 证据链。
