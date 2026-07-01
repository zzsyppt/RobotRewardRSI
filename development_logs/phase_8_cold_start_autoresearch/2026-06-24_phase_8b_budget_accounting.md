# Phase 8B：immutable runner 预算公平性最小实现

日期：2026-06-24

## 设计判断

Phase 8B 的方向是合理的：预算公平单位应该是 `total_env_steps`，不是固定 iteration。

原因：

- code-level candidate 未来可能改变每个 iteration 内的计算结构；
- 固定 iteration 会让不同 recipe 的环境交互量或计算量不可比；
- Phase 7A 的 4000-iteration calibration 来自 512 envs，只能作为 env-step budget 的换算依据；
- Phase 8A 已验证 4096 envs 能跑 1 iteration，但这不等于长训容量已验证。

边界：

- Phase 8B 只做 runner 层 budget accounting / cap；
- 不开放 LLM-authored candidate；
- 不做 OS/container 隔离；
- 不运行 public/eval accept-reject；
- 不把 4096 envs 的 1-iteration smoke 解释成长训容量证明。

## 实现

新增：

```text
src/humanoid_agent/immutable_runner/budget.py
configs/cold_start/phase8b_budget.yaml
scripts/phase8b_budget_runner.py
scripts/phase8b_budget_selftest.py
```

复用并扩展：

```text
scripts/phase8a_min_runner.py
```

## Budget 换算

Phase 7A calibration 参考：

```text
4000 iterations * 512 envs * 24 steps/iteration/env = 49,152,000 env steps
```

Phase 8B 默认并行 envs：

```text
num_envs = 4096
```

因此同等 env-step budget 下：

```text
49,152,000 / (4096 * 24) = 500 iterations
```

Phase 8B config 中启用：

```text
derive_max_iterations_from_env_step_budget = true
```

runner 会从 `env_step_budget` 派生 `max_iterations`，而不是让 candidate 或人工手写 iteration 数作为公平单位。

## Enforcement

当前最小实现会记录并检查：

- planned env steps；
- `config_derived_total_timesteps`，来自训练脚本写入的结构化 `training_accounting.config_derived_total_timesteps`；
- elapsed seconds；
- checkpoint count；
- checkpoint total bytes；
- numeric latest checkpoint selection。

违反 cap 时，runner summary 标为 reject/fail。
如果执行后无法解析 `config_derived_total_timesteps`，则 fail closed。
stdout 中的 `Total timesteps` 只保留为可信历史调试 fallback，Phase 8B 默认禁用。

结构化 `training_accounting` 由 runner-owned train script 写入，包括：

- `num_steps_per_env`；
- `env_steps_per_iteration`；
- `config_derived_total_timesteps`；
- `total_timesteps_semantics=config_derived_not_runtime_observed`；
- `policy_update_iterations`；
- `num_learning_epochs`；
- `num_mini_batches`；
- `estimated_optimizer_steps`；
- `estimated_gradient_steps`；
- `estimated_forward_backward_calls`。

当前 optimizer/grad/forward-backward 是基于 RSL-RL 配置的估算计数，不是底层 optimizer hook 观测值。
真正不可绕过的 grad/optimizer hook 仍应在 Phase 8D/8E 前补齐。

注意：

- grad step / optimizer step 目前没有可靠 hook，只能先作为 Phase 8B 后续增强项；
- wall-clock 只是 safety cap，不是公平主指标；
- checkpoint policy 仍由 runner 外部数字选择，不信 summary 的旧 `latest_checkpoint` fallback。

## 验收

本地 selftest 覆盖：

1. `4096 envs + 49,152,000 env steps + 24 steps/iter/env` 派生 `500 iterations`；
2. Phase8B runner command 固定 `--num-envs 4096`；
3. Phase8B runner command 派生 `--max-iterations 500`；
4. fake train log 的最后一个 `Total timesteps` 被解析；
5. 伪造 stdout overrun 不覆盖结构化 summary；
6. `config_derived_total_timesteps` 精确等于 planned/budget 时 pass；
7. `config_derived_total_timesteps` 超出 budget 时 fail；
8. `config_derived_total_timesteps` 缺失时 fail closed；
9. `num_steps_per_env` 与 budget contract 不一致时 fail；
10. checkpoint count 超 secondary cap 时 fail；
11. summary 中旧污染 `latest_checkpoint=model_900.pt` 不影响 runner 选择 `model_1999.pt`；
12. Phase8B dry-run summary 标记 `phase=8B-budget`，并暴露 derived runner config。

## Review 问题处理记录

- F1：真实存在，已修改。Phase 8B 不再把 stdout grep 当作权威来源；默认使用训练脚本结构化 summary，stdout fallback 默认关闭。
- F2：真实存在，继续部分处理。已修正 `num_learning_epochs` / `num_mini_batches` 读取路径，从 `agent_cfg.algorithm` 读取，使 optimizer/gradient/forward-backward 估算字段不再在真实 run 中恒为 null。但这些仍然只是 config-derived estimates，不是底层 hook。真正 grad/optimizer hook 必须在 8D/8E 前完成，不能宣称 8B 已解决。
- F3：真实存在，已修改。增加 `require_config_derived_matches_planned`，`config_derived_total_timesteps` 与 planned 不一致默认 fail；不靠零余量的 stdout 解析决定通过。
- F4：真实存在，已修改。执行后要求 train summary 的 `num_steps_per_env` 与 budget contract 的 `steps_per_iteration` 一致，不一致 fail。
- F5：真实存在，已修改。candidate fingerprint 从 manifest + entrypoint 扩展为整棵 `training_code/` tree hash。
- F6：真实存在，已修改。stdout regex 只保留为显式 fallback；Phase 8B 默认不用 stdout 作为权威来源。
- `exec_module` 仍在当前进程运行候选代码：属 Phase 8A/8B 设计边界，已记录为 8C 必须解决，不在 8B 内伪装成安全沙盒。

## B1/B2/B3 追加处理

- B1：真实存在，已修改。真实 run 里 `estimated_optimizer_steps` / `num_learning_epochs` / `num_mini_batches` 为 null 的原因是 RSL-RL 配置把 PPO 参数放在 `agent_cfg.algorithm` 下，而旧代码从 `agent_cfg` 顶层读取。现在 `_training_accounting` 会优先读取 `agent_cfg.algorithm.num_learning_epochs` 和 `agent_cfg.algorithm.num_mini_batches`。
- B2：真实存在，已修改。旧字段名 `observed_total_timesteps` 会误导后续 Agent 以为这是 runtime observation。Phase 8B 当前没有 VecEnv.step counter，因此字段改为 `config_derived_total_timesteps`，并写明 `total_timesteps_semantics=config_derived_not_runtime_observed`。真正 runtime observed counter 留到 Phase 8D/8E。
- B3：不影响当前代码路径。服务器上遗留的失败 reject/smoke 目录只作为历史证据保留，不进入 runner 判定；不需要为此改代码。后续清理服务器 run 目录时再按 run-dir 前缀归档/删除。

真实 4096-env 1-iteration Phase8B smoke 还暴露了一个 runner bug：subprocess 之前依赖外层
`PYTHONPATH`，直接执行时训练脚本无法 import `humanoid_agent`。已修改 runner，使 subprocess
环境由 runner 显式注入 `PROJECT_ROOT/src`，不再依赖调用者 shell。

## 后续

Phase 8B 后续增强可以继续做：

- 从训练循环或 RSL-RL runner hook 记录 grad/update/optimizer step；
- 对长训做真实 capacity validation；
- 把 budget accounting 接入后续 Phase8E public/eval summary；
- 在 Phase8C 隔离完成后再运行 LLM-authored code candidate。
