# Phase 8E fixes after first real 3-round self-evolution run

日期：2026-06-28

第一条真正端到端的 3 轮 DeepSeek 自演进跑完(28.9 min,0 accepted)。round 3 真训练 21 min、封 checkpoint、跑 6 rollout、trusted eval 给 0% 成功——正确拒绝。但暴露出几处必须改的地方。

## 三轮结果

- round 1：候选过 8D，但训练运行时崩、没封 checkpoint。
- round 2：候选没过 8D（pretraining 反馈）。
- round 3：候选过 8D，真训练 21 min，0% 成功，动作爆炸(`mean_action_l2≈2.9e8`)，被 gate 拒。

## 修复

### #1（硬阻断）budget_accounting 在合法运行上误判失败
真实 train-start summary 只在 `probes.train_start` 带 `num_envs`/`max_iterations`，不带 `config_derived_total_timesteps`/`num_steps_per_env`/`policy_update_iterations`。原 `build_runner_owned_budget_summary` 把这些当作"子必须回显"，缺失即 violation → **任何真候选永远过不了 budget 门、永不可能 accept**。

改：这些是配置派生、runner 拥有的值，由不可变 Phase 8B 配置补齐(`build_budget_accounting` 收 runner_accounting)；`_reported_budget_field_violations` 改为**只在子上报冲突值时判 violation，缺失不判**。防伪造保持(子报 `num_envs=1` 仍因 mismatch 被拒)。回归：最小真实型 summary 现在 pass、且 `config_derived_total_timesteps=49152000`/`num_steps_per_env=24` 由 runner 填。

### #2 训练异常回传 LLM（之前被吞）
内层 train 脚本本就记录 `errors=[{type,message}]`，但 minimal_rootfs_probe 没传递。改：probe 把首个异常(类型+message[:400])放进 `train_start.training_error`；full-round 把它带进训练失败反馈的 `detail.training_error`。候选自己代码的报错回传给同一 LLM 是它自我纠错所需。`_append_failure_feedback` 加健壮回退：若自由文本触发 public-only 守卫，则去掉自由文本、记 status-only，不丢整轮。

### #3 崩溃误判 failed_safety_gate → reject
"没产出 checkpoint"(候选崩)被标 `failed_safety_gate`。改：先判 `_train_start_checkpoint_absent`(checkpoint_present False 或无 sealed 路径)→ `reject`；`failed_safety_gate` 只留给"checkpoint 存在但不安全/非 runner 封"。回归：checkpoint 缺失 → reject(exit 1)，且 training_error 进反馈。

### #4 门失败反馈带具体指标
失败 gate 之前只回传 status。改：`_project_gate` + `_gate_status_only` 带 public-safe `failures` 摘要(按 metric 去重、留最大幅值的 actual/limit)。LLM 现在能看到"mean_action_l2 爆到 2.9e8"，而非只知"motion gate fail"。

### 轮数
evolution loop 默认 `--max-rounds` 3 → **15**。

## 边界不变
全部失败反馈仍只写公开内容、整记录过 `assert_public_only_payload`；budget/eval/checkpoint/metric 仍 runner-owned，候选不可改；评测端不被腐蚀。
