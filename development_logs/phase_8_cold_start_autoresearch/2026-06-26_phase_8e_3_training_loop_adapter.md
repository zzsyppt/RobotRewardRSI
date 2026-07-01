# Phase 8E-3：callable 接入隔离训练 loop

日期：2026-06-26

## 目标

把 Phase 8E-2 的 callable contract 从“离线契约验证”推进到“真实训练进程内执行”：

- `RewardTerm` 在 runner-owned 训练路径中被调用，并把 guarded candidate reward 加到 env reward；
- `LossTerm` 在训练进程内执行和守护，但本阶段不伪称已插入 PPO optimizer；
- `DiagnosticTerm` 从任意 dict 收紧为带契约 callable，只允许 public numeric summary；
- schedule callable 仍由 runner 预求值为数值表，不在可信 runner/eval 进程里执行候选代码；
- 候选仍不能获得 env、runner、optimizer、checkpoint selector、eval config、success metric。

## 设计判断

8E-3 采用“最小真接入”，不重写 RSL-RL：

1. 训练脚本仍是 immutable runner 的一部分；
2. 候选 package 只提供 `TrainingComponents`；
3. 在隔离训练进程内，runner 加载候选并构造 public `RewardInputs`；
4. `Phase8EVecEnvRewardWrapper` 代理 `VecEnv.step()`，在 step reward 返回后调用候选 `RewardTerm`；
5. 输出必须经过 runtime guard 后才可进入 reward；
6. eval 进程仍不加载候选代码。

这满足“真 code-level”的核心要求：候选写的是可执行 Python reward/diagnostic 逻辑，并被训练进程直接调用；安全边界仍来自 OS 隔离训练进程 + 独立可信 eval，而不是把候选限制成 DSL。

## 对前置提醒的处理

1. `evaluate_*` / `precompute_*` 的执行位置  
   已接入 `phase1_push_recovery_train_smoke.py --candidate-dir`。在服务器 minimal-rootfs train-start 中，候选 callable 会在隔离训练进程里执行，而不是在可信 eval 进程里执行。

2. `diagnostics` 未定义问题  
   新增 `DiagnosticTerm`。`diagnostics` 不再允许任意 dict；执行结果只允许数值、布尔或按 env 的数值向量，并输出聚合摘要。

3. 4096-env `.tolist()` 性能问题  
   新增 `guard_vector_value()`。torch tensor 走 on-device shape / finite 检查，只同步少量标量；非 torch 路径保留 materialized guard，用于本地 selftest 和非训练契约检查。

4. clip 语义  
   明确定义为：先检查 shape/finite，再按 term 的 `clip_min` / `clip_max` 裁剪，最后乘 `weight` 并加入 reward。非有限值不能被 clip 掩盖。

## 实现内容

- `src/humanoid_agent/code_sandbox/components.py`
  - 新增 `DiagnosticTerm`；
  - `TrainingComponents.diagnostics` 进入 callable contract；
  - public summary 不暴露函数对象。

- `src/humanoid_agent/cold_start_lab/recipe_adapter.py`
  - 新增 `GuardedVectorResult`；
  - 新增 `guard_vector_value()`；
  - torch tensor runtime guard 使用 `torch.isfinite`；
  - 新增 `evaluate_diagnostic_terms()`；
  - `validate_training_components()` 校验 diagnostics 类型。

- `src/humanoid_agent/cold_start_lab/training_adapter.py`
  - 新增 `Phase8ETrainingAdapter`；
  - 新增 `Phase8EVecEnvRewardWrapper`；
  - 构造 public `RewardInputs`；
  - 在 `VecEnv.step()` reward 路径加入 guarded candidate reward；
  - 记录 runtime summary。

- `scripts/phase1_push_recovery_train_smoke.py`
  - 新增 `--candidate-dir`；
  - 在 `RslRlVecEnvWrapper` 后、`OnPolicyRunner` 前接入 Phase8E adapter；
  - summary 记录 candidate component summary 和 runtime summary。

- `scripts/phase8e_minimal_rootfs_probe.py`
  - train-start smoke 显式传入 `/candidate`，确保服务器隔离训练进程中真实执行 candidate callable。

- `examples/cold_start_candidates/declarative_curriculum_candidate/`
  - diagnostics 迁移为 `DiagnosticTerm`。

- `scripts/phase8e_training_adapter_selftest.py`
  - 验证 step reward 代理、loss guard、diagnostic execution、非有限 reward 拒绝。

## 明确未完成边界

本阶段没有提前做：

- 正式长训；
- public eval / public_accept / OOD stress / motion-quality gate；
- hidden audit；
- aux `LossTerm` 插入 PPO optimizer；
- dynamic schedule 改写 env config 的完整执行器；
- observation/action wrapper 权限开放。

这些仍属于后续正式 immutable runner / eval runner 工作。8E-3 的完成标准是：callable 已能在隔离训练进程内进入真实训练 reward 路径，并且所有输出由 runner-owned adapter 守护。

## 本地验收

```text
python -m py_compile ...
=> pass

python scripts/phase8e_training_adapter_selftest.py
=> {"status": "pass", "test": "phase8e_training_adapter_selftest"}

python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": false}

python scripts/phase8d_verifier_selftest.py
=> {"status": "pass", "test": "phase8d_verifier_selftest", "hard_isolation": false}

python scripts/phase8e_safety_selftest.py
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": false, "safety_model_status": "pass"}
```

服务器验收将在同步后执行，重点是 minimal-rootfs 内 `--train-start-smoke --candidate-dir /candidate` 是否能真实产出 checkpoint，并且 summary 中包含 Phase8E candidate runtime。

## 服务器验收

服务器提交：`e7aa6ea`

轻量验收：

```text
/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_training_adapter_selftest.py
=> {"status": "pass", "test": "phase8e_training_adapter_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": true}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8d_verifier_selftest.py
=> {"status": "pass", "test": "phase8d_verifier_selftest", "hard_isolation": true}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_safety_selftest.py ...
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": true, "safety_model_status": "pass"}
```

真实隔离 train-start：

```text
unshare -Urmnfp --mount-proc /data/yuyangliu/envs/isaaclab/bin/python \
  scripts/phase8e_minimal_rootfs_probe.py \
  --candidate-dir examples/cold_start_candidates/declarative_curriculum_candidate \
  --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-26_phase8e_3_train_start \
  --hidden-probe-file /data/yuyangliu/secrets/deepseek.env \
  --train-start-smoke \
  --num-envs 4096 \
  --max-iterations 1 \
  --seed 801 \
  --device cuda:0

=> {"summary_file": "/candidate_output/minimal_rootfs_summary.json", "status": "pass"}
```

关键证据：

- `minimal_rootfs_status = pass`
- `cuda_training_smoke_passed = true`
- `runner_inputs_no_protected_tokens = true`
- `hidden_file_read_blocked = true`
- `repo_config_read_blocked = true`
- `repo_write_blocked = true`
- `network_connect_blocked = true`
- `secret_env_absent = true`
- `oldroot_detached = true`
- `train_start_checkpoint_present = true`
- checkpoint：`/candidate_output/train_start_smoke/rsl_rl/model_0.pt`
- `phase8e_candidate_runtime.step_calls = 24`
- `phase8e_candidate_runtime.reward_term_calls = 24`
- `phase8e_candidate_runtime.diagnostic_calls = 1`
- reward 输出：`shape = [4096]`, `dtype = torch.float32`, `finite = true`, `clip_applied = true`
- diagnostic 输出：`height_margin` 与 `upright_penalty` 均为 `[4096]` 向量聚合摘要

结论：Phase 8E-3 已完成“candidate callable 在 OS 隔离训练进程内进入真实训练 reward 路径”的最小真接入。后续若要宣称 aux loss 真的影响 PPO，需要新增 runner-owned PPO loss hook；本阶段没有伪称这一点。

## 2026-06-27：进入 eval/accept 前的契约与产物硬化

本轮逐条核实了 3 个问题，结论如下：

1. **checkpoint / train summary 落在候选可写空间：真实存在，已修。**
   旧路径中 `phase8e_minimal_rootfs_probe.py --train-start-smoke` 把 `--run-dir` 和 `--summary-file` 都指向 `/candidate_output`，而候选 callable 与训练 runner 同进程执行，候选可通过 `atexit`/后台线程在 `raise SystemExit` 退出链上篡改 `model_0.pt` 或 summary。现在改为：
   - `/candidate_output` 仍是候选可写空间，只用于非权威输出、HOME/cache 和普通探针；
   - `/runner_artifacts` 是 train-start 的 runner-owned 权威产物目录；
   - `phase1_push_recovery_train_smoke.py` 新增 `--runner-artifacts-dir`；
   - RSL-RL `log_dir` 改为 `<runner_artifacts_dir>/rsl_rl`；
   - runner 选出的 checkpoint 会复制封存到 `<runner_artifacts_dir>/sealed_checkpoints/`，summary 记录 `source_latest_checkpoint`、`latest_checkpoint`、`latest_checkpoint_sha256`；
   - summary 写入使用预先捕获的低层 `os.open/os.write/os.replace`，脚本退出使用预先捕获的 `os._exit`，避免普通 Python `atexit` 钩子二次篡改。

   说明：这不是最终的“同进程候选不可作恶”证明。同进程执行候选 callable 仍不是完整安全边界；最终 8E-code 仍应把“执行候选代码”和“权威 accept/eval 决策”拆成不同进程/路径。本轮修复的目标是消除 train-start 冷启动 gate 的明显可写路径漏洞，确保进入后续 eval/accept 前权威 checkpoint 有封存与 hash 证据链。

2. **`RewardInputs` 对外宣称字段但实际降维/置零：真实存在，已修。**
   旧 `training_adapter` 把 `base_lin_vel/base_ang_vel/joint_pos/joint_vel/last_action` 转成 L2 范数，并把 `push_active/push_vec_public` 静默置零。这样 LLM 会误以为拿到了完整向量或真实 push 信息。现在改为：
   - 若 runner/env 能提供向量，则直接传原始 per-env 向量；
   - `last_action` 保持原始 action 向量，不再降维；
   - `push_active/push_vec_public` 仍为 Phase 8E-3 未接入字段，但在 `runtime_contract.input_field_status` 中明确标注为 `not_instrumented...zero_filled`；
   - 自测新增对 `last_action` 和 push placeholder 的回归检查。

3. **`_find_reward_index` 启发式扫描 tuple：真实存在，已修。**
   旧实现按“第一个 `(num_envs,)` 非 bool 向量”猜 reward。对 RSL-RL VecEnv 来说 reward slot 应固定为 tuple index 1；否则未来 wrapper 若在后面返回 float dones 或额外数值向量，会被误判。现在改为：
   - 只接受 `result[1]` 作为 numeric reward vector；
   - 若 index 1 不是 `(num_envs,)` 数值向量，直接 `ComponentContractError`；
   - 自测新增“后续 tuple 出现另一个数值向量也不能被当作 reward”的回归。

本轮没有提前实现 public eval / accept / hidden audit / PPO aux-loss hook；修复范围仅限 8E-3 训练接入前必须收紧的契约和权威产物路径。

## 2026-06-27 复审修复：把 checkpoint 安全闸移出候选进程 + 开放 reward_weight_scale + 输入 provenance

复审发现上一版 `assert_safe_checkpoint` 放在 `phase1_push_recovery_train_smoke.py`（候选同进程）里执行，候选 reward callable 同进程运行，可 monkeypatch `assert_safe_checkpoint` / `torch.save` 伪造 `latest_checkpoint_safe_load: true`。服务器实证：同进程 rebind 后 child 报告 true，但 fresh 引用复验为 false（`finding_real: true`）。修复：

- **权威 checkpoint 安全闸移到可信 eval 边界**（`phase8e_trusted_eval_probe.py`，从不加载候选代码）：加载前 `checkpoint_safety_report()`（`weights_only=True`），不安全即 `reject(checkpoint_failed_safe_load_gate)`，写权威 `checkpoint_safety{safe_load, sha256}`；`safety_model` 增聚合检查 `trusted_eval_checkpoint_safe_load`。
- 训练进程内的检查**降级为 advisory**，字段改名 `child_reported_checkpoint_safe_load` 并注释“可被候选伪造，勿据此 gate”。
- 可复用原语集中在 `immutable_runner/safe_checkpoint.py`：`assert_safe_checkpoint` / `checkpoint_safety_report` / `safe_load_checkpoint` / `file_sha256`。前向契约：未来正式 eval runner 在 `runner.load`(weights_only=False) 前必须先过此闸。
- 回归：`phase8e_eval_load_safety_selftest`（benign/恶意/缺失/伪造-但-可信复验拦截）、`phase8e_safety_selftest`（真 benign checkpoint + 恶意 checkpoint 被 trusted gate reject）。

权限开放（训练侧、eval 不受影响）：

- **`reward_weight_scale` 课程生效**：runner 按训练进度从预求值表取该键，clamp [0,10] 乘到候选 reward 权重；`ScheduleTerm` 从“预求值但不生效”升级为真生效的训练杠杆。
- **输入 provenance**：`_build_inputs` 一次性记录每个字段 `observed/zero_fallback/not_instrumented/runner_owned`，消除“静默置零却宣称可用”。`_roll_pitch` 修复每步重复计算两次。

服务器端到端复验（GPU 空闲，256 env、1 iter，真 Isaac + 示例候选）：`status: pass`；candidate reward 跑满 24 step、mean≈0.737；`input_provenance` 中 base_height/roll/pitch/base_lin_vel/base_ang_vel/joint_pos/joint_vel/last_action **全部 observed**（证明富化输入在真 env 上真实生效，非占位）；封存 checkpoint 经可信 gate `safe_load: true`。
