# Phase 8E-2：组件契约改造（数据 → 带契约的 callable）

日期：2026-06-26

## 目标

把 Phase 8E candidate 从旧的声明式 hint（例如 `reward_weight_hint` / `curriculum_hint`）升级为真实 Python callable contract，同时不提前启动正式 Isaac/RSL-RL 训练。

本阶段的完成标准是：

- candidate 能返回可执行 `RewardTerm` / `LossTerm` / `ScheduleTerm`；
- runner-owned adapter 能校验结构、预求值 schedule，并守护 reward/loss 输出；
- prompt / 示例 candidate / selftest 都切到 callable 口径；
- 不伪造“已接入 PPO loop”的结论。

## 设计判断

8E-2 只做 contract 和 adapter，不直接改训练 loop。原因：

- true code-level 的核心是候选写函数、runner 直接调用；这一步必须先有明确输入/输出契约；
- 直接把 callable 接进 Isaac/RSL-RL 训练 loop 会同时牵涉预算计数、checkpoint 权威产物、独立 eval、OOD/motion gate，容易把 8E-3 的工作提前混入；
- 当前安全边界已经验证 `minimal_rootfs_pivot_root` 可承载 Isaac/CUDA，因此 8E-2 可以先把“可调用什么、怎么守护输出”定清楚。

## 实现内容

新增/修改：

- `src/humanoid_agent/code_sandbox/components.py`
  - 新增 `RewardInputs`；
  - 新增 `RewardTerm`、`LossTerm`、`ScheduleTerm`；
  - `TrainingComponents.to_public_summary()` 只公开结构信息，不暴露 function repr。
- `src/humanoid_agent/cold_start_lab/recipe_adapter.py`
  - `validate_training_components()`；
  - `build_reward_inputs()`；
  - `precompute_schedule_table()`；
  - `evaluate_reward_terms()` / `evaluate_loss_terms()`；
  - `guard_vector_output()`。
- `scripts/phase8e_contract_selftest.py`
  - 覆盖 callable candidate、schedule 预求值、非法 schedule key、非 callable、非 finite 输出、错误 shape。
- `examples/cold_start_candidates/declarative_curriculum_candidate/`
  - 从旧 hint 字典改为 `RewardTerm` + `ScheduleTerm` callable。
- `src/humanoid_agent/cold_start_lab/phase8e.py`
  - prompt payload 增加 callable component contract、`RewardInputs` 字段和 runner schedule 白名单。
- `configs/cold_start/phase8e_loop.yaml`
  - candidate capability 描述从 declaration/hint 改为 callable contract。

## 当前边界

已完成：

- schedule 可以被 runner 预求值为 public numeric table；
- reward/loss callable 的输出 shape、finite、dtype summary 能被 adapter 守护；
- callable 示例候选能通过 8D/8E scaffold selftest。

未完成，留给 8E-3：

- 把 reward/loss callable 真接进 Isaac/RSL-RL step/PPO loop；
- runner-owned checkpoint / train summary / candidate output 分离；
- 长训 env-step / grad-step / wall-clock 预算权威计数；
- 独立 trusted eval 的真实 public/OOD/motion gate。

## 验收

本地：

```text
python -m py_compile \
  src/humanoid_agent/code_sandbox/components.py \
  src/humanoid_agent/cold_start_lab/recipe_adapter.py \
  src/humanoid_agent/cold_start_lab/phase8e.py \
  scripts/phase8e_contract_selftest.py

python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

python scripts/phase8d_verifier_selftest.py
=> {"status": "pass", "test": "phase8d_verifier_selftest", ...}

python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", ...}
```

服务器：

```text
/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8d_verifier_selftest.py
=> {"status": "pass", "test": "phase8d_verifier_selftest", "hard_isolation": true}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": true}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_safety_selftest.py \
  --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-26_phase8e_2_safety_regression
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": true, "safety_model_status": "pass"}
```

结论：Phase 8E-2 可以作为后续 8E-3 训练 loop 接入的 contract 基线。

## 2026-06-26 追加修复：materialized shape 作为权威输出形状

发现并修复 `guard_vector_output()` 的一个低危但应在 8E-3 前处理的契约漏洞：旧实现优先信任候选对象自报的 `.shape`，恶意或错误对象可以让 `.shape == (num_envs,)`，但 `.tolist()` 实际返回嵌套二维数据，从而绕过 shape gate。

修复后：

- `guard_vector_output()` 先将输出 materialize 为 Python nested data；
- 用 nested data 推导真实 shape，并要求它等于 `(num_envs,)`；
- 如果对象同时自报 `.shape`，则要求自报 shape 与 materialized shape 一致；
- `scripts/phase8e_contract_selftest.py` 新增 `FakeShapeNestedOutput` 回归，覆盖“伪造 `.shape`、真实 `.tolist()` 为二维嵌套”的情况。

这不改变 8E-2 的边界：仍然只做 callable contract 与 adapter，不提前接入正式 PPO/Isaac 训练 loop。

## 2026-06-27 计划清理迁入摘要

从 `2026-06-23_cold_start_autoresearch_plan.md` 的 `Phase 8E-code` 段迁出以下落地状态，避免计划正文混入开发日志：

- `code_sandbox/components.py` 已定义 `RewardInputs`、`RewardTerm`、`LossTerm`、`ScheduleTerm`；
- `cold_start_lab/recipe_adapter.py` 已提供 runner-owned 结构校验、schedule 预求值、reward/loss 输出 shape/finite 守护；
- 示例候选已从旧 `reward_weight_hint` / `curriculum_hint` 迁移为真实 callable；
- `phase8e_contract_selftest.py` 已覆盖合法 callable、schedule 表、非法 schedule key、非 callable、非 finite 输出和错误 shape。

边界仍保持：8E-2 只完成 callable contract 与 adapter；把 reward/loss 真接入 Isaac/RSL-RL step/PPO loop、runner-owned checkpoint 与 public eval 仍属于 8E-3 及其后续阶段。
