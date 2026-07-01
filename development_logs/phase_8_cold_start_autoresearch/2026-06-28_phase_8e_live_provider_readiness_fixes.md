# 2026-06-28 Phase 8E live provider readiness fixes

## 背景

DeepSeek provider 已能返回 candidate package，但 provider-only smoke 暴露两处训练前阻断：

- 8D 动态隔离加载候选时，候选 `import torch` 会触发 `/dev/urandom` 读取，component-load Landlock 未放行该非敏感系统设备，导致候选在训练前被 reject。
- prompt 只写了 entrypoint 名称，没有明确 `build_training_components` 的三参数调用契约，live 候选生成了零参数 builder。

同时，训练前 reject 没有进入 shared lab ledger，下一轮 prompt 无法看到公开 verifier 失败原因，削弱自演进闭环。

## 修复

- `default_landlock_read_paths` 加入 `/dev/urandom` 与 `/dev/random` 的只读 Landlock 规则，保持 repo/secret/hidden/network 仍不可达。
- Phase 8E prompt 的公开 component contract 增加精确 builder 签名：
  - `def build_training_components(task_spec, base_env_cfg, base_agent_cfg):`
- 第二次 live provider smoke 暴露 LLM 仍会把 builder 当作 API 模块导入符号，并使用错误的 `TrainingComponents` 字段名 / 不存在的 `RewardInputs` 字段。prompt 进一步补充：
  - `build_training_components` 必须由候选在 `training_code/components.py` 中定义，不能从 `humanoid_agent.code_sandbox.components` 导入。
  - API 模块只提供 `RewardInputs`、`RewardTerm`、`LossTerm`、`ScheduleTerm`、`DiagnosticTerm`、`TrainingComponents`。
  - `RewardInputs` 字段列表是完整白名单，不存在 `inputs.action`；动作历史字段为 `last_action`。
  - `TrainingComponents` 字段名固定为 `reward_terms`、`loss_terms`、`schedules`、`train_observation_transforms`、`diagnostics`、`metadata`、`rationale`。
- full-round 在 provider/8D gate 失败且 `append_ledger=true` 时，追加一条 public-only 训练前失败 ledger 记录：
  - 只包含 candidate public projection、verifier public projection、失败阶段与公开 reason。
  - 不包含 hidden eval、eval seed、runner 私有状态或 checkpoint/eval 控制权。
- next-round prompt history 增加 public verifier projection，使 Agent 能根据公开 issue code 修正候选包，而不是盲目重采样。

## 验收

新增/更新的回归覆盖：

- prompt 必须声明真实三参数 builder 签名。
- prompt history 必须包含 public verifier status，同时继续隐藏 public_accept 细节。
- provider/8D 训练前失败不得启动训练，但必须能写入 public-only ledger feedback。
- 8C component-load Landlock 在 Linux 上必须包含 `/dev/urandom`。

本地轻量验收已覆盖 Phase 8E evolution/full-round/provider/linked/trusted-eval/training-adapter/contract/safety/selftest。
