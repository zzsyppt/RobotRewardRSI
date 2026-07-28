# Phase5: Component Smoke 与 Anti-Degenerate Authoring Check

## 目标

Phase5 的目标是在候选进入真实 runner 之前，增加轻量、公开安全、可同轮 repair 的静态 authoring 检查：

1. 对 `training_code/components.py` 做组件 smoke：确认组件可加载，RewardTerm callable 可在任务提供的 RewardInputs fixture 上执行，输出形状符合任务期望且数值有限。
2. 对照 prompt 内 current-best source 做轻量 anti-degenerate 判断：标记 identical / formatting-only / parameter-only / rationale-only / mechanism-delta-present。
3. 失败反馈进入同一 round 的 static repair prompt；真实 train/eval 仍只发生一次，runtime failure 仍只进入下一 round。
4. 信息足量但不爆炸：只携带最新失败源码，受 source catalog 与 authoring frame budget 管控。

## 设计边界

- Shared authoring 层不硬编码 push recovery 字段、shape、failure 语义。
- 任务特殊 RewardInputs fixture 由 `TaskPlugin.component_smoke_adapter()` 提供；默认任务使用 neutral/no-op adapter。
- Push recovery 的 smoke fixture 位于 task-owned 模块，只在 `PushRecoveryTaskPlugin` 中暴露。
- Component smoke 和 anti-degenerate 不是最终 policy accept/reject 逻辑；它们只处理 runner 前的明显静态问题。
- `parameter_only_delta` 只作为 warning，不阻止候选进入 runner；最终是否可接受仍由既有 surface policy / decision 层处理。
- `identical_or_format_only` 是 blocking authoring failure，需要 rewrite。
- 所有 repair feedback、attempt record、ledger record 必须 public-safe，不暴露本地路径、server path、secret、candidate path 或 previous_components_py。

## 实施结果

新增 task-owned smoke 合同：

- `src/true_rl_agent/tasks/component_smoke.py`
- `ComponentSmokeFixture`
- `ComponentSmokeAdapter`
- `NeutralComponentSmokeAdapter`
- `SmokeTensor`

新增 shared authoring checks：

- `src/true_rl_agent/candidates/authoring_checks.py`
- `run_component_smoke_check()`
- `assess_anti_degenerate_candidate()`
- `assess_anti_degenerate_source()`
- `build_component_smoke_repair_feedback()`
- `build_anti_degenerate_repair_feedback()`

新增 push recovery fixture：

- `src/true_rl_agent/tasks/push_recovery/component_smoke.py`
- `PushRecoveryTaskPlugin.component_smoke_adapter()`

接入 authoring/live flow：

- `CandidateAuthoringLoop` 顺序为 provider -> verifier -> component smoke -> anti-degenerate -> candidate_verified。
- smoke fail 或 identical-source blocking fail 会生成 static repair feedback，并把 latest failed source 交给 source catalog。
- max-attempt 仍失败时，`run_live_loop()` 写 `candidate_static_authoring_failed` ledger，不进入 runner。

## Blocking Review 修复: 可执行候选代码隔离

复审发现 Phase5 初版 component smoke 在 authoring 主进程内直接调用候选 `RewardTerm.fn`。如果候选 reward 写成 `while True: pass`，`StaticCandidateVerifier` 会通过结构校验，随后 smoke 会卡住 authoring 主流程。

修复后边界如下：

- `run_component_smoke_check()` 将 candidate load、RewardTerm callable、DiagnosticTerm callable 全部放进隔离 worker process。
- 父进程只等待硬 timeout；超时后 terminate/kill worker，并把结果转换为 public-safe `component_smoke_error`。
- reward hang 返回 `reward_callable_timeout`，diagnostic hang 返回 `diagnostic_callable_timeout`，factory/load hang 返回 `component_load_timeout`。
- `StaticCandidateVerifier` 也不再在主进程直接执行 `build_training_components()`；组件加载和组件摘要提取移入隔离 worker，factory hang 返回 `training_components_load_timeout`。
- worker feedback 只返回 public error category/type，不返回 traceback、本地路径、server path 或 raw candidate path。

同类调用链审计：

- `src/true_rl_agent/candidates/loader.py::load_training_components()` 仍是底层 loader；直接调用它的普通单元测试不是 authoring 主流程。
- authoring/verifier 路径已通过 worker timeout 包住。
- `src/true_rl_agent/tasks/push_recovery/runtime_adapter.py` 仍会在真实训练 runtime 中 load candidate 并调用 reward/diagnostic，但该路径位于 runner-owned train subprocess 内，由 `SubprocessRunner.timeout_s` 控制，不阻塞 authoring 主进程。
- Windows `spawn` 需要可导入的 main module；从 `<stdin>` 形式直接 probe 会安全失败为 `WorkerExit`，不会挂住。CLI/scripts 和 unittest 模块路径可导入，timeout 行为正常。

## 实际 Smoke 产物检查

本地 probe 生成了：

- `.tmp_phase5_smoke_probe/round-001/prompt_payload.json`
- `.tmp_phase5_smoke_probe/round-001/prompt_payload_attempt_002.json`

人工检查第二份 provider-facing payload：

- `authoring_context_frame.frame_kind = static_repair`
- `provider_or_runtime_repair_feedback.failure_category = component_smoke_error`
- `component_smoke_result.status = fail`
- `source_payloads` 只包含最新失败 `attempt/latest_failed/training_code/components.py`
- 失败源码中 `broken_reward` 可见，路径被 scrub 为 `[path]`
- 全文未发现 `D:/humanoid_agent`、`D:\humanoid_agent`、`/data/yuyangliu`、`sk-live-secret`、`candidate_path`、`previous_components_py`
- payload 文件约 26KB，`actual_prompt_payload_bytes = 21321`，低于 256KB hard limit
- latest failed source payload 为 389 bytes，没有历史源码累积

Anti-degenerate probe 另行构造了已接受 current-best source，并让下一候选提交相同 source：

- 第一次 assessment 为 `identical_or_format_only`，`blocking = true`
- 第二次 provider-facing payload 为 `static_repair`
- `provider_or_runtime_repair_feedback.failure_category = anti_degenerate_warning`
- `source_payloads` 同时包含 `current_best/training_code/components.py` 和 `attempt/latest_failed/training_code/components.py`
- current-best 与 latest-failed source hash 匹配，provider 不需要文件系统访问
- 全文未发现本地路径、server path、secret、candidate_path 或 previous_components_py

## 验收

已覆盖并通过：

- component smoke pass/fail/public-safe/unsupported-output tests
- component smoke infinite-loop reward timeout test
- anti-degenerate identical / parameter-only / mechanism-delta tests
- verifier hanging `build_training_components()` timeout test
- authoring loop smoke fail -> static repair -> next candidate success
- live loop max smoke fail -> static authoring ledger -> no training
- existing verifier/provider/runtime/source boundary tests

验证命令：

- `python -m unittest tests.unit.test_component_smoke_and_antidegenerate tests.unit.test_candidate_authoring_loop tests.integration.test_live_loop`
- `python -m unittest discover -s tests`
- `python -m compileall src tests`
- `git diff --check`

## 后续注意

- `SmokeTensor` 是轻量 authoring fixture，不是 torch/runtime 替代品；未来如果候选常用更多 tensor API，应扩展 fixture 能力或把不支持输出转为 repair feedback，不能把真实 runner 放进 shared smoke。
- source follow-up 仍未启用，Phase5 不改变 Phase0-4 的 source request 边界。
- task-specific success 判定仍不进入 shared smoke/anti-degenerate 层。
