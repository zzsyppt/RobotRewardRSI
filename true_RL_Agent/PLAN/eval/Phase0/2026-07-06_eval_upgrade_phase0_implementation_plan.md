# Phase0 实施计划：EvaluationProtocol 与 invariant 收敛

## 目标

Phase0 只解决协议漂移问题，不实现 full eval rollout，也不把 push recovery 的任务指标塞进公共层。完成后，task config、TaskSpec invariant、backend config、EvalJobSpec、seal 和 materialize 输出都必须围绕同一个 canonical eval protocol 对齐。

## 参考边界

- 旧 Phase8E 的可借鉴点：runner-owned checkpoint 验收、候选代码隔离、public-only 投影、预算和运动质量 gate。
- Codex 的可借鉴点：通用 policy/protocol 对象、显式配置映射、规则自测，而不是把业务语义写进公共执行层。
- 新系统必须保留现有分层：task plugin 定义任务语义，backend/runner 物化和执行，公共层只描述通用 eval 协议和 gate 结构。

## Canonical 字段

Phase0 统一使用这些字段：

- `eval_protocol_version`
- `public_seed_count`
- `public_accept_seed_count`
- `sealed_validation_seed_count`
- `eval_num_envs`
- `eval_protocol_hash`

`trusted_seed_count` 只能作为迁移期边界输入存在，并且进入系统后立即转换成 `sealed_validation_seed_count`。协议 hash、TaskInvariantSpec、prompt/public projection、seal payload 和 materialize 输出不得继续把它当作 canonical 字段。

## 具体步骤

1. 新增公共 `EvaluationProtocol`。
   - 位置：`src/true_rl_agent/evaluation/protocol.py`
   - 只包含协议版本、split counts、eval env 数和稳定 hash。
   - 不包含 push recovery 的 fall、drift、height、tilt、push window 等任务语义。

2. 补齐 push recovery task config。
   - 在 `PushRecoveryTaskConfig` 中加入 `sealed_validation_seed_count` 和 `eval_protocol_version`。
   - 增加 helper 生成 `EvaluationProtocol`。

3. 收敛 `PushRecoveryTaskPlugin.task_spec`。
   - `TaskInvariantSpec` 增加 canonical eval fields 和 `eval_protocol_hash`。
   - `fixed_eval_protocol` 使用同一份 `EvaluationProtocol.to_public_dict()`。
   - 不在 task spec 中暴露 `trusted_seed_count`。

4. 收敛 backend materialization。
   - `IsaacLabRslRlBackendConfig` 保留 `trusted_seed_count` 作为 legacy 输入，但新增 `eval_protocol_version`。
   - backend 内部生成 `EvaluationProtocol`，并用它构造 `EvalJobSpec`。
   - invariant 校验要求 canonical eval fields 必须存在且匹配；缺失直接 fail。

5. 收敛 `EvalJobSpec` 和 seal。
   - `EvalJobSpec.to_public_dict()` 输出 `sealed_validation_seed_count`、`eval_protocol_version`、`eval_protocol_hash`。
   - seal hash 使用 canonical eval protocol payload，不再使用 `trusted_seed_count` 作为 hash 字段名。

6. 收敛入口脚本。
   - `run_loop.py` 新增 `--sealed-validation-seed-count`，兼容旧 `--trusted-seed-count` alias。
   - `materialize_job.py` 从 task config 读取 sealed validation count。

## 测试策略

- `EvaluationProtocol` hash 对 count、env 和 version 变化敏感，对 candidate manifest/provider JSON 不敏感。
- task spec 同时包含 fixed protocol 和 invariant protocol 字段。
- backend/task 的 `sealed_validation_seed_count` 不一致会在 materialize 阶段失败。
- task spec 缺失 `sealed_validation_seed_count` 会在 materialize 阶段失败。
- materialize 输出只使用 canonical `sealed_validation_seed_count`。
- prompt-facing payload 继续不出现 `trusted_seed`。

## Phase0 不做

- 不移除 eval CLI 的 candidate path；这是 Phase1。
- 不实现 byte-level checkpoint safe-load；这是 Phase2。
- 不实现 sealed holdout rollout；这是 Phase3。
- 不迁移 current-best stats；这是 Phase5。
- 不把 push recovery 的任务指标写进 `EvaluationProtocol`。
