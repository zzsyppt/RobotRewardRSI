# 2026-07-02 Phase 13 Memory/Context 最小接口

## 本阶段目标

按计划完成 Phase 13：建立最小 Memory/Context 接口，让下一轮 LLM prompt 只来自 public projection，并把 private artifact 与 prompt payload artifact 明确分开。

## 实现内容

- 新增 `ExperimentRecord`、`PrivateArtifactRef`、`PromptPayloadRef`：
  - `ExperimentRecord` 保存 public ledger projection。
  - `PrivateArtifactRef` 只在 private projection 中暴露 URI/checksum/reason。
  - public projection 只记录 non-prompt artifact redaction，不暴露 private artifact URI 或 trusted eval 类型名。
- 新增 `PublicContextBuilder`：
  - 从 `ExperimentRecord` 或 public mapping 构造 next-round public context。
  - 可选接入 `TaskMemoryAdapter`。
  - 生成并保存 public-only prompt payload。
- 新增 `TaskMemoryAdapter` protocol，作为未来 long-term memory 和任务特定 public summary 的扩展点，不要求 AgentCore 主流程改动。
- 新增 `PushTaskMemoryAdapter`：
  - 只压缩 public push eval metrics。
  - 输出 success_rate、latest failure mode、failure breakdown、motion quality。
- 扩展 public context sanitizer 和 prompt payload guard：
  - 继续屏蔽 private/hidden/secret/api_key/token。
  - 新增屏蔽 `trusted_eval`、`trusted_seed`、`checkpoint_selector`。
- 更新 stair extension guide，说明未来 stair 任务可以提供 task-specific public summary adapter，但不能暴露 hidden eval、private seed、checkpoint selector 或 runner-owned terrain internals。

## Review 结果

- 确认 next-round prompt payload 从 public projection 构造，不读取 private artifact projection。
- 确认 hidden eval、private seed、trusted eval/seed、checkpoint selector 和 API key 风险字符串不会进入 prompt payload。
- 确认 push summary adapter 只消费 public eval metrics，不接触 trusted eval 或私有 seed。
- 确认 future task memory extension 通过 `TaskMemoryAdapter` 注入，不需要改 AgentCore loop。
- 额外补强 push adapter，对 malformed public task field 不抛 AttributeError。

## 验证

- `python -m unittest tests.unit.test_memory_context_phase13 -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 13 已完成。当前 Memory/Context 层提供 public-only next prompt 构造、任务特定 public summary 扩展点和 prompt payload 保存路径，同时保持 hidden/trusted/private evidence 不进入 LLM prompt。
