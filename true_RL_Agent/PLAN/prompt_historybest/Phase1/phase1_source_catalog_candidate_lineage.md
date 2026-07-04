# Phase1: Source Catalog 与 Current-Best Lineage

## 目标

把“已接受候选代码没有进入下一轮输入”的问题收敛为通用 prompt source 机制，而不是在 provider 或某个任务 prompt 里临时拼接代码文本。

Phase1 的目标是让下一轮 prompt 能看到：

- 当前候选组件 API 的公开合同入口。
- 最近一次已接受 candidate 的 `training_code/components.py` 逻辑源。
- known-good examples 的逻辑槽位。
- 最新失败尝试源码槽位的保留元数据。
- 当前 best candidate 的 lineage、hash、public score、changed components 与 source delivery 状态。

## 实际实现落点

- 新增 `true_rl_agent.prompting.source_catalog`，统一生成 `source_catalog`、`candidate_lineage`、`source_payloads`、`exploration_directive`。
- `PublicContextBuilder` 在 public history/task sections 之后接入 source context，再交给 canonical prompt assembler。
- `build_prompt_payload()` 将 source sections 作为顶层 canonical sections，且从 transitional `public_context` alias 中移除，避免 source content 重复出现。
- `authoring_context_frame` 记录 source section 引用、hash、payload 摘要，但不复制源码内容。
- `Round` ledger 写入 `private_artifacts`，用于后续 source resolver 找到 accepted candidate package。
- `ExperimentRecord.from_ledger_record()` 把 ledger 中的 candidate path 拆成 private artifact，同时从 public projection 移除 path。

## 边界

- Provider 接口不变：`CandidateProvider.next_candidate()` 仍只返回 candidate package，不新增 source request 能力。
- Provider 不允许访问文件系统；prompt 只暴露 runner 已解析、已 scrub、已 budget 的 source payload。
- Prompt 中不暴露真实 candidate path、本地路径、服务器路径、secret、raw seed、trusted/private 信息。
- Source resolver 只消费 `ExperimentRecord.private_artifacts`，不从 public `candidate.path` 直接解析源码。
- `api/candidate_components.py` 只作为逻辑 catalog item，不内联共享 `components.py` 原始源码；候选 API 以 `required_candidate_package.component_contract` 为准，避免把任务特殊字段带回共享语义层。
- 当前 best source 小于硬上限时内联；超过上限时不放源码，只保留 hash、bytes、公开 summary 与 rewrite directive。
- latest failed source slot 当前只保留 metadata，具体失败候选源码迁移留给后续 Phase；现存 repair feedback 的 `previous_components_py` 仍是旧路径，不属于 Phase1 current-best source catalog 的完成范围。
- `source_payloads.content_sha256` 表示 prompt-visible scrub 后内容的 hash，不承诺等于磁盘原始文件 hash；read-over-budget 且不内联的 current-best source 只暴露 raw source bytes 的 hash、bytes 与粗 summary，不暴露源码内容。

## 参考结论

- 旧 Phase8E 的 `latest_accepted_record` 与 source-pack 思路合理，但旧工程与 push 任务强耦合，新系统只保留“逻辑 source id + runner 解析 + prompt artifact”模式。
- Codex 的上下文窗口/compaction 思路适合用于记录 omitted metadata、hash 与 frame included sections，但不引入 Codex 的 tool/session 语义。
- 任务特殊语义继续停留在 task prompt adapter；共享 source catalog 只描述通用 lineage 和 source delivery。

## 验收

- 无 accepted record 时，prompt 有 closed allowlist source catalog，`candidate_lineage.status=none`，不内联 source。
- 有 accepted record 且 source 在 budget 内时，只内联最近 accepted candidate 的 `training_code/components.py`，并记录 source hash。
- accepted candidate path 只存在 private artifact，不进入 public record、payload、frame 或 provider prompt。
- 超预算 source 不进入 prompt content，仍保留 hash、bytes、公开 summary 与 rewrite directive。
- public candidate path 不能作为 source resolver 的直接输入。
- direct assembler 入口传入的 source sections 必须通过同源校验：payload logical id 必须在 catalog 中，hash/bytes/content 必须一致，lineage 引用必须指向 catalog/payload 中的同一 source。
- `task-owned` 等普通公开词不会被 `sk-` secret 检测误伤，真实 `sk-...` token 仍被拒绝或 redacted。

## 已覆盖测试

- `tests/unit/test_prompt_source_catalog.py`
- `tests/unit/test_prompt_projection_and_mechanism.py`
- `tests/unit/test_prompt_safety_boundaries.py`
- `tests/unit/test_memory_context_phase13.py`
- `tests/unit/test_task_prompt_adapter.py`
- `tests/unit/test_core_schema.py`
- `tests/unit/test_candidate_package.py`
- `tests/integration/test_fake_round.py`
- `tests/integration/test_live_loop.py`
