# Phase 4 后续可提升项

> 本文件只记录不阻断当前冻结实验、但有明确收益与证据的改进项。处理时不得改变 P4 已冻结的训练、评测、验收或候选权限语义。

## F1：调整 provider payload 顺序，提高同代双 lane 的前缀缓存复用

**问题与证据**

- `src/true_rl_agent/prompting/payload.py` 的 `PROMPT_PAYLOAD_SECTION_ORDER` 将 lane 相关的 `required_candidate_package` 与逐请求变化的 `authoring_context_frame` 放在 `source_catalog`、`candidate_lineage`、`source_payloads`、`public_feedback_summary`、`public_history` 之前。对只复用共同前缀的 provider，这会让后半段大量相同内容无法命中缓存。
- validation 第 3 代两个 lane 的首次真实请求分别为 89,475 B 与 90,301 B；逐字节比较得到共同前缀 35,493 B、共同后缀 46,580 B。共同后缀包含相同的当前 best/最近失败源码、两轮反馈与记忆，却位于首个 lane 差异之后。
- lane-b 的真实 provider usage 为 `request_units=23962`、`request_cache_hit_units=9216`、`request_cache_miss_units=14746`；命中规模与 35,493 B 的共同前缀相符，而 46,580 B 共同后缀没有形成可复用前缀。第 1/2/3 代 lane-b 的 cache hit 均约 9.2k units，也与固定首段边界一致。

**建议**

- 在下一次 prompt schema 升级时，仅重排顶层 section：先放同代共享的稳定契约、source、feedback 与 memory，再放 lane/attempt 相关的 candidate package、frame 与 repair feedback。不要删除契约、合并权限语义或改变 section 内容。
- 用同一代 A/B 配对请求验收：provider-visible section 内容集合保持一致，`request_cache_hit_units` 上升且 `request_cache_miss_units` 下降；公共/私有边界与候选权限测试保持不变。

## F2：把未选中 sibling 的公共失败原因纳入现有有界探索记忆

**问题与证据**

- `StairExplorationProjector._assess_unified_trial()` 只按 public-dev objective/guard 差值分类，不消费已有的 `public_dev.runtime_effect_evidence`；`_generation_public_dev_trial_summaries()` 又只输出 identity、objective score 与 named guard scores，丢弃已计算的 `observed_effect`、`reason_codes`、parent-relative delta 和 `trial_status`。
- 第 1 代 lane-a `stair_clearance_aware_v1` 的正式 selection receipt 明确给出 `runtime_effect_evidence.status=fail`、`reason=candidate_reward_operation_runtime_effect_not_observed`；训练回执中新增 `clearance_penalty` 的 `max_abs_weighted_contribution=0.0`，且 checkpoint 与 reference 相同。
- 第 2 代 prompt 只把该 sibling 显示成一条绝对分数比较；到第 3 代，它已不在 `experiment_memory.experiments`，也没有进入 `verifier_equivalent_avoidance_cards`。第 3 代记忆只保留第 1 代被接受的 lane-b、第 2 代被正式拒绝的选中 lane-b，以及第 2 代 lane-a 的一张 avoidance card。因此后续代可以再次提出与这个“运行时无效”机制等价的候选而得不到直接提醒。

**建议**

- 不新增第三套记忆系统；在现有 task-owned exploration projection 中，为每代未选中 sibling 生成一张紧凑公共卡，至少保留 candidate/mechanism identity、`trial_status`、public runtime-effect status/reason、public-dev objective/guard delta、selected 标记。
- 运行时无效应优先成为 avoidance 语义；普通 selector loser 仍按现有有界排序/淘汰。严禁写入 public-accept、sealed-validation、私有 cell/seed 或正式判决细节。
- 以第 1 代样例回放验收：第 3 代仍能看到 `stair_clearance_aware_v1` 的 runtime-inactive 避免信息，同时总记忆字节上限与遗漏计数继续生效。

## F3：修正 `public_history` 的 8192 B 限额字段语义

**问题与证据**

- `agent_core/context.py::_bounded_history_entries()` 的 8192 B 实际只约束 `recent_outcomes` 列表；`build_public_history_section()` 随后把独立上限为 14,336 B 的 `experiment_memory` 放入同一个 `public_history`，但 section 仍声明 `limits.max_serialized_bytes=8192`。
- 第 3 代真实 payload 在已有两轮历史时，`public_history` 为 7,330 B，其中 `experiment_memory` 为 4,725 B。
- 使用这两轮真实条目按现有代码复算 16 轮边界：保留 16 条 experiment memory、5 条 recent outcome、遗漏 11 条 recent outcome；最终 `public_history` 为 15,816 B，却仍声明最大 8,192 B。系统不会失控，因为 experiment memory 自有 14,336 B 上限且完整 prompt 另有 256 KiB 硬上限；问题是对外契约字段声称了代码没有执行的范围。

**建议**

- 低复杂度修正优先：把该字段改名为 `max_recent_outcomes_serialized_bytes`，并沿用 `experiment_memory.limits.max_serialized_bytes` 表达另一部分边界；无需再增加一层压缩器。
- 验收应证明字段名与实际 enforcement 对齐，并用 16 轮构造验证 recent outcomes、experiment memory 和完整 provider payload 三层现有上限均独立生效。
