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
- **该限额在本 Phase 已被真实违反，并非仅是未来风险。** `_bounded_history_entries()` 用 `_serialized_size()` 即 `canonical_prompt_json_bytes`（`json.dumps(indent=2)`）执行该 8192 B；以同一序列化器度量 validation 第 3 代两个 lane 的原始 payload，`public_history` 实际为 **10,113 B**（其中 `experiment_memory` 6,395 B、`recent_outcomes` 2,763 B），已超过其自身声明的 `limits.max_serialized_bytes=8192`。第 1/2 代分别为 1,538 B 与 6,140 B，尚未越界。
- 按现有代码复算 16 轮边界：保留 16 条 experiment memory、5 条 recent outcome、遗漏 11 条 recent outcome，`public_history` 增至约 15.8 KB，声明值仍为 8,192 B。系统不会失控——`recent_outcomes` 的 8192 B 与 `experiment_memory` 的 14,336 B 各自独立生效，完整 prompt 另有 256 KiB 硬上限；问题是该字段挂在整个 `public_history` 上，却只约束其中的 `recent_outcomes`，对读取它的 LLM 而言是一个已经不成立的对外契约。（原记录的 7,330 B / 4,725 B 系用紧凑 `json.dumps` 度量，与该限额实际执行所用的 canonical 序列化器不一致，已按 canonical 值更正。）

**建议**

- 低复杂度修正优先：把该字段改名为 `max_recent_outcomes_serialized_bytes`，并沿用 `experiment_memory.limits.max_serialized_bytes` 表达另一部分边界；无需再增加一层压缩器。
- 验收应证明字段名与实际 enforcement 对齐，并用 16 轮构造验证 recent outcomes、experiment memory 和完整 provider payload 三层现有上限均独立生效。

## F4：训练前 component smoke 不施加 `clip_min/clip_max` 与 `weight`，无效 reward term 只能在整轮训练后被发现

> 与 F2 同源于第 1 代 lane-a 事件：F2 处理“事后是否被记住”，F4 处理“事前能否更早发现”。两者独立，可分别实施。

**问题与证据**

- `candidates/component_smoke_runtime.py::_smoke_reward_term()`（第 407–463 行）只对 `term.fn(fixture.reward_inputs)`
  的原始返回值检查三件事：是否抛异常、形状是否等于 `fixture.expected_reward_output_shape`、是否有限。它
  **不施加 `term.clip_min`/`term.clip_max`，也不乘 `term.weight`**，因此不会观察该 term 在自身声明下的实际量级。
- 训练期的语义与此不同：`tasks/stair_climbing/reward_program.py` 第 617–619 行先 `clamp` 再乘权重，而
  `max_abs_weighted_contribution`（第 501 行，取自 `max_abs_weighted_rate`）度量的是**裁剪并加权之后**的量。
  于是“函数返回负值 + 声明 `clip_min=0`”会使该 term 在训练中恒为 0，而预检查看到的只是一个正常有限的负数。
- 第 1 代 lane-a `stair_clearance_aware_v1` 即为此形态：新增 `clearance_penalty` 返回负值并声明 `clip_min=0`，
  训练回执记录 `max_abs_weighted_contribution=0.0`（调用与采样计数正常），checkpoint 与 official reference 逐字节相同。
  该 checkpoint 相同性是确定性训练（`isaaclab_execution.py` 第 697–701 行的冻结 cudnn 设置）在有效奖励未变时的
  预期结果，不是接线失效。
- 判定本身正确：`tasks/stair_climbing/public_projection.py` 第 886–892 行的 runtime-effect 门要求
  `max_abs_weighted_contribution > 1.0e-12`，该 lane 在 public-dev 筛选阶段即被 `public_dev_runtime_activation_filter`
  过滤，未进入 public-accept 或 sealed，`selection.py` 亦有一等的 `no_runtime_activation_eligible_candidate` 语义。
  问题只在于**发现时机**：该判定发生在 `196,608,000` steps 训练完成之后。
- 代价：validation 共 7 个 train job、`1,376,256,000` steps、`allocated_gpu_capacity_hours_charged=8.537715`；
  该 lane 单独消耗约一个完整训练 job（wall 约 `63:45`，≈1 GPU-hour）及其 public-dev seed units，最终未产出可选候选。
- 所需材料均已存在，不需要新组件：`candidates/components.py` 第 40–43 行的 `RewardTerm.weight/clip_min/clip_max`、
  `tasks/component_smoke.py` 第 68 行已实现的 `SmokeTensor.clamp()`，以及与训练后完全相同的 `> 1.0e-12` 判据。

**建议**

- 在 `_smoke_reward_term()` 内复用训练期顺序：`fn → clamp(clip_min, clip_max) → × weight`，取结果的最大绝对值，
  并用与 runtime-effect 门相同的 `1.0e-12` 判据。不新增门限常量、不改变已冻结的 runtime-effect 语义。
- 结果必须作为**可修复的 authoring 反馈**（具名 reason，例如 `reward_term_inert_under_declared_clip_and_weight`），
  而不是硬性拒收：smoke fixture 是合成输入，“在 fixture 上恒零”不等于“在真实状态分布上恒零”。最终裁决权仍归
  训练后的 runtime-effect 门。该反馈应走现有 author→repair 回路，使 LLM 在本代内改正，而不新增第三套校验框架。
- 不改动 P4 已冻结的训练、评测、验收与候选权限语义；不因此调整 `1.0e-12`、guard 阈值或题池。
- 验收：以第 1 代 lane-a 的候选源码回放，smoke 在任何 train job 之前以新具名 reason 返回可修复反馈；
  一个仅在 fixture 上为零、但声明合法的 term 仍可进入训练（证明该项为 advisory 而非拒收）；
  legacy Stair 与 G1-29 Push 的现有 smoke 结果不变。

## F5：任务 prompt 容量 preflight 不是上界，可达的溢出会在会话中途硬失败

**问题与证据**

- `memory/context_builder.py::preflight_task_prompt_capacity()` 由 `agent_core/live_loop.py` 第 172 行在任何
  invocation 之前调用，是已接线的启动闸门。以当前冻结 unified 配置实跑该闸门，自报
  `serialized_bytes=30798`、`headroom_bytes=1970`（压缩阶梯自身目标为 `32768-512=32256`）。
- 用同一个产品函数 `build_provider_task_prompt_receipt()` 重新度量 validation 的 11 份原始 prompt payload，
  第 3 代三份实测均为 `31433 B`。**真实运行比启动闸门自报的最坏情况还大 635 B**，闸门不是上界。
- `_minimum_core_stair_prompt_sections()`（`tasks/stair_climbing/prompt_adapter.py` 第 1719–1723 行）是压缩阶梯
  的最后一级，超出 `target_bytes` 时 `raise ValueError`，没有更低一级可降。
- 低估有两个可复现的成因：
  1. `_stair_unified_capacity_preflight_feedback()` 把 `failure_priority` 的全部 7 个原因都填成 `rate=1.0`；
     `_compact_unified_stratum_diagnostics()` 按 `(-rate, name)` 排序取 top-2，**全部相等时每个 stratum 都保留
     同样两个字母序最前的名字**，列并集由 7 塌缩为 2。preflight 枚举 7 个原因以建模最坏情况的意图，被它自己
     喂入的截断抵消。实测：preflight 表 2500 B（2 列）对忠实最坏 3600 B（7 列），每 entry 少算 1100 B。
  2. `maximum_number = 1.0` 序列化为 3 字符，而表内数值按 6 位小数舍入后为 8 字符（如 `0.037037`）。
- 忠实最坏情况为 `34348 B`，超出阶梯目标 `2092 B`。溢出**用当前已冻结的 G1 协议即可达到**，不需要新机器人；
  触发条件是本次 validation 中各自真实出现过的两件事的组合（以服务器产品序列化器复算）：

  | 场景 | 字节 | 相对 32256 |
  | --- | ---: | ---: |
  | 第 3 代原样（3 个失败列） | 31433 | +823 |
  | 仅把恰好为 0.0/1.0 的格子换成合法 6 位小数 | 31988 | +268 |
  | 仅增至 4 个失败列（第 1 代 reference_anchor 表真实为 4 列） | 31973 | +283 |
  | **4 列且非整数值** | **32528** | **−272，raise** |
  | 全部 7 个失败原因成列 | 33641 | −1385，raise |

- 测试盲区：唯一的 preflight 容量断言在 `tests/unit/test_prompt_capacity_projection.py` 第 541 行，使用 legacy
  `stair_protocol_v2.json`（不是 Phase 4 实跑的 unified 协议），且断言对象是 `headroom_bytes` ——即 preflight
  自报的那个数字本身，因此原理上无法发现低估。

**建议**

- 让 preflight 忠实建模最坏情况：给每个 stratum 构造**互不相同**的 top-2 失败原因，使列并集达到
  `failure_priority` 全集；数值改用该字段实际能出现的最长合法序列化形式（6 位小数），而非 `1.0`。
- 增加一条把 preflight 与真实 payload 绑定的回归：对 unified 协议断言
  `preflight_bytes >= 任一真实 minimum_core payload 的 bytes`，用本 Phase 的 11 份原始 payload 作为夹具，
  使“闸门低于实际”这一类缺陷不能再次通过。
- 为 minimum-core 溢出提供一条有序的可诊断降级（例如先降为单 entry 并显式记录省略计数），使最后一级不再以
  `ValueError` 中断已经支付了训练成本的会话；不改变已冻结的 32 KiB 上限与 512 B 保留量。
- 与 F7 合并实施更省：去掉 top-2 截断同时消除本项成因 1，且平均反而减少字节。

## F6：淘汰了半数候选的 public-dev selector guard 门对 LLM 完全不可见，且其判据无法由 prompt 反推

**问题与证据**

- selector 规则（`tasks/stair_climbing/selection.py` 第 665–671、750、764–783 行）为：对每个 public-dev seed 取
  `min(j_ood, c0_id, c1_id)`，跨 seed 求均值得到 anchor，与 parent anchor 相减，与
  `min(guard.max_regression)=0.05` 的**硬上限**比较；`guard_rule` 自述为
  `parent_relative_hard_regression_cap_only`，且 `paired_statistics.status=not_applicable`、
  `acceptance_headroom_role=not_evaluated_by_selector`。
- 该门决定了三代中两代的结局：第 2 代 lane-a 出局（`0.058547`）、第 3 代整代 `no_selection`
  （`0.078281`、`0.066065`）；六个候选中三个由它淘汰。
- 由原始 `public_dev.json` 的 per-seed `named_guard_scores` 重算 anchor 与 regression，六条全部与
  `selection.json` 的 `guard_regression_by_trial` 逐位吻合，规则复原无误。
- **prompt 给出的量算不出该判据**：prompt 只提供三条 guard 各自的均值，LLM 至多能算 `min(各guard均值)`，
  而 selector 用的是 `mean(每 seed 的 min)`。二者在真实数据上的差为
  `g1 lane-a 0.03479761`、`g2 lane-b 0.00543058`、`g3 lane-b 0.00365495`；由 Jensen 不等式该差**方向恒为乐观**，
  即 LLM 只会低估自身退化。上限仅 0.05，g2 那次的差已占整个预算的 11%。
- 规则本身在 prompt 中零出现：对 11 份原始 payload、792,328 B 扫描，
  `guard_regression`、`max_regression`、`parent_relative`、`hard_regression`、`minimum_of_named_guard`、
  `sibling`、`not_selected` 命中均为 **0**。唯一相关的一句 `selection_guidance` 描述的是 acceptance 的
  *paired* 容忍度，指向的并非这道门。
- 多轮记忆中该失败的原因是同义反复：`public_history.recent_outcomes` 对第 2 代记录
  `reason="runner_did_not_accept_candidate"`，等价于 `accepted:false` 的复述。
- 对照旧系统：legacy 的 `_generation_trial_comparisons()`（同文件第 918–939 行）确实向 prompt 发出
  `guard_delta`、`guard_max_regression`、`guard_met`、`reason_codes`。unified 改为 public-dev-only 是**正确**的
  收紧（legacy 读的是 `public_accept`），但没有补回这些量的 public-dev 等价物。
- 所需数据已在手：selection receipt 已持久化 `guard_regression_by_trial` 与 `reason`，而
  `_generation_public_dev_trial_summaries()`（第 945–1007 行）已经在读同一个 `generation["selection"]` 映射
  取 `selected_trial_id`。这是接线遗漏，不是新计算。

**建议**

- 在 `latest_generation_trial_comparisons` 的每个条目上补三个 public-dev 派生字段：selector anchor
  （`mean(每 seed 的 min)`）、其 parent-relative regression、以及该 trial 的 selector reason code；并在
  `public_eval_contract` 中声明 selector 的聚合方式与硬上限 `0.05`。这些全部由 public-dev 推导且为协议常数，
  不引入任何 public-accept 或 sealed 内容。
- 把 `recent_outcomes` 的失败原因替换为具体判据（例如 `public_dev_guard_filter` 及其 regression 值），
  不再写 `runner_did_not_accept_candidate` 这类同义反复。
- 严格保持边界：acceptance 阶段的判决细节（例如第 2 代 `c1_id` 的 `−0.09373`）由 public_accept 推导，
  **继续不得进入 prompt**；本项只补 public-dev 侧已有的量。
- 验收：以本 Phase 三代真实数据回放，第 3 代 prompt 能让 LLM 复现出与 receipt 一致的 selector regression；
  同时公共/私有边界扫描（`public_accept.<cell>`、`sealed_validation.<cell>`、私有 cohort hash）命中仍为 0。

## F7：`top_2_per_stratum` 截断是净负收益，且会使非零失败原因整列消失

**问题与证据**

- 列集合是**被保留**名字的并集（`prompt_adapter.py` 第 2091、2110–2118 行）：一个在每个 stratum 都排第 3 的
  原因不会获得列，因此既不是 `0.0` 也不是 `null`，而是完全不存在；而表头仍声明
  `truncated_rates_are_null: true`，对该情形是错的。
- 以产品函数喂入六个 trial 的真实 per-stratum `primary_failure_breakdown`：第 3 代 lane-a 的 `joint_ood`
  真实 `off_course=0.074074`，在表中**整列消失**。（该表因第 3 代为末代而未进入任何 prompt；本次实际下发的
  4 张表均完整。缺陷在真实评测数据上可复现，本轮未造成实际误导。）
- 字节收益为负：对六张真实表比较“截断版”与“完整版并删去两个说明字段”，**平均少 65 B**（−77/−85/−89/−89/
  −89，仅一例 +38）。即去掉截断反而更小，同时信息完整。
- 不提供最坏情况上界：构造 8 个 stratum 各保留不同 top-2，列数仍可达 `failure_priority` 全部 7 个，
  故它并未约束表宽，却稳定地损失真值。
- `66e9c3b`（schema v2）修复的是“有列但被截断 → 写 `null`”，未覆盖“全局被截断 → 无列”这一支。
- 该截断同时是 F5 成因 1：preflight 的等值输入经它塌缩，使最坏情况被少算 1100 B/entry。

**建议**

- 直接取消 `top_2_per_stratum`，输出全部非零失败原因，并移除 `primary_failure_projection` 与
  `truncated_rates_are_null` 两个说明字段；同步 schema 版本。此改动平均减少字节、恢复真值、并消除 F5 的一半成因。
- 若为控制表宽仍需上界，应改为对**列并集**设上限（而非每行 top-k），并对被省略的列给出显式具名计数，
  使“存在但未展示”可被读者识别。
- 验收：以本 Phase 六个 trial 的真实 breakdown 回放，表中非零失败原因集合与真值集合相等；
  minimum-core 字节不高于当前实现。

## Phase 6 前收口结论（2026-08-30）

本轮按当前 G1-V2 / Go2-V2 主线重新核实了 F1–F7。七项问题均真实存在，但 F4、F5、F6 的原建议需要收窄；
修复只涉及提示词投影、公开探索记忆与候选 authoring 诊断，没有改动训练分布、课程、seed、PPO、评测套卷、验收阈值或
acceptance 权限。Stair 特殊语义仍由 Stair adapter 解释，共享层只传递不透明的公开事实。

| 项目 | 结论 | 源头收口 |
| --- | --- | --- |
| F1 | 需要解决，已关闭 | provider payload 的稳定共享段统一前置，lane/attempt/repair 等动态段后置；`public_context` 兼容别名也进入同一份唯一顺序表，不再形成第二条尾部拼接路径。schema 升至 v4。真实 API prefix-cache 收益只能在 Phase 6 调用中观测，不能伪装成本轮正确性证据。 |
| F2 | 需要解决，已关闭 | generation record 将 selection receipt 中的公开 trial screen 作为不透明事实挂回每个 trial；Stair adapter 再把“未入选但 runtime effect 不合格”的 sibling 投影为负向探索记忆。没有新增第二套 memory，也没有让共享层理解 Stair 判据。 |
| F3 | 需要解决，已关闭 | `public_history.limits.max_serialized_bytes` 改名为 `max_recent_outcomes_serialized_bytes`，明确它只约束 recent outcomes；同步提升 history / generation projection schema，避免继续误导为整个 memory 的总预算。 |
| F4 | 问题真实，但原阈值方案过宽；已按更窄规则关闭 | smoke 同时计算 raw 输出与声明 clip/weight 后的 effective 输出。仅当 raw 非零而 clip/weight 将其精确抹为零时给 advisory；raw 本来为零不误报，也未把任意 `1e-12` 阈值写入共享层。advisory 最多获得一次已有 authoring retry，第二次仍 advisory 时进入训练，由 task-owned runtime-effect gate 作最终裁决。 |
| F5 | 需要解决，已关闭 | preflight 改用合法的长非整数统计量和完整失败列，忠实覆盖当前协议的高信息量形状；minimum-core 最后一级改为单条 latest-evaluated entry，并记录省略数量与 projection receipt，不再在已支付训练成本后抛 `ValueError`。未采用绑定 11 份开发期 payload 的长期夹具，以免把历史产物变成第二真相。 |
| F6 | 需要解决，已关闭 | 公开协议现在明确 selector 的 `mean_across_seeds_of_minimum_named_guard_per_seed` 聚合和 hard cap；trial comparison 提供 objective delta、guard floor/regression/cap、eligibility/status、selector reason、runtime-effect observed/reason。没有投影 public-accept、sealed 或 acceptance 私有判决。没有把 formal rejection 的通用原因强行改名为 selector failure；具体 selector 原因只在确有该公开 screen 时进入探索记忆。 |
| F7 | 需要解决，已关闭 | 删除 per-stratum top-2、`truncated_rates_are_null` 和虚假的截断说明，输出所有非零且经任务词汇表校验的失败列；schema 升至 v3。该删除同时消除了 F5 的等值输入塌缩成因。 |

### 容量与回归证据

- G1-V2：`28,983 B / 32,768 B`，余量 `3,785 B`；Go2-V2：`28,337 B / 32,768 B`，余量 `4,431 B`。
- 两行均为 `minimum_core_single_entry`：保留 1 条 latest-evaluated feedback，显式记录省略 1 条，receipt 为
  `stair_climbing.minimum_core_single_entry.v1`。
- 本地全量：`1569 tests`，`29 skipped`，`0 failed`，261.961 s。
- 服务器真实 Torch/Isaac 环境全量：`1569 tests`，`3 skipped`，`0 failed`，970.402 s。
- Phase 5 强验收器 `verify_phase5.py --require-go` 在本地与服务器均为 `status=pass`；G1-V2 与 Go2-V2 的
  `cell_loop_ready` / GO 身份链未改变。服务器 `test_stair_phase5_contract` 为 10/10 通过。
- 代码验收提交：`bc3965667af776f9b23c31856c10c5125fe0b5fe`；当时 local / origin / server 三端一致，服务器 tracked tree 为空。

### 相邻问题复审与边界结论

- 已扫描旧 prompt/history/feedback/schema 标识及废弃 `reward_term_zero` 语义，未发现残留生产入口。
- prompt 顶层顺序只有一份权威常量；容量降级也只有一条 task-owned 有序路径，没有再引入旁路拼装器。
- 未新增公开记忆存储、acceptance 判据或评测解释层；公开 trial screen 在共享层保持 opaque，机器人/任务特殊解释没有进入公用层。
- P5 契约、manifest、reference、训练与评测身份均由原验收器复算通过；本轮没有运行训练、Isaac 评测或 LLM/API 请求。
- 本轮删除的是会丢真值的 top-2 投影和其伪元数据；没有保留仅用于开发期历史 payload 的额外夹具或兼容分支。
- 同步审计发现服务器仅有两条已损坏的 `refs/codex/turn-diffs/...` 开发期辅助引用持有缺失对象；当前分支、origin 与
  `HEAD^{tree}` 均不受影响。精确删除这两条不可恢复的辅助 ref 后，`git fsck --connectivity-only` 通过；随后清除经
  路径核实的旧 `.git/gc.log`，自动 repack 正常结束且未再生成错误日志。未删除任何产品分支、remote 或 reflog。

因此，F1–F7 在进入 Phase 6 前均已按当前主线完成必要收口；剩余的 API prefix-cache 命中率属于 Phase 6 的运行观测量，
不是新的入口阻塞门。

## Phase 6 入口再审计收口（2026-08-30）

进入 Phase 6 前的独立回放发现，上一节 F6 增加的 fail-closed 校验把合法的 `selection=null` 误当成 selector 合同损坏：
`paired_authoring_incomplete`、`candidate_execution_incomplete` 或缺失 selection 的一代，会使下一代 prompt 在寻找最近完整 sibling
比较时抛出 `Stair public_dev selector reason is invalid`。这属于真实阻断缺陷，且发生在已支付 authoring/训练成本之后。

本次从语义源头收口：

- `selection=null` / 缺失 selection 明确表示“本代没有 selector 反馈”，投影返回空并继续查找更早的完整代；非对象 selection 仍 fail-closed。
- selector 的 guard/runtime eligibility 判据收敛到 Stair `selection.py` 的单一分类函数，生产 selector 与 prompt adapter 共用；adapter 只交叉核对 receipt 的 eligible trial 集，不再重写一份边界规则。
- unified trial comparison 的 18 字段集合成为 task-owned 唯一合同，真实投影与 preflight stress 都经过同一校验；字段漂移不再依赖历史 payload 夹具发现。
- `exploration_directive` 移至 lane 专属 `required_candidate_package` 之前，完成 F1 遗漏的稳定前缀收敛。

容量数字同时更正：`28,983 B` / `28,337 B` 是 `minimum_core_single_entry` 降级后的值；未降级 stress 分别为
`36,590 B` / `35,944 B`。它们证明有界降级可工作，不证明双 feedback entry 能完整装入。当前真实 Phase 6 payload 尚未产生，
不得用 Phase 4 的旧 6 字段 comparison 产物替代；待观察量已单独移入
`P6/PHASE6_FOLLOWUP_OBSERVATIONS_ZH.md`。

`10 N` 顶层支撑力阈值本轮不改：当前没有误判证据，RobotProfile 也没有权威质量事实；直接按质量缩放会改变冻结评测身份并制造
新的二源真相。该项只作为 Phase 6 行为观测，不是入口阻断。

本次验收证据：

- 本地全量 `1571 tests`，`29 skipped`，`0 failed`，257.157 s；新增 selectionless、selector 边界、真实/preflight 字段集合和稳定前缀回归均通过。
- 服务器真实 Torch/Isaac 环境四条关键回归 4/4 通过；全量 `1571 tests`，`3 skipped`，`0 failed`，退出码 0，972.026 s。
- 服务器 Phase 5 强验收器仍为 `status=pass`，G1-V2 / Go2-V2 均为 `go=true`、`cell_loop_ready=true`。
- 产品修复提交为 `27cb9a96d03a9b96a07f2c668c26581a08e6f883`；本轮未运行训练、Isaac 评测或 LLM/API 请求。
