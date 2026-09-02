# Phase 8E Prompt 构造改进计划(prompt-evolve)

> 目标:**只改 prompt 构造**(`build_phase8e_prompt_package` 及其投影/source pack),不动评测、预算、隔离、候选 scope。
> 原则总纲:**prompt 只暴露"真实已接入"的能力;尚未实现/未实际接通的功能,代码里可以先留接口,但暂不进 prompt,等真正做好并有行为探针证明后再进。**
> 依据:对真实 round-3 prompt 的实测审计(`layer2_live_15round_20260628_200343/round_003/.../phase8e_prompt.txt`,2188 行 / 127 KB)。
> 日期:2026-06-29

---

## 0. 一句话结论

当前 prompt 是一坨**按字母排序的裸 JSON**,**40%+ 是重复内容**,真正的任务排在**最后一行**,还**向 LLM advertise 了一批它根本用不了的能力**(`push_active`/`push_vec_public` 零占位、4 个 ineffective schedule 键)。这些都让信噪比变差、token 翻倍、注意力被稀释。本计划把 prompt 收敛成"**只讲真的、讲一遍、先讲任务、把信号顶到台面**"。

---

## 1. 范围 / 非范围

**做(本计划):** prompt 构造质量——去重、结构/框架化、任务前置、反馈聚合、source pack 语义、"真实能力门"、路径卫生。纯 runner 侧、纯构造层。

**不做(另立计划 `schedule_push_signal`):**
- 不在本计划接通 push/DR 课程键。
- 不在本计划把 `push_active`/`push_vec_public` 变成真实信号。
- 本计划只负责:**在它们变真之前,把它们从 prompt 里拿掉**;等那份计划做完、有行为探针,再由"真实能力门"自动放它们进来。

---

## 2. 核心原则

1. **单一事实源、零重复**:每份契约在 prompt 里只出现一次;source pack 用 logical id 引用,不重嵌内容。
2. **真实能力门(programmatic gate)**:prompt 是否 advertise 一个字段/调度键,由"它是否真实 instrumented/connected"**程序化决定**,不靠手维护。
   - RewardInputs 字段:`provenance ∈ {observed, derived_from_observed, runner_owned}` 才进 prompt;`not_instrumented` 不进。
   - schedule 键:`effectiveness == effective_now` 才进 prompt 的"可用键";其余不进。
   - 好处:`schedule_push_signal` 把 push 变真后,门**自动放行**,无需改 prompt 代码;反之永远不会"advertise 一个用不了的东西"。
3. **任务前置且唯一**:目标只有一句、在最前(或 system 里),不再首尾各一句、措辞还不同。
4. **先框定再给料**:`目标 → 你能用什么(契约/范例/当前最佳)→ 上轮反馈 → 交付物`,而不是无序 sorted JSON。
5. **信号顶到台面**:反馈给聚合摘要,不让模型自己跨种子/跨轮汇总。
6. **不泄露 host 细节**:prompt 里不出现服务器绝对路径。
7. **构造层不破坏安全边界**:所有改动后,`assert_public_only_payload` 仍覆盖整 prompt;public-only / 评测端边界不受影响。

---

## 3. 实测证据(为什么要改)

| 现象 | 实测 |
|---|---|
| 体积 | 2188 行 / 127 KB(单轮、单次调用) |
| 契约三重复制 | `reward_inputs_contract` / `control_effectiveness` / `field_expansion` 各出现 **3 次**(`component_contract` + source_pack item 的 `content` 字符串 + `parsed` 对象) |
| source_pack 占比 | **917 行(42%)**,大部分重复那 600 行的 `required_candidate_package` |
| 任务位置 | `task.objective` 在**第 2178 行(倒数第一个键)**;另有一句不同措辞的 `objective` 在第 248 行 |
| 反馈冗余 | `primary_label_priority`(常量列表)在 seed_results 里重复 **14 次**;无跨种子聚合摘要 |
| 不可用能力被 advertise | `push_active`/`push_vec_public` = `not_instrumented` 零占位仍在 `field_order`/契约/expansion(出现 18 次);4 个 `accepted_but_ineffective` schedule 键仍在 checklist 与 control 契约 |
| push/反馈错配 | 主导失败 `fall_backward`(被推倒)反复出现,但候选**没有 push 信号杠杆**可用——prompt 如实暴露了这个困境 |
| source 机制冗余 | 完整 content 已推进 prompt,却又配 `source_access_policy` + source-read 拉取轮次(内容都给了,拉取永不触发) |
| 元数据/代码分离 | `current_recipe`/`known_good_reward_examples` 顶部只给 metadata,真代码在 900 行的 source_pack 里(`current_best/training_code/components.py` 第 1946 行) |
| host 路径 | `current_recipe.candidate_dir` / `entrypoint_path` = `/data/yuyangliu/runs/...` 直接进 prompt |

---

## 4. 改造项(分阶段)

### P0 — 真实能力门 + 移除"还没接入"的能力(信号/清晰度最大,且落实总纲原则)
- 新增 prompt-facing 投影:`prompt_facing_reward_inputs_contract()` 从完整 `reward_inputs_contract()` **过滤掉** `provenance == not_instrumented` 的字段(当前即 `push_active`/`push_vec_public`)。
- prompt 的 `reward_inputs_fields` / `field_order` / 自检清单的 `reward_inputs_field_order` 全部改用**过滤后**的字段集。
- `control_effectiveness` 投影到 prompt 时:`schedule_controls` 只列 `effective_keys`(当前只剩 `reward_weight_scale`);`accepted_but_ineffective_keys` **不进 prompt**(连同 `selection_guidance` 里"别依赖 ineffective 键"那句也随之移除——没 advertise 就不用警告)。
- **代码接口保持不变**:`RewardInputs` 仍有 17 字段、完整 `reward_inputs_contract()` 仍存在(供**隔离 fast 校验**构造合成输入用);`DEFAULT_SCHEDULE_ALLOWED_KEYS` 仍含 5 键(runner 仍预计算)。只是**不向 LLM advertise** 未生效的部分。
- 设计取舍:候选若仍引用被隐藏的真实字段(如 `inp.push_active`),不硬拒(它是真零字段,向前兼容),但 prompt 不教、不鼓励。
- **验收**:prompt 文本里 grep 不到 `not_instrumented`、`accepted_precomputed_not_applied`、`push_active`、`push_vec_public`;但 `phase8e_contract_selftest` 里**完整契约**仍含 17 字段、合成校验仍跑全字段。

### P1 — 去重(体积最大,预计砍 35–45%)
- source_pack item **不再同时存 `content` + `parsed`**:JSON 类只留一份(优先 `parsed`,或只留 `content`,二选一,全局统一)。
- **契约不在 source_pack 里重嵌**:`component_contract` 是唯一权威;source_pack 对这些 id 只放**轻量条目**(id + kind + sha256 + 一句 summary + `"see": "required_candidate_package.component_contract..."`),不放正文。
- `public_history` 里把每 seed 重复的 `primary_label_priority` 提取为**一份顶层常量**(或直接删,因为 `failure_breakdown` 已自带标签)。
- **验收**:三份契约 schema_version 在 prompt 里**各只出现 1 次**;prompt 行数/字节较基线下降 ≥ 35%(对同一 ledger 状态测)。

### P2 — 任务前置 + 轻框架化
- 合并 `objective`(248)与 `task.objective`(2178)为**唯一**一句,放在 payload 顶部(或写进 system message)。
- 给 payload 一个**稳定的呈现顺序**(不靠 `sort_keys`):`task → loop_contract → component_contract(含契约)→ known_good_examples → current_recipe → public_history(含聚合)→ checklist → 交付格式`。
  - 实现上:要么改 `messages_to_text` 用固定 key 顺序渲染,要么在 system 里加一段自然语言导航("先读 task,再读 component_contract,按 checklist 自检后只输出 candidate_package JSON")。
- **验收**:prompt 前 ~30 行内能读到完整任务目标 + 交付物要求;`task`/`objective` 不再重复。

### P3 — 反馈聚合
- `public_history` 顶部(或每轮条目顶部)加一行 runner 计算的**聚合摘要**:跨种子 `mean_success_rate`、`dominant_failure`(+ rate)、相对上一轮 accepted 的趋势(↑/↓/持平)。
- 保留 per-seed 明细(可折叠/可选),但聚合摘要必须在最前。
- **验收**:不跑模型也能从 prompt 顶部一眼看出"上轮均值 0.89、主导失败 fall_backward"。

### P4 — 代码就近 + 路径卫生
- `current_recipe` 与 `known_good_reward_examples`:要么直接带上**精简源码**(就近),要么明确"代码见 source_pack/<id>"并去掉重复 metadata;不要两头都给一半。
- 所有进 prompt 的路径改为**相对 lab/run-root** 或纯 logical id,去掉 `/data/yuyangliu/...` 绝对路径。
- **验收**:prompt 文本 grep 不到 `/data/yuyangliu`。

---

## 5. "未接入功能"的接口处理(落实总纲)

- **代码层**:接口/字段/键照常存在并被**隔离 fast 校验**与训练 adapter 使用(例如 `push_active` 仍是 RewardInputs 字段、仍被合成输入构造)。这样 `schedule_push_signal` 计划只需把"真实信号/课程"接上,无需改 prompt 构造。
- **prompt 层**:由 P0 的"真实能力门"自动决定可见性。判据来自**已有的、程序化的真值**:
  - 字段:`reward_inputs_contract` 的 `provenance`(`not_instrumented` ⇒ 隐藏)。
  - 调度键:`control_effectiveness_contract` 的 `effectiveness`(非 `effective_now` ⇒ 隐藏)。
- **变真时的切换成本 = 0 prompt 代码改动**:当 push 被 instrument(provenance 变 `observed`/`runner_owned`)、push 课程键被接通(effectiveness 变 `effective_now`),门自动放行,prompt 自动开始 advertise。**这正是"写好接口、做好了再进 prompt"的机器化保证。**

---

## 6. 测试 / 不回归

每个阶段都要:
1. `assert_public_only_payload` 仍覆盖整 prompt + 全部 source/反馈投影(P0–P4 不得绕过)。
2. 现有 selftest 全绿(`phase8e_selftest`/`phase8e_contract_selftest`/`phase8d_verifier_selftest`/`phase8e_full_round_selftest`/...)。
3. 新增 prompt-构造断言(放进 `phase8e_selftest`):
   - prompt 不含 `not_instrumented` / `accepted_precomputed_not_applied` / `push_active` / `push_vec_public`(P0)。
   - 每份契约 schema_version 在 prompt 里出现 **== 1 次**(P1)。
   - 完整 `reward_inputs_contract()` 仍含 17 字段且合成校验跑全字段(P0 不破坏校验保真)。
   - prompt 顶部存在唯一任务句 + 聚合反馈行(P2/P3)。
   - prompt 不含 `/data/yuyangliu`(P4)。
4. **真实对照**:对同一 ledger 状态重新生成 prompt,记录"改造前后行数/字节",证明体积下降且信息无损(关键信号仍在)。

---

## 7. 排序

P0(真实能力门 + 去除不可用能力)→ P1(去重)→ P2(任务前置/框架)→ P3(反馈聚合)→ P4(就近 + 路径)。

P0/P1 优先:它们既落实总纲原则,又拿走最大的噪声与体积。P2–P4 是可读性/信噪比收尾。全部做完后,prompt 预计从 ~127 KB 降到 ~60–70 KB 量级,且"只讲真的、先讲任务、信号在前"。

---

## 8. 显式的"先别做"

- 不在本计划接 push 信号 / push 课程(留给 `schedule_push_signal`)。
- 不改 `RewardInputs` 字段集、不改 `DEFAULT_SCHEDULE_ALLOWED_KEYS`、不改合成校验用的**完整**契约——只改**prompt-facing 投影**。
- 不动评测/预算/隔离/分类器/候选 scope。
