# 2026-06-30 Prompt 下一步改进优先级

## 这份文档基于什么

这份排序基于两部分：

1. 本地代码和最近两轮 GitHub 改动审计
2. 真实 live run 提示词审计

这次重点看的真实产物是：

- `layer2_live_15round_20260630_194237/round_008/provider_candidate/phase8e_prompt_payload.json`
- `layer2_live_15round_20260630_194237/round_008/provider_candidate/phase8e_prompt.txt`
- `layer2_live_15round_20260630_194237/round_008/provider_candidate/attempts/attempt_001/source_requests.json`
- `layer2_live_15round_20260630_194237/round_008/provider_candidate/attempts/attempt_001/source_payloads.json`

## 先说结论

当前 prompt 已经能驱动正确方向的小步改动，但还有三个明显问题：

1. 有 correctness bug，会直接误导模型
2. 有结构性噪声，已经开始淹没主目标
3. 有一些信息分布不合理，不利于模型围绕 current best 做稳定的小步改动

所以，下一步不该继续往 prompt 里加更多原始信息，而应该先：

1. 修 correctness
2. 压缩噪声
3. 重组关键信号的位置

---

## 真实 prompt 的关键观察

这轮 `round_008` 的真实 prompt 情况：

- `phase8e_prompt.txt` 大约 **74.5 KB**
- `required_candidate_package` 大约 **18.5 KB**
- `public_history` 大约 **19.2 KB**
- provider 实际只做了 **1 次 source-read**
- source-read 只请求了 `current_best/training_code/components.py`
- 随后生成了 `phase8e_push_rec_v8`

这说明两件事：

1. 现在这套 prompt 不是完全失焦，模型能抓到“围绕 accepted current best 做小改”这个主线
2. 但当前 prompt 的历史部分已经过重，而且有些数字和字段本身就是错的

再说一个更具体的观察：

- 这轮模型最后交出的 `v8`，本质上是在 `v5` 基础上增强了 `backward_lean_penalty` 和 `stillness`
- 也就是说，模型真正利用到的是：
  - accepted baseline 代码
  - 最新失败方向
  - 小步改动策略

它并没有真正受益于大量 seed 级原始历史细节。  
这很重要，因为它说明接下来应该压缩噪声，而不是继续堆明细。

---

## 优先级 1：先修 correctness bug

### 1. 修 `dominant_failure.rate` 语义错误

这是当前最明显的 prompt correctness 问题。

在真实 `round_008` prompt 里，`public_feedback_summary` 中出现了这种内容：

- `label = fall_backward`
- `rate = 12.333333333333334`

这显然不对。`rate` 不应该大于 1。  
现在这个字段在某些 shaped `failure_breakdown` 下，实际上把 count 混进了 rate。

这个问题的影响很直接：

- 模型会误判失败强度
- prompt 顶层摘要本来应该是高信号，现在反而在给错信号

下一步应该做的事：

- 明确区分 `rate` 和 `count`
- `rate` 始终保持比例语义
- 如果要保留计数，单独给 `count`
- 修 summary 构造和对应 selftest

这是当前第一优先级。

### 2. 修 provider 坏 `operation_diff.json` / `diagnostics.json` 时逃逸 repair loop 的问题

最近一轮改动把 staging 后的异常收窄到只抓 `CandidateLoadError`。  
这个方向有一半是对的：它避免把内部代码 bug 伪装成 provider 错误。

但副作用是：

- 如果 manifest 是合法的
- 但 `operation_diff.json` 或 `diagnostics.json` 是坏 JSON

这类“provider 产坏候选”的情况现在可能直接炸出原始异常，而不是进入 repair loop。

这会导致：

- inner loop 失去一部分“把坏候选反馈给模型修复”的能力

下一步应该做的事：

- 保留“内部 bug 不伪装成 provider 错误”这个原则
- 但把 provider 自己写坏的可选 JSON 文件重新收回到 repairable provider error

这也是第一优先级。

### 3. 对齐 source-read 的 author contract 和 runtime contract

现在 author prompt 和 runtime 的 source-read 合同还没完全对齐。

当前事实是：

- runtime 允许多轮 source-read，受 `max_source_read_turns` 限制
- follow-up message 也允许再次发 `source_requests`
- 但 author prompt 的 system wording 更像“先 source-read 一次，然后就该直接交 candidate”

这会让合同不自洽。

下一步应该做的事：

- 要么明确写清楚“最多可以 source-read 几次”
- 要么把 runtime 收窄成一次
- `system prompt`、`provider_json_output`、follow-up message 三处统一

这个问题虽然不像前两个那样直接产生错误数字，但它会持续制造行为边界模糊，所以也应放在第一优先级。

---

## 优先级 2：压缩 `public_history`

这是当前真实 prompt 里最大的结构性噪声来源。

真实情况是：

- `public_history` 单独就占了大约 **19.2 KB**
- 已经比 `required_candidate_package` 还大

而模型真正需要的每轮信息其实很有限：

- 这一轮是谁
- accept 还是 reject
- public dev 表现
- 主失败模式
- gate 状态
- 改了什么
- 为什么没过

现在 base prompt 里还有很多低价值重内容，例如：

- 完整 `seed_results`
- `condition_flag_breakdown`
- `primary_label_priority`
- 每个 seed 上重复出现的 breakdown 结构
- verifier fingerprint / hash

这些信息不一定有害，但它们已经明显过重。

下一步应该做的事：

- 把 `public_history` 改成 runner-owned compact digest
- base prompt 每轮只保留高价值摘要
- 需要原始细节时，再通过 source-read 给

建议 base prompt 每轮最多保留：

- `candidate_id`
- `accepted`
- `reason`
- `public_dev_mean_success_rate`
- `public_dev_seed_count`
- `dominant_failure`
- `gate_statuses`
- 简短 `public_failure_feedback`
- 简短 `operation_diff`

这一步不只是“缩短 prompt”，更重要的是让模型不要被历史噪声拖走。

---

## 优先级 3：把 current best 的关键信号放到更集中、更前的位置

这轮真实 provider 的行为已经说明：

- 它会先看摘要
- 然后主动 source-read 当前 accepted best 的代码
- 再围绕 accepted best 做小步改动

这条工作流本身是对的。问题不在于方向，而在于当前相关信息分散得太开。

现在 accepted baseline 的信息分散在：

- `current_recipe`
- `public_feedback_summary.best_public_dev_round`
- `public_history`
- `current_best/public_eval_summary.json`

这会导致模型要自己在多个块之间拼出：

- 当前 best 是谁
- 当前 best 的公开表现是什么
- 最新 reject 相比它差在哪

下一步应该做的事：

- 在 top-level 增加一个更集中的 baseline 摘要块
- 只放模型真正需要的字段

建议这个摘要块至少包含：

- `current_best_candidate_id`
- `current_best_public_dev_mean`
- `current_best_dominant_failure`
- `latest_candidate_id`
- `latest_delta_vs_current_best`
- `latest_reject_reason`

目标不是加更多信息，而是把“围绕谁改、为什么改、最近哪里退化了”讲得更直接。

---

## 优先级 4：删除 prompt 中几乎没有决策价值的字段

当前 prompt 中有一批字段是“安全的”，但对模型决策几乎没用。

典型例子：

- `current_recipe.fingerprint.manifest_sha256`
- `current_recipe.fingerprint.entrypoint_sha256`
- `current_recipe.fingerprint.training_code_tree_sha256`
- `public_history[*].verifier.fingerprint.*`
- `source_pack.items[*].content_sha256`

这些字段不会帮助模型决定下一步怎么改代码，但会占掉注意力和 token。

下一步应该做的事：

- fingerprint / hash 默认不进 base prompt
- 如果确实要保留，放到 source-read 或调试产物里

这一步优先级不如前面几项高，但它是很稳妥的降噪项。

---

## 优先级 5：继续瘦身静态合同，但不要乱砍

当前 prompt 的静态合同仍然偏重，尤其是：

- `reward_inputs_contract`
- `reward_inputs_field_expansion_report`

这里不能简单理解成“越短越好”，因为它们本身承担 prompt-facing contract 的作用。  
但其中确实有一部分更像演进说明，而不是 authoring 必需信息。

下一步应该做的事：

- 优先精简 `reward_inputs_field_expansion_report`
- 尽量保留：
  - `field_order`
  - `field_shapes`
  - 精简后的 `field_status`
- 下沉或删除：
  - 演进历史类字段
  - 对 authoring 没直接帮助的说明性元数据

这里的目标不是把合同削空，而是把“约束信息”和“背景叙事”分开。

---

## 优先级 6：清理会误导维护者的陈旧配置

当前 `configs/cold_start/phase8e_loop.yaml` 里的 `candidate_capabilities.allowed` 仍然带着旧语义，比如：

- `LossTerm`
- training-only domain randomization schedules

现在 prompt 构建已经不再用它生成 `candidate_allowed_controls`，所以这不是当前 live prompt 的直接 bug。  
但它会继续误导后续维护者，以为这些能力应该还在 prompt-facing contract 里。

下一步应该做的事：

- 要么明确标注这个字段不再是 prompt-facing truth source
- 要么直接把它清理到和当前 prompt-facing contract 一致

这个问题可以排在后面收，但最好不要一直留着。

---

## 当前不建议做的事

### 1. 不建议继续往 base prompt 塞更多 per-seed 原始历史

这轮真实 prompt 已经说明：

- 问题不是“信息不够多”
- 而是“有效信息不够集中”

继续加原始 seed 级信息，只会让 prompt 更重。

### 2. 不建议把更多 source pack 内容直接展开到 base prompt

这轮 provider 只读了 `current_best/training_code/components.py`，效果已经够好。  
这说明 source-read 机制是有效的。

正确方向是：

- base prompt 更精
- source-read 继续做按需补充

不是把 source-read 的内容重新摊平进 base prompt。

### 3. 不建议重新 advertise 尚未真实生效的能力面

包括但不限于：

- `LossTerm`
- 无效 schedule keys
- 未真正 prompt-visible 的控制面

这条边界现在是对的，应该继续守住。

---

## 推荐执行顺序

建议按下面顺序推进：

1. 修 `dominant_failure.rate`
2. 修 provider 坏 optional JSON 的 repair loop 回收
3. 对齐 source-read author/runtime 合同
4. 压缩 `public_history`
5. 重组 current best 摘要
6. 删除 hash / fingerprint 噪声
7. 继续瘦身静态合同
8. 清理陈旧配置

---

## 最后一句判断

当前 prompt 不是“完全没法用”，而是“已经能工作，但信号组织还不够成熟”。  
下一步最重要的不是加更多信息，而是把错误信息修掉、把重噪声拿掉、把 current best 相关信号摆到模型最容易抓住的位置。
