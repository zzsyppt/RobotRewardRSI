# P4 生产版 sibling exclusion 小样本复测

## 结论

旧 author-only 探针的 `13/15 = 86.7%` 不能继续作为生产 P4 的 paired-ready 实测值。旧探针向 B 暴露了 A 的 candidate ID、round ID 和逐轮手写自然语言语义；生产实现只允许 verifier-owned 的 8 字段 `single_component_semantic_exclusion.v1`。旧结果只能视为信息更充分条件下的上界。

本次选择 R01、R04、R08、R11、R14 五份冻结真实 prompt，用当前生产投影器构造排除信息，并放入真实 provider 路径：

`required_candidate_package.component_contract.static_verifier_rules.single_component_semantic_exclusion`

每份只请求一次，不重试，不使用 GPU，不写 scientific ledger。请求统一使用 `deepseek-v4-flash`、thinking enabled、`reasoning_effort=max` 和当时的 16,384 output cap。

结果：

| 样本 | provider 结果 | 人工核对 |
|---|---|---|
| R01 | 形成候选 `candidate_foot_step_regularity_v1` | 读取完整源码后确认：使用脚接触和双脚相对高度塑造摆动/双支撑规律，与被排除的 `goal_progress_fraction` 直通稠密奖励在输入、结构和作用机制上均不同；生产比较器亦判 distinct。静态验证通过；随后在服务器基线 `3af99dd` 的 Isaac/Torch CPU 环境中完成 16 个 RewardTerm smoke，全部通过。 |
| R04 | `finish_reason=length` | 16,384 units 全部用于 reasoning，未产生候选 JSON；不可判断机制差异。 |
| R08 | `finish_reason=length` | 16,384 units 全部用于 reasoning，未产生候选 JSON；不可判断机制差异。 |
| R11 | `finish_reason=length` | reasoning 接近上限，只留下不完整 content；不可判断机制差异。 |
| R14 | `finish_reason=stop`，但 final content 为空 | JSON 草稿留在 `reasoning_content`，生产解析器正确拒绝。人工阅读草稿可见它仍在探索 base-clearance pacing 近族，但因为没有正式候选，不能计入 paired-ready 或等价泄漏统计。 |

因此本次可陈述的事实只有：

- 生产版最小排除信息至少在 R01 能引导出真正不同且静态合法的候选；
- R01 的服务器 component smoke 也通过，因此首请求 production-shaped paired-ready 为 `1/5`；
- 1 个可解析候选中没有改名/参数等价泄漏；
- `paired-ready = 1/5`，但其中 4/5 是 provider 成稿失败，不是机制碰撞；这个样本不能估计排除后的真实不同候选概率；
- `deepseek-v4-flash + reasoning_effort=max + O=16,384` 与当前长代码候选合同不匹配，首请求成稿率只有 `1/5`。

五次权威 usage receipt 合计：prompt `117,709`、completion `58,960`、其中 reasoning `56,455` units；prompt cache hit `9,984`、cache miss `107,725`。这里记录实际用量，不把会变化的价格换算成伪冻结金额。

## 处理决定

1. 保留生产 8 字段排除结构，不把旧探针的 candidate ID、round ID 或手写语义塞回 provider prompt。
2. provider 默认切换到 `deepseek-v4-flash` 和 `reasoning_effort=max`；thinking 请求不再发送无效的 temperature 字段。
3. output cap 从 16,384 提升到 32,768。这是对本次 3 个确定性 length 截断的最小保守修正，不是声称 32K 已被本样本证明充分；P6 provider-only gate 仍需在 GPU 前验证五次 attempt 的真实 paired-ready 与物理成本。
4. 本次不追加请求来美化比率，也不从 `reasoning_content` 偷取候选。旧 86.7% 与本次 1/5 都不得授权 GPU pilot。

## 信息流核对

五份生产 payload 均确认：

- exclusion 恰好 8 个字段；
- 不含 A candidate ID、round ID、源码、rationale、response、evidence ref 或路径；
- `canonical_mechanism_delta` 中会暴露 verifier 的组件机制名，它只用于定位被排除的组件，不是等价判断依据；真正的 blocking 比较仍忽略改名并使用 normalized component semantics；
- 没有顶层旧 `sibling_candidate_exclusion`；
- model、thinking、reasoning effort 均写入冻结 request/receipt。

逐份人工源码/响应判断与 request/response SHA-256 见 `outputs/production_exclusion_recheck_20260817/MANUAL_REVIEW_ZH.md`；R01 服务器 smoke 原始 JSON 为 `round_01/server_smoke_result.json`（SHA-256 `a0394c4012e3f8a60fade0d2642ac910130a25841de7a471dd745df6b5397a86`）。

## 权限边界

```text
production exclusion plumbing: VERIFIED ON 5 REQUEST SHAPES
first-attempt provider readiness at 16K: FAIL (1/5)
equivalent leak among parseable candidates: 0/1
32K max-reasoning readiness: OPEN
GPU / P6-B authorization: NOT GRANTED
```
