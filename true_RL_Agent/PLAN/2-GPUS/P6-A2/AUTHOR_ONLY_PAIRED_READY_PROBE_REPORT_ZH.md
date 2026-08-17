# Author-only paired-ready 探针报告

> **2026-08-17 纠偏：** 本报告的 86.7% 使用的是试验性、信息更充分且违反最终 A→B 最小信息流的 exclusion（包含 A candidate/round ID 与逐轮手写语义），不是 P4 最终生产 payload。它只能作为上界，不能再作为生产 paired-ready 实测或 GPU 授权依据。生产 8 字段复测见 `PLAN/2-GPUS/P4/PRODUCTION_EXCLUSION_RECHECK_REPORT_ZH.md`。

## 1. 结论

本次共完成 **15 次实际 provider HTTP 请求**，严格串行、每个样本只请求一次、不重试、不使用 GPU。

- `13/15` 得到可物化、静态验证通过、服务器 CPU component smoke 通过，且经逐份人工阅读后确认与本轮 A **实质不同**的 B；
- `1/15` 为 provider 传输中断：`http.client.IncompleteRead`，只收到 2 bytes；
- `1/15` 为 `finish_reason=length`：16,384 output units 全部进入 reasoning，没有候选 JSON；
- 改名、改权重、数值包装或同结构机制的**漏放为 0**。

因此，本样本的首请求 paired-ready 率为：

```text
13 / 15 = 86.7%
Wilson 95% 区间约为 [62.1%, 96.3%]
```

它通过了主计划预注册的工程门：`valid_structurally_distinct >= 10/15` 且等价漏放为 0。

更精确地说：

- 按全部实际调用计，首请求 operational paired-ready 为 `13/15 = 86.7%`；
- 排除唯一传输失败后为 `13/14 = 92.9%`；
- 在成功得到可解析候选的 13 次中，人工确认的结构异质率为 `13/13 = 100%`。

样本只有 15 个，不能把 86.7% 当成稳定不变的真实概率；它足以回答本次工程问题：**在 A 的单组件语义被明确排除后，B 通常仍能找到一个有效且真正不同的方向，串行 A→B 不是表面上的 width=2。**

## 2. 重要边界

这次是用户明确批准的 author-only 实测，但它**不是主计划中正式 P6-A2 生产接线门的闭合证据**：

- 当前代码树尚无 P4 的生产 `single_component_semantic_exclusion` 路径和 `GenerationCoordinator`；
- 本探针把拟议的 public-safe exclusion 注入 15 份已保存 prompt，并直接复用现有 verifier-owned 单组件 operation delta 与等价比较原语；
- 因此它验证的是 exclusion 语义和 provider 可行性，不证明 P4 coordinator、receipt、budget、continuation 或 production prompt plumbing 已完成；
- 本结果不授权 P6-B GPU pilot，也不改变 temperature、attempt cap、acceptance 或 sealed 协议。

## 3. 冻结输入与执行合同

- 代码 HEAD：`3af99dd6999f5119c5458fd6218efe22e34febda`
- 来源 session：`stair-reference-formal-20260813T125345Z`
- 来源快照压缩包 SHA-256：`c285da71b3c2b6110a8f02f010f61bdb93ae1ac1fb5172e0f236ef84224e2b79`
- 样本：该 session 的 15 份真实首轮 prompt/base context；每轮 A 均相对其 own anchor 投影单组件语义
- model：`deepseek-v4-pro`
- temperature：`0.2`
- reasoning：`adaptive`
- output cap：`16,384`
- 发送方式：A-first 语义已知后构造 B exclusion；15 个 B 请求严格串行；每个样本零重试
- 禁止面：无 train、无 eval、无 sealed、无 primary ledger 写入、无 current-best 更新

首次尝试 R01 前曾因误用 API key 环境变量名而在本地入口前失败；它没有发出 HTTP 请求，不计入 15 次。修正为服务器真实的 `DEEPSEEK_API_KEY` 后，R01–R15 恰好各发出一次实际请求。

## 4. 人工逐项验收

每个有候选的样本均由人工完成以下检查，而不是采用脚本 verdict 代判：

1. 读本轮 A exclusion 的 operation、输入依赖和自然语言语义；
2. 读 B 的实际 `components.py` 及被改函数/组件；
3. 判断是否只是改名、权重、clip、阈值、数值缩放或同计算包装；
4. 核对现有静态 verifier 的单组件 delta；
5. 在正式服务器 Python/Torch 环境做 CPU-only Stair component smoke；
6. 人工写入逐轮结论表。

| 轮次 | 被排除的 A | B | 人工结论 |
|---|---|---|---|
| R01 | `stair_goal_progress_v1` | `stair_base_elevation_v1` | 有效异质 |
| R02 | `stair_base_clearance_v1` | 无候选 | provider 传输失败，不可评价 |
| R03 | `stair_progressive_clearance_v1` | `stair_torso_contact_force_penalty_v1` | 有效异质 |
| R04 | `stair_torso_force_penalty_v1` | `stair_support_clearance_v1` | 有效异质 |
| R05 | `stair_step_span_clearance_v1` | 无候选 | response truncated，不可评价 |
| R06 | `stair_height_conditioned_clearance_v1` | `stair_goal_gated_command_v1` | 有效异质 |
| R07 | `stair_top_margin_clearance_v1` | `stair_foot_height_reward_v1` | 有效异质 |
| R08 | `stair_goal_progress_v1` | `stair_clearance_gated_velocity_v1` | 有效异质；关键重命名绕过样本未复现 |
| R09 | `stair_scan_proximity_clearance_v1` | `stair_top_foot_settle_v1` | 有效异质 |
| R10 | `stair_double_support_progress_v1` | `stair_dense_torso_force_v1` | 有效异质 |
| R11 | `stair_torso_force_penalty_v1` | `stair_progress_orientation_penalty_v1` | 有效异质 |
| R12 | `stair_support_aware_clearance_v1` | `stair_terrain_relief_clearance_v1` | 有效异质 |
| R13 | `stair_scan_clearance_v1` | `stair_span_scheduled_clearance_v1` | 有效异质 |
| R14 | `stair_quadratic_top_clearance_v1` | `stair_top_foot_speed_reward_v1` | 相对 A 有效异质；与探针 R09 属相近机制族 |
| R15 | `stair_footspan_progressive_clearance_v1` | `stair_swing_foot_clearance_v1` | 有效异质 |

逐轮人工机制说明见：

`D:\humanoid_agent\true_RL_Agent\outputs\author_only_paired_ready_probe_20260817\MANUAL_REVIEW_ZH.md`

## 5. 对系统设计的含义

### 5.1 可以保留串行 A→B

这组数据不支持“temperature=0.2 会让 B 在排除 A 后仍经常只会复述 A”的担忧。13 个可解析候选全部绕开了本轮 A，R08 还直接证明：排除 goal-progress 直通奖励后，模型没有用改名/改权重重新提交，而是改为净空门控速度跟踪。

### 5.2 首请求损失主要来自 provider 可靠性，不是机制碰撞

两次失败分别是传输中断和 reasoning 撑满输出上限，没有一次是“得到候选但只能复述 A”。生产 P4 仍需沿用有界 authoring attempts；本探针只测首响应，因此不能据此推导 5-attempt 后的 paired-ready 率。

### 5.3 局部 exclusion 不能替代全局历史

R09 与 R14 都属于“顶端足速稳定”近族；R03 与 R10 都属于“躯干接触力连续惩罚”近族。它们各自相对本轮 A 都是真正不同的 B，所以正确计为 paired-ready；但这也说明跨 generation 的 verifier-owned exact/parameter equivalence 与有界历史证据仍有价值。不能把本轮 A exclusion 扩张成一个会误杀合法近族变体的模糊语义分类器。

## 6. Provider 用量与成本

14 个有 response receipt 的请求合计：

- prompt：371,214 units
- cache hit：136,320 units
- cache miss：234,894 units
- output：174,798 units
- 其中 reasoning：147,945 units
- 已知总量：546,012 units
- 按 P0 采用的单价口径，已知成本：`¥1.756878`

R02 的传输失败没有返回 usage receipt，因此总成本不能写成一个伪精确值；应表述为“已知 ¥1.756878，加 1 次无 receipt 的失败请求成本”。

## 7. 无 GPU 与无科学证据污染验证

- 训练作业：0
- 评测作业：0
- sealed 查询：0
- 新 primary ledger 写入：0
- 候选服务器检查全部是 CPU-only static/component smoke
- 收口时 `nvidia-smi --query-compute-apps` 无 compute process
- probe transport/static-check 进程无残留

输出目录中的 `ledger.jsonl` 仅是从正式 session 复制来的只读 source snapshot，不是本探针生成的科学 ledger。

## 8. 结论状态

```text
authoring feasibility: GO
pre-registered numeric gate: PASS (13/15, equivalent leaks=0)
formal production P6-A2 gate: OPEN
P6-B GPU authorization: NOT GRANTED
```

下一步应先完成 P4 的生产 exclusion/receipt/coordinator 接线及 fake/golden 验收，再用本结果作为“该设计值得接线”的实测依据；不能把一次性探针代码直接搬进生产路径。
