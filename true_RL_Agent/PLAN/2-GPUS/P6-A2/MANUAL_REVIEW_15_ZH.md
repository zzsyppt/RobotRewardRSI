# P6-A2 正式 15 例逐件人工复核

> 复核对象仅为正式输出
> `/data/yuyangliu/true_rl_agent_phase_g/p6a2_author_only_b5dff4d_20260819T165305Z`。
> 人工结论只能维持或降级机器分类，不能把 reasoning 草稿、失败 smoke 或未物化候选升级。

| 轮次 | A exclusion | B 单组件 delta | 机器结论 | 人工结论 |
|---|---|---|---|---|
| R01 | `goal_progress` | 无正式 final candidate | `provider_failed` | 维持；final content 为空，reasoning 草稿不使用 |
| R02 | 固定 base clearance | 新增 `upright_posture_exp` | valid | 维持；gravity-xy 指数姿态奖励与 A 不同 |
| R03 | progress-scheduled clearance | 新增 `arrival_stability_reward` | valid | 维持；近顶端速度、姿态与角速度联合塑形 |
| R04 | torso contact-force penalty | `base_clearance_penalty -> terrain_aware_clearance_penalty` | valid | 维持；height-scan 局部净空增量与接触力惩罚不同 |
| R05 | step-span-conditioned clearance | 新增 `swing_clearance_shape` | valid | 维持；span deficit × swing height 不是 A 的参数变体 |
| R06 | foot-height-conditioned clearance | clearance 加入 forward-progress margin | valid | 维持；新增输入和调度结构，不是参数等价 |
| R07 | top-margin clearance | `flat_orientation_l2 -> orientation_safety_margin` | valid | 维持；姿态 hinge 与 A 的净空机制不同 |
| R08 | `goal_progress_reward` | 新增 `obstacle_headroom_penalty` | valid | 维持，但记录 height-scan 语义过度解释风险 |
| R09 | height-scan proximity penalty | step-quantized base clearance | valid | 维持；输入与函数结构都不同 |
| R10 | top double-contact progress | 无正式 final candidate | `provider_failed` | 维持；final content 为空，reasoning 草稿不使用 |
| R11 | torso contact-force penalty | 新增 `top_stability_reward` | valid | 维持；near-top upright/support/foot-speed 联合门控 |
| R12 | support-aware base clearance | 新增 `top_platform_stability_penalty` | valid | 维持；near-goal 双足接触下 lean+sway 惩罚 |
| R13 | scan-based clearance | support/posture-aware clearance | `authoring_invalid` | 维持；smoke 输出 `[2,2]`，合同要求 `[2]` |
| R14 | quadratic top clearance | 新增 `top_progress_bonus` | valid | **降级为 `authoring_invalid`**；upright 符号错误 |
| R15 | foot-span progressive clearance | 新增 `goal_proximity_reward` | valid | 维持；绝对剩余距离的指数邻近奖励与 A 不同 |

## R14 人工降级依据

R14 使用：

```python
upright = (inputs.projected_gravity[:, 2] > 0.85).float()
```

同一正式 prompt 的 task-owned 输入合同明确给出 `small_angle_xyz=[pitch,-roll,-1]`，并写明
“Use `-z` for upright reward; `+z` favors inversion”。因此直立附近 z 约为 -1，R14 的新增
奖励在正常直立域恒为 0，反而只在倒置方向可能点火；这与其 rationale 的 upright 声明直接
矛盾。现有 smoke 只验证有限值与输出形状，所以机器门未发现这一语义退化。原机器制品保持
`valid_structurally_distinct` 不改写；最终验收单独记录人工降级。

## R08 风险为何不降级

Stair 合同中的 `height_scan` 是向下地形扫描，不提供 overhead/torso sensing。因此 R08 的
“headroom”命名及“前方障碍”解释偏强。但它对 scan 做允许的顺序不变归约，结果有限且形状
正确；`min_scan + 0.5` 仍可作为局部地形净空代理，并且相对本轮 A 的 goal-progress 奖励是
真实不同的机制。这是科学质量风险，不是 identical/parameter-only、合同非法或等价漏放。

## 复核后的门

```text
machine: 12 valid / 2 provider_failed / 1 authoring_invalid / 0 leak
manual:  11 valid / 2 provider_failed / 2 authoring_invalid / 0 leak
gate:    11 >= 10 and leak == 0 -> GO
```

11 个保留 valid 的候选均满足：相对 own anchor 恰好一个 canonical RewardTerm delta；static、
smoke、anti-degenerate、production sibling comparison 全部通过；独立 verifier comparison 为
`None`；人工读取 A exclusion 与 B 源码后确认不是改名、改权重、numeric-only 或 parameter-only。

候选仍明显聚集在净空/地形族与近顶端稳定族。这不构成同一 case 的 sibling leakage，但说明
本探针证明的是“面对一个 A exclusion 能否给出另一个结构”，不证明跨代概念探索已经充分
多样，也不提供候选科学效果证据。
