# Phase 4 unified Stair reference 报告

> 运行日期：2026-08-29
> 运行源码：`5a30129060353fcbddea0224d4544e23b8d2ee35`
> 结论：reference 资格化通过；三个 OOD family 全部进入本 epoch 的冻结目标

## 1. Reference 身份与训练边界

- cell：`stair.g1_37dof.unified_v1`；
- reference：`official-g1-rough-reward-vnext-v1`；
- source checksum：`sha256:3462bdf78d874ce2e4ab527cea3c1530ebc93cf9750d2b4fc64292e3a63adfb8`；
- manifest hash：`4090bd3b80113ee956400870e17dbe60809cf55d0a15e351ff11583942ff924f`；
- checkpoint：`sha256:3fbbf80d5e7e4ee559420ea06759231c00e2d90316101ace4db6e24ec25810b2`；
- 初始化：random，`initialization_checkpoint_hash=null`；
- 训练预算：`196,608,000 env steps`；
- TrainingBinding：`d7aea85d7742c57eae1753b8b236b32555269d9bcd8c25ed2b3742d2e045e14c`；
- EvaluationBinding：`96d04d130cb669337297bbf8a5b53918b2006810c42586ff0ed374f6c18a0af2`。

pilot 与 validation 都从头执行 reference。两次 reference checkpoint 的字节 SHA-256 相同，这是固定
seed 与冻结训练栈下的确定性结果；服务器文件 inode 分别为 `64772:12195678` 和
`64772:12196051`，不是 pilot 文件复用。

## 2. 官方奖励等价性

`PHASE4_REWARD_EQUIVALENCE_RECEIPT.json` 在服务器 Torch `2.7.0+cu128` 上比较官方 G1 rough V1 与
role-projected VNext：

- 15/15 term 均有限；
- 257 个随机样本逐项最大绝对差均为 `0.0`；
- 加权总奖励最大绝对差为 `0.0`；
- feet-air-time 的 single-stance、双足接触、腾空与静止边界案例均符合官方语义；
- receipt hash：`f12a02eaae2bf29de85e6e81d4c164b13e91c0a2d0a2cea0e95dc396026a7b89`。

因此本轮 reference 不是 Phase 3A 中那份被压平关节作用域的 12 项奖励，而是忠实的 15 项官方结构。

## 3. Learnability 与 family mask

资格化只读取 runner 私有 `public_accept`，不进入 provider prompt：

| 层 | macro progress | macro success | 门结果 |
| --- | ---: | ---: | --- |
| C0-ID | 0.942943 | 0.666667 | pass |
| C1-ID | 0.862312 | 0.311111 | pass |

floor-only family 资格规则的结果为：

| family | reference score | success | 固定 seed 向量 SE | qualified |
| --- | ---: | ---: | ---: | --- |
| `F_scalar` | 0.676789 | 0.503704 | 0.019906 | 是 |
| `F_c1` | 0.631754 | 0.355556 | 0.008167 | 是 |
| `F_c2` | 0.552485 | 0.244444 | 0.034203 | 是 |

- `J_OOD = 0.620343 ± 0.007564 SE`；
- `J_ID = 0.695758`；
- `J_primary = 0.645481 ± 0.009299 SE`；
- `objective_family_mask_hash = 04ad43edf20c7dfb7749492cc0b7e4fadcb881136444c0ccbc4cd40cb3284a54`；
- qualification evidence hash：`89537aa93a5bc18f07563cf07e6bd3d68ec4aaf49cfee67eab8afe0a72853f79`。

这里的 family SE 是 reference 在 5 个固定 `public_accept` seed 上的边际标准误；reference 单独没有
“candidate-reference 配对差”这一统计量。所有后续 candidate 保留同一 seed 向量，接受时才计算真正的
paired delta SE。

## 4. `F_c2` 权重与方差观察

三个 family 全部 qualified 后，`F_c2` 按预注册规则占 `J_OOD` 的 `1/3`、占 `J_primary` 的 `2/9`。
它只有一个 stratum，reference SE `0.034203`，约为 `F_scalar` 的 1.72 倍、`F_c1` 的 4.19 倍；这证明
用户预先要求观察的方差风险真实存在，但没有触发事后改权、补题或换门。

第 1 代被接受候选相对本 reference 的同 seed family 配对统计为：

| family | paired delta | paired delta SE |
| --- | ---: | ---: |
| `F_scalar` | +0.078953 | 0.016889 |
| `F_c1` | +0.351579 | 0.008009 |
| `F_c2` | +0.447515 | 0.034203 |
| `J_OOD` | +0.292682 | 0.006283 |

`F_c2` 仍是单 family 中噪声最大的一个，但本次提升远大于其 paired SE；总接受判定继续只使用冻结的
`J_primary` paired threshold 和三个 named guards，没有额外为 C2 增设专用门。

## 5. 三个 cohort 的 reference 结果

| split | `J_primary` | `J_ID` | `J_OOD` | `F_scalar` | `F_c1` | `F_c2` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public-dev | 0.655091 | 0.726746 | 0.619263 | 0.650421 | 0.676520 | 0.530848 |
| public-accept | 0.645481 | 0.695758 | 0.620343 | 0.676789 | 0.631754 | 0.552485 |
| sealed-validation | 0.673229 | 0.740032 | 0.639827 | 0.662227 | 0.669838 | 0.587418 |

三套卷没有出现结构性退化；C2 的 reference 分数和成功率均处于可分辨区间，因此 Phase 3B 的 C2 claim
没有因 floor 失败退出目标。该结论只说明当前 official reference 支持该评测 family，不等于 LLM 一定能
改善它。

## 6. Reference GO

reference readiness、非空 family mask、随机初始化 lineage、训练/评测 binding 和成本门全部通过。
因此系统按计划授权 1 代 pilot；没有调整课程、地形、seed、PPO、预算、suite、family 权重或阈值。
