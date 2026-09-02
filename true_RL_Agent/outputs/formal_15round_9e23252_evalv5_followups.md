# Stair formal-loop follow-up observations

本文件只保留尚未解决、且已由正式运行产物直接观察到的提升点及证据。以下数值来自已停止的 eval-v5 R1-R3；其中成功率属于 eval-v5 的“首次有效到顶”语义，不作为 eval-v6 稳定成功率的证据。

## 35-cell 宏平均会掩盖局部能力退化

- 观察：候选可以在总体分数显著提升时，让最高台阶中的个别困难 tread cell 退化；现有 35-cell 等权宏平均没有单独表达这种局部退化。
- 证据：R1 以 acceptance objective `0.9171` 被接受，最高 `0.40 m` 高度行整体也从 Genesis `0.325` 提升到 `0.495`，但 `height_040/tread_450` 从 `0.30` 降到 `0.125`，`height_040/tread_500` 从 `0.175` 降到 `0.125`；两格都只有 `5 / 40` 次成功。

## 候选解释没有充分利用已提供的局部失败分布

- 观察：authoring prompt 已提供按 cell 分解的失败证据，但候选 rationale 仍可能只引用全局失败率，未说明机制针对的具体几何条件。
- 证据：R1 public-dev 的 `73 / 840` 次 torso-contact 失败中，`66 / 73` 集中在 `0.40 m` 高度；`0.45 m` 与 `0.50 m` tread cell 分别有 `21 / 24`、`20 / 24` 次 torso contact。R2 rationale 只引用全局 `8.7%`。正式评测后，R2 将 `height_040/tread_450` 从 `5 / 40` 提升到 `29 / 40`，却把同一高度的 tread `0.30 / 0.35 / 0.40 / 0.50 m` 从 `21 / 29 / 39 / 5` 降到 `0 / 22 / 26 / 1`，显示其局部 trade-off 没有在候选解释中被充分预判。

## 结构新颖性可能高估奖励语义新颖性

- 观察：单机制结构检查能证明候选只改了一个 canonical term，但不能证明该 term 提供了与现有 reward family 不同的能力信号。
- 证据：R3 新增 `upright_orientation_reward = -projected_gravity[:, 2]`、权重 `0.1`，而当前 accepted head 已有同样依赖 `projected_gravity` 的 `flat_orientation_l2`。该候选通过单-delta/history 检查后完成完整训练与评测，最终因 guard regression 被拒绝。
