# Phase 3B 开发日志

## 1. 目标与边界

本 Phase 将 C2 实现为 Stair 单格内的 evaluation-only 表面机制 OOD。实现始终保持：机器人 × 任务
是系统矩阵的两根轴，terrain regime 不是第三根轴；LLM 仍只能修改完整 reward program；训练分布、
课程、seed、预算、PPO、评测与验收由系统冻结。

## 2. 主要实现

1. 新增 task-owned `rough_surface.py`，以确定性的低频、边界归零分片线性表面同时驱动 mesh 与 query；
2. ordered geometry schema 升为 v3，`surface_profile` 进入 cell digest；
3. 新增 watertight C2 tread solid mesh，start/top platform 保持 flat；
4. decoder 通过 Stair query factory 解析 flat/C2，不在共享层识别 roughness；
5. 三套 formal cohort 各增加 3 道 C2，保留相同 4/6/8 mm 难度阶梯并保持几何隔离；
6. 增加 `F_c2`、per-stratum primary failure breakdown 和 compact prompt 保真；
7. episode 重复数 4→3，使 8-strata 正式评测成本为 1080，低于原 1260 上限；
8. 扩大 Stair execution source manifest，使训练/评测真正绑定 mesh、query、decoder 与 scoring 源码。

## 3. 真实运行与根因修正

### 3.1 训练排除

在 draft 最终语义上独立运行 128-step PPO smoke：16 env 的 terrain 分配为 C0=8、C1=8，C2=0。
随后只改变 `protocol_state` 和由其决定的 protocol/manifest 状态身份，task protocol hash 保持不变。

### 3.2 C2 校准

在 active 最终字节上运行 RTX 5090 headless Isaac：9 个正式 C2 mesh 全部 watertight，mesh/query
最大误差 `3.33e-16 m`；scanner 有 120/187 条 ray 观测到至少 1 mm 差异，最大 `4.52 mm`；
surface-relative failure 分类完全一致，固定 world-z 的 82/896 次翻转全部位于 6 mm 幅值带内。

### 3.3 执行编排问题

最初把 C2 calibration 和 PPO smoke 放在同一个 Isaac 进程中。第一个环境关闭后第二次创建训练环境
长期停在 CPU、GPU 0% 利用率，属于同进程顺序重建 SimulationApp 的执行编排问题，不是 C2
几何或训练失败。根修是将两个必要证据拆成两个独立进程；不新增重试框架或 workflow 状态机。
孤儿进程只按已核对的 PID/PGID 终止，GPU 释放后删除该次空输出和未生成回执。

## 4. 完工审计与清理

- `is_irregular_cell` 在 active `src/`、`scripts/` 中为 0；
- C2/roughness 标记只出现在 `tasks/stair_climbing`，共享矩阵层为 0；
- 统一训练 generator 只含 C0/C1，C2 仅以 forbidden regime 出现在训练契约；
- 未发现残留 `g1_irregular_v1`、独立 C1/C2 cell 或旧 Phase 3A 开发模块；
- 删除 5 条只证明“当前协议是 draft”的重复开发期测试，改为 active manifest 正向入口测试；
- 保留一条真正构造 draft 协议的负测，继续证明 catalog fail-closed；
- 资格 hash 篡改测试不再被状态门遮蔽，而是运行到程序资格与 execution authorization 的真实绑定门。
- 最终定向回归为 `127/127`，完整非 GPU 回归为 `1525 tests` 全通过、`30 skipped`。

## 5. 明确未做

- 未执行完整 reference/candidate 训练，不根据策略结果调整 C2；
- 未新增 terrain cell、C2 reference、C2 ledger 或 C2 LOOP；
- 未修改 acceptance 阈值、failure taxonomy 阈值或 LLM 权限；
- 未把 legacy equal-height final-confirmation 协议强行套到 unified suite；这由 Phase 4 随 reference
  与 objective-family mask 一起完成。

## 6. 验收入口

权威验收入口为 `PLAN/Variant_construction/P3B/verify_phase3b.py`。它不固定 Git HEAD，而是比较最终
协议/manifest、两份运行回执和 17 个关键产品文件的 normalized-LF 内容哈希；文档提交不会让入口
自失效。两份运行回执均用 canonical JSON 内容哈希，跨平台换行不会破坏证据闭包。

## 7. 完工后独立审计修正（2026-08-28）

### 7.1 Prompt 二源根修

审计确认 `prompt_adapter.py` 仍手抄 7 个 strata，并把已经进入正式套卷的 C2 描述成 future，导致
同一 payload 与真实 `stratum_diagnostics` 自相矛盾。根修没有在字符串上补第八项，而是：

- 从 active `public_dev` cells 派生 strata 顺序、每层题数和 claim kind；
- 从 suite score contract 派生 `J_primary` 与 named guards；
- 从 training profile 派生 C1 training phase 和 C2 exposure；
- 将 unified task prompt 升为 v4、public-eval prompt 升为 v2，并明确 C2 是当前
  evaluation-only surface/contact mechanism shift；
- P3B verifier 直接比较 prompt 投影与 suite/profile，新增“删除 C2 stratum 必须失败”的负测。

### 7.2 可感知 claim 收窄

训练期 height-scan 噪声为 `UniformNoiseCfg(±0.100 m)`，标准差约 `57.7 mm`；真实 C1/C2 scanner
最大差只有 `4.52 mm`，约为一个噪声标准差的 7.8%。因此不改变训练噪声、不突破 15 mm 拓扑硬
上限，也不再把 scanner smoke 解释为策略显著感知。它只证明 mesh→sensor 物理链有效；C2 的
Core V1 claim 是训练未见的表面/接触机制迁移，是否获得本 epoch 的研究支持由 Phase 4 reference
family floor-only mask 决定。

### 7.3 目标权重、采样与 hidden 范围

保留 family 等权：它表达三个研究 claim 的优先级，不是方差权重；三 family 均合格时 `F_c2` 占
`J_primary` 的 `2/9`。保留 3 episodes：success 虽较粗，但 score 还含连续 progress，candidate/head
使用 paired seeds，sealed 门消费聚合目标；没有证据支持为此增加成本或改阈值。三套卷继续使用相同
C2 profile 多重集以对齐难度，因此当前不声称对未见 roughness 实现泛化。

### 7.4 证据闭包修正与重跑

原 GPU 回执把纯 prompt 文件误纳入 runtime source closure，导致 prompt-only 修复会错误作废几何
证据。现将 prompt 与 post-rollout scoring 移至静态 contract snapshot 验收，runtime closure 收敛为
真正影响 mesh/scanner/decoder probe 的 11 个文件；随后在 RTX 5090 上按同一 seed 重跑，核心物理
指标不变，新 receipt hash 为
`e8a8e50e5c2225d83d76d0f73c466483b28d86fe7d398750daf85025a66c9a12`，snapshot hash 为
`ac32b92e592eddd6597b2409993de26b385f6a902dc6e447ad78bd198c13625f`。
后审计定向回归为 P3B `67 tests` 通过、`2 skipped`，verifier 自测 `4/4`，legacy Push `76/76`；
完整非 GPU 回归仍为 `1525 tests` 全通过、`30 skipped`。

Phase 4 预实验审计随后把 provider 评分语义升级为 `public_dev-only`，并新增 unified
evaluation/acceptance guard 名称的源头一致性检查。P3B 当前静态契约快照因此重签为
`35739d6cce013381c6cc52bf12fd36e6703666eae12388e5e09fb75eed1cf1ae`。同一轮继续修正 ceiling
family 无保护、空 mask 异常、prompt 反馈容量与评分二源后，当前静态快照再次机械重签为
`75e088a8a69a521a1859a2e718cb74d926a9ffd2ece4dfd999dfad9f9e1da9d1`。服务器回归审计继续修复
calibration cohort 缺少 `F_c1` 时的派生聚合，并将当前静态快照机械重签为
`3fd7300e6fa3a8b725d1a3304424843ce11eac1dbe631bd8f3be3e2b9aa8d5bd`。进入 reference 前继续修复
minimum-core stratum 表把被 top-2 截断的非零失败率伪装为 `0.0` 的真值缺陷，v2 表以 `null`
表示截断、保留真实零值，并将当前静态快照机械重签为
`92b94b749e490e45b8ad9daf70079299174d0827a544b3c930d980b22474b861`；上述 P3B 完工 hash 和中间
静态 hash 均只保留为迁移记录，真实 C2 运行回执与训练排除回执未重跑或改写。当前 verifier 明示
`rollout_geometry_or_training_changed=false`。
