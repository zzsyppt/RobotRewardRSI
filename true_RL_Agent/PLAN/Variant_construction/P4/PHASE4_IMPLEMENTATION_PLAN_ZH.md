# Phase 4 实际实施计划：G1-37 × 统一 Stair reference 与 reward LOOP

## 1. 目标与完成定义

本阶段只在唯一 active 格子 `stair.g1_37dof.unified_v1` 上完成两件事：

1. 从随机初始化训练官方 15-term reference，并用冻结的 `public_accept` 在 OOD family 聚合层机械产生
   唯一的 floor-only `objective_family_mask_hash`；低于 `0.10` 的 family 退出本 epoch 声明，ceiling
   family 保留，避免既有能力成为无保护区；
2. 复用现有 generation-session 内核完成 1 代 pilot，成本闭合后再启动一份全新的固定 3 代
   validation，证明 LLM 仅修改完整 reward program 时，source→verify→train→eval→decision→ledger
   可真实闭环。

工程 GO 的必要条件是 `loop_verified=true`，不是必须观察到性能提升。`improvement_observed`
只如实报告；若为 false，不追加 generation、不调低门、不改题池。

## 2. 冻结边界

- LLM 可变：一个完整 reward program；
- 系统拥有且本阶段不可变：robot/profile、terrain、`D_train`、课程、seed、PPO、训练预算、
  checkpoint selector、评测 suite、成功/失败语义、acceptance 与 sealed 判决；
- C0/C1/C2 是 Stair TaskPlugin 内的数据 regime，不恢复 terrain cell 生命周期；
- `objective_family_mask` 是 reference evaluation 的派生会话契约，不进入通用矩阵层，也不修改
  task protocol hash；reference 生成后，candidate、selection、sealed、ledger 与 replay 只能复用；
- provider/LLM 只消费 `public_dev`；`public_accept` 只允许 runner 在接受判决和终态人工审计中读取，
  不得反投影到 prompt 或 memory；
- 旧 C0 checkpoint 只允许作为历史 calibration/transferred evidence，不进入 reference lineage；
- 不新增一套 LOOP、runner 或评测框架。1 代与 3 代入口映射到现有
  `development_generation_pilot` / `development_generation_validation` 内核；15 代 formal 路径不改语义。

## 3. 实施批次与严格验收

### P4-A：运行前契约收敛

**目标**：修复会影响 Phase 4 结论的反馈和身份缺口，再冻结可运行源码。

**产物**：

- per-stratum 非互斥 `diagnostic_flag_rates`；
- reference-derived `objective_family_mask` 及其候选/selection/prompt/sealed/replay 接线；
- 32 KiB 内仍保留 current-best 与 latest candidate 的 family/八层 stratum 因果反馈，以及公开的
  `J_ID/J_OOD/J_primary` 与 family→strata 聚合结构；
- 终态 selector quality 从 runner 私有 `public_accept` phase receipts 独立复算；
- unified Genesis 的 `public_dev` phase receipt 直接复用为首代 parent guard，不重复评测；
- 显式 1 代 pilot 与固定 3 代 validation CLI；
- 更新后的 P3B 当前静态快照；历史 GPU 回执保持原字节，只明确当前评分协议与历史物理证据的边界。

**验收**：

1. 同一 C2 episode 同时触发 slip 与 insufficient-clearance 时，primary failure 只记 slip，
   C2 stratum 的两个 diagnostic rate 均可复算；
2. mask 只能由 reference `public_accept` 的 family floor-only 规则生成；ceiling family 必须保留；若
   全部低于 floor，生成具名 qualification failure 和 family measurements 后停在诊断，不进入 candidate；
   自洽篡改、reference replay 漂移、candidate 换 mask、同一插件二次换 mask 均被拒；
3. mask 后的 `J_OOD/J_primary`、三个 named guards、prompt objective 使用完全相同的 paired-seed
   投影；
4. 最坏合法 unified prompt 不超过 32 KiB；current-best 与 latest 两条反馈均保留全部 family、八个
   strata、主要失败与非零 diagnostic flags，且 prompt 明示 family→strata 和冻结 mask 规则；
5. validation 终态 selector reversal 与 consistency 只从 runner 私有 `public_accept` receipts 计算，
   不读取已经删除的旧 prompt-summary 字段；provider 投影仍不含 `public_accept`；
6. unified cost/operation topology 中不存在额外 parent-public-dev job、argv 改写或占位 slot；legacy
   Stair 仍保留自己的兼容 bootstrap；
7. P3B verifier、P4 定向单测、legacy Stair/Push 回归通过；线性开发检出的 tracked 文件在运行前干净。

### P4-B：官方 reference 与 1 代 pilot

**目标**：在真实双 GPU 环境中先建立可信 Genesis，再验证一代完整循环与实际成本。

**验收**：

1. reference manifest 指向 `official_g1_rough_vnext_v1`，15 个 term 名称、权重、关节组作用域和
   feet-air-time single-stance 语义与冻结 authority 一致；初始化方式为 random，训练预算为
   `196608000` env steps；
2. reference 的 C0-ID、C1-ID 同时达到 readiness floor，所有 OOD strata 的分数/成功率/progress/
   primary failure/diagnostic flags 有限且可复算；
   若没有任何 OOD family 达到 reference floor，资格报告后停止，不创建 candidate；
3. Genesis public result、private qualification evidence、current-best 与 reference artifacts 绑定同一
   mask hash、checkpoint checksum、manifest、TrainingBindingV2 和 EvaluationBinding；
4. pilot 恰好 1 generation、2 个 sibling candidates；两者只改变 reward source/effective identity，
   runner-owned contracts 与 reference 相同；
5. ledger 恰好闭合一代，decision 可重放；若有 candidate 进入 formal evaluation，acceptance 只读取
   `J_primary`、`J_OOD/C0-ID/C1-ID` guards 与 sealed gate；
6. 实际 GPU/steps/output/provider 使用未超过 pilot contract，且没有残留目标 GPU 进程。

若 reference readiness、mask 非空性、接线或成本任一失败，本阶段停在诊断，不启动 validation。

### P4-C：fresh 3-generation validation

**目标**：仅在 P4-B 全部通过后，用新 output/session/Genesis 执行固定 3 代验证，排除 pilot 状态复用。

**验收**：

1. validation output 在启动前不存在，source commit 与 pilot 相同，Genesis/reference 从头建立；
2. 恰好 3 generations、每代 width=2，不追加代数；每代的 parent head、候选、selection、正式判决、
   ledger height 和 cost journal 连续闭合；
3. 所有 generation 共用 validation Genesis 的唯一 mask；不得复用 pilot mask、checkpoint 或 ledger；
4. prompt/memory 只含 public_dev 聚合；扫描所有 provider request，禁止出现 public_accept 逐题、sealed
   summary、sealed geometry 或 final holdout；终态人工报告可由 runner 私有 receipts 计算
   public_dev selector 相对 public_accept 的 reversal/consistency，但该指标不得回流给 provider；
5. validation 终态报告、ledger replay、generation closure、checkpoint lineage、artifact checksum、
   provider usage 与 GPU resource closure 全部复算一致；
6. 报告 `loop_verified`、`improvement_observed`、每代 J/guards/sealed 状态与 paired uncertainty，不把
   未提升解释成工程失败。

### P4-D：终审与清理

**目标**：形成最小、可复核的 Phase 4 证据包，并删除本阶段无后续价值的临时脚手架。

**产物**：验收报告、开发日志、机器验收器、reference/pilot/validation 摘要与原始 artifact 引用。

**验收**：

1. 机器验收器从冻结文件和真实 session artifacts 独立复算核心身份、mask、成本、ledger、decision、
   lineage 与泄漏扫描；
2. 完整非 GPU 回归通过，旧 G1-37 legacy Stair、G1-29 Push 任务插件无退化；
3. 只清理本阶段明确创建且不再被证据引用的临时文件/进程，不触碰历史 outputs；
4. 本地、origin、服务器最终 commit 一致，服务器线性开发检出的 tracked 文件干净，目标 GPU 无残留进程。

## 4. 不做事项

- 不因某个 family 方差较大而事后改权重、episode 数或 guard；
- 不开放课程、地形、PPO、seed 或评测给 LLM；
- 不把 mask、C2、clearance 等 Stair 专用语义放进 AgentCore、Runner 或通用矩阵组合层；
- 不新增多套“安全框架”、通用工作流引擎或与验收无关的防御测试；
- 不把 pilot/validation 的结果写回 task protocol、suite 或 acceptance 配置。
- P4-A 的 floor-only 评分修正发生在任何 Phase 4 reference/candidate 运行之前；它会改变当前
  suite/protocol/manifest 身份，但不得把历史 P3B GPU 回执冒充为新评分协议下的行为证据。
