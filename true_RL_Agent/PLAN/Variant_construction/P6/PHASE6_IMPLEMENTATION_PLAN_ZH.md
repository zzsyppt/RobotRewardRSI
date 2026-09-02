# Phase 6 实际实施计划：Go2 × 统一 Stair LOOP 与矩阵闭环证据

> 状态：**已完成，GO**；pilot 与 fresh validation_3 均已闭合，后者三代为 `accepted / no_selection / no_selection`
> 编制日期：2026-08-30
> 权威上位计划：`PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md`
> 本阶段对象：`stair.go2_12dof.unified_v2`

## 1. 本阶段只证明什么

Phase 6 在 Go2 行上复用已经由 G1 跑通的受控 reward LOOP，完成真实的
`source -> verify -> train -> public_dev selector -> public_accept decision -> conditional sealed -> ledger/replay`
闭环。它要证明“增加机器人行只增加机器人 profile/binding/reference lineage，不复制 AgentCore、任务插件、
runner、评测器或 LOOP”。

LLM 本阶段仍只可提交完整 reward program。训练分布、地形、课程、seed、训练预算、PPO、评测套卷、
objective、guards、family mask 与 acceptance 均由系统冻结。C0/C1/C2 仍是 Stair 任务内的训练/评测 regime，
不是第三根矩阵轴。

`improvement_observed` 只报告事实，不是工程 GO 门；不得因为没有改进增加 generation、改题、降低门槛或调课程。

## 2. 当前入口事实与边界

- Phase 5 已将 Go2 active V2 资格化为 `cell_loop_ready=true`；reference checkpoint 为
  `sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`。
- Go2 V2 是 evaluation-only successor。Phase 6 的 fresh 含义是新的 output、session、Genesis、ledger、
  current-best 与 generation lifecycle；reference checkpoint、资格化三相证据和 frozen mask 从 Phase 5
  受控导入，不重新训练，也不伪装成新的 reference training。
- pilot 和 validation 是两次独立实验。validation 不继承 pilot 的 candidate、current-best、ledger、
  provider history 或 lifecycle，只共用相同的 Phase 5 reference 与冻结科学身份。若 pilot 后出现只影响其未消费
  路径的必要 blocker 修复或只读验收工具提交，不为追求 source SHA 相等重跑 pilot；验收器必须证明提交祖先关系、
  逐项核对 protocol/binding/mask/reference/首代输入身份未漂移，并在开发日志中解释 source succession。
- Go2 candidate 只能沿 Go2 reference/current-best lineage 演化；任何 G1 checkpoint、mask 或分数进入 Go2
  session 都是阻断错误。
- Phase 4 的 G1-V1 LOOP 仅作“共享主线已经真实跑通”的历史证据，不能当作 active G1-V2 的 LOOP 证据，
  也不能与 Go2-V2 性能数值直接排名。Phase 6 完成后，Core 最终 GO 仍需在 Phase 7 前单独闭合
  active G1-V2 的 `loop_verified`；该缺口不通过文字继承消失，也不扩入本次 Go2 预算。

## 3. 允许的运行中修复

只有满足以下全部条件的问题才作为 blocker 修复：

1. 可由当前源码、原始 receipt、runner summary 或稳定复现用例证明；
2. 会阻止 session 正常启动、完成固定代数、生成正确 prompt/decision/ledger，或会污染冻结身份与权限边界；
3. 修复不改变训练、评测、acceptance 或 LLM 权限语义。

修复后先做定向回归和 source/identity 复核；已完成且身份未受影响的 train/eval receipt 可以复用，只重跑
受影响阶段。能力差、未观察到改善、某个诊断信号区分度不足、经验阈值可能不理想，都不是中途调参授权；
证据尚不足的问题只记录到 `PHASE6_FOLLOWUP_OBSERVATIONS_ZH.md`。

## 4. P6-A：入口与实验身份冻结

### 目标

在不调用 provider、不启动训练/评测的条件下，证明当前产品提交可以选择 Go2 行，并把随后两次实验所需的
代码、reference、GPU、成本和输出身份冻结清楚。

### 工作

- 对齐 local、origin 与服务器产品提交；实验 source SHA 取文档/必要 blocker 修复提交后的最终提交，不把
  工作树 HEAD 本身当永久产品身份。
- 运行当前 `verify_phase5.py --require-go`、Go2 generation 入口/selectionless recovery/prompt capacity/
  pause-resume 定向回归。
- 用生产 `run_stair_generation_session.py` 物化 Go2 foundation，复核 cell manifest、TrainingBinding、
  EvaluationBinding、reference manifest、checkpoint、family mask 和 GPU process binding。
- 运行 `--preflight-only`；它必须只产生 scratch/检查证据，不创建正式 pilot/validation output。
- 运行一次 `--provider-readiness-only` 验证固定 provider 账户可用性；它不构成 generation 请求，也不得启动 GPU。
- 冻结 pilot 与 validation 各自的 session id、绝对 output root、GPU UUID、provider env 路径和 operator caps。

### 验收

1. Go2 使用与 G1 相同的 generation 脚本、runner、verifier、phased evaluator 与 ledger 类型；源码中不存在
   Go2 专用 LOOP/launcher。
2. preflight 报告的 row、reference checkpoint、manifest/binding/mask 均等于 Phase 5 权威身份。
3. provider readiness 明确为可用；余额或权限错误应在任何正式输出、provider generation 或 GPU 工作前停止。
4. 两张 GPU 无残留 compute PID，lease namespace 可用；正式 output 在启动前不存在。
5. prompt capacity preflight 通过，且 preflight/真实 comparison 共用字段契约，而不是两份无约束手写结构。

## 5. P6-B：Go2 单代 pilot

### 目标

用最小真实成本证明 Go2 可以完整穿过共享 LOOP，并检查真实 prompt、反馈、记忆和资源记账是否满足
Phase 6 边界。

### 固定实验

- 新 session、新 output、新 ledger；`cell_variant_id=stair.go2_12dof.unified_v2`。
- 恰好 1 generation、固定 width 2；不得由调用方改代数或扩大 lane。
- 使用 Go2 Phase 5 reference 作为唯一初始 current-best；不导入任何 pilot 之外的 candidate history。

### 运行中监控

- 每次 provider 请求记录 task-section bytes、总 payload bytes、projection level、被保留/省略的 feedback entry；
  原始 payload 做 public/private 泄漏扫描。
- 逐项对照 generation record、public_dev evidence、selector receipt 与后续 prompt，确认目标、三个 guards、
  family/stratum、父代 delta、runtime-effect 和失败归因没有被重新推导或错接。
- 检查多轮记忆边界：source catalog、current-best、最近失败、机制差异信息足量，静态 ABI/完整源码不过量重复。
- 记录 O1-O10 所列观察量，但不据此修改冻结协议。
- 持续核对 train/eval 权威 summary、checkpoint checksum、budget journal、GPU/PID 和 provider usage；曲线或
  progress 只能用于诊断，不能替代 objective/guards/sealed 决策。

### 验收门

1. 一代正常关账，至少一个真实 candidate 完成 source→train→eval→decision→ledger；selection 不存在的合法
   incomplete outcome 可恢复，不得让下一代 prompt 构建崩溃。
2. candidate 只改变完整 reward program；所有 runner-owned hash 与 Genesis 相等。
3. public_dev 决定 selector，public_accept 决定正式判决，sealed 只在协议要求时运行；不得泄漏后两者逐题证据。
4. terminal audit、decision replay、checkpoint checksum、成本和资源闭包可复算。
5. 实际成本不超过 pilot 预冻结上限，且没有未解释的重复 reference/public_dev 工作。

只有以上全部通过，才授权 fresh validation_3。

## 6. P6-C：fresh validation_3

### 目标

在与 pilot 隔离的新 session 上完成固定三代 LOOP，验证跨代反馈、记忆、恢复和正式决策，而不是重复 pilot 状态。

### 固定实验

- 新 output、新 Genesis、新 ledger、新 current-best；只重新导入同一 Phase 5 Go2 reference。
- 恰好 3 generations、固定 width 2。对于只读导入 Phase 5 reference 的 validation，使用
  imported-reference cost contract：本地成功工作仅含 6 次 candidate train 与三代评测，成功预算
  `13.6 GPU h`、retry 预算 `0.2 GPU h`、成功输出 `3 GiB`、retry 输出 `512 MiB`；物理上限为
  `13.8 GPU h / 1,183,555,008 simulated control steps / 3,758,096,384 B`。
- candidate train 每次按协议 `7,200 s` 加 `16 s` runner 清理余量进入成本上界；准入层只检查全局容量，
  不再把历史 G1/P6B 用时变成第二条进程 timeout。评测进程继续使用各自冻结的 operation reserve。
- 已导入且有完整 receipt 的 reference train/qualification 不属于本 session 可执行或可重试工作，不再占用
  本地 operation slot。其余模拟步数与槽位均由 runner-owned cost contract 派生，调用 caps 必须精确相等。
- 不继承 pilot candidate/source/provider history，不因结果增加 generation。

### 严格验收

1. 三代均形成合法终态（accepted、formal_rejected、no_selection 或协议定义的 incomplete），终态报告不在
   最后一步因旧 schema/旧 workload 期望崩溃。
2. 第 2/3 代 prompt 能从公开证据区分 current-best、最近失败、lane 机制差异和 selector 原因；容量降级如发生，
   必须有 receipt，且保留下来的信息仍支持下一轮决策。
3. objective、named guards、family mask、suite、seed、PPO、预算、地形/课程和 reference 身份三代不漂移。
4. Go2 lineage 全程闭合；每个已接受 current-best 的 checkpoint、source/effective reward identity 和父子关系可复算。
5. sealed decision、ledger replay、pause/resume/typed failure 恢复和最终资源释放通过。
6. `loop_verified=true`；`improvement_observed` 如实记录 true/false，不影响本阶段 GO。

## 7. P6-D：终审与交付

### 目标

用独立验收器从原始产物重算 Phase 6 结论，并明确矩阵已经证明到哪里、仍缺什么。

### 交付

- `PHASE6_DEVELOPMENT_LOG_ZH.md`：实际提交、命令、blocker、复用边界、运行与资源时间线。
- `PHASE6_ACCEPTANCE_REPORT_ZH.md`：pilot/validation 结果、成本、decision、lineage、prompt/memory 与矩阵结论。
- `PHASE6_EVIDENCE_INDEX.json` 与 `verify_phase6.py`：从原始 ledger/receipt/summary/checkpoint 重算，不信任报告自报。
- `PHASE6_FOLLOWUP_IMPROVEMENTS_ZH.md`：只收录运行后已有证据、但不阻断 Phase 6 的提升项；不重复已关闭问题。

### 最终 GO 门

1. Phase 5 入口仍为 GO，Go2 active V2 的 reference/manifest/binding/checkpoint 未漂移。
2. pilot 成本门先通过，fresh validation_3 后启动且独立完成。
3. Go2 通过共享 LOOP 达到 `loop_verified`，无 Go2 专用系统或 G1 lineage 混用。
4. LLM 权限始终为 reward-only；训练、评测与 acceptance 语义掌握在系统手中。
5. prompt/feedback/memory 有原始请求和跨代证据支持；无 public_accept/sealed 私有信息泄漏。
6. ledger、decision、sealed、checkpoint、成本和资源闭包可独立复算。
7. G1-29 Push 共享主线与 legacy parity 未退化；共享代码未改变时以当前全量/定向回归证明，不重复无价值 GPU dry-run。
8. 报告不夸大：Phase 6 证明 Go2-V2 闭环和机器人行复用；active G1-V2 的独立 LOOP 证据缺口继续作为
   Core/Phase 7 前置项，不用历史 G1-V1 结果冒充。

## 8. 完成结论

fresh validation 在实验源码 `7e5fa25b607ac118c64424d3e137fa6ce713b2c0` 上完成固定三代：Gen1
正式接受，Gen2/Gen3 被 frozen OOD guard 正确拒绝，未增加 generation。被接受候选将 public_accept
`J_primary` 从 `0.933915` 提升到 `0.974711`，C0-ID/C1-ID 保持 `1.0`，sealed pass。

独立验收器从原始 train/public_dev/public_accept/sealed receipts 重组正式决策与 current-best，并扫描全部
10 次 provider attempts，未发现私有评测泄漏；成本与资源均在冻结上限内闭合。因此本阶段满足全部最终
GO 门，下一阶段为 Phase 7。完整数字、证据边界和非阻断提升项分别见
`PHASE6_ACCEPTANCE_REPORT_ZH.md`、`PHASE6_DEVELOPMENT_LOG_ZH.md` 与
`PHASE6_FOLLOWUP_IMPROVEMENTS_ZH.md`。
