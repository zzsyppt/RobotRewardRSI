# P3 实施计划：分阶段评测、selector 边界与成本收敛

> 日期：2026-08-16  
> 入场 HEAD：`d38385c922d3594b443508abdcd99d33e8290cf8`  
> 本阶段不启动 GPU、API、服务器实验，不实现 GenerationCoordinator、generation ledger 或探索前沿。

## 1. 阶段目标

建立一条固定、可验证的评测路径：

1. A/B 各自只运行 `public_dev`；
2. Stair selector 在任何 `public_accept` 产生前冻结唯一 provisional winner；
3. A/B 均可运行 `public_accept`，但其结果不能回流重选 winner；
4. 只有 winner 可以运行 `sealed_validation`；
5. winner 的三份 phase receipt 聚合后，继续复用现有 `PublicEvalResult`、sealed gate、required gates 和 acceptance；
6. loser 的公开结果只作为后续探索证据，不能产生 accepted decision。

## 2. 关键边界

### 2.1 公共层只拥有执行合同

- 公共层只认识固定的三个 phase、完整 eval plan、任务无关的 phase result、执行成本、设备/资源绑定和 create-once receipt。
- Stair 的 35-cell 证据、objective、guard、selector、sealed 组装与成本含义全部留在 Stair 插件。
- Push/Neutral 默认不支持 phased eval，继续沿用 width=1 完整评测。
- 不引入通用 DAG、Graph 数据库或动态 agent orchestration。

### 2.2 selector 是资源分配，不是验收

- selector 只接收经 task-owned projector 验证的 `StairPublicDevScreenEvidence`。
- A/B 必须共享 task、eval/evaluation/training/capability 和软件执行环境域；`execution_device`、物理 `resource_binding_hash`、subject 和 checkpoint 可以不同，但必须显式记录为允许差异。
- current parent 提供同域 public-dev guard reference。
- 先排除相对 parent 已越过 acceptance protocol 冻结的 guard 硬回退上限的候选；这只是明显风险预筛，不复制 public-accept 的动态 guard、absolute floor、headroom 或 sealed 门。
- 对剩余候选始终按 public-dev generalization objective 排序。`1.5×paired_SE` 只记录 `inside/outside estimated noise band`，不再决定是否弃权。
- objective 完全相同时可用 public-dev guard 次序；只有 objective 与 guard 都完全相同，或没有 guard-eligible 候选时才 no-selection。
- selection 明确 `acceptance_authority=false`，不能产生 decision。

### 2.3 正式组装必须绑定已执行证据

- 每个 phase receipt 同时绑定 trial contract、phase job spec、完整 plan 和 phase result hash。
- Stair assembler 必须重放 selection，并确认被组装 trial 是 frozen winner。
- assembler 输出 typed `EvaluationAssembly`，其中包含 selection、selected trial、完整 summary和三份 phase receipt binding。
- runner 只接受该 typed assembly；它重新读取三份 receipt，核验 canonical bytes、checksum、共同 contract、job spec、phase result、plan、checkpoint、device 与 resource binding。
- 任意进程内 mapping 不能直接投影为正式 `PublicEvalResult`。

### 2.4 单一终局合同

- 当前 round-session `StairFinalConfirmationPlan/Reservation/Result` 保持原样。
- 不在 P3 创建较弱的 generation final-confirmation 克隆。
- 待 P4/P5 的 generation session、continuation chain 和 replay audit 真实存在后，只在现有 task-owned final-confirmation 模块中做版本化扩展，并复用既有 `single_use`、同 cohort 配对顺序、paired delta/mean/SE、diagnostic-only 和 `acceptance_authority=false` 冻结块。

## 3. 实施步骤

### S1：任务无关 phase 合同

- 固定 `EvaluationPhase`、`EvaluationPhaseResult`、`EvalPhaseJobSpec`。
- phase job 引用完整 `EvalJobSpec`，不得另建缩小版 eval protocol。
- phase result 显式保存 execution device 和 P1 resource binding hash。

### S2：runner、backend 与 receipt

- backend 从完整 plan 派生单 split argv。
- runner 复用 P1 attempt、GPU lease、owned process、checkpoint、环境和身份校验。
- `TrialReceiptJournal` 为三个 phase 发布/加载 create-once receipt。
- phase-result/spec 绑定只保留一份 task-neutral validator，runner 与 receipt loader共同调用。

### S3：Stair 组装与 selector

- 单体完整 eval 和三 phase 聚合共用同一个 Stair summary builder。
- selector 使用 objective 排序、parent guard 硬上限预筛和噪声带标签。
- formal assembler 重放 selection，并产生 receipt-bound `EvaluationAssembly`。

### S4：成本合同

- 科学路径上限：`2×public_dev(3) + 2×public_accept(5) + 1×sealed(7) = 23 seed-units = 6,440 episodes`。
- retry physical cost 继续由 P1 attempt receipts 独立累计，不混入 23 seed-units。
- 不在没有真机等价证据时实施 tread-horizon 优化。

## 4. 严格验收

1. 单体三 split 与三 phase assembly 的 public-dev、public-accept、sealed summary/evidence/gate 和 public result科学内容一致。
2. 有效微小 objective 差必须选出 winner；噪声带只改变标签。
3. 明显越过 parent guard 硬回退上限的候选不得成为 winner；完全并列与零 guard-eligible 路径明确 no-selection。
4. public-accept/sealed phase不能伪装成 selector 输入；loser不能组装 formal result。
5. raw mapping 不能进入 assembled projection；selection、receipt checksum、job spec、phase result、plan 或 checkpoint 篡改均拒绝。
6. 跨 `cuda:0/cuda:1` 比较时 device/resource 差异在证据中可见，但不会误判为科学域不一致。
7. width=1、Push/Neutral、round-v7、ledger replay、旧完整 eval 和现有 final-confirmation 全部不退化。
8. 运行 P3 聚焦回归、相关兼容回归、完整测试、编译检查和 `git diff --check`；不以脆弱的 import 名称子串扫描代替类型与行为测试。

## 5. 完成边界

P3 完成只表示“后续 P4 有可信的 phase/selector/assembly 基座”，不表示双卡 coordinator 已接线，也不表示真实双卡效率或 Isaac phase 等价已验证。未经用户验收，不进入 P4，不同步服务器，不启动实验。
