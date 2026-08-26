# Phase 0 验收报告：最新基线与二维边界冻结

> - 验收日期：2026-08-26
> - 结论：**GO——Phase 0 完成，允许进入 Phase 1**
> - 证据范围：基线、身份、边界与迁移债务冻结；不代表矩阵产品代码、Go2 Stair、复杂地形或新性能结果已经实现

## 1. 验收结论

Phase 0 已把当前系统固化为可复算的迁移起点：本地、origin 与服务器 checkout 对齐；G1-37 Stair、G1-29 Push 和 Go2 上游执行基础通过同一运行时 schema 记录；当前 Stair/Push 协议、reward ABI、35-cell 评测、产品 reference runner argv 与关键源码均进入机器快照；Core V1 范围固定为 3 个 cells / 7 个 variants；任务语义继续由 TaskPlugin 拥有，不进入 AgentCore 或机器人公共层。

本次审计修正了四项源头问题：运行时事实不再由自有 hash 自证，而是由服务器回执绑定探针源码并把 G1-37 与当前 Stair runtime protocol 交叉校验；三种 profile 均具有默认姿态、物理尺度、actuator/PD、传感器/噪声和完整资产闭包；legacy 35-cell 矩形考卷债务已经进入 Phase 3A 的前置验收；Phase 0 入口门改为产品内容身份，不再把 live Git HEAD 与捕获提交强制相等。Push 的两个旧“schema hash”被如实降格为 legacy label，另由回执提供内容绑定的 policy-I/O hash。

本阶段没有修改产品代码，没有拉取或复制 Go2 资产，没有升级 Isaac Lab，也没有提前实现 Phase 1。Go2 的结论是 `AVAILABLE_NO_PULL`：冻结的 Isaac Lab 2.3.0 已提供可读 USD、actuator、flat/rough config 与 RSL-RL agent，但本项目仍未支持 Go2 Stair。

## 2. 证据边界

| 对象 | 冻结身份 | 可以声明 | 不可以声明 |
|---|---|---|---|
| Phase 0 捕获仓库 | `f3593567b3901453f7dbf80dd6c450554b957fb2` | 捕获时本地/origin/服务器一致；仅作 provenance | 文档提交后 live HEAD 仍必须等于该 SHA |
| 当前产品代码 | `ab84aa3350700a2f7d46231afefaec4425ffa2ef` | `f359356` 相对它仅增加 P6-D 审计文档 | 文档提交带来新的运行能力 |
| 冻结正式实验 | `c582eca04dbac7d5ee5e2df0f56c646500ead55f` | 有一次真实 15-generation；公共结果 `4 accepted / 7 formal_rejected / 4 no_selection` | 把该结果归属到 `ab84aa3/f359356` 精确字节 |
| Phase 0 快照 | `66598edc3fe3ef43b999c624c4b4ce796355e77b7855ba5847dd02ab24424f3b` | 当前产品入口可由 26 个相关文件、resolved contracts、runner/reward/argv 与运行时回执复算 | Phase 1 的正式 RobotProfile API 已存在 |
| 运行时回执 | receipt=`5153841b0817015351ec40bb6efeef45c9492aa21782e396faa03e0398eafd5b`；file=`62c860f29c58e543a4c0d1192a718947d5f15941fa7193701647734fe1f63063` | `1e-6` 量化后同环境连续两次采集 byte-identical；探针源码 hash 与本地一致 | 把含捕获提交的 `receipt_hash` 当成跨提交 embodiment hash，或把回执称为 Phase 1 正式 manifest |

## 3. 严格验收结果

| 门 | 结果 | 验收证据 |
|---|---|---|
| P0-G1 捕获 provenance | PASS | 捕获时 local/origin/server 均为 `f359356`；快照字段已改名为 `captured_*`，验收器只检查三方捕获记录内部一致，不查询 live HEAD |
| P0-G2 产品入口身份 | PASS | 验收器复算 26 个相关产品文件、Stair/Push protocol、acceptance、seed、capability、binding、Genesis、reward ABI、35 cells 与 runner source/argv；`GIT_DIR` 指向不存在目录时仍 `status=pass`，证明 Git 元数据不是入口依赖 |
| P0-G3 物理身份 | PASS | 三种 profile 使用完全相同的字段集；内部 profile/binding/closure/receipt hash 全部复算，G1-37 又与 `stair_protocol_v1` 逐字段比对；一致重算所有自有 hash 的 joint+decimation 篡改被拒绝；两次量化探针文件 SHA 均为 `62c860f2...` |
| P0-G4 Go2 基础 | PASS | Go2 runtime=`12 joints / 19 bodies / 1 actuator group`；USD used-layer closure 为 `2 files / 18,810,329 bytes / 0 unresolved`，覆盖 8 KB payload wrapper 后的真实几何/物理层；官方 flat/rough config 与 PPO runner 均存在 |
| P0-G5 捕获事实边界 | PASS | `f359356` 捕获时仍无 Go2/Stair roughness 产品接入且无 fresh formal rerun；该结论作为绑定提交的历史事实记录，验收器不再执行 Phase 3 后必然变假的动态负向源码扫描 |
| P0-G6 范围与考卷债务 | PASS | 快照严格要求 3 cells / 7 variants；同时冻结 legacy 7×5=35 矩形考卷的两条结构不变量和 11 个受影响模块；Phase 3A 必须先提供 task-owned explicit-cell suite，Go2×Push 与六足仍为 extension |
| P0-G7 语义归属 | PASS | ownership map 冻结 AgentCore/TaskPlugin/RobotProfile+RuntimeAdapter/binding/runner 权责；楼梯成功、失败、地形、reward/eval/acceptance 留在 TaskPlugin |
| P0-G8 回归 | PASS | 本轮高信号集 `198 tests`，`0 failures / 0 errors / 6 skipped`；含正常快照与一致篡改否定测试，并覆盖 composition、Genesis、evaluation binding、Stair entrypoint/ABI/protocol/eval/evidence、Push task/runtime |
| P0-G9 清理与非目标保护 | PASS | 两次量化探针均 exit code=0；远端临时脚本/回执已删除，随后两张 RTX 5090 均 `6 MiB / 0%` 且 compute-app 为空；未跟踪项共 1530，其中 P0 为 10，排除 P0 后仍为原 1520；产品跟踪工作树无差异 |

六个本地跳过项中四个来自本机 `torch is unavailable`，两个是未开启 Isaac 动态集成环境。本轮变更不触及产品字节；相同产品代码已有
[P6-D follow-up 审计](../../2-GPUS/P6-D/P6D_POSTRUN_FOLLOWUP_AUDIT_ZH.md)记录的 Windows/Linux 全量
`1483 tests` 通过与 Linux Torch CPU 轴通过。本轮没有用该历史测试替代 Phase 0 特有验收：资产 checksum、真实
articulation schema、Go2 可用性和 probe 清理由本次重新核验。

## 4. 冻结的关键事实

### 4.1 机器人资产与执行身份

| 执行基础 | 运行时 schema | 完整资产闭包 | runtime facts hash |
|---|---|---|---|
| G1-37 Stair | 37 joints / 44 bodies / 3 actuator groups；root height `0.7400 m` | USD used layers：1 file / 22,195,512 bytes / 0 unresolved；closure `660130ec...` | `93adedbe9022a62d19fda14057775bd39e15705757d18287e494ce18855c16c2` |
| G1-29 Push | 29 / 35 / 4；root height `0.8000 m` | description tree：209 files / 115,354,479 bytes / 0 unresolved；closure `28742f44...` | `9038bdfdb8e5061adaa7bd2902fa94b65e124f73f76701f0085a155c2c678fbc` |
| Go2 上游基础 | 12 / 19 / 1；root height `0.4000 m` | USD used layers：2 files / 18,810,329 bytes / 0 unresolved；closure `c7fe9da2...` | `c56fed536354249bec90ce22fa52a10f8b899396758e46f8e03747368d985437` |

三份记录还统一覆盖默认 joint pose、全 body 相对位置、支撑尺度，以及逐 actuator/joint 的 effort、velocity、stiffness、damping、armature 和 friction。所有浮点数按 `1e-6` 分辨率量化，`-0.0` 归一为 `0.0`；原 294 个 morphology 数中的 16 个亚微米非零噪声现均为零。`runtime_facts_hash` 可用于相同 runtime/schema 的跨次比较；包含捕获提交 provenance 的 `receipt_hash` 只标识单次完整回执。它们不冒充 Phase 1 将实现的正式 `RobotProfile` schema/hash。

### 4.2 当前 Stair 固定语义

- reward ABI：`d10f3801634e8a77250c38d7a56176309d4061bcd352e469135a75f8dc0791c6`，42 个输入；其中 37-DoF、两只脚和 14 个手指输入是明确迁移债务；
- evaluation profile：35 个 legacy cells，hash `86d8eed0c8b9a8c32af49c69a81d983b3ec826abec408d5664717f61e16939d5`；
- acceptance protocol：`9fccce3984eed8f04602a07511d7afb79cedf8dc8732fe56b7a18d804ef26382`；
- normalized reference runner argv 快照：`c22cdf06920db9ac62b2c7611b150980db71ddc609bab6a045351de7932fbd13`；
- Stair/Push/Go2 upstream binding 的内容绑定 policy-I/O hash 分别为 `33c29649...`、`b9808edd...`、`e87ff59e...`；Push 后端旧常量只保留为 legacy label；
- legacy 35-cell 仍是回归 anchor；非规则 ≤18-cell 套卷必须在 Phase 3A 先经过显式 ordered-cell seam，不得继续依赖矩形切片。

## 5. 可复现验收

在 `D:\humanoid_agent\true_RL_Agent` 执行：

```powershell
.\.venv\Scripts\python.exe PLAN\Variant_construction\P0\verify_phase0_snapshot.py
```

另可运行最小的验收器自身测试：

```powershell
.\.venv\Scripts\python.exe -m unittest PLAN.Variant_construction.P0.test_verify_phase0_snapshot -v
```

这两条命令验证的是 Phase 0 的产品内容基线，而不是当前 Git SHA。提交或推送本目录文档后仍应通过；一旦 Phase 1 修改任一受保护产品源码、resolved contract、runner/reward/argv 身份，则按设计失败并由 Phase 1 迁移回执接管。捕获时的 `f359356` 继续保留为 provenance。远端 runtime 数据已经写入内容绑定回执，因此本地验收不会反复启动 Isaac 或 SSH。

## 6. GO 后的边界

Phase 1 可以开始实现最小 RobotProfile/RuntimeAdapter registry、RobotTaskBinding、ResolvedCellManifest 和 identity 绑定，但必须保持以下约束：

- 先迁移两个 legacy G1 execution profiles，不提前实现 Go2 Stair 或 roughness；
- TaskPlugin 只引用小型 semantic-role 词汇，具体 joint/link/asset 留在 robot runtime/binding；
- composition root 解析一次完整 manifest，现有 LOOP/runner/ledger 只消费解析结果；
- Phase 3A 在 C1 前由 Stair TaskPlugin 提供显式非矩形 cell suite，并保持 legacy V5 35-cell replay；
- 不新增第二套 LOOP、通用 TerrainPlugin、数据库、服务或全矩阵 scheduler。
