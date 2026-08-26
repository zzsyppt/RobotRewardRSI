# Phase 0 实际实施计划：最新基线与二维边界冻结

> - 实施日期：2026-08-26
> - 适用仓库：`D:\humanoid_agent\true_RL_Agent`
> - 当前分支：`true-rl-agent-refactor`
> - 本阶段性质：基线冻结与边界验收，不实现 Phase 1 的矩阵契约

## 1. 本阶段要解决的问题

总目标仍是让现有自进化系统形成“机器人 × 任务”二维矩阵：每个有效单元格解析成自己的机器人、任务、环境、协议与策略谱系，之后进入同一套 LLM LOOP。Phase 0 不新增机器人适配代码，而是先把迁移前的事实定牢，防止后续把下列不同概念混在一起：

- Isaac Lab G1-37DoF Stair 与 `unitree_rl_lab` G1-29DoF Push 是两个不同执行体；
- Stair 的上游类名含 `rough`，不等于当前楼梯踏面已实现 roughness；
- Go2 上游资产与 agent 基础存在，不等于本项目已经支持 Go2 Stair；
- 冻结提交 `c582eca` 上的正式 15-generation 实验存在，不等于修复后的当前产品字节已重新完成同样实验。

Phase 0 完成后，Phase 1 只能在已经冻结的事实和权责边界上建设，不再重新讨论系统是否需要第二套 LOOP、通用 TerrainPlugin 或全矩阵调度器。

## 2. 实施边界

### 2.1 本次允许变更

- 修正主计划中已经过时的同步提交与 P6-D 证据表述；
- 在本目录生成机器可读基线快照、边界/硬编码清单、验收报告和开发日志；
- 增加一个只读的 Phase 0 快照验收脚本；
- 执行本地测试和服务器只读资产/runtime probe。

### 2.2 本次明确不做

- 不新增 `RobotProfile`、`RuntimeAdapter`、`RobotTaskBinding` 或 `ResolvedCellManifest` 产品代码；
- 不修改 AgentCore、runner、ledger、decision、LLM provider、Stair/Push 任务语义或训练预算；
- 不实现 roughness、irregular Stair、Go2 Stair 或任何新 LOOP；
- 不升级 Isaac Lab / Isaac Sim，不安装缺失的 `libGLU`，不拉取已经存在的 Go2；
- 不新增数据库、服务、调度器、通用 TerrainPlugin 或额外安全框架。

## 3. 采用的工程原则

从 `D:\humanoid_agent\codex` 当前提交 `5ce04789fbbd4b12c80340a8246cfb00d0a330d5` 只采用以下与本项目直接相关的思想：

1. 稳定、显式的身份优于 family alias；后续 cell/profile/capability 使用稳定 ID，不能用 `unitree_g1` 代替物理兼容性。
2. robot、task、binding、environment 分层输入在 composition root 解析一次，LOOP 只消费唯一 resolved object。
3. 专有语义由拥有者提供；共享层只处理带身份的契约，不解释楼梯、脚、roughness 或 Push。
4. 内容 hash 只证明内部一致；新采运行时事实必须再与本地 runtime protocol、冻结上游版本和源码身份交叉校验。
5. 只保留会阻止真实回归的测试，不为静态常量和已删除逻辑增加防御层。

不采用 Codex 的沙箱、进程权限、远程插件、并发调度和服务端状态系统，因为它们不能帮助本阶段证明二维边界，反而会扩大系统复杂度。

## 4. 工作包与产物

### P0-A：同步事实与证据分层

记录捕获时本地 HEAD、origin、服务器 checkout 与工作树状态，以及“当前产品提交”和“冻结正式实验提交”两条独立 provenance。Git SHA 不作为文档提交后的入口门；入口门由相关产品源码、resolved contracts、runner source、reward ABI 与物化 argv 的内容身份组成。已有未跟踪产物只读留存；主计划是用户指定的唯一既有编辑目标。

产物：主计划基线修正、快照中的 `repository` 与 `evidence_boundary`。

### P0-B：本地组合与协议快照

从当前产品代码解析 Stair/Push composition，记录 Genesis snapshot、task/eval/acceptance/seed/capability hash、Stair reward ABI、35-cell profile、训练预算与 runner argv builder/entrypoint hash。runner argv 直接通过产品的 Stair reference composition 与公开 `materialize_reference_jobs` 生成，不依赖测试 fixture 私有方法。所有值由代码解析，不从旧报告抄录。

产物：`PHASE0_BASELINE_SNAPSHOT.json`、`verify_phase0_snapshot.py`。

### P0-C：服务器机器人与环境核验

在冻结服务器环境中核验 Isaac Lab 版本/source commit、Unitree 包来源，并用单场景 runtime probe 对三种 articulation 统一采集 joint/body、默认姿态、根高/支撑尺度、actuator/PD 参数和完整资产闭包；同时冻结 Stair、Push 与 Go2 official rough 基础的控制时序、policy I/O、观测噪声和传感器 pattern。所有浮点事实按 `1e-6` 分辨率量化并将 `-0.0` 归一为 `0.0`；同环境连续两次探针必须得到 byte-identical 回执。Go2 同时要求 official rough/flat config 与 RSL-RL agent class 存在。

产物：`capture_phase0_runtime_receipt.py`、`PHASE0_RUNTIME_PROBE_RECEIPT.json`，以及快照中的 `server_environment`、`runtime_probe_receipt`、`go2_availability`。

### P0-D：二维范围、权责与迁移债务冻结

固定 3 个 execution-level cells、7 个 cell variants，形成 AgentCore / TaskPlugin / RobotProfile+RuntimeAdapter / RobotTaskBinding / runner 权责表；既列出会阻止 Go2/六足复用的 G1-37/双足耦合，也列出会在 Phase 3A 阻塞非规则 ≤18-cell 套卷的 legacy 35-cell 矩形网格耦合。Phase 0 只冻结债务，不提前实现 VNext suite。

产物：`OWNERSHIP_SCOPE_AND_HARDCODES_ZH.md`。

### P0-E：严格验收与收尾

运行快照复算、Stair/Push composition 与高信号任务测试；复核本地/服务器提交、未跟踪产物、远端进程和 GPU。发现问题只修本阶段的事实源或文档，不提前修改 Phase 1 产品架构。

产物：`PHASE0_ACCEPTANCE_REPORT_ZH.md`、`DEVELOPMENT_LOG_ZH.md`。

## 5. 严格验收门

| 门 | 通过条件 | 失败处理 |
|---|---|---|
| P0-G1 捕获 provenance | 捕获时本地 HEAD、origin、服务器 checkout 同为冻结提交；这些 SHA 只作来源记录，不与提交后的 live HEAD 比较 | 捕获 provenance 不一致则先查明分叉 |
| P0-G2 产品入口身份 | 快照验收器可从当前工作树复算 26 个相关产品文件、Stair/Push 核心 contract、reward ABI、runner source 与产品 reference argv；不 import 测试类，且在 Git 元数据不可用时仍能通过 | 修正产品事实源；不得手改 hash 规避 |
| P0-G3 物理身份 | 三种执行体使用相同字段集记录完整 runtime facts/资产闭包；G1-37 还须与 Stair runtime protocol 逐字段交叉校验；一致重算所有自有 hash 的篡改仍被拒绝；`1e-6` 量化后的两次独立探针 byte-identical | 不得进入 Phase 1 profile 迁移 |
| P0-G4 Go2 基础 | Isaac Lab 2.3.0 冻结源码中 asset/config/actuator/agent 均存在，Go2 USD 实际可读 | 缺失则记录 Phase 5 前补齐或 NO-GO，不暗升 runtime |
| P0-G5 捕获事实边界 | `f359356` 捕获时无 Go2/Stair roughness 产品接入，且当前 15-generation 证据边界准确；该事实绑定提交，不做未来必然失效的动态负向扫描 | 修正文档中的过度声明 |
| P0-G6 范围与考卷债务 | 恰好 3 cells、7 variants；六足和 Go2×Push 仍为 extension；同时完整记录 legacy 矩形评测拓扑的 11 个受影响模块与 Phase 3A 根治 seam | 删除本阶段扩张项或补全债务清单 |
| P0-G7 语义归属 | ownership map 明确任务语义不进入 AgentCore/robot 公共层 | 重新收敛边界，不增加补丁层 |
| P0-G8 回归 | 选定的 composition、Stair、Push、runner 高信号测试全部通过 | 只对真实回归做根因修复 |
| P0-G9 清理 | 本次 probe 无残留进程/GPU 占用；审计文档及其他非目标产物未被本次工作写入 | 精确清理本次进程并核对差异 |

Phase 0 的 GO 只表示迁移基线与边界已冻结，不表示矩阵、Go2 Stair、roughness 或新策略性能已经实现。
