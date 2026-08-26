# Phase 0 开发日志

> 日期：2026-08-26  
> 范围：最新基线与二维边界冻结

## 1. 实施记录

1. 重读主计划、Variant construction 审计及 P6-C/P6-D 既有证据，确认总目标仍是现有 LLM LOOP 内的“机器人 × 任务”稀疏矩阵，不建设独立 variant 系统。
2. 同步核对本地、origin 与服务器：三者均为 `f3593567b3901453f7dbf80dd6c450554b957fb2`。把当前产品代码 `ab84aa3` 与历史正式实验 `c582eca` 分开记录，并修正主计划中可能混淆精确字节证据的表述。
3. 参考 `D:\humanoid_agent\codex` 提炼四个适用原则：显式稳定身份、分层输入只解析一次、专有语义归 owner、内容身份还需外部 authority 交叉校验。没有移植 Codex 的 sandbox、插件/服务状态或并发调度。
4. 从当前代码解析 Stair/Push composition 与全部核心协议 hash；通过产品 Stair reference composition 和公开 backend API 物化 train/eval argv，不再让验收工具依赖测试类私有 `_materialize`。
5. 在服务器冻结环境核对 Isaac Lab 2.3.0、Isaac Sim 5.1、Unitree RL Lab、G1-37/G1-29/Go2 资产与上游配置。Go2 基础完整，因此决定 `AVAILABLE_NO_PULL`。
6. 使用 headless 单场景实例化三种 articulation，并从同一运行中采回 joint/body、默认姿态、root/support 尺度、完整 actuator/PD 参数和资产闭包；同一探针还解析 Stair、Push、Go2 official rough 的控制时序、policy I/O、观测噪声和传感器 pattern。Go2 的闭包实际为 2 个 USD layers / 18,810,329 bytes，消除了 8 KB payload wrapper 不能代表真实资产的问题。
7. Isaac Sim 5.1 在回执写完后会因可选 Iray `libGLU.so.1` 缺失卡在 Kit close。本阶段没有安装系统依赖；探针先 `fsync` 回执，再让一次性自有进程直接退出。最终重跑 exit code=0，随后两张 GPU 均为 6 MiB / 0%，无探针进程。
8. 审计确认原 `execution_profiles` 仅验证自有 hash。现改为“回执内部完整复算 + 探针源码/上游提交/源码 hash 绑定 + G1-37 对 `stair_protocol_v1.policy_compatibility` 逐字段交叉校验”；一致替换关节名、同步改 actuator/default pose、把 decimation 改为 8 并重算所有 hash 的测试已被拒绝。
9. 三种 profile 使用相同字段集。Push 的 `push_recovery_obs_v1` / `unitree_actions_v1` 明确标成非内容绑定 legacy label，实际 policy I/O 由运行时回执另给 SHA-256；不在 Phase 0 破坏历史身份。
10. 独立扫描确认 legacy 35-cell 链路把评测集合写成 7×5 矩形网格，涉及 11 个 Stair 模块。没有在 Phase 0 批量重写；主计划改为 Phase 3A 先建设 task-owned explicit-cell suite，再运行 C1 非规则 ≤18-cell 套卷。
11. 删除验收器的未来负向源码扫描。`f359356` 上“尚无 Go2/roughness 接入”改为提交绑定的捕获事实。
12. 执行快照复算、验收器一致篡改测试和 12 个既有高信号模块，共 198 项通过、6 项因本机 Torch/Isaac 动态环境不可用而跳过、无失败；旧 Stair/Push 插件未退化。
13. 后续审计发现验收器把 live local/origin HEAD 强制等于捕获时 `f359356`，导致提交 P0 文档后自锁。根修采用产品内容身份：移除 live Git 查询，`repository` 字段改为 `captured_*` provenance；仍复算 26 个相关产品文件、resolved contracts、runner source、reward ABI 与 normalized reference argv。将 `GIT_DIR` 指向不存在目录时验收仍通过。
14. 复现 294 个 morphology 数中有 16 个非零亚微米 float32 噪声。探针 schema 升级后统一按 `1e-6` 量化并把 `-0.0` 归一为 `0.0`；量化不改变任何任务阈值、控制时序或执行器有效参数。
15. 在相同服务器/runtime/提交上连续运行两次完整三机器人探针，两份回执 byte-identical，文件 SHA-256 均为 `62c860f29c58e543a4c0d1192a718947d5f15941fa7193701647734fe1f63063`。跨 Phase 应比较 profile/binding/asset identity；包含捕获 provenance 的完整 `receipt_hash` 仍是单次回执身份。

## 2. 本阶段文件

| 文件 | 用途 |
|---|---|
| `PHASE0_IMPLEMENTATION_PLAN_ZH.md` | 实际工作包、非目标与验收门 |
| `PHASE0_BASELINE_SNAPSHOT.json` | 可复算的代码/协议/runner/环境/资产/执行体基线 |
| `capture_phase0_runtime_receipt.py` | 服务器一次性运行时事实采集器 |
| `PHASE0_RUNTIME_PROBE_RECEIPT.json` | 三种执行体与三份 binding 的内容绑定回执 |
| `verify_phase0_snapshot.py` | 本地只读冻结基线验收，不触发训练或远端运行 |
| `test_verify_phase0_snapshot.py` | 正常回执与一致篡改的两条最小验收器测试 |
| `OWNERSHIP_SCOPE_AND_HARDCODES_ZH.md` | 3×7 范围、权责表和迁移阻塞清单 |
| `PHASE0_ACCEPTANCE_REPORT_ZH.md` | 严格验收证据与 GO 结论 |
| `DEVELOPMENT_LOG_ZH.md` | 本日志 |

主计划 `STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md` 只更新最新基线、证据边界、Phase 0 状态与实施回执。没有修改产品源码、配置、测试、资产或旧审计文档。

## 3. 关键决策

- **不拉 Go2**：冻结 runtime 已有官方资产/config/agent，重复拉取会制造第二身份源。
- **不升级 Isaac Lab**：Phase 0 证明的是当前 2.3.0 基线，不把 3.0 beta 变化带入矩阵基础。
- **不修 `libGLU`**：它不阻止 articulation/asset 身份核验；为只读 probe 安装系统依赖会扩大本阶段范围。
- **不做产品抽象**：正式 RobotProfile/manifest 属于 Phase 1；Phase 0 runtime facts 只用于证据隔离与迁移输入。
- **不重复 formal 15-generation**：本阶段没有产品字节变化，且目标是边界冻结；历史结果继续绑定其原提交，不能用于声称当前精确字节复跑。
- **不增加安全框架**：只增加一个集中式、只读、确定性的快照验收器，替代分散的手工检查。
- **HEAD 只作 provenance**：文档提交不改变产品能力，也不应使 Phase 0 入口门失效；产品相关内容变化才触发失败。
- **量化只稳定身份**：`1e-6` 精度远小于地形、clearance、控制与执行器的有效尺度，不改变任务或评测语义。

## 4. Phase 1 入口

Phase 1 开始时应先运行：

```powershell
.\.venv\Scripts\python.exe PLAN\Variant_construction\P0\verify_phase0_snapshot.py
```

该命令可在提交/推送 Phase 0 文档后继续使用：live HEAD 不是输入，捕获时 `f359356` 仅作 provenance。开始有意修改受保护产品源码或契约后，它按设计失败，后续验收应引用 Phase 1 自己的迁移回执，不能不断改写 Phase 0 历史。Phase 1 的首个实现目标是让两个 legacy G1 profiles 通过同一个最小 compiler 解析到各自的完整 manifest，而不是先接 Go2 或复杂地形。
