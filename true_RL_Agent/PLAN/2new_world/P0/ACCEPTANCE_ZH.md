# P0 基线、清理与验收

日期：2026-09-05。起点：`0773a9a815e5a0ccd151f351fca0e369f157fa05`；工作分支：`true-rl-agent-refactor`。

**P0 总体结论：GO。** 首批清理、基线、问题处置与验收均已完成。本地全量回归及生产环境定向补测通过。以下数字均为实际删除量；本阶段没有启动训练、评测或 provider 请求，也没有推进 P1–P8。

**后续扩围状态：已完成，2026-09-06 更新。** 服务器另删除 8 个闲置工作副本、138 个下载包和 5 个传输包，共 7,320,384,776 B；本地旧 MVP 与下载缓存已实际删除 912 个文件、28,653,612,487 B，删除后 1606 项全量回归通过。审计收尾又删除 258 个文件、204,983,351 B，并修正当前协议说明、根 README 和本记录。扩围与本次收尾均未删除 checkpoint。最新逐项处置和回归结果见 [扩围清理记录](OUTER_CLEANUP_ZH.md)及[收尾回执](audit_followup_result.json)。

收尾全量回归 1604 项通过（30 skip）；发现最后一处测试输出污染后，仅修改该测试模块的临时目录归属，6 项定向回归通过且根级占位文件不再生成。没有增加测试框架或修改产品源码、协议、checkpoint 与正式结果。

## 1. 实际完成的工作

执行范围见 [实施计划](IMPLEMENTATION_PLAN_ZH.md)，原始 55 项问题只维护 [编号处置索引](ISSUE_DISPOSITION_ZH.md)，不重复复制问题正文。

- 删除 replay-only 的 `agent_core/loop.py`、`agent_core/session.py` 及 8 个无当前消费者的一次性脚本：**10 个产品文件、1948 行、72,185 B**。准确路径和删除前 SHA256 见 [源码删除结果](source_cleanup_result.json)。
- 旧 `test_agent_core_loop.py` 的两项有效 sanitizer / missing-runtime-activation 测试迁移到 `tests/integration/test_round_public_boundaries.py`；三项只保护旧包装器的测试退役。原有 live/Genesis/generation/replay 边界继续保留，没有增加兼容包装。
- README、当前 runbook 和旧概览导航指向实际正式 launcher；文档覆盖矩阵与边界扫描同步修改。P1 三份文档的 Go2 × Push 责任统一为 **Extension A**，证据数值不变。
- 发现 4 处测试会向项目 `outputs` 写占位 manifest，已使用 `TemporaryDirectory` 和 unittest cleanup 收回其生命周期；Phase-B 测试复用已有临时目录。原断言和产品逻辑不变，避免每次回归重新污染项目输出。
- F30/F33 已退役，F45 文档修复；F29/F31/F44 仅关闭本阶段明确完成的部分，其他工作继续按问题索引承接。

## 2. 首批 P0 删除量与当时边界

| 类别 | 已执行结果 | 保留边界 / 证据 |
|---|---|---|
| 本地旧审计、干运行副本、缓存和占位物 | 167 个目标，2445 个文件，227,470,405 B | [精确原清单](local_deletion_manifest.json)、[执行前核查](local_cleanup_preflight.json)、[删除结果](local_cleanup_result.json) |
| 本地补充缓存、测试输出和空目录 | 33 个目标（含清空后的 .codex_tmp 父目录），846 个文件，21,534,413 B | [补充删除结果](local_cache_fixture_cleanup_result.json)；不包含虚拟环境、已安装依赖或研究材料 |
| **本地临时产物合计** | **200 个目标，3291 个文件，249,004,818 B** | 全部已核实不存在；源码删除量单独计算 |
| 服务器历史中间 `model_*.pt` | **892 个文件，6,632,805,920 B** | 原 1321 个模型中保留 429 个；[执行前核查](remote_cleanup_preflight.json)、[删除结果](remote_cleanup_result.json) |
| 旧产品与测试文件 | 10 个产品文件 + 1 个旧包装器测试文件，合计删除 84,785 B | 已计入有效测试迁移；不能用毛删除量代替源码净变化 |
| 早前补测自身的临时 fixture | 642 个文件，15,407,214 B | 两个失败审计调用产生的自有 `/tmp` 目录已清理；[结果](test_fixture_cleanup.json)，不冒充历史项目清理量 |

首批本地与服务器临时产物合计删除 **6,881,810,738 B，约 6.88 GB**。数字为文件字节数，不等于磁盘分配簇。首批收尾时 `src/scripts/tests` 的 Python 文件为 **423 个 / 9,293,898 B**，比起点净减少 10 个文件、80,471 B；已计入新测试及目录隔离修正。后续扩围与审计收尾单独计数，见本文顶部的最新状态。

本地使用同一 PowerShell 的 `Microsoft.PowerShell.Management\Remove-Item -LiteralPath`：先验证绝对路径属于 `D:\humanoid_agent\true_RL_Agent`、根与祖先及后代无 reparse point、没有 tracked 文件、内容和数量符合清单、没有活动项目进程，再删除并验证目标消失。补充缓存仅允许 `.pyc`；测试目录仅允许已经定位到写入方的 manifest 或空目录。

服务器只对 `/data/yuyangliu/AGENT/true_RL_Agent/outputs` 下清单中的普通模型文件执行 `Path.unlink`，未递归删除实验目录。执行前重新检查 3688 个相关文本文件、实际作业引用、文件大小和 mtime、源码状态与 GPU/进程；发现引用为 0。删除前后 **3214 个受保护文件、3,761,552,290 B** 哈希完全相同。

原 901 个 checkpoint 候选中的 9 个来自旧 `d444f5b` 正式失败作业，本次改为保留其失败实物证据。因此实际删除为 892，不能写成 901。当前 `c6ab8ba` 正式输出全体保留。历史成本 receipt 和 accounting 不改写；F53 的失败作业 retention 仍由 P8 收敛。

必要保留项和原始引用见 [本地清单](local_inventory.json)、[服务器基线清单](remote_inventory.json)；当前执行状态以 [artifact_cleanup_status.json](artifact_cleanup_status.json) 为准。服务器清单是删除前快照，实际处置以删除结果为准。

首批保留边界如下；后续已获授权的父层缓存、旧 MVP 和无消费者录像清理，以扩围记录为准：

- 父仓库 `.tmp`、其他项目、Git 分支、虚拟环境、已安装依赖及 `/data/yuyangliu/isolated` 均在本次范围外。
- 用户原有的 `PLAN/Variant_construction/P6/PHASE6_FOLLOWUP_OBSERVATIONS_ZH.md`、P5 几何研究材料、录像与后端迁移材料。
- 所有仍被 receipt、报告、诊断、恢复流程或外部 anchor 使用的 reference、best、末态/对比模型。
- V2 依赖的 V1 protocol/binding/lineage，仍有消费者的 manifest validator、pilot 依赖链及 legacy transition normalizer。其后续删除须先收回真实调用者。
- F56 原来未产生的 planned URI 不伪造为文件，也不计作本次删除产生的悬空。

此前两次本地删除曾在执行前返回 `blocked by policy`。本次用户明确要求继续并复核范围后，重新执行同一原生 PowerShell 路线成功；此前拒绝是历史记录，已不是当前清理阻塞。

## 3. 活动格子与权威证据

当前导航入口为项目 [README](../../../README.md)。配置解析及完整 anchor 见 [active_cells.json](active_cells.json)、[reference_anchors.json](reference_anchors.json)、[remote_reference_verification.json](remote_reference_verification.json)。

| Cell | 当前 reference / 能力 | ledger 与 head |
|---|---|---|
| `stair.g1_37dof.unified_v2` | P5 三相重资格化已留证 | P5 reference Genesis 没有 candidate ledger；旧 Phase4 G1-V1 head 不能作为 V2 新结果 |
| `stair.go2_12dof.unified_v2` | P5 reference 导入当前正式会话 | `c6ab8ba` 15 条 generation ledger，唯一接受 Gen8-A |
| `push.g1_29dof.legacy_v1` | legacy manifest 可解析 | 尚无 reference-program Genesis / generation session；Go2 Push 尚未闭合 |

以下路径相对于服务器项目根：

| 实物 | 位置 |
|---|---|
| G1-V2 reference Genesis | `outputs/p5_g1_pseudofall_removed_genesis_0a3f08a/session_genesis.json` |
| G1 reference checkpoint | `outputs/phase4_unified_validation_5a30129_20260829/runner/phase4-unified-validation-5a30129-20260829/checkpoints/best.pt` |
| Go2-V2 reference Genesis | `outputs/p5_go2_pseudofall_removed_genesis_0a3f08a/session_genesis.json` |
| Go2 reference checkpoint | `outputs/p5_go2_derived_reference_ef6826e/runner/p5-go2-derived-reference-ef6826e/checkpoints/best.pt` |
| 当前正式 session | `outputs/formal_go2_stair_15round_20260901_c6ab8ba` |
| 正式 ledger / report | 上述目录的 `ledger.jsonl` / `generation_session_report.json` |
| Gen8-A checkpoint | 上述目录的 `runner/formal-go2-stair-15round-20260901-c6ab8ba-generation-008-lane-a/checkpoints/best.pt` |
| Final 配对结果 | 上述目录的 `final_confirmation/result.json`；原始 summary 在 `runner/final_confirmation/{genesis,final}/eval/sealed_validation/summary.json` |
| 独立终态 replay 历史证据 | `.tmp/formal_terminal_completion_audit_17e819d.json` |

核实的核心身份：

- G1 reference checkpoint：`3fbbf80d5e7e4ee559420ea06759231c00e2d90316101ace4db6e24ec25810b2`。
- Go2 reference checkpoint：`fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`。
- Gen8-A checkpoint：`bcd8de3a9d0e6e6a6c84b8c7833dc4a1b0ed697758ddee6f9c9eb31416a2f93c`；candidate：`stair_swing_foot_level_sweep_penalty_v1`。
- accepted head：`7c2756aeb0ea21d1db378779d4db5fb3bade4081bd2e016302d889f750448eb9`。
- ledger：`052a71e7c438f80789fb22a2fcdf5b921dd864ed4c2cd35019bb82097f900845`；report：`4ad1e53cc72afa0a9fe20ab69a8b5bc41e4ed9249efda4a7958477eb0fc150ca`。
- final Result：`046582e2665e9c301cc88e6cc336d40f8e7bc85637d49f4a801006bc06e99b42`。

正式报告保存的 11 项 checks 为真。本次核实已有真实产物及身份，没有重新执行真实 15 代完整终态 replay，也没有把历史 replay 当作 P0 新测结果。

## 4. 验收结果与环境边界

| 检查 | 当前结果 |
|---|---|
| 本地清理前全量基线 | 1609 项，0 failure/error，30 skip，276.807 s；[基线](local_baseline_tests_result.json) |
| 临时产物删除后全量 | 1606 项，0 failure/error，30 skip，310.967 s；[结果](local_after_artifact_cleanup_tests_result.json) |
| 4 处测试输出隔离回归 | 88 项，0 failure/error/skip，11.074 s；实际删除后重跑，0 个目标重新生成；[结果](fixture_output_isolation_tests_result.json) |
| 最终完整本地回归 | 1606 项，0 failure/error，30 skip，276.919 s；[结果](local_final_tests_result.json)、[日志](local_final_tests.log) |
| 生产环境定向单测 | 26 项，0 failure/error/skip；[结果](production_environment_unit_tests_result.json) |
| 生产环境长集成 | 2 项，0 failure/error/skip，1385.282 s；15 代连续运行/终态确认及三代暂停/崩溃恢复均完整通过；[结果](production_environment_long_tests_result.json)、[日志](production_environment_long_tests.log) |
| 静态与内容保护 | 423 个 Python 文件 AST 通过；749 个原有 tracked 文件字节不变；全部配置、剩余产品源码和用户原有修改保持原样；[检查](closure_check.json) |
| 引用与问题索引 | 当前文档链接无悬空；原始 55 个问题编号与处置表集合一致；旧模块无遗留 import |
| 服务器产物保护 | 删除前后 3214 个受保护文件哈希不变；892 个已删除模型均不存在 |
| CLI / 环境可执行性 | 本地 4 个、生产 5 个、录像目标环境 5 个 CLI help 通过；[本地](local_entrypoints.json)、[生产事实](remote_baseline_facts.json)、[录像目标](video_target_entrypoints.json) |
| 原生 Isaac 动态检查 | 2 项显式 opt-in 未启动；P0 未修改其运行时，不能替代 P2 物理/训练/录像验收 |

完整回归使用标准 `python -B -m unittest discover -s tests -t . -v`，测试进程继承 `PYTHONDONTWRITEBYTECODE=1`。补测只在测试子进程设置 `OMP_NUM_THREADS=1 / MKL_NUM_THREADS=1`，没有更改生产默认环境。

30 个本地 skip 中，26 项依赖 Linux/procfs/信号、符号链接权限或 Torch/NumPy 的检查已经在服务器补过；两项长集成也已取得完整外层断言结果，合计 28 项已补齐；仅两项原生 Isaac opt-in 留给涉及运行时变更的 P2。长集成使用 fake runner 证明编排与恢复，不产生新的科学实验结论。

早前两次长集成审计给整个集合 600 s，但 fixture 的 `final-crash` 单阶段允许 900 s，另有 finish 等阶段，因此两次调用均未闭合。其结果保留在 `production_environment_tests*.json/.log`，不计 PASS。本次外层给足 3600 s，保持 fixture 原有阶段期限与所有断言。本次两个完整 unittest 均通过；worker 自报 GO 没有被用来替代外层断言。

首次源码清理后出现的两处文档/边界测试联动遗漏已修复并全量重跑；历史失败见 [local_test_retry.json](local_test_retry.json)。未以删断言或降低条件消除失败。

| 环境 | 已核实事实 | 本阶段用途 |
|---|---|---|
| 本地 | Python 3.12.14，无 Torch/NumPy/Isaac | 完整轻量回归 |
| 当前生产 | Python 3.11.15；Sim 5.1.0.0；isaaclab distribution 0.47.2；isaaclab-rl 0.4.4；RSL-RL 3.0.1；Torch 2.7.0+cu128；Gymnasium 1.2.0 | 环境定向补测 |
| 录像迁移目标 | Python 3.12.14；Sim 6.0.1.0；isaaclab distribution 6.1.14；isaaclab-rl 0.5.5；RSL-RL 5.0.1；Torch 2.11.0+cu128；Gymnasium 1.2.1 | CLI 可执行性核查 |

录像目标源码为 `/data/yuyangliu/isolated/isaac_sim_6_validation/IsaacLab-v3.0.0-beta2.patch1`，源码 `VERSION=3.0.0`；发布/源码版本和 Python distribution version 分开记录。此处不宣称生产后端迁移完成。

源码清理和测试修正在本地工作区，服务器源码仍为起点提交，未部署本地改动。服务器长集成验证的是与本地字节相同的未修改 runtime/fixture；不将它冒称为本地改动部署后的全量服务器回归。

## 5. 后续预算与阶段出口

P0 训练、评测、provider 调用均为 **0**。普通 unittest、哈希和 CLI help 不算科学实验。后续预算按池分别冻结，未知数不填零：

| 预算池 | 承接 Phase | 冻结内容 |
|---|---|---|
| 迁移对照 | P2 | ABI/一步/短程、完整训练对照、eval/video 对齐与失败收尾 |
| calibration | P3 | 公开题池、机器人尺度、难度与指标区分力，不消费 final 留出 |
| reference | P2、P6 | 每个 cell 的训练 seed、随机初始化完整预算及三相资格化 |
| 候选搜索 | P7 | generations × lanes、修复/重试次数、train/eval 拓扑及 provider 用量 |
| 复现 | P7 | reference/候选的配对多训练 seed 和必要域外验证 |
| final | P7 | 独立题池/seed、单次终态确认，不能挪用给搜索或 calibration |

每池按实际单作业成本冻结 GPU 小时、control steps、API 用量/费用、磁盘和墙钟上限以及停止规则。

**P0 出口已满足：** 删除清单、前后数量、保护哈希、本地全量回归、28 项生产环境补测、CLI 与问题索引均闭合。最终未发现项目测试/train/eval 残留进程，GPU compute 列表为空；两项长集成的临时目录均自动清理，见 [最终运行状态](final_runtime_status.json)。当前文档链接核对通过。F44 的自动 retention、F53 的正式失败模型处置、仍有消费者的旧生成栈按 P8 承接；这些明确保留项不是本阶段“已经删除”的内容。
