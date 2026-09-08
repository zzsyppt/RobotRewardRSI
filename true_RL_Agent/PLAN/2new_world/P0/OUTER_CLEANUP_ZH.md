# 本地外层与服务器扩围清理

更新：2026-09-06。**状态：本地与服务器扩围清理均已完成。** 本记录承接用户在首批 P0 完成后的扩围授权，不把待删除清单算作实际成果，也不与首批 P0 数字重复累计。

## 1. 已确认边界

- 保留 `D:\humanoid_agent` 的 Git 根和 `true_RL_Agent` 现有路径；本地退役旧 MVP，保留必要环境、代理工具与当前项目。
- `p6d-experiment-demo`、`codex` 为独立项目，研究资料、录像迁移材料及仍有消费者的产物保留。
- **新系统完整 15 rounds 实验涉及的全部 checkpoint 保留。此次扩围没有删除任何 checkpoint**，也没有修改正式协议、ledger、接受结果或训练/评测逻辑。
- 服务器只处理 `/data/yuyangliu` 下确认闲置、没有独有工作且可重建的副本，以及已核实的下载/传输缓存。环境、依赖源码、secrets、在用材料及独有未提交工作继续保留。

## 2. 服务器实际删除

| 类别 | 实际文件数 | 文件逻辑字节 |
|---|---:|---:|
| 7 个旧 Git 工作树 | 7,165 | 367,203,665 B |
| 1 个重复旧克隆 | 1,076 | 172,857,125 B |
| Sim 6 验证环境下载包 | 138 | 6,700,389,176 B |
| 已有 Git 历史覆盖的 5 个传输包 | 5 | 79,934,810 B |
| **合计** | **8,384** | **7,320,384,776 B，约 7.32 GB** |

这是 14 个明确目标的文件逻辑字节，不推算文件系统块或共享 Git 对象的额外回收量。目录已经实际移除，没有搬到另一个 archive。

精确路径与执行证据：

- [删除前逐文件清单](outer_remote_cleanup_preflight.json)、[引用与正式证据基线](outer_remote_reference_audit.json)
- [工作树执行结果](outer_remote_worktree_result.json)、[缓存与克隆执行结果](outer_remote_cache_clone_result.json)
- [传输包逐项审查](outer_remote_archive_audit.json)、[传输包执行结果](outer_remote_archive_result.json)
- [合并删除结果](outer_remote_cleanup_result.json)、[最终删除后检查](outer_remote_postcheck.json)

工作树使用 `git worktree remove`，未使用强制选项；删除前均无 tracked/untracked 修改，ignored 内容只有可重建缓存，提交由主仓库引用保留。重复克隆的全部 Git refs 也有主仓库引用覆盖。下载包逐个核实为普通 `.whl` 文件后 unlink，再删除空目录；没有触及已安装环境。传输包仅在全部所需内容可由主仓库 Git 历史恢复时删除。

## 3. 服务器扩围当时的保护与验收结果

- 在主项目 `outputs` 与 `true_rl_agent_phase_g` 登记的 **5,176 个 checkpoint、40,318,483,640 B**，删除前后路径、字节和 `mtime_ns` 全部一致；该数字不是服务器所有位置的 checkpoint 总数。
- 当前 `formal_go2_stair_15round_20260901_c6ab8ba` 下 **801 份 JSON/JSONL** 的 SHA256 全部一致，覆盖本轮报告、ledger、receipt 和结果身份。checkpoint 本体此次比较的是文件清单、大小与修改时间，没有声称重算全部权重哈希。
- 扫描 6,605 份实际产物文本，未发现这 8 个旧工作副本的路径引用；安装环境的 `.pth` 与 editable 入口也未引用删除目标。
- 正式环境和录像环境均成功导入 Torch/RSL-RL，Isaac Sim/Lab 的版本和实际模块位置检查通过。两套环境分别仍为 Torch 2.7 / Sim 5.1 与 Torch 2.11 / Sim 6.0.1 路线；没有启动仿真、训练、评测或 provider 请求。
- 服务器主仓库 HEAD 仍为 `0773a9a815e5a0ccd151f351fca0e369f157fa05`，tracked-clean，检查时 GPU 无计算进程。主 checkout 的源码没有在本批中部署更新；直接删除其受跟踪旧源码会改变正式运行要求的源码状态，本批没有改动该检查规则。

保留的特殊项目有具体理由：

| 保留项 | 理由 |
|---|---|
| `AGENT_AUDIT_STAIR_V3_20260807`、`AGENT_WORKTREES/true_rl_agent_phase_d_404a66d`、`true_rl_agent_eval_profile_audit_20260810` | 存在未提交或独有审计内容，不当作普通重复副本删除 |
| `AGENT_PHASE_D_FC84190` | HEAD 尚无主仓库其他引用覆盖 |
| `AGENT_WORKTREES/p6d_video_c582eca` | 录像材料关联的源码副本 |
| `phase_d_e24f899_full.bundle` | 并非所有 bundle refs 都有主仓库引用覆盖 |
| `p3a_server_pre_sync_backup_20260827T164341.tar.gz` | 实际含 4 个 checkpoint 和独有诊断材料 |
| `phase3a_sync_20260827.tar.gz` | 有 21 个文件内容未被主仓库可达 Git 对象覆盖 |
| `envs`、`vendor`、`tools`、必要运行时与 `isolated` 源码/结果 | 当前环境、资产和后端迁移验证仍有消费者 |

保留的三个传输/备份包在清理后 SHA256 不变。没有因名称包含“旧”“backup”或“phase”就直接删除。

## 4. 本地扩围清理已完成

[精确清单](outer_local_cleanup_manifest.json)中的 **81 个目标、912 个文件、28,653,612,487 B** 已实际删除，包括旧 MVP 源码、配置、样例、阶段脚本、日志和两个 Isaac 安装下载目录。清单没有 checkpoint；执行前检查绝对路径、reparse point 和普通文件范围，使用原生 PowerShell 的 `Remove-Item -LiteralPath`。

[执行回执](outer_local_cleanup_result.json)为 PASS，完成时间为北京时间 2026-09-06 00:09:46。[删除后全量回归](outer_local_tests_result.json)实际运行 **1606 项、0 failure/error、30 skip、exit 0，304.724 s**，不是借用首批测试结果。此前工具的执行前拒绝已是历史，不再是当前阻塞。

[最新状态核对](outer_local_postcheck.json)按实物更新。根 README 与环境说明已对齐当前项目；旧 MVP 不再作为可运行路径。一次性 `RUN_LOCAL_CLEANUP.ps1` 已完成使命，在下述收尾中删除；原始清单、执行回执和测试日志保留，不再保留一个无法重跑原清单的脚手架入口。

## 5. 2026-09-06 审计收尾

用户提出的审计建议逐条按当前消费者和主线边界复核。本批新增删除 **34 个目标、258 个文件、204,983,351 B**，未删除 checkpoint。准确路径、删除前 SHA256、结果与回归见 [收尾回执](audit_followup_result.json)。

| 审计项 | 实际处置 |
|---|---|
| active v6 / 20-step 描述 | 系统概览收敛为当前主线；Stair 指南从活动协议的 `episode_outcomes`、suite 和 acceptance 解释行为。同时移除把旧 35-cell 目标和旧 guard 写成现行规则的描述；没有改协议或运行时 |
| outputs 根级散件 | 删除 3 份无消费者的旧 follow-up 与测试遗留 `resolved_cell_manifest.json`；删除空的几何重训目录及 renderer 缓存父目录 |
| 已失效的父项目抽取地图 | 删除 `docs/reference_inventory.md` 和只检查这份地图文本的 2 项测试；实际运行依赖边界测试保留 |
| 同类旧实施脚手架 | 删除 `docs/superpowers` 中两份已被实现替代、仍指向已删 loop/test 的旧设计和执行清单；无现行消费者，保留实际代码与合同测试 |
| README 与 P0 状态不一致 | 根/项目 README、本文、ACCEPTANCE、问题索引及本地 postcheck 对齐真实完成状态；原执行记录不改写 |
| 权限残项 | 删除 4 条引用已删 MVP 源文件的 scp allow 项，另删 2 条引用已不存在 scratchpad 的临时 allow 项 |
| 构建与解释器缓存 | 删除源树 `true_rl_agent.egg-info`，以及实验/PLAN 内 13 个缓存目录、34 个 `.pyc`（765,353 B）；产品 `src/scripts/tests` 已无缓存。已安装 distribution 和环境内的 1 个缓存目录保留 |
| 旧录像 | 删除父 `artifacts/stair_checkpoint_videos`、项目 `outputs/stair_checkpoint_videos` 及 e24777b 中无消费者的 6 组旧导出；保留下表中有明确审计用途的 2 组 |
| 两份 prep bundle | 删除 `a576420`、`281bef3` bundle；前者由本地当前分支覆盖，后者已现场核实由服务器主仓库 `codex/phase-d-prepilot-281bef3` 引用覆盖 |

这不是把所有历史引用都视为保留理由，也没有把“不能解释 active Go2-V2”推导成“任何旧证据都无价值”。以下保留项有具体用途：

| 保留项 | 当前保留理由 |
|---|---|
| `outputs/current_38818b5_ledger.jsonl`、`outputs/formal_15round_821c961_evalv5_ledger.jsonl` | 分别是 1 条真实 v6、5 条真实 v5 决策的最小历史记录，合计 405,207 B；旧会话/重复副本已清理。这两份精确字节仍是 [历史协议基线](../../2-GPUS/P0/P0_BASELINE_AND_PROTOCOL_FREEZE_ZH.md) 的真实 ledger 样本，不作为新实验输入 |
| `outputs/formal_15round_b445125_evalv5_followups.md` 及对应会话 | 实际第 7 个根级文件并非孤儿：对应会话仍在，[Variant 审计](../../Variant_construction/STAIR_VARIANT_PLAN_AUDIT_ZH.md)使用其真实候选被拒绝的量级；不用于当前接受 |
| `outputs/stair_e24777b_diagnostic_videos/r1_stair035_env404347` 和 `r1_stair040_env485155` | [原始审计](../../stair/formal_15round_e24777b_followups.md)用其核对 0.35/0.40 m 的终态分类差异；仅保留这 14 个文件、6,314,239 B 的历史取证，不冒充原生录像或 unified V2 结果 |
| `.tmp/phase_d_e24f899_full.bundle` | 含当前本地及服务器主仓库引用尚未完全覆盖的对象；没有因另外两份 prep 包可删就连带删除 |
| 完整 15 rounds 的 checkpoint、正式 ledger/receipt、V1 predecessor、最小 validator、Sim 6 迁移材料 | 分别仍支撑复现、身份/lineage、冻结报告校验与 P2 后端对齐，未因本次文档清理改写或删除 |

全量回归实测 **1604 项、0 failure/error、30 skip，299.420 s**。这次回归还暴露第 5 处测试输出污染：`test_materialize_job_script.py` 的缺参测试先物化 composition，随后才报错，因此默认输出根又生成了同 SHA256 的占位 manifest。该文件中两项未指定输出目录的测试现已显式使用 `TemporaryDirectory`，原错误断言不变；修正后该模块 **6 项回归通过（0.476 s）**，项目根 manifest 未再生成。回归自产的 42,419 B 文件另行清除，不重复计入上述历史清理量。

全量日志记录的是发现最后一处 fixture 问题前的完整回归；6 项定向结果覆盖其后的测试修改，没有把前者冒称为最后修改后的再次全量运行。实际产品源码和任务配置保持字节不变，12 份当前文档的 144 个本地链接无悬空。外层原 790 个保护文件中，777 个不变、5 个文档按授权重写、1 个测试做输出隔离、7 个在本轮明确退役，详见 postcheck。

本次没有新增产品层、兼容包装或测试框架。原来的 1606 项测试仅退役上述抽取地图的 2 项文本断言；不修改产品合同断言。F29/F31/F61 的剩余功能建设仍按 [问题索引](ISSUE_DISPOSITION_ZH.md)承接，不能把文档/导出清理写成整个功能已经闭合。
