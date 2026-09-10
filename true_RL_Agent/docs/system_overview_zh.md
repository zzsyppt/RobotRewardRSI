# true_RL_Agent 系统概览

核对日期：2026-09-10。本文说明当前主线的职责和证据入口；具体协议以活动 cell 解析出的配置为准，历史实验数值留在原始报告中。

## 1. 系统要解决什么问题

系统让 LLM 提出受控的 reward program 改动，再由可信系统验证、训练、评测和判决。机器人与任务是两个独立维度：机器人提供资产、形态和状态/动作事实，任务定义成功、失败、分布、指标与反馈。增加机器人不应把新机器人的关节名、体型阈值或任务规则写进共享编排层。

P3 在 9 月 9 日固定 C2 标定基底并重新测量后，`stair.g1_37dof.unified_v3` 与 `stair.go2_12dof.unified_v3` 均通过原 public-accept reference 资格门；剩余卷间差异和泛化限制见 [当前验收](../PLAN/2new_world/P3/ACCEPTANCE_ZH.md)。V2 行保留已有证据解释。Stair composition/reference 入口默认 G1 V3，共享正式入口要求显式 task/cell，成本合同按所选协议的实际工作量和测量来源冻结。`push.g1_29dof.reference_v1` 已接入 reference-program Genesis、三阶段评测和共享 session；这是 P4 的工程通路，真实 Go2 × Push Recovery 资格仍归 P6。旧 `push.g1_29dof.legacy_v1` 保留原协议。准确配置入口见 [README 活动格子表](../README.md)。

## 2. 受控改进闭环

```text
解析 robot × task cell，冻结训练、评测和候选权限
    ↓
从合格 reference 建立或导入 Genesis
    ↓
从已接受源码及允许公开的反馈构建 authoring context
    ↓
候选源码封存 → 静态验证 → 作业物化 → 固定预算训练
    ↓
绑定 checkpoint 的 public / sealed 评测 → 可信接受判决
    ↓
写入 ledger、更新 accepted source parent、形成会话报告
```

Stair 与 Push 的正式宿主是 [run_generation_session.py](../scripts/run_generation_session.py)，显式选择 task/cell，合同固定为 15 generations × width 2。任务拥有评分、reference、选择和 final cohort；公共层拥有生命周期、成本与作业。旧 Stair 命令名只转发共享入口。恢复和终态确认按 [runbook](stair_fresh_15round_runbook.md) 执行；旧 readiness 生成栈已经退役。

候选从冻结的随机初始化协议重新训练；只有 accepted 奖励源码成为后续父代，accepted checkpoint 不作为下一候选的 warm-start。source、checkpoint、job、evaluation binding 和 ledger 的身份必须闭合。候选自述和 rationale 不能代替源码与 runtime activation evidence。

新会话的 Genesis v6 持久化 composition snapshot v3：科学投影由 TaskSpec、ResolvedCellManifest 和任务装配参数形成，执行位置单独记录。backend 环境 manifest v2 绑定 Python ABI、选定依赖实现及本项目 `src/scripts/configs` 的相对路径与源码内容；父进程物化和 child admission 使用同一冻结身份。该证据不增加矩阵维度，`cell_variant_id` 仍是唯一持久化协议 epoch。新 reference 会话拥有内容校验后的 checkpoint 副本，保留原回执和来源链。实现与版本边界见 [P1 记录](../PLAN/2new_world/P1/DEVELOPMENT_LOG_ZH.md)。

## 3. 谁拥有语义

| 层 | 拥有的职责 | 不应取得的权限 |
|---|---|---|
| [robots](../src/true_rl_agent/robots/) | 资产、关节/刚体、形态事实、状态与动作适配 | 成功/失败和接受标准 |
| [tasks](../src/true_rl_agent/tasks/) | TaskPlugin、协议、任务分布、结果分类、指标、公开反馈 | 修改共享作业生命周期或私有评测隔离 |
| [composition](../src/true_rl_agent/composition/) | 机器人任务绑定、能力匹配和不可变执行身份 | 另存一套任务结果规则 |
| [agent_core](../src/true_rl_agent/agent_core/)、[runner](../src/true_rl_agent/runner/) | 有界 authoring、作业生命周期、预算、seed 隔离、checkpoint、可信证据和判决 | 根据机器人名称猜测奖励或任务阈值 |
| LLM candidate | 当前协议授予的 reward program 源码面 | PPO、预算、初始化、评测、checkpoint 选择、成功/失败或接受门 |

这是职责边界，不是“编排已完全通用”的完成声明。当前正式 generation 装配仍以 Stair 为中心；Go2 Push 的主线工作需要收回这些任务特化装配，同时继续让任务拥有任务语义，不能在共享层增加另一套 Stair/Push 分支规则。扩展检查入口见 [扩展与边界指南](extension_and_boundary_guide.md)。

## 4. 当前 Stair 协议

P3 unified V3 的权威入口是（V2 按原配置回放）：

- G1：[task protocol](../configs/tasks/stair_unified_protocol_v3.json)。
- Go2：[task protocol](../configs/tasks/stair_go2_12dof_unified_protocol_v3.json)。
- 各 protocol 的 `component_refs` 指向训练 profile、ordered evaluation suite、acceptance protocol 和 reward authority；机器人阈值与几何不能从另一机器人的文档示例复制。

`StairClimbingTaskConfig` 从所选协议的 `episode_outcomes` 构造结果语义。当前 completion mode 是 `first_valid_top_arrival_latched`：首次有效到顶需要平台边界、姿态和有效支撑满足该 cell 的合同，到顶后立即闭合 success；到顶前的 `forbidden_contact`、`off_course` latch 会阻断成功。没有额外 20-step 顶层稳定等待，成功闭合后也不继续重新判失败。

当前 primary reason 集合是 `success / forbidden_contact / off_course / base_orientation_instability / slip / insufficient_clearance / timeout`；`fall` 不是 unified V2 的独立 primary reason。净空、支撑面投影、持续步数及 timeout 的精确规则由所选协议和 suite 拥有。

当前 suite 采用 C0/C1/C2 相关分层，`J_primary = J_OOD`，接受协议比较配对 seed 的 `J_primary`，另检查 `j_ood`、`c0_id`、`c1_id`、`joint_ood`、`phase_control` guards。报告必须分别给出成功 n/N、成功率、J_primary 和 stratum/family 分数，不能把混合 progress 的分数写成成功率。完整说明见 [Stair 任务指南](task_extension_guides/stair_climbing.md)。

P3 ordered summary v4 / unified summary v3 保存可复算的 episode 事实；完成时间只作诊断，主分仍为 success/progress。三个 OOD family 等权，每个目标 stratum 必须实测存在成功率转折；饱和的 ID 和 phase 控制只保留独立能力守卫。题面先按 profile 的支撑高度尺度生成，再对各机器人用固定 reference 标定，详见 [P3 设计](../PLAN/2new_world/P3/IMPLEMENTATION_PLAN_ZH.md)。历史 ordered summary v3 / unified summary v2 保持原算术；比较投影的统计版本由任务统一选择，原始摘要按其存储格式验证，历史判决按已封证据回放。数值表示版本不新增矩阵维度，也不授予旧 reference 新后端资格，见 [P1 复审](../PLAN/2new_world/P1/AUDIT_FOLLOWUP_ZH.md)。

旧七桶、35-cell 和 standalone evaluation-semantics v6/v7 属于兼容路径或历史合同。尤其 standalone v7 的 stable-top 规则不等于 active unified V2；不能靠修改文档中的版本号把两者合并。历史 checkpoint、结果和 ledger 始终按其原有 binding 解释，不混入新协议的 accepted lineage。

## 5. 如何判断实验是否闭合

一次 eval 完成或 worker 自报 GO 都不等于整个实验完成。应核对 ledger、accepted head、正式 report/checks、可信 receipt、成本与资源记录，以及终态确认和恢复链。sealed 原始结果和私有 seed 不进入 authoring context，公开反馈由任务适配器投影。

已完成的 Go2 15 代实验见 [正式结果](../PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md)。其中 Gen8-A 是唯一 accepted；不能据此宣称跨机器人、跨训练 seed 或所有 OOD 家族均提升。当前 reference、checkpoint 与 final 配对结果的位置和身份见 [P0 验收](../PLAN/2new_world/P0/ACCEPTANCE_ZH.md)。

Phase 0 G1 checkpoint 探针等历史材料仍用于原协议下的兼容性与实验溯源；其数值不作为当前任务的资格证明。reference 饱和、指标区分力、机器人泛化和 Go2 Push 的后续工作由 [新主线计划](../PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md) 承接，本次文档清理没有改变冻结实验合同。

## 6. 训练后端和录像

新生产入口统一使用 Sim 6.0.1 / Isaac Lab 3 / PhysX / RSL-RL 5。完整预算训练、新 checkpoint 评测及原生录像已完成 [P2 工程验收](../PLAN/2new_world/P2/ACCEPTANCE_ZH.md)。旧/新后端数值等价未成立；[P3 reference 资格化](../PLAN/2new_world/P3/ACCEPTANCE_ZH.md)在固定 C2 标定基底后两本体均通过，但这不表示三卷难度完全相等。新正式搜索仍须完成相应资格及 P4 编排和实测成本合同。

Stair 原生录像由正式 eval 执行器的 `--video-dir` 呈现选项生成，使用相同策略、环境、题面、seed 与任务计数。旧的离线轨迹 renderer、独立 actor 与环境安装器已退役，见 [删除回执](../PLAN/2new_world/P2/retired_video_sources.json)。历史 MP4 和 checkpoint 保留原有诊断身份，不形成新的接受证据。

开发与回归命令统一维护在 [README](../README.md)，避免在概览复制另一套会漂移的启动教程。
