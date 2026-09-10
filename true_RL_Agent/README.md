# true_RL_Agent

受控的 RL 代码研究与验证系统。LLM 提出 reward program 的机制改动，可信系统完成验证、固定预算训练、隔离评测、接受判决与账本。机器人和任务是两个独立维度。

项目运行时独立于已退役的父项目 `humanoid_agent`；旧 MVP 源码已删除，当前安装与运行入口均在本项目内。

## 当前可用范围

核对日期：2026-09-10。活动 Stair 行来自 [row catalog](src/true_rl_agent/tasks/stair_climbing/row_catalog.py)，Push 来自 [task binding](src/true_rl_agent/tasks/push_recovery/robot_binding.py)。表内 active 表示组合已注册；新后端 reference 的资格另行验证。

| Cell | 当前能力 | 协议与证据 |
|---|---|---|
| `stair.g1_37dof.unified_v3` | C2 固定标定基底后，当前 reference 资格通过（9 月 9 日修订） | [协议](configs/tasks/stair_unified_protocol_v3.json)、[P3 验收](PLAN/2new_world/P3/ACCEPTANCE_ZH.md) |
| `stair.go2_12dof.unified_v3` | C2 固定标定基底后，当前 reference 资格通过；accept C2 为 48/60、48/60、47/60 | [协议](configs/tasks/stair_go2_12dof_unified_protocol_v3.json)、[P3 验收](PLAN/2new_world/P3/ACCEPTANCE_ZH.md) |
| `stair.g1_37dof.unified_v2` | 已注册的冻结旧行；已有证据只读解释 | [协议](configs/tasks/stair_unified_protocol_v2.json)、[binding](configs/tasks/stair_g1_37dof_unified_binding_v2.json) |
| `stair.go2_12dof.unified_v2` | active；正式 15 代实验已结束 | [协议](configs/tasks/stair_go2_12dof_unified_protocol_v2.json)、[binding](configs/tasks/stair_go2_12dof_unified_binding_v2.json)、[实测结果](PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md) |
| `push.g1_29dof.reference_v1` | reference-program 与三阶段评测已接入共享正式宿主；固定 runner 的 15 代及恢复通过，物理资格归 P6 | [P4 实施记录](PLAN/2new_world/P4/IMPLEMENTATION_PLAN_ZH.md) |
| `push.g1_29dof.legacy_v1` | 已注册的 legacy Push 组合；尚无 reference-program Genesis / generation session | [binding](configs/tasks/push_g1_29dof_legacy_binding_v1.json)、[任务配置](src/true_rl_agent/tasks/push_recovery/config.py) |

Stair composition 与 reference 入口的默认行由同一 row catalog 派生。共享正式 CLI 要求显式指定 task 和 cell；V2 冻结会话使用历史 reader 按原格式审计，不迁入新会话合同。当前 G1/Go2 均通过原 public-accept 资格门；这不代表各卷均有相同区分力。[P4 共享编排工程验收已通过](PLAN/2new_world/P4/ACCEPTANCE_ZH.md)。新物理研究须冻结当前源码、合格 reference 和实测成本合同，预算按所选协议的实际样本量核算。

Go2 × Push Recovery 是 [Extension A](PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md)，尚未闭合。V1 Stair 行虽然 inactive，仍是 V2 的冻结 predecessor，按 lineage 保留。

## 从这里开始

- 当前正式运行、恢复与终态审计：[Stair generation runbook](docs/stair_fresh_15round_runbook.md)。唯一执行入口为 `run_generation_session.py --task-id … --cell-variant-id …`，固定 15 generations × width 2。两个旧 Stair 命令名只转发此入口，不再接受旧 pilot/validation/readiness 模式。
- reference、accepted head、ledger、checkpoint 与历史 replay：[P0 基线及验收](PLAN/2new_world/P0/ACCEPTANCE_ZH.md)。本轮唯一 accepted 是 Gen8-A；最终确认是诊断证据，不再次改写 head。
- 后续建设顺序：[P0–P8 主计划](PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md)。问题背景读[原始 55 项清单](PLAN/Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md)，初始分工读 [P0 编号索引](PLAN/2new_world/P0/ISSUE_DISPOSITION_ZH.md)，身份、执行位置和恢复的实现状态见 [P1 验收](PLAN/2new_world/P1/ACCEPTANCE_ZH.md)。
- 系统闭环与语义归属：[系统概览](docs/system_overview_zh.md)。新任务的职责和验证：[扩展指南](docs/extension_and_boundary_guide.md)，当前 Stair 规则见 [Stair 任务指南](docs/task_extension_guides/stair_climbing.md)。

## 所有权

| 层 | 负责什么 |
|---|---|
| [robots](src/true_rl_agent/robots) | 资产、形态事实、关节/刚体、状态和动作适配 |
| [tasks](src/true_rl_agent/tasks) | 成功/失败、任务分布、评测指标、诊断和任务反馈 |
| [composition](src/true_rl_agent/composition) | 机器人任务绑定与不可变执行身份 |
| [agent_core](src/true_rl_agent/agent_core)、[runner](src/true_rl_agent/runner) | 有界 authoring、生命周期、作业、成本、seed 隔离、checkpoint、可信判决和 ledger |
| LLM candidate | 协议授予的 reward program；不控制 PPO、训练预算、评测、seed 或接受门 |

## 开发与验证

在项目根目录执行：

```powershell
# 首次配置：CPU Torch 足以执行本地 checkpoint / RGB 编码回归。
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/Scripts/python.exe -e '.[test]'
.\.venv\Scripts\python.exe -m unittest discover -s tests -t .
.\.venv\Scripts\python.exe scripts/run_generation_session.py --help
.\.venv\Scripts\python.exe scripts/run_loop.py --help
```

`run_loop.py` 是已有组合/audit 入口；`run_stair_live.py` 和 V1 final-confirmation CLI 是历史 compatibility 工具，均不等于双 lane 正式 generation。编码与 checkpoint tensor 测试属于本地必跑项；Isaac 仿真验证在固定服务器运行时执行，不能用缺依赖的 skip 代替。

涉及运行时、默认格子或冻结协议的变更，发布前还须在目标项目根目录执行完整回归。这里必须是真实 Git 检出，单独的源码压缩包不能验证 运行源码的 Git 身份；保留 `PYTHONPATH`，使测试子进程能导入当前项目：

```bash
TRUE_RL_AGENT_ISAAC_DYNAMIC_TEST=1 PYTHONPATH=.:src /data/yuyangliu/envs/isaaclab6_validation/bin/python -m unittest discover -s tests -t . -v
```

验收分别记录本地与目标运行时的总数、失败及跳过原因，确认 `tests.integration.test_generation_session` 的两任务分支实际执行；进程中断→恢复→15 代 final 的持久化回执位于 P4 验收目录。目标专项不能替代目标全量。

冻结的 `scripts/isaaclab_stair_phase0_probe.py` 仅属于旧 RSL-RL 3 历史验证，不能在目标后端运行；其文件字节继续保留，当前评测使用正式 Stair eval 入口。

训练、评测、原生录像统一使用 `/data/yuyangliu/envs/isaaclab6_validation/bin/python`（Sim 6.0.1 / Lab 3 / PhysX / RSL-RL 5.0.1）。P2 审计修复已部署，完整 196608000 步训练、新 checkpoint 评测/录像、部署后对照、历史保护与资源收尾均通过，**工程 GO**，见 [P2 验收](PLAN/2new_world/P2/ACCEPTANCE_ZH.md)。旧/新后端数值等价未成立；[P3 当前验收](PLAN/2new_world/P3/ACCEPTANCE_ZH.md)记录两本体修订后的 reference 资格通过及剩余卷间差异，六段原生录像均已核验。新正式搜索须满足对应 reference 资格，并完成 P4 实测成本合同。旧 Sim 5.1 环境只保留冻结历史对照用途，新生产入口不接受它。
