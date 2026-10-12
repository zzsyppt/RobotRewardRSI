# true_RL_Agent

受控的 RL 代码研究与验证系统。LLM 提出 reward program 的机制改动，可信系统完成验证、固定预算训练、隔离评测、接受判决与账本。机器人和任务是两个独立维度。

项目运行时独立于已退役的父项目 `humanoid_agent`；旧 MVP 源码已删除，当前安装与运行入口均在本项目内。

## 当前可用范围

Push 核对日期：2026-09-22。活动 Stair 行来自 [row catalog](src/true_rl_agent/tasks/stair_climbing/row_catalog.py)，Push 来自 [task binding](src/true_rl_agent/tasks/push_recovery/robot_binding.py)。活动格子表示当前可解析的组合；注册、协议冻结、工程验收和 reference 资格分别记录。Push 两行统一为 v3 外力评价，已有 checkpoint 的 public-dev 物理与回放验收通过：Go2为141/192，G1为33/192。按用户同日后续修订，当前Push新reference默认使用 `observed_outcomes_v1`：完整有效dev证据须同时出现成功与失败，不再要求固定成功率区间。两台已有测量复核此dev门均PASS；是否接近饱和仍看分层结果，自动门通过本身不证明余量充分。新域三相完整资格仍待验证，不能宣布科学GO。Stair既有P3题池、mask和评分规则不变。详见[准入修订](PLAN/2new_world/P6/REFERENCE_ADMISSION_REVISION_ZH.md)、[本次验收](PLAN/2new_world/P6/FORCE_UNIFICATION_ACCEPTANCE_ZH.md)与[P6历史资格](PLAN/2new_world/P6/ACCEPTANCE_ZH.md)；历史结果和原规则保留原义。

| Cell | 当前能力 | 协议与证据 |
|---|---|---|
| `stair.g1_37dof.unified_v3` | C2 固定标定基底后，当前 reference 资格通过（9 月 9 日修订） | [协议](configs/tasks/stair_unified_protocol_v3.json)、[P3 验收](PLAN/2new_world/P3/ACCEPTANCE_ZH.md) |
| `stair.go2_12dof.unified_v3` | C2 固定标定基底后，当前 reference 资格通过；accept C2 为 48/60、48/60、47/60 | [协议](configs/tasks/stair_go2_12dof_unified_protocol_v3.json)、[P3 验收](PLAN/2new_world/P3/ACCEPTANCE_ZH.md) |
| `stair.g1_37dof.unified_v2` | 已注册的冻结旧行；已有证据只读解释 | [协议](configs/tasks/stair_unified_protocol_v2.json)、[binding](configs/tasks/stair_g1_37dof_unified_binding_v2.json) |
| `stair.go2_12dof.unified_v2` | active；正式 15 代实验已结束 | [协议](configs/tasks/stair_go2_12dof_unified_protocol_v2.json)、[binding](configs/tasks/stair_go2_12dof_unified_binding_v2.json)、[实测结果](PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md) |
| `push.g1_37dof.recovery_v3` | 统一外力评价；dev为33/192，已有恢复正例和改进空间，新域三相完整资格仍待验证 | [任务指南](docs/task_extension_guides/push_recovery.md)、[评价协议](src/true_rl_agent/tasks/push_recovery/force_protocol.py) |
| `push.go2_12dof.recovery_v3` | 同一外力评价，经Go2机器人事实解析；dev为141/192，仍未完成新reference正式资格 | [任务指南](docs/task_extension_guides/push_recovery.md)、[binding](src/true_rl_agent/tasks/push_recovery/robot_binding.py) |

Stair composition 与 reference 入口的默认行由同一 row catalog 派生。Push 新会话默认 G1 v3，两行统一使用任务拥有的 `ForceEvaluationProtocol`，新评价 base seed 为1500000000；训练条件由独立的 task-owned 行配置保存，已有 checkpoint 验收显式带入原训练协议和 reference identity。力大小、恢复阈值和脉宽经各机器人尺度解析，无力空档固定0.22秒，位移不再造成失败。共享正式及 reference 资格 CLI 要求显式指定 task 和 cell；冻结会话不能由新默认替换身份。当前 Stair G1/Go2 均通过原 public-accept 资格门；这不代表各卷均有相同区分力。[P4 共享编排工程验收已通过](PLAN/2new_world/P4/ACCEPTANCE_ZH.md)。新物理研究须冻结当前源码、reference 与实测成本合同。

旧Push v2保留历史manifest读取及显式回放；G1-29保留明确的历史审计入口，不属于当前矩阵。G1原资格只在原域有效，原Go2 NO-GO不被新诊断覆盖。[P4共享编排证据](PLAN/2new_world/P4/IMPLEMENTATION_PLAN_ZH.md)、外力诊断或成功/失败录像均不能替代新评价域的资格闭合。P6完整目标未完成，不能据本次dev验收启动P7。V1 Stair行虽然inactive，仍是V2的冻结predecessor，按lineage保留。

## 从这里开始

- 当前正式运行、恢复与终态审计：[Stair generation runbook](docs/stair_fresh_15round_runbook.md)。唯一执行入口为 `run_generation_session.py --task-id … --cell-variant-id …`，固定 15 generations × width 2。两个旧 Stair 命令名只转发此入口，不再接受旧 pilot/validation/readiness 模式。
- reference、accepted head、ledger、checkpoint 与历史 replay：[P0 基线及验收](PLAN/2new_world/P0/ACCEPTANCE_ZH.md)。本轮唯一 accepted 是 Gen8-A；最终确认是诊断证据，不再次改写 head。
- 后续建设顺序：[P0–P8 主计划](PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md)。问题背景读[原始 55 项清单](PLAN/Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md)，初始分工读 [P0 编号索引](PLAN/2new_world/P0/ISSUE_DISPOSITION_ZH.md)，身份、执行位置和恢复的实现状态见 [P1 验收](PLAN/2new_world/P1/ACCEPTANCE_ZH.md)。
- 系统闭环与语义归属：[系统概览](docs/system_overview_zh.md)。新任务的职责和验证：[扩展指南](docs/extension_and_boundary_guide.md)，当前任务规则见 [Stair 任务指南](docs/task_extension_guides/stair_climbing.md)和 [Push 任务指南](docs/task_extension_guides/push_recovery.md)。

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

`run_loop.py` 是已有组合/audit 入口；旧 Push legacy bootstrap/dry-run 回执仅解释其历史身份，不为新外力评价合成 reference 资格。`run_stair_live.py` 和 V1 final-confirmation CLI 是历史 compatibility 工具，均不等于双 lane 正式 generation。编码与 checkpoint tensor 测试属于本地必跑项；Isaac 仿真验证在固定服务器运行时执行，不能用缺依赖的 skip 代替。

涉及运行时、默认格子或冻结协议的变更，发布前还须在目标项目根目录执行完整回归。这里必须是真实 Git 检出，单独的源码压缩包不能验证 运行源码的 Git 身份；保留 `PYTHONPATH`，使测试子进程能导入当前项目：

```bash
TRUE_RL_AGENT_ISAAC_DYNAMIC_TEST=1 PYTHONPATH=.:src /data/yuyangliu/envs/isaaclab6_validation/bin/python -m unittest discover -s tests -t . -v
```

验收分别记录本地与目标运行时的总数、失败及跳过原因，确认 `tests.integration.test_generation_session` 的两任务分支实际执行；进程中断→恢复→15 代 final 的持久化回执位于 P4 验收目录。目标专项不能替代目标全量。

冻结的 `scripts/isaaclab_stair_phase0_probe.py` 仅属于旧 RSL-RL 3 历史验证，不能在目标后端运行；其文件字节继续保留，当前评测使用正式 Stair eval 入口。

训练、评测、原生录像统一使用 `/data/yuyangliu/envs/isaaclab6_validation/bin/python`（Sim 6.0.1 / Lab 3 / PhysX / RSL-RL 5.0.1）。P2 审计修复已部署，完整 196608000 步训练、新 checkpoint 评测/录像、部署后对照、历史保护与资源收尾均通过，**工程 GO**，见 [P2 验收](PLAN/2new_world/P2/ACCEPTANCE_ZH.md)。旧/新后端数值等价未成立；[P3 当前验收](PLAN/2new_world/P3/ACCEPTANCE_ZH.md)记录两本体修订后的 reference 资格通过及剩余卷间差异，六段原生录像均已核验。新正式搜索须满足对应 reference 资格，并完成 P4 实测成本合同。旧 Sim 5.1 环境只保留冻结历史对照用途，新生产入口不接受它。
