# RobotRewardRSI

基于大语言模型的机器人奖励函数自进化系统。通过奖励代码生成、固定预算 PPO 训练、分层仿真评测和结构化失败反馈，持续迭代机器人任务的奖励逻辑。

## 工作流程

1. 解析机器人与任务配置，冻结训练预算、评测协议和候选修改范围。
2. 从已接受的奖励程序、任务约束和公开反馈构建模型上下文。
3. 生成候选 Python 奖励代码，执行静态检查和数值验证。
4. 使用 Isaac Sim / Isaac Lab 与 RSL-RL / PPO 训练候选策略。
5. 按独立任务指标进行分层评测，接受或拒绝候选，并记录后续迭代反馈。

## 任务与反馈

- **机器人**：Unitree Go2 与 G1；机器人形态、状态接口和任务语义分别配置。
- **任务**：爬楼梯与抗扰恢复，支持通过 cell 配置选择机器人和协议变体。
- **反馈**：动作轨迹、接触状态、奖励分项与任务结果；数值证据进一步转换为测量、事件和行为描述。
- **执行**：双 GPU 候选训练、作业隔离、检查点管理、会话恢复和结果追踪。

## 代码结构

| 目录 | 职责 |
|---|---|
| `true_RL_Agent/src/true_rl_agent/agent_core/` | 奖励代码生成、验证与候选修订 |
| `true_RL_Agent/src/true_rl_agent/runner/` | 作业执行、GPU 绑定及进程生命周期 |
| `true_RL_Agent/src/true_rl_agent/tasks/` | 任务逻辑、奖励接口、评测指标与反馈 |
| `true_RL_Agent/src/true_rl_agent/robots/` | 机器人配置与运行时适配 |
| `true_RL_Agent/src/true_rl_agent/evaluation/` | 评测绑定、协议与结果处理 |
| `true_RL_Agent/configs/` | 机器人事实、任务协议和实验配置 |
| `true_RL_Agent/scripts/` | 会话入口、训练、评测与轨迹处理 |
| `true_RL_Agent/tests/` | 单元测试、集成测试与固定测试数据 |
| `true_RL_Agent/docs/` | 系统架构、扩展接口和运行说明 |
| `tools/prepare_first_stair_smoke.py` | 生成 Go2 PPO 微型检查的作业参数 |

## 环境与入口

仿真环境使用 Python 3.12、Isaac Sim 6、Isaac Lab 3 系列源码及 RSL-RL 5；GPU 训练由 PyTorch 执行。LLM 通过 OpenAI 兼容接口接入，可配置 DeepSeek 等服务。

在已安装仿真依赖的环境中，从仓库根目录查看入口参数：

```bash
export PYTHONPATH="$PWD/true_RL_Agent:$PWD/true_RL_Agent/src"
python true_RL_Agent/scripts/run_generation_session.py --help
python true_RL_Agent/scripts/training_numeric_smoke.py --help
```

正式实验需要配置模型服务、选定任务与 cell、满足 reference 资格要求，并冻结成本合同。训练、评测和录像产物写入实验目录；Python 环境、API 凭据、检查点和录像不纳入版本管理。

更多技术说明：

- [系统概览](true_RL_Agent/docs/system_overview_zh.md)
- [扩展接口与职责边界](true_RL_Agent/docs/extension_and_boundary_guide.md)
- [正式实验运行说明](true_RL_Agent/docs/stair_fresh_15round_runbook.md)
