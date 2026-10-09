# RobotRewardRSI

基于大语言模型的机器人奖励函数自进化系统，结合奖励代码生成、PPO 策略训练、Isaac Sim / Isaac Lab 仿真评测与结构化行为反馈。

## 功能

- 支持 Unitree Go2 / G1 的爬楼梯与抗扰恢复任务。
- 验证候选奖励程序，按固定训练预算生成策略并进行分层评测。
- 将轨迹测量、接触状态和奖励分项整理为后续迭代反馈。
- 分离机器人配置、任务语义、作业执行和评测协议。

## 代码

运行模块位于 `true_RL_Agent/src/true_rl_agent/`，任务配置位于 `true_RL_Agent/configs/`，训练与评测入口位于 `true_RL_Agent/scripts/`。

技术说明见 [系统概览](true_RL_Agent/docs/system_overview_zh.md) 和 [扩展接口](true_RL_Agent/docs/extension_and_boundary_guide.md)。
