# Humanoid Agent 工作区

当前系统位于 [true_RL_Agent](true_RL_Agent/README.md)：在固定训练、评测和接受协议内生成、验证并比较 reward program。机器人事实、任务语义和实验编排的边界以该项目的当前实现与文档为准。

`D:\humanoid_agent` 继续作为 Git 根，当前项目保留 `D:\humanoid_agent\true_RL_Agent` 路径。

## 阅读入口

- [当前项目与运行入口](true_RL_Agent/README.md)
- [新系统分阶段计划](true_RL_Agent/PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md)
- [本地外层与服务器清理的实际状态](true_RL_Agent/PLAN/2new_world/P0/OUTER_CLEANUP_ZH.md)
- [环境与必要工具](setup/environment_reproduction.md)
- [按需代理](setup/proxy_control.md)

## 目录用途

| 目录 | 用途 |
|---|---|
| `true_RL_Agent/` | 当前源码、配置、测试、运行脚本与实验计划 |
| `scripts/`、`setup/` | 环境安装、激活、检查与按需代理工具 |
| `.tmp/`、`artifacts/` | 经清单确认仍需保留的录像迁移材料、实验取证与展示产物 |
| `reference/` | 研究资料 |
| `p6d-experiment-demo/`、`codex/` | 独立项目，有各自的 Git 历史和维护边界 |

父目录中的 `src/humanoid_agent`、旧 `configs`、`examples`、阶段脚本和开发日志已经删除；必要环境与代理脚本保留。外层本批共删除 912 个文件、28,653,612,487 B，删除后当前项目 1606 项回归通过，清单与回执见上面的清理记录。

完整 15 rounds 实验涉及的 checkpoint、reference 及正式证据保留。大体积环境与实验产物主要位于服务器 `/data/yuyangliu`，不进入 Git。
