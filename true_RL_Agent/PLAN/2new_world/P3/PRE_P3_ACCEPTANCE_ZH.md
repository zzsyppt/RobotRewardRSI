# P3 前置审计验收

2026-09-08。结论：**前置修复 GO；P3 新评测尚未签发。** 起点为 `6f17a4218f3039657147f5bbf61ce034d95b396d`，已将 12 个修复文件同步到服务器生产目录；没有提交 Git、修改旧协议或晋升新 reference。

## 修复与边界

逐项判断见 [PRE_P3_AUDIT_ZH.md](PRE_P3_AUDIT_ZH.md)。历史 timeout 由 TaskConfig 已加载的协议派生，执行 deadline 仍归 composition；没有新增配置源。Stair 和 Push eval 在加载 checkpoint 前核验实际 learner，并将结果保存为摘要元数据。冻结的 Phase-0 probe 字节不动，只在当前指南标明后端限制。

动态 recorder 测试修复了错误 fixture，同时修正测试进程的所有权：普通 unittest 进程启动独立 Isaac 用例，子进程先打印完整结果，再将结果退出码交给 Kit。保留原两项用例，并直接断言 XYZW → WXYZ 转换后的 recorder 数据。没有在产品适配器里容忍缺失字段。

`fast_shutdown=False` 的中间方案没有采用：它打印 `OK` 后仍在底层析构时 SIGSEGV。原错误方案和这次失败均保留日志身份，没有按成功验收。

Push 的历史 7200 s 合同与 Stair 的 2000-iteration 成本不能直接类比；未注册的 Go2 Push 不在本次资格化范围。后续新格子须按 P4/P6 实测签发 deadline。已提交的 P2 Kit 日志保持原样；P3 原始日志留磁盘，版本化小型结果与校验和。

## 实测结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 本地完整回归 | 1612 项，0 failure/error，19 项按环境跳过，exit 0，364.857 s | [本地回执](results/preflight_local_tests.json) |
| 目标 Python 专项 | 28 项，0 skip，exit 0；包含两项动态 recorder、checkpoint 张量、469 帧编码与评测摘要测试 | [目标回执](results/preflight_target_tests.json) |
| 动态反例 | 临时副本删去 `root_lin_vel_w` 后，原 AttributeError、子进程 ERROR 汇总、父进程 FAILED 汇总均可见，最终 exit 1；临时副本已删除 | 同上 |
| Go2 Stair 正式评测入口 | 同一 P2 随机初始化训练所得 checkpoint，3 seeds × 72 episodes = 216；成功 **165/216**，`J_primary=0.883938290781893`；所有任务统计与 P2 相同 | [实际评测回执](results/preflight_actual_evaluations.json) |
| G1-29 Push checkpoint eval | 实际入口通过，actor 输入 480、critic 输入 495、动作 29，保存完整构造核验；使用 384-step 诊断 checkpoint、全新 seed、100-step rollout，不据此宣称可学性或恢复成功 | 同上 |
| 推理加载 | 两任务均记录 `rsl_rl_5`，actor/critic 已加载，optimizer/iteration 未加载；核验发生在 load 前 | 同上 |
| 科学身份 | G1/Go2 task、cell manifest、机器人 PPO 身份逐项与修改前相同；历史 timeout 7200、实际 deadline 36000，归属不变 | [身份对照](results/preflight_scientific_identity.json) |
| 冻结 probe | SHA256 `7ee95526df69ce7e5cbfafe1f8e3bba66d2ce5d2bd987bab23aa8687350673aa`，与 HEAD 相同 | [本地回执](results/preflight_local_tests.json) |
| 完整 15 rounds | 71 个 checkpoint 全部存在且受保护：48 个中间模型核对路径/大小/mtime，23 个权威模型复算 SHA256；884 个受保护产物复算 SHA256，无变化 | [历史核对](results/preflight_history_preservation.json) |
| 部署及占位残留 | 12 个文件逐个核对原 HEAD 和部署后 SHA256；仅删除已确认的 42419 B 占位 manifest 与空目录；生成源已改为临时目录 | [部署回执](results/preflight_deployment.json) |
| 资源收尾 | 无 GPU 计算进程，既有 lease 均未持有；未删除共享锁文件 | [资源回执](results/preflight_resource_closure.json) |

本次源文件字节变化使 backend manifest 从 `2d42240c…` 变为 `ce7eaa4d…`，实际与声明一致。这是执行源码身份变化；上表中的科学配置与任务统计没有变化。新增 learner 元数据不会进入任务评分或改变任务证据的数学结果。

## 复算与继续开发

[已测试差异](results/preflight_source.patch) 可在基线提交上复原本次产品与测试修复；原始运行结果保留在 `/data/yuyangliu/isolated/new_world_p3/true_RL_Agent/PLAN/2new_world/P3/results`，各 JSON 回执给出具体路径、命令和校验和。P3 继续开发后，以此差异和回执核对当时版本，不能把随后工作区当成本次冻结源码。

本次只重跑与修复有关的实际评测和测试，P2 已完成的完整训练及原生录像验收继续引用其原记录。后续 P3 要独立完成新题池、真实表面 roughness、指标区分力与 reference 资格化，不能以本次工程通过替代。
