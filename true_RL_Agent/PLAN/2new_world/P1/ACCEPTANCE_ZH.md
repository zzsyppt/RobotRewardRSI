# P1 验收记录

日期：2026-09-07。**P1 复审修复完成，重新验收 GO**。逐项判断、实际改动、旧版本退役条件及证据限制见 [复审记录](AUDIT_FOLLOWUP_ZH.md)，本次机器记录见 [audit_validation_results.json](audit_validation_results.json)。范围见 [实施计划](IMPLEMENTATION_PLAN_ZH.md)，首轮变更见 [开发记录](DEVELOPMENT_LOG_ZH.md)。

## 2026-09-07 复审结论

混合旧 reference / 新 candidate 的比较统计已在任务投影入口统一，selector 的一致性检查保留；原 summary/phase 不重写，历史 formal assembly 使用已封 screen 的统计版本。当前聚合链从原始 episode 起使用明确算术，重复统计公式收敛到共享数值模块；任务与机器人语义没有移入共享层。

- Windows 全量：1615 项，0 failure/error，31 skip；Linux 3.11/3.12 各 250 项通过。
- Linux 完整 15 代及 final 硬退出恢复通过，835.849 秒。使用 fake runner，实际覆盖首代双 lane / 接受、后续 incomplete 关账及终态重复审计。
- 最后删除一个无用导入后，本地及 Linux 双版本各补 22 项通过；最终源码 hash 为 `d0c45575636c7a30d87443069b110e36bd9085ca45641b012ed92d581da784ee`。详细源码差异与各批测试身份在复审记录中分别保留。
- 两个 active V2 cell 从相同原始 episode 生成的摘要和投影跨 Python 身份一致；真实 P5 摘要保持原 hash，24 份几何实验的旧 seed 统计复算一致。
- 重新检查 71 个正式 checkpoint 的大小/mtime、801 份正式证据和 6 份 reference 文件 hash，均不变。未部署服务器正式 checkout，未启动真实 RL/provider 作业，临时验证副本按复用价值保留。

这次 GO 不表示旧 P5 自动兼容新 Genesis/后端，也不替代 P2 资格化、指标修改或 P4 通用编排。以下为 **2026-09-06 首轮记录**，其测试数量和源码身份不代表复审后的最终代码；原 [validation_results.json](validation_results.json) 保持原样。

## 1. 已取得的证据

| 检查 | 实际结果 | 证据 |
|---|---|---|
| Windows Python 3.12 完整回归 | 1609 项，0 failure/error，31 skip，313.999 秒 | `local_regression.log` |
| 本地恢复与 Genesis 专项 | 89 项全部通过 | `recovery_identity_tests.log` |
| 本地 composition / 环境 / readiness | 33 项全部通过 | `composition_environment_tests.log` |
| 本地 final confirmation | 14 项全部通过 | `final_confirmation_tests.log` |
| Linux Python 3.11.15 关键专项 | 200 项全部通过，0 skip，80.681 秒 | `linux_identity_recovery_py311.log` |
| Linux Python 3.12.14 同一专项 | 200 项全部通过，0 skip，83.126 秒 | `linux_identity_recovery_py312.log` |
| Linux 完整终态 | candidate-head 15 代、3 代崩溃恢复、reference-head 15 代三种场景全部通过；最后一项单独复核 490.514 秒 | `linux_formal_tests.log`、`linux_reference_head_tests.log`，见下文 |
| G1 / Go2 数值身份与旧摘要读取 | Windows 3.12、Linux 3.11/3.12 的新统计与历史表示读取身份相同 | `numeric_identity_*.json`、`numeric_identity_probe.py` |
| 生产环境实际 child 准入 | Python 3.11 独立 child 复算 backend v2 manifest 成功；0 训练作业 | `backend_admission_py311.json` |
| 保护历史产物 | 71 个正式 15-round checkpoint 的大小/mtime 与 P0 一致；801 份正式证据及 6 份 reference 实物 hash 不变 | `protected_artifacts_check.json` |

本地 skip 不计通过。Linux 专项覆盖本阶段涉及的 procfs、进程组、恢复、cost journal 与旧格式 reader；完整终态使用现有 Isaac 源码环境和 fake runner，验证编排与证据，不是物理训练或跨后端性能实验。

完整终态首轮 3 项中有 1 项失败，该日志保留：fake runner 按 final 槽位名称生成高分，导致同一 reference checkpoint 的两次合成评测不同。已改为按实际 subject 生成结果；“checkpoint 相同、两个 job 不同、配对差和标准误为零”的断言保留。随后复核又发现测试辅助函数在第二次审计时重复创建无消费者的报告副本，已删除该辅助写出。两处都是夹具修正，产品实现不变。修正后的完整 reference-head 用例通过；其他两个已通过的长用例未重复执行，不能把首轮日志本身读作 3 项全绿。

## 2. 行为闭合与所有权

| 项目 | 验收落点 |
|---|---|
| F11 | 新摘要按明确数值版本复算；旧 ordered v2 / unified v1 的已存单 ULP 表示保留原 hash。计数和原始源码/文件 hash 没有放宽 |
| F13 | 实际 Push 装配改变位置身份不变、改变允许的 seed/预算身份变化；Stair 位置身份不变、G1→Go2 cell 改变身份；科学参数仍来自任务和矩阵绑定 |
| F18 | 三种 resume/terminal 模式在 probe boot A、当前 boot B 下签发并加载新 readiness；历史会话字节不改写。现有生命周期检查 boot/PID incarnation、资源租约与 once-only 结算 |
| F19 | 原有 typed-spec rematerialization 的五项 metadata 身份断言通过；保持现有正确实现 |
| F26 | 新 reference/import 保存会话自有只读副本；删除测试夹具的原 checkpoint 及来源副本后仍可恢复，train/eval 调用不增加，原 Genesis 字节不变 |
| F48 | train 完成、public_dev timeout、trial 已封而 generation 未封后退出：恢复保持原 incomplete 投影；provider/train/eval 实际调用不增加；再次 resume 不重复 commit |
| F50 | 同一源字节换路径仍被 child 接受；改变源码在作业前拒绝。Windows/服务器 323 个执行文件的内容身份一致，生产环境独立 child 准入成功 |
| F55 | 15 代保持 reference-head、accepted=0、head hash 不变；final 中断恢复后两个 job 身份不同且 checkpoint 相同，配对均值和标准误均为 0；两次审计不改 ledger/report、不增加实际 job |
| F56 | 不存在的 summary/checkpoint 只留在计划中；私有实物列表只收集实际产生且有 checksum 的文件；正常与失败路径保持公开信息隔离 |

任务 × 机器人两根轴不变。没有新增 terrain 路由、第三个矩阵维度、第二套 ledger 或任务专用共享调度分支。现有协议配置文件没有改动。新的 source snapshot 与 derived statistics 模块没有 Stair、Go2 或 Push 的特殊语义。

## 3. 可复核命令与内容身份

本地完整回归：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Linux 关键专项分别在 `/data/yuyangliu/envs/isaaclab/bin/python` 和 `/data/yuyangliu/envs/isaaclab6_validation/bin/python` 下运行同一集合：

```text
tests.unit.test_isaac_lab_execution_environment
tests.unit.test_reference_genesis
tests.unit.test_stair_phase_b_composition
tests.unit.test_stair_generation_launch_readiness
tests.unit.test_stair_generation_session
tests.unit.test_stair_final_confirmation
tests.unit.test_generation_lifecycle
tests.unit.test_stair_generation_cost_recovery
tests.unit.test_p6b_attempt_chain_fault_probe
tests.integration.test_fixed_width_generation
tests.unit.test_stair_ordered_evaluation
tests.unit.test_acceptance_contract
tests.unit.test_acceptance_replay
```

执行方式：`PYTHONPATH=src <python> -m unittest <上述模块> -q`。完整终态模块是 `tests.integration.test_stair_generation_session`，reference-head 复核使用其 `StairGenerationSessionProductionShapeTests.test_formal_fifteen_generations_keep_reference_head_and_finalize_twice`。

- 323 个执行源码文件的 snapshot hash：`01fcdfd24f1fafac3139b0ea15896408cb8e433df94d474db953fe02ac122fdc`。
- 数值 probe 合并身份：`87008c86094d86a4169e3171507f270ca0f6002fe8a1a80cda9d9de7f8b94008`。
- 生产 Python 3.11 backend manifest：`cf40b8090202c69f04bc9d50c73f4f0bbfb1a4774eb5e2ec325dfa9d2321c8ec`。

## 4. 清理与范围限制

P1 清理已完成，详见 [cleanup_result.json](cleanup_result.json)：本地删除 8 份已被替代的专项日志/传输包，共 7,148,287 字节；服务器删除两个本阶段隔离验证目录，共 85,461,977 字节。删除前逐个核对绝对路径、进程占用与已归档日志 hash，删除后确认目标不存在；正式 checkpoint 删除数为 0。实现、必要验收证据和首次完整终态失败日志保留。

本次没有部署到服务器正式 checkout，没有启动真实 RL 训练、provider、评测或录像，没有重启主机。跨 boot 使用确定性 host/probe 证据注入；Linux 进程恢复另有实际子进程检查。新 session 自有 checkpoint 不授予删除来源 Genesis、原评测回执或旧 15 rounds checkpoint 的权限。

P2 的单一训练/录像后端迁移、P3 指标设计、P4 通用编排和 Go2 Push 任务闭合尚未实施。不能从 P1 的证据身份一致推断这些目标已完成。
