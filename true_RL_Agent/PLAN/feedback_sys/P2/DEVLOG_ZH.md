# P2 开发与复现记录

本轮从 `b24ddfe9e54c7cc93318a4c6a2675eb9fb799d20` 开始，没有提交 Git。完整验收与覆盖边界以 [ACCEPTANCE_ZH.md](ACCEPTANCE_ZH.md) 为准。

2026-09-24 独立审计后的修复另存 [audit_response](audit_response/)：[逐项裁定](audit_response/AUDIT_RESPONSE_ZH.md)、[执行卡](audit_response/IMPLEMENTATION_ZH.md)、[新增验收](audit_response/ACCEPTANCE_ZH.md)。该次评估与训练使用新的两个不可变源码快照，差异仅为训练启动锚点的 sampler/checker，见 [源码差分](audit_response/source_equivalence_audit.json)。历史 r2/r5/r6 的事实不重写；当前字段目录已升级。

## 实际落点

- P1：零调用解释增加真实执行/声明前提；删除审计中间字典格式迁移。
- 来源：共享 runner 仅提供 task-owned checkpoint admission hook；Stair 核逐级 evaluation-only 祖先合同，Push 核旧 reference 的物理 ABI/训练定义与来源回执。
- 采集：共享 numeric writer 与固定训练采样器；Stair 从整 seed 堆积改为有界分块；Push 用独立 reducer 重算现有窗口；两任务通过现有 backend/argv/receipt 接入。
- 作者：训练 coverage 走同一公开 allowlist；候选训练失败沿原有封闭错误合同和绑定 execution receipt 传递，不生成虚假成功训练记录。没有增加记忆表列或任意附件读取接口。

## 源码冻结与真实运行

隔离根目录：`/data/yuyangliu/AGENT/feedback_p2_20260923`。真实 repo、既有训练输出与历史权重只读。运行前检查 GPU 并使用现有 lease；本轮结束检查两 GPU 均空闲，无遗留的本轮模拟器进程。

| 快照 | 用途与结果 |
|---|---|
| r1 | parent/Gen8 两个前置 off；后续源码调整，未计入最终配对，成本保留 |
| r2 | 14 个评估进程，7 组 on/off 与独立复算通过 |
| r4 | 第一条 Push off 在导入 Isaac Lab 前失败；保留 venv 路径修复后重新开始 |
| r5 | Push 四个训练进程、两组配对通过；Stair Go2 两个训练进程实际完成，但 on 的外层 checker 错把历史/当前协议当同一身份，原失败保留 |
| r6 | 修正 checker 后新准备、新输出运行 Stair G1/Go2 四个训练进程，两组配对通过 |

r3 是未执行的中间传输包，已删除其 ZIP 和本地冗余指纹；本地 r5/r6 临时 ZIP 已清理，远端受测源码、实际失败与最终来源清单保留。没有删除用户其他目录或修改 P0/P1 历史审计。

`source_snapshot_r2/r5/r6.json` 记录实际打包文件；[source_equivalence_audit.json](source_equivalence_audit.json) 将最终生产文件与三版逐文件比较。后续只有公开消费链改变，受测采集、奖励、PPO、评估与配置保持同字节。原 r5 Stair on 附件只读复算见 `training_smoke_protocol_identity_r5_readonly_recheck.log`，不据此覆盖它的失败 receipt。

## 验收命令与证据

实际 GPU 命令、source、计时与日志 hash 均在 [运行总账](results/run_inventory.json)，避免将手写示例命令冒充已执行。`acceptance_cases.json` 为评估配方，`training_cases.json` 为最终训练配方；`prepare_training_smoke.py` 在目标真实 Python 中只作合同准备，不启动模拟器。

最终合并测试使用本地 `.venv/Scripts/python.exe`，`PYTHONPATH=src`、`PYTHONIOENCODING=utf-8`，执行以下 unittest 模块（共同前缀 `tests.unit.`）：

```text
test_numeric_evidence test_stair_numeric_capture test_push_numeric_capture
test_training_numeric_capture test_training_numeric_smoke
test_stair_checkpoint_ancestry test_push_checkpoint_admission test_stair_numeric_dev_replay
test_isaac_lab_backend test_subprocess_runner test_stair_phased_evaluation test_push_phased_evaluation
test_runtime_evidence_projection test_training_sampling_projection test_eval_public_projection
test_stair_public_projection test_prompt_projection_and_mechanism test_feedback_memory_retention
test_memory_context_phase13 test_push_recovery_runtime_adapter test_feedback_evidence_boundary
test_generation_historical_wire test_generation_receipts test_generation_finalization
```

结果 317 项，316 pass、1 本机 symlink 能力 skip；`verification_final.log`。`tests.integration.test_fixed_width_generation` 两项通过；`verification_training_sampling_generation.log`。三个 P2 checker 测试模块共 12 项通过；`verification_acceptance_tools.log`。此前一次合并命令误把 integration 模块写成 unit 路径的失败日志保留为 `verification_combined_initial.log`，已更正并完成上述真实检查。工具负例会打印模拟 FAIL，以最终测试断言/退出码判断。

独立同类复查确认：存储异常不归为候选程序故障；失败采样不参与接受或奖励执行判定；generation 从绑定 receipt 取证，不从裸 trial 拼接；默认 source/score 选择规则保持。全部 11 份真实配对报告 PASS，预算与成本统计不遗漏失败、前置运行和重跑。

P3 的下一输入已具备：有限事件词表所需的数值、可定位控制步、覆盖与缺失、同 cell/seed 成功反例，以及有界训练贡献样本。视觉输入、自动高层姿态描述、主动证据查询和作者修改收益实验尚不属于本次交付。
