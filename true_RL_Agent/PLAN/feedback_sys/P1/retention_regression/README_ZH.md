# 真实会话记忆回归验收（2026-09-23）

本轮采用原审计驱动的 `--memory-variant full`，没有启用 `sparse`、`sparse_notop` 或 `token` monkeypatch。服务器在 `/data/yuyangliu/p1_retention_fix_20260923_01` 从 `fe96455ddd1a8c6ba30ee16e6a3a015422afe8d3` 导出两个隔离源码副本；current 只覆盖当前变更的 10 份生产源码。没有启动训练、模拟器或模型调用。本轮只验证保存会话的证据重建、记忆保留与确定性，不提供诊断质量或 RL 收益证据。

| 真实 prompt 检查点 | 基线保留 / 省略 / 字节 | 修复后保留 / 省略 / 字节 | 预先冻结的保留下限 | 字节上限 |
| --- | --- | --- | --- | --- |
| formal c6 Stair，g13 前 | 12 / 0 / 14,080 | 12 / 0 / 13,814 | 11 | 14,336 |
| formal c6 Stair，g16 前 | 14 / 1 / 13,912 | 13 / 2 / 14,111 | 13 | 14,336 |
| Go2 Push attempt_08，g6 前 | 5 / 0 / 6,087 | 5 / 0 / 6,525 | 5 | 14,336 |

三个检查点均通过，字节数采用生产 `canonical_prompt_json_bytes` 的 UTF-8 序列化口径。g16 比基线少保留一代，没有提高预算或下调门槛。

Stair 的 33/33 份、Push 的 15/15 份公开评估摘要与原账本逐值相等。六次基线/current prompt 重建比较均 deterministic。对 ledger 的公开 summary 单独修改一个数值后，两个 current 回放均拒绝，报 `generation public summary/phase receipt mismatch`；两个基线仍接受，与原审计记录一致。两个任务的最终 current_best checkpoint 均与基线相同，完整 hash 在 `run_summary.json`。

使用已生成的 payload 做了额外只读比较，没有重跑：三个检查点分别比较 12 / 13 / 5 个共有 generation、24 / 26 / 10 行 trial，前 9 列逐值相等；修复后保留代均在对应基线记忆内。末代 12 / 15 / 5、最近接受代 8 / 8 / 4 均保留；保留数与省略数之和分别等于实际 prefix 的 12 / 15 / 5 代。所有行均恰为 10 列，第 10 列为合法短状态，所有 generation 均没有顶层 `runtime_observation` 副本。原驱动的 `last_row_measurements` 只打印字典，故 current 报告该字段为 null；实际字符串检查来自保存的 payload，见 `payload_measurements` 与 `payload_comparison_checks`。

普通单元测试位于 `tests/unit/test_feedback_memory_retention.py`，只读取 `tests/fixtures/feedback_memory_baseline.json`，运行时不依赖 PLAN 或服务器。该 fixture 共 48,737 字节，保留三份冻结公开 payload 的紧凑记忆内容与实际对应的 scalar runtime evidence，记录 41 个来源 SHA-256、7 个去重 runtime profile；36 行可见旧 overlay measurement 已核对一致。没有补回基线已省略的历史。一个测试含三个 subtest，调用生产行编码与 compactor，检测保留门槛、预算、最新/最近接受代、计数闭合和幂等性；本地结果与上表三个 current 结果一致。它验证行编码和压缩回归，不代替真实 receipt replay。

本轮文件与复查口径：

- `sync_manifest.json`：基线 commit、原驱动完整 SHA-256、10 份同步源码的本地/远端/基线 hash。运行结束后，远端 10 份源码及本地 10 份源码仍与同步值一致。
- `report_baseline_stair.json`、`report_baseline_push.json`、`report_current_stair.json`、`report_current_push.json`：原驱动输出原样取回。原驱动位于 `../audit_replay/p1_replay_audit.py`，SHA-256 为 `44de525e8643c92dec7a2bb43c58f4df9d7f01c80d6374ad69195b24f6d04425`。
- `run_summary.json`：四条完整驱动命令、各报告 hash、耗时、实际 payload 路径与 hash、短状态计数、额外逐列比较、current_best 对照，以及两个 session 的 ledger/genesis/manifest/contract 共 8 份身份文件前后 hash。
- `run_remote_replay.py`：本轮隔离导出、仅同步变更源码并调用原驱动的执行脚本；不修改原驱动。三个检查点的门槛及额外字段比较由 summary 单独记录，不能仅将原驱动的 `status=PASS` 当作全部验收。

四项驱动墙钟合计约 668.3 秒（baseline Stair 284.503、baseline Push 40.124、current Stair 298.525、current Push 45.144）；不包含传输与额外只读核对时间。所有新完整 payload 只留在远端隔离目录，没有复制到本地。

历史目录的写入边界来自驱动：冻结 session 作为输入，Push composition 使用 `read_only=True`，Stair composition 与所有输出指向新 scratch。8 份身份文件前后 hash 均一致；此检查范围不等于历史目录每个文件都做了前后 hash。原 `audit_replay` 报告、原 `acceptance_checks.json` 和原验证日志没有被本轮覆盖。
