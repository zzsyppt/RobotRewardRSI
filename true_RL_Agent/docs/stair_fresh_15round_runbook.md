# Stair 正式 generation 入口

核对日期：2026-09-07；适用当前 active G1/Go2 unified V2。正式模式是 15 generations × width 2，轮数、lane 数和生命周期边界由协议固定。P2 后端工程验收已完成；**P3 reference 资格化与 P4 新成本合同完成前，不启动新正式搜索。** 旧后端 reference 的策略可读不代表新环境兼容资格成立。

旧单 lane 的 `run_stair_live.py`、`create_stair_final_confirmation_plan.py` 和 `run_stair_final_confirmation.py` 属于历史 compatibility 路径。它们不构成当前正式会话的启动链；旧 `outputs/launch_*.sh` 也不作为新会话模板。

## 正式链

1. 用当前 cell 的协议完成 reference 资格化，或选择已经合格且科学身份兼容的 reference Genesis。创建入口是 [create_stair_reference_genesis.py](../scripts/create_stair_reference_genesis.py)；导入使用 generation 的 `--source-reference-genesis-dir` 与 `--source-reference-session-id`。
2. 冻结本次源码内容、环境、session/output、host、GPU UUID、lease 根目录和各项预算；commit 用于追溯，backend 环境 manifest v2 另存本项目实际源码字节身份。新会话使用新 output，resume 绑定原会话。由 [run_stair_generation_launch_readiness.py](../scripts/run_stair_generation_launch_readiness.py) 签发对应 readiness receipt；不能手写或复制 receipt。
3. 用 [launch_stair_generation_session.py](../scripts/launch_stair_generation_session.py) 包装 [run_stair_generation_session.py](../scripts/run_stair_generation_session.py)，generation 参数包含 `--formal-session` 和明确的 `--cell-variant-id`。每个物理 attempt 使用独立目录和 exit receipt。
4. 恢复使用原 session/output 的 `--resume-existing`，重新采集当前主机与资源 readiness。新签发的 v4 receipt 在 `binding.boot_id` 记录本次 boot，在 `prelaunch_evidence.probe_boot_id` 保留历史 attempt/teardown probe 的 boot；原 probe 文件和旧 v3 receipt 不改写。源码、环境、会话归属和当前资源检查仍须通过。
5. 全部 generation 关闭后，同一会话链执行 final confirmation 和终态报告。终态复核模式为 `--formal-session --resume-existing --terminal-audit-only`，参数绑定同一已结束会话，不产生新训练/provider/eval。该模式也可补全已 terminal 会话缺失的报告；读取现有报告与补全报告应分别记录。

新 Genesis v6 保存 composition snapshot 的科学条件与执行位置。路径、GPU ordinal 的变化不改变科学投影；Python ABI、依赖实现和源码变化仍由环境身份检查。新 reference 会话封存自己的 `reference_checkpoint.pt`，恢复使用这份校验过的实物；原来源 Genesis 和评测回执仍须可解析。旧 Genesis/摘要按原版本读取，不自动改写为新会话，也不凭跨 Python 统计一致性宣称训练后端等价。边界及验收见 [P1 实施记录](../PLAN/2new_world/P1/DEVELOPMENT_LOG_ZH.md)。

对已经通过身份兼容检查的旧 reference 摘要，当前 public-dev screen 使用统一的比较统计投影，与新候选一起选择；原始 phase hash 保留，已封历史 screen 按其原统计版本回放。该投影不替代 reference 资格化或 Genesis import 检查，详见 [P1 复审](../PLAN/2new_world/P1/AUDIT_FOLLOWUP_ZH.md)。

## 查看参数

在服务器项目根 `/data/yuyangliu/AGENT/true_RL_Agent`，以当前生产解释器检查：

```bash
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/launch_stair_generation_session.py --help
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/run_stair_generation_session.py --help
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/create_stair_reference_genesis.py --help
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/run_stair_generation_launch_readiness.py --help
```

Launcher 参数包括 `--python-executable`、`--attempt-dir`、`--timeout-s` 和 `--launch-readiness-receipt`；其后的 generation 参数由 generation parser 负责。GPU 小时、control steps、输出字节预算来自该轮冻结的成本测算，不能照抄旧上限。provider env 文件只传路径，密钥不进入命令行正文或报告。

Stair 的单作业 `--runner-timeout-s` 当前默认 36000 秒；它归执行位置和成本预约所有，选择 cell 不得覆盖显式值。冻结协议中历史的 7200 秒字段仅保留原身份解释，不控制新 runner 的截止时间。Launcher 的整次 attempt `--timeout-s` 是另一层墙钟限制，须覆盖所授权的整个 attempt，不能直接填单次训练时长。

## 验收依据

结束时核对 generation receipts/closures、ledger、attempt reservation/result、checkpoint/summary hash、资源和成本闭包、final plan→reservation→phase receipts→result→report。报告的 checks 是入口，必要时用终态 reader 复算；进程退出或 ledger 行数不能单独证明成功。

现存 Go2 正式 15 代的证据入口在[结果记录](../PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md)，reference/head/ledger/replay 的精确位置在 [P0 基线及验收](../PLAN/2new_world/P0/ACCEPTANCE_ZH.md)。下一轮后端与协议按[建设主计划](../PLAN/2new_world/NEW_WORLD_PHASED_PLAN_ZH.md)重新冻结；本页不授予训练预算或替代运行验收。
