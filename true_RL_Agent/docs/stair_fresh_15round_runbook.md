# 共享正式 generation 入口

宿主流程核对日期：2026-09-10；Push 入口与代数边界更新于2026-09-22。正式宿主默认为 15 generations × width 2；`--max-generations N` 可显式指定正整数代数，成本合同、会话合同和 final plan 必须冻结同一个 N。Stair 与 Push 使用同一个 [run_generation_session.py](../scripts/run_generation_session.py)。机器人、任务含义和资格门分别由 profile、task 与 cell 绑定提供。

P3 的 G1/Go2 Stair V3 reference 已通过原资格门，见 [P3 验收](../PLAN/2new_world/P3/ACCEPTANCE_ZH.md)。P4 的 Push reference cell 只闭合历史工程装配；当前 Push 两行统一为外力评价 V3，原 G1 资格只在原速度增量域有效，新评价域的完整资格仍未闭合，见 [Push 指南](task_extension_guides/push_recovery.md)。固定 runner 或新 public-dev 工程验收通过不代表 reference 资格 GO。

## 准备与执行

1. 明确 `--task-id` 和 `--cell-variant-id`。Stair 当前新研究选择 `stair.g1_37dof.unified_v3` 或 `stair.go2_12dof.unified_v3`；Push 活跃行是 `push.g1_37dof.recovery_v3` 和 `push.go2_12dof.recovery_v3`，完成对应新评价域资格前不得据注册或工程验收启动正式 generation。`push.g1_29dof.reference_v1` 只保留显式历史审计入口。正式入口不隐式选择机器人。
2. 使用当前 cell 合格的 reference，或通过 `--source-reference-genesis-dir` 与 `--source-reference-session-id` 导入身份兼容的 Genesis。reference 资格不是 CLI prepare 检查可以替代的。
3. 提供 `--cost-contract` 与 `--provider-policy` JSON。成本合同从所选 cell 的真实 workload、测量来源及各 operation 的保守时间/空间预约构造；必须计入 reference、两个 dev、选中的 accept/sealed、retry、final 和会话文件。不得把 P4 固定 runner 的测试数值用作物理训练预算。provider 请求配置与其尝试/输入/输出额度一起冻结。新会话还须提供 `model_context`：`context_window_tokens`、`max_output_tokens`、`tokenizer_count_source="provider_usage_prompt_tokens"` 和官方能力依据 `specification_source`。输入预约加请求输出不得超过窗口；请求后的实际输入 token 以 provider usage 为准，不能把字节估算当精确 tokenizer 结果。
4. 显式提供 session/output、干净的源码 commit、环境、host、两块 GPU UUID、lease root 与总限额。`--prepare-only` 完成真实 composition 和冻结文件核验，不启动训练或模型请求。首次准备会创建 manifest、cost、continuation、session contract 和任务 final plan。
5. 去掉 `--prepare-only` 后执行同一组参数。首次执行及尚未终结的恢复都在后续工作前检查本地 provider 凭据，不发送网络探测；prepare-only 和只读终态审计不要求凭据。正式入口核对主机恰好拥有冻结的两卡，拒绝重复 index/UUID、错误映射及额外卡。每个物理 attempt 使用独立目录、成本预约和退出回执；接受判决仍以任务协议为准。确定性 public-accept 拒绝可省略 sealed，并明确记录 `not_run_due_to_early_reject`。

在目标项目根查看完整参数：

```bash
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/run_generation_session.py --help
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/create_stair_reference_genesis.py --help
```

`launch_stair_generation_session.py` 与 `run_stair_generation_session.py` 只保留兼容文件名，转发相同参数。旧 `--formal-session`、pilot/validation、readiness receipt 与 launcher 专属参数已经退役，不要照抄历史 shell 文件。

## 暂停、恢复和审计

- `--pause-after-generation N` 在第 N 代完整关账后退出，N 必须大于等于 1 且小于冻结的 `--max-generations`（默认 15 代时为 1–14）；省略暂停参数则连续运行，暂停/恢复仍受冻结的段数和成本上限约束。旧会话只允许其已冻结的暂停边界。恢复使用原 session/output、相同 `--max-generations` 和 `--resume-existing`，继续复用冻结文件。改变代数、cell、成本或合同不是恢复操作；缺失的权威基础文件不会自动重建。5 代验证应使用 `--max-generations 5` 并完成 final，不能用 15 代会话暂停于第 5 代替代。
- Linux CLI 在启动前拥有独立进程组；崩溃恢复仍检查原进程/进程组已经退出，并按现有 attempt/lease/cost 回执恢复。不能手改进程身份来绕过仍在运行的作业。
- 正常执行负责关闭 final 并生成报告。`--resume-existing --terminal-audit-only` 只核验已经完整的终态证据和报告，不补写缺失文件，不运行 GPU 或 provider。未完成 final/报告的会话用普通 resume 完成后再审计。
- 新会话的 final plan 由任务提供。Stair 在 Genesis 前承诺私有 study 实例，恢复复用 `final_confirmation/study_holdout.json`；`--final-replication-source` 明确声明复现实例。Push P4 final 复用已有 sealed cohort，不声明新的 seed 或地形泛化。

单作业 `--runner-timeout-s` 默认 36000 秒，属于执行位置与成本预约；冻结协议的历史 7200 秒字段只保留原身份解释。共享宿主没有第二个 Stair launcher 的总超时参数。provider env 只传文件路径，密钥不写进命令或回执。

## 旧证据与完成标准

旧 V2/旧 generation ledger/report 不迁写为新版本。旧宿主的 Stair v6 报告使用 `scripts/audit_stair_generation_history.py`；新共享报告使用上述正式入口的 `--terminal-audit-only`。历史入口从原成本文件读取上限，校验原始来源和完整报告，不要求 provider 凭据、不采样 GPU、不补造缺失文件。它要求任务协议和后端兼容的审计 checkout，并不是任意旧环境的转换器；需要旧执行条件时从原 Git commit 获取。重放入口与固定格式验证见 [P4 复审](../PLAN/2new_world/P4/AUDIT_FOLLOWUP_ZH.md)。

```bash
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/audit_stair_generation_history.py --help
```

验收必须同时核对 ledger、generation receipts/closures、attempt reservation/result、checkpoint/summary hash、成本与资源闭包，以及 final plan→reservation→phase receipts→result→report。退出成功或 ledger 达到冻结代数不能单独证明完成。

原 Go2 15 代实验的 reference、head、71 个 checkpoint 与原始回执继续保留，入口见 [P0 基线](../PLAN/2new_world/P0/ACCEPTANCE_ZH.md) 和 [实验结果](../PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_RESULTS_ZH.md)。本页不授予新的物理训练预算。

## 两任务的接受边界

P4 历史工程证据中，Stair V3 与 Push reference_v1 共用编排，采用各自冻结的科学判据。该旧 Push 行的 `min_delta=0.05`，使用非配对的噪声余量，主分数下限为 0.05，至少 3 个 accept 样本，没有 named guard；sealed 分数只要求达到绝对下限，没有 public→sealed 下降门。这些数值仅解释 P4 原记录，不是当前 Push 外力评价的规则。Stair 使用配对 headroom 改进、5 个样本、5 个 named guard 和 0.05 的候选 public→sealed 下降门。当前 Push 合同由所选 cell 的 task config、reference contract 和 phased adapter 提供，入口见 [Push 指南](task_extension_guides/push_recovery.md)；不得移植历史 Push 或 Stair 的阈值。

## 目标环境完整回归

在最终代码版本下运行整套测试；`PYTHONPATH` 同时传给测试启动的子进程。动态 Isaac 开关使目标环境执行开发机无法覆盖的配置和资源检查。

```bash
cd /data/yuyangliu/AGENT/true_RL_Agent
PYTHONPATH="$PWD:$PWD/src" TRUE_RL_AGENT_ISAAC_DYNAMIC_TEST=1 \
  /data/yuyangliu/envs/isaaclab6_validation/bin/python \
  -m unittest discover -s tests -t . -v
```

测试发现失败并修改源码后，应重新执行此完整命令。记录使用的源码快照、Python、退出码、failure/error/skip；旧批次的成功 ID 并集不能替代最终代码全量。
