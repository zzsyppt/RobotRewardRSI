# P6 进场审计处置

2026-09-11。本记录承接 P5 的残留审计；旧 P5 验收、原始 prompt、科学账本与回执不改写。专项回归只覆盖以下工程变动，不代表四格真实物理验收或 P6 完成。

| 项目 | 当前处置与实际影响 | 保留边界 / 证据 |
|---|---|---|
| Stair 私有失败 reason 白名单、Push 缺少同类 reason | 由共享宿主公开原因投影统一修复，属于本轮公开反馈链工作。任务不应维护宿主生命周期 code 的第二份白名单。 | 共享原因改动及完整链回归由最终验收统一列证据，本表不以小项专项替代其验收。 |
| 八列记忆自动补成九列 | 删除 `_compact_entry` 的 `len(row) == 8` 补列。当前内部 writer `_trial_outcome` 已产九列；未找到正式执行入口把旧八列记忆读入这个重压缩函数。 | P5 保存的旧 prompt 按原字节验证；从 ledger 重建时使用当前 writer。旧 prompt 存在不等于兼容分支有执行消费者。不新增版本转换器或形式校验框架。 |
| Stair minimum-core 版本未随结构变化更新 | 当前 feedback schema 与常规 capacity projection 改为 v6，single-entry capacity projection 改为 v2。 | 新结构保留共享表头及行含义；旧封存 prompt 不迁写，也不为它增加新的重投影 reader。 |
| Push runtime 按整条 record 选来源 | 改用 `latest_nonaccepted_trial`，跨未训练代选择同一 trained nonaccepted trial；没有此项时保留有训练证据的 accepted current-best。输出包含 source trial/candidate/program 身份，runtime schema 改为 v2。 | 同一代 A accepted / B nonaccepted 时取 B；B 缺 runtime 时明确 unavailable，不能借 A 的观测或编造零次调用。无历史训练来源仍为 no-prior；Genesis reference runtime 未进入现有 Push prompt 输入，因此不编造参考观测。 |
| generation ledger v3 启用 exact exclusion | 保留 `generation_lanes.py` 的版本边界，本轮不另建 policy registry。 | P5 DEVLOG 的版本消费者表记录 P4 与既有实验 v1/v2 的冻结解码/回放；当前新入口写 v3。删除边界或重解释旧合同需单独版本迁移，不由清理暗中改变。 |
| 冻结的 False 字段 | 保留 `failed_lane_does_not_cancel_sibling_authoring=False`。它是 authoring policy 的明确冻结事实，不是仍在工作的独立 authoring 分支。 | `generation_preparation.py` 将整个 authoring policy hash 写入 session contract。P5 已封存 v2 policy；删除字段会改变合同 hash，并不能减少现有执行分支。 |
| 复用失败 reason 的 phase 后缀 | 当前检查发现多个同类拼接位置，未发现生产 writer 向 trial decision 写已带 phase 后缀 reason 的证据，因此不宣称已经发生双后缀污染。 | fresh 失败以基础 reason 封存 decision；phase 只进入 lane status。宿主 reason 所有者统一检查，本小项不另加规范化 helper。 |
| source helper 名称与行为不符 | `_latest_failed_generation_trial_source` 改名 `_latest_nonaccepted_generation_trial_source`，唯一内部调用同步改名。 | 函数实际解析 selected trained nonaccepted lane；包括没有失败的未接受 sibling。无公开别名或兼容包装。 |
| exact exclusion 临时 dict | 保留现有有界 assessment/repair dict，不新增 dataclass、注册器或平行分析器。 | assessment 写入封存 attempt，并由 `prompt_artifacts.project_authoring_status` 读取 unresolved reason；exact key 仍由同一 canonical delta 派生，旧 exploration card hash 保持原算法。 |

参考此前明确留存范围：`../P5/DEVLOG_ZH.md` 的“版本与保留消费者”表及 exact hash 说明、`../P5/REPLAY_CONTEXT_FOLLOWUP_ZH.md` 的原始 prompt/正式历史只读回放说明。

## 小项专项验证

执行命令：

```text
.venv/Scripts/python.exe -B -m unittest tests.unit.test_push_recovery_task tests.unit.test_memory_context_phase13 tests.unit.test_prompt_capacity_projection tests.unit.test_prompt_source_catalog -q
```

首次运行 151 项，7 errors / 1 failure：七个纯诊断 fixture 没有 candidate/train 身份，另一个刻意无效的 decision 过去仍被 runtime 读取。没有为旧 fixture 加运行时 fallback；诊断 fixture 补齐已有训练与候选证据，无效 decision 的断言改为不能提供当前源码的 runtime。

新增组合回归覆盖：accepted sibling 与 nonaccepted sibling、后续未训练代、选定 trial 缺 runtime 不借其他 trial，以及没有 nonaccepted 时保留 accepted current-best。修正后 152 项全部通过，14.360 秒。同期 `git diff --check` 通过。测试时其他 P6 分区仍在修改，最终需对整体冻结代码运行完整回归；本记录不声称此时全仓验收完成。

## 真实历史与冻结回放

服务器只部署 A 阶段的 17 个源文件及新的 `audit_shared_public_history.py`。18 文件逐项 SHA256 与本地一致，见 `audit_shared_public_history_deployment.json`；源清单见 `audit_shared_public_history_sources.json`。没有上传 P6 新 Push 的未完成源文件。

`shared_history_c6_01/result.json` 为原 c6 实验的实际只读结果。脚本调用生产 `generation_runtime.generation_public_history`，给只读 foundation 明确绑定 `comp.task_plugin.phased_evaluation_adapter()`，再调用实际 task scheduler 与同一 adapter 的 `PublicContextBuilder`。没有启动完整会话准备、provider、runner、物理环境或新训练，也没有 patch c6 的 policy 或科学回放。

| 下一代提示 | prompt bytes | memory bytes | 保留 trial | 结果 |
|---|---:|---:|---:|---|
| G4 | 94,946 | 8,307 | 6 | PASS |
| G13 | 102,263 | 14,080 | 24 | PASS |

两帧都保留 G3-A 的目标分差 `+0.000336`、guard 分差 `+0.000504`，共享历史、scheduler 和 prompt 取同一 parent checkpoint `sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`。G3-A 被判为 positive / medium / eligible，但未自动获得 priority；两条调度 lane 仍为 novel。未测 G3-B 没有 objective/guard 分差。G13 保留此前 12 代及全部 24 trial，memory 在 14 KiB 上限内；完整 prompt 的 private-safety 检查通过。五处 adapter 注入另由源码 AST 核对，不把该检查称为五个独立正式会话实跑。

全 c6 当前目录的 1,271 文件、619,640,092 bytes（包括 71 个 `.pt` checkpoint）在前后 size、mtime_ns、SHA256 比较中全部不变，见同目录 `protected_before.json` 与 `protected_after.json`。这是当前完整目录范围，不能与 P0 的 932 文件保护子集混为同一统计口径。c6 脚本耗时 121.67 秒，执行进程返回 0。

P4 使用既有且未改写的 `../P5/audit_frozen_replay.py` 对 `/data/yuyangliu/AGENT/p4_baseline_cd48221` 执行生产正式 CLI，结果 `frozen_p4_replay_01.json` 为 PASS；251 文件的 bytes、mtime_ns、SHA256 不变，原报告 hash 仍为 `cec022d738bae577a715a1bfa41b839f50514f5f8b5255e65f3a6dd04437941a`。沿用 P4 原验收的 backend/checkout fixture override，仅用于该 fixture，不是物理或策略验证。回放耗时 21.13 秒，进程返回 0。

两条回放的输出、完整新 prompt、保护快照、日志及执行回执已下载。`shared_history_download_verification.json` 对 12 份下载产物逐项校验 SHA256 全部通过。原 P5 脚本、结果与历史科学文件未修改。以上证据闭合 A 阶段此次共享公开历史修复；P6 四格物理与策略资格化仍需后续独立验收。


### P6 V2 Push prompt、同源反馈与 smoke 接入

仅 `config.push_protocol` 非空走 V2：明确 `push_complete_recovery_v2` 完整奖励程序、实际注入当前事件、L / sqrt(gL) / sqrt(L/g) 空间尺度、reward rates 总和由 task RewardManager 在 reset 前只乘一次 dt；完整恢复与事件/episode 成功计数仍由 task controller 拥有。V2 保留既有三个候选 capability，不增加权限。

V2 task history 在共享 authority identity 保持不变的条件下，仅从 canonical public-dev phase summary 投影分数、恢复计数、主分类和可用四档 n/N；feedback 与 runtime 共用 latest trained nonaccepted / accepted current-best 来源选择。接受分数与 private canary 不进入完整 provider 或 minimum-core，sealed gate 只有 status。新旧协议输入与旧消费者分支独立保留。

smoke 使用 G1 37 / Go2 12 实际关节与动作维数并提供全部 V2 字段；新 reference 和改动候选均实际调用通过，错维访问被拒绝。初轮发现 shared SmokeTensor 缺 exp，补充三行 unary 方法；初轮 full-provider 发现投影未保留 authority 字段，已按共享 identity 契约修正。最终四模块 130 tests PASS / 21.815 s，其中新专项 8 tests；Ruff F、diff check PASS。证据及逐文件哈希见 `push_v2_prompt_smoke_validation.json`。未执行 GPU/API/模拟器或策略训练，不将 authoring smoke 作为恢复效果证据。


### 全量回归后的旧 bootstrap 审计入口边界

`run_push_recovery_loop_audit` 仍有 `run_loop --mode dry-run-audit` 与 `run_round` CLI 消费者；它固定 synthetic 初始分数 0.10、15 个预置候选，属于旧 bootstrap 审计，不是 reference 训练资格验证。保留入口，在任何 audit Genesis/ledger/candidate 写入前拒绝非 bootstrap composition，并明确要求 `--cell-variant-id push.g1_29dof.legacy_v1`。未改新 reference 默认值，未向新 reference 填入假 bootstrap 分数。

旧 integration helper 与相应 CLI/materialize fixtures 显式 legacy，原 15 轮、接受门、ledger、prompt 断言全部保留。新增 3 个 reference cells 和两个默认 CLI 的拒绝/无 audit 产物边界。最终 `tests.integration.test_dry_run_audit_loop` 12 tests PASS / 66.295 s，Ruff F、diff check PASS；日志 `legacy_audit_regression_02.log`，逐文件哈希 `legacy_audit_boundary_validation.json`。
