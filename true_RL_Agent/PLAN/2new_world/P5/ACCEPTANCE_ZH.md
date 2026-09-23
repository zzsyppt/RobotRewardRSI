# P5 验收：authoring、反馈选择与机制准入

状态：**GO，2026-09-11 历史回放与上下文复审已闭合**。冻结正式 CLI 回放、内容取舍、未选择 trial 的已测 dev 与失败原因均已修复并直接复验；两端各一次完整 1611 项回归通过。实施与同类问题自审见 [REPLAY_CONTEXT_FOLLOWUP_ZH.md](REPLAY_CONTEXT_FOLLOWUP_ZH.md)。下文区分本轮结果与前一批仍有效的证据。

## 当前实现与边界

- 最终 trial 回执决定恢复终态；训练后任务门拒绝不会在恢复时二次 finalize。
- 未完成 candidate repair 跨 transport/source-request 保留；A authoring 失败时 B 为 not_started。双就绪执行、冻结额度和单次科学提交保持不变。
- 一处 selector 选择最近有训练证据的未接受 trial；源码、runtime 和详细 public-dev 反馈共用身份。active unified 的 prompt feedback 只从 public-dev 派生，不再成为第二份科学数据。
- 当前所有预算按实际 compact UTF-8 计量；14 KiB 跨代记忆保留紧凑机制结果。历史缩进 prompt/容量回执按原格式读取，当前回执显式声明新格式。
- 合成夹具的全零/截断观察不消耗额外 provider 修复；真实训练后的同 transition 差分门保持原范围。数值准入包含同方向比较严格性的调整，不改冻结 delta/hash。
- 已授权的推理耗尽降级在同一 lane 内持续；删除无消费者的 continuation 翻译、无收益独立 authoring 和猜测供应商错误码。

本次未修改 configs、地形、reference、科学评分、成功定义、PPO 或正式训练/接受预算。任务语义留在 task adapter，机器人形态来自绑定 profile，共享层只处理生命周期、身份、通用结果和容量。P6 的 Go2 Push 资格化与 P7 的正式搜索尚未完成。

## 当前历史回放复审的直接证据

| 项目 | 当前结果与边界 |
|---|---|
| 冻结正式回放 | `replay_frozen_cli_final.json`：实际 P4 正式 CLI PASS，251 文件内容及 mtime 不变，原报告 SHA256 `cec022d738bae577a715a1bfa41b839f50514f5f8b5255e65f3a6dd04437941a`。只读复用冻结 authoring 身份，实际续跑不放宽。 |
| 已测但没有选择的 trial | `replay_context_results.json`：c6 G3-A 的 dev 目标 delta `+0.000336`、guard delta `+0.000504` 与原始 dev/parent 回执重算一致。共享层仅验证并读取绑定证据，任务层解释分数；B lane 仍为未测。 |
| 失败原因 | 同一真实 G13 帧保留 `candidate_runtime_failed` 与 `authoring_provider_failed`，未复制原始私有错误；旧八列读入、新九列写出。 |
| 内容取舍 | current-best/latest 保留详情，其余保留关键成绩和主要失败。G13 反馈 36,389→18,483 B，整帧 114,943→102,380 B；G15 保留 28 trial，记忆 14 KiB 上限不变。shared 表头、身份和非零列对齐见复审记录。 |
| 实际分支 | c6 G1/4/7/9/13/15、c8 G9、实际修复帧和受控 source-followup 均重建并独立核对。没有新增 provider 请求、RL 训练或 token 用量测量。 |

详细比较、同类问题自审及本轮最终回归以 [REPLAY_CONTEXT_FOLLOWUP_ZH.md](REPLAY_CONTEXT_FOLLOWUP_ZH.md) 为准。前一批 provider token 数不作为压缩后新帧的实测。

## 前一批直接证据（本轮未追加运行）

| 项目 | 结果与回执 |
|---|---|
| H1 原问题与恢复 | `audit_h1_before.log` 复现二次 finalize；修复后中断、恢复和重复执行不增加已完成作业/门调用，见 `audit_h1_after.log` 及完整 fixed-width 回归。 |
| 两场旧实验及 P4 Genesis | 生产 builder 直接重建成功；G13 保留前 12 代/24 trial，G15 保留前 14 代/28 trial；memory 实发字节等于计量字节。见 `audit_history_after_c6`、`audit_history_after_c8`、`audit_history_after_p4`。 |
| 原 prompt 与回执 | 两场 ledger 绑定的 45 份 prompt 验证通过；P4 的 15 份原始 prompt 及内含任务段回执按旧格式验证通过。见 `audit_original_prompt_artifacts.json`、`audit_historical_receipts.json`。 |
| 实际候选 | `audit_candidate_history.json`：37 个真实训练候选的源码与进场分析器结果保持一致；原 F62 候选被当前参数族规则拒绝。旧训练采样只有 1 个改动项为零；22 个当前合成观察不能替代真实训练证据。 |
| 实际 provider 容量 | `audit_capacity/result.json`：初代/真实晚代/真实失败修复/受控 source-followup 分别为 16669/37367/13391/33645 输入 token，总输入 101072、输出 4；全部低于冻结窗口。 |
| 原实际候选的当前准入 | `audit_provider_readmission.json`：原 provider 候选保持原源码校验值，在当前代码下通过作者 delta 检查、封存资格化及执行授权；没有追加 provider 请求或训练。 |
| 同 lane 推理状态 | `audit_reasoning_resume.log`：耗尽→verifier 失败→中断→恢复后继续 disabled，未启用降级时仍保持原配置。 |
| 当前 Isaac 接口 | `audit_dynamic_tests/result.json`：两项显式 opt-in 测试全部通过，运行约 15 秒，使用生产 process/lease；不冒充正式 RL 实验。 |

容量测量每帧只请求 1 token，预期 finish_reason=length，不是候选生成成功。两场旧实验没有 source-followup 事件，该帧明确是对真实晚代上下文执行实际源码请求服务的受控分支。

## 仍有效的原批次运行证据

原 `provider_live_01/result.json` 在固定 5 次尝试、32768 输出上限内，用 3 次请求取得并封存单机制候选，实际训练入口 admission 为 PASS。共 49639 输入、62573 输出 token。旧实现第三次重新开启思考，本轮已修复；该探针不作为新版持续降级的实测。

原 `runtime_acceptance_summary.json` 的四次实际 Isaac 短 PPO 诊断证明 dispatcher/sidecar/差分链。Go2/G1 的零作用 modify 都为 0/144；非零 modify 都为 128/144，最大加权 rate 差分分别 0.11456525/14.19705772。128 个 PPO 环境步外另有 16 个 activation preflight transition。这些结果不证明策略收益或完整训练性能；本轮未扩大训练后门。

原 `capacity_01` 是初代与预检内容组合，已撤销其“晚代容量”的证据用途；当前脚本直接测量历史重建帧。原失败日志保留用于追溯，不拼接成功结果。

## 最终完整回归与交付

两端最终完整 discovery 均已通过，每端均为一次完整运行，没有合并中止片段或定向重试结果。

| 环境 | 完整 discovery | 结果 |
|---|---:|---|
| 本地 Python | 1611 项，约 1291 s | 0 failure/error，19 项平台/Isaac 安装条件跳过；`replay_local_final_tests.json`。 |
| 目标 Python | 1611 项，约 2737 s | 0 failure/error，仅两项显式 opt-in 跳过；`replay_target_final_tests.json`。 |

本地跳过的四组 Stair 会话及 Linux 进程/符号链接检查在目标完整运行中通过。两项显式 opt-in 物理接口测试本轮未重跑；本轮没有改动其物理执行路径，前一批 `audit_dynamic_tests/result.json` 的两项通过保留原证据范围。

- 本地命令：`.venv/Scripts/python.exe -X utf8 PLAN/2new_world/P5/run_full_tests.py replay_local`；目标 Python 使用同一脚本的 `replay_target` 标签。底层均为 `unittest discover -v -s tests -t .`。
- Ruff `uvx --offline ruff check --select F src scripts tests` 与 `git diff --check` 通过。源文件同步清单 `replay_source_files.json` 覆盖 515 文件，SHA256 为 `cb571c766e5257561a429395b5a0748482d33d3a28599a55ee876d0f0671b9ce`；活动 Python 文件无漏项，两端内容一致，见 `replay_source_sync.json`。
- `replay_protected_after.json` 在最终回归后确认完整 c6 实验的 932 文件、517212889 B、71 checkpoint 与 P0 清单相同。c8 原始 Genesis/ledger 也未变。
- `replay_resource_closeout.json` 确认本轮测试进程无遗留，四个已有 GPU lease 均未上锁；没有改写历史 lease 元数据或停止其他工作负载。
- 本轮逐文件交付清单、源码清单及两端完整测试回执由 `replay_deployment_receipt.json` 绑定。测试后未修改产品代码或测试。前一批 1610 项记录仍可追溯，但不替代本轮验收。
