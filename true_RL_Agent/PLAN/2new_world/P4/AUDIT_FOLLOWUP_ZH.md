# P4 复审与实施记录

2026-09-10。基于已交付但未提交的 P4 工作树复审。当前修改前的 98 个源码/测试文件与 `final_source_snapshot.json` 一致。本轮完成前，不沿用上一轮 GO 作为修复后的验收结论。

## 顺序与边界

1. 修复 holdout 几何碰撞、恢复凭据和 GPU 清点：先统一实际几何展开，再检查源卷和卷内重复；只在提交前对结构性碰撞重抽。恢复、replication 和既有科学解释不变。GPU 格式/清点归 runner，固定两卡要求归正式入口。
2. 收敛终态及历史接口：finalizer 唯一拥有关闭后的完整终态重放，循环保留关闭前成本/资源核验；final 作业后的资源关闭独立核对。将证据/成本接口从活动循环中移出，不复制第二套算法；历史格式需要固定样本回归及可使用的只读入口。
3. 校正验收与文档：使用真实 Stair V3 phase 投影和明确预期值检查早拒；记录 Push 工程 cell 与 Stair 的门差异，不在 P4 偷换 Push 科学协议。补齐目标环境可直接运行的全量命令。
4. 对最终源码运行本地与目标全量、历史只读回放和相关闭环验证，再同步代码/记录并逐项复核。本轮全量结果单独保存，不把前一轮成功 ID 的并集称为本轮全量。

## 初步判断

| 审计项 | 当前证据与处置 |
|---|---|
| P4-1 | 成立。`freeze` 在源卷重合检查前就可能报 sealed 内重复；在同一几何展开路径上统一判定两类结构性碰撞，限制首次抽样次数。 |
| P4-2 | 前一轮记录确为分批闭合，不能替代本轮最终源码全量。用户新运行结果是背景；本轮继续取得自己的完整回执。 |
| P4-3 | 成立。恢复被 `not resume_existing` 排除；恢复到需要执行的会话也必须检查本地凭据，离线准备/终态审计保持独立。 |
| P4-4 | 成立。内联 dict 丢失重复 index 与严格格式信息；恢复清点合同，并复用 runner 的 GPU UUID 校验。 |
| P4-5 | 成立。`**_boundary` 丢弃调用方的已验证终态；收敛所有权与参数，区分代循环关闭和 final 作业后资源证据。 |
| P4-6/7 | 私有跨界和历史读取无正式入口成立；但部分所谓“成本算法”只是 receipt 的 path/hash/event 投影，共享稳定规则本身不是二源。需要形成明确的证据/成本公共接口，固定历史格式回归，提供只读入口。 |
| P4-8 | 核对真实协议后记录差异；P4 工程等价不表示任务门等强，Push 科学判据修订归 P6。 |
| P4-9 | “完全恒真”不准确：现有测试另有独立的 accepted 布尔预期。但实际 Stair 分数投影缺少直接验证，已补充真实任务证据与独立预期。 |
| P4-10 | README 已有目标命令，runbook 缺少；补到运行者实际阅读的位置。 |
| 次要项 | GPU 正则单源化；Push sealed evidence 必须显式接收协议。报告文件名先核对是否同一语义，避免为一个字符串引入无用层。 |

全程保留原完整 15 rounds 的 checkpoint 与证据，不开展新科学训练或按 reference 成绩调题。

## 已实现的源头修复

- Holdout 的 `bind_suite` 在 `freeze` 前使用同一个 `resolve_stair_suite_cells` 展开实际几何，同时检查 sealed 内重复和与所有来源题面重合。首次创建最多抽 64 次，只有该结构性碰撞可以重抽；其他错误原样报出，replication 一次即失败且不写文件。G1/Go2 两个卷内碰撞 nonce 见 `holdout_internal_collision_before.json`；此前源卷碰撞用例保留。固定有效 nonce 的完整 suite hash 仍与迁移前相同，见 `audit_holdout_frozen_mapping.json`。没有修改偏移分布、成功率门或根据 rollout 结果挑题。
- CLI 的凭据谓词覆盖新建和未终结恢复；准备、已有报告及显式终态审计不查 provider。GPU 的完整 UUID 规则统一归 runner，清点保留全部行并拒绝畸形/重复记录，正式 CLI 单独要求主机恰好为冻结的两卡。录像退出探针也复用同一 UUID 校验。
- 删除 finalizer 的 `**_boundary`，删除循环在调用 finalizer 前的完整 terminal 重放和未消费参数。关闭前 prospective terminal 只核对计数、成本头及资源；关闭后由 finalizer 重放完整 ledger/lifecycle/attempt。最终计时覆盖这一遍完整重放，final 作业后仍核对新的资源关闭范围。
- 成本查找、引用投影、实际量核对归 `generation_cost_journal`，计费结算归 `generation_accounting`；closure 的 v1 格式和已有 execution receipt 读取归 `generation_closure`。历史 reader 不再依赖活动循环。operation/resource/provider/job-evidence 的真实跨模块接口改为公共名称并修正调用方，Push 分数/质量投影同样由任务层公开。没有复制历史算法、保留旧私有别名或引入通用工作流框架。
- `tests/fixtures/generation_cost_closure_cd48221.json` 是迁移前完整会话产出的固定成本/closure 样本，含来源 hash。回归直接比对既有数值、字段和序列化字节，只替换测试机的本地路径。它与完整历史只读重放分别保护 wire 格式和证据链，期望值不是调用当前成本函数现场生成。
- 增加 `scripts/audit_stair_generation_history.py` 正式只读入口，读取冻结成本上限，保留原来源与当前审计 checkout 的区别。旧配置移除无消费者的 provider-env 字段；测试停止把 `generation_history` 伪装成已删 live session。历史接口只重放兼容任务/后端下的旧 Stair v6 报告，不声称自动适配任意旧环境。
- 真实 Stair V3 phase 测试同时覆盖 G1/Go2，用不同 stratum 成功数、非恒定的 5 项配对向量，独立计算 J_OOD 和五项 guard 预期，检查字段映射、顺序、版本和协议身份。保留原通用早拒的独立结果预期及真实共享会话早拒集成场景。

## 两任务门强度的诚实边界

| 条件 | Stair unified V3 | Push reference_v1 |
|---|---|---|
| `min_delta` | 0.0，另有 20% headroom 与配对 SE 余量 | 0.05，另有非配对噪声余量 |
| 主分数绝对下限 | 0.2 | 0.05 |
| accept 最少样本 | 5 | 3 |
| named guard | 5 | 0 |
| 配对样本比较 | 是 | 否 |
| 候选 public→sealed 最大下降 | 0.05 | 未声明此门 |
| sealed 行为分数门 | 候选检查分数一致性；reference 单独处理 | 均值至少 0.05 |
| P4 final 的新研究泛化声明 | 事前承诺新 study，结果仍需实际测量 | 复用原 sealed cohort，不声明新 seed 泛化 |

Push 的 public-accept 0.9 / sealed 0.06 可以通过其 sealed 分数门，但仍须通过其他接受条件。不能写成“Push 没有 sealed 验证”，也不能把工程编排等价写成科学门等强。P6 已补记独立验证域与对应接受判据的预声明要求；应由 Push 的恢复目的决定，不能机械照搬 Stair 阈值或看过 holdout 后改门。

## 同类问题复审与复杂度边界

`audit_interfaces.json` 记录跨模块私有导入由 48 个名称降至 13 个，P4 新编排、历史回放、Push phase 之间的 35 个已收敛。剩余项是既有 Genesis/lifecycle 的 10 个进程和日志内部工具、Stair 标定的一个 surface-control 工具及原 resource-shadow 的两个训练探针工具；不为清零计数重写 P1 进程身份或扩展旧探针框架。本次共享 generation 模块没有 task 层导入。

新报告与旧 Stair 报告均叫 `generation_session_report.json`，但 schema、写入所有者和读取入口不同。保留各自命名常量，避免历史 reader 反向依赖新应用，也不为同名字符串新建兼容层。Push sealed projector 取消默认协议；已有 public 原始摘要工具的默认协议仍仅提供其原有检查/诊断语义，没有把该默认带入正式 sealed 构造。

主计划中指向已删 Stair session 的链接已改为共享宿主。另核实：当前 V3 protocol 实际引用文件名为 `unified_evaluation_suite_v2.json` 的题池，保留这个真实入口，不能按 protocol 的版本号猜改文件名。runbook 补齐目标环境 `PYTHONPATH`、动态开关及完整 discovery 命令。

参考 Codex 的范围仍限于统一生命周期所有权和持久化后的完成声明。这里采用“关闭后的完整终态由 finalizer 唯一构造”，没有引入新的 task trait、事件总线或第二套调度系统。

## 验收过程

第一版复审全量暴露本轮接口迁移漏导入：暂停边界调用 `assert_actual_cost_receipt_matches_usage` 时发生 `NameError`。已补齐其成本模块导入，真实 Push accepted→暂停→恢复→final 场景通过后，重新冻结 `audit_final_v2_source_snapshot.json` 并重跑两端整套测试。中止批次的说明及原日志见 `audit_aborted_batch.json`，不能充当最终成功覆盖。

最终版本的[本地全量](audit_local_full_v2.json)已完成：1599 项，0 failure、0 error，19 项因 Windows 或目标 Isaac 条件跳过，耗时 972.390 秒。测试后核对 102 个源码/测试文件的 hash 均与冻结快照一致。[目标环境完整 discovery](audit_target_full_v2.json)也已完成：1599 项，0 failure、0 error、0 skip，耗时 2694.560 秒。逐项从本批日志核对 1599 个唯一成功 ID，本地 19 个跳过项全部在目标环境成功，见[最终核对](audit_final_verification.json)。这些结果来自最终 v2 代码各自的一次完整运行，没有合并之前批次的成功记录。

[最终版本历史 CLI 复核](audit_historical_cli_v2.json)重新构造迁移前的完整 15 代 fixture 报告，251 个文件的字节、mtime 和 SHA256 均未变化。它使用记录明确列出的固定 backend/source 测试替身，不冒充原 Go2 科学实验的目标后端重放。另将首次 P4 归档与当前实现的七个成本/closure 函数比较：仅归一化列出的公开符号改名及 docstring 后，完整 AST 均相同，见[迁移核对](audit_cost_closure_migration.json)；旧样本的固定字段/字节断言仍独立执行。

**本轮 P4 复审工程 GO。** [源码同步回执](audit_source_deployment_receipt.json)记录服务器主检出更新 36 个文件、保留 66 个相同文件，全部 102 个源码/测试文件与最终验收快照一致；11 个已退役路径继续不存在，本次没有新增删除。同步前 114 个相关路径与前次交付一致，未覆盖并行修改。

[同步后保护检查](audit_protected_after.json)按 P0 的原完整 Go2 15 代保护范围核对 932 个文件、517212889 字节；其中 71 个 checkpoint 全部核对 SHA256，未发现变化。[部署入口检查](audit_deployed_cli.json)中的共享 CLI、两个旧命令转发及历史只读 CLI 均可运行。

[资源收尾](audit_resource_closeout.json)确认本轮全量进程已退出，两张 GPU 分别使用 7/6 MiB、计算进程为空。没有运行新的科学训练、外部 provider 请求或更改评测题池参数。本次未创建 Git commit；后续真实研究仍须冻结实测成本、合格 reference 和最终源码 commit。P5、Go2 Push 物理资格化 P6 及真实搜索 P7 不在本轮工程 GO 内。
