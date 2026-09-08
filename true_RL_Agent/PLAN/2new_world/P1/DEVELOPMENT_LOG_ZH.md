# P1 开发记录：身份、执行位置与恢复

日期：2026-09-06。范围和实施顺序见 [实施计划](IMPLEMENTATION_PLAN_ZH.md)。本次只处理 P1；没有启动 RL 训练、provider、评测或录像，没有重启服务器。服务器验证使用 `.tmp/p1_validation_20260906` 和后续 `_final` 隔离副本，已安装环境和正式 checkout 未部署修改。

当前状态：**2026-09-07 复审修复后 P1 GO**。混合版本比较、整条聚合算术和相关消费者已修复，最新证据与退役条件见 [复审记录](AUDIT_FOLLOWUP_ZH.md)和 [audit_validation_results.json](audit_validation_results.json)。本文件后文保留 2026-09-06 首轮开发过程及其 [validation_results.json](validation_results.json)，不将首轮通过范围误记为复审覆盖。

## 1. 保持矩阵与所有权

构造边界重新核对了 `PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md` §2 与历史机器人矩阵 P1：只有任务和机器人两根轴，`cell_variant_id` 是唯一持久化协议 epoch。terrain regimes、执行路径、boot、GPU 编号均不新增 cell。任务成功、评分、资格化、分布和机器人尺度仍由已有 task/profile/binding 拥有。

科学身份并非一个新增的“全局超级 hash”。composition 的科学 snapshot、现有 EvaluationBinding/TrainingBinding、backend 环境身份继续各自负责原有合同，最终由 Genesis 绑定。snapshot 保存已有合同的内容投影及其身份，便于说明差异；不会在 core 中复制 Stair 或 Push 配置。

参考本地 Codex 的 rollout/create/resume 与 session 工作位置分离思想，没有移植数据库、调度器或新的执行框架。

## 2. 实际变更

| 问题 | 根部处置 | 明确边界 |
|---|---|---|
| F13 | `ResolvedTaskComposition` 输出 snapshot v3，`scientific` 来自 TaskSpec 固定协议、resolved cell、reference program 与任务明确参数；`execution_placement` 由装配处记录。新 Genesis v6 保存实体并核验科学投影 hash | 不按共享层黑名单从 `task_defaults` 删除几个键。Stair 的科学设置仍由选定 cell 冻结，CLI 默认值不能改写它 |
| F26 | 建立新 reference Genesis 或 import 时，校验并封存独立的 `reference_checkpoint.pt`；eval plan 解析为本会话实物位置 | 原 anchor、来源 Genesis、train/eval receipt 不改写。只解除外部 checkpoint 的运行依赖；没有宣称整个历史来源目录可以删除或任意迁移 |
| F50 | backend 环境 manifest v2 加入 `src/true_rl_agent`、`scripts`、`configs` 的相对文件路径与内容 SHA256。backend 在构造时冻结来源，之后物化与既有 child 环境准入均核对 | 两个新增共享模块只有源码快照和派生统计；没有 Stair/Go2 分支。Python ABI、8 个依赖版本和 5 个优化热路径模块身份保持约束 |
| F18 | readiness 新 writer 产出 v4；历史 probe boot 单独记录，resume 验证历史证据原归属，再采集本次 boot/主机/资源 | fresh launch 仍要求 probe 与当前 boot 一致；旧 v3 reader 按原规则读取。没有修改历史 teardown 来冒充当前证据 |
| F48 | reused trial 先恢复已存在 train receipt，再按原 phase receipts 重建完成前缀；已封失败 trial 遇到缺失 phase 时保持 incomplete | 不重跑训练/评测，不因缺 public result 丢弃已有 checkpoint/train 投影 |
| F56 | private artifacts 仅列实际存在、可计算 checksum 的 summary/checkpoint 文件；计划路径留在原 job plan，目录不冒充文件证据 | dry/failure 路径不创建占位文件；已有文件内容身份不符仍拒绝 |
| F11 | 派生统计 v1 使用精确二进制输入的有理均值/样本方差、固定 Decimal 精度 50 与舍入；ordered summary v3、unified summary v2 显式选择新数值契约；配对 selector/acceptance/final 同步消费版本 | 没有全局放宽 `canonical_json_hash`，没有改原始事实/源码 checksum，也没有修改评分公式、阈值、seed 或任务协议 |
| F19 | 复核现有 typed EvalJobSpec rematerialization 已同步五项身份，保留实现 | 通过既有直接断言和完整 final 两条链验收，没有再加一层 metadata 修补 |
| F55 | 扩展现有 formal fixture，15 代始终 reference-head 后运行两个独立 final job，并在 final 已封后恢复与重复审计 | 该分支通过 authoring-incomplete 保持零接受，验证终态编排；不声称完成了 15 次真实候选训练或证明科学改进 |

## 3. 版本与保留内容

- 新写：composition snapshot v3、Genesis v6、backend 环境 manifest v2、readiness v4、ordered summary v3 / unified summary v2、derived statistics v1。
- 旧格式 reader 保留原 hash 与字段解释；历史原始文件没有批量重算、重签或迁移。旧格式可读取，不等于其 checkpoint 自动获得新后端/协议的 reference 资格。
- 新统计版本只规范派生证据表示。跨 Python 一致性测试不能替代 Python/物理后端兼容性验收；后者仍归 P2。
- 已有完整 15 rounds 的 checkpoint、ledger、协议、回执和外部 P5 anchor 全部保留。本次没有为了身份改造删除旧 checkpoint。

## 4. 验收中发现并处理的事项

第一次完整回归运行了 1608 项，发现 2 个 failure、1 个 error：旧 entrypoint 测试仍访问已退役 snapshot 字段；Push dry-run 测试把未产生的 checkpoint 当实物；恢复测试把读取 authoring journal 的函数进出事件当作 provider 执行。分别更新到新合同，并直接断言 provider 请求数、train/eval 事件与回执投影均不增加，没有删掉原有公开信息隔离或判决断言。

新增 composition 对照曾试图通过 Stair CLI 改 seed/预算。实码显示这些输入由选定 cell 冻结，因此改为验证真实 G1→Go2 cell 切换；Push 则验证其允许配置的 seed/预算变化。保持底层已有所有权，没有为测试开放新入口。

专项恢复/Genesis 89 项和 composition/环境/readiness 33 项已通过。完整回归、Linux 完整终态、跨 Python 数值及历史 reader 结果统一写入最终 [验收记录](ACCEPTANCE_ZH.md)，未完成项不能由这些局部结果代替。

Linux 完整终态首轮 3 项中，candidate-head 15 代和 3 代崩溃恢复通过；新 reference-head 用例的零差断言失败。定位到原 fake runner 按 `final-eval` 槽位名称固定返回候选高分，未考虑该槽位仍评估 reference。只修正夹具按实际 subject 生成结果，保留零配对差断言并补标准误为零；产品逻辑未为测试改写。该用例独立重跑，不重复已通过的其他长用例。

重复终态审计还暴露了测试辅助函数的无用写出：产品 reader 成功返回后，夹具再次创建同名 `formal-worker-audit.json`，导致第二次调用报文件已存在。该文件没有消费者，因此审计分支直接返回，不再另存报告副本；ledger/report 不变及实际 job 不增加的断言保留。

最终消费者审计确认，活动 generation 的 bootstrap、reference screen、finalization 和终态报告均通过 `VerifiedSessionGenesis.reference_eval_plan()` 使用已校验实物。唯一仍直接解释旧 anchor URI 的单 lane final 脚本属于文档明确标出的历史 compatibility 入口，保持旧证据读取边界，不作为新 formal 主线。P1 没有顺带扩建或迁移该旧入口。

最终 reference-head 完整用例在两处夹具修正后通过，用时 490.514 秒；服务器测试文件 SHA256 与本地一致。未重复已经通过且不受夹具修正影响的 candidate-head / 3 代恢复长用例。323 个执行源码文件与通过验收的服务器副本内容身份一致。

本阶段创建的 8 份冗余本地日志/传输包、服务器两个隔离验证目录已删除，清单与实删结果记录在 `cleanup_result.json`。首轮完整终态失败日志保留以解释修正过程；正式 15 rounds 的 checkpoint、历史证据和 P5 anchor 均未删除。最终交付停在 P1 边界，P2 后端迁移另行实施。

临时产物的清理按复用和排障价值判断，按用户最新要求不作为每轮必须清空的验收条件。
