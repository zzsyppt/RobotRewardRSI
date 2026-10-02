# P3 验收与交付记录

日期：2026-09-24。状态：工程、真实生产重放、48/48 作者实验及匿名材料交付已完成，根复核通过。人工价值门待评；不是 Phase 3 质量收益门通过声明。

用户选择先交付全部实现、实验和盲评材料，人工评阅待补。**人工价值门没有通过，Phase 4 未启动。** 工程正确、模型生成完成、建议事实可靠与实际 RL 收益是不同结论。

## 实际架构

数值采集档案 → 任务内确定性行为派生 → 封存在 public-dev 回执的有限视图 → 任务公开投影 → 正式作者上下文。公用 generation 层只传递已绑定的真实 parent 回执，不解释姿态、脚、台阶或扰动。生产实现位于 `src` / `scripts`，不依赖本目录。

Stair 保留有符号根部姿态与位移范围、逐身体接触和移动事件、持续条件、进展窗口、第一立面锚点和终局前时间线。根部参考点不是通用“躯干”；支撑身体参考点不是脚尖。接触传感器历史是缓冲测量，不能恢复子步顺序。低净空不是碰撞证明，成功回合也可能出现打滑。

Push 保留稳定条件分别失守的区间、稳定计时被打断、恢复达标及其后的再次失稳。Velocity 协议按每次扰动对齐；force 协议按完整序列最后一次脉冲结束对齐。已经恢复过不代表最终成功。根部相对扰动方向的运动不能证明迈步策略。

观察和奖励权限分别登记：瞬时量可作为奖励代理；历史持续窗口、未来观察窗和精确表面查询不会因此变成奖励输入。Stair 既有 ABI 已提供当前 `forbidden_contact` mask 与瞬时禁触力，可以使用；禁触力历史缓冲没有暴露，不能将当前 mask 当成历史序列。不增加候选权限，不改变评分、接受标准或 PPO。

## 上下文与记忆边界

完整行为导出留给复核。作者视图保留所有 stratum 的分母、原成功/失败口径、按固定规则选出的有限代表与省略说明。成功反证和失败代表先于第三个可选例子；不会为了满足字节上限删空时间线。

Stair 封存 profile 上限 8 KiB，Push 12 KiB。Stair 的来源包装及合法父差分另计，并全部进入 56 KiB task sections 限额。8 KiB 不是整个作者请求大小。长期记忆仍为 14 KiB，旧历史不逐条追加时间线。旧回执缺字段就显示缺失，构建 prompt 时不回读 NPZ 补写历史。

Push 保持既有反馈对象身份；没有对应 public-dev 时不把别的策略的行为挂上去。当前 best 和最近失败的统一受保护记忆槽仍属于 Phase 5。

## 验收范围与证据

| 项目 | 已验证范围 | 证据 |
|---|---|---|
| 历史档案终局 | 11 组、5,472 回合；Stair 两形态，Push 两形态及 velocity/force；原成功和失败数一致 | `results/projection_capacity_r4.json`、各组 provenance；机器语义报告见下 |
| Stair 语义核对 | 全档对照 P2 冻结的旧三类事件、锚点及终局；此部分共享 decoder，不能当成独立物理真值。新增谓词另用独立 raw 算术抽核 6 个真实回合、2,647 步、1,421 个区间，含两例规则平面净空 | `results/stair_independent_semantics.json`、`results/stair_raw_event_spotchecks_r3.json` |
| Push 独立核对 | 7 个真实预选过程实例，条件、锚点、dwell、恢复后失稳及方向 | `results/push_semantic_spotchecks_r2.json` |
| 旧历史兼容 | Stair 33 个旧 phase、Push 15 个旧 phase 重建一致；既有记忆保留量达标 | `verification_stair_history_replay_r3.json`、`results/history_replay_push/report.json` |
| 正式 generation 入口 | 正常选优与无选优路径均传递已绑定 parent public-dev | `verification_generation_r4.log`：2 tests，60.463 s，OK |
| 公共边界与记忆 | 通用项目边界、已有记忆、训练采样公开投影 | `verification_shared.log`：56 tests，OK |
| Stair 语义与生产边界 | 105 项 CPU 测试通过；含身体重排反例、镜像符号、8,185 B 高熵两代表容量 | `verification_stair_r5.log`，最终运行 24.918 s，OK |
| Push 生产与容量 | 真实 evaluator checksum 接线；两代表及合法差分容量回归 | `results/push_validation_r4.log`：29 tests；`results/push_diff_capacity_r4.json` |
| 作者流程 | 真实资格 → builder → 静态准入/CPU smoke → author journal；离线 transport 仅验接线 | `author_validation.log`：6 tests，11.586 s，OK |
| 输出偏差与总预算 | 实际输出不截断，输入边界/总单位额度保持；无总额余量时仍阻断，未知调用保守预留 | `verification_provider_accounting_r7.log`：5 tests（含多个边界子场景），2.762 s，OK |
| 服务恢复与预算 | 401/402/403 原失败回执保留、未知用量全额预留、恢复不扩尝试额度；普通未知调用仍阻断 | `verification_provider_service_recovery_r6.log`：111 tests，36.370 s，OK |

Stair 的同源码历史 fca→Gen8 可作真实父子描述比较；其他源码或协议不同的组合显式不可比。Push 没有已证明的父子配对就登记缺失。合成容量 fixture 只验证尺寸和保留规则，不当成真实策略比较。

数值时间轴是派生测量图，**不是原生仿真视频或视觉验收**。旧 PyBullet 视频与本批档案不同源，不借其证明当前语义。镜像/身体重排测试只验证数据轴、符号和规则映射，不证明物理镜像运动一致。

## 作者实验与人工交付

预登记 12 个焦点问题，Stair / Push 各 6，每臂两次新生成，总计 48 个作者 episode。底层只有 2 份观察包、2 个策略簇；重复生成反映作者波动，不能当成独立谱系。A/T 共享资格父本、数据时点、源码权限、模型、修复政策和全部预算，仅 behavior 视图及其派生容量计量不同；Stair 另登记由两份合法公开投影分别计算的 `source_public_result_hash`，该精确差异必须通过 canonical 对象重算核对，不能放宽任意来源差异。

资格来源、案例、预算、模型与提示对照在输出前冻结。静态/CPU smoke 通过只表示候选符合当前合同，不证明它在仿真中有效。失败作者 episode 保留在分母。机器引用定位只说明引用对象是否出现，不能替代人工判断数值含义和事实支持率。

已交付 48 份作者终态、5 个隐藏复阅项（53 项材料）、空评分表、实际作者最后可见输入、共同父源码、两份完整 public-dev 真值与四张派生时间轴。48 项均通过静态准入和 CPU 检查；1080 个文件逐 hash/size 验证通过。随机映射与评阅材料分开。人工还需评阅事实支持、反证、权限、动作相关性及可检验性；允许不确定。一个评阅者只能报告自身复阅一致性。

评阅入口：[匿名材料索引](results/author_final_bundle/author_blind_review/reviewer/INDEX_ZH.md)、[仅含评阅材料的 ZIP](results/author_final_bundle/author_blind_review_reviewer.zip)。评分填入索引同目录的 `review_sheet.json`；若要保持盲评，不打开其上一级的 `withhold_from_reviewer.json` 和 `withhold`。审计入口：[最终汇总](results/author_summary_final.json)、[完整清单](results/author_final_bundle.json)、[根复核](results/final_root_delivery_review.json)。

本次事前选择质量收益门，费用作描述，不在看输出后改走成本优势路线。达到 95% 事实支持率与两个过程类别的决策相关改善必须等人工结果；当前状态为 `pending_human_review`。

## 资源与限制

阶段额度 provider USD 15、GPU capacity 2 h，不执行候选训练。截至全部真实采集结束，5 次运行（含失败）合计 2,236.0073 GPU capacity s，即 0.6211 h。所有失败、替代重放都计入资源，详见 `results/resource_accounting.json`；GPU lease 时长与纯 rollout 时长分别报告。

最多 240 次 provider 请求包含源码跟进、修复和未知结局，无免费重试池。40k 输入是预留，实际 provider usage 为权威，不称为精确 tokenizer 上限证明。输入/费用超限或模型身份漂移保留回执并停止，不丢弃不利结果。

最终 92 个账本槽：89 次有权威 usage，两次 HTTP 402 和一次已保存 prompt、发送前被旧守卫拦截的槽没有 usage，仍按完整预留计入。已知用量按事前峰价折算 USD 2.618139576，三次未知预留 USD 0.1539648，合计 **USD 2.772104376 / 15**；这是记录用量加保守预留，不是服务商实付账单，也不包括 Codex 工作费用。48 个原 episode 的机会没有扩容，最大实际占用为 4/5，原源码查询限制 2 次、每次 2 个 ID、48 KiB 均保持。全部实际响应模型为 `deepseek-flash`，指纹一致。

两次 HTTP 402 均在用户确认恢复额度后从原账本继续；旧失败不改写，既有结果不重做。r6 修复已记录服务拒绝的保守恢复规则；r7 区分请求输出限制与事后真实消耗。一次 Push T 请求 32,768 输出 units，实际回执为 32,769 且最终正文为空；该偏差完整保留、实际计费不截断，不能宣称原单次输出合同完全合规。总 input/output/费用和尝试上限仍保持。四段执行的日志与停止现场均保留，恢复说明见 [r6修订](AUTHOR_RECOVERY_AMENDMENT_R6_ZH.md)、[r7修订](AUTHOR_OUTPUT_ACCOUNTING_AMENDMENT_R7_ZH.md)。

| 任务与臂 | 作者终态 | 占用尝试 | 峰价实际或预留 USD |
|---|---:|---:|---:|
| Stair A | 12 | 23 | 0.723629328 |
| Stair T | 12 | 24 | 0.769932384 |
| Push A | 12 | 18 | 0.519567492 |
| Push T | 12 | 27 | 0.758975172 |

费用只作描述；服务中断、输出偏差、修复次数及同策略问题相关性都在，不能把臂间差异全部归因于反馈表达。没有进行候选 RL 训练，静态通过也不说明建议内容正确。

最终 Stair 语义边界及新增事件独立抽核已完成，见 `results/stair_boundary_review_r5.json` 与 `results/stair_raw_event_spotchecks_r3.json`。冻结版真实 producer 已通过，Stair profile 8,187 B、864 回合；完整导出逐值重现封存视图。48 次作者实验与匿名材料已完成根复核，详见 `results/final_root_delivery_review.json`。人工与 RL 效果不在本轮伪造闭合。

## 可重跑入口

本地在仓库根目录，使用 `.venv/Scripts/python.exe`，设 `PYTHONPATH=src`、`PYTHONUTF8=1`。相关命令：

```powershell
.venv/Scripts/python.exe -m unittest tests.unit.test_stair_behavior tests.unit.test_stair_numeric_capture tests.unit.test_stair_phased_evaluation tests.unit.test_stair_public_projection tests.unit.test_stair_transition_materialization tests.unit.test_stair_morphology_projection tests.unit.test_stair_termination_capture tests.unit.test_stair_complete_reward_contract -q
.venv/Scripts/python.exe -m unittest tests.unit.test_push_behavior tests.unit.test_push_recovery_v2_prompt_smoke tests.unit.test_push_phased_evaluation tests.unit.test_push_force_consumers -q
.venv/Scripts/python.exe -m unittest tests.unit.test_project_boundaries tests.unit.test_memory_context_phase13 tests.unit.test_training_sampling_projection -q
.venv/Scripts/python.exe -m unittest tests.integration.test_fixed_width_generation -q
.venv/Scripts/python.exe PLAN/feedback_sys/P3/author_checks.py
```

真实档案及模拟器位于服务器，不能把 Windows CPU 测试当成仿真。采集命令登记在 `simulator_cases.json` 与远端每次 `*.launch.json`；独立 verifier 报告内保存实际参数、源码 hash 和输入 hash。行为采集冻结见 `source_snapshot_r5.json`；公共预算恢复与输出计量修订见 `source_snapshot_r6.json`、`source_snapshot_r7.json`，远端源码目录同名。不覆盖旧目录重跑，也不修改旧 phase 来让新实现通过。旧 r2/r3/r4/r5 的报告保留它们实际验证的版本。

## 机器核查的范围与缺失

原 `results/author_partial_qa_r7.json` 核对了全部 47 项实际末次输入，显式身份标记警告为 0；这不等于评阅者完全无法推测反馈样式。后续 `results/author_parent_reward_supplement_r7.json` 只补审阅资料：三个修复末次输入没有父程序，使用已绑定且与冻结首包相等的实际 attempt 1 核对父源码，再派生 before/after 奖励项符号。其中两个 Stair 的已保存 raw/clip/weight 样本算术可直接补核。末次输入原件及当时缺少 current_best/public_feedback_summary 的事实保持，不替作者补写可见上下文。

最终 48 项均有父子奖励项符号对照，24 项 Stair 已保存样本算术通过；24 项 Push 原 CPU 回执只保存绝对值范围与计数，未保存逐样本 raw，故逐样本算术仍为 N/A。不得把绝对值极值当成带符号证据，也未为填表重新执行候选。以上都不是奖励意图正确、事实支持率达标或 RL 收益证明。原 47 项报告作为停止时快照保留；最终全量见 `results/author_final_delivery_checks.json`。三个父本补充对应四个匿名评阅项（含一项隐藏复阅）；含臂名和内部路径的原补充仅放审计区，匿名附件只引用评阅资料。所有事实支持、数值含义及奖励意图保持人工待评。
