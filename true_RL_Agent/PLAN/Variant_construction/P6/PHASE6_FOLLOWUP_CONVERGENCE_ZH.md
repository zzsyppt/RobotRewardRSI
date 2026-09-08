# Phase 6 follow-up 收敛记录

## 1. 目标与边界

本轮只收敛 `PHASE6_FOLLOWUP_IMPROVEMENTS_ZH.md` 中会影响下一轮 Go2 × Stair 正式实验的真实问题。机器人形态仍归机器人行，楼梯评测、失败分类和提示语义仍归 Stair 任务；训练、评测、seed、预算、PPO 与验收权威不进入 LLM 权限。没有重建会话框架，也没有修改冻结评测结果来追认结论。

## 2. 逐项决定

| 项 | 决定 | 收敛结果或依据 |
|---|---|---|
| F1 `joint_ood` 名称歧义 | 修复 | 不改冻结协议标签；任务提示明确其含义是楼梯高度与踏深同时作标量外推，不是机器人关节状态 OOD。 |
| F2 诊断被表述成因果证据 | 修复 | 提示与 reward input 契约统一声明：诊断是可重叠的历史/近期相关量，可与成功共存，只用于机制假设，不能证明因果。接触历史长度从绑定 ABI 派生，不再写死为 3。 |
| F3 `goal_progress_fraction` 时序语义缺失 | 修复 | 明确它是每控制步重算、可回退、非锁存的空间位置；完成事件仍由任务拥有的 `first_valid_top_arrival` 表达。 |
| F4 `rename` 可读投影失真 | 修复 | receipt 身份不变，新增派生的 `operation_effect`；仅名字变化为 `pure_rename`，名字与结构/参数同时变化为 `rename_and_modify`。 |
| F5 lane/提示统计错误及无选择仍双跑 `public_accept` | 修复 | 两条 lane 均完成 train + public_dev；只有 public_dev 选中的 lane 进入 public_accept 和 sealed，无选择时两者均不运行。报告从 formal/selected trial 解析 lane，并从全部 authoring journal attempts 统计提示次数与字节。旧 dual-public ledger 仍可回放，但不能参与当前 selector。预算槽位保留为冻结最大上界，不会启动作业。 |
| F6 中间 checkpoint 占用过大 | 修复 | 最终 checkpoint 已复制、逐字节哈希一致且运行/预算证据闭合后，只删除该 job 独占 RSL-RL 目录中的 `model_*.pt` 中间文件，保留最终迭代与所有非 checkpoint 日志。 |
| F7 `task_scales` 进入身份但不执行 | 暂缓 | 当前 V2 已冻结并完成资格化。此时让字段突然生效或删除都会重定义 cell；应在后继协议中选择唯一事实源并重新资格化，不作为本轮表面补丁。 |
| F8 无生产消费者的两套 registry | 删除 | 删除 `TaskRegistry` / `BackendRegistry` 开发脚手架；扩展文档统一指向实际使用的 `TaskCompositionRegistry`，不再保留平行注册真相。 |
| F9 no-selection 后最新失败源码缺失 | 修复 | 公开反馈继续使用 task 投影；源码解析单独使用已验证 ledger identity 与 runner-owned artifact refs。generation 按最新未接受 lane 解析对应 artifact，并新增真实 fixed-width 集成覆盖。 |
| F10 零消费者符号 | 部分清理 | 删除确定为模块私有且零消费者的 `_is_required_feedback_entry`、`_GATE_FIELDS`。显式导出的常量/Protocol/数据类可能有仓库外消费者，没有弃用契约，保留；全量回归曾实际证明“本文件未使用”不等于“可删除的重导出”。 |
| F11 跨 Python 版本 1 ULP hash | 暂缓 | 正式环境解释器内可复算，差异不改变排序或判决。现在量化会改变历史 receipt；后继 schema 再对派生统计量做版本化 canonicalization。 |
| F12 容量先删证据后删静态文本 | 修复 | 先压缩每轮重复的 task/ABI 静态描述，再动已验证反馈；每条压缩链只生成一个直接绑定原始 sections 的 receipt，禁止 receipt 叠层。服务器真实第三代在 40/44/48/52 KiB 下仍会删除逐 cell/factor 证据，56 KiB 才完整保留，因此 task-section 上限版本化提升到 56 KiB；新回执为 v2，历史 v1 仍按原 32 KiB 重算。 |
| F13 Genesis 含设备/绝对路径且不存快照体 | 暂缓 | 属身份 schema 迁移，不阻断同一声明环境的下一轮实验；需以后版本化拆分科学身份与执行位置，不能在本轮改历史 Genesis。 |
| F14 cost contract `isinstance` 分派 | 暂缓 | 当前 V4 运行与复验均明确，重写会扩散到预算/恢复/receipt 多层；没有当前表现或正确性证据支持在实验前重构。 |
| F15 会话编排仍绑定 Stair | 后续 Phase | 这是任务轴的后续通用化工作，不是 Go2 × Stair 本轮阻断项；禁止在实验前重写约 9k 行会话编排。 |
| F16 审计 loader 位于任务产品模块 | 暂缓 | 仅位置/依赖方向问题，不进入训练、评测、判决热路径；迁移收益不足以承担现阶段兼容成本。 |

## 3. 容量与信息保真门

容量不是单一的“字节不超限”门。下一轮实验前必须同时满足：

1. current-best 与最新失败候选源码仍可解析；
2. `family_scores`、qualified family、stratum diagnostics、失败分类和 runtime effect 没有因压缩消失；
3. 任务 sections 不超过当前 56 KiB，完整 provider payload 不超过 256 KiB；
4. 若服务器真实三代重放仍需删除上述信息，则升级 receipt schema 后适度扩容，不允许用裁掉高价值反馈换绿灯。

服务器现有 Go2 三代真实 ledger 的第三代只读阈值标定：

- 40/44/48/52 KiB：均落到 `static_contract_and_feedback_core`，删除 `success_rate_by_cell`、`factor_diagnostics` 与逐 cell progress；
- 56 KiB：落到只压缩重复静态描述的 `static_contract_core`，task sections 为 54,652 B，完整 provider payload 为 94,287 B；8 个 stratum、失败分类、诊断率、runtime activation、逐 cell/factor/progress 以及 current-best/latest-failed 源码均保留；
- 94,287 B 仅占 256 KiB 总上限约 36.0%，没有把信息恢复转化为上下文失控。

当前本地最坏形状预检仍是故意塞满合法字段的压力形状：G1 V2 为 38,744 B，Go2 V2 为 38,098 B，均落到 `minimum_core`，但同时保留 current-best 与 latest-evaluated 两条反馈，不再退化为单条。该压力形状只负责证明硬上界闭合；真实三代则负责证明高价值信息没有因容量投影消失。

## 4. 本地验收

- `python -m compileall -q src/true_rl_agent tests`：通过；
- 定向 prompt/source/generation/retention 回归：通过；
- fixed-width 真实集成路径：通过；
- Windows 全量：`Ran 1584 tests in 251.360s`，`OK (skipped=29)`；
- `git diff --check`：通过。

本轮未启动训练、评测或 provider API 请求。

## 5. 服务器验收

- 产品代码提交 `17255889f2db0f5dd8d0646acad98baf8dd5b91b` 已 fast-forward 到 `humanoid-server`，tracked tree 无漂移；
- 项目 IsaacLab 解释器全量：`Ran 1584 tests in 949.468s`，`OK (skipped=3)`；
- 已通过 `load_completed_stair_generation_session` 复验现有 `p6-go2-validation3-v4-successor` 三代 typed closure；
- 提交后只读重放的 task/provider 字节分别为：generation 1 `50,284 / 78,806`，generation 2 `54,664 / 87,956`，generation 3 `54,652 / 94,287`；三代 provider task receipt 均为 v2，`max_serialized_bytes = 57,344`；
- 三代均只使用 `static_contract_core` 去除重复 task identity 与 reward 字段说明；完整 `factor_diagnostics`、`success_rate_by_cell`、逐 cell progress、失败分类与诊断率仍在。第三代 `stratum_diagnostics` 为 8 层完整映射，runtime activation 可用；
- generation 1/2 保留 current-best 内联源码；generation 3 同时保留 current-best 与 latest-failed 内联源码；
- 结束后无 Stair/Isaac 进程，两张 GPU 均为 `0% / 6 MiB`。

本轮没有启动训练、评测或 provider API 请求。上述证据关闭 F12 的信息保真风险；是否进入新的正式 Phase 6 实验仍应由正式实验前的独立 GO 检查决定，不能把本次静态/重放验收冒充策略表现结果。

## 6. 本轮提示证据去冗余复核（后续追加）

本节记录对现有三代产物按当前产品代码进行的只读重放，不改写原始 evaluator summary、ledger、selection 或历史 receipt。上一节的字节数是当时实现的历史证据，不应覆盖；本节才是当前提示投影结果。

- 真实 generation 1/2/3 每次 provider 调用均只有一条完整反馈，双条反馈是压力预检形状，不是生产事实；
- factor diagnostics 保留全部 factor group、level、cell count、success rate 与 mean progress，只把每轮重复的固定 cell ID 列表投影成带 schema 的表；饱和 factor 仍保留，不以“无改进空间”为由删除能力证据；
- stratum diagnostics 保留所有非零失败率和诊断率；在 first-valid-top-arrival 语义下，只把与 success rate 恒等的 `first_valid_top_arrival_achieved` 从提示投影中去重。`goal_not_reached_detected` 还要求最大进度未达到目标，不是失败率的补集，继续保留其真实动态证据；
- 当前三代 task sections 分别为 `45,862 B`、`49,857 B`、`49,845 B`，均无需容量降级；factor/stratum 动态块由第三代原来的约 `10,543/6,220 B` 降为 `3,518/2,107 B`；
- 压力预检原始合法形状为 `72,751 B`，投影后为 `41,052 B`，仍保留两条反馈；56 KiB task-section 上限不需要继续扩大；
- 历史第三代 lane-b 曾把高 slip rate 当作因果先兆并新增 `contact_slip_excess`，其 `J_OOD` 相对 parent 回退约 `0.06444` 后被拒绝；该候选早于 F2 语义修复，当前尚无新 provider 请求，故只列入观察文档，不宣称风险复现或实验关闭。

本轮新旧事项分离如下：尚未解决的工程项见 `15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md`；仅需后续真实运行观察的事项见 `15_rounds_formal/PHASE6_OBSERVATIONS_ZH.md`。

## 7. completion 派生关系纠正（开跑前复核）

对 Phase 6 pilot 与三代 validation 的 18 个 eval phase、432 个 cell-phase 独立复算：`first_valid_top_arrival_achieved == success` 为 `432/432`；`goal_not_reached_detected == 1-success` 仅为 `430/432`。两个反例均已到达目标进度，但分别因 `forbidden_contact` 与 `off_course` 失败。提示层因此只派生并省略 arrival 列，恢复 `goal_not_reached_detected` 的全局与逐 stratum 真实测量；原始 evaluator summary、ledger 与历史 receipt 不变。
