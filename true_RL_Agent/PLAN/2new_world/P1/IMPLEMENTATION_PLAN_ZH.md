# P1 实施计划：证据身份、执行位置与恢复

日期：2026-09-06。主计划：[NEW_WORLD_PHASED_PLAN_ZH.md](../NEW_WORLD_PHASED_PLAN_ZH.md)。前置：[P0 验收](../P0/ACCEPTANCE_ZH.md)与[最后收尾回执](../P0/audit_followup_result.json)均已通过。

执行状态：**P1 已完成，2026-09-07 复审修复后重新验收 GO**。混合版本比较、上游算术、必要边界测试及旧版本退役条件见 [复审记录](AUDIT_FOLLOWUP_ZH.md)；最终结果见 [验收记录](ACCEPTANCE_ZH.md)，首轮实现见 [开发记录](DEVELOPMENT_LOG_ZH.md)。下文保留实施时的步骤与验收要求。

本阶段在现有工作区实现；起点 HEAD 为 `0773a9a815e5a0ccd151f351fca0e369f157fa05`，包含已经验收的 P0 未提交清理和测试修正。旧 `PLAN/Variant_construction/P1` 是机器人矩阵阶段的历史材料，本次开发记录只写入 `PLAN/2new_world/P1`，不覆盖前者。所有既有 15 rounds checkpoint、协议、ledger 和 receipt 保留。

## 1. 目标与不变量

让同一科学条件下的实验能判断自身身份、诊断执行位置变化，并从已有证据恢复；机器重启、路径变化和失败收尾不应制造第二份训练事实或第二次收费。

- 科学条件由现有 task protocol、robot/binding、reward/observation/action、训练设置、源码内容与 backend 环境合同派生；不复制配置定义。
- 路径、PID、boot、设备编号属于执行位置。Python ABI、依赖实现、物理/控制与会影响数值行为的软硬件条件仍需明确绑定或兼容验证；不能以“位置可迁移”为由忽略这些差异。
- task-specific 规则继续由 task 拥有。共享层只处理内容身份、运行位置、文件实物、journal、job 和 receipt；不识别 Stair 成功、Go2 足端或 Push 恢复含义。
- 不修改评分目标、接受阈值、题池、训练预算、PPO、seed、candidate 权限或搜索策略，不推进 P2 后端迁移和 P4 通用 session 抽取。
- 新结构以版本边界落地；旧结构保持原 hash 和 reader，不批量重写历史产物，不增加“忽略 hash”或“强制恢复”入口。

## 2. 已核实问题与处置

原始问题正文统一引用 [F 清单](../../Variant_construction/15_rounds_formal/PHASE6_UNRESOLVED_FOLLOWUP_ZH.md)，这里只记录实现责任。

| 项目 | 当前核对 | 本阶段落点 |
|---|---|---|
| F11 派生浮点身份 | 派生均值/标准差参与回执重建，部分代码直接依赖解释器的统计实现 | 在现有统计/回执边界给派生数值定义版本化规范；原始文件 checksum、整数计数、源码和协议 hash 仍精确验证 |
| F13 科学身份与位置 | composition 把 `task_defaults` 整体入 hash；Genesis 只存 `runtime_snapshot_hash`，无法解释漂移 | 保存版本化 snapshot 实体，科学投影与位置投影由各自已有所有者明确构造；新 import/resume 使用对应版本的兼容规则 |
| F18 跨 boot 恢复 | readiness 的当前 boot 与历史 attempt/teardown probe 的创建 boot 被要求相等 | 保留历史 probe，续接当前启动的 readiness；关系绑定到同一会话、源码与已验证的历史基线，重新取得当前资源证据 |
| F19 final metadata | 当前 `rematerialize_eval_plan()` 已从顶层字段同步五项身份；旧问题描述不能直接视为仍有缺陷 | 通过实际 candidate/reference 两条 final 链验证；若现有逻辑成立，只补必要断言并关闭旧记录，不叠加同步补丁 |
| F26 reference 实物 | 新 formal import 仍可指向会话外的 P5 checkpoint | 新会话只复制一次内容校验后的 reference anchor，提供会话内只读实物；原来源及原 receipt 保持可追溯，不改造旧实验 |
| F48 incomplete 恢复 | `_run_train_lane()` 在 reused trial 缺 public result 时，早退发生在重建 train receipt 之前 | 从同一 trial journal 先重建已存在 train 投影，再保持 incomplete 结论；不重跑训练和评测 |
| F50 child 源码准入 | 正式入口只检查一次 checkout；后续子进程没有逐 job 校验父进程冻结的本项目源码内容 | 在现有 source-checkout、job admission/argv 边界共享一份源码内容身份；启动前和 child admission 校验同一冻结来源，避免任务各写一份 validator |
| F55 reference-head final | 两条源码分支已经存在，缺完整 reference-head 终态实测覆盖 | 扩展现有 formal fake-runner fixture，验证 15 代无接受时的双 final job、相同 checkpoint、配对差近零和 head 不变 |
| F56 planned URI | round 的 private artifacts 无条件从 materialized metadata 收集 checkpoint/summary 计划路径 | 私有实物从实际产生的证据派生；计划输出保留在原 job plan 或明确标为未产生，不伪造文件、不建第二个 artifact registry |

## 3. 借鉴 Codex 的范围

矩阵边界同时遵守 `PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md` §2：只有任务与机器人两根轴，`cell_variant_id` 仍是唯一持久化协议 epoch 字段。P1 的 snapshot 是现有合同的内容证据，不是新 cell 或第三根轴；路径、boot、GPU 编号也不创建新协议 epoch。terrain regimes 不参与 Genesis/ledger/reference 路由。机器人的资产/能力仍通过既有 ResolvedCellManifest 绑定，任务语义仍由 TaskPlugin 提供。

参考仓库：`D:\humanoid_agent\codex`，核对提交 `ddf04ad26789d040f9ef6a96736f76602e35a6cc`，只读。

- `codex-rs/rollout/src/recorder.rs`：逻辑 thread/session identity 与具体 rollout 文件分离；create/resume 是明确的生命周期分支。本项目对应复用已有 session identity 与 journal，不从目录名重新发明实验身份。
- `codex-rs/tui/src/session_resume.rs`：历史中记录的工作位置与本次恢复选择的位置分开处理。本项目对应将旧执行证据保留为历史事实，再验证当前 placement；不把旧 boot 改成新 boot。
- 采用有明确所有者、可从持久记录恢复的边界思想；不移植 Codex 的 UI、数据库、调度器、远端执行框架或庞大适配层。

## 4. 实施顺序

### A. 固定当前契约，先修回执投影

1. 以 P0 的完整回归、文件身份和现有正式产物为基线，记录本次实际修改范围。
2. 修复 F48：复用 train receipt 的时机与正常路径一致，保留原错误状态与原 attempt。
3. 修复 F56：区分作业计划与产生的私有实物；核对正常、训练失败和评测中断三种结果。
4. 复核 F19 的现有 typed-spec rematerialization，避免继续维护无消费者的身份副本或增加局部补丁。

出口：真实 journal 故障用例通过，train/eval 调用次数与收费事实不增，合法原始 receipt 不改写。

### B. 建立版本化身份与 reference 所有权

1. 从现有 TaskSpec/绑定、backend 合同和明确运行参数构造科学 snapshot；placement 单独记录并可诊断。
2. snapshot 实体与其规范身份一起持久化在 Genesis 边界；旧 reader 按原版本解释旧 hash，不能在共享层临时删除几个字段来兼容。
3. 新会话建立/导入时复制 reference checkpoint，校验内容和字节数，封存为会话自有 anchor；重复恢复只读取已封实物。
4. 训练/评测继续消费同一 checkpoint 内容身份。历史 source Genesis、train receipt 与本地 anchor 的位置映射清楚，不改写旧来源来伪装迁移。

出口：路径和 GPU 编号变化的正例、源码/控制/预算等实际变化的反例均有行为验证；外部原 reference 不可用时，新会话仍能使用自己的已封 anchor。旧会话读取保持原样。

### C. 收敛源码准入和跨 boot 恢复

1. 复用并收回现有 source-checkout 职责，以本项目实际执行内容形成一份冻结 snapshot，记录相对逻辑路径及内容 hash；commit 作为可追踪来源，不能用终态 HEAD 代替运行期字节证据。
2. 在共享 runner/backend 的现有 job 准入和参数物化处传递冻结身份；child 在真正作业启动前验证。校验实现不得写入机器人或任务分支。
3. resume 从现有 attempt/trial/cost journal 重建状态，校验旧证据自身的 hash 与归属；新的 host/boot/resource readiness 作为本次 invocation 的证据续接，不重签历史 teardown。
4. 当前源码、环境兼容性或资源不满足时，在作业启动前失败；不能因此重复 train/eval 或增加一次结算。

出口：同 boot、变 boot、重复 resume、源码漂移和路径变化均覆盖；检查一次提交、一次计费、历史证明不变及无残留子进程。

### D. 规范派生统计的身份

1. 优先复用现有统计入口，在最小必要的版本边界定义派生数值规范化；不全局修改 `canonical_json_hash`，不对原始事实宽松比较。
2. 计数由整数事实派生；均值、方差、标准误和配对统计按明确契约生成规范表示。统计容差只用于其声明的派生比较，不能接受错误的文件/receipt 身份。
3. 在实际 Python 3.11 与 3.12 上读取同一组原始输入，复算新规范身份；同时验证旧版本证据读取。

出口：跨解释器一致；真正不同的原始事实仍可区分；不把软件/物理环境的差异误写成“科学等价”。

### E. 完整终态与恢复验收

复用现有测试层级，不新增通用故障注入框架：

| 场景 | 必须观察的结果 |
|---|---|
| train 已完成后退出 | train receipt 被复用；训练不重复、checkpoint 与费用不变 |
| eval timeout 后封 trial，generation 未封即退出 | 恢复后 incomplete 结论和 train 投影与正常失败路径一致 |
| eval phase 已封后退出 | 已封 phase 不再执行；同一 generation 只提交一次 |
| 重复 resume | ledger、attempt、cost 无新增重复事实 |
| boot 改变 | 历史证据保持旧 boot；当前 readiness 验证新 boot 并明确续接 |
| 源码改变 | 下一 child 作业启动前拒绝，不能运行父旧/子新混合字节 |
| 只改变执行路径/位置 | 新身份合同允许经校验的等价位置；科学参数改变仍拒绝 |
| 全程 candidate-head | accepted lineage、最终配对身份、checkpoint 与 ledger 闭合 |
| 15 代始终 reference-head | 两个 final job/slot 不同、checkpoint 相同、配对差近零，accepted_count=0、head_moved=false |
| 训练失败且没有输出 checkpoint/eval | 私有实物清单不声称不存在的文件已产生；计划本身仍可追踪 |
| 旧版本 receipt / Genesis / ledger | 原始字节不改写，按旧版本读取，不伪造新版本字段 |

## 5. 验证成本和交付

- 日常运行直接受影响的现有单测/集成；收尾执行本地完整回归。只在新增改动、失败或具体未解问题需要时重跑。
- Linux/procfs、进程组、实际 Python 3.11/3.12 和 formal 终态使用服务器现有环境及隔离的验证工作区；不部署到现有正式 checkout，不覆盖其产物。
- 本阶段不启动真实 RL 训练、评测、provider 或录像，不重启服务器；boot 变化用已有可注入 host/procfs 证据做确定性验证。formal fixture 的 fake runner 只证明编排与恢复，不形成科学提升结论。
- 正式交付保留实施计划、必要开发日志、简洁验收记录和机器可读结果；日志引用已有测试与原始证据，不复制整套旧报告。临时产物按复用、排障价值决定保留，不要求每轮清空。
- 被替代且确认无活消费者的实现同步删除；保留必须读取的历史格式，不为“未来可能用”增加包装器或开关。
- P1 GO 要求上述行为检查闭合，关键环境 skip 得到真实补测，源码/层级/旧证据审计通过。F19 等已成立的现有实现如经验证无需改动，明确记录证据，不为凑修改量重写。

实际实现中若字段、版本或模块落点需要调整，以代码依赖和行为验收为依据，在开发日志记录原因；本节的权威归属、旧证据保护和 P1 范围不变。
