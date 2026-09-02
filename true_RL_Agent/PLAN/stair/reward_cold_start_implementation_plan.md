# Stair 奖励源码冷启动详细实施计划

## 0. 目标与结论

本阶段在现有 Stair 完整奖励闭环上增加一种显式的 `reward_cold_start` 会话启动模式，同时保留现有 `reference` 模式并继续作为默认值。

两种模式必须共享同一套已经冻结和验证的科学合同：

- G1、楼梯环境、完整奖励 ABI 与 pre-reset 奖励权威；
- `4096 x 24 x 2000` 固定训练预算；
- 相同网络、PPO/optimizer、训练 seed 和随机策略初始状态；
- 相同七桶 public/sealed 评估、失败闭合、acceptance floor、`min_delta` 与 required gates；
- 相同 verifier、runner、checkpoint、seal、ledger、replay 和 provider 边界。

两种模式只允许在 session 起点上不同：

| 模式 | Genesis | 初始 current-best | 首轮提示词源码 | 首次接受 |
| --- | --- | --- | --- | --- |
| `reference` | `reference_program` | 正式 R0 完整评估 | R0 完整奖励源码 | 相对 R0 做 noise-aware 比较 |
| `reward_cold_start` | `bootstrap` | 无已评估奖励程序 | 无父奖励源码，LLM 从公开合同生成完整奖励 | 通过既有 `absolute_accept_floor` 后成为首个 accepted head |

这里的“冷启动”指**奖励源码冷启动**，不是放弃公平的策略初始化。每轮策略仍从零随机初始化训练，但必须在 PPO 开始前匹配同一个已验证的随机策略状态哈希。冷启动不得使用 R0 checkpoint、R0 奖励源码、R0 评估结果或 accepted checkpoint 作为 warm start。

Push Recovery 只能作为 bootstrap 控制流参考，不能被描述为等价实验：Push 当前在没有 accepted candidate 时仍保留环境基础奖励，而 Stair 的 `complete_reward_program` 会完全替换环境奖励，因此 Stair 冷启动的首个候选必须独立给出可训练的完整 locomotion/stability/stair reward。

## 1. 当前基座与不可破坏边界

### 1.1 已具备的能力

- Stair 完整奖励由 task-owned pre-reset runtime 单一计算，PPO、episode accumulator、日志和 evidence 消费同一权威值。
- candidate artifact 已经过 seal、静态验证、源码身份绑定和运行前字节复核。
- bootstrap current-best、无父源码 authoring、best-observed bootstrap feedback 和首个候选 absolute floor 接受逻辑已存在于共享主干。
- reference Genesis 已绑定 R0 源码、训练合同、随机策略初始状态、checkpoint、public/sealed eval 和资格门。
- 现有正式 R0/R9/R14 证据已证明固定随机初始化状态哈希；实现只从 task-owned start-profile 信任锚读取该值，计划文档不复制派生哈希。该值只用于随机初始化公平性，不授予任何奖励信息或策略权重。

### 1.2 必须保持不变

- 不修改冻结的 `stair_protocol_v2.json`、训练预算、评估 seed/bin、success/failure 语义和 acceptance 参数。
- 不修改现有 `reference` 模式产生的 TaskSpec、task defaults snapshot、evaluation binding 或 Genesis 验证结果。
- 不迁移、不重写现有 reference Genesis、R0/R9/R14 ledger、checkpoint 或 prompt artifact。
- 默认任务仍是 Push；显式选择 Stair 时默认启动模式仍是 `reference`。
- Stair 高度、失败原因、奖励字段、冷启动 profile 和提示内容留在 `tasks/stair_climbing/`，共享层只表达 task-neutral 的 Genesis/初始化证据合同。

## 2. 源头设计

### 2.1 Task-owned session start profile

新增 Stair 自有的不可变 session-start profile，公开两个严格枚举值：

- `reference`：`genesis_anchor_kind=reference_program`，需要 verified official reference artifact；
- `reward_cold_start`：`genesis_anchor_kind=bootstrap`，`initial_best_score=0.0`，绑定固定随机策略状态哈希，不允许携带 reference/baseline artifact。

profile 只描述 session 起点，不复制 V2 的训练、评估、楼梯或 acceptance 语义。profile identity 由以下字段派生：profile ID、Genesis kind、Final V2 hash、formal TrainingBinding hash、初始化状态哈希。这样配置或训练绑定漂移时会在训练前失败，而不是生成一个看似可比较的 session。

`reference` 的 resolved `task_defaults` 保持当前字节语义，避免破坏旧 Genesis runtime snapshot；只有新建的 `reward_cold_start` snapshot 才增加显式 start-profile identity。该兼容分支只处理既有持久 artifact 的版本边界，不复制生产执行路径。

### 2.2 Bootstrap initialization anchor

对共享 `BootstrapGenesisAnchor` 做最窄的 task-neutral 扩展：允许携带可选的 `initial_policy_state_hash`。旧 bootstrap Genesis 缺少该字段时继续按旧合同加载；新 Stair cold-start Genesis 必须包含它。

统一行为：

1. composition 把 task-owned profile 的初始化哈希交给 Genesis 创建器；
2. Genesis canonical bytes 和 runtime snapshot 同时绑定该哈希；
3. `VerifiedSessionGenesis.expected_initial_policy_state_hash` 对 reference 和带锚的 bootstrap 返回同一抽象值；
4. 现有 backend、TrainJobSpec、argv、Stair runtime、TrainJobResult、checkpoint lineage、eval job、seal 和 replay 继续复用当前 pairing 链；
5. 哈希错误时必须在 PPO `runner.learn()` 前失败，且不得产出可接受的 checkpoint/eval/decision。

不新增“首轮训练后自我认证”的弱证据，也不把候选运行结果反向写入 immutable Genesis。初始化锚来自已完成的 runner-owned 正式随机初始化证据，并由相同 TrainingBinding 在实际冷启动训练中重新验证。

### 2.3 唯一 composition 路径

`StairClimbingCompositionProvider` 改为先解析严格 start mode，再决定是否解析 reference：

- `reference`：恰好一次解析并验证官方 R0 artifact，行为与当前生产路径一致；
- `reward_cold_start`：不得调用 reference resolver，resolved composition 中 reference/baseline 必须均为空；
- 两种模式均构造同一个 Stair TaskConfig、backend、runner、verifier、memory adapter 和完整奖励 runtime；
- plugin 只根据 task-owned start profile 投影 `genesis_anchor_policy`，不把 start mode 注入共享 TaskSpec 字段。

生产 registry 继续只注册一个 lazy Stair provider。generic `run_loop.py` 只消费 resolved composition，不出现 Stair 枚举、R0 路径或初始化哈希。

### 2.4 Prompt、反馈与记忆语义

冷启动第一轮必须满足：

- `current_program.status=none`，parent/source checksum 为 `null`；
- 不出现 R0/reference source、R0 score、reference evaluation 或 reference qualification；
- 明确要求输出完整 `training_code/components.py`，不能输出 delta、权重 patch 或依赖隐藏基础奖励；
- 保留 RewardInputs、目标、约束、完整奖励 ABI、评估定义和 failure diagnostics 等真实公开合同。

后续轮次继续使用现有 accepted-only source inheritance：

- 首个通过 absolute floor 的候选成为 accepted current-head；
- denied 源码不成为 parent，但其公开结果可进入 bounded feedback/best-observed 辅助分析；
- accepted 后只携带当前 accepted 完整源码、必要的 current-best 证据、最近失败和 best nonaccepted，避免逐轮无限累积；
- private seed、checkpoint 路径、R0 私有证据和未公开 sealed 细节不得进入 provider prompt。

## 3. 分阶段实施

### 阶段 C0：基座冻结与计划发布

**目标**

- 记录当前 local/origin HEAD、worktree 和正式 R0 初始化哈希来源；
- 确认 reference 模式、Push 默认路径、现有 replay 和测试基线；
- 发布本计划后再改代码。

**严格验收**

- 当前工作区用户改动不被覆盖；
- 已确认 Stair 当前是“策略随机初始化 + R0 奖励源码 reference”，不是奖励源码冷启动；
- 已确认 Push bootstrap 仍叠加基础奖励，不能作为 Stair 能力结论。

### 阶段 C1：Stair start profile 与 composition 收敛

**目标**

- 新增 task-owned start profile 及严格 resolver；
- Stair composition CLI 增加 `start_mode`，默认 `reference`；
- provider 在 cold-start 路径完全跳过 official reference resolver；
- plugin 根据 profile 选择 `reference_program` 或 `bootstrap`。

**严格验收**

- 非字符串、未知 mode、错误 profile/training-binding/hash 均在 composition resolve 阶段拒绝；
- `reference` 默认 composition 的 TaskSpec、task defaults snapshot 和 reference artifact 身份与修改前一致；
- `reward_cold_start` composition 为 random + bootstrap，reference/baseline 均为空，且 reference resolver 调用次数为 0；
- Push provider、默认 registry 和 Push snapshot 无任何变化。

### 阶段 C2：Bootstrap 初始化状态证据闭环

**目标**

- 扩展 bootstrap Genesis 的可选初始化哈希；
- 让 composition -> Genesis -> train spec -> argv -> runtime -> result/checkpoint -> eval/seal -> ledger replay 使用同一哈希；
- 保持旧 bootstrap Genesis 和全部 reference Genesis 可加载、可重放。

**严格验收**

- 新 cold-start Genesis canonical bytes 包含唯一初始化哈希，修改、删除、重复键或错配均 fail closed；
- 首轮候选与后续候选都在 PPO 前验证 actual hash 等于 Genesis hash；
- checkpoint 和 eval job pairing hash 闭合，ledger replay 对任一篡改拒绝 continuation；
- 旧 Push bootstrap 无初始化哈希时保持现有行为；旧 Stair reference Genesis 的 canonical bytes 和 hash 不变。

### 阶段 C3：入口、prompt 与 dry-run/materialization 验收

**目标**

- generic live 入口从 composition 传递 bootstrap 初始化锚；
- Stair thin wrapper 允许显式 cold-start 新 Genesis，同时 reference continuation 仍要求原有 Genesis；
- materialize、dry-run/audit 和 prompt preflight 复用同一个 resolved composition；
- 冷启动首轮提示词不含 reference 残留。

**严格验收**

- cold-start materialized train job 是 candidate subject、无 init checkpoint、包含 expected initial policy hash；
- argv 与 TrainJobSpec 完整绑定且 checkpoint/hash 参数唯一；
- 第一轮 source catalog 无 parent，完整奖励能力和输入合同齐全；
- reference 模式首轮 prompt 仍以 R0 为 current source；
- 任一模式错误复用对方 Genesis 时在 provider 调用前拒绝。

### 阶段 C4：本地严格回归与故障注入

**目标**

- 覆盖 start-mode 隔离、Genesis migration、初始化 pairing、prompt source、acceptance/replay 和旧任务回归；
- 删除实现中无调用者的兼容 helper 或重复 resolver，不保留开发偏差分支。

**严格验收**

- 定向 unit/integration tests 全通过；
- 标准 `unittest discover`、全量 pytest、Ruff `E9,F,I`、compileall、`git diff --check` 全通过且无新增 warning/误收集；
- 故障注入覆盖：cold 模式解析 R0、reference 缺失误伤 cold、错误初始化哈希、Genesis mode 漂移、R0 源码泄漏、accepted/denied parent 漂移；
- shared 目录扫描不含 Stair 高度、七桶、失败原因、RewardInputs、R0 ID 或 start-mode 枚举。

### 阶段 C5：提交、推送、服务器同步与真实 preflight

**目标**

- 在本地验收干净后提交并推送；
- 服务器正式仓库 fast-forward 到同一 commit；
- 在不启动 2000-iteration PPO 前完成 server composition、Genesis、prompt 和 materialization preflight。

**严格验收**

- local HEAD、origin HEAD、server HEAD 完全一致且正式仓库 clean；
- server reference 模式可 load-only 重放旧 Genesis；
- server cold-start 新建 bootstrap Genesis 不训练 R0、不解析 R0 artifact，首轮 prompt 无 R0；
- preflight 不遗留 simulator/Python 子进程，GPU 回到运行前基线。

### 阶段 C6：冷启动 15-round 正式实验

**目标**

- 使用 `/data/yuyangliu/secrets/deepseek.env`、新 session、新 output root 和固定 15 rounds；
- 每个可训练候选均从相同随机策略状态、固定 2000 iterations 预算训练，完成 public/sealed 七桶评估；
- 持续监控 provider attempts、prompt、反馈、source parent、memory、训练/eval 状态、进程和 GPU。

**运行纪律**

- 正式运行前冻结 commit、命令、session ID、output root、provider 配置和实验审计模板；
- 只允许因硬错误、证据失真、提示词错误、反馈不可行动、memory 漂移或基础设施失败而停机修复；
- 候选能力差或未接受本身不是修改协议、预算、seed、acceptance floor 或评估定义的理由；
- 每次修复必须本地验收、提交、推送、同步，再从新的 clean session/epoch 重启；不得手工改 ledger、Genesis、candidate、checkpoint 或 result。

**严格验收**

- 15 条轮次记录完整闭合，或每个未进入训练的轮次有合法 bounded failure evidence；
- 每个成功训练候选的 TrainingBinding、initial policy hash、预算、checkpoint lineage、public/sealed binding 一致；
- load-only replay 恢复的 current-best/current-head 与在线结果一致；
- 无 R0 source/score 进入冷启动 prompt；accepted 之前 parent 始终为空，accepted 之后只继承最新 accepted 源码；
- 运行结束后清理本实验产生的进程，不影响服务器其他任务。

### 阶段 C7：结果、公平性、prompt/feedback/memory 审计与迭代

**目标**

- 逐轮分析冷启动是否产生真实上楼能力，而不是 score 漏洞；
- 与同预算 R0/reference 模式做解释清楚的对照；
- 发现系统性不足后按优先级修复并在新 clean 实验中验证。

**公平性审计**

- 核对 score 由每 seed 七桶等权宏平均形成，success + primary failure 数量闭合；
- 同时审计各高度 success、max progress、top hold、base contact/fall/slip/clearance/pitch 等诊断，禁止只看总分；
- 检查 public 与 sealed 趋势、方差、standard error、required gates 和最高高度能力；
- 将“超过 bootstrap floor”“超过 R0”“超过 R14”“0.40 m 成功”分别报告，不能互相替代；
- accepted 只证明在冻结协议下相对 current-best 改善，不自动证明奖励机制因果归因。

**Prompt/feedback/memory 审计**

- 对实际发送 payload 做 section 级字节/token 统计、重复片段检测和跨轮增长检查；
- 检查 RewardInputs、完整奖励 ABI、当前源码、失败证据和 acceptance 目标是否足量且相互一致；
- 检查反馈是否把七桶表现、主要失败、进度、稳定性和上一候选机制绑定到同一 candidate/source；
- 检查 memory 是否保留 accepted current source、current-best、最近关键失败和 best nonaccepted，同时丢弃重复、过旧、不可行动和私有内容；
- 将 Codex 的 typed context、稳定前缀、增量历史和独立 compaction metadata 思路作为结构参考，但只在真实 payload 证据显示重复或增长问题时改动。

**最终验收**

- 输出逐轮表、accepted lineage、能力分桶、失败迁移、prompt/feedback/memory 审计和 R0/R14 对照；
- 每个“能力提升”结论都有 success/progress/failure/sealed 证据，不以单一 scalar 代替；
- 若迭代修复，旧实验保留只读诊断身份，新实验使用新 commit 和新 session；
- local/origin/server 最终 commit 一致，测试干净，资源清理完成。

## 4. 预计修改面

Task-owned：

- `src/true_rl_agent/tasks/stair_climbing/session_start.py`
- `src/true_rl_agent/tasks/stair_climbing/composition.py`
- `src/true_rl_agent/tasks/stair_climbing/plugin.py`
- Stair composition、prompt、entrypoint 和 production tests

Task-neutral 最窄扩展：

- `src/true_rl_agent/composition/base.py`
- `src/true_rl_agent/agent_core/session_genesis.py`
- `scripts/run_loop.py`
- Genesis、live loop、replay 和 compatibility tests

应用 wiring：

- `src/true_rl_agent/composition/defaults.py` 只负责惰性 provider 构造，不包含 Stair start-mode 语义。

原则上不修改 reward runtime、Final V2、acceptance、evaluation、seed resolver、PPO/agent config 和完整奖励 ABI。只有后续真实故障证据证明这些实现存在硬错误时，才回到对应源头修复并重跑受影响验收。

## 5. 完成定义

本目标只有同时满足以下条件才算完成：

1. 现有 R0/reference 模式和历史 replay 完全不退化；
2. Stair 可显式创建无父奖励源码、无 R0 证据、无 warm start 的 bootstrap session；
3. 冷启动所有轮次严格匹配同一随机策略初始状态和固定训练/评估合同；
4. 首个 accepted 源码由 LLM 从公开合同完整生成，之后只继承 accepted 源码；
5. Push 默认路径和旧任务插件行为不变，任务特殊语义未进入共享层；
6. 本地与服务器严格验收通过，代码和证据 commit 可重放；
7. 完成至少一次干净的 15-round 冷启动实验及公平性、prompt、feedback、memory 和能力效果审计；
8. 实验暴露的硬错误已在源头修复并由新的 clean 运行验证，最终进程/GPU 和仓库状态干净。
