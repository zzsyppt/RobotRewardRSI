# Stair 阶段 E 详细实施计划：正式 R0 与 ReferenceProgramGenesisAnchor

## 1. 阶段定位

本计划以提交 `51911dc542416757b03995934b1ea37787261240` 的 Phase D 结果为不可随意改写的基座。

Phase D 已冻结并验证以下事实，Phase E 只消费，不重新定义：

- 正式训练预算为 `4096 envs × 24 steps × 2000 iterations = 196,608,000 env-steps`；
- 正式训练使用随机初始化，不允许 warm-start；
- 正式训练 seed 域与 Phase D pilot seed 域隔离；
- production TrainingBinding、Final Stair V2、正式七桶评估协议和 reference capability gate 已冻结；
- 官方 R0 是仓库内 `training_code/components.py` 的完整奖励源码；
- eval 只加载 checkpoint，不加载 reference/candidate 奖励源码。

除非发现会导致正式 R0 证据不可信、无法执行或破坏后续循环的硬性错误，本阶段不修改 Phase D 的奖励、环境、训练分布、2000-iteration 预算、协议阈值和 pilot 证据。

## 2. 阶段目标

在 Final Stair V2 下，通过正式 Runner 完成一次官方 R0 的随机初始化训练和固定协议评估，并以一个 create-once 的 `session_genesis.json` 原子建立 session 起点。

完成后必须满足：

1. R0 与后续 LLM candidate 使用同一训练配置和同一 Runner 信任边界；
2. R0 的 reference 身份、实际执行源码、TrainingBinding、checkpoint、评估结果和 seal 形成单一链路；
3. Stair 资格门由 Stair 插件计算，共享层只保存不透明 gate 和证据 hash；
4. candidate ledger 在 Genesis 后仍为空；
5. initial current-best 和 prompt current-head 可仅从 Genesis 及其私有证据恢复；
6. R0 任一硬门失败时不创建 Genesis，也不调用 provider。

## 3. 不做的事情

- 不复用 Phase D pilot checkpoint 作为正式 R0；
- 不把 pilot 结果写进正式 current-best、ledger 或 prompt；
- 不修改 acceptance delta、absolute floor 或 reference capability floor；
- 不把楼梯高度分桶、成功定义、progress 阈值或失败分类加入共享 Runner/Genesis；
- 不把 Stair 注册进默认生产 registry；
- 不实现 R1/R2 provider 循环，那属于阶段 F；
- 不为失败的 R0 伪造 candidate acceptance 或 sequence-0 ledger 记录；
- 不新增 warm-start、课程或其他未来权限。

## 4. 当前边界与真实缺口

### 4.1 已可直接复用

- `VerifiedReferenceProgramArtifact` 已绑定官方源码真实字节、manifest 和 capability profile；
- `ExperimentSubject(kind="reference")`、`TrainJobResult`、`EvalJobSpec`、可信 eval seal 已支持 reference 身份；
- Stair eval 入口已支持 `candidate/reference`，且不加载训练奖励源码；
- `ReferenceProgramGenesisAnchor`、reference current-best 恢复、prompt source head 和 ledger replay 已有通用合同；
- Stair public projection 已严格重算七桶公开统计、失败闭合、预算和 runtime activation；
- SubprocessRunner 已统一执行 checkpoint 字节预检、summary 身份复核和 sealed validation；
- `session_genesis.json` 已有 canonical bytes、create-once 和 reload 校验。

### 4.2 必须补齐

1. reference 源码目前只有内存中的可信字节，没有 Runner 可执行的不可变本地 artifact；
2. backend 只有 candidate train materializer，没有 reference train materializer；
3. `TrainJobSpec` 和 train argv 校验只绑定 candidate artifact；
4. Stair train 入口只接受 `--candidate`，无法加载并再次验证 reference artifact；
5. `prepare_session_genesis()` 对 reference 仍主动报错，没有正式 train/eval/publish 流程；
6. reference capability gate 还没有基于 Final V2 正式 public_accept 结果的 task-owned 投影；
7. reference 私有证据没有绑定 train job/result、runtime evidence 和可复核 checkpoint 字节位置；
8. 缺少不触发 provider 的正式 Stair R0 Genesis 命令入口。

## 5. 设计原则

### 5.1 单一身份链

正式链路固定为：

`VerifiedReferenceProgramArtifact`
→ Runner-owned materialized reference artifact
→ `TrainJobSpec(subject=reference)`
→ `TrainJobResult`
→ sealed eval checkpoint
→ `EvalJobSpec(subject=reference)`
→ Stair public projection
→ Stair reference qualification
→ trusted eval seal
→ private reference evidence
→ `ReferenceProgramGenesisAnchor`
→ initial current-best/current-head。

任何一层都必须从上一层的强类型对象派生，不允许从 metadata、CLI 默认值或手写 JSON 重新声明身份。

### 5.2 planned bytes 与 verified bytes 分离

- 内存中的官方源码能力不直接作为子进程路径；
- 通用 materializer 将其写成 canonical、内容绑定、无 symlink 的 Runner-owned artifact；
- train spec 和 argv 只接收该已物化能力；
- 子进程启动前和入口 preflight 都重新核对 artifact hash、manifest hash、源码 checksum 和 size；
- Stair 入口基于同一份已物化字节执行 task-owned qualification。

### 5.3 共享层不理解 Stair

共享层只新增：

- reference artifact 的通用物化/加载能力；
- reference train job 的通用强类型字段及 argv 绑定；
- reference Genesis train/eval/publish 编排；
- task 返回的不透明 reference qualification 结果合同。

共享层不得出现 `stair_010`、七桶、高度、success/progress floor、楼梯失败原因等字段。

### 5.4 Genesis 是唯一有效 session 提交点

- 训练、评估和私有证据先完成并校验；
- checkpoint 和私有证据可在 Genesis 前持久化，但它们单独不构成 session；
- 只有 `session_genesis.json` create-once 成功后，session 才有效；
- 无 Genesis 时，ledger 不得存在，live loop 不得启动，provider 不得调用；
- 发现同 session 的不完整固定产物时 fail-closed，不隐式拼接或伪恢复。

## 6. 实施步骤

### E1. 通用 Runner-owned reference artifact

修改 `training_programs.py`，新增一个不能由普通调用方直接伪造的已物化 reference artifact capability。

实现要求：

- canonical 目录只包含内部 manifest 和 `training_code/components.py`；
- artifact identity 绑定 reference manifest、verification hash、源码 checksum、size 和文件证据；
- 先写同盘临时目录，完整复核后 no-clobber 发布到内容寻址目录；
- loader 使用 strict canonical JSON，拒绝重复键、非有限数、额外文件、symlink、路径逃逸和字节漂移；
- loader 重新构造 `VerifiedReferenceProgramArtifact`，证明磁盘字节与上游内存能力相同；
- provider-authored candidate allowlist 不因该内部 artifact 改动。

验收：

- 修改源码、manifest、artifact hash、文件数量或路径后均在启动子进程前拒绝；
- 同一 reference 重复物化得到同一 identity；
- candidate artifact 路径与行为不变。

### E2. reference train job 与 backend materializer

扩展 task-neutral backend 接口，增加 reference train/eval bundle 的正式物化能力。

实现要求：

- 输入必须是 `VerifiedReferenceProgramArtifact`；
- backend 先生成 Runner-owned reference artifact，再构造 `TrainJobSpec`；
- `TrainJobSpec` 根据 subject kind 要求且只允许对应 artifact：candidate 对 candidate，reference 对 reference；
- candidate/reference 共享同一个训练配置、TrainingBinding、预算、checkpoint plan 和 eval builder；
- reference argv 只出现 reference artifact 参数，candidate 参数必须不存在；
- reference eval argv 不出现任何 source path，只保留 checksum/lineage 和 checkpoint；
- 提取现有 checkpoint-to-eval 绑定逻辑为 task-neutral helper，candidate round 与 reference Genesis 共用。

验收：

- reference spec、argv、train summary、checkpoint metadata、eval spec 的 subject/manifest/source/training/capability 完全一致；
- 重复参数、交叉 artifact 参数、metadata/argv 不一致均在 subprocess 启动前失败；
- candidate round 物化快照与既有测试保持一致。

### E3. Stair reference 训练入口

把 Stair train 入口改为 candidate/reference 两种互斥且对称的正式输入。

实现要求：

- parser 使用互斥参数组，不为 reference 复用 `--candidate`；
- `validate_stair_train_invocation()` 根据结构化 subject 选择唯一 artifact loader；
- reference loader 复核 reference ID、task ID、manifest hash、source checksum、artifact hash 和 capability profile；
- reference 通过 `qualify_stair_reference_reward_program()`；candidate 路径维持原有 verifier/qualification；
- 两类程序最终都变成同一个 `QualifiedStairRewardProgram` 并进入同一 `execute_stair_training()`；
- summary 的程序身份来自实际 qualified program，不从 argv 盲拷贝。

验收：

- reference/candidate 参数同时出现或全部缺失时拒绝；
- reference artifact 修改后 preflight 失败且 Isaac App 未启动；
- reference R0 的 runtime program subject 为 `reference`；
- candidate 现有 smoke 和正式 materialization 不退化。

### E4. Final V2 task-owned reference capability gate

在 Stair 包内新增正式 reference qualification 模块，不修改 Phase D pilot 证据实现。

实现要求：

- 输入是已经过 `project_stair_public_result()` 重算的正式 public result；
- 只从 Final V2 canonical JSON 读取 source split、seed count、low bins 和两个 floor；
- success 使用每 seed、每低阶 bin 等权宏平均后跨 seed 平均；
- progress 在固定且等量 episode 覆盖下使用等价的低阶 bin 宏平均，并显式记录等价依据；
- 生成完整 task-private evidence 和最小公共 gate；
- 公共 gate 仅含 schema/status/protocol hash/evidence hash；
- 将 `reference_capability_gate` 加入 public metrics 和 required gates；
- gate fail 时不允许创建 Genesis。

验收：

- 缺 seed、缺 bin、计数不闭合、阈值下调、错误 protocol hash、伪造 evidence hash 全部拒绝；
- Final V2 的 low-bin 和阈值不在 Python 中复制；
- shared task/base、runner、Genesis 中没有 Stair 特殊字段。

### E5. Reference Genesis 原子编排

扩展 `prepare_session_genesis()`，使 reference anchor 由正式 Runner 创建，而不是手工拼装。

实现顺序固定：

1. 解析 session evaluation binding；
2. backend 物化 reference train/eval bundle；
3. Runner 执行正式随机初始化训练；
4. 验证 TrainingBinding、预算核算、initial policy hash、runtime activation 和 checkpoint 字节；
5. 将 eval spec 绑定到该 sealed checkpoint；
6. Runner 执行正式 public/trusted eval；
7. Stair 插件重算 public projection 并运行 reference qualification；
8. shared 层构造 trusted eval seal；
9. 写入并复核 private reference evidence；
10. 构造 `ReferenceProgramGenesisAnchor`；
11. create-once 写入 `session_genesis.json`；
12. 立即从磁盘 reload，恢复 initial current-best 和 current-head。

私有证据至少绑定：

- reference manifest 和实际程序源码；
- materialized artifact identity；
- TrainJobSpec 和 TrainJobResult；
- TrainingBinding；
- budget accounting、initial policy state hash、runtime activation evidence；
- checkpoint URI、绝对本地路径和字节 hash；
- EvalJobSpec、public result、task-private qualification evidence；
- trusted eval seal。

Anchor 只保存跨重放所需的通用公开身份、TrainingBinding、checkpoint hash、公共结果、seal、qualification gate 和 private artifact ref；详细 Stair 证据留在私有 artifact。

验收：

- 任一训练/评估/gate/seal/字节校验失败时 `session_genesis.json` 不存在；
- Genesis 存在时 private artifact 和 checkpoint 字节必须可重验；
- ledger 在创建后不存在或为空；
- 第二次相同调用只 reload，不重新训练/评估；
- runtime snapshot、reference source、TrainingBinding 或协议变化时拒绝加载；
- provider 在整个 E 阶段不参与调用。

### E6. 正式命令入口与 source catalog

新增显式 Stair Phase E 命令入口，用 staging registry 构造真实 composition 并调用统一 Genesis 流程。

实现要求：

- 默认 application registry 仍只包含 Push；
- 命令从 canonical Final V2 和仓库 R0 源码解析 composition；
- 仅接受 session/output 等运行参数，不允许覆盖 2000 iterations、预算、协议、threshold 或 seed；
- 成功输出 Genesis/current-best 的公开摘要；
- 失败返回非零并不创建 ledger；
- 不构造 LLM provider；
- source catalog 从 reload 后的 Genesis 注册 R0 为 `current-head/reference_program_anchor`。

验收：

- 命令物化的 TrainJobSpec 为 reference；
- eval argv 无 reference/candidate 源码路径；
- reload 后 prompt source payload 的内容和 checksum 与官方 R0 完全一致；
- Push 的 `run_loop.py` 默认路径无行为变化。

## 7. 测试与严格验收

### 7.1 单元和故障注入

至少覆盖：

- reference artifact canonical publication、重复物化、symlink/extra-file/byte drift；
- TrainJobSpec artifact kind 互斥；
- spec-to-argv 完整绑定及重复/错误参数；
- Stair candidate/reference train preflight；
- 实际执行源码与 summary/checkpoint lineage 一致；
- Final V2 reference gate pass/fail、覆盖缺失和阈值不可漂移；
- reference Genesis 创建、reload、current-best、source catalog；
- train/eval/gate/private artifact/Genesis 写入各阶段失败注入；
- Genesis/private evidence/checkpoint 任一字节修改后 fail-closed；
- provider 调用计数保持 0；
- candidate ledger 初始为空且首个 candidate 仍从 round 1 开始。

### 7.2 旧任务回归

- 全量 Push 单元/集成测试；
- Push candidate seal→verify→materialize→train/eval dry-run；
- Push bootstrap/baseline Genesis reload；
- 默认 registry 仍不包含 Stair；
- candidate artifact、argv 和 SubprocessRunner 安全边界不退化。

### 7.3 静态质量门

- `unittest discover` 全量通过且无误收集/警告；
- 相关 pytest 入口通过；
- `ruff check`、`compileall`、`git diff --check` 通过；
- 对新增和修改文件做 task-semantics leakage 扫描；
- 检查无无效 compatibility helper、无第二套预算/协议常量、无 metadata 身份旁路。

### 7.4 服务器动态门

在服务器新鲜 worktree 上依次执行：

1. reference artifact/materialization preflight，不启动 Isaac；
2. 小型动态 smoke，证明 reference 进入同一训练 runtime，candidate 回归通过；
3. 正式 R0：固定 `2000 iterations`、正式私有 training seed、正式七桶 public/trusted eval；
4. 验证 capability gate、seal、checkpoint 和 Genesis；
5. 重新启动独立进程，仅从磁盘 reload Genesis 并恢复相同 current-best/source head；
6. 确认无 provider 调用、ledger 为空、GPU/子进程正常退出。

正式 R0 不得因耗时而改用 Phase D pilot checkpoint或缩减预算。若正式 R0 未通过已冻结 capability gate，Phase E 判定失败并停止，不进入阶段 F。

## 8. 阶段完成定义

只有同时满足以下条件，Phase E 才可标记完成：

- 本计划 E1-E6 全部实现；
- 本地和服务器全部验收通过；
- 正式 2000-iteration R0 train/eval 已完成且通过 Final V2 reference gate；
- create-once Genesis 已生成并可独立 reload/replay；
- R0 源码成为 prompt current-head，ledger 仍为空；
- Push 全量回归无退化；
- 代码、计划、证据提交并推送，服务器正式仓库与远端 HEAD 一致。

若仅完成代码和 smoke、尚未跑完正式 2000-iteration R0，则只能报告“Phase E 基础设施完成，正式证据未完成”，不得宣称 Phase E 完成。

## 9. 正式运行硬阻断修正：执行设备单一合同

### 9.1 触发证据

- 首次正式 R0 已按 Final V2 完成 `4096 x 24 x 2000 = 196,608,000` env-steps，训练和 sealed checkpoint 均通过；
- 随后的 15-seed 正式评估在 Runner 固定的 `7200s` 内未完成，Runner 正确终止子进程且未创建 Genesis/ledger；
- 同一 checkpoint、同一正式协议、同一 56-env 评估，使用全卡可见并显式 `--device cuda:1` 时约 46 秒完成两个 seed，与 Phase D 历史速度一致；
- 因此硬错误不是 2000-iteration 预算、R0 奖励、七桶协议或评估统计，而是 production job 没有绑定执行设备，只能依赖进程外 `CUDA_VISIBLE_DEVICES` 重映射和 AppLauncher 默认值，形成第二套设备真值。

### 9.2 源头修复边界

新增 task-neutral 的 canonical execution-device 合同，并固定唯一数据流：

`task-owned composition default/CLI`
-> `IsaacLabRslRlBackendConfig`
-> `TrainJobSpec/EvalJobSpec/BaselineEvalJobSpec`
-> 唯一 `--device`
-> 入口实际环境/agent device
-> summary
-> `SubprocessRunner` 复核
-> `TrainJobResult` / checkpoint lineage / public eval result
-> trusted eval seal / Genesis 私有证据。

约束：

- 仅接受 canonical `cpu` 或 `cuda:<非负整数>`，拒绝布尔值、空值、隐式 `cuda`、前导零和任意字符串；
- train/eval argv 中 `--device` 必须各恰好出现一次并与 spec 完全一致；
- 三类 job spec 和 backend config 均不得保留隐藏设备默认值；设备只能由已验证 composition snapshot 提供；
- composition runtime snapshot 必须包含同一 execution device，使 R0 与后续候选在同一 session 内不能漂移；
- Stair 和 Push provider 均从各自已验证 composition config 构造 backend，不在脚本、metadata 或环境变量中另设默认；
- 正式 Stair Genesis 入口复用统一 task runtime CLI；`--execution-device cuda:1` 必须进入同一 composition snapshot，而不是入口专属 metadata；
- 入口 summary 必须回报实际 device，Runner 在接受训练 checkpoint 或评估结果前复核；
- checkpoint lineage、候选/参考 eval binding、public metrics 与 Genesis train/eval 证据必须继续绑定同一 device；
- trusted eval seal 升级为 v4，设备同时进入 `eval_config_hash` 和 seal hash，不能在 seal 后替换；
- `TrainingBinding` 继续只描述可比较的算法/网络/优化器配置，不混入硬件位置；设备公平性由 session snapshot 和上述执行证据链约束；
- 正式服务器运行使用全卡可见、显式 `cuda:1`，不再使用 `CUDA_VISIBLE_DEVICES=1` + `cuda:0` 重映射；
- 不修改 Final V2 JSON、2000 iterations、训练种子、七桶/seed 数、奖励源码、capability gate、acceptance 参数或 Phase D pilot 实现；
- 不因本次故障扩大 `runner_timeout_s`。

### 9.3 严格验收

- 缺失、重复或与 spec 不一致的 `--device` 在 subprocess 启动前失败；
- summary 声明的实际 device 与 spec 不一致时，checkpoint/result 不被 Runner 接受；
- checkpoint lineage、Genesis 证据或 trusted seal 中的 device 漂移均 fail-closed；
- Stair materialize/preflight 明确产生 `cuda:1` train/eval 作业；
- Push 默认 `cuda:0` 物化、dry-run 和现有测试行为保持成立；
- 本地全量 unittest/pytest、ruff、compileall、diff-check 通过；
- 服务器定向测试通过，显式 `cuda:1` 正式评估速度恢复到有界范围；
- 由于首次正式 session 未提交 Genesis，修复后使用 fresh output/session 完整重跑 2000-iteration R0 train/eval，不把诊断 checkpoint 伪装为正式新 session 证据；
- fresh Genesis 创建成功后，在独立进程中只 reload，不重新训练/评估，且 provider 调用为 0、ledger 不存在或为空、GPU/子进程全部回收。

### 9.4 正式多 split 评估的进程清理边界

设备合同修复后，15-seed 评估暴露了第二个独立硬阻断：同一个 Isaac
runtime 顺序完成 `public_dev/public_accept/sealed_validation` 后，长生命周期环境在
`env.close()` 内无法有界返回。按 split 重建环境也不可行，因为同一个
`SimulationApp` 内第二次创建 `SimulationContext` 会挂起。该问题不属于 Phase D
奖励、Final V2、2000 iterations 或七桶协议，不能通过缩减 seed、扩大 timeout 或
复用 pilot checkpoint 规避。

最终边界固定为：

- 正式多 split Stair eval 仍只创建一个环境并顺序运行全部 `3+5+7` seed；
- task-owned `StairEvaluationRuntimeLease` 持有该环境直到证据完成投影；
- 多 split 路径不在 Python 内调用 `env.close()`，由 Runner 启动的隔离子进程退出
  统一回收 Isaac/CUDA 资源；
- `StairEvaluationExecutionResult` 内部递归只读，跨 subprocess 的唯一出口是脱离内部
  对象、严格有限且 JSON-compatible 的 `public_summaries()`；
- AppLauncher 创建后，生产入口在成功、rollout 失败和 summary 发布失败三条路径上
  都从同一个 `finally` 执行 `os._exit(status_code)`；summary 必须先完成写入和标准流
  flush；
- Phase D 的单 split pilot 不持有 runtime lease，继续显式 `env.close()`；Push 入口和
  shared Runner 不引入 Stair 生命周期语义。

故障注入和真实动态证据：

- `c27e443` 上全部 15 个 seed 已完成，但内部 `MappingProxyType` 被直接放入 summary，
  `json.dumps()` 失败；该失败同时证明原发布异常路径没有到达硬退出边界；
- `a0bf441` 将内部只读结果与公开 JSON 投影分离，并把 post-launch 退出收敛到统一
  `finally`；
- 服务器使用全卡可见和显式 `cuda:1`，对旧失败 session 的 checkpoint 仅做诊断评估：
  `public_dev=3`、`public_accept=5`、`sealed_validation=7` 全部完成，进程退出码为 0；
- public accept mean 为 `0.5464285714`，sealed validation mean 为
  `0.5586734694`；checkpoint safety、holdout、协议 hash、evaluation binding 和隔离门
  全部为 pass；
- summary 在进程退出前完整落盘，父子进程均消失，GPU1 回到仅保留既有外部进程的
  498 MiB 占用；GPU0 的既有进程未受影响；
- 该 checkpoint 仍只用于验证修复后的执行链，不能进入新 Genesis。正式完成 Phase E
  仍要求在当前已提交源码上创建 fresh session，从随机初始化完整训练 2000 iterations，
  再由 Runner 完成同一 15-seed eval、qualification、seal 和原子 Genesis。
