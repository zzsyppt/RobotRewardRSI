# Stair Phase C 详细实施计划

> 基座：`209c03d Harden stair Phase B reference verification`
>
> 对应总计划：`revised_plan.md` 的“阶段 C：pre-reset 完整奖励运行时、ABI 与 R0 等价性”。
>
> 本阶段只建立完整奖励的可信执行面和官方 R0 等价性，不运行收敛 pilot，不创建 Genesis，不开放课程、观测、动作、终止、命令或 terrain 修改权限。

## 1. 阶段目标

Phase C 完成后，Stair 训练环境必须满足以下事实：

1. Isaac Lab 的 `RewardManager` 只有一个有效的 Stair complete-reward term；
2. 该 term 在 termination 计算之后、auto-reset 之前，从当前 transition 构造一次 task-owned 输入并执行一次当前完整奖励程序；
3. PPO 返回 reward、Isaac episode accumulator、RSL-RL 训练日志和 runner runtime evidence 都引用这一次计算，不存在 post-step 覆盖或第二次奖励调用；
4. 候选只接收 detached clone，不能通过输入、全局状态或 RNG 污染训练；
5. canonical R0 在同一真实 transition 上与 Isaac Lab 2.3.0 官方 G1 rough reward 数值等价；
6. Push Recovery 继续使用原 additive wrapper，行为和合同不变。

## 2. 已确认的真实运行时事实

服务器当前 Isaac Lab 2.3.0 的 `ManagerBasedRLEnv.step()` 顺序为：

```text
physics
  -> episode/common counters
  -> termination_manager.compute()
  -> reward_manager.compute(dt)
  -> recorder.record_post_step()
  -> reset done envs
  -> observation
```

`RewardManager.compute()` 对每个 term 执行：

```text
value = term(env, ...) * term.weight * dt
reward_buf += value
episode_sum[term] += value
```

因此正确接入点是 Stair env cfg 的唯一 `RewardTermCfg`，不是 `env.step()` 返回后的 wrapper。只要 dispatcher 返回 reward rate，Isaac 自身会把同一个乘 `dt` 后的 tensor交给 PPO并累积到 episode log。

官方 G1 rough R0 的非零项包括：termination、线速度/角速度跟踪、足端 air-time/slide、姿态、action rate、关节力矩/加速度/限位，以及 hip/arm/finger/torso 关节偏移。它们依赖 termination mask、root/body/world 状态、命令、动作历史、关节分组和足端接触历史。

参考 Codex 的 capability 构造方式，本阶段遵循同一原则：字符串或 manifest 不能自行声明能力；重复、缺失或相互矛盾的状态在严格构造入口直接拒绝，只有完整验证流程能产生可执行 capability。

## 3. 边界设计

### 3.1 共享层允许的改动

共享层只增加 task-neutral 的候选源码纯函数安全规则：

- 禁止 attribute/subscript 写入和删除；
- 禁止 `AugAssign`、`global`、`nonlocal`、可变默认参数；
- 禁止 tensor 原地方法和显式 `__setitem__`/`__setattr__`；
- 禁止 `getattr/setattr/delattr`、`__builtins__` 等动态反射旁路和 console/debug 副作用；
- 禁止 Python、NumPy、Torch 及 tensor RNG 调用；
- `RewardTerm.fn`/`DiagnosticTerm.fn` 必须是 canonical source 中定义的单参数、无闭包 Python 函数，不能携带 builtin、foreign callable 或隐藏可变状态；
- 保留现有文件、网络、进程、动态 import/exec 禁区；
- verifier 输出绑定一个明确的 policy version/hash，避免“旧规则验证”和“新规则验证”同名。

这些约束对任何候选代码都成立，不包含 Stair 字段、楼梯几何或评价语义。Push 现有纯函数候选应继续通过；若旧样例依赖副作用，应修正样例而不是为旧漏洞保留旁路。

### 3.2 必须留在 Stair task-owned 包的内容

以下实现全部位于 `tasks/stair_climbing/` 或 Stair reference source 目录：

- `StairRewardInputsV2` schema 与 prompt projection；
- Isaac scene tensor 到 detached-clone 输入的 builder；
- joint/body 名称解析和 runtime schema guard；
- complete-reward dispatcher、per-term guard 和 runtime evidence；
- RNG/input/source alias guard；
- RewardManager 配置替换；
- canonical R0、官方逐 term oracle 和等价性报告；
- Stair train entrypoint 的 RSL-RL 执行与 task-owned summary；
- Stair 性能门限和 benchmark artifact。

共享 runner、evaluator 和 Push runtime 不解释任何 Stair 字段。

### 3.3 复用 Phase B verified reference 基座

canonical R0 源码写入真实 `training_code/components.py` 布局后，继续使用 Phase B 的 `resolve_reference_program_artifact()` 从实际 UTF-8 bytes 原子派生 URI、checksum、size、manifest hash 和 verification hash。

`VerifiedReferenceProgramArtifact` 只证明 bytes/identity/provenance 绑定。本阶段另由 Stair-owned qualification 证明：

- 静态纯度策略通过；
- 完整奖励 ABI 合法；
- 官方来源映射完整；
- 数值等价通过。

二者不合并，也不把 `authorship_origin` 当作数值等价证明。

## 4. StairRewardInputsV2

### 4.1 单一来源

在 draft V2 canonical JSON 中新增 `complete_reward_runtime`，冻结：

- ABI schema/version；
- 字段名、dtype 类别和 trailing shape；
- output shape/device/dtype/finite/per-step bound；
- dispatcher、purity guard 和 evidence schema version；
- R0 等价 `atol/rtol`；
- deterministic activation preflight 次数；
- 1024-env 性能门限；
- canonical R0 source identity/provenance revision。

Python loader 只校验结构、关系和可信 protocol hash anchor，不再复制整份字段实例。runtime、prompt 和测试都从已验证 JSON 派生。

### 4.2 输入字段原则

输入同时覆盖“官方 R0 可精确复现”和“后续 Stair reward 有足够信息”两类需求：

- 官方运动/控制：command、yaw-frame/base 速度、world/body 角速度、projected gravity、action/previous action；
- 官方关节项：ankle limits、leg torque/acc、hip/arm/finger/torso position error；
- 官方足端项：air/contact time、contact force history、foot velocity；
- Stair 状态：base/feet 相对 terrain origin、height scan、forward progress、remaining goal distance、episode progress；
- 终止与接触：current non-timeout termination、base contact force；
- 必要的完整姿态/关节公共状态，供后续完整重写使用。

不暴露 env、manager、sensor object、terrain generator、seed、bin identity、acceptance 或 checkpoint。

### 4.3 clone 与 alias 合同

builder 每个 control step：

1. 只读当前 pre-reset scene/action/termination buffers；
2. 先计算 task-owned derived tensors；
3. 对每个输出执行 `detach().clone()`；
4. 校验 shape、device 和 dtype；
5. 校验输出 storage 不与任何环境 source tensor alias；
6. 生成只包含 metadata、storage identity 和 tensor version 的轻量 input fingerprint。

不在每步将大 tensor 拷回 CPU 做 SHA-256。数值全量比较用于 activation/fault-injection 测试；生产每步依靠 clone、storage non-alias、tensor version 和静态禁区组合守卫，避免把完整 GPU 同步引入热路径。

## 5. CompleteRewardDispatcher

### 5.1 唯一 RewardManager term

`build_stair_train_env_cfg()` 必须显式接收一个已验证的 Stair reward program execution spec，并把原 `G1Rewards` 替换为只有 `complete_reward` 的配置对象。不能只把旧 term 权重设零，也不能保留旧 term 给日志使用。

eval env 不加载任何 reward source。评估 reward 不参与评分，仍由 task-owned recorder/evaluator 产生结果。

### 5.2 每步执行

dispatcher 被 Isaac `RewardManager.compute()` 调用一次后：

1. 构造一份 `StairRewardInputsV2`；
2. 保存 input/RNG guard；
3. 顺序调用源码中声明的 `RewardTerm.fn`，对每项校验 `(num_envs,)`、floating dtype、同 device、finite 和 clip；
4. 按源码中的 weight 汇总为完整 reward rate；
5. 校验总值 per-step bound；
6. 核对输入 tensor version、Python/NumPy/Torch CPU/CUDA RNG state 未变；
7. 保存同一次调用的 per-term 与 total runtime evidence；
8. 返回该 tensor，之后由 Isaac 统一乘 `dt`、累计和记录。

禁止为日志、evidence 或 recorder 再调用 reward program。

### 5.3 确定性与“每步一次”的协调

源码结构、纯度和 exact-bytes 身份在 AppLauncher 前完成。env 创建并 reset 后、RSL-RL runner/PPO 创建前，先通过正常 termination capture 和 dispatcher 执行一个零 action 的真实 preflight transition，并保留该 transition 的 detached input。随后只在 dispatcher 外对这份冻结输入重复执行纯函数确定性检查，结果全部丢弃；检查完成后 reset 全部环境，再交给 PPO。runtime evidence 将这一次 preflight dispatcher call 与正式 PPO control steps 分开记录。

这样既不在 reset 后伪造“当前 transition”，也保持每个真实 env step 只有一次权威 dispatcher 调用；确定性检查的重复结果不进入 RewardManager、episode accumulator 或训练。

### 5.4 evidence 与日志同源

runtime evidence 记录：

- dispatcher call count / environment common step count；
- input schema hash、program source checksum、subject manifest hash；
- per-term call count、finite/shape、last mean、max absolute contribution；
- total reward rate mean/max 与乘 `dt` 后 reward mean；
- RNG/input/source-alias guard 状态；
- 共享 mechanism gate 使用的 `reward_term_calls`、nonzero flag、component/capability activation；
- duplicate-call/legacy-base-term/recorder warnings；
- performance counters。

训练完成后以 Isaac `reward_manager._episode_sums["complete_reward"]` 和 RSL-RL episode log 的一致性探针验收。evidence 只读取已产生的 reward，不重算。

## 6. Canonical R0 与数值等价

### 6.1 源码布局

canonical source 固定为：

```text
src/true_rl_agent/tasks/stair_climbing/reference_programs/
  official_g1_rough_v1/
    training_code/components.py
```

entrypoint 仍为 `training_code.components:build_training_components`。源码只使用公开 component API 和 `StairRewardInputsV2` 字段，不导入 Isaac env/manager。

R0 将官方 G1 rough 的每个非零 term 转写为 `RewardTerm`，保留官方名称、weight 和参数。官方零权重项在等价报告中显式记录为 zero/skipped，不制造第二套有效 reward。

### 6.2 等价 oracle

等价测试不拿两套独立手写公式相互比较。权威 oracle 是真实 Isaac env 原 `G1Rewards`：

1. 在同一真实 environment transition、reset 前读取官方 RewardManager 各 term raw output/weighted rate；
2. 从相同 scene buffers 和 termination term callable 的 current-step capture 构造 detached `StairRewardInputsV2`；
3. 执行 R0 dispatcher；
4. 逐 term 对齐官方名称，比较 raw、weighted rate 和 total；
5. 覆盖正常 step、timeout、base-contact 以及 timeout+contact 并发；
6. 强制覆盖 `base_contact -> auto-reset -> 下一回合正常 step`，证明公开 ABI 不读取历史 `_term_dones`；
7. 报告 max abs/rel error、首个 mismatch 和样本数。

必须在冻结容差内全部通过。若某个官方实现无法通过公开 ABI 精确表达，应扩充 task-owned ABI并更新 draft hash，不允许在 oracle 或 R0 中加入补偿常数掩盖误差。

## 7. Stair 训练入口

Phase B 的 fail-closed entrypoint 原地升级为真实 Stair trainer：

1. 在 AppLauncher 前完成 argv/config/budget/identity/artifact bytes preflight；
2. 加载 exact source bytes并执行 task-owned complete-program structural/purity preflight；
3. 启动 Isaac App；
4. 构造 task-owned train env cfg，安装唯一 dispatcher，确认 recorder 为 `None`，再用真实 pre-reset 输入完成 determinism preflight；
5. 构造官方 `G1RoughPPORunnerCfg`，应用并回显 Phase A canonical agent configuration hash；
6. 以 fixed random seed 创建 RSL-RL runner并训练；
7. 原子复制 checkpoint并校验 bytes hash；
8. 在关闭 SimulationApp 前写入并 flush summary；
9. summary 必须包含 budget、program identity、runtime evidence、RewardManager schema 和 checkpoint lineage。

本阶段只要求短 smoke 真正执行；49M+ convergence 留给 Phase D。

## 8. 实施步骤

### C1. 冻结合同与纯函数安全

- 更新 draft V2 JSON/hash anchor；
- 新增 task-owned reward input schema/projection；
- 抽取/强化 task-neutral AST purity audit；
- 让 verifier evidence 绑定 policy version/hash；
- 补 fault injection：`zero_()`、subscript write、attribute write、global/nonlocal、mutable default、Python/NumPy/Torch/tensor RNG。

验收：非法源码在 seal/verify 后、backend materialization 前失败；现有 Push 纯函数候选继续通过。

### C2. 输入 builder 与 dispatcher

- 实现 runtime body/joint schema resolver；
- 实现 detached-clone builder和 non-alias/fingerprint guard；
- 实现 per-term/output/RNG guard；
- 实现唯一 RewardManager cfg replacement；
- 实现 runtime evidence snapshot。

验收：fake env 行为测试和真实 Isaac 1–16 env fault injection均通过；done step 使用 reset 前状态。

### C3. R0 与等价性

- 写 canonical R0；
- 用 Phase B verified reference resolver绑定真实 bytes/provenance；
- 实现真实官方 RewardManager oracle；
- 输出机器可读 equivalence report。

验收：逐 term/total 在冻结容差内通过，term 名集合闭合，R0 source 满足 inline 24 KiB 上限。

### C4. 训练链与日志同源

- 激活 Stair train entrypoint；
- 运行 1/16/1024 env 短 smoke；
- 验证 PPO reward、episode sum、RSL log、runtime evidence 同源；
- 注入 sentinel legacy base reward并证明所有出口不受影响；
- 验证每 step dispatcher call count 恰为一。

### C5. 性能与完整回归

- 在同一服务器/GPU、相同 1024 env/steps 下比较：官方 reward、Stair R0 dispatcher、Stair dispatcher+eval recorder；
- 报告 env-steps/s、clone/dispatcher 时间和相对开销；
- train benchmark 明确 recorder `None`；
- 运行全量 unittest、Ruff E9/F/I、compileall、diff-check；
- 重新审计 identity、protocol hash、prompt exposure、reward/log/evidence、Push 回归和无用 Phase B fail-closed 代码。

## 9. 严格验收矩阵

| 要求 | 权威证据 |
|---|---|
| pre-reset | 真实 Isaac done transition 行为测试；reward/termination/recorder common-step stamp 一致 |
| 唯一 reward | RewardManager active term 集合严格为 `complete_reward`；sentinel legacy term 无法进入 cfg |
| 每步一次 | dispatcher calls == env common steps；per-term calls == dispatcher calls × declared terms |
| 同源 | returned reward、`reward_buf`、episode accumulator、RSL log、evidence 的同一 transition 数值一致 |
| 输入隔离 | 每字段独立 storage；故障注入后 env/action source bytes不变；input version guard pass |
| RNG 隔离 | Python/NumPy/Torch CPU/CUDA state 调用前后相同；随机源码被静态或 activation guard拒绝 |
| 确定性 | activation fixture 两次输出逐 bit相同；正式 step只执行一次 |
| 输出合同 | shape/device/dtype/finite/bound 全部 fail closed |
| R0 等价 | 真实 Isaac逐 term/weighted/total report 全通过，含 terminal并发 transition |
| 性能 | 1024-env benchmark达到 draft V2 门限；train recorder 为 `None` |
| 身份 | R0 verified artifact 的 bytes/hash/size/provenance与 equivalence report/source checksum一致 |
| 旧任务 | Push 全量单元/集成测试通过，additive wrapper source与行为未改 |
| 无二源 | ABI、容差、bound、性能门只来自 canonical draft V2；runtime/prompt/evidence读取同一对象 |

## 10. 明确不做

- 不运行 Phase D convergence pilot；
- 不创建或提交 ReferenceProgramGenesisAnchor；
- 不把 R0 checkpoint 当 current-best；
- 不开放 curriculum/observation/action/termination/command/terrain authoring；
- 不让 candidate diagnostics替代 task recorder；
- 不把 Stair 字段放入共享 `RewardInputs`；
- 不把 Push post-step wrapper迁移为 complete replacement；
- 不以 mock-only、字符串顺序测试或 source grep替代真实行为验收。

## 11. 完成判定

只有验收矩阵全部有当前代码和当前服务器运行证据、审计未发现仍需本阶段解决的 P0/P1/P2 边界问题，并且本地/origin/server HEAD一致时，Phase C 才可标记完成并进入 Phase D。

## 12. 实施与验收结果（2026-07-29）

Phase C 的功能实现与当前服务器动态验收已按上述边界完成。实际实现保留 Phase B 的 reference/artifact/composition 基座，并在其上增加 task-owned complete-reward 执行能力；未回滚 Phase B 修复，也未提前实现 Phase D–F。当前 Phase C 变更尚未进入 Git HEAD，因此在本地/origin/server 形成同一提交前，不满足第 11 节的正式交付条件，也不得进入 Phase D。

### 12.1 动态证据

- `phase_c_reward_equivalence_report.json`：真实 Isaac 官方 RewardManager oracle 覆盖 normal、timeout、base-contact、timeout+base-contact，以及 base-contact auto-reset 后下一回合的当前步语义；16 env、131 个 transition、2096 个 env-transition 样本，逐 term/weighted/total mismatch 为 0，16 个 post-reset 环境均无历史终止原因泄漏；
- `phase_c_reward_runtime_report.json`：4 env × 8 control steps，RewardManager 只有 `complete_reward`；一次真实 transition preflight 加 8 个测量 step，共 dispatcher 9 次、15 项 term 135 次，训练 recorder 关闭；共享 mechanism evidence 同时闭合；
- `phase_c_reward_fault_report.json`：CUDA 上 finite 越界和 non-finite 两类故障均 fail closed，故障后 CUDA context 继续可用；
- `phase_c_reward_performance_report.json`：1024 env 串行同 GPU 测量，dispatcher/官方吞吐比 1.14721179，通过 0.85 下限；recorder 对 dispatcher 的比值单独作为诊断值记录；
- `phase_c_rsl_rl_smoke_report.json`：一次真实 transition preflight 后执行 16 env × 8 PPO steps、1 次真实 PPO update；PPO reward、RewardManager episode sum、RSL log、TensorBoard scalar 和 checkpoint 证据闭合。

五份报告均为 canonical JSON hash 自校验通过、同一 draft V2 hash、同一当前 R0 qualification 身份；报告中的派生 hash 不在本文复制，防止形成第二套手工真值。

### 12.2 审计中完成的源头修复

- 共享 verifier 的可执行 policy facts/hash 覆盖 AST、反射、RNG、写入、callable 来源/签名/闭包规则，prompt projection 与 verifier 使用同一来源；
- Stair runtime evidence 同时满足 task-owned gate 和既有 task-neutral `RuntimeActivationEvidence`，避免 Phase F 被 mechanism gate 静默判为未激活；
- `QualifiedStairRewardProgram` 只持有 immutable term tuple，不暴露可变 `TrainingComponents`；
- 所有新旧生产 Isaac entrypoint 使用 task-neutral lifecycle helper，失败退出码不会被 `SimulationApp.close()` 覆盖；历史 Phase 0 probe 保持冻结，不重写其已取证源码；
- 性能编译器将 recorder 比值明确绑定 dispatcher 分母，消除字段名称与计算语义二源；
- 共享 runtime evidence 拒绝字符串、浮点和布尔值之间的隐式强制转换。
- Stair 训练、Oracle、评估统一在 termination term callable 返回处捕获当前 transition mask，并由 term 的 `time_out` 配置推导 timeout/non-timeout 聚合；奖励 ABI 和 Oracle 不再读取会跨 reset 保留历史原因的 `TerminationManager.get_term()`；
- determinism preflight 不再在 reset 后构造伪 current-transition 输入，而是在 PPO 前执行一次真实、受 dispatcher guard 保护的 transition，并复用该次冻结输入完成重复性检查。

### 12.3 回归结果

- 本地：670 tests 全部通过，9 项因本地环境条件跳过；
- 服务器 Isaac/Torch 环境：670 tests 全部通过，仅 2 项环境条件跳过；
- `ruff --select E9,F,I`、`compileall`、`git diff --check` 全部通过；
- shared source 任务语义扫描未发现 Stair 字段、楼梯失败分类或几何常量泄漏；共享层只保留 task-neutral complete-reward capability 和纯函数/证据合同；
- Push reward additive wrapper 未改，Push 相关测试包含在两端全量回归中。

服务器的 headless Isaac 运行仍打印缺少可选 Iray `libGLU.so.1` 和 display 的上游告警，但所有目标作业、CUDA probe 和退出码验收均成功；该告警不进入任务合同，也不影响本阶段结果。
