# Stair Phase G 详细实施计划：全量回归与生产启用

## 0. 当前状态（2026-08-04 最终验收）

- 生产 registry 惰性接入、staging 二源删除和 Push 默认路径隔离均已实现，可以保留。
- 旧 Phase F v2 ledger 的训练、评估、checkpoint 与 decision 结果保持有效且不可改写，但其 authoring provenance 仍明确标为 legacy unverified，不用于 provider continuation。
- 当前可写 ledger v5 显式区分 `provider_authoring` 与 `non_provider`，强制绑定 ledger epoch、prompt 和 evidence；未知版本、epoch 漂移和模式漂移 fail closed。历史 v4 + 完整 `authoring_evidence.v2` 只允许验证与 replay，新记录只允许 v5 + `authoring_evidence.v3`。旧 v2 不迁移、不回填。
- 正式 v5 epoch 已完成精确 prompt、逐 attempt authoring evidence、candidate、checkpoint、public/sealed eval、decision 和 replay 闭环。
- 修正 prompt 证据边界后的独立 15-round 正式实验已完成：15 个候选全部跑满固定 2000 iterations 并完成 public/sealed eval，R9 与 R14 被接受，最终 current-best 为 `stair_stable_progress_contact_v3`，public `0.621429`、sealed `0.602041`。
- Phase G 的生产接入、根因修复、全量回归、服务器 replay、逐字节证据审计和正式效果实验均已完成。0.40 m 与稳定 0.35 m 能力仍是后续优化目标，不通过修改 Final V2 或预算掩盖。

## 1. 阶段定位

Phase G 主要负责把已经完成真实两轮验收的 Stair 任务接入生产组合根，并完成最终回归、故障注入和发布审计。已验证的奖励运行时、Final V2 协议、训练算法、评估统计和 2000 iterations 固定预算是默认基座，不因接入便利或候选效果差而重写；但这不等于冻结 Phase A-F。若真实运行或故障注入证明既有实现存在执行错误、证据失真、反馈不可行动、信任链绕过或边界不合理，必须回到对应权威源修复，并重跑全部受影响验收，不能在 Phase G 表面兼容。

本阶段最终实现基于 `f19ef37`：既保留此前 Phase D-F 的固定协议与训练结果，也补齐 v5 authoring provenance、ledger epoch 和生产 registry 边界；服务器使用同一提交完成正式证据 replay、Push 默认路径和显式 Stair 路径验证。

## 2. 不变量与边界

- 默认任务继续是 `unitree_push_recovery`；无 `--task-id` 的所有旧 CLI 行为不得改变。
- Stair 只能由显式 `--task-id unitree_stair_climbing` 或 Stair 薄入口选择。
- 生产注册只装配 task-owned provider，不把楼梯高度、失败原因、RewardInputs、seed/bin 或 reward 语义放入共享层。
- Stair provider 必须继续从 canonical Final V2、官方 R0 artifact 和唯一 `StairClimbingCompositionProvider` 构造。
- 官方 R0 的解析采用惰性装配：构建默认 registry 或运行 Push 不读取、训练或校验 Stair R0；只有显式选择 Stair 才解析并验证。
- live loop 在 provider 调用前必须先获得完整、已验证的 reference Genesis。创建新 Genesis 时，必须先完成 R0 的固定 2000 iterations 训练、正式评估和资格门；已有会话则只允许 load-only 验证。
- `scripts/run_stair_live.py` 继续要求 existing Genesis；每个请求轮必须形成可 replay 的闭合 ledger 记录，每个成功训练的候选必须完成 public eval 和 sealed validation。provider/static/candidate-runtime 的合法拒绝允许进入下一轮，不能把这种预期控制流误判为整个 live session 的基础设施失败。
- 不开放 curriculum、训练分布、PPO、网络、seed、预算或评估协议修改权限。
- 15-round 正式实验不作为放宽 Phase G 实施门的手段；它只在 Phase G 实现和回归全部通过后启动，并已在独立 epoch 中完成。

## 3. 源头设计

### 3.1 唯一生产注册源

`build_application_composition_registry()` 成为 Push 与 Stair 的唯一生产注册源：

- Push provider 保持当前直接注册和默认任务身份；
- Stair 以 task-neutral 的 lazy provider 壳注册；
- lazy 壳只保存 `task_id + provider_factory`，第一次调用 `cli_defaults()`、`resolve()` 或 `run_audit()` 时创建并缓存真实 provider；
- factory 位于应用 wiring 层，内部调用 Stair task-owned 的 config、reference resolver 和 composition provider；
- lazy 壳必须验证 factory 返回 `TaskCompositionProvider` 且 task ID 完全一致。

这样既显式启用 Stair，又不会让 Stair artifact 的文件错误或协议错误阻断默认 Push 的 CLI 启动。lazy 壳不解释任何任务语义。

### 3.2 删除开发期二源

- `scripts/run_stair_live.py` 和 `scripts/materialize_stair_job.py` 改为调用默认生产 registry，不再自行解析 config/reference 或注入 staging registry。
- `scripts/create_stair_reference_genesis.py` 同样从默认生产 registry 解析一次 composition；正式 R0 artifact 只取自 resolved composition，不再由脚本独立解析第二份 official reference。
- 删除无生产必要的 `tasks/stair_climbing/staging.py`。
- Phase B 的 provider 注入测试直接构造 `TaskCompositionRegistry + StairClimbingCompositionProvider`，不保留一套 staging 注册 helper。
- 文档和测试中“生产 registry 排除 Stair”的历史断言改成：Phase 0 activation artifact 保持历史不可变，但当前生产 registry 已显式注册 Stair，默认任务仍为 Push。

## 4. 分步实施

### G0：冻结 Phase F 入口证据

**工作**

- 验证真实 ledger 至少包含两条完成 train、public eval 和 sealed validation 的候选记录；此前或其间允许存在合同闭合的 provider/static/candidate-runtime 拒绝记录；
- replay 恢复 current-best/current-head，并核对第二轮 parent source；
- 核对每轮 2000 iterations、initial policy hash、TrainingBinding、checkpoint、public result、seal 和 decision；
- 确认无残留训练/评估进程，GPU1 回到运行前基线。

**严格验收**

- 任一成功训练的候选缺 eval/sealed evidence、出现 infrastructure error、hash 不闭合或 replay 不一致，Phase G 不得开始；
- 合法拒绝记录必须有对应的 bounded failure evidence，不能伪造 train/eval 成功计数；Phase F 只有在累计至少两次完整评估后才通过实验门；
- 不允许通过手工修改 Genesis、ledger、checkpoint 或 summary 继续。

### G1：生产 registry 惰性接入

**工作**

- 在 composition wiring 中实现最小 task-neutral lazy provider；
- 在默认 registry 注册 Push 和 Stair，默认 ID 保持 Push；
- Stair factory 只调用 task-owned canonical resolver。

**严格验收**

- `registry.task_ids()` 精确为 Push 与 Stair，`default_task_id` 仍为 Push；
- 构建 registry、选择默认 provider、解析 Push 均不得调用 Stair reference resolver；
- 显式选择 Stair 时 resolver 恰好调用一次，返回 Final V2 默认值，`train_max_iterations == 2000`；
- factory 返回错误类型、错误 task ID 或损坏 R0 时 fail closed，且同一故障不影响 Push 路径。

### G2：入口收敛与 staging 删除

**工作**

- Stair live/materialize/Genesis 薄入口改用生产 registry；
- 删除 staging helper 及所有生产调用；
- 更新历史测试和扩展指南中的当前激活描述。

**严格验收**

- `rg` 证明不存在第二个 Stair registry builder；
- generic `run_loop.py` 无参数仍选择 Push；显式 Stair 才使用 Stair defaults；
- Stair live 入口仍拒绝调用方覆盖 task/mode，仍要求 existing Genesis、全部请求轮闭合以及每个成功训练候选完成完整评估；
- Stair materializer 仍拒绝覆盖 task/job kind，并只能物化 candidate 作业；
- Stair Genesis preflight/create/load-only 只使用 resolved composition 内的 reference artifact，脚本中不存在第二次 config/reference resolver；
- 删除 helper 后无失效 import、兼容分支或未调用代码。

### G3：身份、hash 与启动边界故障注入

**工作**

- 覆盖 source、manifest、TrainingBinding、initial hash、checkpoint、evaluation binding、seal 和 Genesis 漂移；
- 覆盖 argv 重复/错配、candidate source mutation、strict JSON 重复键、RNG/输入变异、auto-reset 终态和统计二源；
- 覆盖“Stair reference 缺失/损坏但 Push 正常”和“Stair 无完整 Genesis 时 provider 不调用”。

**严格验收**

- 所有漂移在对应可信边界 fail closed，错误 checkpoint/argv 在 subprocess 启动前拒绝；
- candidate 失败与 infrastructure 失败权限语义不混淆；
- denied 记录不推进 current-head，accepted 才推进；
- 公共 prompt/ledger 不出现私有 seed、env-to-bin、服务器路径或未公开异常。

### G4：旧任务与全量回归

**工作**

- 运行 Push unit/integration、dry-run、materialize 和 live-loop 回归；
- 运行 Stair、Genesis/replay、Runner safety 和脚本入口回归；
- 运行标准 unittest discover、pytest 收集、Ruff `E9,F,I`、compileall、`git diff --check`。

**严格验收**

- Push 的默认 config、预算、CLI snapshot、baseline eval-only 和 warm-start 安全行为完全不变；
- 所有测试通过，无误收集、warning 或隐藏 skip 增量；
- shared 目录扫描不含 Stair 高度、失败分类、RewardInputs 或 task protocol 字段；应用 wiring 中仅允许 task ID/provider 装配信息。

### G5：服务器 production smoke 与发布

**工作**

- 提交并推送后 fast-forward 同步服务器正式仓库；
- 服务器验证默认 Push 解析不触发 Stair resolver；
- 显式 Stair 执行 load-only Genesis、materialization/preflight 和不启动 PPO 的 prompt 构造；
- 核对本地、origin、服务器 HEAD 与 clean 状态。

**严格验收**

- Stair 从正式 R0 Genesis 进入 candidate round 的 prompt、source head、公开反馈和 argv 均可重建；
- smoke 不创建伪 ledger，不修改 Genesis，不留下 simulator/Python 子进程；
- GPU 回到 smoke 前基线；
- 只有全部验收通过后才标记 Phase G 完成并启动 15-round 正式实验。

## 5. 预计修改面

- `src/true_rl_agent/composition/defaults.py`：唯一生产 registry 与惰性 provider wiring；如 lazy 壳需要独立复用文件，只能包含 task-neutral provider 生命周期逻辑。
- `scripts/run_stair_live.py`、`scripts/materialize_stair_job.py`：删除 staging 组合构建，复用默认 registry。
- `scripts/create_stair_reference_genesis.py`：复用默认 registry，并从 resolved composition 获取唯一 reference artifact/config。
- 删除 `src/true_rl_agent/tasks/stair_climbing/staging.py`。
- 更新 composition、Stair entrypoint、Phase B 历史边界和 extension guide 测试。
- 更新 `revised_plan.md`、Phase F/Phase G 实施记录及机器生成证据引用；派生 hash 不手写复制。

## 6. 完成定义

Phase G 只有在以下条件同时满足时完成：

1. Phase F 已累计至少两条真实完整评估记录，所有拒绝记录闭合，且整体 replay 已通过；
2. 生产 registry 只有一个 Stair provider 来源且按显式选择惰性解析；
3. 默认 Push 全路径无行为退化；
4. Stair 只有完整 reference Genesis 后才调用 provider；
5. 所有身份、hash、argv、源码和统计故障注入 fail closed；
6. task-specific 语义未进入共享层；
7. 本地、origin、服务器代码和测试结果一致；
8. 无 staging 二源、无残留进程、无未解释的临时兼容代码。

## 7. 原实施完成记录与审计更正（2026-08-03）

以下生产接入工作已经完成，且没有重写 Genesis、ledger 或 checkpoint；中间审计曾因旧 v2 authoring provenance 不完整而撤回完成结论，后续 v5 正式重跑已闭合该门：

- `build_application_composition_registry()` 现在是 Push 与 Stair 的唯一生产注册源；Push 保持默认且直接装配，Stair 通过 task-neutral lazy provider 在显式选择后才解析官方 R0。
- Stair live、materialize 与 reference Genesis 三个入口均消费同一个 production registry；开发期 staging registry 已删除，不再存在第二套 reference/config 装配路径。
- 新记录的 ledger epoch、provider prompt 精确字节、每次 provider attempt 与最终选中 prompt 由 append-only authoring journal 和 ledger v5 绑定；中断后保留旧 attempt，并只在该轮剩余封存预算内分配新 ID。正式 Phase F ledger v2 保持原字节，但只保留训练/评估等科学证据，authoring provenance 明确不完整。
- 正式 Phase F v2 artifact 的 Genesis 与 ledger 摘要保持不变；直接从旧 v2 ledger continuation 仍必须返回 `authoring_evidence_legacy_unverified`。新的 v5 epoch 只复用同一不可变 Genesis，不把旧记录回填成新证据。
- 默认 Push 路径的全新进程探针证明不会导入 Stair 模块；显式 Stair production preflight 从 canonical Final V2 和官方 R0 解析出固定 `2000` iterations 与 `196,608,000` env-steps，且不启动 PPO。
- 早期生产接入快照的本地 `pytest` 为 `798 passed, 16 skipped, 588 subtests passed`；本地标准 `unittest` 为 `814 tests, OK, skipped=16`。最终验收结果见 7.2。
- Ruff `E9,F,I`、compileall 和 `git diff --check` 均通过；共享层扫描与 Push/Stair 回归未发现 Stair reward、terrain、七桶或失败分类语义泄漏。

上述是撤回完成结论时的历史状态；正式 v5 重跑与后续根因审计已经完成，当前结论由 7.2 取代。

### 7.1 v2 到 v5 的可执行 ledger epoch 边界

迁移不复制、不迁移也不重建 Genesis。通用 live 入口现在显式区分两个目录：

- `--session-genesis-dir` 指向原正式目录，只读加载并验证原 Genesis、reference 私有证据和 evaluation binding；
- `--output-dir` 指向一个新的可写 epoch 根目录，只承载新 ledger、round authoring、candidate、runner 和 summary；
- `--ledger-epoch-id` 是新 epoch 的稳定身份；当两个目录不同时必须显式提供。

首次启动会在新目录 create-once 发布 `ledger_epoch.json`，原子绑定原 Genesis 文件字节、`session_genesis_hash`、原目录旧 ledger 的存在状态/字节哈希，以及新 `ledger.jsonl` 的绝对路径。其 manifest hash 进入每条 v5 ledger 记录并由 replay 验证。后续 resume 在 provider 调用前重新验证这些绑定；原 Genesis、旧 ledger 或新 ledger 的 epoch 身份发生任何变化都会 fail closed。旧目录中的 v2 ledger 不参与新 ledger replay，新 epoch 明确从同一个 R0 Genesis current-best 开始，不能把其他 epoch 的 candidate 决策伪装成本 epoch history。

正式重跑命令形状为：

```powershell
python scripts/run_stair_live.py `
  --session-id <formal-session-id> `
  --session-genesis-dir <existing-formal-v2-root> `
  --output-dir <fresh-phase-f-v5-epoch-root> `
  --ledger-epoch-id phase-f-v5-rerun `
  --rounds 2
```

同一 epoch 的整个 live loop 由进程持有的 session lease 串行化，单条 ledger append 另由 ledger writer lock 串行化；进程异常退出后 OS 自动释放所有权，遗留 sidecar 文件不代表 stale owner。正式服务器重跑已使用该路径完成，没有复制 Genesis 或改写旧 ledger。

### 7.2 正式 v5 证据与最终验收

- 正式 v5 artifact 位于服务器 `/data/yuyangliu/true_rl_agent_phase_f/formal_v5_0dbb214_20260803T084840Z`，复用原正式 R0 Genesis；新 epoch 共两条 ledger v5 记录，两条都完成 train、public eval 和 sealed validation。
- 第一轮第一次 draft 通过 verifier 但 smoke 失败，第二次修复 draft 的 verifier、smoke 与 anti-degenerate gate 全部通过后才进入训练；第二轮一次 authoring attempt 通过。selected attempt 均为该轮最后一次已完成 attempt。
- 两个候选都使用与 R0 相同的 TrainingBinding、相同 initial policy state、随机初始化、无 warm-start checkpoint，以及 `4096 × 24 × 2000` 固定预算；七桶 public/sealed 证据完整。两者 public accept 均为零且 holdout floor 失败，因此都被合理 denied，R0 current-best/current-head 不变。
- `f19ef37` 进一步禁止从含 accepted 决策或外来 Genesis 的 predecessor ledger 创建新 epoch，强制 authoring evidence 选择最终 attempt，并保证 lazy composition 在并发首次访问时仍只解析一次。
- 同提交服务器 load-only 从两条记录恢复，执行轮数为零，Genesis、旧 ledger、epoch manifest 与新 ledger 的字节摘要前后不变；默认 Push 两轮 dry-run 的 13 项检查全过，显式 Stair preflight 保持 2000 iterations 且未启动 PPO。
- 最终本地标准 `unittest` 为 `846 tests, OK, skipped=16`；`pytest` 为 `830 passed, 16 skipped, 594 subtests passed`。服务器标准 `unittest` 为 `846 tests, OK, skipped=2`。Ruff `E9,F,I`、compileall、`git diff --check`、仓库 clean 和资源清理均通过。

**最终结论：** Phase G 已完成。生产 registry 只有一个装配源，Push 默认路径不解析 Stair，显式 Stair 路径从不可变 R0 Genesis 进入 v5 provider loop；Phase F/G 的 evidence、replay、任务隔离和旧任务回归门均已满足，可以启动新的 15-round 正式实验。

### 7.3 首次 15-round 诊断运行与 prompt 边界修正（2026-08-04）

首次独立长运行位于服务器 `/data/yuyangliu/true_rl_agent_phase_g/formal_15round_f6c2806_20260803T141333Z`，基于提交 `f6c2806`、正式 R0 Genesis 和新 ledger epoch。运行在第三条完整 ledger v5 记录落盘后主动停止，未把已知错误继续传播到 R4-R15；无关服务器进程未被终止。

- R1 `public_accept=0.164286`，以 `base_pitch_instability` 和 stalled progress 为主要公开反馈，denied；
- R2 `public_accept=0`，训练奖励的无界 forward-progress 机制导致 base contact/stall 崩塌，denied；
- R3 删除无界进度机制并加入有界 goal progress 与接触/高度约束。固定 `4096 × 24 × 2000` 训练完整结束，initial-policy pairing、预算、runtime activation、七桶 public eval 和 sealed validation 全部通过；`public_accept=0.517857`、sealed mean `0.515306`，但低于 R0 `0.546429` 且未满足 `min_delta=0.03`，因此以 `noise_aware_delta_not_met` 合理 denied。

官方 load-only continuation 以 `--rounds 3` 重放该目录，round、prompt、authoring evidence、runner candidate 和 eval audit 十项一致性检查全部通过。该目录保留为诊断证据，不再继续 authoring。

运行同时暴露两项 prompt 真值问题：

1. shared `public_history` 把 eval 的 `candidate_package_loaded=false` 误解释为训练候选未加载，和真实 `runtime_activation_gate=pass`、R3 的 `864018` 次 reward-term 调用矛盾；
2. Stair prompt 只携带 latest 的简化反馈，遗漏七桶成功率和完整 primary failure breakdown，且 R1 best-observed 源码与 R2 latest 反馈没有形成同一候选身份下的配对。

修正保持任务边界：shared 层只把 task-neutral 的“训练 runtime 激活”和“eval 代码隔离”拆成两个字段；七桶、失败原因、diagnostics、反馈选择和大小边界全部留在 `tasks/stair_climbing/`。Stair 反馈由已验证 `public_dev` 单源重建，按 candidate ID/source checksum 绑定，保留 reference current-best、最近三条、best nonaccepted 和 accepted current-best 的去重集合，总条数不超过 5。真实 R0-R3 ledger 投影得到 4 条反馈，task-owned sections 为 `16958` bytes，未接近 `32 KiB` 上限。

新的正式 15-round 运行必须使用修正后的同一提交、新 output root 和新 ledger epoch；在服务器同提交 replay、真实 prompt 重建、Push 回归及全量测试通过前不得启动。

### 7.4 修正后正式 15-round 结果与最终关闭（2026-08-04）

修正后正式运行位于 `/data/yuyangliu/true_rl_agent_phase_g/formal_15round_eb4ede9_20260804T063045Z`，基于 clean commit `eb4ede94d01b48f8033e85d7ada257c8d6e6b4fb`、原不可变 R0 Genesis 和新 ledger epoch。15 个候选全部使用同一 TrainingBinding、相同随机初始化策略状态、无 warm start、`4096 x 24 x 2000` 固定预算，并全部完成 public 与 sealed 七桶评估；没有 eval incomplete 或 infrastructure failure。

- R9 `stair_stable_progress_v2` 以 public `0.589286`、sealed `0.576531` 首次超过 R0 与 `min_delta=0.03`，被接受；
- R14 `stair_stable_progress_contact_v3` 从 R9 accepted 源码生成，以 public `0.621429`、sealed `0.602041` 再次接受；
- 其余 13 轮均按 threshold 或 required gate 正确拒绝，未推进 current-head；
- R15 从 R14 生成但 public 仅 `0.396429`，拒绝后最终 head 仍为 R14；
- 最终相对 R0 的 public 绝对提升为 `0.075`，0.30 m 桶从 `0.175` 提升到 `0.675`，base-contact 失败率从 `0.378571` 降到 `0.25`；0.40 m 仍为 0。

生产入口 load-only replay 得到 `resumed_from_round_count=15`、执行新轮数为 0，live summary 10 项与 eval audit 12 项检查全部通过。额外逐字节审计重新验证了 15 个 candidate artifact、15 个 checkpoint、24 份 attempt prompt、parent/source head、预算、初始化、eval 隔离、七桶计数和 required gate，19 项全部通过。运行进程和 GPU1 已清理，服务器原有无关进程未受影响。完整逐轮证据见 `phase_g_15round_experiment_audit.md`。

最终本地标准 unittest 为 `850 tests, OK, skipped=16`，pytest 为 `834 passed, 16 skipped, 595 subtests passed`；服务器正式仓库 unittest 为 `850 tests, OK, skipped=2`。Ruff `E9,F,I`、compileall 和 `git diff --check` 全部通过。

**最终结论保持：Phase G 已完成。** 本次正式效果实验没有发现需要回改 Phase F/G 合同的硬错误；失败轮均属于候选奖励机制问题，并被现有边界正确隔离。后续优化应从 R14 accepted 源码在新 epoch 中继续，不改变固定协议、预算或随机初始化比较合同。
