# Phase 6 后 Agent 持续提升方案（批判性采纳版）

日期：2026-06-21

背景：

- 任务：Unitree G1 29DoF 站立抗 push recovery。
- 最近正式运行：`2026-06-21_phase6_live_deepseek_001`。
- Phase 6 结果：Round 3 找到一次大幅有效改进，Round 4/5 未继续提升。
- 最终接受结果：Round 3。
- Round 3 public 成功率均值：`0.854166`。
- Round 3 前 baseline public 成功率均值：约 `0.557292`。

这份文档不是把所有建议简单相加，而是对已有实验结果、当前系统架构、提示词设计、ENPIRE/Eureka 思路和用户补充判断做批判性合并，给出下一步最值得实现的路线。

## 1. 对补充判断的批判性采纳

我基本接受以下判断：

1. **config-only 空间的结构性杠杆太少**。

   这确实是头号问题。Round 3 的成功来自“训练时加入适中 push 暴露”这个结构性杠杆；这个杠杆打开后，继续只调 push 幅度、interval 和通用 reward weight，边际收益会快速下降。

2. **当前验证噪声太高，难以判断小增益**。

   这个判断成立。Round 3 后剩余可提升空间已经不大，而当前 3-seed、500 iteration、128 env eval 的方差足以吞掉 `+2-3pp` 的真实增益。没有更低噪声的验证，持续微爬升很难成立。

3. **单提案贪心搜索不适合这个阶段**。

   这个判断成立。Round 4 太保守，Round 2/5 太激进，说明 LLM 对 push 幅度和训练稳定性之间的响应面没有足够模型。best-of-K / 群体候选比单提案更合理。

4. **反馈太粗，Agent 容易乱拧 reward weight**。

   成立。现在 prompt 的 public summary 还不够机制化，缺少 reward term 贡献、失败时序、push-conditioned breakdown 和 secondary metrics。

我需要加条件或修正的地方：

1. **“固定 eval 饱和后没有梯度”是重要问题，但不应急着把更难 eval 纳入每轮优化目标。**

   如果马上让 Agent 围绕更难 OOD 层做每轮优化，容易把任务边界搞乱。更合理的是先增加 protected report-only OOD / stress 层，用于最终诊断和阶段性审计；等 public_accept 稳定后再决定是否把更难层级纳入正式 accept/reject。

2. **“训练更久/更多 env”不是所有候选都要做。**

   如果每个候选都 1000+ iteration、512 eval episode，算力会被低质量候选浪费。但这里不能简单理解为“短训 success 可以给候选排序”。对这个任务，curriculum 类候选往往需要足够 iteration 才显现效果，短训最多适合作为 runability / 明确崩溃过滤器。正式排序必须依赖统一、足够的固定训练预算，或者先证明 short-to-full rank correlation。

3. **“Extension C 代码级权限”方向正确，但不应作为近期直接实施项。**

   ENPIRE 的强处确实来自更大的代码级动作空间，但我们当前 verifier 还没强到能安全放开 reward implementation、observation、aux loss 等源码改动。近期应先把 config-level 的候选池、噪声控制、反馈机制做好。

4. **“ENPIRE 能持续爬”不能直接等价迁移到我们当前任务。**

   ENPIRE 的 manipulation 任务、真机 reset/verification、policy learning 范式和我们的人形站立抗 push PPO 不完全相同。它提供系统设计启发：大动作空间、群体搜索、低方差验证、经验迁移；但不代表我们应照搬全部权限和流程。

## 2. 根因：为什么 Phase 6 一跳后停住

### 2.1 只有一个大杠杆：训练时 push 暴露

Round 3 的有效改动是：

- `phase2_train.push.velocity_range_mps.y: [0.0, 0.0] -> [0.0, 0.3]`
- `phase2_train.push.interval_range_s: [1.0, 1.0] -> [0.8, 1.2]`

它把 public success 从约 `0.557` 推到 `0.854`，说明 baseline 最大缺陷是训练时没有足够 push recovery 暴露。

但这个问题被解决后，当前 action space 里没有第二个同量级结构杠杆：

- reward weight 微调在 Round 1/5 没证明有效；
- push 幅度继续增大会造成训练退化；
- 小 lateral push 在 Round 4 没带来增益；
- 当前没有 schedule、domain randomization、observation history、reward-term registry 等新自由度。

### 2.2 噪声地板吞掉小增益

Round 3 后 best 已达到 `85.4%`。从这个位置继续提升，很多真实有效改动可能只有 `+2-5pp`。

但当前决策门槛是：

`accept iff candidate_mean - best_mean > max(5pp, pooled_SE)`

在 3 seeds 下，pooled SE 经常接近或高于小增益。也就是说，当前系统只能可靠识别“大改进”或“明显退化”，不擅长识别精修。

### 2.3 单提案贪心没有响应面

当前 loop 每轮只测一个候选。LLM 没有机会同时比较：

- 小幅 push ramp；
- 中等 push ramp；
- reward-only；
- PPO-only；
- curriculum-only；
- combined candidate。

所以它只能靠语言推理猜一个点。Round 4 猜得太弱，Round 2/5 猜得太猛。

### 2.4 fixed public eval 缺少更高层级的进步目标

public eval 是固定 `phase2_public` 分布。它必须受保护，Agent 不可修改，这是正确的。

但当当前分布成功率已经较高时，单一 binary success 会变钝。后续要继续进步，需要：

- 更细的 public diagnostics；
- 更稳定的 public_accept；
- 人定义、Agent 不可改的 OOD/stress 层做阶段性检查。

### 2.5 反馈不是机制级反馈

当前 Agent 主要看到 success、failure label、condition flags 和 operation diff。

它还不知道：

- 每个 reward term 是否过大、过小或无效；
- 哪个 push 方向导致失败；
- 失败发生在 push 后多久；
- 是欠反应还是过反应；
- 动作率、能耗、振荡是否恶化；
- 训练早期是否已 collapse。

因此它会倾向于“加大姿态/高度惩罚”这类直觉操作，但实验表明这类操作不是稳定杠杆。

## 3. ENPIRE / Eureka 给我们的启发

ENPIRE 能持续爬升，不是因为单个 LLM proposal 神奇，而是因为系统具备四个条件：

1. **大动作空间**

   ENPIRE Agent 可以改算法、数据采样、BC/RL 组合、控制器补偿和训练代码。我们当前只开放 config。

2. **多候选 / 多 agent / 群体搜索**

   ENPIRE 用 fleet 和多 agent 并行测试假设。Eureka 也会一次生成多个 reward 候选再选优。我们当前是一轮一个候选。

3. **低方差验证**

   ENPIRE 的任务可以用大量 rollout 或连续成功目标降低方差。我们当前 public 验证方差较高。

4. **经验积累**

   ENPIRE 会把成功 recipe 和失败经验沉淀下来，迁移到后续任务。我们目前只有 round history，还没有稳定 public-only 经验台账。

因此，我们下一步不应该只是“继续多跑 round”，而应优先补齐：低方差验证、候选池、反馈机制、经验记忆，再逐步扩动作空间。

## 4. 优先级 P0：先降低噪声地板

对应根因：小真实增益无法被可靠判出。

### 4.1 方案

1. 先离线标定一个统一固定训练预算。

   不在 in-loop 里判断“每个候选是否已经各自收敛”。RL 收敛难判，特别是 push curriculum 类候选可能前期很差、后期才出现质变。动态收敛判定容易误杀这类候选，也会引入候选间预算不公平。

   更稳的做法是：

   - 先用 baseline 和 Round 3 accepted best 跑训练曲线；
   - 找到 success / episode length / termination rate 进入平台期的大致 iteration；
   - 选一个所有正式候选共用的固定预算；
   - 在 loop 内只对“明确崩溃 / 发散”早停，例如 reward NaN、loss 爆炸、success 长时间卡 0 且 episode length 不增长、训练进程异常退出。

2. 分层使用预算，但短训不直接作为成功率排序依据。

   - runability screen：可以短训，用于过滤 reward NaN、loss 爆炸、环境不可运行、明显训练崩溃；
   - full eval：正式候选使用统一固定预算；
   - final confirm：必要时更多 seeds / 更多 eval episodes，而不是给某个候选临时追加训练直到“看起来收敛”。

3. 增加 final eval episode 数。

   - runability screen 可继续 128 env episodes；
   - full eval / final confirm 建议 256 或 512 env episodes。

4. 增加 seed 数，但只给有希望的候选。

   - runability screen：1 seed；
   - full eval：3 seeds；
   - near-threshold / small-gain candidate：5-8 seeds。

5. 使用 paired / CRN 思路减方差。

   候选和当前 best 尽量用相同 seed 子集、相同 public eval 条件比较，decision 重点看 paired delta。

### 4.2 理由

如果没有这一步，后面的 prompt、候选池、reward reflection 都很难发挥作用，因为系统看不见小改进。

### 4.3 风险和约束

不能把所有候选都直接升级到高预算，否则算力浪费严重。但短训阶段只能作为 runability / 明确崩溃过滤器，不能在没有证据前承担候选排序功能。若未来要用短训排序，必须先做 short-to-full rank correlation 审计。

## 5. 优先级 P1：扩展 config 级动作空间

对应根因：当前 action space 缺少第二个高杠杆结构旋钮。

### 5.1 增加 push curriculum schedule

从静态 push range：

```yaml
velocity_range_mps:
  y: [0.0, 0.3]
```

升级为训练过程 schedule：

```yaml
push_curriculum:
  stages:
    - until_iter: 100
      velocity_range_mps:
        y: [0.0, 0.05]
    - until_iter: 300
      velocity_range_mps:
        y: [0.0, 0.20]
    - until_iter: 1000
      velocity_range_mps:
        y: [0.0, 0.30]
```

进一步可以支持 metric-gated curriculum：

- episode length 足够长才升难度；
- bad_orientation 高则暂缓升难度；
- action_rate / energy 爆炸则回退或冻结难度。

执行边界：

- curriculum schedule 的执行器应写在受保护训练代码里；
- Agent 只能填写 schedule config；
- 每个新旋钮都必须在 verifier 中设置边界，包括单调性、最大幅度、阶段数量、每阶段最小长度；
- metric-gated curriculum 比静态 schedule 风险更高，建议先实现固定阶段 schedule，等收敛诊断稳定后再考虑动态 gating。

理由：Round 2/5 不是证明强 push 无效，而是证明“直接强 push”容易让 PPO 训练崩。schedule 是比静态幅度更合理的杠杆，但 schedule 执行逻辑本身不能交给 Agent 任意写。

### 5.2 增加训练-only domain randomization

允许 Agent 在 verifier 约束下调整训练随机化：

- friction 小范围扰动；
- mass / COM 小范围扰动；
- motor strength 小范围扰动；
- observation noise / delay；
- push timing jitter。

评估 profile 仍然 protected，Agent 不可改。

理由：抗 push recovery 不只取决于 push velocity，还取决于接触、惯量、延迟和控制鲁棒性。训练-only DR 是 config 级但高杠杆。

### 5.3 暴露更多 PPO schedule 旋钮

可考虑开放：

- learning rate schedule；
- entropy coefficient schedule；
- clip range；
- value loss coefficient；
- minibatch size；
- gamma / lambda schedule。

明确不开放：

- PPO `epochs`。

理由：训练 collapse 可能与 PPO 更新稳定性有关，不一定只靠 reward 和 push range 能解决。但 `epochs` 直接改变优化强度和计算预算，容易成为预算作弊通道，也违背计划里锁死算力预算的要求。近期只保留 lr / entropy / clip / gamma / lambda / minibatch size 这类更可控旋钮。

### 5.4 增加 observation history 长度旋钮

push recovery 有历史依赖。短 history 可以帮助 policy 从姿态/速度变化中推断扰动后状态，而不需要显式泄露真实 push 参数。

## 6. 优先级 P2：从单提案升级为候选池 best-of-K

对应根因：单提案 greedy 搜索太脆弱。

### 6.1 方案

实现 Phase 7A candidate-pool loop：

1. 每轮 LLM 生成 `K=4-8` 个候选；
2. verifier 静态检查全部候选；
3. 每个候选先 runability screen，过滤明确崩溃 / 不可运行；
4. 通过 runability 的候选进入统一固定预算 full eval；
5. 使用 `public_dev` 做候选池内部选择；
6. top `1-2` 候选进入 `public_accept` 确认；
7. near-threshold 候选扩展到 5-8 seeds；
8. 最终仍按 public-only accept/reject。

### 6.2 候选类型约束

Prompt 要求候选必须多样化，例如：

- curriculum-only；
- reward-only；
- PPO-only；
- training DR-only；
- conservative mutation around best；
- exploratory mutation with rollback risk。

### 6.3 信任域约束

避免 Round 5 式过激改动：

- push 不允许从 `[0.0, 0.3]` 一步跳到 `[0.0, 0.5]`，除非通过 schedule ramp；
- reward weight 单轮变化幅度受限；
- 候选必须解释为什么没有重复历史失败模式；
- 超出 trust region 的候选只能 cheap screen，不能直接 full eval。

### 6.4 理由

候选池是当前最接近 ENPIRE / Eureka 的低风险改造。它不需要马上开放源码权限，却能显著提升搜索质量。

关键约束：

- 不能用同一个 public 集合同时做 best-of-K 选择和 accept/reject；
- 否则会产生 winner's curse，系统性高估被选中候选；
- 因此 candidate pool 必须和 `public_dev` / `public_accept` split 一起实现。

## 7. 优先级 P3：升级 public feedback，做机制级反思

对应根因：Agent 缺少 credit assignment。

### 7.1 增加 per-reward-term 统计

summary 中加入：

- 每个 reward term 的均值；
- 每个 reward term 的标准差；
- 哪些 term 接近 0；
- 哪些 term 主导总 reward；
- early/mid/final 阶段 reward component 曲线。

理由：让 Agent 判断 reward weight 是否值得调，而不是凭失败标签乱调。

### 7.2 增加 push-conditioned breakdown

按以下维度统计结果：

- push 方向象限；
- push magnitude bin；
- push 后时间窗口；
- final drift bin。

理由：`fall_backward` 只是表层标签。Agent 需要知道哪个扰动条件触发失败。

### 7.3 增加失败时序

区分：

- push 后立即失败；
- 恢复中段失败；
- episode 末端漂移 / 慢性失稳；
- 训练早期 collapse。

理由：不同失败时序对应不同改法。立即失败可能需要 recovery response；后段漂移可能需要 damping 或 command tracking；训练早期 collapse 说明 curriculum 太猛。

### 7.4 修 condition flags 语义

增加：

- pre-fall first violation；
- final-step-only diagnostic；
- max value before failure；
- time-to-first-threshold-crossing。

理由：避免 condition flags 变成“摔倒后所有 flag 都为真”的冗余失败率代理。

### 7.5 加 secondary metrics

加入：

- mean action L2；
- mean action-rate L2；
- energy proxy；
- oscillation proxy；
- mean max pitch / roll；
- mean min base height；
- recovery time。

理由：这些指标可以区分欠反应、过反应、动作过猛和振荡。

## 8. 优先级 P4：建立 public-only 实验记忆台账

对应根因：Agent 容易重复坏区域，跨 run 学习不稳定。

### 8.1 方案

建立：

`development_logs/agent_memory/public_experiment_ledger.md`

每条记录包括：

- candidate id；
- operation diff；
- public mean / seed values；
- accepted or rejected；
- 一句话经验；
- 不建议重复的区域。

Phase 6 初始记忆：

- `[0.0, 0.3]` y-push + `[0.8, 1.2]` interval 是高杠杆有效改动；
- `[-0.5, 0.5]` 双轴大 push 会 collapse；
- `[0.0, 0.5]` y-push + 更强姿态/高度 penalty 明显退化；
- `[-0.1, 0.1]` lateral push 对当前 best 无明显帮助。

### 8.2 理由

ENPIRE 的 Evolution 模块强调经验沉淀和配方迁移。我们的 loop 需要把 public-only 历史压缩成可复用知识，但不能引入 hidden 信息。

### 8.3 安全要求

public-only ledger 必须有 allowlist projection 和 canary 禁词测试。

如果未来在 prompt 中预置类似“push curriculum 是大杠杆”的经验，当前 best 和对照候选都必须从同一份 public-only 记忆状态出发，避免给某个候选额外提示优势。

## 9. 优先级 P5：增加受保护的更难评估层

对应根因：固定 public 分布饱和后缺少更高层级目标。

### 9.1 方案

增加人定义、Agent 不可改的评估层：

1. `public_dev`

   用于 LLM 可见开发反馈。

2. `public_accept`

   用于正式 accept/reject。

3. `iid_hidden`

   最终 audit only，绝不进入 prompt。

4. `ood_stress`

   更强 push、未见方向组合、轻微动力学扰动。初期只做 report，不参与每轮 accept/reject。

### 9.2 理由

这一步要谨慎。更难 eval 可以提供新目标，但不能让 Agent 控制 eval，也不应太早把 OOD stress 变成优化目标。初期只作为诊断层更安全。

与候选池的关系：

- `public_dev` 用于候选池内部选择；
- `public_accept` 用于最终接受判定；
- best-of-K 不能在同一个 public 集合上既选择又确认；
- `iid_hidden` 和 `ood_stress` 初期只做审计 / 报告，不进入 Agent prompt。

## 10. 优先级 P6：Extension C 代码级权限

对应根因：config-only 最终存在结构性天花板。

### 10.1 可开放能力

未来可以允许 Agent 修改或实例化：

- 新 reward term；
- observation history / features；
- auxiliary loss；
- BC regularization；
- recovery-specific policy architecture；
- controller compensation；
- curriculum implementation。

### 10.2 前置 verifier 要求

必须先实现：

- source file allowlist；
- import / runtime smoke；
- reward finite test；
- tensor shape / device test；
- 禁止外部网络和 secrets；
- 禁止 eval / seeds / verifier / failure classifier 修改；
- deterministic operation diff；
- failed candidate quarantine；
- 自动 rollback。

### 10.3 理由

ENPIRE 的核心威力来自代码级动作空间，但这也是最大风险源。当前阶段不应直接进入 P6。应先完成 P0-P4，保证实验判定、搜索和反馈质量足够高。

## 11. 推荐近期实施路线

最近一个最值得做的阶段应定义为：

**Phase 7A：协议修正与低方差候选池闭环**

包含：

1. 训练预算标定与固定 full-eval budget；
2. runability screen，只过滤明确崩溃 / 不可运行；
3. `public_dev` / `public_accept` split；
4. candidate-pool best-of-K；
5. trust-region verifier；
6. public-only experiment ledger；
7. canary prompt/ledger 泄露测试。

明确不包含：

- hidden/IID 进入 prompt；
- Agent 修改 eval difficulty；
- Agent 任意改源码；
- 直接进入 Extension C。
- curriculum schedule / training DR 的完整执行器；
- feedback v2 的全部诊断字段。

建议把后两者放到 Phase 7B：

- Phase 7B：curriculum schedule / training DR / feedback v2；
- Phase 7C：Extension C 代码级权限预研。

## 12. 最终优先级表

| 优先级 | 改造 | 主要解决的根因 | 为什么现在做 |
| --- | --- | --- | --- |
| P0 | 降低噪声地板 | 小增益不可判别 | 没有低方差验证，持续微爬升不成立 |
| P1 | 扩 config 级结构杠杆 | config-only 空间太窄 | 先补 schedule/DR/PPO 这类低风险高杠杆旋钮 |
| P2 | 候选池 best-of-K | 单提案贪心脆弱 | 用群体搜索替代单点猜测 |
| P3 | 机制级 public feedback | 诊断太粗 | 让 Agent 知道为什么失败、该动哪个旋钮 |
| P4 | public-only 记忆台账 | 重复坏区域 | 把失败和成功 recipe 转成可复用知识 |
| P5 | 受保护更难评估层 | public 饱和 | 作为诊断和阶段审计，暂不急着进每轮优化 |
| P6 | 代码级权限 | config 天花板 | 方向正确，但必须等 verifier 强化后再做 |

结论：

下一步不应只是继续多跑 Phase 6 round，也不应直接开放代码级权限。最稳的路线是先做 Phase 7A：把训练预算、公平评估、候选池选择、dev/accept split、trust-region 和 public-only 记忆这些协议问题先修好。等协议稳定后，再在 Phase 7B 引入 curriculum schedule、training DR 和更丰富 feedback。
