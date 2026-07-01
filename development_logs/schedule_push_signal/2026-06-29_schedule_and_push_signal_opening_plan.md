# Phase 8E 自由度开放计划:接通 push/DR 课程键 + 真实 push 信号

> 目标:只开放两点 —— (1) 把已"accepted/precomputed 但未生效"的 schedule 键真正接进训练;(2) 把 `push_active`/`push_vec_public` 从占位变成真实 instrumentation。
> 范围:**全部 runner 侧改动**,候选 scope 不变(仍只写 reward/loss/schedule/diagnostic)。
> 日期:2026-06-29

---

## 0. 一句话结论

两点都该做、都符合项目目标与 ENPIRE 原则,且因为"评测固定、不可动"而**训练侧自由不可作弊**。但**必须配评测加固一起做**(见 §1),否则评测分辨不出它们带来的提升(round 2/3 已并列 0.890625,差异淹没在噪声里)。

---

## 1. 前置依赖与排序(关键,不能跳过)

这两点都在**扩大训练自由度**。当前瓶颈是**评测分辨不出相似候选**。所以:

- **Proposal 1(push/DR 课程)⊗ OOD 压力 gate**:训练侧能调 push 课程 ↔ 评测侧要有更难的 push 测鲁棒性,否则选不出"更鲁棒"的配方。
- **Proposal 2(真实 push 信号)⊗ 连续恢复评分**:reward 用真实 push 信息变聪明,只有评测能测"恢复质量"(连续分,而非 64 步二值成功)时才会被选出来。

**结论排序**:评测加固(多种子/多 episode + OOD 压力 gate + 连续恢复分)与本计划**并行或先行**。本文档只覆盖这两点的开放;评测加固另立计划。但每一项的"验收"都以"硬化后的评测能测出差异"为准。

---

## 2. 统一作弊/安全原则(通用尺子)

> **候选可任意塑造"训练"(reward + 课程 + 使用真实训练-env 信号);"评测"永远是固定、runner 拥有、不跑候选代码、种子与指标隐藏的裁判。** 正是这个不可动的裁判,让所有训练侧自由都不可作弊。

每加一个自由度,过这五关(本计划每点都按此验收):

1. **只在训练侧** —— 绝不进 eval、绝不进策略观测。
2. **clamp 到非退化区间** —— 不能把任务 trivialize。
3. **契约如实标注** —— effective+bounds / source+shape+frame+fallback,并有行为探针证明契约不撒谎。
4. **进隔离 fast 校验的合成输入(非退化值)** —— bug 在训练前被拦。
5. **加"独立性"测试** —— 候选改了 X,eval 配置/eval 观测不变。

---

## 3. Proposal 1:把 push/DR 课程键真接进训练

### 3.1 现状
- `reward_weight_scale` 已生效([training_adapter.py](../../src/humanoid_agent/cold_start_lab/training_adapter.py) `_current_reward_weight_scale`,真乘进 reward weight)。
- `push_magnitude_scale / push_start_fraction / push_interval_s / domain_randomization_scale` 仅被 `precompute_schedule_table` 预计算,executor 未接 —— `training_adapter_runtime_contract()` 里 `schedule_curriculum_applied_to_env_push_or_dr: False`,`control_effectiveness_contract` 据此把它们派生为 `accepted_but_ineffective`。

### 3.2 关键架构决定:训练专用 runtime executor(不走 overlay)

> 审计修正(P1):`apply_phase1_overlay` 被**训练与评测共用**([train_smoke:231](../../scripts/phase1_push_recovery_train_smoke.py) / [eval:463](../../scripts/phase1_push_recovery_eval.py));overlay 只配 `push_robot` 事件的 `interval_range_s` 与 `velocity_range`([isaac_overlay.py:65-69](../../src/humanoid_agent/phase1_push_recovery/isaac_overlay.py))。**绝不能把课程应用塞进 overlay**,否则同时改了 eval。

> 审计修正(P1 load-order):`env_cfg` 在 [train_smoke:247](../../scripts/phase1_push_recovery_train_smoke.py) `gym.make()` 处就**冻结**,而 `Phase8ETrainingAdapter` + schedule 表在其后 [:253](../../scripts/phase1_push_recovery_train_smoke.py) 才构造。⇒ config-time 改 `env_cfg` 在时序上拿不到 schedule。课程又是 progress-based 的。**二者择一,结论:用 runtime 执行器**。

**改动点(runner 侧,训练专用)**:
1. **新增 `Phase8ECurriculumExecutor`**,**只由训练侧的 `Phase8EVecEnvRewardWrapper.step`**([training_adapter.py](../../src/humanoid_agent/cold_start_lab/training_adapter.py) `Phase8EVecEnvRewardWrapper`)在每个课程检查点调用 —— eval **完全不 import/不调用**它(eval 是 checkpoint-only、不走这个 wrapper,天然隔离)。
   - executor 读取**已预计算**的 schedule 表里这四个键(候选 schedule fn 不进 env 热循环、更不进评测),clamp 后**在运行时改 live env 的 event manager**:`push_robot.interval_range_s` / `push_robot.params["velocity_range"]`(magnitude=对 velocity_range 缩放,start_fraction=课程开始的 progress 阈,interval=间隔)与 DR 事件 params。
   - **不碰** `env_step_budget`(预算不变)。
2. **可行性 spike(实现前必做,见 §3.6)**:确认 Isaac event manager 支持**运行时改 `push_robot` / DR EventTerm 的 params 并即时生效**;若 event manager 缓存了 params,需找到正确的更新入口或退而用"分段重建 event term"。spike 不通过则本点不上线。
3. **clamp + 目标难度地板(防作弊核心,值写进 immutable runner config)**:
   - `push_magnitude_scale ∈ [0.5, 2.0]`、`push_start_fraction ∈ [0.0, 0.8]`、`push_interval_s ∈ [0.5, 5.0]`、`domain_randomization_scale ∈ [0.5, 1.5]`(默认值,可调)。
   - **(审计修正 P2)目标难度地板**:训练**最后 X%(如 20%)progress 必须回到 runner-owned 目标难度**(`push_magnitude_scale≥1.0`、DR≥1.0),且**每 episode 最小 push 次数 / 最小 DR 强度**由 runner 强制 —— 防止 LLM 在 clamp 内长期停在 easy 端、靠噪声蹭 public eval。地板由 runner 在 clamp 之后再叠加(取 max),候选改不动。
   - clamp/地板在 runner 应用,候选输出超界一律截断/抬升,summary 记录原值与生效值。
4. **`domain_randomization_scale` 显式 knob 映射(审计修正 P2)**:必须写清它到底缩放哪些 **runner-owned** DR knob,否则 `effective_now` 是撒谎。建议明确:**摩擦系数范围、base mass 扰动、motor strength 扰动**(物理域随机化);**明确排除** observation corruption(overlay 现在就把 `enable_corruption=False`,obs 噪声不在本次范围)与 terrain(不在本次范围)。映射写进 immutable config 并在 contract 的 `training_effect` 里逐项列出。
5. **runtime_contract 翻转**:executor + spike 通过后,`training_adapter_runtime_contract()["schedule_curriculum_applied_to_env_push_or_dr"]` 改为 **True**。因 `control_effectiveness_contract` 从该 flag 派生,这四个键**自动**从 `accepted_but_ineffective` 升为 `effective_now`(单一事实源)。同时给每个键补 `bounds`、目标难度地板说明、`training_effect`(含 DR knob 映射)。

### 3.3 契约/合成同步
- `SCHEDULE_CONTROL_DETAILS` 给这四个键补 `bounds`、把 `training_effect` 从 "not yet connected..." 改成真实描述。
- 不涉及 RewardInputs 字段,合成校验无需改。

### 3.4 测试/验证(五关)
- **行为探针(契约不撒谎)**:扩展现有的 `control_effectiveness` 行为探针 —— 断言"改 `push_magnitude_scale` 确实改变 live env 的 `push_robot.params['velocity_range']`;改 DR 键确实改变映射到的那几个 DR EventTerm params",且 `effective_keys` 现含这四个键。**flag True ⟺ 真生效**,二者必须同源/一致,杜绝 `schedule_curriculum_applied_to_env_push_or_dr` 与实际脱节。
- **clamp + 地板测试**:候选 `push_magnitude_scale=0`/`=99` → 训练侧实际值被截到 `[0.5,2.0]`;且训练**最后 X% progress 的生效值 ≥ 目标难度地板**(≥1.0);summary 记录原值/截断值/地板值。
- **eval 独立性测试(强化,审计修正 P1)**:eval 走共用 `apply_phase1_overlay` → `push_robot` 是 runner-owned **固定目标难度**;断言**任何候选 schedule 都不进入 eval 的 `Phase8ECurriculumExecutor` 调用路径**(eval 不构造 `Phase8EVecEnvRewardWrapper`),即 eval 的 push/DR 配置与候选课程完全解耦。
- **预算不变测试**:课程改动后 `env_step_budget` 仍 = 49,152,000。

### 3.5 作弊边界(为什么安全)
- 候选不知 eval 种子(隐藏)→ 无法把课程调成迎合 eval。
- 把训练调简单(push 调小/推迟/DR 关)→ 策略没练过推力 → **固定评测直接惩罚** → 自我纠正(第一道防线)。
- **目标难度地板**:训练末段强制回到 runner-owned 目标难度 + 每 episode 最小 push/DR(第二道防线)→ 杜绝"clamp 内长期 easy 端蹭噪声"。
- clamp 防退化极值;**executor 只在训练 step wrapper、eval 不调用** → "训练=评测"过拟合在结构上不可能。

### 3.6 可行性 spike(实现前必做,gate 本点上线)
- **目标**:确认 Isaac event manager 是否支持**运行时**改 `push_robot`(及映射的 DR)EventTerm 的 params 并即时生效。
- **做法**:小脚本起一个 G1 env,运行中改 `env.event_manager` 里 `push_robot.params["velocity_range"]`,跑若干步,核实施加的推力幅度确实随之改变(读 root velocity 跳变或 event 触发记录)。
- **判定**:能即时生效 → 走 §3.2 runtime executor;若 params 被缓存不生效 → 改用"按 progress 分段、在段边界重建/重配 event term"的方案,或本点降级为"少数离散难度档"。**spike 不过不接 executor。**

---

## 4. Proposal 2:把 `push_active` / `push_vec_public` 变成真实信号

### 4.1 现状
- 二者是占位:`_build_inputs` 里 `push_active = _constant_like(rewards, 0.0)`、`push_vec_public = _zero_matrix_like(rewards, width=2)`;契约标 `not_instrumented`;合成输入也填 0。

### 4.2 结构性安全前提(已验证)
- 候选 scope 只有 reward/loss/schedule/diagnostic;`train_observation_transforms` 被 [recipe_adapter.py:68](../../src/humanoid_agent/cold_start_lab/recipe_adapter.py) **硬拒**。
- ⇒ push 信号**只能进奖励函数**(privileged shaping 信息),**绝无可能进入被评测的策略观测**。评测时策略只靠自身本体感觉行动,任务不变。这是标准 privileged-reward。

### 4.3 改动点(runner 侧 instrumentation)

> 审计修正(P1 #3):**现状没有"实际施加 push 向量"的 instrumentation**。overlay 只配 `push_robot` 事件的 `interval_range_s` 与 `velocity_range`([isaac_overlay.py:65-69](../../src/humanoid_agent/phase1_push_recovery/isaac_overlay.py));Isaac 的 `push_by_setting_velocity` 在 event manager 内部采样并 set root velocity,**采样值默认不落任何 public buffer**。所以"用真实 push 状态填充"需要**先做 instrumentation spike(§4.6),拿不到就不许把 contract 翻成 real**。

0. **instrumentation spike(实现前必做,见 §4.6)**:确认能否拿到"本步实际施加的 push 向量 + 是否触发"。
1. **真实填充(spike 通过后)**:runner 在**训练** env 的 `_build_inputs` 用 push 事件 instrumentation 填充:
   - `push_active`:`[N]`,本步该 env 是否触发推力(1/0 或强度)。
   - `push_vec_public`:`[N, 2]`,本步实际施加的 xy 推力向量,**声明坐标系**(建议 base-frame xy;若用 world 必须标明)。
   - 来源只能是**训练** env 的 push instrumentation,**严禁**从任何 eval/隐藏状态取值。
   - **spike 拿不到真实值时**:**不翻 contract**,保持 `not_instrumented`(或提供一个**明确标注为 proxy** 的粗信号,如由 root velocity 跳变推断的"疑似 push",contract 里写明 `source_status: derived_proxy_not_true_applied_push`)。绝不把 proxy 伪装成 real。
2. **契约更新**(`REWARD_INPUT_FIELD_METADATA` + `REWARD_INPUT_FIELD_SHAPES` 已含 shape):
   - `push_active` / `push_vec_public` 的 `source_status` 从 `not_instrumented` → `observed_when_push_active_else_zero`(或类似);
   - 补 `units`、`coordinate_frame`、`fallback_behavior`("无推力本步为零向量;reward 须把零向量当成'本步无推力'");
   - 因 `_input_field_status` 从 metadata 派生,改 metadata 自动传播到 contract,无需手改第二处。
3. **合成校验同步(关键,否则契约撒谎 + fast 校验失效)**:
   - `_synthetic_field_value` 给 `push_active`/`push_vec_public` 填**非退化**值(如部分 env 有推力、push_vec 非零),让隔离 fast 校验能跑出"只在有推力时暴露的 reward bug";
   - 这样 contract→synthetic→real 三处一致的闭环继续成立。

### 4.4 测试/验证(五关)
- **真实性测试**:有推力的 env step,`push_active=1`、`push_vec_public` 非零且方向与施加推力一致;无推力 step 为零。
- **eval 零接触测试**:断言 trusted eval(checkpoint-only、不跑候选 reward)**不构造 RewardInputs / 不暴露 push_vec 给策略**;eval 观测不含 push。
- **obs 隔离测试**:候选试图通过 `train_observation_transforms` 注入 push → 仍被硬拒(回归测试)。
- **合成非退化测试**:`build_synthetic_reward_inputs` 的 `push_vec_public` 非全零;一个"用 push_vec 的 reward"在 fast 校验里能跑通,一个"误用 push_vec shape"能被拦并定位到 term。
- **契约不撒谎**:`reward_inputs_field_status()` / contract 对这两字段不再是 `not_instrumented`,且与 `_build_inputs` 实际行为一致。

### 4.5 作弊边界(为什么安全)
- **只进 reward、永不进策略观测**(由 scope + obs 硬拒结构性保证)→ 评测时策略拿不到 push_vec,无泄露、任务不变。
- push_vec 是**训练**推力(runner 课程),非 eval 推导;候选看不到 eval 推力 → reward 无法过拟合 eval → 泛化 → 固定评测自我纠正。
- instrumentation 只在训练 `_build_inputs` 填充;实现时审查不得读 eval/隐藏状态。

### 4.6 instrumentation spike(实现前必做,gate 本点上线)
- **目标**:能否在每步拿到"实际施加的 push 向量 + 是否触发",且来源是训练 env(非 eval)。
- **候选做法**(spike 里逐一验证):
  1. **包裹/挂钩 event 函数**:用 runner 自己的 wrapper 替换 `push_robot` 的 event func,采样后把 velocity 写进 runner buffer(最忠实)。
  2. **读 root velocity 跳变**:在 push 触发步比较前后 root lin vel 差(近似,需对齐触发时刻)。
  3. **读 event term 内部状态**:若 Isaac 暴露上次采样值。
- **判定**:方案 1/3 能拿到忠实值 → 填 real、翻 contract;只有方案 2 → 标 `derived_proxy`,**不翻 real**;都拿不到 → 维持 `not_instrumented`。
- spike 同时核实:该 buffer **只在训练侧存在**,trusted eval 路径不构造、不读取它(对齐 §4.4 eval 零接触测试)。

---

## 5. 实施顺序与验收

| 步 | 内容 | 验收(以"硬化后评测能测出差异"为准) |
|---|---|---|
| 0 | (并行/先行)评测加固:多种子/多 episode + OOD 压力 gate + 连续恢复分 | 评测能区分相似候选,round 2/3 那类并列消失 |
| 1a | **§3.6 spike**:Isaac event manager 运行时改 push/DR params 可行性 | 即时生效 / 给出降级方案;**不过则步 1 不接** |
| 1b | Proposal 1 训练专用 `Phase8ECurriculumExecutor`(step wrapper)+ clamp + 目标难度地板 + DR knob 映射 + runtime flag 翻转 + 契约派生 | §3.4 四类测试全过;`effective_keys` 含四键;eval 独立性测试过 |
| 2 | Proposal 1 ⊗ OOD:在硬化评测上确认"更难课程→更鲁棒策略→评测更高" | 课程改动能在评测上产生可测增益 |
| 3a | **§4.6 spike**:能否拿到本步真实施加 push 向量 | 拿到 real / 退 proxy / 维持 not_instrumented;**结论决定步 3b 是否翻 contract** |
| 3b | Proposal 2 instrumentation(real 或明确 proxy)+ 契约 + 合成非退化同步 | §4.4 五类测试全过;contract/synthetic/real 一致;无 real 时 contract 不撒谎 |
| 4 | Proposal 2 ⊗ 连续分:确认"用 push_vec 的 reward 能被评测选出来" | push-aware reward 在连续分上可被区分 |

每步并入主干前:5 个 selftest 全绿 + 对应行为/独立性探针 + 服务器实测;`commit→push→服务器同步` 按既有流程。

---

## 6. 风险与回滚

- **契约/flag 脱节**(最大风险):`schedule_curriculum_applied_to_env_push_or_dr` 或 push 字段 metadata 与实际 executor 不一致 → 契约撒谎。**缓解**:每点都用行为探针把"契约声明"与"真实行为"对拍,作为 selftest 一部分。
- **课程误入共用 overlay → 评测被同时改**(审计修正 P1,高危):`apply_phase1_overlay` 训练/评测共用。**缓解**:课程只走训练专用 `Phase8ECurriculumExecutor`(step wrapper),**绝不进 overlay**;§3.4 eval 独立性测试守门。
- **spike 没做就硬接 / 把 proxy 当 real**(审计修正 P1 #3 + load-order):runtime param 改不动、或拿不到真实 push 向量却翻 contract → 撒谎或无效执行。**缓解**:§3.6/§4.6 spike 作为 gate,不过不上线;拿不到真实值只标 `derived_proxy`/`not_instrumented`。
- **clamp/地板缺位** → 候选在 clamp 内长期 easy 端蹭噪声。**缓解**:clamp + **目标难度地板**(末段回到目标难度 + 每 episode 最小 push/DR)在 runner 强制;无此不得上线。
- **`domain_randomization_scale` 无 knob 映射** → `effective_now` 撒谎。**缓解**:显式映射到摩擦/质量/motor strength,排除 obs 噪声与 terrain,写进 contract `training_effect`。
- **push 信号误入观测**(Proposal 2)→ 评测任务被改。**缓解**:obs 硬拒回归测试 + eval 零接触测试。
- **回滚**:两点均为可开关的 runner 侧能力。Proposal 1 把 flag 设 False + executor 短路即回到"accepted_but_ineffective";Proposal 2 把字段填充改回零 + metadata 标回 `not_instrumented`。契约与合成随派生自动回退。

---

## 7. 真 code-level 对齐审计与保障(关键)

**核心判断:本计划不破坏 code-level 接口,但 Proposal 1 带来一个 incentive 层面的偏离风险,必须加保障。**

- **Proposal 2 —— 强化 code-level**:`push_vec_public` 是 reward **代码**的新输入,候选仍须用代码写出 novel reward 逻辑(如"惩罚沿推力反方向的位移")。它加深而非削弱"候选写真实 reward 代码"。✓
- **Proposal 1 —— 有偏离风险(非接口违规)**:`ScheduleTerm` 确是声明过的 code-level callable;但这四个键是 **runner 拥有的 config 旋钮**,经一个很薄的 callable 暴露(`return {"push_magnitude_scale": 1.5}`)—— 候选只是选值,curriculum 的**实现是 runner 写的**。它处在 code-level 谱系里"偏声明"的一端。

**两个具体偏离风险**:
- **风险 A(已用代码核实)**:候选 reward 是**叠加**到 base 任务 reward 上的([training_adapter.py:327-331](../../src/humanoid_agent/cold_start_lab/training_adapter.py),无候选 reward 时 `return rewards`),且 contract **允许零 reward term**(noop 合法)。⇒ **一个只写 schedule、零 reward 代码的候选也能训练(吃 base reward)并可能获胜** → 直接偏离"code-level reward 发现"。
- **风险 B**:eval 加固后,若调旋钮比精修 reward 代码更易拿分,LLM 会走最省力路径去调旋钮 → autoresearch 退化成 **curriculum 超参搜索**,掩盖"reward 代码发现其实没在工作"。

**保障(部分已在计划,3/4 为新增,必须落实)**:
1. **reward 代码优先**:contract `primary_mechanism_required` 已声明"须靠 reward terms + reward_weight_scale 改善";保留为硬设计约束,prompt/checklist 持续强调旋钮是**次要 curriculum 辅助**,不是主机制。
2. **clamp + 目标难度地板**(已在 §3.2):旋钮拿不到大分,策略仍须靠 reward 代码学会恢复才能过硬化 eval。
3. **eval 加固依赖**(§1):OOD + 连续分让"旋钮单独赢不了",reward 代码与课程都得贡献 —— 这是防 B 的结构性保障。
4. **【新增】改进归因(必做)**:每个 accepted round 在 ledger/diagnostics 记录"增益可归因于 **reward-code-diff** 还是 **schedule/knob-diff**",并显式标记:`无 reward term`、`reward 代码相对 inherited best 未变`、`增益仅可归因于课程`。让退化**可被检测** —— 否则 loop 悄悄变旋钮搜索而不自知。
5. **【新增】纯旋钮候选 surface**:对"零 reward term / 仅改 schedule"的候选,在公开 diagnostics 里显式标出(不一定拒,但必须可见),用于监控 autoresearch 是否真在做 code-level 工作。可选更强约束:接受时要求"至少有一个非平凡 reward term 或 reward 代码相对 best 有实质 diff"。

**结论**:Proposal 2 直接对齐目标、放心做;Proposal 1 在**保留 reward 代码优先 + clamp/地板 + 依赖 eval 加固 + 加改进归因/纯旋钮 surface** 的前提下不会偏离 —— 旋钮停留在"次要 curriculum 辅助",loop 的进步仍可归因于 code-level reward 发现。**缺了第 4、5 条保障,Proposal 1 就有真实偏离风险。**

---

## 8. 不在本计划范围

- 评测加固本身(多种子/OOD/连续分)—— 另立计划,但为本计划前置依赖。
- LossTerm 接进 PPO optimizer、其它新 RewardInputs 字段 —— 不在本次开放范围。
