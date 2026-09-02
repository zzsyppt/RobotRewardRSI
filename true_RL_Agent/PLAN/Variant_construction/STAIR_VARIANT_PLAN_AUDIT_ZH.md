# 机器人 × 任务矩阵计划 —— 第二轮独立审计

> - 文档状态：审计意见，未修改被审计计划
> - 审计日期：2026-08-25
> - 被审计对象：`PLAN/Variant_construction/STAIR_VARIANT_CONSTRUCTION_PLAN_ZH.md`（733 行，2026-08-24 改版）
> - 本文**替代** 2026-08-24 那版审计（那版针对已被替换的 Stair variant family 草案）
> - 代码基线：`c582eca`，分支 `true-rl-agent-refactor`
> - 审计方式：逐条对照源码与冻结 config 复核；未运行 Isaac/GPU

---

## 0. 结论

**改版是一次真正的方向性修正，架构判断正确，可以按它施工。**
目标从"造一个楼梯基准"改成"造一个机器人 × 任务矩阵，每格挂 LLM LOOP"，
这与系统实际结构、与 Isaac Lab 自己的 Task/Robot 组织方式都对得上。

本轮审计发现 **1 个必须改的内部矛盾、1 个已被代码证伪的验收条款、3 个高优先级缺口**，
以及 **4 处可以删掉或降级的验证复杂度**。都是局部修补，不需要推翻结构。

一个必须说清的前提（见 §2.3）：**矩阵能把软件成本降到 O(行+列)，但降不了实验成本 O(格)。**
这不是设计缺陷，是物理事实，计划里应当写明，否则会对 3×3 矩阵产生错误预期。

---

## 1. 你的需求本身合理吗

先回答这个问题，因为后面所有判断都建立在它上面。

### 1.1 需求复述

> 机器人和任务是两根独立轴；加一行或加一列只需一小块适配层；每格都能挂 LLM LOOP 在自己的复杂
> 环境上持续改进策略；**LLM 的改进权限可分级、可扩展（现在只有奖励函数，未来可能有课程等），
> 但训练、评测、接受这些语义必须掌握在我们手里。**

### 1.2 判断：合理，而且是这个系统的正确长期形态

三条理由：

1. **和上游的组织方式一致。** Isaac Lab 自己就是 `locomotion/velocity/config/{g1,go2,h1,anymal}/`——
   共享任务层 + 每个机器人一份薄配置。你要的正是这个结构，只是把它提升到 TaskPlugin 之上。
   这不是发明抽象，是对齐已被验证的工程模式。
2. **权限分级是科学控制的必要条件，不是额外治理。** 自进化实验的可信度完全取决于"LLM 只能动
   X，其余全部冻结"。如果评测或预算也能被候选影响，任何"改进"都不可归因。而且系统里**已经有
   这根轴**：`CandidateCapabilityProfile`（当前 Stair 为 `["complete_reward_program"]`，Push 为
   `diagnostic_term / reward_term / reward_weight_scale_schedule`）+ `CandidateSurfacePolicy` +
   verifier。所以"未来开课程"是往已有枚举里加一项，不是新建系统。
3. **系统主干已经是任务中立的。** `run_live_loop` / `round.py` 不解释任务或机器人，composition
   registry 按 `task_id` 解析，ledger/current-best/memory 全部按 task 身份分家。矩阵要做的事发生在
   **composition 解析阶段**，AgentCore 一行都不用改——计划 §2.1 的这个判断经复核成立。

### 1.3 需求里唯一需要加限定词的地方

"加一行或加一列只需一小块适配层"——**对软件成立，对实验不成立。**

| 成本类型 | 复杂度 | 说明 |
|---|---|---|
| RobotProfile / RuntimeAdapter / TaskPlugin / binding 代码 | **O(行 + 列)** | 矩阵抽象的收益在这里，真实且可观 |
| 每格的 reference reward（Genesis 锚点） | **O(格)** | 训练路径强制要求 qualified program（`isaaclab_execution.py:681`），而官方奖励是**按机器人分别编写**的 |
| 每格的难度标定与评测套件资格化 | **O(格)** | 每个机器人的可行几何范围不同 |
| 每格的 LOOP 实验 GPU | **O(格)** | 各自独立谱系，不能共享 |

计划 §5.3 的"接入工作量预算"只列了第一行。建议补上后三行，并明确写：
**矩阵降低的是软件与协议成本，不降低实验成本；扩张矩阵前必须先批预算。**

---

## 2. 相比上一版，已经改对的地方

复核后确认这些都成立，不必再改：

1. **机器人轴被抽出来，且要求被两个任务共同验证**——这是最关键的一改。
   现实依据（我复核过）：`push_recovery` 和 `stair_climbing` **各自独立硬编码了同一组 G1 常数**：
   `torso_link`（两处）、摔倒高度 `0.45`（两处）、姿态阈值 `0.65`（两处）。
   2 任务 × 1 机器人已经抄了两遍，不抽出来必然继续复制。
2. **`loop_verified` 与 `improvement_observed` 分开**——最好的一处新增。
   它把"工程闭环"和"科研结果"分成两个不会互相冒充的状态。
3. **`TrainingBindingV2` 只加 `task_protocol_hash`**（上一版建议 C2）——采纳。
   复核确认 `TrainingBinding` v1 确实无任务身份（`training_binding.py:21-31`），
   而 `EvaluationBindingV1` 已有 `task_protocol_hash`（`evaluation/binding.py:53-58`），
   两者现在用同一个身份载体，没有引入第二套 variant binding。
4. **C1 不等高/不等深 排在 C2 roughness 之前**（上一版建议 I2）——部分采纳，见 §4.1。
5. **评测成本口径改成 `eval_num_envs ≤ 512`**（上一版 B2）——采纳，这才是真实成本变量。
6. **跨机器人不产生 winner**（上一版 C1）——采纳，`common anchors 只作描述性表格`。
   上一版建议删掉的那整套跨机器人统计比较机制，确实已经不见了。
7. **难度标定用 calibration seeds、在任何 LOOP 前冻结 easy/target/hard**（上一版 B3）——采纳。
8. **子进程必须重载 manifest 并重算 hash、summary 回传 runtime-derived identity**——
   这条防的是"父进程贴标签"，是本计划里质量最高的一条工程验收。
9. **1 / 3 / 15 代枚举**——好消息：**引擎里已经有了**，不是新建。
   `agent_core/generation_contract.py:45-51` 定义了
   `development_generation_pilot / development_generation_validation / formal_generation_session`，
   `generation_session.py:379-388` 的 `max_generations` 映射为 `1 / 3 / 15`。
   缺的只是**生产 CLI 与 launcher/readiness 没有暴露非 formal 模式**
   （`run_stair_generation_session.py` 只有 `--formal-session`；readiness 全是
   `STAIR_FORMAL_LAUNCH_SUBJECT_*`）。计划 Phase 4 的描述准确，且实际风险比字面看起来低。

---

## 3. 必须改的问题

### M1【内部矛盾】Core V1 的 GO 门里塞进了一个不可控的科研结果

**§9.1 最终 GO 要求**：

> G1/Go2 complex Stair 都至少出现一次由 LLM reward 候选导致、经完整 acceptance 接受的 current-best 更新

**§9.2 又说**：

> 3-generation LOOP 闭合但无 accepted improvement：记录 `loop_verified=true, improvement_observed=false`，
> 审查 reward ABI、反馈质量和难度标定，不宣称自进化成功

这两条互相打架：§9.2 把"没有改进"当成一种**正常可记录的结果**，§9.1 却把它当成 **Core V1 失败**。

而且这个门在统计上很难过。3-generation × width=2 = **每个机器人只有 6 个候选**，
要在其中至少有一个跑赢官方奖励基线。历史证据不支持这个假设：eval-v5 那次正式会话里
R1 的 generalization 是 `0.246` 对 Genesis 的 `0.590`，anchor guard `0.132` 对 `0.636`——
被决定性拒绝。多数候选被拒是这个系统的常态，不是异常。

**建议**：
- Core V1 的 GO 门 = **`loop_verified`（工程闭环闭合）**；
- `improvement_observed` 降为**必须如实报告的结果项**，不作门；
- 若两个 cell 都是 `false`，触发 §9.2 已经写好的诊断分支（难度标定偏难？公开反馈信息量不足？
  reward ABI 表达力不够？），并把"是否追加 generation"变成**需要单独授权的决定**，
  而不是为了过门而偷偷加代数——后者恰恰是计划 Phase 6 验收 6 明令禁止的
  （`不得结果驱动增加 generation 数`）。

把门改成 `loop_verified`，§9.1 和 §9.2 就一致了，同时也和 Phase 6 的禁令一致。

### M2【已被代码证伪】"Stair 与 Push 报告完全相同的 G1 RobotProfile hash"做不到

**Phase 1 严格验收 2** 要求：

> Stair 与 Push composition 报告**完全相同的 G1 RobotProfile hash**

**事实**：这两个任务用的根本不是同一个 G1 关节体。

| | Stair | Push Recovery |
|---|---|---|
| 关节数 | **37**（`stair_protocol_v1.json: policy_compatibility.robot_schema.joint_names`，动作 `dim: 37`，含双手手指） | **29**（`push_recovery/isaac_config.py:15-17`：`Unitree-G1-29dof-Velocity`） |
| 来源包 | Isaac Lab `G1RoughEnvCfg`（G1_MINIMAL_CFG） | 厂商包 `unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg` |

**37 DoF 与 29 DoF 是两个不同的机器人模型，还来自两个不同的上游包。**
强行让它们产出同一个 profile hash，只有两条路：要么把 profile 掏空到只剩"它叫 G1"（抽象变成装饰），
要么把其中一个任务改成用另一个的资产（超出本计划范围，且会破坏 Push 的历史 replay）。

**建议**：把这条验收改成**可达成且仍然有说服力**的版本：

1. `robots/` 注册表里是 `g1_29dof` 与 `g1_37dof` **两个 profile 条目**，
   共享同一个 **profile schema、semantic role 词汇表和尺度事实来源**（同一个 G1 家族的髋高、腿长等）；
2. 两个 TaskPlugin **都通过 role 访问机器人**，源码中不出现 USD 路径、关节名、link 名——
   这条用一个否定测试就能证明，比 hash 相等更有力；
3. 真正的"同一份 profile 被两个任务复用"，用 **Go2** 来证明更干净：
   Go2 在 Stair 和（Extension A）Push 上用的是同一个关节体，不存在 29/37 分裂。

也就是说：**Phase 1 证明"机器人轴存在且任务不碰裸名字"，把"同一 profile 跨任务复用"的
强证明挪到 Extension A（Go2 × Push）**。这样 Core V1 的验收不会卡在一个物理上不成立的条件上。

顺带提醒：Push 走的是**厂商包 `unitree_rl_lab`**，不在 Isaac Lab 内。
`RobotRuntimeAdapter` 因此要跨两个上游包，这是 Phase 1 里被低估的一块工作量。

### M3【缺口】acceptance 的绝对地板没有进 manifest，可能让 complex cell 的 Genesis 直接开不起来

计划 §3.3 的 manifest 冻结清单里有 evaluation profile hash，**但没有 acceptance protocol hash**。

**为什么这是硬问题**：`agent_core/decision.py:214-230` 在 bootstrap（尚无 current-best）时要求
`objective ≥ absolute_accept_floor` **且** `guard ≥ guard_absolute_accept_floor`，
当前两者都是 `0.2`（`stair_acceptance_protocol_v5.json`）。
这些数字是按 **G1 × clean 35-cell** 的分数尺度定的（当时 Genesis 是 0.590 / 0.636）。

complex cell 的 Genesis 分数必然更低。如果落到 0.2 以下，
**reference program 自己无法成为 current-best，整个 LOOP session 起不来**——
而且失败会发生在你已经付完一次完整训练 + phased eval 的钱之后。

同理，`reference_capability_gate` 的门（`low_step_bin_ids: stair_010/015/020`、
成功率 ≥0.05、进度 ≥0.5 m）也是 G1 绝对几何写死的，Go2 的缩放几何下这三个 bin id 根本不存在。

**建议**（改动很小）：

1. `acceptance_protocol_hash` 与 `reference_capability_gate` 加入 §3.3 的 manifest 冻结清单；
2. 两者的阈值在 **Phase 3 难度标定阶段**、用 calibration seeds 和固定 reference 一并重新冻结，
   规则事前声明（例如 `absolute_accept_floor = round(0.5 × 标定得到的 reference objective, 2)`）；
3. 加一条 Phase 3 验收：**标定结果必须证明该 cell 的 reference 能通过自己的 bootstrap 门**，
   否则该 profile NO-GO，回去改 terrain，而不是等到 Phase 4 烧掉一次 Genesis 才发现。

---

## 4. 高优先级改进

### I1. C2 roughness 进 Core V1 会把"判据失真"问题带回来，而计划只解决了 C0 那一半

Phase 3 验收 1 保证 **C0 的 35-cell 结果不变**——很好，但那是对照组。
真正的风险在 C1/C2：几条 outcome 判据的容差和 roughness 幅值是同一个量级
（`stair_evaluation_semantics_v6.json`）：

| 判据 | 阈值 | 与 C2 的冲突 |
|---|---|---|
| `insufficient_clearance` | 非支撑足距投影踏面 `< 0.03 m` 连续 3 步 | roughness 峰值若达 3 cm，摆动足掠过凸起即触发**伪失败** |
| `fall` | `maximum_projected_surface_penetration_m = 0.03` | `terrain_evidence.py` 靠它逐级回退判定支撑层级，rough 下会**判错支撑台阶** |
| `slip` | 接触中平面速度 `> 0.5 m/s` | 粗糙面微滑移增加，失败归因漂移 |

§3.2 把"少量形态阈值"放进 RobotTaskBinding——这解决了**机器人尺度**，
但没解决**地形幅值**。结果是 complex cell 的失败分类与 clean 不可比，
而失败分类正是喂给 LLM 的主要反馈（prompt v7 的 failure grids）。

**建议**：在 Phase 3 增加两条验收：

1. **量级约束**：`roughness 最大局部偏差 ≤ 0.5 × insufficient_clearance 阈值`，
   或者反过来，把这些容差声明为 `f(environment_profile)` 的派生量并写进 manifest；
2. **分类漂移对照**：同一个固定 reference policy，在 C1 与 C1+C2 上跑同一批 cells，
   报告 primary outcome 分布的变化，并要求 `insufficient_clearance` / `slip` 的占比变化
   在事前声明的范围内。超出即说明测的是判据而不是能力。

### I2. 把 Phase 3 拆成 3a / 3b，让第一个闭环早两周出现

Phase 3 是全计划最大的一块（10–14 人日），同时装了：C1 生成器 + C2 roughness +
runtime surface evidence + 五套 seed split + covering array + 难度标定 + 预算冻结。
而 Phase 4（G1 第一个真实 LOOP）**完全依赖它整块完成**。

C1 和 C2 的工程性质差别很大：

| | C1 不等高/不等深 | C2 踏面 roughness |
|---|---|---|
| 踏面 | 仍是平的 | 起伏 |
| decoder | `level*h` → 累积高度表，**仍然精确** | 必须换成采样式 local surface query + 容差 |
| 判据阈值 | **不受影响** | 需要 I1 那一整套 |
| mesh 规模 / 吞吐 | 基本不变 | 显著上升，需实测 `steps/s` 与 `runner_timeout_s=7200` 的余量 |

**建议**：

- **3a**（约 5–6 人日）= C1 + 精确 decoder + seed split + 难度标定 + 预算冻结
  → **Phase 4 可以在这里就启动**，用 `stair.g1.complex_v1 = C1` 拿到第一条真实闭环证据；
- **3b**（约 5–7 人日）= C2 roughness + local surface query + I1 的两条验收
  → 完成后把 complex profile 升级为 C1+C2，重跑一次标定。

收益：**第一个 `loop_verified` 提前约 2 周**；且如果 C2 在吞吐或判据上出问题，
你手里已经有一个能用的复杂变体，不会连闭环证据都没有。

代价：`stair.g1.complex_v1` 会有两次冻结（C1 版、C1+C2 版）。
这在计划的身份体系里是干净的——它们是两个 `cell_variant_id`，本来就该有不同 manifest hash。

### I3. Go2 的 reference reward 是全计划最大的单点风险，而计划选了较贵的路线

§3.4 给 Go2 reference 的两条路是：**task-owned 人工基础 reward + 固定预算 qualification**，
或 **reward-cold-start/bootstrap**。两条都是开放式搜索，每次尝试 ≈ 1 GPU 小时，
而且要通过 `reference_capability_gate`（成功率 ≥0.05、进度 ≥0.5 m，5 seeds）。
尝试次数不可预测，这是整个 Core V1 里方差最大的一项。

**计划漏掉了最便宜、且与 G1 完全同构的第三条路**：
**移植 Isaac Lab v2.3.0 官方 Go2 rough 奖励**。

依据：G1 那份 `official_g1_rough_v1` 就是这么来的——
它对官方 15 个 reward term 做 1e-6 等价复现（`complete_reward_runtime.reference_equivalence`
列出了 `feet_air_time / feet_slide / dof_pos_limits / joint_deviation_* / ...`），
而且这份"只是官方 locomotion 奖励"的参考程序在 35-cell 上拿到了 **0.590**。
Go2 在 v2.3.0 里有结构完全对应的 `go2/rough_env_cfg.py`（不同 term 集合：
`undesired_contacts`、`feet_air_time` 等），移植路径、等价 oracle、资格化流程可以整套复用。

**建议**：把"官方 Go2 rough 奖励移植 + 等价性验证"设为 **Phase 5 的首选路线**，
人工 reward / cold-start 作为它不过门时的兜底。
这条同时也解释了为什么 §3.4 那句"新机器人不能复制 G1 reference reward"是对的
但表述不完整——不能复制的是 **G1 的那份程序**，
可以（而且应该）复制的是**"移植该机器人官方奖励"这个方法**。

### I4. Core V1 的 GPU 预算应该在 Phase 0 给出量级，而不是等到 Phase 3 才冻结

计划把预算冻结放在 Phase 3 末尾（治理上合理），但**没有任何量级估算**，
而这个量级决定 Core V1 是 6 周还是 4 个月的项目。

按已知基线（4096×24×2000 = 196.6M 步 ≈ 62 分钟训练 + 14 分钟评测，width=2 双卡并行）粗算：

| 项目 | 训练次数 | 说明 |
|---|---:|---|
| G1 complex Genesis | 1 | |
| G1 1-gen pilot | 2 | width=2 |
| G1 fresh 3-gen validation | 6 | |
| Go2 reference 资格化 | 1–4 | I3 的风险项；走官方移植路线取下界 |
| Go2 clean qualification | 1–3 | |
| Go2 complex Genesis | 1 | |
| Go2 1-gen pilot + 3-gen validation | 8 | |
| 难度标定 / holdout / parity | 0 训练，若干次评测 | 复用已有 checkpoint |
| **合计** | **20–25 次训练** | 一次全对的理想情况 |

≈ **25–35 GPU-小时纯计算**；双卡并行下墙钟约 **20–30 小时**；
考虑失败重跑、排队和 provider 往返，现实预期 **2–3 倍**。
再加上 47–67 人日工程（P0–P7 累加）≈ **10–14 周单人串行**。

**建议**：在 Phase 0 交付物里加一张这样的量级表（不需要精确，需要有量级），
并同时给出**两档 scope dial** 供决策（见 §6）。

---

## 5. 可以砍掉或降级的验证复杂度

### 5.1 covering-array 生成器 → 手工设计 + 20 行 pairwise 检查器

§4.3 要求"由 covering-array generator 给出最小 cells，并证明所有可行二元组合 100% coverage"。
对 **≤18 个 cells、约 4 因素 × 3 水平**的规模，
手工设计一组 cells 再用一个小脚本枚举验证 pairwise 覆盖，结论完全等价。
**保留因素模型和覆盖率证明，删掉生成器这件工程品**（少一个要维护和测试的组件）。

### 5.2 `improvement_observed` 从 GO 门降为报告项

见 M1。这一改同时消除了"为了过门而加代数"的制度性压力。

### 5.3 Phase 1 的 "相同 profile hash" 降为 "相同 role 词汇表 + 否定测试"

见 M2。**否定测试（TaskPlugin 源码中不得出现 USD 路径 / 关节名 / link 名）比 hash 相等更强**，
因为它直接约束了将来所有新任务的写法，而 hash 相等只是一个一次性事实。

### 5.4 Phase 7 的 formal 预注册模板 → 一页纸，不做实现

§7 Phase 7 交付里有"formal 15-generation / 多 session 研究的预注册模板"。
在 formal 真正获批之前，这份模板只需要是一页 Markdown（样本量、paired effect、区间、通过标准的占位），
不需要 schema、校验器或代码。等真要做 formal 时再实现。

### 5.5 明确**不要**砍的四项

为避免误读，这几条虽然看起来"重"，但都是本计划的承重墙，建议原样保留：

1. **子进程重载 manifest + summary 回传 runtime-derived identity**——防"父进程贴标签"，
   是"roughness/Go2 真的生效"的唯一硬证据；
2. **cross-cell negative tests**（交换 Genesis / reference / checkpoint / ledger 必须被拒）——
   谱系隔离靠这个，成本很低；
3. **`unseen_final_holdout` 一次性使用**——泛化结论的唯一合法来源；
4. **难度标定在 LOOP 前冻结、只用 calibration seeds**——没有它，后面所有比较都不可信。

### 5.6 一个关于 holdout 的小修正

§4.3 规定 holdout ≤4 cells / 128 envs，且"只运行一次，运行后不得继续 authoring"。
配合 §9.2 的"返回上一 Phase"，会出现一个耗尽问题：
每次返工都要新建一份未使用的 holdout，而 4 个 cell 的池子撑不了几次。

**建议**：把 holdout 定义为一个**较大的保留 seed/几何池 + 每次终局评测抽取不相交子集**的规则，
而不是固定的 4 个 cell。这样重复实验不会把"未见"资源用光，规则本身不变。

---

## 6. 建议的修订顺序与两档 scope

保持计划的 Phase 骨架，只做三处调整（3 拆分、4 提前、Phase 1 验收改写）：

| Phase | 内容 | 人日 | 变化 |
|---|---|---:|---|
| 0 | 基线冻结 + **GPU 量级表** + Go2 资产只读确认 | 2 | 加量级表 |
| 1 | 矩阵契约 + 共享机器人轴（验收按 M2 改写） | 5–7 | 验收改写 |
| 2 | Stair 任务语义与形态解耦 | 5–7 | 不变 |
| **3a** | **C1 不等高/不等深 + 精确 decoder + seed split + 难度标定 + 预算冻结**（含 M3 的地板重冻结） | 5–6 | **拆分** |
| 4 | **G1 × C1-complex 接入 generation LOOP**（1-gen pilot → fresh 3-gen） | 8–12 | **提前** |
| **3b** | **C2 roughness + local surface query + I1 的两条验收** | 5–7 | **拆分后置** |
| 5 | Go2 机器人行 + Go2 × Stair（reference 走**官方移植**优先） | 7–10 | I3 |
| 6 | Go2 × complex LOOP + 跨单元格隔离 | 6–9 | 不变 |
| 7 | 一次性 holdout + 矩阵收口 | 4–6 | 5.4 降级 |

**两档 scope（建议在 Phase 0 就选定）**：

| | 完整档 | 收敛档 |
|---|---|---|
| G1 complex | 1-gen pilot + 3-gen validation | 同左 |
| Go2 complex | 1-gen pilot + 3-gen validation | **只做 1-gen pilot**，3-gen 移入 Extension |
| 训练次数 | 20–25 | **14–17** |
| 证明了什么 | 两个机器人都完成多代闭环 | 机器人轴能走到 LOOP；多代闭环由 G1 证明 |
| 省下 | — | 约 6 次训练 + 1–2 周 |

收敛档的逻辑是：**"多代闭环可行"由 G1 证明一次就够，Go2 需要证明的是"新机器人行能接进同一个
LOOP"，这一点 1-gen pilot 已经完全覆盖。** 如果 GPU 或时间紧，这是最划算的一刀。

---

## 7. 需要拍板的四件事

1. **M1**：Core V1 的 GO 门改成 `loop_verified`，`improvement_observed` 只报告不设门？
   （审计意见：必须改，否则 §9.1 与 §9.2 自相矛盾，且门本身不可控）
2. **M2**：Phase 1 验收改成"共享 role 词汇表 + 否定测试"，把"同一 profile 跨任务复用"的强证明
   挪到 Go2 × Push（Extension A）？（审计意见：必须改，37 vs 29 DoF 是硬事实）
3. **I2**：接受 Phase 3 拆成 3a/3b、把 G1 的第一个真实闭环提前到 C1 阶段？
   （审计意见：强烈建议，提前约 2 周且显著降风险）
4. **scope 档位**：完整档还是收敛档？

前两条是纠错，后两条是取舍。你定完这四条，我可以直接把计划正文改到位。

---

## 8. 本轮复核的证据索引

| 结论 | 证据位置 |
|---|---|
| Push 与 Stair 各自硬编码同一组 G1 常数 | `push_recovery/isaac_config.py`（`torso_link` / `0.45` / `0.65`）对 `stair_evaluation_semantics_v6.json` |
| Push = G1 29-DoF / 厂商包；Stair = G1 37-DoF / Isaac Lab | `push_recovery/isaac_config.py:15-17`；`stair_protocol_v1.json: policy_compatibility.action.terms[0].dim = 37` |
| Push composition 硬编码机器人 | `push_recovery/composition.py:32` (`PUSH_RECOVERY_ROBOT = "unitree_g1"`) |
| 1/3/15 代模式引擎侧已存在 | `agent_core/generation_contract.py:45-51`；`generation_session.py:379-388` |
| 生产 CLI 只暴露 formal | `scripts/run_stair_generation_session.py`（mode group 仅 `--formal-session` + 两个 helper） |
| live composition 强制 15 rounds | `tasks/stair_climbing/composition.py:299-301` |
| bootstrap 绝对地板 0.2 | `agent_core/decision.py:214-230`；`configs/tasks/stair_acceptance_protocol_v5.json` |
| reference capability gate 绑 G1 绝对 bin | `stair_protocol_v2.json: reference_capability_gate`（`stair_010/015/020`, 0.05, 0.5 m） |
| 训练强制单一 complete_reward 项 | `tasks/stair_climbing/isaaclab_execution.py:681` |
| G1 官方奖励参考程序与等价口径 | `stair_protocol_v2.json: complete_reward_runtime.reference_equivalence`（15 terms, 1e-6） |
| 判据容差与 roughness 同量级 | `configs/tasks/stair_evaluation_semantics_v6.json` |
| 候选权限轴已存在 | `TaskSpec.candidate_capability_profile`；Stair = `["complete_reward_program"]` |
| LOOP 主干任务中立 | `agent_core/live_loop.py`、`agent_core/round.py`、`composition/defaults.py` |
| 历史候选被决定性拒绝的量级 | `outputs/formal_15round_b445125_evalv5_followups.md`（R1: 0.246 vs Genesis 0.590） |
