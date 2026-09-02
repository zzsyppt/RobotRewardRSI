# P0 基线闭合与协议冻结报告

> 基线审计：2026-08-15；verifier 根因复审：2026-08-16
>
> 结论：P0 PASS，等待用户验收；P1–P6 未实施，禁止启动新实验

## 1. 权威身份

| 对象 | 结果 |
|---|---|
| 本地分支 | `true-rl-agent-refactor` |
| 基础 commit | `e8b1e9e80f4b2a12af993b691cb8a76692cc3dfd`；干净提交测试基线为 `1150/20` |
| origin HEAD | 与本地一致 |
| 服务器 checkout | `/data/yuyangliu/AGENT`，与本地 HEAD 一致且 tracked worktree clean |
| 正式 session | `stair-reference-formal-20260813T125345Z` |
| 正式 output | `/data/yuyangliu/true_rl_agent_phase_g/fresh_formal_15round_20260813T125345Z` |
| plan hash | `ce5583fd35ac715987df315719dda3e06e95d9d080ff11b0d3fa6bbd7ebc0480` |
| Session Genesis | v3，`744cc76d31704ebf2f5571c9bc8a7261242471d42a604fd86b3b05c0e9a2059b` |
| backend environment manifest | `3e314c3aa47a414cb84bafcb55d7bafe7bf64f3cb7f775933600dfb05cb6aeda`；Python 3.11.15、torch 2.7+cu128、Isaac Sim 5.1、rsl-rl 3.0.1 |
| ledger epoch | `571169d92e7fbabfb6e3f9f6440096617c066fc9c3d62e0cd931d1e2bfb138ca` |

本地工作树在 P0 开始前已有用户修改：

- `scripts/run_stair_final_confirmation.py`
- `tests/unit/test_stair_final_confirmation_entrypoint.py`

P0 未修改、还原、stage 或同步这两处内容；也未移动或删除任何 output。P0 入场工作树（基础 commit + 上述用户修复）的测试基线为 `1153/20`。随后按用户复审证据，P0 只在 verifier 历史判重根源、历史 anchor 投影及其测试上做了纠偏；因此当前验收对象是“基础 commit + 用户 final-confirmation 修复 + P0 verifier 根修”，不再把未提交工作树的测试数冒充 `e8b1e9e` 本身。相关文件 SHA256 与最终测试数记录在 `DEV_LOG.md`；在形成用户批准的 commit 前不得称为不可变 release 基线。

## 2. 正式单卡 session 闭合结果

### 2.1 Lifecycle 与服务器状态

| 检查 | 结果 |
|---|---|
| ledger | 15 条 `true_rl_agent.ledger_record.v7`，全部闭合 |
| train/eval | 15 个 train summary pass；15 个完整 eval summary pass |
| live invocation | 恰好 1 个；`completed / normal / exit_code=0` |
| live summary | v6；15/15；10 个 completion checks 全为 true |
| invocation 时间 | 2026-08-13 14:15:06Z → 2026-08-14 09:45:52Z |
| 总墙钟 | `70,246.431 s`，约 `19.513 h` |
| 当前 tmux/目标进程 | 无 |
| 当前 GPU | 两卡均 0% SM；审计时无目标计算 PID |
| output 大小 | 约 `5.0 GiB` |

ledger 文件：

- bytes：`1,911,425`
- SHA256：`1d49e22d0530b40332ced06e1f562d57855ba9ded927d258c32850b5634a95e2`

没有 partial round 被计入 ledger，也没有基础设施失败被伪装成科学拒绝。

### 2.2 科学结果

| 项目 | 结果 |
|---|---|
| accepted | R2、R3，共 2 条 |
| rejected | 13 条 |
| infrastructure failure | 0 条 |
| 最终 current-best | R3 `stair_progressive_clearance_v1` |
| objective | `0.9492857142857144` |
| guard | `0.9821428571428571` |
| sealed | `0.9408163265306122` |
| checkpoint | `sha256:dad4980344bcaf1552cba834cd7192de4400895ea3896a6b141a5cbfe2032d3c` |

authoring 共 17 次 provider attempt；R9、R10 各有一次 `response_truncated`，都在既有 attempt budget 内完成同轮 repair。它们是 authoring repair，不是 train/eval job retry。

### 2.3 Final confirmation 的精确状态

正式 live 已闭合，不等于 session-end final-confirmation 已完成。

- `final_confirmation/` 当前只有 create-once `plan.json`；没有 reservation/result；
- 终局配对诊断尚未执行；
- 服务器 `e8b1e9e` 入口会把公开 `artifact://.../best.pt` 当作本地 checkpoint 路径，形成确定性阻断；
- 本地工作树中的用户修复改为解析同一 record 内 checksum-bound 的 private `runner_checkpoint`，相关回归通过，但 P0 没有同步或运行它。

因此当前表述只能是：**15-round scientific live closed；final-confirmation pending**。用户验收、发布同步和服务器预检完成前，不得在原正式目录启动终局诊断。

## 3. 单卡成本基线

### 3.1 Authoring 与 provider

| 指标 | 结果 |
|---|---:|
| attempts | 17 |
| request units | 444,776 |
| cache-hit input | 195,456 |
| cache-miss input | 249,320 |
| response units | 213,995 |
| reasoning units | 184,570（response 的子集，不另加） |
| provider total units | 658,771 |
| selected prompt bytes | min 71,312；mean 96,070.867；median 101,915；max 104,092 |
| authoring evidence bytes | min 6,977；mean 7,571.933；max 8,967 |
| authoring wall proxy | min 107.986s；mean 208.297s；median 210.466s；max 496.073s |

authoring 墙钟来自首个 prompt 文件时间到 evidence 文件时间，只是可复核 proxy，不冒充 runner-owned duration。

### 3.2 Training

| 指标 | 结果 |
|---|---:|
| jobs | 15 |
| 每 job env-steps | 196,608,000 |
| 总 env-steps | 2,949,120,000 |
| 每 job wall proxy | min 3,530.382s；mean 3,579.803s；median 3,549.168s；max 3,763.772s |
| 15 jobs wall proxy 合计 | 53,697.043s |

训练墙钟以首个 checkpoint 到 summary 的持久时间计算，缺少 materialization/start 的少量时间，因此仍标为 proxy。

### 3.3 Evaluation

每个 round 为 35 cells × 8 episodes × `3/5/7` seeds，共 4,200 episodes。

| split | 每 job episodes | wall min | wall mean | wall median | wall max | 15 jobs 合计 |
|---|---:|---:|---:|---:|---:|---:|
| public-dev | 840 | 150.482s | 165.183s | 167.438s | 174.022s | 2,477.739s |
| public-accept | 1,400 | 256.436s | 270.591s | 272.476s | 285.408s | 4,058.869s |
| sealed | 1,960 | 343.214s | 378.167s | 382.726s | 398.175s | 5,672.503s |
| full eval | 4,200 | 750.431s | 813.941s | 823.437s | 857.605s | 12,209.111s |

15 个 candidate eval 合计 63,000 episodes。Genesis reference 另有 4,200 episodes；若将来合法执行当前终局配对诊断，还会增加 `2×7 seeds = 3,920 episodes`。

### 3.4 当前缺失的资源数据

现有持久 summary 没有权威 GPU 显存峰值、GPU 利用率分布或主机 RAM 峰值。P0 不使用过去人工采样值替代它们。

处理决定：

- 不为补该数字修改当前正式产物；
- P1 logical-job receipt 设计保留资源计量位置；
- P6-A resource shadow pilot 必须先产出单 lane/双 lane 的 GPU/PID/device、RAM、swap、CPU 与 IO 峰值；
- `64 GiB 可用 RAM` 仅是待 P6-A 验证的初始保留值，不是 P0 已测常量；在此之前，双卡内存/CPU 资源门不能宣称已冻结。

## 4. Provider 合同与金额草案

### 4.1 当前事实

| 字段 | 当前值 |
|---|---|
| endpoint | `https://api.deepseek.com` |
| model | `deepseek-v4-pro` |
| temperature | `0.2` |
| reasoning | `adaptive` |
| output cap `O` | 16,384 units |
| response envelope | 524,288 bytes |
| transport | `true_rl_agent.openai_compatible_transport.https_or_loopback_http_no_redirect.v1` |

价格快照（2026-08-15）：DeepSeek V4 Pro cache-hit input `¥0.025/1M`、cache-miss input `¥3/1M`、output `¥6/1M`。来源：[DeepSeek 官方价格](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)、[限流说明](https://api-docs.deepseek.com/quick_start/rate_limit)、[思考模式](https://api-docs.deepseek.com/guides/thinking_mode)。

本次实际 authoring 费用：

```text
195,456 × 0.025 / 1,000,000 = ¥0.0048864
249,320 × 3     / 1,000,000 = ¥0.7479600
213,995 × 6     / 1,000,000 = ¥1.2839700
合计                              ¥2.0368164
```

reasoning units 已包含在 response units 内，未重复计价。

### 4.2 供 P1 实现和用户审批的简单上限草案

- `I = 300,000 request units / attempt`
- `O = 16,384 response units / attempt`
- 每 generation 最多 10 attempts
- 15 generations 最多 150 attempts
- 以全部 cache-miss 的保守价格计算：
  - 每 attempt 最坏 `¥0.998304`
  - `C_generation = ¥10`（向上取整）
  - `C_session = ¥150`（向上取整）

`I=300,000` 是宽松成本预留草案，不是当前代码已有硬门，也不是 P6 可以直接冻结的最终值。它是本次最大 28,114 request units 的约 10.7 倍；`C_session=¥150` 也明显高于本次 17 attempts 的 `¥2.0368164` 实付，只适合作为开发期不漏算的外层保守值。P1 必须先实现 provider-compatible 发送前计量，再用 15 轮实测分布与最大 contract-valid prompt 形状重新校准 `I/C_generation/C_session`，最后由用户批准 P6 的冻结值。

## 5. 历史兼容与 golden 基线

| 能力/文件 | records | bytes | SHA256 |
|---|---:|---:|---|
| v6 `outputs/current_38818b5_ledger.jsonl` | 1 | 114,420 | `b8ef9b47d19c5350fe16a3e390e3d977e644dcd847b78584111b6c09dcc265ea` |
| v5 `outputs/formal_15round_821c961_evalv5_ledger.jsonl` | 5 | 290,787 | `2a1fd7878678bcbc9069add66683b05fc00a998e37791b1f927c3f82318e3250` |
| v5 15-round prompt-capacity `outputs/prompt_capacity_post15/frozen_r15/ledger.jsonl` | 15 | 703,964 | `baed20ed40beb3c5bbfb1eadd39e0a43b86911e458b0c662b7a5bf8f7126942f` |
| current v7 server ledger | 15 | 1,911,425 | `1d49e22d0530b40332ced06e1f562d57855ba9ded927d258c32850b5634a95e2` |

现有 synthetic/golden tests 继续覆盖 v2/v4/v5/v6/v7，v3 继续 fail closed。P0 不复制这些 ledger 到新目录，不建立新 replay 实现，只冻结路径和 hash。

## 6. Stair-owned 探索价值草案

该策略将来必须实现于 Stair task-owned projector；公共层只可承载通用枚举、绑定和哈希。

### 6.1 可比较 context

只有以下身份相同或由协议声明可比较时，才能比较 trial：

- task/evaluation/acceptance protocol；
- common parent program/checkpoint；
- training initialization、seed schedule 和 backend environment；
- public-dev/public-accept cohorts；
- selector/value-policy version。

跨 parent 的旧结果只能作为重访假设，不是同一机制的全局因果分。

### 6.2 观察方向

| 值 | Stair-owned 含义 | 重访处理 |
|---|---|---|
| `strong_positive` | public evidence 完整；public gates 通过；public-accept objective 达到现有 task-owned 改善阈值且 guard 通过；dev 方向无实质矛盾 | 未选 sibling 进入 priority；accepted 记 incorporated |
| `partial_or_mixed` | dev/accept 方向冲突，或 objective 有改善但未达阈值，或 objective/guard 冲突且没有硬安全失败 | 保留 evidence，有限 defer/eligible |
| `negative` | public evidence 完整且出现 guard regression/公开硬门失败，或 objective 与 guard 均不优于共同 parent | 只抑制 exact implementation/context |
| `unknown` | 未训练、evidence 不完整、author/static/runtime/infra failure | 不作科学降权 |

不增加新的数值权重；materiality 和 guard 复用现有 Stair 比较合同。sealed 只允许投影 generic formal failure reason，不进入 observed value，不暴露 raw hidden evidence。

## 7. Repeat disposition 单一真相

### 7.1 权威来源

唯一判断者仍是 candidate verifier 的 `component_source_semantics`：

- 当前候选和历史候选分别相对**各自 authoring anchor**提取唯一变化组件；组件名与函数名不作为语义身份。数值归一化覆盖外层 `weight/clip/value` 等声明参数，以及 callable 中的普通算术、缩放和阈值幅值；tensor 下标/slice，以及 reduction/shape 调用的维度/轴配置保留为结构身份；
- 单组件 identical/parameter-only 等价阻断为 `historical_mechanism_repeat`；该规则可跨 parent 捕捉改名或改权重重复；
- rejected history 中输入依赖相同但函数/clip 结构不同的实现，保留非阻断 `historical_variant_assessment`；不再要求 same-parent；
- 每次 anti-degenerate evidence 无论是否命中 repeat，都记录 `history_sources_with_anchor`、`history_sources_without_anchor` 和 candidate anchor 是否可用；历史 prompt I/O 缺失不再静默伪装成“强比较已完成”。legacy full-source fallback 继续服务旧记录兼容，但不得作为未来 A/B sibling exclusion 的 distinct 证明；
- frontier/evidence card 只引用 sealed disposition，不再运行 AST、名称、rationale、embedding 或“机制家族”推断。

### 7.2 已证实的边界

本次正式 session 中唯一完成实物级复核的浪费是 R8：R1 为 `goal_progress_fraction`、weight `0.5`，R8 仅改成 `goal_progress_reward`、weight `0.25`，函数体与 clip 相同；R8 约消耗 59 分钟训练和 13.7 分钟 eval，共约 73 分钟，占 19.5 小时 session 的约 6.2%。生产函数复现实验同时证明两条独立漏口：

1. **anchor 漂移**：把 R8 与 R1 全组件源码直接比较时，中间 accepted head 带来的其他组件变化会被误判成 `compound_mechanism_delta`；
2. **重命名**：即使 anchor 完全相同且 R1 source 已明确提供，只改 term/function 名和权重也会绕过按名称 keyed 的整组件集比较。

因此旧报告“只是 supplied source 不完整”的根因判断错误；补更多 source 本身解决不了问题。根修已经收敛为同一个 verifier 原语：两侧各自先相对自己的 anchor 投影唯一变化组件，再比较不含名称的精确/参数归一化结构。真实 R1↔R8 在新原语下为 `parameter_only_delta`，必须 blocking。

2026-08-16 的复审又确认，最初实现曾把 callable 内所有数值常量统一归一化，导致 `projected_gravity[:,2]` 与 `[:,0]`、`sum(dim=1)` 与 `sum(dim=2)` 被误判为 parameter-only。该问题已在同一 AST 归一化源头修复：普通算术、`.mul(...)` 缩放和 `clamp(...)` 阈值仍归一化，tensor 索引/slice 及 reduction/shape 的维度配置不归一化；R1↔R8 仍由外层 weight 差异正确阻断。没有加入 Stair 字段名或张量轴含义表。

R11 不属于同一缺陷：R4 使用 `norm → sum → /100 → clamp[0,1]`、最大贡献 `-2.0`；R11 使用 `norm → squeeze`、clip `[0,50]`、最大贡献 `-0.5`，聚合结构与作用量级都不同。真实 source 在新原语下返回 distinct，并继续产生同输入、结构不同的 advisory；这是应保留的合法探索。R14/R15 未完成同等级 component-relative 对照，不再与 R8 并列宣称为重复。

## 8. Authoring 角色冻结

第一版 generation 冻结：

1. A-first、B-second；
2. A 完成 verify 后，由同一个 verifier 封存 canonical delta 与“相对共同 anchor 的单组件语义 exclusion”；
3. 从 A 动态流向 B 的唯一 provider-visible 信息，是 verifier-owned、public-safe 的 exclusion payload/hash；B 仅改名、改权重或给出 identical/parameter-only 等价实现时必须 blocking；同输入但 tensor 下标、轴、函数/clip 结构不同只作 advisory；
4. P4 必须调用唯一公开的 `compare_single_component_deltas`。该函数只有在两侧均为恰好一个可归因 delta 时才返回 equivalence 或 structurally-distinct；否则抛 `SingleComponentDeltaComparisonUnavailable`。B 必须把该异常作为可修复 authoring invalid，禁止当作 distinct 后训练；
5. B 不得看到 A source、rationale、provider response、private evidence ref/path、train/eval 结果；
6. A/B 各自最多 5 attempts，B exclusion repair 消耗 B 的既有预算，不新增第三条 lane；
7. A/B 任一未产生有效 candidate evidence，generation 零训练闭合，不降为 effective-width=1；
8. 两者齐备后才允许 `cuda:0/1` 并行训练；
9. temperature 固定 0.2、reasoning 固定 adaptive；只能在新 generation session 的 pre-Genesis contract 中改变，不能看见本 session 结果后调参。

在 P4 真实 exclusion 路径完成后、P6-B 使用任何训练预算前，新增一次 author-only paired-ready 探针：复用 15 份已保存的真实 prompt，每份注入该轮自身的 verifier-owned 单组件语义 exclusion并只请求一次 B；不使用 GPU、sealed 或 ledger。必须先由用户批准最多 15 次 API 和金额上限。验收要求等价漏放为 0，且至少 10/15 个首响应得到有效、结构不同候选；否则回到 P4 改 authoring/exclusion 表达，不用 temperature sweep 掩盖问题。本轮 P0 没有执行该 API 探针。

## 9. 从 Codex 借鉴的实现原则

只借鉴 `D:\humanoid_agent\codex` 中成熟的执行边界，不复制其产品架构：

- 长任务必须由 typed process handle 拥有，支持显式取消、终止确认和 wait/reap；
- 第一个失败是权威失败，清理失败只能附加记录，不能覆盖原错误；
- 外部副作用要有明确 reservation/result 状态，不能靠目录是否存在猜完成；
- 必须用 fault-injection/integration test 验证取消、异常和恢复，而不只测试 schema；
- 不引入 Codex server/UI/event bus，也不把其通用状态机变成本项目第二套 ledger。

这支持 P1 抽出小型 runner-owned process/attempt primitive，而不是在 `GenerationCoordinator` 外层给现有阻塞 round 套线程。

## 10. 已解决、仍存在与后续归属

### 已解决并应复用

- eval-v6 stable-top primary success；
- sealed 可重算证据；
- candidate/mechanism identity 单源；
- round-v7、Session Genesis v3、authoring evidence v4、live summary v6；
- authoring continuation/预算；
- backend execution-environment manifest；
- single-candidate正式 live 的生命周期审计。

### 真实存在，进入后续阶段

| 问题 | 证据 | 归属 |
|---|---|---|
| runner 仍是 blocking `subprocess.run` | 每 job 单次执行、无 typed job retry | P1 |
| train/eval job 无 infra retry | provider repair 存在，但 runner 零 retry | P1 |
| contract-valid 最大 prompt 终点仍可能 raise | 当前阶梯 fail closed，没有 guaranteed minimum core | P1 |
| GPU/RAM 峰值未持久化 | 本次正式 artifacts 无权威字段 | P1 receipt / P6 pilot |
| bounded memory 会遗忘旧负向证据 | R8 在 R1 离开 bounded window 后重复 | P5 evidence index；重复判断继续复用已根修的 verifier 单源 |
| B 的 paired-ready 率尚无生产路径实测 | 当前 generation/exclusion 尚未实现 | P4 完成后、P6-B 前运行 15-prompt author-only 探针 |
| generation 与双卡能力不存在 | current round仍是 singular writer | P2–P4 |
| 服务器终局 checkpoint path 缺口 | plan 已建、result 未跑；本地用户修复未同步 | 单独发布/验收，不混入 P0 |

### 不纳入本轮

- 新 Graph/DAG runtime；
- 全局机制价值分、UCB/MCTS、机制家族自动推断；
- 跨另一场完整 15-generation session 搬运 receipt/head/budget；
- 依据训练 reward 曲线接受、早停或合并 checkpoint。

## 11. 验收结果

| 门 | 结果 |
|---|---|
| 代码/服务器身份一致 | PASS |
| 正式 lifecycle/ledger/current-best 闭合 | PASS |
| final-confirmation 状态未夸大 | PASS，明确 pending |
| provider/训练/eval/磁盘成本基线 | PASS |
| GPU/RAM 峰值 | 未持久化，记录为非阻断测量缺口 |
| Stair value/repeat/authoring 边界 | PASS；R8 根因已纠偏并源修，R11 合法变体未被误杀；P4 继续复用同一比较原语 |
| 纠偏前 P0 定向基线 | `Ran 215 tests; OK (skipped=1)`；仅作历史记录，不冒充根修后的验收 |
| verifier/authoring 根修定向测试 | `Ran 82 tests; OK` |
| prompt/replay/Stair/Push 兼容集合 | `Ran 127 tests; OK (skipped=2)`；最终全量再次覆盖 |
| 干净 commit 全量测试 | `e8b1e9e: Ran 1150 tests; OK (skipped=20)` |
| P0 入场工作树全量测试 | `e8b1e9e + 用户 final-confirmation 修复: Ran 1153 tests; OK (skipped=20)` |
| 当前 P0 集成候选全量测试 | `Ran 1156 tests; OK (skipped=20)`；不得归因给未包含这些改动的基础 commit |
| compileall | PASS |
| `git diff --check` | PASS |
| GPU/API/服务器写入 | 0 |

P0 可以提交用户验收。未经用户明确批准，不进入 P1，不同步服务器，不执行 final-confirmation，不启动任何新实验。
