# Fresh 15-round 发布后观察与后验升级登记

本文只保留有实物证据、但不应在新一轮正式 15 rounds 开始前继续扩实现的观察项。每项必须满足“先观测、再触发、最后根修”；不得因为存在 follow-up 就改变当前 acceptance、seed、训练预算或任务权威。

## 上一轮正式性裁决

- R1–R9 的历史 v6 ledger 在当前严格重放中有效；R2 是该有效前缀唯一被接受的 current-best。
- R10 的 compact static-repair 只内联了同轮失败源码，旧实现因看不到 round 初始 current-best 而把 anti-degenerate 评为 `unknown`，却仍允许训练和接受；ledger 缺少 `authoring_mechanism_delta`（读取投影为 `None`）。这不是可用离线补证追认的缺字段，而是当时 pre-train authority 未被证明。
- 因此 R10 的 declared accept 无正式 authority，R11–R14 的 parent/current-best 链随之失效；这些轮次只保留为开发和行为证据，不改写旧 ledger。
- 下一次必须从新 Session Genesis、空 ledger、新 ledger v7/authoring evidence v4 开始完整 15 rounds。旧输出保持只读。

## 新 Genesis 前已从源头关闭的发布项

以下项目不再列为 pending；发布门必须证明实现、兼容和回放均通过。

### 每轮冻结 runner-private authoring anchor

首个真实 prompt 在 provider 调用前原子绑定 round anchor；后续 compact repair/source-followup 仍可只展示失败源，但 anti-degenerate 与最终 canonical mechanism delta 始终相对首帧权威 source。已有 prompt 而无 anchor、anchor/ref/hash/parent/lineage 漂移、重复 logical source 或 parent-bound `delta=null` 均 fail closed。纯 `interrupted_before_prompt` reservation 仍可由下一个真实 prompt 建立 anchor。ledger v7 要求 authoring evidence v4；历史 v6 只读重放并重新核验 selected-prompt authority。

### provider transport 有界、安全且可诊断

内置 urllib transport 只允许 HTTPS 或 loopback HTTP，拒绝 redirect、userinfo/query/fragment、异常 credential header，并把 timeout、HTTP、TLS、网络、JSON/UTF-8/response-envelope 边界投影成脱敏、定长的 `ProviderCandidateError`。明确 transient 失败只在既有 attempt 总预算内继续；明确配置/鉴权/TLS 证书验证失败立即终止；没有隐藏重试或第二预算。URL policy 只属于内置 transport，custom `JsonTransport` 仍可使用自己的 URI。receipt v2 绑定 transport policy 与 response byte cap。

### public-safe key/value 判定唯一化

prompt、eval projection、provider error 与 subprocess diagnostic 共用 task-neutral、边界感知的 credential/key 真相源：authoring/public payload 使用保守 predicate，不误伤普通 `bearer` 英文；provider、runner、Push runtime、subprocess 等不可信诊断使用同一递归 strict sanitizer。普通 `tokenizer_version`、`token_count`、`secretary_score`、`privateer_progress`、`Task-neutral`、`task-specific` 不再误伤；真实 credential/private-seed 键、标准 Authorization header/Bearer credential 与独立 `sk-*` token 仍阻断或脱敏。Stair/Push 特殊语义未进入共享规则。

### backend-owned execution environment identity

新正式 Isaac Genesis 绑定唯一 backend manifest：实际 `isaac_lab_python`、关键 Python/package 版本以及训练主路径的少量源码 checksum；train/eval 子进程在 simulator 启动前重算并回显同一 hash，runner 对 receipt 做 exact 验证。正式控制器必须由配置的 Isaac Python 运行；dry-run 与手工 materialization 不采样、也不伪造该环境身份。五个 hot-path source 使用有界、同一 FD identity 读取。Stair 环境与任务语义仍由 task-owned upstream source preflight 在 train/eval/final-confirmation 中重验，没有塞入 common backend manifest。硬件/driver 不进入权威身份，未做脆弱的全环境目录 hash。非 Isaac backend 可显式 `None`；Push 未来若做正式实验，其 vendor task source 应由 Push task-owned manifest 绑定，不塞入 common manifest。

### never-fed final confirmation 预承诺

新 Genesis 前由 Stair 插件 create-once 冻结独立 namespace、private cohort commitment、Genesis-vs-final 配对顺序、统计定义和 single-use 规则；只把安全 plan commitment 纳入 Stair task defaults，从而进入 Genesis identity。第 15 轮后才执行同 cohort 的 Genesis/final paired confirmation；结果为独立、不可反馈、无 acceptance authority 的 task-owned artifact，不写 ledger、prompt、memory 或普通 PublicEvalResult。计划与结果不持久化 raw seeds，并保存可重验的完整 35-cell canonical generalization summaries，而不是只留均值。

## 下一轮 15 rounds 逐轮观察项

### 1. tread horizon 同质分组的真实性能试验

证据：R2 三 split 合计固定上限为 2,982,000 simulated env control steps、841.85 秒；按 490/545/600/655/710 五个 horizon 理论可降至 2,520,000，少 15.49%。但直接把 280 env 拆为 5×56 会改变 vector RNG、env index、terrain assignment 和 GPU 数值路径。

- 本轮裁决：不改 formal evaluator。
- 观察：每轮 split wall time、fixed/closed-first steps、GPU 利用与完整 outcome/count/hash。
- 升级触发：先建立稳定 `(split, seed, cell, episode)` 初始状态身份，并实证逐 episode outcome、aggregate 和 task-evidence hash 等价，同时墙钟确有收益。
- 禁止：不能用理论步数节省宣称等价；不等价时必须升 eval/runtime identity。

### 2. progress 只作为 task-owned 非排序诊断

证据：R1 stable-success 0.20 / mean progress 4.5145；R2 0.855714 / 4.3014；R5 stable-success 0 / progress 5.2908，且 35/35 cells 比 R2 差。stable-success 提前吸收使 raw max distance 不适合跨候选排序。当前 progress 已退出 nonaccepted tiebreak，recency 决定同 objective anchor。

- 本轮裁决：保持 acceptance/objective/tiebreak 不变。
- 观察：本轮只监视现有 `goal_not_reached` count/rate 与全体 episode 的 per-tread progress；当前 summary 没有 failure×progress 联合统计，不能把两者做条件归因。
- 升级触发：若新轮继续出现 raw progress 与 stable-success 反向，才设计 task-owned、failure-conditioned 的 bounded progress fraction/normalized deficit 投影，并先离线证明解释质量。
- 禁止：不把 progress 恢复为能力分、接受门或共享 RewardInputs 特殊语义。

### 3. reward activation 的累计尺度与 clip 饱和

事实更正：当前 Stair per-term receipt 只有 observation/call count、last weighted mean 和历史 max absolute weighted contribution；没有 per-term nonzero count/rate。只有整体 activation 有非零判据。

- 本轮裁决：不在开跑前扩 runtime schema。
- 观察：逐轮核对新 term calls/observations/last/max、finite、总 reward tensor 与 eval 行为；记录“被调用/曾非零”与“有效改善”的区别。
- 升级触发：多轮出现 max 恰达 clip 而 last 接近零、且正式行为难以解释时，才增加固定内存 online sum/abs-sum 与 pre-clamp low/high hit counters。
- 禁止：不落逐步轨迹、不二次执行 candidate callable、不建通用 telemetry 框架、不把 activation 当因果证据。

### 4. 跨 parent 的相似被拒机制归因

证据：R13 memory 中仍保留 R6 rejected orientation family，selection guidance 也要求说明最近先例；但 R13 hypothesis 未引用它，随后 25/35 cells 退化、objective 0.821429、guard 0.842857。问题不是 memory 缺失，而是相关负证未被作者使用。

- 本轮裁决：保持 `historical_variant_assessment` nonblocking，不增加硬门或自然语言因果 verifier。
- 观察：每轮记录 parent、依赖输入、机制家族、最近相似拒绝项是否在 memory、hypothesis 是否点名及代码级差异、后验 cell 结果。
- 升级触发：新 15 轮再次出现“相关历史在场但未引用”的同族失败，才 A/B 测试从现有 `experiment_memory` 结构化机制记录生成非阻断提示；same-parent 时可附加现有 historical assessment。
- 禁止：same-input/same-family 不能直接阻断；不得新增第二套机制 analyzer。

### 5. production training diagnostics receipt

证据：旧轮中 TensorBoard reward/complete reward 可上升而 formal eval 归零，terrain/base-contact 已能早期揭示背离；但 production summary 没有有界 TB receipt。现有 pilot summarizer 是 pilot-specific，并非可直接复用的通用内核。

- 本轮裁决：继续逐轮人工/审计脚本读取 allowlisted tags，不在开跑前扩大训练失败面。
- 观察：tag steps 0..1999、finite、terrain/base-contact/episode-length/mean reward 的同阶段 tail，对照 formal eval，明确 training-only/non-causal。
- 升级触发：新轮再次证明这些量能稳定提前发现塌缩，且可抽出不改变 pilot 行为的最小 bounded summarizer。
- 禁止：不进入 acceptance、provider prompt 或因果归因；receipt 与 train summary 不形成互相 hash 的环。

### 6. first-episode finite 生命周期

证据：合成复现证明已闭合 env 的 auto-reset 尾部 NaN 会触发全批 finite false-reject；这是 liveness/假拒，不是假接受。正式 R0–R6、R8–R14 所有实际 eval 均 pass；R7 没有 eval，旧实物未受影响。

- 本轮裁决：保持当前 evaluator，逐轮严查 finite/status/errors。
- 升级触发：正式新 15 轮实际出现此 false-reject，或 eval-v7 专项证明逐 env active lifecycle 与旧活跃 episode outcome bitwise/aggregate 等价。
- 根修边界：done/stable-success 后 inactive；关闭步本身仍严格验证 action/reward/transition；auto-reset obs 排除；decoder 跳过 inactive。
- 禁止：不能只删除全局检查或只遮罩 wrapper `done`。

### 7. task-section aggregate 的 post-hoc receipt

事实更正：32,768B 是 shared hard limit，512B 是 Stair adapter reserved headroom。R8 可借当前 builder 重构 31,440/32,256B，但 artifact 自身不能独立复算原 task aggregate；promotion/shared guidance 已改变最终表示。“不复制内容且独立复算原 aggregate”目前不可同时满足。

- 本轮裁决：继续在生成时 hard fail，并逐轮重构检查；不先造伪 receipt。
- 升级触发：先决定唯一 canonical representation（独立未变换 task aggregate 或可逆 assembly），再绑定 hash/bytes/serializer/headroom。
- 禁止：不扩大预算、不加入 Stair 特殊 receipt 字段、不用额外 hash 掩盖不可逆 assembly。

### 8. authoring evidence / attempt-result 的统一有界 envelope

证据：live summary v6 已不再复制完整 provider attempt，只保留数量与不可变 evidence 引用；但 `authoring_evidence.json`、`attempt_result.json` 与 budget artifact 的既有加载器仍没有统一的 producer/reader 字节上限。本地旧实物与 ledger refs 中 authoring evidence 均小于 17 KiB、attempt-result 小于 28 KiB，不构成本轮容量阻断，但不能据此宣称全部 authority read 已有界。

- 本轮裁决：不在发布前临时塞一个拍脑袋的 2 MiB 常量；继续逐轮记录 evidence/attempt-result 实际 bytes。
- 缺口状态：无界 producer/reader 已确认；当前因 artifact 由可信本地 runner 生成、且上游 prompt/provider/package 已有边界而评为非阻断，但这不等于缺口可以不修。
- 升级前提：本轮持续采样实际 bytes 只用于 cap sizing。共享配置目前未把 5 attempts 设为通用硬上限，因此不能直接按 Stair 默认值拍出公用 cap；下一版必须先确定通用 durable-attempt、路径和诊断字节边界，或改为 evidence refs/hash chain，再统一根修。
- 根修边界：Ref 构造、producer 落盘前、resume loader 与 verifier 共用同一上限和 cap+1 FD-bound read；budget 使用独立小上限。
- 禁止：不能为迁就 summary 缩减诊断证据，也不能只给 reader 加 cap 而让 producer 在第 15 轮结束时才失败。

## 删除的提案

删除“跨版本恢复时按当前 provider contract 重新投影旧 repair 自然语言”。R8 的一次性旧 v1 repair 与 v2 contract 冲突说明跨版本恢复需要明确兼容边界；正式会话只允许在同一批准 HEAD 和 clean release state 下恢复，跨版本恢复不受支持并在发布状态检查处 fail closed。旧 repair 只作 immutable evidence，共享层不维护跨版本自然语言迁移器。

## 每轮实时审计矩阵

新 15 轮每轮必须在下一轮 authoring 前闭合以下证据，并跨轮维护探索关系：

1. **Authoring**：frame/omissions/section bytes、task aggregate headroom、source owner/hash、memory retained/omitted 及 current-best/latest-failure保护、provider model/thinking/output cap/transport receipt、attempt预算与失败分类。
2. **单机制 authority**：frozen round anchor、lineage parent、candidate source hash、canonical hypothesis、changed mechanism count/name、repair 是否只修声明问题；任何 unknown/null delta 不得训练。
3. **Package/runtime**：5-file contract、sealed hash、admission/authorization、组件 smoke、anti-degenerate、execution environment manifest、初始 policy/hash、seed/budget/checkpoint、runtime activation 和 finite。
4. **Train**：完整 2,000 iterations/196,608,000 steps、启动时枚举并冻结的实际 TensorBoard scalar tag set 与核心健康 tags 的 step 覆盖/finite；健康代理只作诊断，不替代 eval。
5. **Eval**：public-dev/public-accept/sealed 的 3/5/7 seeds × 35 cells × 8 episodes、per-seed/per-cell counts重组、task evidence hash、checkpoint/identity/device、candidate isolation、actual argv、wall/closed-first cost。
6. **Decision/lifecycle**：objective/guard/sealed 与动态阈值、accept/reject reason、current-best 更新、train→eval→ledger 时间序列、lease/process/GPU/磁盘清洁。
7. **跨轮探索**：parent/current-best、最近失败与相似机制、memory是否在场、hypothesis是否吸收负证、cell级收益/回退、provider和eval累计成本；不得只按单轮 reward 或总分叙事。

## 第 15 轮后唯一计划动作

完成 ledger 15 行、acceptance replay 和进程清洁后，按 Genesis 前冻结的 Stair final-confirmation plan 单次运行 Genesis vs final current-best paired cohort。先验证 plan/Genesis/ledger/checkpoint/backend identity，再创建不可逆 reservation；完整两份 canonical 35-cell summaries 和派生 paired delta 写入独立 result。失败后不得换 cohort、改统计或重跑。该结果只支持最终泛化说明，不改变 current-best，不开启第 16 轮，也不反馈给任何后续 authoring。
