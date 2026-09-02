# Fresh formal 15-round（20260813T102722Z）follow-up

本文件只登记本轮正式实验需要继续观察、或实验后可能值得升级的事项及证据。它不改变当前
acceptance、训练预算、seed、eval protocol、provider attempt 预算或正式运行中的 release HEAD。
发现权威 blocker 时另行停止并根修；非 blocker 只在这里累积证据，15 轮与 never-fed 最终确认完成后统一裁决。

正式运行身份：

- release commit：`6042e72950cd7a8de6edf00b204d260075d906d3`
- session：`stair-reference-formal-20260813T102722Z`
- output：`/data/yuyangliu/true_rl_agent_phase_g/fresh_formal_15round_20260813T102722Z`
- Session Genesis hash：`23e65099740bcc72a674ae8ef2c2c54f5782bc1dd3e490a8c692e1667452ca79`
- reference comparison objective：`0.5385714285714286`
- reference guard：`0.575`
- reference sealed：`0.5459183673469388`

## 从上一轮 follow-up 迁入的未关闭项

来源：`outputs/formal_15round_38818b5_evalv6_followups.md`。以下 8 项仍需本轮实物观察；旧文档中已经由代码、回放和发布门关闭的 authoring anchor、ledger v7、backend manifest、final-confirmation precommit 等不作为 pending 重复迁入。

### P1. tread horizon 同质分组是否真的节省评估成本

- 现状：formal evaluator 保持不变；reference 三 split 实测共 `2,982,000` simulated control steps、`831.455s`。
- 本轮采集：每轮三 split 的 wall time、vector calls、simulated steps、closed-first steps、GPU 利用与 task-evidence hash。
- 升级触发：只有在固定 episode identity 后逐 episode outcome/aggregate 等价，且墙钟实测有收益，才试验 5 个 horizon 分组。
- 禁止：不能用理论 15.49% step 节省宣称行为等价。

### P2. progress 仅作为 task-owned 非排序诊断

- 本轮采集：`goal_not_reached` count/rate、全 episode per-tread progress、stable-success 与 35-cell objective 的方向关系。
- 升级触发：若再次出现 progress 上升而能力明显下降，才设计 failure-conditioned bounded projection。
- 禁止：不恢复为排序、接受门或共享 RewardInputs 语义。

### P3. reward activation 累计尺度与 clip 饱和

- 本轮采集：per-term calls/observations/last/max、finite、overall nonzero、正式行为结果。
- 升级触发：多轮出现 max 贴 clip、last 接近零且行为难解释时，才考虑固定内存 sum/abs-sum 与 pre-clamp hit counters。
- 禁止：不落逐步轨迹、不二次执行 candidate callable、不把 activation 当因果证据。

### P4. 跨 parent 的相似被拒机制是否真正被作者利用

- 本轮采集：parent、依赖输入、机制家族、相似拒绝项是否在 memory、hypothesis 是否点名、代码差异与 cell 结果。
- 升级触发：再次出现“负证在场但作者未引用”的同族失败，才 A/B 测试结构化非阻断提示。
- 禁止：不新增第二套机制 analyzer，不把自然语言相似度变成硬门。

### P5. production training diagnostics receipt

- 本轮采集：TensorBoard 实际 scalar tag set、steps `0..1999`、finite、terrain/base-contact/episode-length/mean reward 的阶段尾部，并与 formal eval 对照。
- 升级触发：若这些 training-only 指标在多轮稳定提前揭示塌缩，再抽最小 bounded summarizer。
- 禁止：不进入 acceptance/provider prompt，不做因果宣称，不与 train summary 形成 hash 环。

### P6. first-episode finite 生命周期

- 本轮采集：每轮 eval finite/status/errors，以及 active first-episode count、closed-first steps 与 primary outcomes 是否闭合。
- 升级触发：正式出现 auto-reset tail 导致的 false reject，或专项证明逐 env active lifecycle 与旧活跃 outcome 等价。
- 禁止：不直接删除全批 finite 检查或只遮罩 wrapper `done`。

### P7. task-section aggregate 的 post-hoc receipt

- 本轮采集：task aggregate/headroom、最终 prompt payload bytes、assembly 后的 included/omitted sections 与重构一致性。
- 升级触发：先选定唯一 canonical representation，才能新增 hash/bytes/serializer/headroom receipt。
- 禁止：不扩大 shared 预算、不加 Stair 专属共享字段、不用额外 hash 掩盖不可逆 assembly。

### P8. authoring evidence / attempt-result 的统一有界 envelope

- 本轮采集：每个 `authoring_evidence.json`、`attempt_result.json`、budget artifact 的实际 bytes，以及 durable attempts 数。
- 升级前提：先决定通用 durable-attempt/path/diagnostic 边界，或改 evidence refs/hash chain；不能按 Stair 默认 5 attempts 拍一个通用 cap。
- 根修范围：Ref、producer 写前、resume loader、verifier 共用同一 contract；budget 用独立小上限。

## 本轮刚发布改动的重点回归观察

这些项目已有测试与发布证据，不预设仍有缺陷；本轮要用真实 15-round artifacts 复核，而不是只依赖单测。

### N1. frozen authoring anchor / evidence v4 / ledger v7

- 每轮核：首个真实 prompt anchor、parent program checksum、public prompt content hash、selected prompt、canonical mechanism delta 与 current-best before 完全绑定。
- repair/source-followup 只能改变展示帧，不能改变 round 初始比较锚。
- 任何 replay reason、unknown/null delta、parent 漂移均为 blocker。

### N2. live receipt v6 与单次 `0 -> 15`

- 每轮核：invocation start identity、round accounting、ledger append 顺序、root summary/immutable snapshot/exit receipt。
- 最终必须 `resumed_from_round_count=0`、`rounds_executed_this_invocation=15`；`14 -> 15` 与 `15 -> 15` 不具最终确认资格。
- 通用 resume 仍保留，Stair formal 的单次闭合要求只在 task-owned final gate。
- 阻断实证：当前 final finder 只统计 verified completed invocation，会忽略同 OUT 中先前 interrupted/failed journal；临时 fixture 的 `interrupted(0) -> completed(0→15)` 被 task-owned final gate 接受，与 runbook 的单次 invocation 契约冲突。
- 处置：R1 训练中优雅中断，当前 OUT 只保留开发证据；exit=`interrupted/keyboard_interrupt`、ledger=0、GPU/PID 清洁。根修只在 Stair final gate 要求 `live_invocations` 目录恰好一个且为完成 receipt 对应目录；不收紧 shared resume/finder。

### N3. final-confirmation plan 与 never-fed 执行绑定

- 15 轮前只核 plan/Genesis commitment 未变；不得提前运行 confirmation。
- 15 轮后核 fresh ledger replay、nonzero trained/full/sealed counts、Genesis/final checkpoint、namespace/cohort、backend/source/device receipt 与 single-use reservation。
- result 只作 diagnostic，不写回 ledger/prompt/history/memory。

### N4. backend execution manifest 的正式/非正式作用域

- 正式 live 的 Genesis、train/eval child 必须重算并精确回显 manifest。
- Push dry-run、manual materialization 与非-Isaac 插件不得因该身份采样退化。
- Stair task/upstream source 仍由 task-owned preflight 绑定，不进入 common manifest。

### N5. 共享 `fs_contract` 收敛后的文件语义

- 观察 create-once 与 atomic replace 调用点是否保持原语义、无残留短 temp、无 `FileExists`/长路径异常。
- bounded read 只承担 task-neutral regular-file/cap；领域 JSON、checksum、cap 和目录生命周期仍由调用方负责。
- 不再增加同进程攻击者、竞态 fuzz 或 site-packages 防黑客式复杂防护。

### N6. provider transport 与正常失败路径

- 正式路径核：HTTPS、query 原样传递、no redirect、response cap、HTTP/TLS/network 分类、retry 只消费既有 attempt budget、receipt v2。
- 本轮 provider endpoint 无 query；Azure query 兼容已由定向测试覆盖，不在 live 中另造请求。
- 不做恶意容器、多层转义、自定义 transport fuzz 等超出项目威胁模型的审查。

### N7. public projection `hostname` 与正式诊断脱敏

- eval/runner public projection 的 `hostname` 必须拒绝；普通 `ghost_writer` 等不得误伤。
- 只审真实 provider/runner/Push diagnostic 流，不继续扩展通用安全规则。

### N8. prompt/history/memory 充分性与膨胀

- 每轮记录 first/repair frame、payload bytes、included/omitted sections、history recent/omitted、memory retained/omitted、public/runtime feedback、best-observed/current-best 与 provider usage。
- R9 后 recent history 达上限可正常 compact；15 轮内 experiment memory 不应因条目上限丢失，只有字节预算压缩才可 omission。
- 关注相关负证是否被作者实际吸收，而不只判断“字段存在”。

## Reference 与启动阶段已采证

- reference：3/5/7 seeds × 35 cells × 8 episodes，共 4,200 first-episode outcomes；三 split objective `0.560714 / 0.538571 / 0.545918`。
- reference 成本：wall `168.033 / 275.557 / 387.865s`；总 vector calls `10,650`，simulated steps `2,982,000`，closed-first steps `1,941,058`。
- raw ingestion、Stair projection、reference qualification、Genesis 及 load-only 全部通过；14,520 个数值节点 finite；candidate code/package/path 均未进入 eval。
- 第一次 tmux 包装命令因 shell 多行转义产生空白 argv，在 argparse 前退出；没有 invocation、ledger 或 runner round。日志保留为 `live_cli_parse_error*`，随后以参数数组启动唯一正式 invocation。该操作问题不改变项目协议；不为此新增运行时抽象。
- 正式 invocation：`22f3f164ec554771b159674884f9b44e`，从空 ledger 开始；不得自动重启。

## 逐轮证据登记

### R1（运行中）

- authoring：attempt 1/5 即 `candidate_verified`；候选 `stair_goal_progress_v1`，单一机制 checksum `sha256:b4919c...`。
- prompt：71,312B / 262,144B；reference components 4,978B 完整内联；history=0、memory=0 合理；约 0.5KiB 空 optional/legacy alias 冗余，当前非 blocker。
- task section：31,336B / 32,256B Stair target，剩余 920B；当前未触发 compaction，但从 R2 起需重点观察反馈增长是否导致 headroom 不足。
- provider：request 18,234、response 11,257、reasoning 9,458、total 29,491 units；cache hit 11,008 / miss 7,226；finish=`stop`，一次完成。
- anchor：parent/current-best 为 Genesis reference `sha256:a125...a3a3`，comparison objective `0.538571`；ledger 尚未闭合前不作 decision 结论。
- train：4096 env、2,000 iterations、random init、`cuda:0`；持续更新中。
- 终止：静态审计发现 N2 正式性 blocker 后，在约 `model_1500+` 阶段向唯一 live PID 发送 SIGINT；wrapper exit 130，invocation exit=`interrupted/keyboard_interrupt`，closed ledger count=0，ledger 未创建，全部实验 PID/GPU context 已清理。该轮只作开发证据，不进入正式结果。

后续每轮在 ledger 新行出现后补：authority/replay、train/eval budgets、三 split outcome/cost、decision/current-best、prompt/memory/provider、训练诊断和非阻断提升证据。
