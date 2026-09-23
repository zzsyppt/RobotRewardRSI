# P0 策略、父子关系与评估来源目录

更新：2026-09-23。已按第四次审计补入最近 Go2 Push，并把原病例缓存改为来源目录。重跑 [inventory_extract.py](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/inventory_extract.py) 生成 [inventory_manifest.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/inventory_manifest.json)，不调用模型或模拟器。

## 可直接使用的两条修改链

| 搜索会话 | 完成训练的候选条目 | 不同权重 | 实际 parent→candidate 边 | accepted | 作者材料 |
|---|---:|---:|---:|---:|---|
| formal Go2 Stair c6，15代 | 23 | 22 | 23 | 1：Gen8-A | 53份实际prompt/attempt回执，36份候选源码rationale |
| Go2 Push attempt_08，5代，force v3 | 10 | 10 | 10 | 2：Gen1-A、Gen4-B | 15份实际prompt/attempt回执，10份候选rationale |

formal 的 Gen2-B 与 Gen3-A 都是 `sha256:55a38c938e7d13876f14ffeaf9a8913d01e5ede2d4c511ff0102314d5fc29337`，目录合并权重身份，同时保留两次训练与两个作者关系。24份训练 summary 中 Gen3-B 失败，没有正式发布的 best checkpoint；23个完成训练的lane各有一条真实父边。**15是代数，23才是已训练候选的父边数。**正式 PT 历史远端核验为71/71存在且匹配，其中包括最终权重副本和中间迭代。此次另核原parent的best.pt，6,882,293字节且完整hash匹配；其train summary与resolved manifest已只读取回，见 [parent核验](./inventory_parent_remote_verification.json)。

formal 已训练的 Gen1–8 以 reference `fca6a59c...` 为作者 parent，Gen9–15 以 Gen8-A `bcd8de3a...` 为 parent。Push 的 Gen1 以 reference `1495390b...` 为 parent，Gen2–4 以 Gen1-A `0fcb1ddc...` 为 parent，Gen5 以 Gen4-B `5714a1f0...` 为 parent。完整 hash、每条 trial、prompt、rationale、summary 和 ledger pointer 均在机器目录中。

两会话均采用 random policy initialization，而非从作者 parent 的权重续训。formal 24次训练共享初始状态 `a32148c6...`、seed schedule `8f4c7a40...`；Push 10次训练共享初始状态 `9a9c8718...`、schedule `e065c0f1...`，各自训练 reservation 对应 stdout 均记载 `Environment seed: 42`。formal 23个成功训练及原parent的summary还直接记录 `effective_agent_execution.seed=1262835026`，目录据原件列出；失败训练未发布相同字段时保留缺失。父子关系表示代码修改历史，训练初始化单独列出。

原parent的训练manifest为 `54cd180c9f39f65bffd71482ec8a87cda912d0e378e8b434d0c44c839a3ab1c2`（v1），Gen8-A为 `193fa9038a6f46539fb71a7e73edc44bdc024be2b899785394b8f82113c91f37`（v2）。两者训练binding及机器人I/O合同一致，评估successor规则有变化。当前v3 runner仅接纳直接前驱，fca跨v1→v3的合法祖先准入仍是P2启动前置，不能用Gen8-A的PASS代替；完整差异和范围见 [P2卡](./RESOURCE_SCOPE_AND_MIN_LOOP_CARD_ZH.md)。

## 当前评估域已有双策略材料

| 对象 | Public-dev 已存结果 | 用途 |
|---|---|---|
| Go2 Stair v3 reference `7b582643...` | 689/864；J_primary=0.7466970595 | 当前reference画像 |
| formal Gen8-A `bcd8de3a...` 的v3重放 | 822/864；J_primary=0.9244268914 | 与上行做同套卷描述差分 |
| Go2 Push force v3 reference `1495390b...` | 141/192=0.734375；保存reference gate pass | 当前Push reference画像，有指标余量 |
| Push Gen1-A / Gen4-B | 169/192、175/192 | 已接受修改的差分样例 |
| Push Gen3-A | 0/192 | 真实退化样例；保留其parent与作者材料 |
| G1 Stair v3 / G1 Push reference | 618/864；96/192 | 跨机器人兼容与诊断材料 |

旧结论“Go2 Push D是192/192，因此没有可用Push修改链”已纠正：同一reference权重在旧协议达到192/192，在attempt_08的force v3协议为141/192；协议身份随评估记录保存。formal 其余候选的历史v2 summary只有聚合，新目录为它们登记真实权重与训练身份，后续可用现有重放入口获得当前域facts。

Gen8-A的v3原件在 [public_dev_summary.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/audit4_replay/gen008a_v3/public_dev_summary.json)。它与P3 reference具有同eval binding与seed cohort，但执行源码/环境记录不同，当前可做描述差分。P3 reference `7b582643...` 不是 Gen8-A 的真实作者parent；目录保留这一差别。

主要源根目录：

- formal：[inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba)。rationale已取回 [author_evidence_raw/formal_c6/provider_candidates](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/author_evidence_raw/formal_c6/provider_candidates)。
- Push：[Go2_push/attempt_08/session](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/Go2_push/attempt_08/session)。本轮只读远端核验10候选与reference，共11个权重，全部hash匹配：[inventory_push_checkpoint_verification.json](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/P0/inventory_push_checkpoint_verification.json)。
- Stair v3 reference：[cohort_alignment_20260909](D:/humanoid_agent/true_RL_Agent/PLAN/2new_world/P3/results/cohort_alignment_20260909)。Go2原始训练summary在P2目录且已绑定；G1原始训练summary本地缺失，机器目录登记远端记录路径与缺项。

## 目录用法与成本

`sources` 保存短ID、仓库相对POSIX路径、文件hash与大小；消费者从仓库根目录解析，不依赖本机盘符；`policies` 按权重hash合并，保存训练attempt/seed/init、checkpoint位置和public-dev评估摘要/事实pointer；`parent_candidate_pairs` 保存33条真实关系及实际prompt/rationale来源；`sessions` 给出会话计数、失败或未训练trial与初始化绑定。共37个策略身份、302个来源，额外reference用于当前域对照与跨机器人验证。原始episode facts留在summary中，已删除9,880,541字节的重复展开JSONL。

简易检查覆盖来源存在、父子ID闭合、ledger与训练checkpoint一致、实际prompt校验和匹配、rationale路径、会话计数、真实public-dev split、执行版本来源及primary计数守恒。全部通过。38份评估的来源pointer已逐一回读，33对候选的训练初始化与session绑定已核对。脚本按被引用文件取hash，不再重复验证全部177份下载原件，也不建立封存系统。

formal历史结算仍为45.926567 GPU容量小时、24收费训练槽、69训练/评估attempt、203 eval seed units、53 provider attempts；其中一次失败训练按完整预算收费，reference训练成本来自此前导入。该数字用于后续阶段预算参照，当前目录工作没有新增GPU训练、重放或API调用。作者如何使用这些证据见独立作者基线报告，此目录只负责来源。

**边界：**以上是现存策略与作者历史的开发材料。共享初始化、同会话以及同策略多个episode仍保持关联；它们适合画像、差分和事后规则验证，正式多种子收益研究需要另行冻结独立search sessions。public-dev原始facts可直接回读，私有训练seed与路径留在审计目录。既有v3重放由此前审计执行，本轮只读纳入。
