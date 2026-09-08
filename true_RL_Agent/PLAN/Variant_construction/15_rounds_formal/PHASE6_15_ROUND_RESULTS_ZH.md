# Go2 × Stair：15 代正式实验逐代结果

> 状态：**实验工程验收通过**。15 代、final-confirmation、正式报告与生产只读重放均已完成，作业已退出、GPU/lease 已释放。科学结论是一次正式接受及新评测 seed 上的平均改善，不是 15 代持续单调提升。本报告区分成功率、目标分数、科学判决与合法 incomplete，不把缺测写成零。

## 1. 范围与计量口径

- 会话：`formal-go2-stair-15round-20260901-c6ab8ba`；15 代生成/训练版本：`c6ab8bae78201e91d8a6c02911505ced01d7d6c8`；终态修复执行版本：`17e819d50979e10c8a59bd39c497364e88ebc80b`；机器人任务格：`stair.go2_12dof.unified_v2`。原 ledger/训练身份没有改写，终态来源由独立 sidecar 绑定。
- 15 代中：`1 accepted + 9 formal_rejected + 1 no_selection + 4 incomplete`；11 代有完整科学判决，不能写成“15代均完成两候选训练和比较”。Gen8-A 是当前唯一被接受的候选，Gen9–15 没有改变 current-best。
- **成功率**：原始 `primary_outcome_counts.success / total_episode_count`，成功语义为首次合法到顶锁存。下表同时列整数 `n/N` 与百分比。
- **分数**：单 cell 得分为 `0.5×成功率 + 0.5×平均目标进度`；固定层级汇总得到各 stratum/family、`J_ID` 和 `J_OOD`，最终 `J_primary=(2×J_OOD+J_ID)/3`。这些都不是成功率。本会话 mask 的三个 family 均合格；`F_scalar={height_ood,tread_ood,joint_ood}`，`F_c1={phase_composition_ood,c1_conditional_frontier}`，`F_c2={c2_mechanism_ood}`。
- 一份 public-dev 为 `3 seeds×24 cells×3 episodes=216` episodes；public-accept 为 `360`；sealed-validation 为 `504`。三套卷几何互不相同，分数不能用跨套卷单次大小直接替代同套卷的配对比较。候选筛选只消费 public-dev；正式接受消费被选中候选的私有评测。
- `—` 表示没有完成该项评测/没有选择，绝非 0；例如 Gen4-B 的 `0/216` 是真实完成评测后的零成功，与 Gen3-B 的缺测不同。

## 2. 导入 reference 的原始评测

15 代启动时只导入 reference，没有重新训练或重复其原三套卷评测；以下来自 P5 已冻结的三份 phase receipt，不是另行进行的 final-confirmation 新 seed 结果。

| 套卷 | 成功 n/N（成功率） | J_primary | J_ID | J_OOD | F_scalar | F_c1 | F_c2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| public_dev | 187/216（86.57%） | 0.934438 | 1.000000 | 0.901658 | 0.704973 | 1.000000 | 1.000000 |
| public_accept | 310/360（86.11%） | 0.933915 | 1.000000 | 0.900873 | 0.702619 | 1.000000 | 1.000000 |
| sealed_validation | 429/504（85.12%） | 0.932641 | 1.000000 | 0.898962 | 0.696885 | 1.000000 | 1.000000 |

Reference checkpoint：`sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`。

## 3. 每代 A/B 公开筛选与被选中候选正式评测

表内每个 `J` 列均为 `J_primary`，不是成功率。“选择”列是公开 selector 的结果，不等于正式接受。完整 `J_ID/J_OOD/F_scalar/F_c1/F_c2` 及 candidate ID、证据 SHA256 在[逐相位完整指标 CSV](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_EVAL_METRICS.csv)中。

| 代 | A dev 成功 n/N（率） | A J | B dev 成功 n/N（率） | B J | 选择 | selected PA 成功 n/N（率） | PA J | selected sealed 成功 n/N（率） | sealed J | Outcome |
| --- | ---: | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: | --- |
| 1 | 179/216（82.87%） | 0.918430 | 177/216（81.94%） | 0.913689 | A | 302/360（83.89%） | 0.924348 | 411/504（81.55%） | 0.914466 | 正式拒绝 |
| 2 | 181/216（83.80%） | 0.919597 | 187/216（86.57%） | 0.934775 | B | 316/360（87.78%） | 0.941493 | 418/504（82.94%） | 0.920791 | 正式拒绝 |
| 3 | 187/216（86.57%） | 0.934775 | — | — | — | — | — | — | — | 执行不完整 |
| 4 | 180/216（83.33%） | 0.922439 | 0/216（0.00%） | 0.067330 | A | 298/360（82.78%） | 0.921121 | 404/504（80.16%） | 0.908253 | 正式拒绝 |
| 5 | 171/216（79.17%） | 0.896827 | 163/216（75.46%） | 0.880709 | — | — | — | — | — | 无合格选择 |
| 6 | — | — | — | — | — | — | — | — | — | 成对写作不完整 |
| 7 | 177/216（81.94%） | 0.911484 | 193/216（89.35%） | 0.952738 | B | 317/360（88.06%） | 0.945472 | 434/504（86.11%） | 0.935756 | 正式拒绝 |
| 8 | 191/216（88.43%） | 0.948267 | 188/216（87.04%） | 0.937378 | A | 329/360（91.39%） | 0.960632 | 444/504（88.10%） | 0.944597 | 接受；更新 best |
| 9 | 176/216（81.48%） | 0.916905 | 178/216（82.41%） | 0.917860 | B | 292/360（81.11%） | 0.911802 | 394/504（78.17%） | 0.900884 | 正式拒绝 |
| 10 | 176/216（81.48%） | 0.916869 | 174/216（80.56%） | 0.908502 | A | 286/360（79.44%） | 0.909855 | 394/504（78.17%） | 0.900333 | 正式拒绝 |
| 11 | 176/216（81.48%） | 0.909893 | 182/216（84.26%） | 0.926323 | B | 296/360（82.22%） | 0.917555 | 422/504（83.73%） | 0.922242 | 正式拒绝 |
| 12 | 172/216（79.63%） | 0.899839 | 191/216（88.43%） | 0.940546 | B | 321/360（89.17%） | 0.944124 | 439/504（87.10%） | 0.936455 | 正式拒绝 |
| 13 | — | — | — | — | — | — | — | — | — | 成对写作不完整 |
| 14 | — | — | — | — | — | — | — | — | — | 成对写作不完整 |
| 15 | 198/216（91.67%） | 0.959695 | 184/216（85.19%） | 0.931876 | A | 318/360（88.33%） | 0.944599 | 447/504（88.69%） | 0.946853 | 正式拒绝 |

### 每代被选中候选 public-accept 的八层成功率

下表每格为成功 `n/45（百分比）`，每层固定 `3 cells×3 episodes×5 seeds=45`；没有选择或未完成正式评测仍写 `—`。A/B 两条 public-dev 的逐层明细均保留在CSV中，不以被选中候选代表未选中 sibling。

| 代/选择 | C0-ID | C1-ID | 高度OOD | 踏深OOD | 双轴OOD | 相位OOD | C1前沿 | C2机制OOD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1/A | 45/45（100.0%） | 45/45（100.0%） | 30/45（66.7%） | 41/45（91.1%） | 6/45（13.3%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 2/B | 45/45（100.0%） | 45/45（100.0%） | 30/45（66.7%） | 40/45（88.9%） | 21/45（46.7%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 3/— | — | — | — | — | — | — | — | — |
| 4/A | 45/45（100.0%） | 45/45（100.0%） | 29/45（64.4%） | 36/45（80.0%） | 8/45（17.8%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 5/— | — | — | — | — | — | — | — | — |
| 6/— | — | — | — | — | — | — | — | — |
| 7/B | 45/45（100.0%） | 45/45（100.0%） | 35/45（77.8%） | 38/45（84.4%） | 19/45（42.2%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 8/A | 45/45（100.0%） | 45/45（100.0%） | 44/45（97.8%） | 42/45（93.3%） | 18/45（40.0%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 9/B | 45/45（100.0%） | 45/45（100.0%） | 22/45（48.9%） | 41/45（91.1%） | 4/45（8.9%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 10/A | 45/45（100.0%） | 45/45（100.0%） | 21/45（46.7%） | 34/45（75.6%） | 6/45（13.3%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 11/B | 45/45（100.0%） | 45/45（100.0%） | 33/45（73.3%） | 30/45（66.7%） | 8/45（17.8%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 12/B | 45/45（100.0%） | 45/45（100.0%） | 30/45（66.7%） | 45/45（100.0%） | 21/45（46.7%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |
| 13/— | — | — | — | — | — | — | — | — |
| 14/— | — | — | — | — | — | — | — | — |
| 15/A | 45/45（100.0%） | 45/45（100.0%） | 34/45（75.6%） | 35/45（77.8%） | 24/45（53.3%） | 45/45（100.0%） | 45/45（100.0%） | 45/45（100.0%） |

### 四个 incomplete 的具体含义

| 代 | 已有实物 | 缺失项及权威原因 | 是否把缺测当作性能失败 |
| --- | --- | --- | --- |
| 3 | A 已完成 public-dev：187/216 | B 为 `candidate_failed/candidate_runtime_failed`；没有双路完整证据，因此没有 selector/formal | 否；A 成绩保留，B/正式评测标 — |
| 6 | A 已验证但未训练；B authoring 失败 | `paired_authoring_incomplete`；未启动单边训练 | 否 |
| 13 | A 已验证但未训练；B 耗尽 authoring 预算 | `paired_authoring_incomplete` | 否 |
| 14 | A 烟测修复后已验证但未训练；B 耗尽 authoring 预算 | `paired_authoring_incomplete` | 否 |

Gen5 是另一种状态：A/B 均完成 public-dev，但两者都未通过 selector guard；它属于 `no_selection`，不是 incomplete。

### 正式判决原因

| 代 | 权威 reason / 选择结果 |
| --- | --- |
| 1 | `named_guards_paired_headroom_delta_not_met` |
| 2 | `named_guards_paired_headroom_delta_not_met` |
| 3 | `candidate_execution_incomplete` |
| 4 | `named_guards_paired_headroom_delta_not_met` |
| 5 | `no_guard_eligible_candidate` |
| 6 | `paired_authoring_incomplete` |
| 7 | `named_guards_paired_headroom_delta_not_met` |
| 8 | `formal_acceptance_passed` |
| 9 | `named_guard_score_regression` |
| 10 | `named_guard_score_regression` |
| 11 | `named_guard_score_regression` |
| 12 | `named_guards_paired_headroom_delta_not_met` |
| 13 | `paired_authoring_incomplete` |
| 14 | `paired_authoring_incomplete` |
| 15 | `named_guards_paired_headroom_delta_not_met` |

## 4. 当前 best 与 reference：同套卷比较，不替代最终确认

当前 best 是 Gen8-A `stair_swing_foot_level_sweep_penalty_v1`；checkpoint 为 `sha256:bcd8de3a9d0e6e6a6c84b8c7833dc4a1b0ed697758ddee6f9c9eb31416a2f93c`，source 为 `sha256:29da35bd2a7aa6f68c0095422d79e118c68ef07f099c420a986b9a98435b3f45`。

| 套卷 | Reference 成功率 | Gen8-A 成功率 | 成功率差（百分点） | Reference J_primary | Gen8-A J_primary | J_primary 差 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| public_dev | 86.57% | 88.43% | +1.85 | 0.934438 | 0.948267 | +0.013829 |
| public_accept | 86.11% | 91.39% | +5.28 | 0.933915 | 0.960632 | +0.026717 |
| sealed_validation | 85.12% | 88.10% | +2.98 | 0.932641 | 0.944597 | +0.011956 |

Gen15-A 的 public-dev 虽达到 `198/216`、`J_primary=0.959695`，但 public-accept 为 `318/360`、`J_primary=0.944599`，未胜过 current-best 同卷的 `329/360`、`0.960632`，权威门仍拒绝。不能按最好看的单个套卷或最后一代替换 accepted best。

## 5. 八层成功率附录与数据核验

- [逐相位完整指标 CSV](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_EVAL_METRICS.csv)：`48` 行，即本次 `43` 份 pass candidate eval、`3` 份导入 reference 和 `2` 份 final-confirmation；成功计数/率、全部 J 与 family 分数、失败计数、candidate/checkpoint/source 身份和原始文件路径/哈希分列保存。
- [逐八层成功率 CSV](D:/humanoid_agent/true_RL_Agent/PLAN/Variant_construction/15_rounds_formal/PHASE6_15_ROUND_STRATUM_SUCCESS.csv)：`384` 行，即候选 `43×8=344` 行、reference `3×8=24` 行、final-confirmation `2×8=16` 行；按 evaluation_stage/generation/lane/split/stratum 给出成功 `n/N`、成功率、stratum score 与六类互斥 primary failure 计数。stage 明确区分 `generation/imported_reference/final_confirmation`；generation `0` 表示导入reference，final-confirmation的generation留空并用lane `genesis/final`标识两角色，不伪造第16代。
- 候选评测覆盖 `13,608` episodes，导入reference覆盖 `1,080`，final-confirmation覆盖 `1,008`，本附录共 `15,696` episodes。每份评测均重算 `sum(primary)=N`、逐 cell 成功数之和与整体成功数一致、每层分母等于 `3 cells×3 episodes×seed_count`；计数断言全部通过。J 与 family 数值直接抄自对应权威 summary，没有从成功率重新构造。
- 人工另读 reference、Gen8、Gen12、Gen15 的原始 `primary_outcome_counts` 和 `ordered_summary.seed_summaries`，不只依赖脚本绿灯。例如 reference PA 的逐 seed 成功数 `61+62+62+62+63=310`；Gen8-A PA 为 `67+64+65+66+67=329`；Gen12-B PA 为 `64+65+63+64+65=321`；Gen15-A PA 为 `65+61+63+62+67=318`。sealed 逐 seed 和分别为 `429/444/439/447`，均与全局 primary 成功计数一致。Gen8-A PA 的互斥 primary 又是 `329 success+30 forbidden+1 orientation=360`；Gen15-A PA 为 `318 success+21 forbidden+5 orientation+1 off_course+15 timeout=360`。

原始证据位置：

- ledger：`/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba/ledger.jsonl`；本稿抽取时 SHA256 为 `052a71e7c438f80789fb22a2fcdf5b921dd864ed4c2cd35019bb82097f900845`。
- 候选 summary：`/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba/runner/formal-go2-stair-15round-20260901-c6ab8ba-generation-NNN-lane-{a,b}/eval/{public_dev,public_accept,sealed_validation}/summary.json`。每条实际文件的完整路径及 SHA256 均在两个 CSV 中，不用通配路径代替证据身份。
- reference artifact：`/data/yuyangliu/AGENT/true_RL_Agent/outputs/p5_go2_pseudofall_removed_genesis_0a3f08a/reference_eval_artifact.json`；其 `evaluation_assembly.phase_receipts` 指向三份一等 phase receipt，本次先验证各 receipt 文件 checksum，再读取其中 `phase_result.task_evidence.summary`。

## 6. Final-confirmation：新 seed 配对确认已完成

两份权威 summary 均为 `pass`，并与 `final_confirmation/result.json` 对应的 `generalization_summary` 完全相同。reference 与 Gen8 accepted-current-best 使用**同一组7个新seed、同一批24道 sealed-validation 几何**，每道每seed为3个episode；两路各504、合计1,008 episodes。这是新seed的最终配对确认，**不是新几何 holdout，也不是第16代候选实验**。

Result 明确声明 `acceptance_authority=false / feedback_allowed=false / public_history_projection=false`，角色为 `diagnostic_final_generalization_confirmation_only`；该结果不反馈给 LLM、不参与候选接受或再次选择 best。

| 角色 | 成功 n/N（成功率） | J_primary | J_ID | J_OOD | F_scalar | F_c1 | F_c2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference（genesis） | 425/504（84.33%） | 0.928296 | 1.000000 | 0.892443 | 0.677330 | 1.000000 | 1.000000 |
| Accepted Gen8-A（final） | 441/504（87.50%） | 0.944074 | 1.000000 | 0.916111 | 0.748332 | 1.000000 | 1.000000 |

整体增加 `16` 个成功 episode，成功率提高 `3.17` 个百分点（`16/504`）。Result 的配对 `J_primary` 平均差为 `+0.015778147119341588`，标准误为 `0.002852792995655226`；7个seed的J差均为正（范围 `+0.008381..+0.029530`）。这是本次固定新seed样本上的改善证据，不是对未测机器人、未测几何或所有失败机制的泛化保证。

### 最终配对的八层成功率与分数

每层两路均为63个episode。成功计数与stratum score独立列出，不能把二者互换。

| Stratum | Reference 成功 n/63（率） | Gen8-A 成功 n/63（率） | 成功数差 | Reference stratum score | Gen8-A stratum score |
| --- | ---: | ---: | ---: | ---: | ---: |
| c0_id | 63/63（100.00%） | 63/63（100.00%） | +0 | 1.000000 | 1.000000 |
| c1_id | 63/63（100.00%） | 63/63（100.00%） | +0 | 1.000000 | 1.000000 |
| height_ood | 47/63（74.60%） | 60/63（95.24%） | +13 | 0.819229 | 0.962373 |
| tread_ood | 44/63（69.84%） | 49/63（77.78%） | +5 | 0.777488 | 0.849337 |
| joint_ood | 19/63（30.16%） | 17/63（26.98%） | -2 | 0.435273 | 0.433284 |
| phase_composition_ood | 63/63（100.00%） | 63/63（100.00%） | +0 | 1.000000 | 1.000000 |
| c1_conditional_frontier | 63/63（100.00%） | 63/63（100.00%） | +0 | 1.000000 | 1.000000 |
| c2_mechanism_ood | 63/63（100.00%） | 63/63（100.00%） | +0 | 1.000000 | 1.000000 |

改善集中在 height-OOD（`47→60/63`）与 tread-OOD（`44→49/63`）；joint-OOD 为 `19→17/63`，没有改善，其余五层两路均为 `63/63`。因此总分/成功率改善不等于“每种难度全面改善”。原始逐seed成功数经人工复核：reference `61+61+61+62+58+62+60=425`，Gen8-A `62+63+63+63+63+63+64=441`；八层成功数之和分别为 `5×63+47+44+19=425` 和 `5×63+60+49+17=441`。互斥 primary 也闭合为 reference `425 success+69 forbidden+1 orientation+1 off_course+8 timeout=504`，Gen8-A `441 success+53 forbidden+6 orientation+4 timeout=504`。

证据身份：

- 新seed cohort hash：`edada614fb88b6c5a95069d383349cade5158b9847f738224d6a2c879b8e1d87`。
- Result：`/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba/final_confirmation/result.json`；内容 `result_hash=c133660bcc3dc68863ec31d8ad723e60bd4bc4a0a373dc1d1f73e4fa46a9f1fc`，文件 SHA256 `046582e2665e9c301cc88e6cc336d40f8e7bc85637d49f4a801006bc06e99b42`。
- Reference原始summary：`/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba/runner/final_confirmation/genesis/eval/sealed_validation/summary.json`；SHA256 `74b8fdd0ddde35a88330d5a584f2eadeb4300f712bbbf15a79f3afce1bbc969f`。
- Gen8-A原始summary：`/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba/runner/final_confirmation/final/eval/sealed_validation/summary.json`；SHA256 `71fa278cee8092c6f900b3d48598d79f8b8ccf4ab3313c6a4b7d69fa87c9a143`。
- 两份summary使用的checkpoint分别精确等于导入reference的 `fca6a59c…` 与Gen8 accepted的 `bcd8de3a…`；配对7个seed的J差逐项重算与Result完全一致，均值和标准误复算误差低于 `1e-15`。

## 7. 终态验收与资源清理

- 正式 launcher 于 `2026-09-03T11:05:14.164273Z` 正常结束，`returncode=0`、stderr 为空；原 supervisor/launcher/session 与两条 final Isaac eval 进程均已退出。现场 `nvidia-smi --query-compute-apps` 为空；两张 GPU 无计算占用。
- `generation_session_report.json` 为 `868,429 B`，SHA256=`4ad1e53cc72afa0a9fe20ab69a8b5bc41e4ed9249efda4a7958477eb0fc150ca`，记录 `engineering_status=go`、11 项检查全部为真。独立检查实际 `15` 份 generation receipt/closure、`68` 个 runner summary 字节哈希、`69` 对 attempt reservation/result：`open=0`，68 completed 和 1 个既有 candidate failure。source→plan→reservation→phase receipts→Result→report、ledger/head 与成本链相符；两 GPU lease 同时重取检查通过。另一次服务器生产 `run_stair_generation_pilot(...terminal_audit_only=True)` 全重放正常返回 `pass`，重建报告与已存报告逐字段相同；输出根 `1,271` 个文件的大小/mtime 清单前后完全不变，ledger SHA256 仍为 `052a71e7c438f80789fb22a2fcdf5b921dd864ed4c2cd35019bb82097f900845`。重放证据保存于服务器 `.tmp/formal_terminal_completion_audit_17e819d.json`，不是只信报告自报的 checks。
- 最终累计 `45.926567` 计费 GPU-capacity 小时、`203` eval seed-units、`53` provider attempts；`24` 个计费 train jobs/`4,718,592,000` train steps 包含 Gen3-B 失败尝试的完整预算保守计费，实际完成训练为 23 个。最终确认仅新增 `14` eval seed-units、`2` jobs、`0.168702` GPU 小时，训练与 provider 增量均为零。所有预算上限检查通过。
- 清理范围是本实验的作业进程、GPU lease 与已由成功训练 retention 处理的未引用中间 checkpoint；最终 checkpoint、receipt、summary、ledger、报告和失败诊断保留。Gen3-B 的失败中间 checkpoint 及历史 lineage 产物未在已冻结成本之后手工删除，避免破坏可复算性；相关非阻断清理缺口保留在 F44/F53。没有删除其他任务的文件或停止其他服务，也未为“清理”另做产品架构改造。

## 8. 为什么没有连续多代提升

1. **搜索空间的可测余量集中在 scalar-OOD。** Reference 的 ID、F_c1、F_c2 已接近或处于天花板；在 final pair 中这五层全部 100% 成功。Gen8 的收益来自 height/tread，而 joint-OOD 仍弱；后续候选常在这些层之间交换收益，Gen12/15 的具体计数已列在表中，不能用开发卷的单项改善替代整体接受。
2. **15 代不等于 15 次完整有效比较。** 3 代因成对 authoring 不完整没有训练，1 代 candidate runtime failure 未形成双路比较，1 代两路均不合格；另有 Gen2-B→Gen3-A 的 exact reward 改名重跑浪费一条 lane（F52）。合法关账保障了会话继续，但没有把这些代变成科学成功。
3. **有界记忆仍有信息选择缺口，而不是已证实的上下文爆炸。** F24/F43/F46 记录 repair 连续性丢失、sibling 源码与详细反馈错位、半成功候选身份匿名化；F54/F59 记录形态尺度、坐标及 reward-rate 量纲误读。晚代输入仍成功进入 provider，三次截断发生在 reasoning 耗尽输出预算。不能靠删分层反馈来解释或修复这些问题。
4. **诊断观察不等于失败因果。** 全程 clearance diagnostic 618 次而 primary 仅 1 次；Go2 forbidden-contact 绑定的是 base，不是脚。Gen8 奖励改变与性能收益有实证，但“修好了脚撞竖板”没有当前轨迹/视频因果证据。对应 F41/F60/F63 继续只作提升项，不改本次门或结果。

结论是：**本次产生一次正式接受，最终新 seed 配对也保留了平均收益；没有证明 15 代单调进化、多训练 seed 稳定复现或所有 OOD 层改善。** 完整观察和未解决项分别见同目录 `PHASE6_OBSERVATIONS_ZH.md`、`PHASE6_UNRESOLVED_FOLLOWUP_ZH.md`。终态报告重复重放的额外 CPU 开销单列 F65，不影响本次判决，也未在结束阶段改代码绕过验收。
