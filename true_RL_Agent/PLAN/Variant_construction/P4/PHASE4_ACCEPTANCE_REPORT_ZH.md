# Phase 4 验收报告：G1-37 × unified Stair reference 与 reward LOOP

> 验收日期：2026-08-29
> 状态：**PASS**
> 工程门：`loop_verified=true`
> 观察结果：`improvement_observed=true`

## 1. 结论

唯一 active cell `stair.g1_37dof.unified_v1` 已完成：

1. official 15-term reference 从随机初始化训练并通过 ID learnability 与 OOD family 资格化；
2. 一份 1-generation pilot 完整闭合；
3. 一份全新 output/session/Genesis 的固定 3-generation validation 完整闭合；
4. LLM 的六个 validation candidate 都只修改完整 reward program；terrain、训练分布、课程、seed、
   budget、PPO、评测和 acceptance 始终由系统掌握；
5. 第 1 代观察到真实、通过 sealed 的改进；第 2 代候选因 C1-ID guard 退化被拒；第 3 代两个候选都在
   public-dev guard 筛选阶段未获选择，系统没有为追求结果追加代数。

这是开发验证闭环，不拥有 formal acceptance authority，也不代替 Phase 7 的最终 formal/final-holdout
研究结论。

## 2. 身份闭包

| 身份 | hash |
| --- | --- |
| ResolvedCellManifest | `cb9c93c1239d2f20eb58737b684cc74e21b6675a724198c470a58410f9ccdae2` |
| resolved task protocol | `3328f6600513d2e34f9b5f1cfab125505a9144fea7e2be23bc22ceb181d3b42e` |
| TrainingBinding | `d7aea85d7742c57eae1753b8b236b32555269d9bcd8c25ed2b3742d2e045e14c` |
| EvaluationBinding | `96d04d130cb669337297bbf8a5b53918b2006810c42586ff0ed374f6c18a0af2` |
| acceptance protocol | `d18b5345c8b3112546af401a8dd1560e6c6e840ad8c5ef4d102bd3e1ea5b4c24` |
| evaluation protocol | `aaea86a49cb01ea488eb3508685509dbee55c31589a268a922afd24ed082a942` |
| ordered suite | `4ee307fe359122671cf4f7056b601135b28679818a9b4bd419f3e7697891ceb7` |
| objective family mask | `04ad43edf20c7dfb7749492cc0b7e4fadcb881136444c0ccbc4cd40cb3284a54` |

pilot 与 validation 的 source commit 都是
`5a30129060353fcbddea0224d4544e23b8d2ee35`；validation output 在启动前不存在，没有继承 pilot ledger、
mask 文件、checkpoint 文件或 session lifecycle。

## 3. Pilot

- session：`phase4-unified-pilot-5a30129-20260829`；
- 模式：`development_generation_pilot`；
- 恰好 1 generation、width 2；
- 被接受候选：`stair_vnext_lateral_course_alignment_v1`；
- public-dev `J_primary=0.922423`；
- public-accept `J_primary=0.930234`；
- sealed `J_primary=0.915441`；
- ledger 恰好 1 行，terminal binding 的 closed/required generation 均为 1；
- 实际成本：3 train jobs、589,824,000 train steps、38 eval seed-units、3 provider attempts、
  3.8365 allocated GPU-hours；所有 cap checks 通过；
- 最终两个目标 GPU 均无 compute process，operation reservation 为 0。

pilot 成本闭合后才启动 validation。

## 4. Fresh 3-generation validation

session：`phase4-unified-validation-5a30129-20260829`。

| 代 | lane | candidate | public-dev `Jp` | public-accept `Jp` | 结果 |
| ---: | --- | --- | ---: | ---: | --- |
| 1 | A | `stair_clearance_aware_v1` | 0.655091 | 0.645481 | runtime effect 未观察到，未选中 |
| 1 | B | `g1_rough_lateral_course_deviation_v1` | 0.922433 | 0.942017 | accepted |
| 2 | A | `g1_progress_curriculum_steering_v1` | 0.880315 | 0.893764 | 未选中 |
| 2 | B | `g1_upright_goal_progress_v1` | 0.883497 | 0.906766 | formal rejected |
| 3 | A | `g1_forbidden_contact_dense_penalty_v1` | 0.863865 | 0.854112 | guard 不合格，未选中 |
| 3 | B | `g1_stair_proximity_speed_gate_v1` | 0.872304 | 0.874805 | guard 不合格，未选中 |

### 第 1 代接受

- accepted checkpoint：`sha256:65ee712a12b09945a44157cdf49550974299a9f241fe60c875f0092322890ef2`；
- public-accept：`J_primary=0.942017`、`J_ID=1.0`、`J_OOD=0.913025`；
- `delta=+0.296536`；冻结 threshold `0.070904`；paired SE `0.010592`；
- sealed：`J_primary=0.928965 ± 0.005295 SE`、`J_ID=1.0`、`J_OOD=0.893448`；
- sealed family：`F_scalar=0.696217`、`F_c1=0.984127`、`F_c2=1.0`；
- 三个 named guards、runtime activation、behavior、budget、accept-evidence、sealed gate 均通过。

### 第 2/3 代拒绝是有说服力的

第 2 代 lane B 相对 current-best 的 `J_primary delta=-0.035251`，低于 `0.017247` threshold；更直接的
拒绝原因是 `C1-ID delta=-0.093730`，超过冻结最大退化 `0.05`。这证明 objective 改善/接近并不能绕过
ID guard。

第 3 代两个候选的 public-dev guard regression 分别为 `0.078281` 和 `0.066065`，均无法进入 formal；
因此没有额外 sealed 运行。系统没有把“无候选”误当异常，也没有消耗 retry train/eval 预算。

## 5. Ledger、lineage 与终态

- ledger 恰好 3 行，outcome 顺序为 `accepted / formal_rejected / no_selection`；
- 每行 canonical record hash 与对应 execution receipt、generation report 相等；
- 第 2/3 代 parent hash 等于上一代 resulting current-best；第 2/3 代均未改写第 1 代 current-best；
- terminal current head：`8fe3f82b54639f64692ff4e6ae43a395bc9442d1ed9c9999f8feccf734788f26`；
- terminal ledger snapshot：`d85365fbb875f2b6b622f8dacb35da0f6f7948fe6250112a31494adcd126a4bf`；
- 三代 paired-ready rate 为 `3/3=1.0`；
- private public-accept selector consistency 为 `2/2=1.0`，reversal 为 0；该指标只在终态人工报告中
  生成，没有进入 provider；
- terminal-audit-only 复跑 return code 为 0；最终 compute rows 为空，open reservation count 为 0。

## 6. Prompt、失败反馈与多轮记忆人工审计

本项不仅运行脚本，还逐项阅读构造代码、11 份真实 prompt payload、generation records、public feedback
和 experiment memory：

- 第 3 代 minimum-core task section 为 `31,433 / 32,768 B`，仍保留 3 个 family、8 个 strata、
  family→strata 目标结构、三个 guards、C2 claim、seed 数和两条完整反馈；
- stratum failure table 的被 top-k 截断非零值为 `null`，真实零为 `0.0`，不再产生错误反馈；
- 第 1/2/3 代 `public_history` 为渐进式有界构建；第 3 代为 7,330 B，其中 experiment memory 4,725 B、
  2 条实验记忆，没有爆炸；
- repair prompt 删除历史/反馈但保留失败源码和静态契约，避免同轮修复重复灌入整段上下文；
- 11 份 validation prompt 共 792,328 B；扫描 `public_accept.<cell>`、`sealed_validation.<cell>`、私有
  cohort hash 和 reference qualification evidence hash，命中均为 0；
- 未发现会阻断本轮的冗余或信息缺失。3 个有证据但不应改变冻结实验的改进项单独记录在
  `PHASE4_FOLLOWUP_IMPROVEMENTS_ZH.md`，本报告不把它们伪装成当前缺陷。

## 7. 成本与资源

validation 实际成本：

| 指标 | 实际 | success plan / physical cap | 结果 |
| --- | ---: | ---: | --- |
| train jobs | 7 | 7 / 8 | pass |
| train steps | 1,376,256,000 | 1,376,256,000 / 1,572,864,000 | pass |
| eval seed-units | 77 | 84 / 91 | pass |
| simulated control steps | 1,380,177,624 | 1,380,574,272 / 1,577,542,128 | pass |
| allocated GPU-hours | 8.5377 | 11.0 / 12.5 | pass |
| provider attempts | 11 | frozen cap 内 | pass |
| provider total units | 319,633 | frozen cap 内 | pass |
| accounted output | 2,346,524,421 B | 3.0 / 3.5 GiB | pass |

success plan 与实际 eval seed-units 的差 7 来自第 3 代没有候选进入 sealed，不是少跑规定阶段。duplicate
parent public-dev 为 0，retry train/eval 均为 0。

## 8. 回归与机器验收

- 产品冻结提交 `66e9c3bb97267663406e91b63f826acd97cc9e12` 在服务器 Torch 环境：1549 tests、
  0 failure、3 skipped、1008.815 s；随后到实验提交只增加文档，冻结的产品字节未变；
- P3B verifier PASS，snapshot hash
  `92b94b749e490e45b8ad9daf70079299174d0827a544b3c930d980b22474b861`；
- Phase 4 evidence snapshot hash：见 `PHASE4_EVIDENCE_SNAPSHOT.json`；
- 本地执行：

```powershell
.\.venv\Scripts\python.exe PLAN\Variant_construction\P4\verify_phase4.py
.\.venv\Scripts\python.exe -m unittest PLAN.Variant_construction.P4.test_verify_phase4
```

- 服务器可追加原始证据复核：

```bash
/data/yuyangliu/envs/isaaclab/bin/python PLAN/Variant_construction/P4/verify_phase4.py \
  --pilot-root /data/yuyangliu/AGENT/true_RL_Agent/outputs/phase4_unified_pilot_5a30129_20260829 \
  --validation-root /data/yuyangliu/AGENT/true_RL_Agent/outputs/phase4_unified_validation_5a30129_20260829
```

该验收器以 39 个冻结产品文件和原始 artifact 字节为入口，不把 Git HEAD 相等当产品身份，所以提交本
报告后仍可复跑。

## 9. Phase GO

`loop_verified=true`：source→verify→train→public-dev selection→public-accept decision→sealed→ledger→
terminal audit 已在 fresh 3-generation session 中真实闭合。

`improvement_observed=true`：第 1 代候选通过 paired objective、三个 guards 与 sealed gate，被接受为
current-best。该结果是观察项，不改变工程 GO 定义；若本次没有提升，系统仍应以 loop 是否闭合作为门。

Phase 4 完成，下一工程 Phase 为 Go2 机器人行资格化；不得把当前 G1 分数、checkpoint 或 family mask
直接复用为 Go2 的能力结论。
