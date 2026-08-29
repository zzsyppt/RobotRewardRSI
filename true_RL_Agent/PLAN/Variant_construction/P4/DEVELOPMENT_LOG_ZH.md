# Phase 4 开发日志

## 1. 目标

在唯一 `G1-37 × unified Stair` cell 上建立 official reference，并以现有共享 generation 内核证明 LLM
只改完整 reward program 时的真实多代闭环。整个阶段没有恢复“地形 cell”轴，也没有把 Stair 的
family、stratum、C2 或失败语义上提到共享层。

## 2. 实施与审计历程

### P4-A：运行前收敛

- 建立 reference-derived floor-only objective family mask，并绑定 Genesis、candidate、selection、
  decision、memory 和 replay；
- provider 只接收 public-dev；public-accept 与 sealed 保持 runner 私有；
- 删除 unified 重复 parent public-dev 评测和对应 argv 手术；
- 将 validation 终态 selector 指标改为从 runner 私有 phase receipts 复算；
- minimum-core 保留目标结构、3 family、8 strata、主要失败、非零 diagnostics 和 3/5/7 seed 数；
- 修复 stratum top-k 表把“被截断非零”写成错误 `0.0` 的真值缺陷，schema v2 以 `null` 表示截断；
- 服务器 commit `66e9c3bb97267663406e91b63f826acd97cc9e12` 完整回归：1549/1549 通过，
  3 skipped。

详细静态证据见 `PHASE4_PRE_REFERENCE_STATIC_AUDIT_ZH.md`。

### P4-B：reference 与 pilot

- reference 从 random 初始化训练 196,608,000 steps；
- C0/C1 readiness 通过，`F_scalar/F_c1/F_c2` 全部达到 floor；
- 冻结 mask：`04ad43edf20c7dfb7749492cc0b7e4fadcb881136444c0ccbc4cd40cb3284a54`；
- pilot output：
  `/data/yuyangliu/AGENT/true_RL_Agent/outputs/phase4_unified_pilot_5a30129_20260829`；
- pilot 恰好闭合一代并接受 `stair_vnext_lateral_course_alignment_v1`；成本和 GPU closure 通过。

### P4-C：fresh validation

- validation output：
  `/data/yuyangliu/AGENT/true_RL_Agent/outputs/phase4_unified_validation_5a30129_20260829`；
- session：`phase4-unified-validation-5a30129-20260829`；
- 恰好运行 3 代、width 2；outcomes 为 accepted / formal_rejected / no_selection；
- 第 1 代接受 `g1_rough_lateral_course_deviation_v1`，第 2 代因 C1-ID guard 拒绝，第 3 代无可选候选；
- normal launcher 和后续 `--terminal-audit-only --resume-existing` 均 return code 0；
- 终态 GPU compute rows 为空，open operation reservation 为 0。

### P4-D：终审

- 人工逐段审计真实 prompt、反馈和多轮记忆，没有发现阻断项；
- 非阻断提升只写入 `PHASE4_FOLLOWUP_IMPROVEMENTS_ZH.md`；
- 建立 `PHASE4_EVIDENCE_SNAPSHOT.json`、reference/acceptance 报告和 `verify_phase4.py`；
- verifier 不绑定当前 HEAD，避免文档提交令验收器自毁；它绑定产品字节、当前 cell 身份、reward
  equivalence receipt 与可选的原始服务器 session artifacts。

## 3. 最终证据索引

- 实施边界：`PHASE4_IMPLEMENTATION_PLAN_ZH.md`；
- pre-reference 静态审计：`PHASE4_PRE_REFERENCE_STATIC_AUDIT_ZH.md`；
- reward 数值等价：`PHASE4_REWARD_EQUIVALENCE_RECEIPT.json`；
- reference 统计：`PHASE4_REFERENCE_REPORT_ZH.md`；
- 总验收：`PHASE4_ACCEPTANCE_REPORT_ZH.md`；
- 冻结快照：`PHASE4_EVIDENCE_SNAPSHOT.json`；
- 后续提升：`PHASE4_FOLLOWUP_IMPROVEMENTS_ZH.md`；
- 机器验收：`verify_phase4.py` 与 `test_verify_phase4.py`。

validation 原始顶层 artifact SHA-256：

- `session_genesis.json`：`09a771de0b14b1666d53f20615e42d9095f6846c489e1c60164a53162d79e05c`；
- `reference_eval_artifact.json`：`8dfdece31028ec90c33c0b6a19dbda56e817282efbd18166eee9c4ca4b2032b8`；
- `ledger.jsonl`：`09fcf7413bfcd37bddc8ef44637cb5ff297e8b2c6599d367a46d70c103b9eaf9`；
- `generation_session_report.json`：`93c605f9b4b75d424808b862d3c9d4a7e06a3458da91d5aff09f39a7409516e5`。

## 4. 保留边界与清理

- 保留 pilot/validation 原始 outputs，它们是验收器引用的证据，不做清理；
- 已删除本地一次性 prompt 审计脚本；没有保留第二套 runner、launcher、memory 或评测套件；
- 历史 outputs 和用户未跟踪文件均未触碰；
- follow-up 仅是下一协议 epoch 的候选改进，不得在当前冻结实验上回填。

## 5. 下一 Phase 交接

Phase 5 只增加 Go2 RobotProfile、runtime adapter、薄 binding 与该行协议配置。复用本 Phase 的共享 LOOP
和 Stair TaskPlugin，不复制 G1 文件，不继承 G1 checkpoint/mask/分数，不向 LLM 开放训练分布、课程、
seed、PPO 或评测语义。
