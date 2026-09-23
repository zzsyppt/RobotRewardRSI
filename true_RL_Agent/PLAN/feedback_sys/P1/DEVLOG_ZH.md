# Phase 1 开发与复核记录

日期：2026-09-23。范围以 [实施卡](./IMPLEMENTATION_PLAN_ZH.md) 为准，结果见 [验收报告](./ACCEPTANCE_ZH.md)。

## 源头修复依据

1. **摘要独立于回执。** 原完整 generation fixture 中，只改 `public_dev.public_summary.score`，其他 hash / 回执不变，结构校验仍通过且公开历史输出假值。修复位于 phase 表示构建与作者消费入口，不向结构验证器塞任务统计公式。live/replay 共用一个构建函数，消费时比对 task adapter 从绑定回执重新生成的摘要。
2. **缺失被补成零。** 原 `RuntimeActivationEvidence.from_mapping(None)` 与实测零完全相同。改为可选测量值；来源 producer 提供它真实知道的声明数和调用数；context 与两个任务反馈使用同一归一化。不把 capability 名称、加载标志当作执行证据。
3. **任务投影修改验证基底。** 原任务 projector 能原地修改传入字典，使“修改前/后”身份比较实际上比较同一个已修改对象。只在所有权交界传副本，既有身份/决策检查继续使用原记录。
4. **同类问题复查：逐 lane 记忆丢状态。** generation 训练回执与任务视图已有每 lane runtime，但 memory 只读顶层 formal 候选。沿既有行追加测量列，避免另建身份映射。旧行补未知；压力摘要与重复压缩保留该列。相应旧 9 列测试更新为“前 9 列不变、末列明确缺测”，不删除其原有断言。

保留观察 / 任务筛选结果 / 接受决定的区别。没有实现新的根因分类器，也没有把统计共现升级为因果结论。Codex 的适用思想和本地源码锚点见 [参考取舍](./CODEX_REFERENCE_ZH.md)。

## 实施中的有界调整

- P0 已完成的一次性 prompt prepare / provider probe runner 直接删除，保留原请求和回执，由现有 verifier 校验；没有新写一套“冻结模式”脚手架。
- P0 historical source baseline 不再作为“所有未来生产代码必须不变”的门；保留原始快照身份，当前源码差异仅报告，本阶段运行自己的测试。
- 先用单轮运行状态反例修复，再审阅主线 generation 的逐 lane 路径，补第 10 列。验收目标没有放宽。
- 本地 producer 测试 fixture 初次缺必填参数并使用了 adapter 不支持的 NumPy 返回容器；修正为正常的带权 RewardTerm 和 Python 数值列表。没有为测试扩展生产容器支持。
- 集成 fixture 的 reference 从不存在的占位路径改为生产 journal 发布的本地文件；第 10 列增加后，原 `not_started` 全行期望同步保留缺测。最终完整模块重新通过。
- P0 小样本 token 拟合保留为基线测量，不直接作为所有模型的生产估计器上线。本次保留既有容量与输出预留合同；完整上下文预算改进由 Phase 5 依据实际 provider 再推进。
- 最后复核普通 width-one round：提交时已有完整 execution receipt 比对，但该旧路径的 ledger 未保存可供重放引用的 final receipt；重放信任已提交 ledger 的公开结果。保留该既有信任层，不扩建旧 schema 的认证框架。验收明确区分它与当前 generation 的 phase 回执重建，不把通用 fixture 或旧 round 回归包装成逐字段认证。

## 可复跑命令

工作目录：仓库 `true_RL_Agent`。下面使用本地 `.venv` Python，不调用 API 或模拟器。

```powershell
.venv\Scripts\python.exe -m unittest tests.unit.test_acceptance_replay tests.unit.test_generation_receipts tests.unit.test_generation_historical_wire tests.unit.test_generation_finalization tests.unit.test_generation_early_rejection tests.unit.test_generation_contract tests.unit.test_prompt_projection_and_mechanism tests.unit.test_memory_context_phase13 tests.unit.test_push_feedback_projection tests.unit.test_stair_public_projection tests.unit.test_task_prompt_adapter tests.unit.test_prompt_capacity_projection tests.unit.test_prompt_safety_boundaries tests.unit.test_eval_public_projection tests.unit.test_runtime_evidence_projection tests.unit.test_feedback_evidence_boundary tests.unit.test_push_recovery_runtime_adapter tests.unit.test_stair_complete_reward_contract -v
.venv\Scripts\python.exe -m unittest tests.integration.test_fixed_width_generation -q
.venv\Scripts\python.exe -m unittest tests.integration.test_fake_round tests.integration.test_round_public_boundaries tests.integration.test_live_loop -v
.venv\Scripts\python.exe PLAN/feedback_sys/P0/verify_p0.py
git diff --check
```

源代码与测试执行在正常目录；P1 只保留实施卡、参考说明、验收记录和日志。补充长批次未完成，不充作已通过证据；独立审阅中的重复测试也不累计增加 265 的数量。

## P1独立审计后的修订

原审计指出的6/7代保留退化属实。本轮将完整测量与长期索引分开：`RuntimeActivationEvidence.reward_execution_state()`从原计量生成短状态；generation仅保留逐lane第10列，删除顶层重复；普通round也采用短状态。完整计数的生产者、回执和详细runtime反馈不变。

分类保留 `unknown` 与 `executed_value_unknown`：只有加载、诊断记录或单独false不能推出奖励项已执行；正调用数且实测贡献false才为 `executed_zero`。重压缩支持已有9列和10列测量字典，保留前9列原值并幂等输出状态，不建立迁移服务或新持久协议。语义串明确available只表示存在记录、null仍未测。

增加正常测试fixture `tests/fixtures/feedback_memory_baseline.json` 与 `test_feedback_memory_retention.py`，由审计公开payload和对应公开runtime提取，保存源hash；测试只读自身fixture，不读取PLAN。门槛在真实新回放前登记于实施卡。复跑上方原18模块与两个集成批次，再增加：

```powershell
.venv\Scripts\python.exe -m unittest tests.unit.test_feedback_memory_retention -v
```

未安装项目时显式使用 `PYTHONPATH=src`（Linux示例：`PYTHONPATH=src python -m unittest tests.unit.test_feedback_memory_retention -v`）。本轮新日志使用 `verification_audit_*.log`；原日志与原 `acceptance_checks.json` 保留为首次验收证据。本轮实际结果和真实会话回放见 [审计处置](./AUDIT_RESPONSE_ZH.md)。Stair别名在现有多个合同出口中保留，P2演化与路径边界登记在范围卡，没有为小项增加兼容框架。

最终267项当前版本测试通过。另在服务器scratch `/data/yuyangliu/p1_retention_fix_20260923_01` 从fe96455d建立baseline/current两个源码副本，current仅覆盖10个生产文件；沿原会话绝对路径，用未修改的旧审计驱动full模式完整重放，再构建Stair第13/16代、Push第6代前的最终payload各两次。完整参数和四份报告hash在[run_summary](./retention_regression/run_summary.json)，同步指纹在[sync_manifest](./retention_regression/sync_manifest.json)。结果为12/13/5代，phase相等、确定性、摘要篡改拒绝及保留内容对照均通过。只导出小报告，不复制新整份payload；0模型调用、0新训练、0模拟器运行。
