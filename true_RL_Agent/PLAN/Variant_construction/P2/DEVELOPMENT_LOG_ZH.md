# Phase 2 开发日志：Stair 任务语义与机器人形态解耦

> 日期：2026-08-26  
> 最终验收：2026-08-27  
> 实施入口：`5513076d5cbfe279987310c12c89938ff1a7691e`（仅作捕获 provenance）  
> 工作范围：Stair task-owned runtime/reward/evidence 形态投影与 G1-37 legacy 兼容

## 1. 实施前重新冻结的边界

本阶段没有新建机器人系统，也没有进入 Go2、C1/C2 地形或新 LOOP 权限。目标只是在现有矩阵主线内建立以下唯一数据链：

```text
ResolvedCellManifest
  -> StairRuntimeProjection
  -> morphology-neutral reward/evidence
  -> Stair task-owned success/failure/surface/evaluation

legacy G1-37:
  VNext snapshot -> G1-37 reward adapter -> frozen V2 reward program
  frozen V2 transition -> legacy reader -> generic decoder/tracker
```

公共 `robots/` 与 composition 只增加通用的“必须选择 profile 为某 role 声明的全部 targets”约束，不解释 support、楼梯或 surface。Stair 自己继续拥有成功/失败判据、surface query、sensor 投影与兼容读取。

## 2. 实施记录

1. 将 Stair 的 `support_effector` requirement 从固定 `(2, 2)` 改为 `minimum=1 / maximum=None / require_all_profile_targets=true`；compiler 对漏选和乱序选择给出稳定的 `role_selection_incomplete`，同时继续允许 base 与 forbidden-contact 复用同一刚体、拒绝 support 与 forbidden-contact 冲突。
2. 新增 `StairRuntimeProjection`。它只从 exact profile/adapter/binding 解析显式 kinematic-base source、logical roles、joint/action schema、contact/height sensor 和内容身份；实际 Isaac joint/body/contact-sensor ID 在环境构造时集中解析并写回 summary。manifest hash 作为 provenance 单独报告，不参与 effective projection hash。
3. 新增 morphology-neutral `StairRewardInputsVNext`。support/joint/action/forbidden-contact/history 轴来自 projection；字段表不含 ankle/arm/finger/torso/foot 等形态假设，不同 2/4/6 support 产生不同 schema 与 capability hash。
4. 把 G1-37 的 joint-group 解释集中到 `legacy_reward_adapter.py`。生产 builder 先形成 VNext snapshot，再由该 adapter 映射到冻结的 42-field `StairRewardInputsV2`；通用 builder 不再写任何 G1 link/joint 常量。
5. recorder、decoder、`StairEnvTransition` 与 tracker 统一改为变长 `support_*` / `forbidden_contact_*` 数据。旧 transition V2 字节不改，由 `legacy_transition_reader.py` 读取；VNext payload 自带 projection/schema/count 身份。
6. 新增 task-owned `StairSurfaceQuery` 与当前 equal-height flat 实现。decoder 只调用 query，不把 future rough mesh 逻辑塞进公共层；C1/C2 实现仍留在 Phase 3A/3B。
7. train/eval child 重载的 resolved manifest 现在显式进入 Stair execution，runtime projection 继续传给 reward builder、recorder 和 decoder；没有 manifest 的历史测试入口仅从当前 G1 profile + legacy binding 构造兼容 projection。
8. TaskSpec 与 LLM task section 的展示名从 `Unitree G1 Stair Climbing` 改为中性的 `Stair Climbing`。`unitree_stair_climbing` 暂时只作为历史兼容 wire ID 保留，不再承载形态判断。
9. 对 non-legacy projection 增加一条必要的 schema 门：只有 qualified reward program 的 input schema 与 VNext projection 完全一致才可安装。当前 Phase 2 不伪装已有 V2 program 已支持新机器人；对应 authoring/qualification 接线属于首个 non-legacy cell 的 Phase 3A/5 责任。
10. `RobotTaskBinding.task_projection` 新增显式 `kinematic_base.source`。G1-37 clean 绑定冻结为 `articulation_root`；reward 与 recorder 统一调用同一 frame reader，legacy adapter 只选择 ABI，不再选择物理刚体。
11. 活跃 evaluation semantics 升级为 morphology-neutral v7：`forbidden_contact`、`support_effector`、`kinematic_base` 替代 torso/ankle/foot 描述。v6 文件原样保留为历史身份，数值阈值与优先级次序不变。
12. Stair catalog 在 manifest 中写入 `reward_input_vnext_schema_hash` 与 `reward_capability_vnext_hash`，Stair-owned compiler wrapper 对缺失/不匹配做必选门；共享 compiler 未增加任务语义。

## 3. 实施中发现并根治的问题

### 3.1 动态测试曾把错误吞成成功

原 Isaac unittest 在 `SimulationApp.close()` 时会因冻结主机缺少可选 `libGLU.so.1` 提前结束，导致 recorder 用例已经报 `ERROR`，shell 仍得到 `RC=0`。真实 traceback 显示测试错误地用 280-env private assignment 构造只允许 56 episodes/seed 的 `public_dev` summary，属于原动态测试自身的考卷规模混用。

修复后测试直接按 `public_dev.episodes_per_seed_per_bin` 构造 7-bin assignment；动态验收使用手动 test runner 在 Kit 关闭前打印每个用例结果并以 `os._exit` 保留真实退出码。两项用例均明确 `MANUAL_PASS`。

### 3.2 临时内容覆盖目录没有 `.git`

首次真实 runtime probe 已完成环境 step、recorder 和 reward evaluation，但在记录 `git rev-parse HEAD` 时失败。根修不是伪造仓库，而是把 HEAD 明确降为可显式传入的 capture provenance；当前产品有效性由 normalized-LF source closure、manifest/projection identity 与 canonical receipt hash 验证。

### 3.3 旧展示名仍泄漏 G1

边界复扫发现 plugin/prompt 的展示名会在未来 Go2 cell 中给 LLM 错误形态信息。它们现已改为通用任务名；没有批量改写历史 task ID、protocol 或 evidence。

### 3.4 审计 A–F 的根因收敛

- A 真实存在：旧分支取 articulation root，VNext 曾取 `base=torso_link`。修复后物理 frame 由 binding 明示，adapter ID 不再参与选择。服务器同帧 exact parity 覆盖 7 组量；没有采用“所有机器人 body pose 必须等于 articulation root”的错误通用断言。
- B 真实存在：v6 validator 与公开 taxonomy 写死 torso/ankle/foot。活跃 v7 改为任务逻辑词汇，旧 v6 只保留为历史字节。
- C 真实存在：projection hash 曾混入 manifest 是否存在。现在 hash 只覆盖有效投影内容，manifest provenance 单独报告；同一 cell 两种入口 hash 相同。
- D 真实存在：8 个字面量黑名单只能防回归。verifier 现从 `RobotRegistry` 自动收集全部 body/joint 名并扫描通用文件 AST 字符串；审计给出的三个注入名会被稳定捕获，未引入例外名单。
- E 真实存在：两个 VNext hash 原为可选检查。catalog 现在必写，Stair compiler 与 runtime projection 必验；任务语义没有下沉共享 manifest compiler。
- F 真实存在但无需重写历史 V1：通用 decoder 现从 `joint_pos` runtime tensor 推导宽度并与 projection 交叉校验。V1 G1 schema 仅由 legacy adapter/reader 消费；VNext TaskSpec 的彻底移除责任留给 Phase 3A。

同类复扫另外发现 P2 verifier 的 `emit_current()` 在无 `.git` 归档中仍强制读取 HEAD，虽 HEAD 只标注 provenance。现改为可空记录；产品身份仍只由内容闭包与 canonical receipt 决定。

## 4. 身份变化与历史边界

Phase 2 有意改变 Stair role requirement，因此当前 legacy cell 的组合身份从 Phase 1 冻结值迁移为：

- legacy grant hash：`b1000b648ecbb9574a3da18a82b8bbb9cc8530542ad1999f2800c2ac8cc76fbc`；
- manifest hash：`4f3cba01f11abcfadea7f262df9a1001a217c0654e8ff050a4b38d1dc4a6b3ca`；
- resolved cell task protocol hash：`7cd355c164d6aa3aad7f2aa89e76a8c67351f1f3a3d0d12bcbdb61a64e9ed963`；
- runtime projection hash：`69405038c92b4a679cd58205a95d003f22af7cb174a079eb574111fe5e6367ff`；
- evaluation semantics v7 hash：`df35aa0fb8c24983570e2c634fd0cc40f8d39beefc191d400c37f099aa8cb766`。

因此 P1 snapshot verifier 会按设计因受保护产品树/组合身份改变而失败；P2 当前入口改为 `verify_phase2_snapshot.py`，不能回写 P1 历史快照。旧 artifact 仍可在其冻结源码/身份下审计，但不能被当前 manifest 冒充续跑。

旧 G1 执行行为保持：observation `[1,310]`、action `[1,37]`、transition V2、42-field reward ABI、15 个官方 reference terms、35-cell 与 evaluation 数值阈值均未改变。公开形态词汇迁移到 v7，因此当前 task identity 按设计改变，历史 v6 不被回写。

## 5. 动态验收

隔离目录：`/data/yuyangliu/true_rl_agent_phase2_followup_PQJYOl`（验收后已删除）；服务器主 checkout 未修改。真实 RTX 5090 / `cuda:0` 单环境 probe 完成：

- robot logical base body ID `[4]`，support body IDs `[24,25]`；
- contact-sensor support IDs `[6,13]`，forbidden-contact ID `[14]`；
- 37 个 joint IDs 全部按 profile 顺序解析；
- recorder 生成 frozen transition V2，并成功规范化成 variable-support payload；
- 通用 VNext snapshot 经 legacy adapter 得到 42-field V2，官方 15-term reference reward 输出 finite；
- binding 明示 `articulation_root`；只移除 legacy adapter 标识后，pos/quat、世界/体坐标线角速度与 projected gravity 的最大绝对差全部为 `0`；
- runtime receipt hash：`bea2c4678e82c10e2a220ab12077f3238979b2d5f4b5527834a4f2d34c963e26`；
- probe 结束后服务器无遗留 GPU compute process。

最终冻结的 29 文件 product-tree hash 为 `148984daa4fa917f104bcb8a65bce348fc2c7760d472f24b11df1cb356bc4a11`，P2 snapshot receipt hash 为 `077a3dc3a6c9113492ca90b92a8379f6cc65a2c29b45010ae45484b36dab8654`。服务器静态契约 42/42、snapshot verifier 与 5 个 verifier 自测均通过。

最终稳定字节上的全仓回归为 1505 tests 全部通过、29 skipped、987.665 s、exit code 0。测试过程打印的 synthetic child failure、无效 GPU lease、publication failure、backend/source drift 均来自预期否定用例；unittest 最终汇总为 `OK`。

## 6. 验收命令

```powershell
$env:PYTHONPATH='src'
uv run python PLAN\Variant_construction\P2\verify_phase2_snapshot.py
uv run python -m unittest discover -s PLAN\Variant_construction\P2 -p 'test_verify_phase2_snapshot.py' -q
uv run python -m unittest -q tests.unit.test_stair_morphology_projection tests.integration.test_stair_evaluation_runtime_pipeline tests.unit.test_robot_matrix_contract tests.unit.test_stair_episode_evidence tests.unit.test_stair_terrain_evidence tests.unit.test_stair_complete_reward_contract tests.unit.test_stair_transition_materialization tests.unit.test_stair_reward_cold_start
uv run python -m unittest discover -s tests -p 'test_*.py' -q
```

最终测试数量、snapshot hash 与完整 GO/NO-GO 判定见同目录 `PHASE2_ACCEPTANCE_REPORT_ZH.md`。

## 7. 留给后续 Phase 的明确责任

- Phase 3A：显式非矩形 evaluation suite、C1 surface query、resolved task/eval binding，并为首个 non-legacy G1 cell 冻结 VNext reward program/prompt qualification；VNext TaskSpec/EvaluationBinding 不得再携带或读取历史 `StairProtocolV1.policy_compatibility.robot_schema`；
- Phase 3B：rough mesh local-surface query 与 failure-classification drift 校准；
- Phase 5：Go2 RobotProfile/RuntimeAdapter/cell-owned EnvCfg 与 agent config、Go2 official rough reference reward 的 VNext 移植；
- 启用任何 non-legacy LOOP 前，必须让其 prompt contract 与 qualified runtime contract 使用同一个 projection-derived VNext schema hash。

本阶段没有增加通用 TerrainPlugin、第二套 LOOP/runner、数据库、服务或宽泛防御框架。
