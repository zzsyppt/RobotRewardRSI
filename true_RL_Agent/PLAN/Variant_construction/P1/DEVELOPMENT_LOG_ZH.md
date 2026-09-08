# Phase 1 开发日志

> 日期：2026-08-26  
> 目标：完成机器人 × 任务矩阵的最小组合契约，并把 Stair G1-37 / Push G1-29 两个 legacy cells 接回现有 LLM LOOP

## 1. 实施前复核

- 重读主计划 Phase 1、Phase 0 baseline/runtime receipt/ownership/hardcode/audit 全部产物；
- 在任何产品修改前复跑 Phase 0 入口验收，确认提交 P0 文档后的新 HEAD 不再导致自失效；
- 使用 CodeGraph 从 `ResolvedTaskComposition`、两个 composition provider、`TrainingBinding`、Isaac backend 和四个 child 入口向下审计真实调用链；
- 复核 `D:\humanoid_agent\codex` 当前本地提交 `5ce04789`，只采用严格 manifest 解析、稳定 ID 唯一 registry、descriptor 与 execution 分离三点；未引入插件安装、权限沙箱、远程同步或服务管理框架；
- 形成 `PHASE1_IMPLEMENTATION_PLAN_ZH.md`，冻结“不做 Go2/C1/C2/Phase 2 runtime 泛化”的范围。

## 2. 关键架构决策

### 2.1 不建立第二套 RobotPlugin 系统

机器人公共层只放：

- 完整运行时/资产事实；
- `base`、`support_effector`、`forbidden_contact`、`joint_group.*` 共享角色；
- backend/runtime adapter descriptor；
- task-neutral capabilities。

TaskPlugin 只声明 role cardinality 与 capability requirements。具体 scanner mount、噪声、action scale、policy observation、reward/eval/acceptance 和地形仍由 task binding / TaskPlugin 拥有。
role target 不做全局唯一：`base` 与 `forbidden_contact` 可以合法重用躯干；公共层只声明
`support_effector × forbidden_contact` 互斥。

### 2.2 legacy V1 与未来 V2 只有一个显式分界

- `stair.g1_37dof.clean_legacy_v1` 与 `push.g1_29dof.legacy_v1` 通过 exact content-bound grant 保持 `TrainingBindingV1`；
- reader 同时严格支持 V1/V2；
- 非 legacy manifest 由同一 backend 自动把 `resolved_cell_task_protocol_hash` 写入 TrainingBinding V2；
- 不增加第三种 binding 模式，不用修改历史 hash 换取形式统一。

### 2.3 child 回传是运行证据，不信父标签

父进程 materialize canonical manifest 并传绝对 path/hash；child 在 App 创建前重新读取、校验 canonical bytes、manifest hash、task/base protocol，然后在所有成功/失败 summary 回传 cell identity。通用 runner 再与磁盘 manifest 对照；checkpoint 继承 cell hash，跨 cell mismatch 在启动前拒绝。

## 3. 实施记录

1. 新增 `robots/base.py`、`registry.py`、`defaults.py`，并从 Phase 0 receipt 机械生成两份 canonical profile JSON；
2. 新增 `composition/cell_manifest.py`，实现 binding/environment/catalog/compiler/manifest 及严格 child loader；
3. 机械生成两份 legacy task binding JSON，完整嵌入对应 Phase 0 cell binding facts；
4. Stair/Push TaskPlugin 增加仅使用共享词汇的 `robot_requirements()`；各任务新增自己的 `robot_binding.py`；
5. 两个 provider 增加闭集 `cell_variant_id`，解析并 materialize manifest，再附着到原 `ResolvedTaskComposition`；
6. runtime composition snapshot 在有 matrix cell 时升级为 V2，绑定 manifest hash 与 resolved cell protocol；旧的无 manifest snapshot reader 保持 V1；
7. 原 Isaac backend 统一传递 manifest；四个 child 在 Isaac App 创建前加载；SubprocessRunner 统一验证 summary/checkpoint cell identity；
8. `TrainingBinding` 增加严格 V2 schema 与外部 task-protocol 对照，V1 schema 常量和序列化保持兼容；
9. 新增 13 个聚合测试场景，没有为 role/capability 的所有排列建立大规模测试矩阵。

## 4. 实施中发现并在源头修正的问题

### 4.1 cell ID 一度只在 provider 输入中，未进入 task defaults

高层 composition 回归发现 `provider.cli_defaults()` 与冻结 `composition.task_defaults` 不再完全一致。根因是 cell ID 被当作临时选择值。修正为：resolved cell ID 同时进入 composition snapshot，确保 CLI、runtime 与 Genesis 使用同一份已验证输入。

### 4.2 base task protocol 与 effective cell protocol 不能复用一个 argv 字段

如果未来非 legacy V2 使用 `--task-protocol-hash` 表示 resolved protocol，同时 child 又用它校验 manifest 的 base protocol，会形成语义冲突。根修为两个明确字段：

- `--base-task-protocol-hash`：只用于 manifest 重新加载；
- `--task-protocol-hash`：训练 V2 / evaluation binding 的有效比较域。

两个 legacy cells 的后者仍是原 task protocol，不改变历史身份。

### 4.3 Push legacy 占位 schema 标签不能直接替换

把 `push_recovery_obs_v1` / `unitree_actions_v1` 直接换成新内容 hash 会破坏 legacy job/baseline 身份。处理方式不是打兼容分支，而是在新 manifest 的 contract hashes 中同时加入 P0 `policy_io_schema_hash`、observation/action content hash；旧标签只作为 legacy backend 兼容字段保留。首个 non-legacy Push cell（Extension A：Go2 × Push）必须改用 manifest observation/action content hash，未完成前不得运行。

### 4.4 dry-run 产物不应成为长期脚手架

共享 LOOP dry-run 完成后，保留 audit summary、ledger、resolved manifest、session Genesis，以及被 summary/ledger 直接引用的 candidate、round 与 trial 原始回执（共 24 个文件、约 183 KB）。只把运行锁移到项目 `.tmp`；此前的试跑仍留在 `.tmp`，没有混入 P1 权威交付。

### 4.5 公共 compiler 不应理解 Phase 0 迁移字段

初版公共 `RobotTaskBinding` 会专门复算 `phase0_binding_facts`。这会把一次迁移过程固化成所有未来任务都要理解的公共语义。根修为：公共层只把 `task_projection` 视为 task-owned opaque content；Stair/Push 配置与 Phase 0 外部回执的逐字段一致性由验收测试负责。

### 4.6 legacy 辅助函数兼容不能依赖完整 CLI Namespace

首轮全仓回归发现两个现有 Stair 单测直接调用 `_status_summary`，传入的精简 Namespace 没有新字段。根修为 summary helper 通过可选 manifest 生成公共字段；真实 job 仍由 runner 强制要求 child 回传，不把测试对象补成假完整 CLI。修正后旧调用与新运行契约同时成立。

### 4.7 V2 与 child/checkpoint 边界必须在首个复杂 cell 前闭合

最终边界审计进一步修正三项同源问题：V2 的 `task_protocol_hash=null` 现在由严格 reader 拒绝；manifest path/hash 任一单独出现都会在 App 创建前失败；非 legacy checkpoint 缺少或错配 cell hash 均在 child 启动前失败。legacy V1 checkpoint 可继续读取没有该新字段的历史元数据。曾尝试在 backend 构造期再比对 base protocol 与 evaluation binding，但全仓回归证明这会抢占既有 baseline-eval 权威拒绝点，因此删除该重复防御，继续由 composition、现有 eval authority 与 child manifest 校验负责。

### 4.8 B1–B7 独立审计后的源头修正

- B1：确认全局 role-target 唯一规则错误。公共层改为最小互斥对；Stair G1-37 profile/binding 正式加入
  `forbidden_contact=torso_link`，`base=torso_link` 同时合法；support/forbidden 重叠仍在 composition 拒绝；
- B2：确认 Push 从未消费逐足支撑语义。Push requirements/binding 只保留 `base`，删除
  `support_effector=(2,2)` 与 `support_contact` 要求，因此 Go2 × Push 不再被虚假双足门阻塞；
- B3：Stair 的 `(2,2)` 仍与当前双足 runtime 一致，不能在 decoder 泛化前放宽；主计划 Phase 2 已把
  `StairClimbingTaskPlugin.robot_requirements()` 列为同一批必须迁移的债务；
- B4：Phase 1 的两个 legacy EvaluationBinding 必须保留 base hash，否则历史 comparison identity 改写；主计划
  已改为 Phase 3A 首个 non-legacy cell 在建 Genesis 前把 resolved hash 同时写入 effective TaskSpec、
  EvaluationBinding 与 TrainingBindingV2；
- B5：删除可随手复制的 bool，改为绑定 profile/adapter/binding/environment/requirements/base protocol 的
  grant hash；cell 任一输入变化而未重新授权时确定拒绝，其他 cell 默认 V2；
- B6：G1-29 `articulation_source` 改为真实资产符号
  `unitree_rl_lab.assets.robots.unitree:UNITREE_G1_29DOF_CFG`，task env module 只留在 Push task-owned runtime；
- B7：legacy 占位 policy-I/O 标签继续只为历史身份保留；清除执行路径的责任与验收门已明确归 Extension A，
  Go2 × Push 必须直接消费 manifest content hashes。

### 4.9 P1 回执可复验性与后续边界审计

- 确认原 24 个 LOOP 证据文件会受 Git 行尾转换影响，这是阻塞项。仅把外层
  `evidence_closure_hash` 改成解析后 canonical hash 不能根治：sealed candidate manifest 还保存了
  `components.py`、`candidate_manifest.yaml` 的原始字节 hash 与大小；外层通过后内层身份仍会失效。
- 根修为在 `.gitattributes` 对小型冻结目录 `PUSH_SHARED_LOOP_DRY_RUN/**` 使用 `-text -eol`，保留证据原始字节；
  新增永久 `verify_phase1_snapshot.py`：冻结产品树只校验内部闭包，当前门精确复算两份 legacy G1 profiles 与
  两个 legacy cell 的机器人/任务轴，并复算 24 文件闭包、Git 属性、sealed candidate 内部源码身份及
  frozen/current Push cell axes。这样新增机器人行或后续任务协议演进不会注销 P1 历史证据。自测用临时 Git 仓库设置 `core.autocrlf=true`，真实 add +
  checkout 后 24/24 字节相同，另有语义篡改拒绝测试。收尾时把 29 个产品文件与 24 个证据文件一起做同样的
  全新检出探针，53/53 均保持原字节；产品文件原有 LF 规则有效，无需增加更广的行尾策略。
- 确认 P1 原先缺少自动验收器；现已与 Phase 0 一样提供可重复 CLI，而不是依赖人工复算。
- 确认 cell resolve 中 registry 曾加载两次。Push/Stair 现在各自只构建一次 registry，并在 grant/compile 间复用；
  不引入缓存或全局生命周期。
- Go2 thigh/calf 是 Phase 5 的真实前向约束，不应提前伪造 Go2 profile。主计划已要求未来 Go2 profile 的
  `forbidden_contact` 候选全集覆盖 P0 已确认的 4 个 thigh 与 4 个 calf body，具体禁碰子集仍由 task-owned
  cell binding 选择。

## 5. 验收命令

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe -m unittest -v tests.unit.test_robot_matrix_contract
.\.venv\Scripts\python.exe -m unittest -v PLAN.Variant_construction.P1.test_verify_phase1_snapshot
.\.venv\Scripts\python.exe PLAN\Variant_construction\P1\verify_phase1_snapshot.py
.\.venv\Scripts\python.exe -m unittest -q `
  tests.unit.test_phase_a_contracts `
  tests.unit.test_isaac_lab_backend `
  tests.unit.test_push_recovery_task `
  tests.unit.test_materialize_job_script `
  tests.unit.test_subprocess_runner `
  tests.unit.test_composition `
  tests.unit.test_session_genesis `
  tests.integration.test_session_genesis_live_loop `
  tests.integration.test_dry_run_audit_loop `
  tests.unit.test_robot_matrix_contract
.\.venv\Scripts\python.exe scripts/run_loop.py `
  --task-id unitree_push_recovery `
  --mode dry-run-audit `
  --rounds 1 `
  --output-dir PLAN/Variant_construction/P1/PUSH_SHARED_LOOP_DRY_RUN
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
uvx ruff check --select F,E9 <本阶段产品与测试文件>
git diff --check
```

最终结果：矩阵契约 13/13；Phase 1 自动验收器 3/3（含 Git 全新检出往返）；相关回归 237/237（1 skipped，92.733 秒）；全仓 1496/1496（29 skipped，823.073 秒）；compileall、Ruff F/E9、`git diff --check` 全部通过。机器回执冻结 29 个产品/测试文件，产品树 hash 为 `ef3b2c3316e041e0380b987e180f6cfe40f20051ef73aa1230dcced069a1a711`，receipt hash 为 `59eb454bdcd6c504d887dda38a71dca3643e3e4d1d532861d95395afa55732d3`。B1–B7 修复后的 Push 共享 LOOP 证据已冻结为字节稳定目录；运行锁和测试在仓库 `outputs/` 根生成的单个 manifest 均已移动到项目 `.tmp`，未删除且未触碰其他历史输出。

## 6. 当前范围声明

- 没有修改 AgentCore、CandidateAuthoringLoop、decision 或 ledger 抽象；
- 没有实现 Go2 cell、复杂楼梯、roughness、变长 support evidence 或通用 TerrainPlugin；
- 没有改变 Stair/Push reward、评测阈值、预算、seed、acceptance 或 LLM 权限；
- 没有要求 LLM 候选必须观察到性能提升；
- 下一阶段只应按主计划进入 Phase 2 的 Stair runtime 形态语义解耦。
