# Phase 1 验收报告：矩阵组合契约、共享角色词汇与两个 legacy G1 profiles

> 验收日期：2026-08-26  
> 实施入口：`4392aa2a6df2a45c7dd09456df35a8788ef41f6d`  
> 结论：**Phase 1 GO**；仅代表机器人轴与两个 legacy cells 完成接入，不代表 Go2、形态无关 Stair runtime、C1/C2 地形或策略性能改进已经完成

## 1. 交付结果

Phase 1 已把当前系统中的机器人执行身份从 task composition 中干净抽出，并保持原 TaskPlugin、AgentCore、runner、decision 与 ledger 主线不分叉：

```text
RobotProfile + RobotRuntimeAdapter
              × TaskPlugin.robot_requirements()
              × task-owned RobotTaskBinding
              × task-owned CellEnvironmentDescriptor
              -> shared compile_cell_manifest()
              -> ResolvedCellManifestV1
              -> 原 ResolvedTaskComposition / LOOP / runner / ledger
```

公共层新增的职责只有：精确机器人事实、共享角色/能力词汇、唯一 registry、确定性组合和不可变 cell identity。楼梯、推倒、scanner、噪声、action scale、reward/eval/acceptance 与环境定义仍在任务侧。

主要产品交付：

- `src/true_rl_agent/robots/`：`RobotProfile`、`RobotRuntimeAdapter` 与唯一 registry；
- `src/true_rl_agent/composition/cell_manifest.py`：binding、cell catalog、capability compiler 与 resolved manifest；
- `configs/robots/*.json`：G1-37 Isaac Lab 与 G1-29 Unitree RL Lab 两份精确 profile；
- `configs/tasks/*legacy_binding_v1.json`：Stair/Push 两份 task-owned legacy binding；
- 两个 task provider 的 cell catalog/解析入口，以及原 backend、四个 train/eval child 和通用 runner 的 manifest 传递；
- `TrainingBindingV2(task_protocol_hash)` 严格 reader/writer，同时保持 legacy V1 字节身份与回放；
- `tests/unit/test_robot_matrix_contract.py`：只覆盖本阶段边界所需的 13 个聚合验收场景；
- `verify_phase1_snapshot.py` 与 3 个自测：自动复算产品树、两个 profile/cell、Push LOOP 证据闭包和 sealed candidate 内部身份，并验证 Windows/Linux 检出不会改写冻结证据字节。

## 2. 冻结身份

### 2.1 机器人执行体

| Profile | DoF / body | Phase 0 runtime facts hash | RobotProfile hash | RuntimeAdapter hash | asset closure hash |
|---|---:|---|---|---|---|
| `unitree_g1_37dof_isaaclab_v1` | 37 / 44 | `93adedbe9022a62d19fda14057775bd39e15705757d18287e494ce18855c16c2` | `d23a5ffdca4aae1845ee37947653cefd4a21e96eeffde029cd319864014b7d18` | `d0fbd7e34deb7f8265cf0b6ad969cdfa2c3b059f068d1564d7d9ec5f9056557c` | `660130ec88b6d500cfa7d9528d67c1450504d07ef3efff8b911cb4c2c2648cc7` |
| `unitree_g1_29dof_unitree_rl_lab_v1` | 29 / 35 | `9038bdfdb8e5061adaa7bd2902fa94b65e124f73f76701f0085a155c2c678fbc` | `fea0020490fd7bfbed6136eb0e8b68cb16fb1d0050f67aeb8df28321e34c0ba8` | `610ca42773eaf5fe6c449796735855520919362f707df8819ae01ec42f436f1b` | `28742f449fef956840f289178834c9401a6b4a19a76fb006fd0402da3a4f10d4` |

两份 profile 使用相同 schema、role vocabulary 和 compiler，但没有把 `robot_family_id=unitree_g1` 当作可交换身份。完整 joint/body、默认姿态、形态尺度、actuator/PD 和资产闭包均与 Phase 0 回执逐字段相等。

### 2.2 两个 legacy cells

| cell variant | binding hash | legacy V1 grant hash | manifest hash | resolved cell task protocol hash |
|---|---|---|---|---|
| `stair.g1_37dof.clean_legacy_v1` | `1054a912270cee9a098064c1763e81245a94f071984fa5fa11b8cff2633cf379` | `6e5b5ce4c54264f4b34a1558858085cdca2b70efc4abcdcf28a0e2fc71b40c65` | `373a6afd63ce432ee2488211f43406a825a84fc695cd53978bf1ea9705d85d4e` | `42c69428ccb9c8f04f6a87d4c7eee31fcf5746a8d4839a6d499db74859d5863e` |
| `push.g1_29dof.legacy_v1` | `25acca996a57bd833912e320d3f4e970b59c1945ad48202499a26c2c8c6065fc` | `6cd183e4faba7141967f183d4786be7e7232ae7ab08bea8375e5d4e1775da856` | `f7835d62b6ebb384fa1cd1266bfa29d1d0bac4f8758a0664a654f1d6e133ae95` | `35258c76dc7ad0ccda07cad149a0a5facf0f2560dd3746e7caf13e3a41dd9346` |

manifest 同时保留 `base_task_protocol_hash` 与派生的 `resolved_cell_task_protocol_hash`。两个 legacy cells 通过
`legacy_training_binding_v1_grant_hash` 绑定 exact profile/adapter/binding/environment/requirements/base protocol；
普通 bool 降级入口已经删除。新生成的 legacy job 传递 manifest，但原正式 Stair training binding hash
`2a469d700e2cff853d525e720173483381139f0901852ac031c681c0192fc2b2` 未被重写。

## 3. 严格验收门

| 门 | 结果 | 实际证据 |
|---|---|---|
| P1-G1 确定性 | PASS | Stair/Push 各在两个全新 Python 进程中生成 byte-identical canonical manifest；profile、base task protocol、binding、environment、observation contract 任一变化都改变 manifest 与 resolved protocol hash |
| P1-G2 真实 profile | PASS | 两份 profile 逐字段等于 Phase 0 runtime receipt；37/29 DoF、上游 revision、完整 asset closure、profile/adapter hash 均不同；scanner/policy projection 不在 profile |
| P1-G3 compiler | PASS | 正常两格为 `supported`；互换 G1-37/G1-29 得到稳定拒绝；缺 role、缺 capability、互斥 role 均为排序稳定的 `unsupported + reasons`；`base + forbidden_contact` 同 torso 合法，`support_effector + forbidden_contact` 同 target 拒绝；重复 capability 声明作为非法 schema 在 compiler 前被严格 loader 拒绝；不 import Isaac |
| P1-G4 legacy parity | PASS | 现有 composition/backend/reference/Genesis/TrainingBinding V1 测试通过；Stair 正式 V1 binding hash 保持；Push 删除从未消费的逐足 requirement，但不改变 reward/eval/runtime；`push_recovery_obs_v1` / `unitree_actions_v1` 兼容标签未重写，同时 manifest 保存真实 P0 policy-I/O 与 observation/action 内容 hash |
| P1-G5 LOOP 复用 | PASS | Push 通过原 `scripts/run_loop.py` 完成 1-round shared LOOP dry-run；13/13 checks 为 true，train/full eval/sealed validation 均完成；24 个证据文件在 `core.autocrlf=true` 的全新 Git 检出后逐字节相同且 sealed artifact 可重载；`accepted_count=0` 是正常结果，不是 GO 门失败 |
| P1-G6 child binding | PASS | 新 train/eval argv 均携带绝对 manifest path/hash 和 base protocol；path/hash 单边输入被拒绝；四个 child 在 App 创建前严格重载；runner 对 child 回传的 manifest hash 与完整 identity 复核，错误回传被拒绝 |
| P1-G7 lineage | PASS | V2 binding 把非空 resolved task protocol 纳入自身 hash；legacy V1 grant 任一内容漂移即拒绝；非 legacy checkpoint 缺失或错配 cell hash 均在子进程启动前失败；base protocol 漂移继续由既有 baseline-eval 权威边界和 child manifest 校验拒绝；session/Genesis/ledger replay 回归通过 |
| P1-G8 所有权 | PASS | `robots/` 对任何 task 包零 import；两个 TaskPlugin 源码无具体 joint/link/USD/上游包；binding 无 budget/seed/acceptance；AgentCore/authoring/decision/ledger 无机器人分支 |

## 4. LOOP 实证

权威精简回执位于 `PUSH_SHARED_LOOP_DRY_RUN/`：

- session：`audit-20260826T115124Z`；
- manifest：`f7835d62b6ebb384fa1cd1266bfa29d1d0bac4f8758a0664a654f1d6e133ae95`；
- session Genesis：`31a03dd2f65408d0e6a7aaacf18c61a3770a0ca887dcea6462cb82b735016e35`；
- runtime snapshot：`02fb3ca597bd5f0252d8553e6d9a982e501d7b5bf601346ec89a2b455a8276e5`；
- 1 个候选完成 train、full eval 和 sealed validation，decision 为 reject；
- 13 项审计检查全部为 true，所有 violation 列表均为空。

该运行证明 Push 仍接入共享 LOOP 和原决策语义，不证明 GPU 策略提升。P1 保留 summary/ledger/manifest/Genesis 以及它们直接引用的 candidate、round、trial 原始回执（24 个文件、约 183 KB），保证路径可追溯；仅运行锁和此前试跑留在项目 `.tmp`。

这些回执内部以原始源码字节的 SHA-256 与大小绑定 sealed candidate，不能只把外层闭包改成 canonical JSON。`.gitattributes` 因而对 `PUSH_SHARED_LOOP_DRY_RUN/**` 明确使用 `-text -eol`，将该小型冻结目录视作字节证据；自动验收器同时重算外层 24 文件闭包和内层 candidate/source 身份。证据闭包 hash 为 `64fd517af2291d93c89d4c9865cd1d5d33ca57715f2ed9ad513306bfaf17422f`。

## 5. 回归结果

- 定向矩阵契约：13 tests，全部通过；
- Phase 1 自动验收器自测：3 tests，全部通过，含 `core.autocrlf=true` 全新检出逐字节往返与语义篡改拒绝，1.029 秒；
- composition/backend/runner/Push/Genesis/LOOP 相关回归：237 tests，全部通过，1 skipped，92.733 秒；
- 全量仓库：1496 tests，全部通过，29 skipped，823.073 秒；
- Python compileall：通过；
- Ruff `F` / `E9`：所有本阶段涉及的产品与测试文件通过；
- `git diff --check`：通过。

机器回执为 `P1_MATRIX_CONTRACT_SNAPSHOT.json`：冻结 29 个 Phase 1 产品/测试文件，`product_tree_hash=ef3b2c3316e041e0380b987e180f6cfe40f20051ef73aa1230dcced069a1a711`，`receipt_hash=59eb454bdcd6c504d887dda38a71dca3643e3e4d1d532861d95395afa55732d3`。该源码闭包与入口 HEAD 只作历史 provenance；持续回归门校验两份 legacy G1 profile 的精确事实、两个 legacy cell 的机器人/任务轴、24 文件证据闭包与 sealed artifact，不要求后续 Phase 的源码字节等于 P1。可复现验收命令为：

```powershell
.\.venv\Scripts\python.exe PLAN\Variant_construction\P1\verify_phase1_snapshot.py
```

### 5.1 Phase 0 基线验收器的移交边界

Phase 1 有意让两个 legacy cell 的新 composition 从 `runtime_composition_snapshot.v1` 升到含
`resolved_cell` 的 V2；因此 P0 记录到当前 P1 的 Genesis snapshot hash 分别由 Stair
`264f213d… → 4b26b227…`、Push `b0924edd… → 02fb3ca5…` 发生预期变化。自提交 `43cfac2` 起，
`verify_phase0_snapshot.py` 会只因这两个受保护字段按设计失败，不再是当前入口门；当前权威入口是本阶段的
`verify_phase1_snapshot.py`，不得回写 P0 历史快照来消除失败。历史 Genesis artifact reader 与 ledger 的只读回放
仍受支持，但边界必须说准确：`verify_session_genesis_for_runtime()` 不重算该 hash，而
`prepare_session_genesis()` 在用当前 V2 composition 重开会话时仍会比较它；旧 V1 identity 会被按设计拒绝为
`session Genesis identity drift`，历史会话应在其冻结源码身份下审计，不能被当前 cell 身份续跑。

## 6. 明确保留到后续 Phase 的边界

- 当前 Stair runtime/reward/evidence 与 `StairClimbingTaskPlugin.robot_requirements()` 仍有双 support 假设；这四处属于同一 Phase 2 迁移，不在 Phase 1 提前放宽 capability 门；
- 35-cell 矩形网格、C1 irregular suite 和 C2 roughness 未改，仍按 Phase 3A/3B 处理；
- Go2 profile/cell 未加入 Core V1，Phase 1 只使用 Phase 0 Go2 回执作为未来输入依据；
- backend 已能对非 legacy manifest 产生 TrainingBinding V2；首个 non-legacy Stair cell 还必须在 Phase 3A 把 resolved hash 写入 effective TaskSpec/EvaluationBinding，真正的 child/runtime ABI 迁移仍需 Phase 2 配套；
- Push legacy 的两个占位 policy-I/O 标签只保留历史 reader/path；Extension A 的 Go2 × Push 必须使用 manifest content hashes，不能复制占位标签；
- 本阶段没有进行 GPU 策略性能实验，也没有把“观察到改进”设为验收条件。

## 7. 最终结论

Phase 1 达到主计划的完成定义：机器人轴已经以精确 profile/runtime adapter 抽出；两个 legacy G1 cells 使用同一 role/capability compiler 和 manifest contract 接回原系统；任务特殊语义没有进入公共层；旧插件、LOOP 与历史 V1 身份未退化。可以进入 Phase 2，但不得跳过 Phase 2 直接宣称 Go2 或复杂楼梯已经通用。
