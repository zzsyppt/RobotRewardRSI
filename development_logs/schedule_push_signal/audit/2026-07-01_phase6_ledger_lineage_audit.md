# Phase 6: Ledger、Lineage、Immutable Context 与 Next-Round Feedback 审计记录

## 审计阶段

Phase 6: Ledger、Lineage、Immutable Context 与 Next-Round Feedback。

## 审计对象

- `src/humanoid_agent/cold_start_lab/curriculum_compiler.py`
- `src/humanoid_agent/cold_start_lab/phase8e.py`
- `src/humanoid_agent/cold_start_lab/phase8e_linked.py`
- `src/humanoid_agent/cold_start_lab/phase8e_full_round.py`
- `src/humanoid_agent/cold_start_lab/phase8e_projection.py`
- `src/humanoid_agent/cold_start_lab/phase8e_prompt.py`
- `scripts/phase8e_linked_round.py`

## 关键风险

- accepted curriculum 必须能从 ledger 追溯到 runner-owned compiler 的 raw/effective/clamps/attribution。
- accepted recipe snapshot 必须保存 runner curriculum snapshot，否则下一轮继承只能继承 candidate code，不能复现训练侧 curriculum。
- immutable context hash 必须覆盖 capability registry、prompt-facing contract、compiler、public evidence、training config 和 eval config。
- 下一轮 prompt 只能看到 public-safe lineage 与 attribution feedback，不能暴露 hidden/private eval 信息或 seed。
- lineage 缺失、summary 缺失或 snapshot 写入失败时必须有可发现的失败信号，不能静默接受。

## 已检查内容

- 正常 full-round 设计路径：`run_phase8e_full_round` 会把 compiler 产出的 `curriculum_lineage_file` 传给 linked round。
- 正常 accepted 路径：传入 `curriculum_lineage_path` 时，ledger 记录 public lineage，accepted snapshot 写入 `runner_curriculum_snapshot`，下一轮 prompt 生成 `curriculum_attribution_summary`。
- 负向 accepted 路径：不传 `curriculum_lineage_path` 时，同一个合法 curriculum candidate 仍会被 accept 并继承。
- CLI 薄层：`scripts/phase8e_linked_round.py` 暴露 `--append-ledger` 和 `--inherit-if-accepted`，但不支持 `--curriculum-lineage-path`。
- 本地与服务器轻量回归自测。

## 证据路径

- 自定义 probe report:
  - `development_logs/schedule_push_signal/audit/phase6_lineage_probe_20260701/run_iv4u59bx/phase6_probe_report.json`
- 正常 lineage:
  - `development_logs/schedule_push_signal/audit/phase6_lineage_probe_20260701/run_iv4u59bx/compile_round/fixed_budget_training/curriculum_compiler/curriculum_lineage.json`
- 正常 accepted snapshot:
  - `development_logs/schedule_push_signal/audit/phase6_lineage_probe_20260701/run_iv4u59bx/lab/accepted_recipes/0000_phase6_curriculum_probe_candidate/runner_curriculum_snapshot/`
- 缺失-lineage accepted snapshot:
  - `development_logs/schedule_push_signal/audit/phase6_lineage_probe_20260701/run_iv4u59bx/missing_lineage_lab/accepted_recipes/0000_phase6_curriculum_probe_candidate/`

## 发现的问题

### P1: accepted + append-ledger + inherit-if-accepted 允许缺失 curriculum lineage，导致可接受课程不可追溯

`run_phase8e_round` 中 `curriculum_lineage_path` 是可选参数。当前逻辑会先把缺失 lineage 投影为 `None`，但在 candidate 被接受且 `append_ledger=True`、`inherit_if_accepted=True` 时没有强制要求 lineage 存在，仍然会调用 `_snapshot_accepted_recipe` 并追加 ledger。

实测结果：

- `status=accept`
- `accepted=true`
- `summary_curriculum_lineage=null`
- `ledger_curriculum_lineage=null`
- `snapshot_runner_curriculum_snapshot_exists=false`

这会破坏 Phase 6 的核心目标：accepted curriculum 不能完整复现和追责，下一轮 LLM 也看不到 attribution feedback。

相关代码位置：

- `src/humanoid_agent/cold_start_lab/phase8e.py:101` 允许 `_load_curriculum_lineage(None)` 返回 `None`。
- `src/humanoid_agent/cold_start_lab/phase8e.py:142-164` accepted ledger append 没有校验 `curriculum_lineage`。
- `src/humanoid_agent/cold_start_lab/phase8e.py:480-512` snapshot 只有在 lineage 是 dict 时才写 `runner_curriculum_snapshot`，否则静默只复制 candidate。

### P1: linked round CLI 无法传入 curriculum lineage

`run_phase8e_linked_round` 函数本身支持 `curriculum_lineage_path`，full-round 驱动也会传入 compiler 产物。但 `scripts/phase8e_linked_round.py` 没有 CLI 参数，因此单独使用 linked round 脚本并开启 `--append-ledger --inherit-if-accepted` 时无法把 lineage 传入。

实测：

- `scripts/phase8e_linked_round.py --help` 只显示 `--append-ledger`、`--inherit-if-accepted`，没有 curriculum/lineage 参数。
- 显式传 `--curriculum-lineage-path dummy` 会报 `unrecognized arguments`。

相关代码位置：

- `scripts/phase8e_linked_round.py:20-37` parser 未定义 lineage 参数。
- `scripts/phase8e_linked_round.py:43-55` 调用 `run_phase8e_linked_round` 时未传 `curriculum_lineage_path`。

## 正常路径结论

在传入 compiler lineage 的路径上，Phase 6 目标基本成立：

- lineage 包含 `public_summary`、`raw_curriculum_design`、`effective_curriculum`、`clamps`、`immutable_context`。
- `immutable_context_hash` 在 compiler summary 和 public summary 中一致，完整 context 的 `sha256` 覆盖 capability registry、prompt-facing contract、compiler、public evidence、training config、eval config。
- accepted snapshot 写出 `runner_curriculum_snapshot/curriculum_lineage.json`、`public_summary.json`、`raw_curriculum_design.json`、`effective_curriculum.json`。
- ledger 只投影 public-safe lineage keys，包括 attribution、capabilities、clamp_count、stage_count、immutable_context_hash。
- 下一轮 prompt 的 `curriculum_attribution_summary` 能统计 accepted lineage、pure knob streak、gain attribution 和 latest accepted attribution。

## 验证命令

- 本地:
  - `python scripts/phase8e_selftest.py` 使用 Codex bundled Python，pass。
  - `python scripts/phase8e_curriculum_compiler_selftest.py` 使用 Codex bundled Python，pass。
  - `python scripts/phase8e_full_round_selftest.py` 使用 Codex bundled Python，pass。
  - `python scripts/phase8e_linked_round_selftest.py` 使用 Codex bundled Python，pass，local `linked_round=skipped_no_torch`。
  - `python scripts/phase8e_evolution_loop_selftest.py` 使用 Codex bundled Python，pass。
- 服务器:
  - `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_selftest.py` pass，`hard_isolation=true`。
  - `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_curriculum_compiler_selftest.py` pass。
  - `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_linked_round_selftest.py` pass，`linked_round=pass`。
  - `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_full_round_selftest.py` pass。
  - `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_evolution_loop_selftest.py` pass。

## 建议修复顺序

1. 在 `run_phase8e_round` 中增加强制校验：当 `decision.accepted is True` 且 `append_ledger=True` 且 `inherit_if_accepted=True` 时，必须存在可投影的 `curriculum_lineage`，否则抛出 `Phase8EError`。
2. `_snapshot_accepted_recipe` 不应静默缺失 runner curriculum snapshot；如果调用方处于 accepted/inherit 语义，应要求 lineage 是 dict，并在写入失败时失败而不是继续。
3. 给 `scripts/phase8e_linked_round.py` 增加 `--curriculum-lineage-path` 参数并传给 `run_phase8e_linked_round`。
4. 增加回归测试：accepted+inherit 缺失 lineage 必须 reject/fail；linked CLI 必须能传 lineage；传 lineage 时 ledger/snapshot/next prompt attribution 保持现有正向行为。

## 风险等级

P1。full-round 主路径当前会传 lineage，风险没有直接打断主编排；但底层 API 和 linked CLI 允许 accepted ledger/snapshot 静默断链，违反 Phase 6 的审计目标，并可能在手动 linked round 或后续编排复用时污染 ledger。
