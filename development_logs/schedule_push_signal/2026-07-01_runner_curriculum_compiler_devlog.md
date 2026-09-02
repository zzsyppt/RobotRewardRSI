# Runner Curriculum Compiler Dev Log

日期: 2026-07-01

## 本次目标

把 V1 LLM `curriculum_design` 从 prompt/provider artifact 推进到 runner 真实训练路径,避免只停留在"LLM 写了课程计划但训练仍然 no-push"。

## 架构变更

- 新增 runner-owned `curriculum_compiler`。
  - 输入: `operation_diff.json.curriculum_design` 和 `diagnostics.json.curriculum_hypothesis`。
  - 复用 V1 curriculum artifact validation。
  - 额外要求 stage 覆盖完整 `[0, 1]` training progress,连续无 gap。
  - 输出: `compiled_phase1_push_recovery.yaml`、`compiled_phase2_baseline.yaml`、`compiled_curriculum_summary.json`。

- 新增 runner-owned `curriculum_runtime`。
  - 训练脚本读取 compiled phase1 config 中的 `runner_curriculum`。
  - 在 `RslRlVecEnvWrapper` 后、candidate reward wrapper 前插入 `Phase8EVecEnvCurriculumWrapper`。
  - 每个 env step 前按 runner-owned progress 选择 stage,并更新 `env.unwrapped.cfg.events.push_robot` 的 push interval 和 velocity range。
  - runtime summary 记录 active stage、stage transitions、last applied push 和 mutation report。

- full round 接入 compiler。
  - provider/Phase 8D 通过后、fixed-budget training 前编译课程。
  - 编译失败时训练前 reject,并只追加 public-safe feedback。
  - 编译成功后训练命令使用 generated phase1/phase2 config。
  - eval rollout 仍使用 budget runner 原始 phase1 config/profile,不读取训练课程。

- 新增 curriculum lineage / attribution 记录。
  - compiler 写出 `curriculum_lineage.json`,包含 public summary、raw curriculum design、effective curriculum、clamps。
  - full round 把 lineage artifact 传给 linked round。
  - linked round 最终把 public lineage 写入 ledger decision/detail 和 top-level ledger field。
  - accepted recipe snapshot 下写入 `runner_curriculum_snapshot/`:
    - `curriculum_lineage.json`
    - `public_summary.json`
    - `raw_curriculum_design.json`
    - `effective_curriculum.json`
  - prompt-facing ledger projection现在包含 public attribution:
    - `reward_code_changed`
    - `curriculum_changed`
    - `diagnostics_changed`
    - `capability_changed`
    - `gain_attributable_to`
    - `pure_knob_candidate`

## 真实执行边界

- `external_push_curriculum` 已进入训练路径:
  - LLM 只能提交 stage/capability 结构。
  - runner compiler 决定 stage push strength,并强制 final consolidation 回到 public target push distribution。
  - candidate code 不接触 env object、runner object、eval config、budget、seed、metric 或 checkpoint selector。

- `train_only_dr_scale` 第一版是 train-only 静态鲁棒性开关:
  - compiler 在 compiled phase1 config 中开启训练侧 observation corruption。
  - 这是实际训练配置变更,但不是 per-stage 动态 DR。
  - artifact 明确标注 `mode: train_only_static_observation_corruption`。

- 旧 `ScheduleTerm` key 仍不直接控制 env push/DR:
  - `reward_weight_scale` 仍是唯一 prompt-facing effective schedule key。
- `push_magnitude_scale`、`push_start_fraction`、`push_interval_s`、`domain_randomization_scale` 不再被错误标成 prompt-facing effective schedule key。
- push/DR 通过 runner-audited `curriculum_design` 编译执行,不是通过 candidate schedule hook 执行。

- P6/P7 lineage 已进入 ledger/snapshot:
  - raw/effective curriculum 被保存为 accepted snapshot 的 runner-owned 附件。
  - 下一轮 public feedback 可看到课程 attribution。
  - `pure_knob_candidate` 由 runner 基于 stage count 和 capability 使用判定。

## 本地验证

- `scripts/phase8e_curriculum_compiler_selftest.py`
- `scripts/phase8e_full_round_selftest.py`
- `scripts/phase8e_selftest.py`
- `scripts/phase8e_provider_ingest_selftest.py`
- `scripts/phase8e_training_adapter_selftest.py`
- `python -m py_compile` on changed Python files
- `git diff --check`

## 后续必须验证

- 在服务器 IsaacLab Python 下跑同一批轻量测试。
- 跑至少一轮 full-round smoke,确认 child training summary 中:
  - training command 使用 compiled phase1/phase2 config。
  - `phase8e_compiled_curriculum.enabled == true`。
  - runtime `last_apply_report.status == applied`。
  - final stage push range 非 `[0.0, 0.0]`。

## 服务器验证结果

服务器 commit: `bde56d4`

轻量测试:

- `scripts/phase8e_curriculum_compiler_selftest.py`: pass
- `scripts/phase8e_full_round_selftest.py`: pass
- `scripts/phase8e_provider_ingest_selftest.py`: pass
- `scripts/phase8e_selftest.py`: pass, `hard_isolation=true`
- `scripts/phase8e_training_adapter_selftest.py`: pass

缩小预算 full-round smoke:

- run: `/data/yuyangliu/runs/phase_8_curriculum_smoke/curriculum_compiler_smoke_20260701_032838`
- status: `accept`
- curriculum compiler stage: `pass`
- training stage: `pass`
- compiled curriculum enabled: `true`
- execution mode: `env_step_progress`
- runtime transitions:
  - `balance_warmup` at step 0
  - `target_consolidation` at step 17
- runtime stage counts:
  - `balance_warmup`: 17
  - `target_consolidation`: 31
- final `last_apply_report.status`: `applied`
- final push range:
  - x: `[-0.25, 0.25]`
  - y: `[-0.25, 0.25]`

注意: 这是小预算接线 smoke,不是性能评估。它只证明 compiler/runtime/full-round 路径真实执行。

追加服务器验证:

- commit: `923d4fa`
- linked selftest: pass, `linked_round=pass`
- lineage smoke run: `/data/yuyangliu/runs/phase_8_curriculum_smoke/curriculum_lineage_smoke_20260701_033921`
- full-round status: `accept`
- runtime `last_apply_report.status`: `applied`
- runtime final push:
  - x: `[-0.25, 0.25]`
  - y: `[-0.25, 0.25]`
- ledger `curriculum_lineage.attribution`:
  - `curriculum_changed=true`
  - `capability_changed=true`
  - `gain_attributable_to=curriculum`
  - `pure_knob_candidate=false`
- accepted snapshot includes:
  - `runner_curriculum_snapshot/curriculum_lineage.json`
  - `runner_curriculum_snapshot/public_summary.json`
  - `runner_curriculum_snapshot/raw_curriculum_design.json`
  - `runner_curriculum_snapshot/effective_curriculum.json`

## 仍未完成

- `push_active` / `push_vec_public` 仍未开放给 candidate reward surface。
- per-stage dynamic DR scaling 仍未实现; 当前只做 train-only static observation robustness。
- recovery-window diagnostic 仍是 report-only 后续项,不是 accept/reject metric。

## 2026-07-01 provider 失败摘要修复与 15-round 状态

- commit: `0968bbe`
- 修复项: provider 异常进入 `phase8e_provider_candidate_summary.json` 时, `_safe_error_message` 现在会遮盖 public-only forbidden token, 包括 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `api_key` / `secret` 等 credential marker。
- 回归测试:
  - 本地 `scripts/phase8e_provider_ingest_selftest.py`: pass
  - 本地 `scripts/phase8e_selftest.py`: pass
  - 本地 `scripts/phase8e_full_round_selftest.py`: pass
  - 本地 `python -m py_compile` changed files: pass
  - 本地 `git diff --check`: pass
  - 服务器 `scripts/phase8e_provider_ingest_selftest.py`: pass
  - 服务器 `scripts/phase8e_selftest.py`: pass, `hard_isolation=true`
- 默认 deepseek 15-round 验证:
  - run: `/data/yuyangliu/runs/phase_8_cold_start_autoresearch/llm_curriculum_v1_15round_20260701_035537`
  - rounds_executed: `15`
  - accepted_rounds: `0`
  - formal_training_executed: `false`
  - public-only crash: 已消除
  - 首轮 provider error: `Missing API key environment variable for deepseek: <redacted>`
  - summary 中未再出现 `DEEPSEEK_API_KEY` / `OPENAI_API_KEY` / `api_key` / `secret`
- 结论: 这次 15-round 只验证了 provider failure path 的安全净化, 没有发生真实 LLM 课程设计、compiler 或训练。真实 LLM 15-round 仍需要在服务器运行环境提供 `DEEPSEEK_API_KEY` 或改用配置好的 provider。

## 2026-07-01 runner-owned push signal instrumentation

- commit: `bd0b6ee`
- 背景: 架构计划 P1 要求 `push_active` / `push_vec_public` 成为 reward-only signal。此前这两个字段仍是 `not_instrumented` placeholder, 因此 prompt-facing contract 会隐藏它们。
- 实现:
  - 新增 `cold_start_lab.push_signal.Phase8EPushSignalBuffer`。
  - 训练脚本在 candidate training 时安装 runner-owned `push_robot` event wrapper。
  - wrapper 保持 IsaacLab `push_by_setting_velocity` 语义, 但记录真实采样的 public xy velocity delta。
  - `Phase8ETrainingAdapter` 每步消费 buffer, 填充 `RewardInputs.push_active` 与 `RewardInputs.push_vec_public`。
  - eval 脚本未安装该 wrapper, trusted eval 仍 checkpoint-only, 不读取 candidate code 或训练课程。
- prompt-facing contract:
  - `push_active.source_status = runner_owned_push_event_instrumentation`
  - `push_vec_public.coordinate_frame = world_xy_velocity_delta`
  - 两个字段现在进入 recipe prompt, 但 candidate 仍只接收 `RewardInputs`, 不接收 env/event manager/seed/budget/eval/metric。
- 本地验证:
  - `scripts/phase8e_training_adapter_selftest.py`: pass
  - `scripts/phase8e_selftest.py`: pass
  - `scripts/phase8e_provider_ingest_selftest.py`: pass
  - `scripts/phase8e_full_round_selftest.py`: pass
  - `scripts/phase8e_curriculum_compiler_selftest.py`: pass
  - `scripts/phase8e_linked_round_selftest.py`: pass, local linked path skipped because no torch
  - changed-file `py_compile`: pass
  - `git diff --check`: pass
- 服务器验证:
  - server commit: `bd0b6ee`
  - `scripts/phase8e_training_adapter_selftest.py`: pass
  - `scripts/phase8e_selftest.py`: pass, `hard_isolation=true`
  - smoke run: `/data/yuyangliu/runs/phase_8_curriculum_smoke/push_signal_train_smoke_20260701_041410`
  - smoke status: `pass`
  - `phase8e_push_signal_probe.status = installed`
  - `phase8e_push_signal_runtime.trigger_count = 1`
  - `phase8e_candidate_runtime.input_provenance.push_active = runner_owned_push_event_instrumentation`
  - `phase8e_candidate_runtime.input_provenance.push_vec_public = runner_owned_push_event_instrumentation`
- 注意: `push_magnitude_scale` / `push_start_fraction` / `push_interval_s` 这些旧 ScheduleTerm key 仍不是 prompt-facing effective schedule key。push curriculum 继续通过 runner-owned `curriculum_design` compiler/runtime 执行。

## 2026-07-01 accepted curriculum immutable context snapshot

- 背景: 架构计划 P6 要求 accepted curriculum 快照不仅保存 raw/effective curriculum,还要能复现实验 lineage 所依赖的 registry、compiler、public evidence、prompt-facing contract、training/eval config。
- 实现:
  - provider summary 新增 `prompt.payload_file`,指向 runner 已写出的 `phase8e_prompt_payload.json`。
  - full-round 在 curriculum compile 成功后读取 provider prompt payload,并把 runner-owned train/eval artifact 信息传给 compiler。
  - compiler 为 `curriculum_lineage.json` 增加 `immutable_context`:
    - `capability_registry.sha256`
    - `curriculum_compiler.sha256`
    - `public_evidence_bundle.sha256`
    - `prompt_facing_contract.sha256`
    - `training_config.sha256`
    - `eval_config.sha256`
  - public lineage projection 只增加 `immutable_context_hash`,不投影 prompt payload 路径、raw rollout seeds 或内部 context。
- 边界:
  - LLM 没有获得新控制权限。
  - eval 仍 checkpoint-only,不读取 candidate code 或训练课程。
  - raw rollout seed 只用于 runner-owned eval context hash,public summary 不暴露 seed 值。
- 本地 RED/GREEN:
  - `scripts/phase8e_provider_ingest_selftest.py` 先失败于 `KeyError: 'payload_file'`,实现后 pass。
  - `scripts/phase8e_full_round_selftest.py` 先失败于缺少 `immutable_context`,实现后 pass。
- 本地验证:
  - `scripts/phase8e_provider_ingest_selftest.py`: pass
  - `scripts/phase8e_full_round_selftest.py`: pass
  - `scripts/phase8e_curriculum_compiler_selftest.py`: pass
  - `scripts/phase8e_selftest.py`: pass
  - `scripts/phase8e_linked_round_selftest.py`: pass, local linked path skipped because no torch
  - `scripts/phase8e_training_adapter_selftest.py`: pass
  - changed-file `py_compile`: pass
  - `git diff --check`: pass
- 服务器验证:
  - server commit: `b9bc3de`
  - `scripts/phase8e_provider_ingest_selftest.py`: pass
  - `scripts/phase8e_full_round_selftest.py`: pass
  - `scripts/phase8e_curriculum_compiler_selftest.py`: pass
  - `scripts/phase8e_selftest.py`: pass, `hard_isolation=true`
  - `scripts/phase8e_linked_round_selftest.py`: pass, linked path skipped because server Python env has no torch
  - `scripts/phase8e_training_adapter_selftest.py`: pass
  - server git status: clean on `codex/llm-curriculum-agent-design`

## 2026-07-01 P7 curriculum attribution monitoring

- 背景: 架构计划 P7 要求显式监控系统是否退化成课程超参/纯 knob 搜索。
- 实现:
  - `build_public_history_summary` 新增 `curriculum_attribution_summary`。
  - summary 只从 prompt-safe public history / ledger projection 读取,不读取 hidden eval、raw seed 或 runner private state。
  - 记录:
    - accepted rounds with curriculum lineage count
    - accepted pure-knob count
    - accepted curriculum-changed count
    - current accepted pure-knob streak
    - gain attribution counts
    - latest accepted attribution
  - 如果最近连续 accepted lineage 都是 `pure_knob_candidate=true` 且 streak >= 2,输出 warning: `accepted_curriculum_degenerated_to_pure_knob_streak`。
- 边界:
  - 这是 prompt-facing public feedback,不是 accept/reject gate。
  - 不改变 trusted eval、checkpoint selector、seed、budget、success metric。
- 本地 RED/GREEN:
  - `scripts/phase8e_selftest.py` 先失败于缺少 `curriculum_attribution_summary`,实现后 pass。
- 本地验证:
  - `scripts/phase8e_selftest.py`: pass
  - `scripts/phase8e_provider_ingest_selftest.py`: pass
  - `scripts/phase8e_full_round_selftest.py`: pass
  - `scripts/phase8e_linked_round_selftest.py`: pass, local linked path skipped because no torch
  - changed-file `py_compile`: pass
- 服务器验证:
  - server commit: `66b94eb`
  - `scripts/phase8e_selftest.py`: pass, `hard_isolation=true`
  - `scripts/phase8e_provider_ingest_selftest.py`: pass
  - `scripts/phase8e_full_round_selftest.py`: pass
  - changed-file `py_compile`: pass
