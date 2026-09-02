# True RL Agent Live 15 Rounds 审计 Checklist

本文用于启动并跟踪一轮真实 `live 15 rounds`。清单以当前工程实际代码为准，并结合 `PLAN/prompt_historybest/history_best_authoring_architecture_plan.md` 及 Phase0-Phase9 已完成改造制定。

## 审计目标

- 确认 `history-best source`、`source_catalog`、`candidate_lineage`、`source_request/source_followup`、`CandidateAuthoringLoop`、component smoke、anti-degenerate、capability registry 与 runner-owned schema preview 在真实 live 链路中平稳运行。
- 确认 LLM 获得足量上下文：任务语义、RewardInputs 合同、public eval 合同、current-best lineage、必要源码、静态 repair feedback、runtime feedback summary。
- 确认上下文不会爆炸：每轮 prompt 有预算、source payload 有 hash/bytes/omitted 记录，repair/source followup 不重复无限历史。
- 确认没有越权：不暴露真实路径、secret、raw seed、trusted/private eval、checkpoint path、runner source/code；不开放课程/环境/runner patch；不把任务特殊语义写入共享层。
- 确认没有退化：候选不是只改数字、只改名字、只改 rationale、复制 current-best；parameter-only 候选不能被当作 policy improvement 接受。
- 确认训练/eval/acceptance 仍由 runner-owned 固定协议控制，不被 LLM 调参或修改。

## 当前项目主链路

真实 live 调用链：

```text
scripts/run_loop.py --mode live
-> true_rl_agent.agent_core.live_loop.run_live_loop()
-> PublicContextBuilder(task_memory_adapter, task_prompt_adapter)
-> CandidateAuthoringLoop.run_round()
-> build_authoring_prompt_payload()
-> build_prompt_payload()
-> OpenAICompatibleCandidateProvider
-> provider response: candidate_draft 或 source_request
-> candidate materialize / source resolver
-> StaticCandidateVerifier
-> component smoke
-> anti-degenerate check
-> run_candidate_round()
-> backend_adapter materialize train/eval jobs
-> SubprocessRunner
-> scripts/isaaclab_rsl_rl_train.py
-> scripts/isaaclab_rsl_rl_eval.py
-> decision accept/reject
-> ledger.jsonl
-> 下一 round 的 public context / source context / task summary
```

关键产物位置：

```text
<RUN_ROOT>/live_summary.json
<RUN_ROOT>/ledger.jsonl
<RUN_ROOT>/rounds/<SESSION_ID>-round-###/prompt_payload.json
<RUN_ROOT>/rounds/<SESSION_ID>-round-###/prompt_payload_attempt_###.json
<RUN_ROOT>/candidates/<candidate-id>/
<RUN_ROOT>/runner/<session-id>/<round-id>/train/
<RUN_ROOT>/runner/<session-id>/<round-id>/eval/
```

服务器默认位置：

```text
/data/yuyangliu/AGENT/true_RL_Agent
/data/yuyangliu/secrets/deepseek.env
/data/yuyangliu/envs/isaaclab/bin/python
```

默认 live 预算预期：

```text
rounds = 15
train_num_envs = 4096
train_steps_per_iteration = 24
train_max_iterations = 500
budget_steps = 49,152,000
eval_num_envs = 64
public_seed_count = 3
public_accept_seed_count = 1
runner_timeout_s = 7200
provider_max_attempts = 5
provider_model = deepseek-v4-pro
```

## 启动前检查

- [ ] 本地与服务器代码一致。
  - 预期：本地 `HEAD`、`origin/true-rl-agent-refactor`、服务器 `/data/yuyangliu/AGENT/true_RL_Agent` 的 `HEAD` 一致。
  - 预期：服务器 `git status --short --branch` 业务代码干净。

- [ ] 服务器环境可运行。
  - 预期：`/data/yuyangliu/secrets/deepseek.env` 存在，但不打印内容。
  - 预期：`/data/yuyangliu/envs/isaaclab/bin/python` 可执行。
  - 预期：没有正在运行的 `scripts/run_loop.py`、`isaaclab_rsl_rl_train.py`、`isaaclab_rsl_rl_eval.py`。
  - 预期：GPU 空闲或负载符合预期。

- [ ] 本地与服务器 smoke 已通过。
  - 预期：`python3 -m compileall -q src scripts tests` 通过。
  - 预期：`PYTHONPATH=src python3 -m unittest discover -s tests` 通过。
  - 预期：15-round `dry-run-audit` 的 checks 全为 true。

- [ ] 启动命令不覆盖旧 run。
  - 预期：新的 `<RUN_ROOT>` 不存在 `ledger.jsonl`。
  - 预期：`session_id` 唯一，例如 `live-15round-YYYYMMDDTHHMMSSZ`。

## 启动后 0-5 分钟检查

- [ ] `live_summary.json` 尚未生成时，先看 run root 和 round 目录是否创建。
  - 预期：出现 `<RUN_ROOT>/rounds/<session>-round-001/`。
  - 预期：第 1 轮先生成 `prompt_payload.json` 或 attempt prompt。

- [ ] 第 1 个 prompt 是完整 payload，而不是只传 `public_context`。
  - 预期：`prompt_payload.json` 顶层包含 `task`、`loop_contract`、`required_candidate_package`、`candidate_capabilities`、`reward_inputs_contract`、`public_eval_contract`、`control_effectiveness_contract`、`source_catalog`、`candidate_lineage`、`source_payloads`、`authoring_context_frame`。
  - 预期：provider request 和保存的 prompt artifact 同源。

- [ ] prompt section 顺序可读，不是字母排序。
  - 预期：先任务/loop/候选输出合同，再 capability/reward/eval/control/source/history/feedback。
  - 预期：`authoring_context_frame.included_sections` 能覆盖 provider-visible legacy alias sections。

- [ ] prompt 大小在预算内。
  - 预期：`authoring_context_frame.budget_metadata.actual_prompt_payload_bytes < 262144`。
  - 预期：单 section 不超过 hard limit。
  - 预期：如有省略，`omitted_sections` 写明原因和可恢复来源。

## 每轮 Prompt Payload 通用检查

每轮都检查当前 round 的 `prompt_payload.json`，若有 `prompt_payload_attempt_###.json`，也要检查每个 attempt。

- [ ] public-only。
  - 预期：没有真实 Windows 路径、服务器绝对路径、secret、API key、raw seed id、trusted/private eval、checkpoint path、candidate path。
  - 预期：出现 `raw seed identifiers are omitted` 这类说明可以接受，但不能有真实 seed 值。
  - 预期：出现 `operation_diff` 只能在候选包合同或禁止说明中，不能作为 LLM 可写 root key。

- [ ] task-owned 语义位置正确。
  - 预期：push recovery 的 RewardInputs 字段语义在 `reward_inputs_contract` / task prompt sections 中。
  - 预期：共享层不解释 push recovery 专属 failure label、success 语义或具体字段用法。
  - 预期：非任务通用 section 不出现硬编码 `push_active` 示例文案。

- [ ] capability registry 投影正确。
  - 预期：默认 `candidate_capabilities.prompt_visible_capabilities` 只包含：
    - `reward_term`
    - `diagnostic_term`
    - `reward_weight_scale_schedule`
  - 预期：`curriculum_design` 和 `train_environment_design` 默认只在 `disabled_or_reserved_capabilities` 中，且：
    - `prompt_visible=false`
    - `candidate_writable=false`
    - `effective_now=false`
    - `schema_status=schema_draft_hidden_until_ready`
    - 不携带完整 `output_schema`
  - 预期：`required_candidate_package.provider_json_output.allowed_keys` 不新增课程、环境或 runner patch key。

- [ ] provider 输出合同不漂移。
  - 预期：candidate draft 只允许：
    - `response_mode`
    - `candidate_id`
    - `rationale`
    - `declared_changed_components`
    - `components_py`
  - 预期：`source_request` 只允许：
    - `response_mode`
    - `logical_ids`
    - `rationale`
  - 预期：`declared_changed_components` prompt-facing 值只鼓励当前真实组件能力，不鼓励未实现的 runner/control 类方向。

- [ ] source catalog 是 closed allowlist。
  - 预期：只出现这些 logical id：
    - `api/candidate_components.py`
    - `current_best/training_code/components.py`
    - `examples/known_good_reward_terms.py`
    - `attempt/latest_failed/training_code/components.py`
  - 预期：provider filesystem access 为 false。
  - 预期：每个 source item 有 role/status/requestable/max_bytes/hash 或 pending 信息。

- [ ] candidate lineage 合理。
  - 第 1 轮若没有已接受候选，预期：`candidate_lineage.status=none`。
  - 一旦某轮 accepted，之后预期：`candidate_lineage.status=accepted_current_best`，且引用 `current_best/training_code/components.py`。
  - 预期：current-best source 小于预算时可内联；超预算时只保留 summary/hash/bytes/directive。
  - 预期：lineage 不把 failed candidate 混成 accepted current best。

- [ ] exploration directive 不退化。
  - 预期：能看到鼓励机制级改进、继承/重写/反证 current-best 的 directive。
  - 预期：不鼓励单纯调权重、复制 current-best 或只改 rationale。

## Round 1 特别检查

- [ ] 冷启动时没有伪造 current-best。
  - 预期：`candidate_lineage.status=none` 或清楚说明 bootstrap，不把 bootstrap 当 performance baseline。

- [ ] task contract 足量。
  - 预期：`task` 中可看到固定 train/eval 协议、budget、public seed count、success predicate 的只读说明。
  - 预期：LLM 只能把这些作为 read-only evidence，不能修改。

- [ ] RewardInputs 合同足量。
  - 预期：包含字段 shape、语义、fallback、限制、候选使用建议。
  - 预期：`push_active` / `push_vec_public` 如果出现，只出现在 task-owned reward contract，且说明当前 generic runtime 路径可能是 zero fallback，不能让模型误以为稳定可用。

## Provider Response 与 Candidate Materialization 检查

每轮 provider 返回后检查 candidate 包和 attempt record。

- [ ] provider response 模式正确。
  - 预期：正常候选为 `candidate_draft` 或省略 `response_mode`。
  - 预期：如返回 `source_request`，不得同时携带 candidate draft 字段。
  - 预期：未知 key、非法 JSON、非法 schema 进入 static repair，不进入 runner。

- [ ] candidate 包结构正确。
  - 预期：有 `candidate_manifest.yaml`、`training_code/components.py`、`rationale.md`、`operation_diff.json`、`diagnostics.json` 等 runner/materializer 管理的文件。
  - 预期：`components_py` 中只定义候选 TrainingComponents，不包含文件系统、进程、网络、runner/env/eval/checkpoint/PPO 权限。

- [ ] declared 和 verifier-derived change set 不冲突。
  - 预期：最终以 verifier-derived change set 为准。
  - 预期：如果 provider 声明了不支持或 legacy 的 component，不能污染 prompt contract 或 decision。

- [ ] prompt artifact traceability 完整。
  - 预期：ledger record 中有 `prompt_payload.path`。
  - 预期：`provider_attempt_prompt_payload_paths` 与每次 provider attempt 对应。
  - 预期：provider failure/static failure 记录中不出现 nested `candidate_path`。

## Source Request / Source Followup 检查

如果某轮模型请求源码，必须逐项审计。

- [ ] request logical id 合法。
  - 预期：`logical_ids` 只能来自本轮 `source_catalog.items[].logical_id`。
  - 预期：不能请求真实路径、Windows drive path、反斜杠、`..`、catalog 外 id。

- [ ] resolver 不读任意磁盘。
  - 预期：resolver 只消费本轮完整 source repository 中已经构建好的 `source_catalog` / `source_payloads`。
  - 预期：返回前重新验证 hash、bytes、delivery、lineage 同源关系。

- [ ] source followup 上下文足量但不爆炸。
  - 预期：`authoring_context_frame.frame_kind=source_followup`。
  - 预期：followup prompt 包含累计已返回 source payloads。
  - 预期：followup prompt 不重复完整 recent public history。
  - 预期：多次 source request 仍能请求同一轮 catalog 中尚未返回的其他 logical id，不因 followup 缩小 payload 而误拒绝。

- [ ] source request 有预算与次数限制。
  - 预期：超过每轮 request 次数或 bytes 限制时返回 public-safe feedback，例如 `source_request_turn_limit_exceeded`。
  - 预期：拒绝原因不暴露真实路径。

- [ ] source followup 后必须回到候选草稿。
  - 预期：读取源码后 provider 最终返回 `candidate_draft`。
  - 预期：不能用 source request 当作绕过候选输出合同的通道。

## Static Repair 检查

出现 provider/schema/materialize/verifier/component-smoke/anti-degenerate 问题时检查。

- [ ] static repair 不跑真实 runner。
  - 预期：同一 round 内只允许静态 feedback 触发下一次 provider attempt。
  - 预期：真实 train/eval 只对最终通过静态检查的 candidate 执行一次。

- [ ] repair frame 正确。
  - 预期：`authoring_context_frame.frame_kind=static_repair`。
  - 预期：省略完整 recent history，并在 `omitted_sections` 说明。
  - 预期：repair feedback 只包含最近 blocking failure，不累计无限错误历史。

- [ ] latest failed source 合法。
  - 预期：失败候选源码通过 `attempt/latest_failed/training_code/components.py` 暴露。
  - 预期：有 hash/bytes/scrub status/source logical id。
  - 预期：不出现 `previous_components_py`、真实 candidate path、本地路径或 server path。

- [ ] component smoke 安全。
  - 预期：candidate load、reward callable、diagnostic callable 都在隔离 worker 中执行。
  - 预期：timeout 转为 public-safe `component_smoke_error`，不会卡死 authoring 主进程。

- [ ] anti-degenerate 生效。
  - 预期：`identical_or_format_only` 是 blocking，需要 rewrite。
  - 预期：`parameter_only_delta` 不能被当作 policy improvement eligible。
  - 预期：机制级差异能被识别为可进入 runner 的候选。

## Runtime Failure 边界检查

真实 train/eval 发生 `CandidateRuntimeError` 或 train summary fail 时检查。

- [ ] runtime failure 不进入同轮 provider retry。
  - 预期：该 round 结束，ledger decision reason 为 `candidate_runtime_failed`。
  - 预期：不会在同一 round 构造 runtime repair prompt 后再次调用 provider。
  - 预期：不会在同一 round 再次调用 runner。

- [ ] runtime failure 进入下一轮 public context。
  - 预期：下一 round prompt 的 public history / task summary 能看到 public-safe runtime failure。
  - 预期：如需要失败源码，通过 `attempt/latest_failed/training_code/components.py` 进入 source catalog，而不是旧 path reader。

- [ ] runtime feedback public-safe。
  - 预期：错误类型、公开摘要、candidate loaded/enabled 等信息可见。
  - 预期：不包含 traceback 真实路径、checkpoint path、raw seed、private artifact。

## Train / Eval Job 检查

每轮进入 runner 后检查 train/eval job metadata 和 summary。

- [ ] budget 固定。
  - 预期：`train_num_envs=4096`。
  - 预期：`train_steps_per_iteration=24`。
  - 预期：`train_max_iterations=500`。
  - 预期：`budget_steps=49152000`。
  - 预期：如果 CLI 参数与乘积不一致，run_loop 会拒绝启动。

- [ ] LLM 没有控制 train/eval 协议。
  - 预期：train/eval job 中没有 LLM schedule、eval protocol 修改、seed 修改、success metric 修改、checkpoint policy 修改、PPO 配置修改。
  - 预期：`candidate_changed_components` 只反映 candidate components，不等价于 runner patch。

- [ ] candidate runtime activation 有证据。
  - 预期：train summary / eval summary 能显示 candidate package loaded 或 runtime enabled。
  - 预期：`reward_weight_scale` 如果被使用，有 schedule runtime activation record 和 bounds。
  - 预期：reward callable 真正进入训练路径，不只是写进文件。

- [ ] public eval 与 trusted 边界分离。
  - 预期：prompt 只看到 public-safe summary。
  - 预期：trusted/private seed、raw seed id、private benchmark 不进入 prompt。
  - 预期：public accept seed count 和 eval env count 被 seal/hash 或固定记录。

## Accept / Reject 检查

每轮 ledger 追加后检查 `ledger.jsonl` 中最新 record。

- [ ] record 数量与 round 对齐。
  - 预期：第 N 轮结束后 ledger 有 N 条 record。
  - 预期：每轮只有一个最终 decision。

- [ ] accepted 逻辑合理。
  - 预期：只有 code-level policy improvement 且满足 public decision 逻辑时才 accepted。
  - 预期：parameter-only、rationale-only、diagnostic-only、engineering-only 不应被当作 policy improvement accept。
  - 预期：如果出现 diagnostic/engineering improvement record，应清楚不是 policy accept。

- [ ] decision 证据足量。
  - 预期：`candidate_surface`、`code_change_classification`、`runtime_activation_summary`、`mechanism_attribution`、`anti_tuning_decision` 能解释为什么接受或拒绝。
  - 预期：public score / public metrics / budget gate / motion quality gate 信息足量。

- [ ] current best 更新正确。
  - 预期：accepted 后下一轮 `candidate_lineage` 指向最新 accepted candidate。
  - 预期：若本轮 rejected，current-best 不应被替换。
  - 预期：lineage source hash 与 accepted candidate `components.py` 的 prompt-visible source hash 对得上。

## 跨轮继承与抛弃检查

在 Round 2、第一次 accepted 后的下一轮、最后一轮重点检查。

- [ ] public history 有边界。
  - 预期：recent outcomes 是压缩后的 public projection。
  - 预期：不携带任务私有字段、raw seed、candidate path。
  - 预期：static repair/source followup 不反复展开完整 public history。

- [ ] current-best source 继承合理。
  - 预期：有 accepted current-best 后，后续 prompt 能看到 lineage 摘要。
  - 预期：小源码可内联，大源码降级为 summary/hash/logical id。
  - 预期：LLM 被鼓励机制级继承、重写或反证，而不是小改常数。

- [ ] stale failed source 不长期粘住。
  - 预期：runtime-failed source 只作为下一轮 repair/evidence，不应在后续 accepted 后继续冒充 latest failed repair context。
  - 预期：static failed source 只服务同轮 repair/source followup。

- [ ] 多次 source request 稳定。
  - 预期：同一轮多次 source request 不会因为 followup prompt 只保留已返回 payload 而丢失完整可服务 source repository。
  - 预期：不同 round 的 source repository 彼此独立，不复用旧 round 的不一致 payload。

## Phase0-Phase9 改造专项检查

- [ ] Phase0 AuthoringContextFrame。
  - 预期：每个 provider-facing payload 都有 frame metadata、included sections、legacy alias sections、bytes budget、omitted sections。

- [ ] Phase1 SourceCatalog / CandidateLineage。
  - 预期：current-best source 只从 accepted private artifact 解析；prompt 不出现 candidate path。

- [ ] Phase2 Failed-Attempt Source Resolver。
  - 预期：failed source 和 current-best source 共用 logical id/hash/bytes/scrub/budget 协议。

- [ ] Phase3 Runtime Failure Boundary。
  - 预期：runtime failure 结束当前 round，只进入下一 round public context。

- [ ] Phase4 CandidateAuthoringLoop。
  - 预期：provider attempts、static repair、prompt artifact 保存、public-safe failure ledger 都在 authoring loop 内收敛。

- [ ] Phase5 Component Smoke / Anti-Degenerate。
  - 预期：smoke worker timeout 有效；identical/form-only 阻断；parameter-only 不被 policy accept。

- [ ] Phase6 AuthoringProviderResponse。
  - 预期：candidate-only provider 兼容；source_request 被识别为协议模式；非法 source_request 被 public-safe 拒绝。

- [ ] Phase7 Bounded Source Request / Followup。
  - 预期：resolver closed allowlist、同源校验、请求预算、累计 returned payload、followup 压缩均生效。

- [ ] Phase8 Candidate Capability Registry。
  - 预期：prompt、legacy alias、verifier/change-set/runtime 对 `reward_weight_scale` 的合同一致。

- [ ] Phase9 Runner-Owned Schema Preview。
  - 预期：默认不开放课程/环境；显式 readiness 只开放 schema preview，不改变 provider 输出合同。

## 异常触发停止条件

出现以下任一情况，应暂停 live run，保留 run root 并先审计。

- [ ] prompt 或 ledger 出现 secret、真实路径、raw seed、trusted/private eval、checkpoint path。
- [ ] provider 输出合同出现 runner patch、课程/环境写入、eval/budget/seed/checkpoint/PPO 修改。
- [ ] runtime failure 后同一 round 又调用 provider 或 runner。
- [ ] source_request 请求 catalog 外 logical id 被接受。
- [ ] source_payload hash/bytes/catalog 不一致仍被返回。
- [ ] component smoke 或 verifier 在主进程卡住。
- [ ] parameter-only / rationale-only / identical candidate 被 policy accept。
- [ ] train budget 不是 `4096 * 24 * 500`。
- [ ] public eval 或 trusted eval 边界混淆。
- [ ] prompt payload 接近或超过预算，且没有明确 omitted/summary/hash 降级。
- [ ] ledger round 数、prompt payload 数、candidate/run/eval 数明显不一致且 summary checks 未能解释。

## 每轮人工记录模板

```markdown
### Round ###

- 状态：
  - provider attempts:
  - source_request:
  - static repair:
  - runner status:
  - decision:
  - public score:

- Prompt 审计：
  - frame_kind:
  - prompt bytes:
  - visible capabilities:
  - reserved curriculum/env:
  - source lineage:
  - source payload ids:
  - forbidden/private hit:

- Candidate 审计：
  - declared_changed_components:
  - verifier-derived change set:
  - code-level / parameter-only / diagnostic-only:
  - smoke result:
  - anti-degenerate result:

- Train/Eval 审计：
  - budget:
  - candidate loaded:
  - reward callable active:
  - schedule active:
  - public metrics:
  - failure mode:

- 结论：
  - 是否符合 history-best/source request 目标：
  - 是否有退化为调参：
  - 是否有越权或泄露：
  - 是否需要暂停：
```

## 结束后总复盘

- [ ] `live_summary.json.checks` 全部解释清楚。
- [ ] `rounds_completed=15`，`prompt_payload_paths` 数量为 15。
- [ ] `ledger.jsonl` 有 15 条 record。
- [ ] 每个 accepted round 都能追到 prompt、candidate、train、eval、decision、current-best lineage。
- [ ] 每个 rejected round 都有足量 public-safe reason。
- [ ] 如果有 source_request，逐个 request/reply/followup 形成闭环。
- [ ] 如果有 static repair，latest failed source 没有泄漏路径且没有无限累积。
- [ ] 如果有 runtime failure，确认只进入下一轮反馈。
- [ ] 横向比较 15 轮：
  - 是否逐步利用 current-best。
  - 是否有机制级探索。
  - 是否出现反复小调权重。
  - 是否出现 prompt 信息不足。
  - 是否出现上下文过大。
  - 是否有真正 policy improvement。
- [ ] 给出最终建议：
  - 可以继续更长轮数。
  - 需要修 prompt/context。
  - 需要修 provider/source request。
  - 需要修 runtime adapter。
  - 需要修 accept/reject。
  - 需要停止并回滚某项能力。
