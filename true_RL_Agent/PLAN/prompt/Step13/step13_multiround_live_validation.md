# Step13 多轮 Live/Audit 验收记录

## 结论

Step13 已完成。最终服务器 15-round live run 完整跑完，prompt 在多轮迭代中持续保留任务、边界、候选包、公开评估、runtime feedback、best/latest、真实拒绝原因和 repair feedback，没有退化成薄 API contract。

## 中途发现并修复的问题

第一次服务器 live run 使用 `live_15round_prompt_step13_20260702T183734Z` 启动后，在 round 002/003 prompt 中发现同一 payload 内存在 runtime 证据冲突：

- `runtime_feedback_summary.candidate_package_loaded` 为 `true`
- `public_history.recent_outcomes[].candidate_package_loaded` 为 `false`

根因是共享 public history 投影把 public eval metrics 中的 checkpoint-only eval 证据优先当成训练侧候选包加载证据。修复点是共享层 `agent_core.context._candidate_package_loaded()`：优先使用 train summary / candidate summary，再回退 public metrics。该修复不加入 push recovery 特殊语义。

修复提交：

- `96ecb18047a8dabc61ca17242f2c45a837172d10` - `Fix prompt history candidate load evidence`

修复后重新同步服务器并重跑 15 rounds。

## 最终服务器 run

- 服务器路径：`/data/yuyangliu/AGENT/true_RL_Agent`
- Run id：`live_15round_prompt_step13_20260702T192853Z`
- Run dir：`runs/live_15round_prompt_step13_20260702T192853Z`
- Ledger：`runs/live_15round_prompt_step13_20260702T192853Z/ledger.jsonl`
- Summary：`runs/live_15round_prompt_step13_20260702T192853Z/live_summary.json`
- 训练预算：每 round `4096 envs * 24 steps/env * 500 iterations = 49152000 steps`

`live_summary.json` 结果：

- `rounds_requested = 15`
- `rounds_completed = 15`
- `decision_outcome_counts = {"policy_improvement_accept": 3, "reject": 12}`
- `final_best_score = 0.84375`
- `provider_failure_count = 0`
- `prompt_payload_paths = 15`
- `provider_attempt_prompt_payload_paths = 21`
- `provider_attempt_records = 21`

## Round 结果

| Round | Candidate | Decision | Public score | Reason | Surface | Runtime activation | LLM curriculum |
|---|---|---:|---:|---|---|---|---|
| 001 | `push_recovery_v2` | reject | 0.0 | `required_gate_failed` | code_level | reward_callable, diagnostic, reward_weight_scale | false |
| 002 | `push_recovery_v3` | policy_improvement_accept | 0.484375 | `absolute_accept_floor` | code_level | reward_callable, diagnostic | false |
| 003 | `push_recovery_v5` | policy_improvement_accept | 0.671875 | `noise_aware_delta` | code_level | reward_callable, diagnostic | false |
| 004 | `push_recovery_v6_drift_mitigation` | reject | 0.21875 | `required_gate_failed` | code_level | reward_callable, diagnostic | false |
| 005 | `push_recovery_v7_balanced` | reject | 0.515625 | `noise_aware_delta_not_met` | code_level | reward_callable, diagnostic | false |
| 006 | `push_recovery_v8_drift_penalty` | reject | 0.5 | `noise_aware_delta_not_met` | code_level | reward_callable, diagnostic | false |
| 007 | `push_recovery_v10_lateral_stability` | reject | 0.65625 | `noise_aware_delta_not_met` | code_level | reward_callable, diagnostic | false |
| 008 | `push_recovery_v12_lateral_stability` | reject | 0.28125 | `required_gate_failed` | code_level | reward_callable, diagnostic | false |
| 009 | `push_recovery_v13_forward_tilt_emphasis` | reject | 0.671875 | `noise_aware_delta_not_met` | code_level | reward_callable, diagnostic | false |
| 010 | `push_recovery_v15_no_torch` | reject | 0.015625 | `required_gate_failed` | code_level | reward_callable, diagnostic | false |
| 011 | `push_recovery_v17_balanced_no_torch` | reject | 0.03125 | `required_gate_failed` | code_level | reward_callable, diagnostic | false |
| 012 | `push_recovery_v18_posture_vel_drift` | reject | 0.40625 | `noise_aware_delta_not_met` | code_level | reward_callable, diagnostic | false |
| 013 | `push_recovery_v19_upright_drift_penalty` | reject | 0.25 | `required_gate_failed` | code_level | reward_callable, diagnostic | false |
| 014 | `push_recovery_v20_upright_drift_penalty` | reject | 0.21875 | `required_gate_failed` | code_level | reward_callable, diagnostic | false |
| 015 | `push_recovery_v21_fall_risk_penalty` | policy_improvement_accept | 0.84375 | `noise_aware_delta` | code_level | reward_callable, diagnostic | false |

## Prompt 验收

所有 21 个 provider attempt prompt 均通过 `assert_public_only_prompt_payload()`。

所有 prompt 均包含必需顶层 sections：

- `task`
- `loop_contract`
- `public_feedback_summary`
- `runtime_feedback_summary`
- `required_candidate_package`
- `reward_inputs_contract`
- `public_eval_contract`
- `control_effectiveness_contract`
- `public_history`
- `known_good_examples`
- `selection_guidance`

多轮反馈验证：

- round 002 prompt 正确包含 round 001 的 `required_gate_failed`、`fall_dominated` 和 runtime activation。
- round 006 prompt 正确包含 round 005 的 `noise_aware_delta_not_met`、latest score `0.515625` 和 best score `0.671875`。
- round 010 prompt 在前序 9 轮时保持 `recent_len = 8`、`omitted_round_count = 1`，history 受控增长。
- round 012 prompt 在前序 11 轮时保持 `recent_len = 8`、`omitted_round_count = 3`。
- round 015 prompt 在前序 14 轮时保持 `recent_len = 8`、`omitted_round_count = 6`。

安全边界验证：

- 未发现 `private_seed`、`hidden_eval`、`trusted_eval`、`checkpoint_selector`、`runner.patch`、Windows 本地路径、server secret path 进入 prompt。
- 未暴露 `policy_accept_delta` 或 `initial_best_score`。
- `threshold` 只作为 anti-tuning notice 或候选 reward shaping 局部常量出现，不是 runner acceptance 阈值。

## Candidate / Ledger 验收

所有 15 个实际进入 ledger 的候选均为 `code_level`，未出现 parameter-only accepted。

所有 15 条 ledger：

- `feedback_safe.public_only = true`
- `curriculum.llm_controlled = false`
- `runtime_activation_summary.capability_activation_records` 非空
- `prompt_payload.path` 存在且指向实际保存的 prompt payload

被训练/评估的 15 个候选包没有命中 runner/eval/seed/budget/checkpoint/PPO/optimizer/open/subprocess/os/sys/import torch 等 forbidden scan。部分未选中的 provider 尝试候选含 `import torch`，但均被 verifier repair 拦截，没有进入训练 ledger。

## 验收判断

Step13 通过。多轮 live 证明：

- prompt 信息足量且稳定，不是薄 API contract；
- repair feedback 能进入下一次 attempt，且 public-safe；
- public history 和 repair feedback bounded，不会无限膨胀；
- 真实 reject reason 会进入后续 prompt，包括 `noise_aware_delta_not_met` 这类非简单 gate fail；
- 任务特殊语义保留在 task-owned prompt adapter / candidate component contract 下，没有为了 Step13 把 push recovery 语义写进共享 prompt assembler；
- 接受逻辑保持 runner-owned，LLM 只产生候选组件，不控制课程、eval、seed、budget、checkpoint 或 acceptance。
