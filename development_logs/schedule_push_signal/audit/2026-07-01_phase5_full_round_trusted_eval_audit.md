# Phase 5 Full-Round、隔离训练与 Trusted Eval 边界审计记录

## 审计阶段

Phase 5: Full-Round、隔离训练与 Trusted Eval 边界审计。

## 审计对象

- `src/humanoid_agent/cold_start_lab/phase8e_full_round.py`
- `src/humanoid_agent/cold_start_lab/phase8e_linked.py`
- `src/humanoid_agent/cold_start_lab/public_eval.py`
- `src/humanoid_agent/immutable_runner/safe_checkpoint.py`
- `src/humanoid_agent/immutable_runner/train_start_paths.py`
- `configs/cold_start/phase8e_loop.yaml`
- `configs/cold_start/phase8b_budget.yaml`
- `scripts/phase8e_full_round_selftest.py`
- `scripts/phase8e_linked_round_selftest.py`
- `scripts/phase8e_trusted_public_eval_selftest.py`
- `scripts/phase8e_eval_load_safety_selftest.py`
- `scripts/phase8e_safety_selftest.py`

## 关键风险

- 课程编译结果越过训练侧，影响 eval 或 accept/reject。
- candidate 或训练子进程伪造预算、checkpoint path、checkpoint safety 或 eval 结果。
- full-round 在 provider、compiler、training、trusted eval 任一失败后仍继续执行后续阶段。
- trusted eval 导入 candidate code 或信任 child-reported checkpoint safety。
- public result 暴露 hidden/private 信息、原始 seed 或 runner-private 状态。
- failure classification 混淆 training failure、trusted eval failure 和 checkpoint safety failure。

## 已检查内容

- 当前本地与服务器均同步到 `fc04bb4 fix: bound phase8e diagnostic summaries`。
- full-round 顺序为 provider/8D gate -> curriculum compiler -> fixed-budget isolated training -> checkpoint resolver -> public rollouts -> linked decision。
- fixed budget 来自 runner-owned `configs/cold_start/phase8b_budget.yaml`，训练命令使用 `num_envs=4096`、`max_iterations=500`、`steps_per_iteration=24`，并校验 `planned_env_steps=49152000`。
- training 使用 compiled phase1/phase2 config，public rollout 使用原始 eval config，保持课程只影响训练侧。
- checkpoint eval 只接受 `/runner_artifacts/train_start_smoke/sealed_checkpoints/model_<n>.pt`，并在 trusted side 重新 safe-load。
- public result projection 会拒绝 hidden/iid source token，并不投影原始 public dev/public accept seed。
- provider reject、compiler reject、training no-checkpoint、checkpoint safe-load failure、trusted eval failure 的中止位置均做了定向 probe。
- 服务器 IsaacLab Python 环境下验证了 torch checkpoint safety、linked round 和 trusted public eval probe。

## 证据路径与命令

- `codegraph explore "src/humanoid_agent/cold_start_lab/phase8e_full_round.py run_phase8e_full_round ..."`
- `codegraph explore "resolve_train_start_checkpoint_for_eval run_phase8e_linked_round checkpoint_safety_report ..."`
- `python scripts/phase8e_full_round_selftest.py`
- `python scripts/phase8e_linked_round_selftest.py`
- `python scripts/phase8e_trusted_public_eval_selftest.py`
- `python scripts/phase8e_safety_selftest.py`
- Server: `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_eval_load_safety_selftest.py`
- Server: `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_linked_round_selftest.py`
- Server: `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_trusted_public_eval_selftest.py`
- Server: `/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_safety_selftest.py`
- Custom probe: forged budget fields are rejected by `build_runner_owned_budget_summary`.
- Custom probe: forged checkpoint paths under candidate output, host absolute path, traversal, and invalid filename are rejected.
- Custom probe: hidden eval rollout summary fails public projection without leaking hidden token or success rate.
- Custom probe: compiler reject stops before training.
- Custom probe: trusted eval/public result failure is currently classified as `failed_safety_gate`.

## 发现的问题

### P1: trusted eval/public result failure is mislabeled as checkpoint safety failure

`src/humanoid_agent/cold_start_lab/phase8e_full_round.py::_classify_linked_error()` treats the generic marker `"trusted eval probe failed"` as a safety marker. As a result, linked errors such as `trusted eval probe failed: public_result_status_not_pass` and `trusted eval probe failed: budget_accounting_status_not_pass` produce:

- top-level `status`: `failed_safety_gate`
- `failure_classification.reason`: `trusted_eval_gate_failed`

This conflates public-result/trusted-eval gate failure with checkpoint safety failure. The accept/reject boundary remains safe because the candidate is not accepted, but Phase 5 requires distinct failure classification for trusted eval failure vs checkpoint safety failure. This can mislead automation, triage, and next-round audit records.

Suggested fix: classify only checkpoint path/safe-load/candidate-dir-forbidden/public-result-file-required errors as `failed_safety_gate`; classify ordinary trusted eval probe failures as `reject` with reason `trusted_eval_gate_failed`, unless the trusted eval summary explicitly indicates checkpoint safety failure.

## 风险等级

P1 for audit correctness and operational triage. It does not create an accept bypass, but it violates Phase 5 failure-classification semantics.

## 建议修复顺序

1. Add a regression test in `scripts/phase8e_full_round_selftest.py` for trusted eval/public result failure classification.
2. Narrow `_classify_linked_error()` so generic `"trusted eval probe failed"` is not a safety marker.
3. Keep checkpoint safe-load and sealed checkpoint contract failures mapped to `failed_safety_gate`.
4. Re-run local full-round/linked/public eval selftests and server torch-backed checkpoint/eval selftests.

## 结论

Phase 5 的核心安全边界整体成立：课程编译结果只进入训练侧，预算由 runner-owned Phase 8B config 决定，checkpoint path 与 checkpoint safety 在 trusted side 重新验证，trusted eval 不导入 candidate code，public result projection 不暴露 hidden/private 信息。

但当前存在一个 failure classification bug：普通 trusted eval/public result gate failure 被顶层 full-round summary 标成 `failed_safety_gate`。建议先修复该问题后再进入 Phase 6。
