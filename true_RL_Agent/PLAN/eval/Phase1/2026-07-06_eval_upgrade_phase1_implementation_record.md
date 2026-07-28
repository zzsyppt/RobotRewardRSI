# Phase1 Eval CLI Isolation Implementation Record

日期：2026-07-06

## 实施结论

Phase1 已完成：eval 执行边界不再接收 candidate package path，backend 物化 eval job 时只传 `--candidate-id` 和 runner-owned checkpoint；eval summary 显式记录 candidate isolation gate；SubprocessRunner 会投影并校验 `sealed_validation_gate`。

训练边界没有退化：train argv 仍保留 `--candidate <candidate.path>`，因为训练进程需要加载 candidate package 执行候选 reward/components。eval 边界只保留 candidate id 作为记账身份，不再把 candidate package path 传入 eval CLI。

## 文件变更

- 新增 `src/true_rl_agent/runner/eval_isolation.py`
  - 通用 helper：拒绝 eval argv 中的 candidate path flags。
  - 校验 `candidate_id` 不是 path-like 值。
  - 构建和校验 `sealed_validation_gate` 的 candidate isolation 部分。
  - 统一拒绝 public eval record 中自相矛盾的 candidate contamination boolean。
  - 不包含 push recovery 的 fall、drift、motion、姿态等任务语义。
- 修改 `src/true_rl_agent/backends/isaac_lab_rsl_rl.py`
  - `_eval_argv()` / `_eval_command()` 改为接收 `candidate_id: str`，不再接收 `CandidatePackageRef`。
  - eval argv 从 `--candidate <candidate.path>` 改为 `--candidate-id <candidate_id>`。
  - eval metadata 增加 `candidate_identity: {"candidate_id": ...}`。
- 修改 `scripts/isaaclab_rsl_rl_eval.py`
  - parser 移除 `--candidate`，新增必需 `--candidate-id`。
  - IsaacLab 初始化前调用 `reject_candidate_path_arguments(sys.argv[1:])`，收到 `--candidate` / `--candidate-path` / `--candidate-dir` 直接 exit 2。
  - 解析后调用 `validate_candidate_id()`，拒绝空值、`artifact://...`、`dir/name`、`dir\name` 这类 path-like 身份。
  - pass/fail summary 记录 `candidate_package_loaded_in_eval = false`、`candidate_code_loaded_in_eval = false`、`sealed_validation_gate`。
- 修改 `src/true_rl_agent/runner/subprocess_runner.py`
  - public metrics 继续兼容 `candidate_package_loaded`。
  - 新增投影 `candidate_package_loaded_in_eval` 和 `candidate_path_argument_present`。
  - eval summary 的 top-level 与 `metrics` 若携带 candidate contamination boolean，必须为 bool 且不得为 true；否则 fail-closed 为 `RunnerExecutionError`。
  - 当 `sealed_validation_gate` 存在时校验其为 object，且 `status` 与 `candidate_isolation_status` 都必须 pass；同时要求 canonical contamination evidence 字段存在且均为 false，否则 fail-closed 为 `RunnerExecutionError`。
- 修改 `src/true_rl_agent/runner/jobs.py`
  - dry-run eval command 从 `--candidate` 改为 `--candidate-id`，避免测试和 fake 继续强化旧 eval 语义。

## 测试覆盖

新增或加厚的测试：

- `tests/unit/test_eval_isolation.py`
  - candidate path 参数 fail-closed。
  - candidate id path-like 值拒绝。
  - isolation gate pass/fail 校验。
  - 自相矛盾的 pass gate 不得携带 true contamination evidence。
  - sealed validation gate 必须携带 canonical contamination evidence。
- `tests/unit/test_isaac_lab_backend.py`
  - backend eval argv 包含 `--candidate-id`。
  - backend eval argv/command 不包含 candidate package path。
- `tests/unit/test_materialize_job_script.py`
  - materialized `runner_execution.eval.cli_argv` 不包含 candidate package path。
- `tests/unit/test_push_recovery_task.py`
  - push task backend eval argv 不包含 candidate package path。
  - eval script source 记录 `sealed_validation_gate` 和 canonical `candidate_package_loaded_in_eval`。
- `tests/unit/test_subprocess_runner.py`
  - `sealed_validation_gate` 会进入 public metrics。
  - candidate isolation fail gate 会被 runner 拒绝。
  - pass gate、top-level summary、metrics 三个入口都不得携带 true contamination evidence。
- `tests/unit/test_runner_jobs.py`
  - dry-run eval projection 使用 `--candidate-id`，不再使用 eval-side `--candidate`。

## 验证记录

使用 Python：

`C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`

红绿过程：

- `python -m unittest tests.unit.test_eval_isolation -v`
  - RED：`ModuleNotFoundError: No module named 'true_rl_agent.runner.eval_isolation'`
  - GREEN：5 tests OK
- `python -m unittest tests.unit.test_isaac_lab_backend tests.unit.test_materialize_job_script tests.unit.test_push_recovery_task -v`
  - RED：eval argv 仍缺 `--candidate-id` 且含 `--candidate <candidate.path>`
  - GREEN：29 tests OK
- `python -m unittest tests.unit.test_subprocess_runner -v`
  - RED：`sealed_validation_gate` 未投影，fail gate 未拒绝
  - GREEN：10 tests OK
- `python -m unittest tests.unit.test_runner_jobs.RunnerJobTests.test_eval_dry_run_command_uses_eval_checkpoint_only -v`
  - RED：dry-run eval command 仍为 `--candidate candidate-001`
  - GREEN：1 test OK

最终验证：

- `python -m unittest tests.unit.test_eval_isolation tests.unit.test_isaac_lab_backend tests.unit.test_materialize_job_script tests.unit.test_push_recovery_task tests.unit.test_subprocess_runner tests.unit.test_runner_jobs -v`
  - 56 tests OK
- `python -m unittest discover -s tests -v`
  - 263 tests OK
- `python -m compileall -q src scripts tests`
  - exit 0
- `git diff --check`
  - exit 0

## P1 复查加固

复查结论：`sealed_validation_gate` fail-closed 不足的问题真实存在，并且属于 Phase1 边界问题，应立即修复。旧实现只检查 `status == "pass"` 和 `candidate_isolation_status == "pass"`，会接受 `candidate_code_loaded_in_eval = true` 这类自相矛盾证据；`SubprocessRunner` 还会把 top-level contamination 字段投影到 public metrics，导致污染证据被公开保留但没有触发拒绝。

本次采用的源头修法：

- `validate_sealed_validation_gate()` 不只看状态字段，还要求 `candidate_code_loaded_in_eval`、`candidate_package_loaded_in_eval`、`candidate_path_argument_present` 三个 canonical evidence 字段存在且为 `False`。
- `candidate_package_loaded` 作为历史/兼容字段如果出现在 eval public record 中，也必须是 bool 且不得为 `True`。
- `SubprocessRunner._public_result_from_summary()` 在投影 public metrics 前，先对 `metrics` 和 top-level summary 执行同一个 contamination check，避免绕过 gate。
- 训练 summary 中的 `candidate_package_loaded = true` 没有被禁止，因为训练边界仍需要加载 candidate code；该规则只作用于 eval summary ingestion。

横向审计后没有把 `sealed_validation_gate` 的 decision 消费提前接入 `decide_candidate`，因为那属于 Phase5 的迁移范围；Phase1 只负责 eval 执行与 summary ingestion 的隔离 fail-closed。

## 边界审计

- eval 真实 argv 中的 candidate package path 已移除。
- train 真实 argv 中的 candidate package path 保留，这是训练加载 candidate code 所必需。
- shared runner helper 没有引入 push recovery 特殊语义；任务指标仍留在 task adapter / eval rollout 逻辑内。
- `trusted_seed_count` 仍只作为 Phase0 认可的 legacy constructor alias 存在；本 Phase 未新增 legacy 使用。
- `SubprocessRunner` 不强制所有 fake eval summary 都必须有 `sealed_validation_gate`，但一旦 summary 提供该 gate，就会校验并 fail-close。真实 Isaac eval script 会产出该 gate。

## Phase2+ 剩余工作

- Phase2：byte-level checkpoint safety report 进入 seal，当前 Phase1 仍只保留已有 `assert_safe_checkpoint()` 行为。
- Phase3：真正 sealed holdout rollout、raw artifact / public projection 分层。
- Phase4：public-accept 统计加厚和 standard error。
- Phase5：decision 消费 `sealed_validation_gate`，从 current-best scalar 迁移到 eval stats。
- Phase6+：behavior gates、prompt-safe eval feedback、run-level audit。
