# Phase 8E-4：训练/评估分进程与 checkpoint 权威边界

## 目标

本阶段落实 Phase 8E-code 计划中的“训练/评估分进程”边界，但不启动正式长训、不进入候选池、不做 Phase 9。

核心目标：

1. 候选 code-level callable 只能影响隔离训练进程；
2. eval/accept 所需 checkpoint 必须由未加载候选代码的可信进程重新验证；
3. 训练子进程的 summary 只能作为 public 证据，不能自证 checkpoint 可被 trusted eval 加载。

## 设计判断

8E-4 的设计是合理且必要的。Phase 8E-3 已经把候选 reward callable 接入真实 train-start，但候选代码与训练 runner 同进程执行，因此任何“训练进程内 checkpoint 安全检查”都只能是 advisory：候选可以通过 monkeypatch、atexit 或污染模块全局状态伪造结果。

正确边界不是继续堆 Python 内部 guard，而是把权威结论移到候选污染不到的父进程/可信 eval 进程：

- 训练子进程：产生 checkpoint 与 public train summary；
- minimal-rootfs probe 父进程：在加载候选组件前验证 train-start checkpoint；
- safety model 父进程：按 host 路径再次验证隔离内 checkpoint；
- trusted eval 进程：正式 eval 前仍必须独立 safe-load gate。

## 本阶段改动

### 1. `phase8e_minimal_rootfs_probe.py`

- train-start 结束后新增父进程 `checkpoint_safety_report()`；
- 通过条件收紧为：
  - 子训练进程 exit code 为 0；
  - 子训练 summary status 为 `pass`；
  - checkpoint 文件存在；
  - 父进程 `weights_only=True` safe-load 通过；
- 新增 reject reason：`train_start_checkpoint_failed_safe_load_gate`；
- 将 candidate package 加载移动到 train-start checkpoint gate 之后，避免候选代码污染 probe 父进程里的 checkpoint safety helper；
- summary 增加：
  - `checkpoint_safe_load`;
  - `checkpoint_safety`;
  - `checks.train_start_checkpoint_safe_load`。

### 2. `safety_model.py`

- 新增 host 侧父进程复验：
  - 将隔离内 `/runner_artifacts/...` 映射为 `minimal_run_dir/runner_artifacts/...`；
  - 将隔离内 `/candidate_output/...` 映射为 `minimal_run_dir/candidate_output/...`；
  - 用从未 import candidate 的父进程运行 `checkpoint_safety_report()`；
- 若 child summary 报 pass 但父进程 safe-load 失败，则 training boundary 直接 fail，不允许进入可信 eval；
- `project_training_boundary()` 增加 `train_start_checkpoint_safety` 投影，方便审计证据链。

### 3. 回归测试

`phase8e_eval_load_safety_selftest.py` 增加 minimal-rootfs 父进程路径映射回归：

- benign checkpoint：父进程 safe-load 通过；
- malicious pickle payload checkpoint：即使 child summary 指向存在的 checkpoint，父进程也必须拒绝。

## 边界说明

本阶段没有开放：

- eval observation/action wrapper；
- success metric；
- checkpoint selection policy；
- PPO loop hook / optimizer hook；
- 正式 LLM-authored 多轮训练；
- Phase 9 candidate pool。

这仍是 Phase 8E 内部的安全边界硬化，目标是让后续“真 code-level reward/loss callable”进入训练时不会把候选进程内的自证结果误当成裁判结论。

## 验收记录

本地验收：

```text
python -m py_compile scripts/phase8e_minimal_rootfs_probe.py scripts/phase8e_eval_load_safety_selftest.py src/humanoid_agent/cold_start_lab/safety_model.py scripts/phase1_push_recovery_train_smoke.py
=> pass

python scripts/phase8e_selftest.py
=> pass

python scripts/phase8e_safety_selftest.py
=> pass（本地无 torch/hard isolation 时 safety_model fail-closed，selftest 通过）

python scripts/phase8e_contract_selftest.py
=> pass

python scripts/phase8e_training_adapter_selftest.py
=> pass
```

服务器验收：

```text
/data/yuyangliu/envs/isaaclab/bin/python -m py_compile ...
=> pass

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_eval_load_safety_selftest.py
=> {"status": "pass", "test": "phase8e_eval_load_safety_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_training_adapter_selftest.py
=> {"status": "pass", "test": "phase8e_training_adapter_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": true}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_safety_selftest.py --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_phase8e4_safety_selftest
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": true, "safety_model_status": "pass"}
```

真实 train-start：

```text
unshare -Urmnfp /data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_minimal_rootfs_probe.py \
  --candidate-dir examples/cold_start_candidates/declarative_curriculum_candidate \
  --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_phase8e4_train_start \
  --train-start-smoke \
  --num-envs 4096 \
  --max-iterations 1 \
  --seed 801 \
  --device cuda:0
=> {"summary_file": "/candidate_output/minimal_rootfs_summary.json", "status": "pass"}
```

## 2026-06-27 计划清理迁入摘要

从 `2026-06-23_cold_start_autoresearch_plan.md` 的 `Phase 8E-code` 段迁出以下落地状态，避免计划正文混入开发日志：

- `phase8e_minimal_rootfs_probe.py` 的 train-start checkpoint gate 已前移到 probe 进程加载候选组件之前执行；候选训练子进程只产出 checkpoint 与 public summary，不能用同进程 monkeypatch 伪造可进入 eval 的 checkpoint 安全结论；
- train-start 通过条件已从“子 summary pass + checkpoint 存在”收紧为“子 summary pass + checkpoint 存在 + 父进程 `weights_only=True` safe-load 通过”；
- `safety_model.py` 外层父进程已增加 host 侧复验：把隔离内 `/runner_artifacts/...` 路径映射回 `minimal_run_dir/runner_artifacts/...`，再由从未 import candidate 的父进程重新运行 `checkpoint_safety_report()`；
- 训练进程内 `child_reported_checkpoint_safe_load` 仍仅作为 advisory；权威判定来自 minimal-rootfs 父进程与 trusted eval 进程。

边界仍保持：这只是 train-start/checkpoint 安全链路硬化；真实 public eval metric、闭环 orchestrator、LLM provider -> candidate package 自动落盘仍是进入首轮从 0 自演进测试前的阻断前置工作。

关键字段：

```text
cuda_training_smoke_passed=true
runner_inputs_no_protected_tokens=true
oldroot_detached=true
train_start_checkpoint_present=true
train_start_checkpoint_safe_load=true
phase8e_candidate_runtime.step_calls=24
phase8e_candidate_runtime.reward_term_calls=24
phase8e_candidate_runtime.diagnostic_calls=1
```

对抗候选回归：

```text
candidate: patch torch.save + forge child_reported_checkpoint_safe_load=true
run: /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_phase8e4_checkpoint_tamper_fixed
result:
  status=reject
  train_start_status=reject
  checkpoint_present=true
  child_claim=true
  checkpoint_safe_load=false
  reject_reason=train_start_checkpoint_failed_safe_load_gate
```

## 2026-06-27 追加审计：train-start parent gate 路径收窄

审计结论：

- 路径 containment 问题真实存在：`_isolated_training_path_to_host()` 原实现接受 `/candidate_output/...`、`/runner_artifacts/../candidate_output/...`，并且在 host 侧绝对路径存在时会直接读取该文件。这会让不可信 child summary 牵引可信父进程读取错误 checkpoint，从而绕开 train-start parent gate 的“固定 runner 产物”语义。
- `runner_artifacts` 在训练进程内可写是当前 true code-level 训练的边界事实，不单独当作 crash/correctness bug；它的可信性不来自不可写，而来自固定 sealed checkpoint 路径、父进程 safe-load、以及后续 trusted eval gate。
- 暂未扩大改动到 Phase 9/候选池/正式长训；本次只收窄 train-start checkpoint 的可信路径。

修复：

- 新增 `src/humanoid_agent/immutable_runner/train_start_paths.py`，统一定义 train-start checkpoint 路径规则。
- 只接受 `/runner_artifacts/train_start_smoke/sealed_checkpoints/model_<n>.pt`。
- 拒绝 `/candidate_output`、path traversal、host 绝对路径、嵌套文件、非 `model_<n>.pt` 命名。
- `safety_model._parent_train_start_checkpoint_safety()` 与 `phase8e_minimal_rootfs_probe._run_train_start_smoke()` 共用该规则。
- `phase8e_eval_load_safety_selftest.py` 增加回归：修复前服务器实测失败于 `/candidate_output/train_start_smoke_public/model_0.pt` 被错误接受；修复后通过。

验收：

```text
local:
python -m py_compile scripts/phase8e_minimal_rootfs_probe.py scripts/phase8e_eval_load_safety_selftest.py src/humanoid_agent/cold_start_lab/safety_model.py src/humanoid_agent/immutable_runner/train_start_paths.py
=> pass

python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

python scripts/phase8e_training_adapter_selftest.py
=> {"status": "pass", "test": "phase8e_training_adapter_selftest"}

python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": false}
```

```text
server:
/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_eval_load_safety_selftest.py
=> {"status": "pass", "test": "phase8e_eval_load_safety_selftest"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_safety_selftest.py --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_pathfix_safety_selftest
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": true, "safety_model_status": "pass"}

unshare -Urmnfp /data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_minimal_rootfs_probe.py \
  --candidate-dir examples/cold_start_candidates/declarative_curriculum_candidate \
  --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_pathfix_train_start \
  --summary-file minimal_rootfs_summary.json \
  --repo-root /data/yuyangliu/AGENT \
  --train-start-smoke \
  --num-envs 4096 \
  --max-iterations 1 \
  --seed 811 \
  --device cuda:0
=> {"summary_file": "/candidate_output/minimal_rootfs_summary.json", "status": "pass"}
```

关键字段：

```text
status=pass
runner_inputs_no_protected_tokens=true
train_start_smoke_passed=true
train_start_checkpoint_present=true
train_start_checkpoint_safe_load=true
latest_checkpoint=/runner_artifacts/train_start_smoke/sealed_checkpoints/model_0.pt
trusted_latest_checkpoint=/runner_artifacts/train_start_smoke/sealed_checkpoints/model_0.pt
checkpoint_safe_load=true
reject_reason=null
```

## 2026-06-27 追加审计：checkpoint unsafe 不得被 CUDA blocker 覆盖

审计结论：

- 问题真实存在：`run_minimal_rootfs_training_boundary_probe()` 原状态顺序是先用 parent checkpoint safety 把 `pass` 改成 `fail`，随后又允许 `_child_security_checks_passed(child_summary) && _child_cuda_probe_failed(child_summary)` 把任意 `fail` 降级成 `blocked`。
- 这会让候选伪造 child summary 中的 CUDA smoke fail，同时 parent 真实检查发现 checkpoint unsafe 时，最终状态从安全失败变成可重试 blocker。
- 风险级别为低到中：trusted eval 仍会独立 safe-load checkpoint，不会因此执行候选 payload；但 safety model 的分类会误导调度层，把安全失败当成环境 blocker。

修复：

- 增加 `parent_checkpoint_unsafe` 局部判定。
- 只有 `not parent_checkpoint_unsafe` 时，CUDA smoke fail 才允许把 `fail` 降级成 `blocked`。
- `scripts/phase8e_safety_selftest.py` 增加回归：模拟 child 自报安全、CUDA fail，但 trusted parent checkpoint report 为 unsafe；修复前状态为 `blocked`，修复后保持 `fail`。

验收：

```text
local:
python scripts/phase8e_safety_selftest.py
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": false, "safety_model_status": "fail"}

python -m py_compile scripts/phase8e_safety_selftest.py src/humanoid_agent/cold_start_lab/safety_model.py
=> pass

python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

python scripts/phase8e_training_adapter_selftest.py
=> {"status": "pass", "test": "phase8e_training_adapter_selftest"}

python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": false}
```

```text
server:
/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_safety_selftest.py --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_blocked_checkpoint_fix_safety_selftest
=> {"status": "pass", "test": "phase8e_safety_selftest", "hard_isolation": true, "safety_model_status": "pass"}

/data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_eval_load_safety_selftest.py
=> {"status": "pass", "test": "phase8e_eval_load_safety_selftest"}

unshare -Urmnfp /data/yuyangliu/envs/isaaclab/bin/python scripts/phase8e_minimal_rootfs_probe.py \
  --candidate-dir examples/cold_start_candidates/declarative_curriculum_candidate \
  --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-27_blocked_checkpoint_fix_train_start \
  --summary-file minimal_rootfs_summary.json \
  --repo-root /data/yuyangliu/AGENT \
  --train-start-smoke \
  --num-envs 4096 \
  --max-iterations 1 \
  --seed 821 \
  --device cuda:0
=> {"summary_file": "/candidate_output/minimal_rootfs_summary.json", "status": "pass"}
```
