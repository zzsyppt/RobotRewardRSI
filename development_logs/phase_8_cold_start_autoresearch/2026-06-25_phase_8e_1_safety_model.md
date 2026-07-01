# Phase 8E-1：true code-level 安全模型

日期：2026-06-25

## 设计判断

Phase 8E-1 先实现安全模型，而不是直接接 reward/loss callable。原因：

- 真 code-level 候选最终会写任意 Python callable，进程内 Python 沙箱不可靠；
- 因此主防线必须是“训练进程 OS 隔离 + 独立可信 eval 进程”；
- 8D 静态检查继续作为 defense-in-depth，但不能被当成 true code-level 的主安全边界；
- 在接入真实训练前，必须先有可执行 probe 证明边界存在。

## 已实现边界

### 1. OS 隔离训练边界

新增一个 isolated training-process safety probe：

- 父进程不 import candidate code；
- candidate code 只在 child 进程中加载；
- child 进程使用 8C 同类隔离：
  - network namespace；
  - PID namespace；
  - Landlock；
  - sanitized env；
  - candidate output 可写；
  - repo 主线不可写；
  - protected repo config 不可读；
  - hidden probe file 不可读；
  - network 不可达；
  - secret-looking env 不进入 child。

训练边界 probe 不启动正式训练，只验证未来训练进程的 blast radius。

### 2. 独立可信 eval 边界

新增 trusted eval probe：

- 只接受 checkpoint；
- 显式拒绝 `candidate_dir`；
- 不 import `training_code.*`；
- 声明 eval observation/action interface frozen；
- 声明 success metric immutable；
- 输出 public-only placeholder summary。

这保证后续 eval 进程不执行候选 reward/obs/action 代码。候选最多影响训练出的 checkpoint，不能直接影响裁判。

## 新增/修改文件

```text
configs/cold_start/phase8e_safety_model.yaml
src/humanoid_agent/cold_start_lab/safety_model.py
scripts/phase8e_safety_probe.py
scripts/phase8e_safety_selftest.py
scripts/phase8e_training_safety_child.py
scripts/phase8e_trusted_eval_probe.py
src/humanoid_agent/code_sandbox/isolation.py
configs/cold_start/phase8e_loop.yaml
```

`isolation.py` 新增 `default_training_landlock_read_paths(...)`，只供未来训练进程使用；不会放宽 8C component-load 的默认最小读权限。

## 服务器验收中发现并修复的问题

服务器硬隔离 selftest 初次失败，根因不是安全模型逻辑，而是 Landlock rule 对 `/dev/nvidia*` 这类字符设备使用了目录/执行权限，`landlock_add_rule(/dev/nvidia-modeset)` 返回 `EINVAL`。

已修复：

- `_add_landlock_path_rule(...)` 对非目录/非普通文件只保留 `READ_FILE/WRITE_FILE` 权限，避免把 `READ_DIR/EXECUTE` 套到设备节点；
- `phase8e_training_safety_child.py` 中 network probe 改为单次采样，避免 `checks` 和 `probes` 日志不一致；
- `phase8e_safety_selftest.py` 增加 `--run-dir`，服务器失败时可保留完整 summary/log 供审计。

## 2026-06-26 复核：CUDA 训练边界 blocker

对“训练安全 probe 是 CPU-only 替身”的质疑成立。已修改：

- `phase8e_training_safety_child.py` 增加真实 CUDA smoke：`import torch`、查询 CUDA、在 `cuda:0` 上分配 tensor 并同步；
- `phase8e_safety_model.yaml` 增加 `require_cuda_training_smoke=true`；
- `safety_model.py` 增加三态语义：`pass` / `blocked` / `fail`。当安全隔离检查通过但 CUDA smoke 失败时，状态为 `blocked`，不再假 pass；
- 新增 `phase8e_cuda_isolation_probe.py`，用于复现不同隔离组合下的 CUDA 可用性；
- `isolation.py` 仅最小放行 `/dev/urandom` 读、`/dev/random` 读、`/dev/null` 写，避免把 `/dev/shm` 等主机共享写路径直接放进候选训练进程。

服务器实测：

```text
bare CUDA                                  => pass, torch 2.7.0+cu128, CUDA 12.8, device_count=2
unshare -Urn                               => pass
unshare -Urnfp --mount-proc                => pass
unshare -Urnfp --mount-proc + Landlock     => fail, cudaGetDeviceCount Error 304
phase8e_safety_selftest.py                 => selftest pass, safety_model_status=blocked
```

结论：

- namespace 本身与 CUDA 兼容；
- Landlock 与 CUDA 初始化不兼容，即使补 `/dev/urandom`、`/dev/null` 后仍失败；
- 8E 后续不能继续按“Landlock GPU 长训沙箱”推进；
- 在接 `RewardInputs` / `recipe_adapter` / reward-loss callable 前，必须先实现并验证新的 GPU 训练隔离方案，例如 namespace + mount-hide/等价机制；如果服务器不支持该机制，应停下来讨论，不得假装安全模型可用。

### 2026-06-26 修复：minimal rootfs / pivot_root

已选择并验证 `namespace + minimal rootfs/pivot_root`，不依赖 `bwrap`（服务器未安装），也不使用 Landlock 作为 GPU 长训边界。

实现：

- 新增 `phase8e_minimal_rootfs_probe.py`；
- 父命令使用 `unshare -Urmnfp` 建立 user/mount/network/PID namespace；
- 子进程创建 tmpfs rootfs，只 bind：
  - `/usr`、`/bin`、`/lib*`、`/dev`、`/proc`、`/sys`；
  - 当前 Isaac/Python env：`/data/yuyangliu/envs/isaaclab`；
  - `/repo/src`，只让候选 import public API；
  - `/candidate` 只读；
  - `/candidate_output` 可写；
- 不 bind `/data/yuyangliu/secrets`；
- 不 bind host repo root，也不 bind `configs/`；
- `pivot_root` 后 lazy unmount oldroot，并检查 `/oldroot` 不存在；
- `phase8e_safety_model.yaml` 的 training boundary 改为 `filesystem_policy=minimal_rootfs_pivot_root`；
- `phase8e_safety_selftest.py` 在服务器已从 `blocked` 变为 `pass`。

服务器最终验收：

```text
python scripts/phase8e_safety_selftest.py --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-26_phase8e_safety_minimal_rootfs
=> status=pass, hard_isolation=true, safety_model_status=pass

关键 checks:
cuda_training_smoke_passed=true
hidden_file_read_blocked=true
repo_hidden_config_read_blocked=true
repo_write_blocked=true
network_connect_blocked=true
secret_env_absent=true
oldroot_detached=true
```

因此，Phase 8E-1 的安全模型现在不再是 CPU-only 替身，也不再是 `blocked`；它已经有一个服务器可运行的 CUDA 训练隔离边界 PoC。下一步仍需做的是在这个边界里跑 Isaac/RSL-RL train-start，而不是直接接 LLM reward/loss。

### 2026-06-26 修复：真实 Isaac/RSL-RL train-start gate

前向提醒成立：`torch.ones(device="cuda")` 只能证明 CUDA 在 rootfs 里可用，不能证明 Isaac/RSL-RL 训练链路可用。

已完成：

- `phase8e_minimal_rootfs_probe.py` 新增 `--train-start-smoke`；
- rootfs 中只读 bind：
  - 当前 Isaac/Python env；
  - `/data/yuyangliu/vendor`（IsaacLab / Unitree editable install 的源码与 system libs）；
  - `/repo/src`；
  - 单个 runner-owned 训练脚本 `/runner_scripts/train_start_smoke.py`；
  - 单个 runner-owned 输入配置 `/runner_inputs/phase1_push_recovery.yaml`、`/runner_inputs/phase2_baseline.yaml`；
- 不 bind host repo root，不 bind `configs/` 目录，不 bind secrets；
- `HOME`、`XDG_CACHE_HOME`、`XDG_CONFIG_HOME`、`XDG_STATE_HOME`、pip/conda cache 全部指到 `/candidate_output/home`；
- train-start 子进程失败、timeout、缺 summary、缺 checkpoint 均输出 `status=reject`，不会进入 eval。

服务器验收：

```text
256 env, 1 iteration:
python scripts/phase8e_minimal_rootfs_probe.py --train-start-smoke --num-envs 256 ...
=> status=pass

4096 env, 1 iteration:
python scripts/phase8e_minimal_rootfs_probe.py --train-start-smoke --num-envs 4096 ...
=> status=pass
```

4096 env 关键结果：

```text
train_start.status=pass
train_start.exit_code=0
train_start.checkpoint_present=true
train_start.latest_checkpoint=/candidate_output/train_start_smoke/rsl_rl/model_0.pt
train_start.elapsed_seconds=65.27

cuda_training_smoke_passed=true
hidden_file_read_blocked=true
repo_hidden_config_read_blocked=true
repo_write_blocked=true
network_connect_blocked=true
secret_env_absent=true
oldroot_detached=true
```

结论：Phase 8E 后续可以把 `minimal_rootfs_pivot_root + train-start gate` 作为接 `RewardInputs` / `recipe_adapter` 前的真实运行边界。后续任何候选导致训练进程崩溃或不产出 checkpoint，都按 reject 处理，而不是尝试修复候选自毁行为。

### 2026-06-26 修复：脱敏训练配置与 `/runner_inputs` 禁词回归

`phase8e_minimal_rootfs_probe.py` 不再把原始 `configs/phase1_push_recovery.yaml` / `configs/phase2_baseline.yaml` 直接 bind 到 `/runner_inputs`。runner 先生成训练专用脱敏配置，再以只读方式 bind 给隔离训练进程：

- 删除 `evaluation`；
- 删除 public / hidden / `iid_hidden` eval profiles；
- 对 `/runner_inputs` 做禁词扫描，禁止 `1001`、`2001`、`iid_hidden`、`hidden_eval`、`hidden_audit` 通过 train-start 输入泄露；
- 脱敏输出采用 JSON 语法的 YAML 兼容文件，训练端仍可用 `yaml.safe_load` 读取，本地回归不再硬依赖 PyYAML。

这是 8E-code 的必要前置修复：候选代码仍能运行真实 train-start，但只能看到训练所需配置，不能通过 runner 输入读取 eval seed 或 iid-hidden 定义。

2026-06-26 服务器最终复验：

```text
unshare -Urmnfp python scripts/phase8e_minimal_rootfs_probe.py \
  --candidate-dir examples/cold_start_candidates/noop_candidate \
  --train-start-smoke --num-envs 4096 --max-iterations 1

=> status=pass
runner_inputs_no_protected_tokens=true
train_start_smoke_passed=true
train_start_checkpoint_present=true
train_start.latest_checkpoint=/candidate_output/train_start_smoke/rsl_rl/model_0.pt
train_start.elapsed_seconds=62.57
```

### 2026-06-26 修复：blocked 降级与 summary 篡改回归

两个审计完整性问题已修：

- `fail -> blocked` 降级不再维护硬编码安全检查白名单；现在默认要求 child summary 中除 CUDA / train-start blocker 外的所有安全检查为 `true`，因此后续新增的 `runner_inputs_no_protected_tokens`、`oldroot_detached` 等检查会自动参与降级判定；
- `phase8e_training_safety_child.py` 和 `phase8e_minimal_rootfs_probe.py` 的权威 summary 改为低层原子写入，写完后直接 `os._exit(exit_code)`，避免候选代码注册 `atexit` hook 在探针写完 summary 后把结果改成 `pass`。

已新增本地回归：

- CUDA-only 失败可降级为 `blocked`；
- `runner_inputs_no_protected_tokens=false` 或 `oldroot_detached=false` 不得降级为 `blocked`；
- 候选 `atexit` 篡改 `public_train_boundary_summary.json` 不能把 reject 伪造成 pass。

服务器复验：

- `phase8e_safety_selftest.py --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-26_safety_audit_integrity`：`status=pass, hard_isolation=true`；
- `phase8e_minimal_rootfs_probe.py --train-start-smoke --num-envs 4096 --max-iterations 1`：`status=pass`，`runner_inputs_no_protected_tokens=true`，`oldroot_detached=true`，`train_start_checkpoint_present=true`，checkpoint 为 `/candidate_output/train_start_smoke/rsl_rl/model_0.pt`。

边界说明：这仍是 8E-1 safety probe 的审计硬化。真 8E-code 训练 runner 仍应继续把候选可写输出、runner-owned checkpoint/summary、accept/reject 决策拆成不同权威路径/进程；不要把当前 probe 当成完整训练 runner。

## 验收覆盖

selftest 覆盖：

- 本地无 Linux hard isolation 时，training hard checks 明确标记 skipped；
- 服务器必须启用 hard isolation；
- attack candidate 在训练边界内尝试：
  - 读取 hidden 文件；
  - 读取 protected repo config；
  - 写 repo；
  - 访问 network；
  - 读取 secret env；
  - 写 candidate output；
- trusted eval checkpoint-only probe 通过；
- trusted eval 一旦收到 candidate_dir 就 reject；
- formal training 未执行。
- 2026-06-26 起，服务器 hard isolation 还必须报告真实 CUDA 训练边界：通过则 `status=pass`，安全检查通过但 CUDA 不可用则 `status=blocked`。
- 2026-06-26 起，minimal-rootfs train-start gate 还必须报告 `runner_inputs_no_protected_tokens=true`。

2026-06-25 服务器复验：

```text
python scripts/phase8e_safety_selftest.py --run-dir /data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-25_phase8e_safety_selftest_debug2
=> status=pass, hard_isolation=true

python scripts/phase8c_isolation_selftest.py --require-server-enforcement
=> status=pass

python scripts/phase8d_verifier_selftest.py
=> status=pass, hard_isolation=true

python scripts/phase8e_selftest.py
=> status=pass, hard_isolation=true

python scripts/phase8a_selftest.py
python scripts/phase8b_budget_selftest.py
=> status=pass
```

## 当前边界

本阶段仍未实现：

- `RewardInputs`；
- reward/loss/schedule callable contract；
- `recipe_adapter.py`；
- Isaac/RSL-RL 训练进程接入；
- 真实 public eval / OOD gate / motion gate。

这些属于后续 8E-2/8E-3。8E-1 的完成标准是安全模型能被独立验证。

## 2026-06-27 计划清理迁入摘要

从 `2026-06-23_cold_start_autoresearch_plan.md` 的 `Phase 8E-code` 段迁出以下落地状态，避免计划正文混入开发日志：

- `phase8e_safety_model.yaml` 已把训练进程和评估进程边界固化为机器可验 contract；
- isolated training-process probe 已验证候选代码只能在隔离子进程内运行，不能读 hidden/protected config、不能写 repo、不能联网、不能看到 secret env，只能写 candidate output；
- 真实 CUDA smoke 已从 `torch.ones(device=cuda)` 级探针升级为可承载后续 Isaac/RSL-RL train-start 的前置验证；
- `minimal_rootfs_pivot_root` 训练边界已成为 GPU 训练主边界；`Landlock + CUDA` 已证伪，Landlock 退回 component-load/static probe；
- `/runner_inputs` 已改为 runner 生成的训练专用脱敏配置，并加入禁词回归扫描，避免 evaluation/public/hidden profile、`1001/2001`、`iid_hidden` 暴露给候选训练代码；
- trusted eval probe 已验证 eval 边界只接受 checkpoint，拒绝 candidate_dir，不加载候选代码。

边界仍保持：8E-1 不定义 callable contract，不接 reward/loss 到正式训练 loop，不实现真实 public eval。这些分别属于 8E-2/8E-3/后续 trusted eval 前置工作。
