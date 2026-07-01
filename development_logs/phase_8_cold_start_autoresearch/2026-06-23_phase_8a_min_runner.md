# Phase 8A：最小代码沙盒 MVP

日期：2026-06-23

## 范围

Phase 8A 只建立最小 code-candidate package 与 immutable runner 链路。

本阶段不做：

- LLM-authored candidate code；
- OS/container 级隔离；
- 完整 env-step budget accounting；
- public/eval accept-reject；
- 多候选池或 Git evolution。

这些属于 Phase 8B-8E。

## 设计决策

### 0. 安全姿态说明

Phase 8A 的 loader 会在当前进程内 import / execute candidate Python module。
因此 Phase 8A 不是安全沙盒，只是最小 candidate schema 和 runner plumbing。

当前安全性来自流程约束：

- 只运行 repo 内我们手写的 no-op candidate；
- LLM-authored candidate 必须等 Phase 8C 的 OS/进程隔离后才允许运行；
- no-op 检查只能验证返回结构，不能阻止候选代码在 import/builder 阶段执行副作用；
- Phase 8A 的 loader 不能被理解成 RCE 防线。

这个姿态必须保持到 Phase 8C 完成。

### 1. candidate 只提交组件，不控制 runner

新增候选接口：

```text
examples/cold_start_candidates/noop_candidate/
  candidate_manifest.yaml
  training_code/components.py
```

候选 entrypoint：

```python
def build_training_components(task_spec, base_env_cfg, base_agent_cfg) -> TrainingComponents
```

Phase 8A 只接受 no-op components。这样可以先验证代码加载、schema、runner 调用和失败处理，不提前开放训练行为修改。

### 2. loader 不是安全沙盒

新增：

```text
src/humanoid_agent/code_sandbox/
```

这个包只负责 schema 和加载。它不是 OS 级隔离。计划中的真正隔离仍在 Phase 8C。

因此本阶段硬门是：

- `llm_authored: true` 直接 reject；
- 候选 provenance 不能只信 manifest 自报，必须匹配 runner 配置中的外部 human-candidate allowlist；
- allowlist 同时校验候选目录、candidate id、manifest sha256 和 entrypoint sha256；
- 只运行我们手写的 no-op demo candidate；
- candidate 不允许改变 reward/loss/schedule/observation/diagnostics。

### 3. runner 固定执行协议

新增：

```text
scripts/phase8a_min_runner.py
configs/cold_start/phase8a_min.yaml
```

runner 固定：

- task；
- seed；
- device；
- `num_envs`；
- `max_iterations`；
- Phase1/Phase2 config；
- checkpoint selection；
- TensorBoard/log/summary 输出位置。

候选不能选择 checkpoint，不能运行 eval，不能改变预算。
Phase 8A config 中的 `budget_contract` 只是 summary 元数据和未来契约，不是完整 runtime
budget accounting。真正的 `total_env_steps` / grad step / optimizer step cap 要到 Phase 8B
实现。

### 4. 并行 envs 目标改为 4096

Phase 8A 配置将 train-start smoke 的并行 env 数设为：

```text
num_envs = 4096
```

原因：

- Phase -1 已验证官方 Ant 4096 envs 可启动；
- 当前服务器 GPU 如果能承载 G1 4096 envs，后续 cold-start 搜索可以显著提高采样吞吐；
- 预算公平性仍然要在 Phase 8B 用 `total_env_steps` 约束，不能把 4096 envs 与固定 iteration 混为一谈。

验收要求：

- 服务器必须跑一次 4096 envs、1 iteration、no-op candidate train-start smoke；
- 若失败，不能声称 4096 已可用，应回退或标注为未通过。

## 新增文件

```text
configs/cold_start/phase8a_min.yaml
examples/cold_start_candidates/noop_candidate/
scripts/phase8a_min_runner.py
scripts/phase8a_selftest.py
src/humanoid_agent/code_sandbox/
```

## 验收标准

本阶段完成必须满足：

1. no-op candidate 能通过 loader；
2. `llm_authored: true` candidate 被 reject；
3. runner dry-run 能写出可复现 `commands.sh` 和 summary；
4. runner command 中固定 `--num-envs 4096`；
5. 本地 selftest / py_compile / diff check 通过；
6. 服务器 selftest 通过；
7. 服务器 4096 envs train-start smoke 通过后，4096 才算正式可用；
8. 旧 Phase6/7 脚本不受影响。

## 验收结果

本地验收：

- `scripts/phase8a_selftest.py`：通过；
- Phase8A CLI dry-run：通过，summary 中 `num_envs=4096`、`executed=false`；
- `py_compile`：通过；
- `git diff --check`：通过；
- Phase7 checkpoint selection selftest：通过；
- 新增 Phase8A 代码路径泄露关键词扫描：无 `DEEPSEEK_API_KEY`、hidden/iid eval seed 等命中。

服务器验收：

- GPU：2 x NVIDIA GeForce RTX 5090，每张约 32GB 显存；
- `scripts/phase8a_selftest.py`：通过；
- `py_compile`：通过；
- Phase7 checkpoint selection selftest：通过；
- `git diff --check`：通过；
- 4096 envs、1 iteration、no-op candidate train-start smoke：通过。

服务器 smoke 运行目录：

```text
/data/yuyangliu/runs/phase_8_cold_start_autoresearch/2026-06-23_phase8a_noop_4096_smoke
```

关键结果：

- runner summary：`status=pass`；
- smoke：`executed=true`；
- smoke exit code：`0`；
- train summary：`status=pass`；
- train summary：`num_envs=4096`；
- train summary：`max_iterations=1`；
- train summary：`checkpoint_count=1`；
- latest checkpoint：`model_0.pt`；
- checkpoint 文件大小约 9.6MB。

结论：

当前服务器可以承载 Phase8A 的 4096 并行 env train-start smoke。因此 Phase8A 默认 `num_envs=4096` 可以保留。后续 Phase8B 仍必须按 `total_env_steps` 做预算公平性，而不是继续用固定 iteration 当公平单位。

注意：这个结论只证明 4096 envs 能启动、跑 1 iteration、并写出 checkpoint；它不等于长训容量已经验证。4000-iteration 或更长预算下的显存碎片、日志增长和 OOM 风险应归入 Phase 8B 的完整预算/容量验收。

## Review 问题处理记录

- V1：真实存在，已修改。Phase 8A 不再只依赖 manifest 的 `llm_authored` 自报；runner 在执行候选代码前先检查外部 trusted human candidate allowlist，并校验 path / candidate id / manifest hash / entrypoint hash。
- V2：真实存在，记录为安全姿态。Phase 8A loader 不是安全沙盒，no-op 检查发生在候选代码执行之后，只能防返回结构，不能防 RCE。该问题不阻断 8A，因为 8A 只运行手写 no-op；真正防线留给 Phase 8C。
- V3：真实存在，已修改。`find_latest_checkpoint_from_summary` 默认不再 fallback 到旧 summary 的 `training.latest_checkpoint`；没有可扫描 `log_dir` 时 fail closed。仅手动可信历史恢复可显式开启 legacy fallback。
- V4：低风险，已记录。`budget_contract` 在 Phase 8A 只是 metadata，不是 runner runtime accounting；完整预算强制执行属于 Phase 8B。
- V7：低风险，已记录。4096 smoke 只证明 1-iteration 启动能力，不证明长训容量。
- `commands.sh` 中默认 `/data/yuyangliu/AGENT` 是服务器真实 repo 路径，当前不需要修改。

补充：后续 review 指出只 hash manifest + entrypoint 仍可能漏掉被 import 的辅助文件。已在 Phase 8B 前置修复中把 provenance 扩展为整棵 `training_code/` tree hash。
