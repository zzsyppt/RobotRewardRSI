# Phase 8E-6-1：真实公开评测 metric

日期：2026-06-27

## 目标

把 `phase8e_trusted_eval_probe.py` 从“只证明 trusted eval 边界存在”的 placeholder，推进到能产出 Phase 8E decision 可消费的真实 public_result。

本阶段不启动正式训练、不调用 LLM、不接候选目录、不做 OOD gate 研究量；只接入可信评测 metric 的最小闭环接口。

## 成功评定设计

首轮任务成功不直接由 reward 决定，而由固定 public push rollout 的成功率与安全 gate 决定：

1. 主指标：每个 runner-owned public rollout summary 的 `aggregate.success_rate`。
2. split 统计：
   - `public_dev.success_rate_stats`
   - `public_accept.success_rate_stats`
   - per-run `failure_breakdown`
   - per-run `condition_flag_breakdown`
   - per-run public secondary metrics
3. `budget_accounting` gate：
   - 只从 runner-owned training/budget summary 投影；
   - 不接受候选输出的 budget 结论。
4. `motion_quality_gate`：
   - 本阶段是最小 sanity gate；
   - 检查 rollout summary `status == pass`；
   - 检查 success rate 与 secondary metrics 有限；
   - 检查 action/action-rate/drift/姿态指标不超过宽松上限；
   - 失败时 public_result 仍表示 eval 已完成，但 decision 会因 required gate fail 而拒绝。
5. `ood_stress_gate`：
   - 本阶段输出 `status: not_run` 作为诊断；
   - 不放入首轮 `required_pass_gates`；
   - 真正 OOD 分布与阈值留到后续 9A/评测增强阶段。

## 实现

新增：

- `src/humanoid_agent/cold_start_lab/public_eval.py`
  - 纯 Python public_result 投影模块；
  - 不 import Isaac、torch 或 candidate code；
  - 从可信 rollout summaries 构建 Phase 8E public_result；
  - 主动剥离 raw seed/path 等不该进入 prompt 的信息。

- `scripts/phase8e_trusted_public_eval_selftest.py`
  - 覆盖 public_result schema；
  - 覆盖 public-only 禁词回归；
  - 覆盖 motion gate fail 与 eval completion 分离；
  - 本地无 torch 时跳过 CLI checkpoint 子测，服务器环境继续验收。

修改：

- `scripts/phase8e_trusted_eval_probe.py`
  - 保留 `--candidate-dir` 禁止；
  - 保留 trusted checkpoint safety gate；
  - 新增 `--public-dev-summary`、`--public-accept-summary`、`--budget-summary`、`--public-result-file`；
  - 在真实 metric 模式下写出 public_result，不再写 placeholder。

- `configs/cold_start/phase8e_loop.yaml`
  - 首轮 `required_pass_gates` 收口为已真实产出的 `budget_accounting` 和 `motion_quality_gate`；
  - `ood_stress_gate` 暂不作为必过 gate。

## 验收

本地 Codex bundled Python：

```text
python scripts/phase8e_trusted_public_eval_selftest.py
=> {"status": "pass", "test": "phase8e_trusted_public_eval_selftest", "trusted_eval_probe_cli": "skipped_no_torch"}

python scripts/phase8e_selftest.py
=> {"status": "pass", "test": "phase8e_selftest", "hard_isolation": false}

python scripts/phase8e_contract_selftest.py
=> {"status": "pass", "test": "phase8e_contract_selftest"}

python -m compileall -q src/humanoid_agent/cold_start_lab/public_eval.py scripts/phase8e_trusted_eval_probe.py scripts/phase8e_trusted_public_eval_selftest.py
=> pass
```

说明：本地 bundled Python 没有 torch，所以 CLI checkpoint safety 的真实 `torch.load(weights_only=True)` 路径需要在服务器环境继续跑。

## 后续边界

- 本阶段完成的是可信 public metric 投影和 gate schema，不是 full-budget training。
- public eval runner 后续要接 sealed checkpoint 的真实 rollout 生产流程。
- OOD gate 不应在没有真实 OOD 分布和阈值前被写成 required gate。

## 2026-06-27 追加安全修复

代码审计发现两个真实缺口：

1. `phase1_push_recovery_eval.py` 的 checkpoint policy 在 `runner.load(checkpoint)` 前没有先跑 `assert_safe_checkpoint`。这意味着如果 rollout 直接加载候选产出的恶意 checkpoint，RSL-RL 内部的 `torch.load(weights_only=False)` 可能在可信评测进程中执行 pickle payload。
2. `public_eval.py` 的 summary 聚合器本身是 seed/profile agnostic。如果后续编排误把 iid/hidden summary 传进来，它会把 hidden success rate 投影成 public_result。

处理：

- 在 `runner.load(args_cli.checkpoint)` 前加入 `assert_safe_checkpoint(args_cli.checkpoint)`；
- 在 `build_phase8e_public_result()` 投影前扫描输入 summary 的 key/string value，发现 `iid_hidden` / `hidden_eval` / `hidden_audit` 等来源标记时直接拒绝该 split，不投影 success_rate，也不把 forbidden token 回写到 public output；
- 回归测试加入：
  - hidden/iid summary 不得投影 `0.99` success；
  - `phase1_push_recovery_eval.py` 必须在 `runner.load()` 前出现 `assert_safe_checkpoint()`。

仍保持的设计口径：

- `budget_accounting` 当前是 runner-owned summary 投影，不声称不可绕过 runtime counter；
- `motion_quality_gate` 仍是首轮 sanity gate，不提前做复杂 OOD/motion 研究量。
