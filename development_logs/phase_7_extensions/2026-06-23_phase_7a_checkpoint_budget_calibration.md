# Phase 7A：checkpoint selection 与预算校准收口

日期：2026-06-23

## 范围

Phase 7A 只处理进入代码沙盒前必须收口的两个工程问题：

1. 修复 checkpoint selection 的数字排序 bug。
2. 固化 4000-iteration calibration 的观察结论。

本阶段不改变 Phase6 默认训练预算，不进入 code-level sandbox，也不改变 accept/reject 协议。

## 问题确认

旧实现使用：

```python
sorted(log_dir.glob("model_*.pt"))[-1]
```

这是字符串排序。训练超过 999 iteration 后，可能把 `model_900.pt` 排在 `model_1999.pt` 或 `model_3999.pt` 后面，导致 eval 加载错误 checkpoint。

这个问题会污染两处下游链路：

- `phase2_baseline_variance.sh` 从 `train_summary.json` 取 `training.latest_checkpoint` 后直接 eval；
- `phase6_run_candidate_public_eval.sh` 也从 summary 取 checkpoint 后直接 eval。

因此修复不能只改训练 summary，还要把 selection 抽成公共 runner 逻辑。

## 实现

新增公共模块：

```text
src/humanoid_agent/immutable_runner/checkpoints.py
```

规则：

- 只接受 `model_<integer>.pt`；
- 按 `<integer>` 数字最大值选择最新 checkpoint；
- 非标准命名忽略，不作为 fallback；
- 旧 summary 中如果 `latest_checkpoint` 被污染，优先使用 `training.log_dir` 重新扫描；
- 如果没有合法 checkpoint，返回 `None`，调用方 fail closed。

接入点：

- `scripts/phase1_push_recovery_train_smoke.py`
  - summary 中写入数字选择后的 `latest_checkpoint`；
  - 增加 `latest_checkpoint_iteration` 便于后续审计。
- `scripts/phase2_baseline_variance.sh`
  - eval 前调用公共 selector。
- `scripts/phase6_run_candidate_public_eval.sh`
  - eval 前调用公共 selector。

新增自测：

```text
scripts/phase7_checkpoint_selection_selftest.py
```

覆盖：

- `model_900.pt`、`model_1999.pt`、`model_3999.pt` 选择 `3999`；
- 非标准命名如 `model_final.pt`、`model_abc.pt` 不参与选择；
- 旧 summary 中 `latest_checkpoint=model_900.pt` 但 log_dir 有 `model_3999.pt` 时，重新选择 `3999`；
- 无 checkpoint 或只有非法命名时 fail closed。

## 4000-iteration calibration 记录

实验目录：

```text
/data/yuyangliu/runs/phase_7_budget_calibration/2026-06-22_round3_accepted_4000iter
```

实验对象：

- Phase6 Round3 accepted 配置；
- fresh random policy；
- seeds：101 / 102 / 103；
- 512 envs；
- 4000 iterations；
- isolated run dir；
- TensorBoard：`http://127.0.0.1:16008/#timeseries`。

完成情况：

- seed 101：完成 4000/4000，exit 0；
- seed 102：完成 4000/4000，exit 0；
- seed 103：完成 4000/4000，exit 0。

关键观察：

- 500 iteration 明显太短；
- 约 600-800 iteration 开始学会基本站住；
- 约 2000 iteration 后，timeout / bad_orientation 等 success 相关指标基本稳定；
- 2000-4000 iteration reward 仍继续提升，但 success 相关指标已经接近饱和。

尾部标量摘要：

| seed | iterations | reward tail | episode length tail | timeout tail | bad_orientation tail |
|---:|---:|---:|---:|---:|---:|
| 101 | 4000 | ~10.14 | ~199.08 | ~0.993 | ~0.0067 |
| 102 | 4000 | ~10.72 | ~199.88 | ~0.997 | ~0.0031 |
| 103 | 4000 | ~10.22 | ~199.77 | ~0.997 | ~0.0033 |

## 结论

1. Phase6/Phase7 现有链路不应继续使用 500-iteration 结果判断候选质量。
2. 2000 iteration 是当前任务 success 指标开始稳定的下限候选；4000 iteration 可作为预算校准上界参考。
3. 进入 code-level sandbox 后，不能把“4000 iterations”直接作为公平预算单位；Phase8B 应转换为 env-step budget，并记录 grad/update/optimizer-step 等次级 cap。
4. Phase7A 不改默认预算，避免把仍未正式讨论完的预算策略固化到 Phase6 协议里。

## 验收

本阶段验收条件：

- checkpoint selector 自测通过；
- `phase1_push_recovery_train_smoke.py` 不再用字符串排序选 checkpoint；
- Phase2/Phase6 eval wrapper 使用同一个公共 selector；
- calibration 结论已固化；
- 不改变 Phase6 默认配置和 accept/reject 协议。
