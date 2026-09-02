# Push Recovery Success 聚合与轮间接受逻辑整理

## 旧项目的有效做法

旧项目把“评估是否跑完”和“候选是否可接受”分开：

- `public_dev`：给下一轮提示词使用的公开开发评估摘要。
- `public_accept`：用于轮间接受判断的公开接受评估摘要。
- `success_rate_stats`：对多个公开 rollout summary 的 success rate 计算 `count/mean/std/standard_error/values`。
- `motion_quality_gate`：动作幅度、动作变化、漂移、pitch/roll 的公开 sanity gate。它失败不代表 eval 没跑完，但接受时可以作为 required gate。
- `budget_accounting`：训练预算必须由 runner 侧记录并通过 gate。
- 接受逻辑：先检查 public result status 和 required gates，再用 `public_accept` 的 mean 做接受分数。若当前 best 接近 0，要求 absolute floor；否则要求 `candidate_mean - best_mean > max(public_min_delta, pooled_standard_error)`。

这套逻辑合理的部分是：success 统计有 split、有统计误差、有 gate；接受不依赖单次浮点小波动；public projection 不暴露 raw seed 或 hidden/trusted 数据。

## 不直接照搬的部分

旧项目是 push recovery 强耦合工程。新项目不能把以下语义写入共享判定层：

- fall/height/roll/pitch/drift 如何定义成功。
- public rollout summary 如何投影为 success rate。
- motion quality gate 的具体指标和阈值。
- push recovery 的 seed split 数量、eval env 数量、budget 字段名称。

这些都属于 `unitree_push_recovery` 的任务语义。

## 新项目采用的边界

共享层只负责轮间接受/拒绝的通用结构：

- 候选 surface 必须 eligible。
- public result status 必须是 `pass`。
- 任务投影出的 required gates 必须全部 `status=pass`。
- public score 必须是有限数。
- 如果任务投影给出 `absolute_accept_floor`，在 near-zero best 时使用 absolute floor。
- 如果任务投影给出 `score_stats/min_delta/best_score_stats`，使用 noise-aware delta。

push recovery 任务侧负责：

- `build_push_public_result_from_summaries()` 聚合 `public_dev/public_accept`。
- 隐藏 raw seed，只保留 run index、count 和公开统计值。
- 生成 `motion_quality_gate`、`budget_accounting`、`acceptance` metadata。
- eval 脚本按 `--public-seeds` 和 `--public-accept-seeds` 生成 split summaries，再调用任务聚合函数。
- `PushRecoveryTaskPlugin.project_public_result()` 把训练 summary 中 runner-owned budget accounting 接入 public result。

## 当前改进

- `policy_accept_delta=0` 不再意味着有一点点分数提升就接受；如果任务投影提供 `min_delta=0.05`，共享接受逻辑会执行 noise-aware threshold。
- motion gate 和 budget gate 从“仅诊断信息”变成 push recovery 的 required gates。
- live 默认 eval env 数量回到 64，避免 8 env 的粗粒度 success rate 主导接受判断。
- public seed count、public accept seed count、eval env count 都进入 push task fixed eval protocol/invariants。

## 仍需注意

`trusted_seed_count` 仍不能进入 public prompt/result。当前 public accept 是公开接受 split，不等同 hidden/trusted eval。未来如果加入真正 hidden eval，它只能用于 runner 内部安全判定，不能进入 LLM 可见反馈。
