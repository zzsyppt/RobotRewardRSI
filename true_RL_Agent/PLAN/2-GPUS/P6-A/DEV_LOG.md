# P6-A 开发日志

## 2026-08-18：范围冻结

- 重读总计划与 P0–P5 交付，确认 P6-A 只验证资源和进程，不进入 provider、eval、sealed、ledger、selector、acceptance 或 exploration。
- 参考 Codex 的有用思想仅限：lease-before-spawn、进程组所有权、取消后等待清理、active/full-wall 分离；没有引入通用 Agent、DAG、gRPC、TTL、沙箱或任意宽度调度器。
- 将 workload 固定为 `4096 × 24 × 64 = 6,291,456 env-steps/job`，矩阵固定为 13 cases / 18 jobs，不允许选择性 replacement。

## `4b8473e`：初版固定资源 shadow

- 增加 task-neutral resource plan、GPU inventory/lease binding、1 Hz resource snapshot 与纯统计汇总。
- 增加 Stair-owned reference fixture、固定 workload、child marker/summary 和 coordinator。
- 复用唯一 `_execute_stair_training()`，未复制训练主体或科学协议。
- fake/CPU 接线验证了两张 lease 必须在任一 spawn 前同时取得、paired barrier、create-once receipt 和 authority-zero。

## `c9e77b4`：Isaac CLI 注册修正

- 修复 AppLauncher 参数注册顺序，避免 shadow child 自定义参数与 Isaac parser 冲突。
- 第一次真机诊断在首个 GPU0 warm-up 后 no-go：同一 PID 在 GPU1 建立约 540 MiB compute context。

## `811b789`：renderer MGPU 显式关闭

- Stair-owned shadow 入口固定 `multi_gpu=false`，并把资源运行条件写入 workload identity。
- 第二次真机诊断仍在首个 GPU0 warm-up 后 no-go：非目标 GPU 固定约 500 MiB compute context；目标训练、initial hash、summary 和 checkpoint 均正常。
- 结论：仅关闭 renderer MGPU 不能保证物理 GPU 隔离，不能通过白名单放宽 cross-device 门。

## `0c46b68`：物理 UUID 与进程内逻辑设备分层

- runner-owned spawn 对每条 child 设置 `CUDA_VISIBLE_DEVICES=<full GPU UUID>`，删除 `NVIDIA_VISIBLE_DEVICES`；父环境保持未 remap。
- 宿主 plan/lease/采样仍使用物理 `cuda:0/cuda:1 + UUID`，两条 child 内都使用逻辑 `cuda:0`；Kit renderer 保留宿主物理 ordinal。
- marker/summary 绑定 physical device、UUID、logical device、renderer ordinal；loader 复核内层 `training.device`。
- GPU0/GPU1 的短 AppLauncher full-UUID 探针均只在目标 UUID 建立 compute context，退出后无残留。
- 完整 13-case 矩阵首次跑满。性能和资源门全部通过，但 report no-go：fault ready 将 marker 前样本误算为 active 后样本，且中断判定错误依赖 `returncode != 0`。

诊断输出：

- `/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_c9e77b4_20260818T131516Z`
- `/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_811b789_20260818T133055Z`
- `/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_0c46b68_20260818T141210Z`

三份输出保持不可变 no-go，不作重投影，不与最终报告拼接。

## `4b2aefb`：fault proof 根修

- fault ready 只计每条 lane 自己 active marker 之后的 exact PID+UUID 样本。
- receipt 改为保存 `post_active_matched_sample_counts_by_lane`，两条 lane 都必须达到冻结的 3 个样本。
- fault gate 绑定 signal、target job/PID/PGID、marker hash 与 monotonic 时序；lane-a 无正常 summary且已清理，lane-b 必须完整 pass。
- 删除“SIGTERM 后 return code 必须非零”的错误假设；return code 只保留为诊断。
- case schema 升为 v3，core summary/report 升为 v2。

验证：

- 本地聚焦：31 tests（Windows 上 Linux-only 1 skip）；
- 本地最终全量：`1270 tests / OK / skipped=21`；
- 服务器 Linux 聚焦：`31 tests / OK / 0 skip`；
- `compileall`、`git diff --check`：通过；
- local/origin/server HEAD：`4b2aefbe0bb0a22226dc9055a02eb3f83cea0a04`，服务器 tracked clean。

## 最终真机运行

输出：

`/data/yuyangliu/true_rl_agent_phase_g/p6a_resource_shadow_4b2aefb_20260818T151328Z`

结果：

- exit `0`，report v2 `go=true`；
- 13/13 cases、18/18 jobs、17 正常 summaries；
- overlap median `0.9999694815`；
- slowdown median A/B `1.0294992890 / 1.0341294103`，max `1.0524981051`；
- paired/matched allocated-active GPU-hours `1.0349108098`，paired/one-single `2.0698216196`；
- fault 两 lane post-active exact samples `3/3`；只终止 lane-a，lane-b 完整闭合；
- 0 cross-device、unknown PID、OOM、orphan、sampler error、swap growth；
- 结束后 compute PID、process group、active lease holder 与 tmux 均为 0。

三路独立只读审计分别重算性能/资源、fault 证据和运行后清理，均同意 G2 GO。

## 阶段收口

- P6-A 只证明资源运行模式，不证明科学 generation。
- shadow 的 UUID 隔离策略尚未自动接入正式 Stair train/eval；这是 P6-B 前置接线与实证门，不在 P6-A 扩面修改。
- 没有启动 P6-A2 或 P6-B；等待用户验收并另行授权。

## 2026-08-19：验收后口径复核

- 重新从 measured snapshots 按 exact PID+UUID 复算：完整 process-resident 窗口的
  `nvidia-smi utilization.gpu` 均值为 single/paired `24.05% / 23.27%`；冻结
  workload-active 窗口为 `25.91% / 24.97%`。该字段是 kernel busy-time，不是 SM
  occupancy，不能据此声称尚有 75% 可叠加算力。
- 同序号两条 single full-job wall 之和相对 `max(同次 paired 两 lane full_job_wall)` 的训练 shadow 吞吐提升
  中位数为 `1.92365×`。它不含串行 authoring、分阶段 eval、selector、sealed 或
  generation barrier，不外推成正式 generation/2,000-iteration 的端到端加速。
- 报告中的 GPU-hour 明确为 `allocated_gpu_hours = sum(active_seconds) / 3600`；
  `paired/one-single` 是 `2 × paired/matched-singles` 的代数冗余展示门，不是独立证据。
- `4b2aefb` 是错误 fault 代理的协议根修：旧 no-go 不重投影、资源阈值不变、schema
  升级后全矩阵新跑；不是在同一次 pilot 中看到结果后放宽门。
- 生产 UUID 接线确实尚未交付，但总计划已把它冻结为 P6-B 的首个阻断项。遵守阶段门，
  本次不越过尚未批准的 P6-A2/P6-B 修改生产执行协议，也不自动启动同卡双 lane 探针。
