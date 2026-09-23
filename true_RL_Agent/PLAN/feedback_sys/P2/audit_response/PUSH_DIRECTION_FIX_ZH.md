# Push 审计 5.2 / 6.1 处置（2026-09-24）

## 5.2：采纳，保留有符号世界系运动量

新产出的数值档案为 `true_rl_agent.push_recovery.numeric.v2`。这里的 v2 是**数值 profile** 版本，与 velocity 扰动协议 v2、force 扰动协议 v3 无关；两种扰动共用此次采集修复。

`runtime.py::PushRuntimeState._observe_pose` 在原来的 pre-reset 观察点保存 `planar_motion[...,4]`：

| 列 | 原始定义 | 单位 |
|---|---|---|
| `root_lin_vel_w_x_mps` / `root_lin_vel_w_y_mps` | `root_lin_vel_w.xy` | m/s，世界系，有符号 |
| `reset_relative_root_pos_w_x_m` / `reset_relative_root_pos_w_y_m` | `root_pos_w.xy - reset_root_pos_w.xy` | m，世界系，有符号 |

仍保留原 `pose[6]` 及其中的 speed/drift 范数：它们是当时运行时五条件和终止判据真正使用的原 dtype 操作数。位移只计算一次，方向量与范数来自同一帧、同一位移张量，不运行第二个姿态估计器。离线重算以原范数重放判据，并用相同 dtype 的向量范数检查来源一致性（相对容差 `4 × dtype.eps`、绝对容差 `dtype.tiny`）；不一致使 `matches_runtime=false`。这样避免用 CPU NumPy 新算的范数替换 GPU Torch 操作数而引入阈值末位漂移。

代价为每 env 每控制步追加 4 个原 dtype 浮点数。按既有真实 64-env 首 episode 的完整帧数，float32 原始字节增量为 G1 每 seed 421,888 B、Go2 每 seed 310,272 B；这是容量估算，不是新采集的压缩体积或性能实测。128 帧 staging 额外 131,072 B，原有预算计算会自动计入新增数组及主机副本。

`numeric_capture.py` 读取已知 v1/v2，引用校验要求 identity 与 metadata 的版本一致；新生产者只写 v2。v1 复算明确返回 `direction_status=not_collected_in_v1`，既不回填方向，也不把缺失记为零。v2 缺少 `planar_motion` 会拒绝。`rollout.py` 在结束后释放新的临时张量。根速度/位移可以支持后续运动方向计算，但不能单独证明迈步、脚接触或致因；本轮未实现 P3 词条或解释。

## 6.1：部分采纳，合并终局分类，保留逐步独立复算

现有任务模块 `failure_classifier.py` 统一消费标量极值、已观察事件和恢复状态：velocity 继续使用 `classify_recovery_episode`，force 从运行时原分支等价迁入 `classify_force_recovery_episode`，两者复用 `_recovery_fall`。`force_runtime.py::episode_records` 与离线 reducer 调用同一终局函数；reducer 传入自己从逐步档案重建的事实，不取封存结果来决定标签。

保留协议差异：velocity 的无效注入优先于物理跌倒，force 的物理跌倒优先于注入错误；早跌倒时缺失的后续推力不额外计为注入错误；连续 dwell 达标后仍要求最终姿态稳定。既有主统计字段、force 注入/physics hook、完整 pulse 检查、序列最后一脉冲末端锚点、窗口包含性和锁存时序均未改。

没有将其余约 60 行全部抽成 Torch/NumPy 通用状态机。运行时设备张量与离线 CPU 数组有不同的数据布局、阈值转换和观察职责；引入适配层会扩大此次修复。五子条件、有效掩码、dwell 和恢复锁存继续独立重算，作为检验运行时的 oracle；此后协议若改这些判据，仍必须同时更新重算与配对验收。终局标签改为共用分类后，标签本身不再宣称是独立实现的 oracle。

## 已完成与待实测

- CPU：`test_push_numeric_capture`、`test_push_force_runtime`、`test_push_physical_protocol`，32 项通过；日志 [push_direction_unit_tests.log](push_direction_unit_tests.log)。新增覆盖相同范数/相反方向、非零 reset 原点、force 终止帧和闭合后排除、v1 显式缺失、v2 漏字段/向量不一致、on/off 全 per-episode 相等、两协议标签优先级与严格阈值。
- 真实旧证据：本地保留的 12 份原 manifest 与 42 个 NPZ 副本逐块核 hash。当前源码重算 768 回合，除新增的方向缺失声明外，与封存的原复算 JSON 逐字段完全相等；见 [push_v1_recompute_regression.json](push_v1_recompute_regression.json)。临时重定位只复制原字节，没有改写旧 manifest、chunk 或 receipt。
- 新 profile 的真实 simulator on/off 配对由根任务统一执行；本笔记不把 CPU fixture 或旧 v1 回归当作新方向量的真实物理采集验收。未执行 GPU、训练或模型调用。
