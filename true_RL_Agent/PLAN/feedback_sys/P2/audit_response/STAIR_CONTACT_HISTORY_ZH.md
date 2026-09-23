# Audit P2 §5.1：Stair 接触判据操作数补齐

日期：2026-09-24。范围只含 Stair 已有传感器的公开评估数值档案；不修改奖励、终止判据、接受门、训练采样和机器人绑定。本文是源码修复及 CPU 验证记录，真实 v2 GPU on/off 验收由主任务另行记录。

## 裁决与观测边界

审计指出的操作数缺失成立。旧 recorder 保存 `net_forces_w` 瞬时向量；forbidden manager 判据依赖 `net_forces_w_history`。任务 reward 的 support-contact 判据同样读取 history。仅凭瞬时向量不能重算该判据，不能把缺失帧解释为“实际无接触”。

但审计把 history 槽位直接解释成“控制步内三个物理子步”的推断未获源码支持：当前安装版本默认 lazy 更新，配置的 `update_period == physics_dt` 不代表缓冲每个物理子步实际取样。不能由旧档案缺力，进一步断言接触必定只发生在两个已知物理时刻之间。本次不强制刷新传感器，不改变它的实际更新调度。

真实源码证据：

- 本仓库 `src/true_rl_agent/robots/isaaclab_locomotion.py:48` 将 contact sensor 的配置周期设为 `sim.dt`。
- 服务器安装根为 `/data/yuyangliu/isolated/isaac_sim_6_validation/IsaacLab-v3.0.0-beta2.patch1/source/`。其中 `isaaclab/isaaclab/scene/interactive_scene_cfg.py:80–84` 明确 `lazy_sensor_update=True`，data 被访问时更新；`isaaclab/isaaclab/scene/interactive_scene.py:648–650` 将 `force_recompute=not lazy_sensor_update` 传给传感器；`isaaclab/isaaclab/sensors/sensor_base.py:194–213` 的普通 update 先更新时间戳/过期状态，只有强制或可视化才立即刷新 buffer。
- 同安装树 `isaaclab_physx/isaaclab_physx/sensors/contact_sensor/kernels.py:248–254` 每次实际更新先后移 history，将当前值写入 index 0，因此顺序为 newest → oldest，不是 oldest → newest。
- 同安装树 `isaaclab/isaaclab/envs/mdp/terminations.py:157–165` 的 `illegal_contact` 计算 `any(max(norm(history, xyz), history) > threshold, bodies)`。本仓库 `isaaclab_runtime_env.py:567–573` 用已解析 forbidden body 角色和协议阈值绑定 `base_contact`。
- `src/true_rl_agent/tasks/stair_climbing/isaaclab_reward_runtime.py:231–244` 的 support-contact 也是逐 body history 最大向量范数；而 `transition_decoder.py:552–564` 的现有离线 support-contact 定义来自瞬时力。本次保留这一区别，不用新增 history 悄悄改动原有 slip/clearance 统计。

这些力是既有绑定 body 的世界系净法向接触力向量；没有接触对、接触点、立面编号或表面法线。即使方向存在，也不能据此确定撞到了哪一块立面。其他身体部位没有新增采集。

## 源头修复

新档案 profile 为 `stair.unified_first_episode.v2`，增加：

| 字段 | 形状 | 语义 |
|---|---|---|
| `support_contact_forces_w_history` | `[control_step, env, history, support_body, xyz]` | 已绑定 support 角色，保持其声明顺序、原始 dtype 和完整带符号向量 |
| `forbidden_contact_forces_w_history` | `[control_step, env, history, forbidden_body, xyz]` | 已绑定 forbidden 角色，保持其声明顺序、原始 dtype 和完整带符号向量 |
| `metadata.contact_force_history` 与 chunk header 同名合同 | 小字典 | `status=recorded`、H=3、newest-to-oldest、实际 lazy 设置、配置周期、physics/control dt、覆盖和缺失时间信息 |

时钟明确命名为 `contact_sensor_buffer_update_events`。逐槽时间戳没有采集；声明 `configured_update_period_is_not_observed_sample_spacing`，不生成 `-i * physics_dt` 的假时间戳。覆盖是首回合每个有效控制步结束、reset 前的现有 buffer 快照；reset 后补零 history 槽本身无独立有效位，不能把每槽零都解释为已观察到无接触。档案层的 `valid=false` 仍明确表示首回合终止后的无效 padding，不是观测。

实现入口：

- `runtime_projection.py:26–58`：复制既有传感器 history，按既有 resolved role IDs 取值；不新增 sensor、不刷新 buffer。
- `isaaclab_execution.py:1217` → `isaaclab_runtime_env.py:409–417,472` → recorder `:172–179`：单一布尔门控，默认关闭，仅 numeric capture 开启且 unified public-dev 才复制 history。capture off、普通 acceptance/sealed 都没有新增 history clone/host materialization 成本。
- `legacy_transition_reader.py:19–22,79–82`：新增两数组和小合同必须成组出现，旧 payload 不补零。`transition_decoder.py:79–198` 仅将这一可选组随原输入有界搬运，原标量解码不改。
- `numeric_capture.py:27–28,76–115,133–218`：v2 要求完整 history；读取 v1 返回 `contact_force_history.status=not_recorded`。v2 缺数组/合同或形状不一致会拒绝，不伪造缺失数据。既有逐回合/summary 复算继续使用同一 decoder/tracker。

Go2 绑定只有 4 个 foot support + `base` forbidden；G1 绑定为 2 个 ankle-roll support + `torso_link` forbidden。profile 复用真实绑定，不将 Go2 的五个 body 下标硬套 G1。G1 active composition 同样默认采集，因此主任务已安排 G1 独立真实 on/off；Go2 CPU/GPU 结果不能冒充 G1 的真实采集验收。

## 容量与验证

按远端已有三个真实 v1 manifest（逐一核 SHA-256）的实际 shape/dtype 精确计算，新增 raw 大小是两种瞬时力 raw 总量的 3 倍。详见 `stair_contact_history_capacity.json`。

| 已有帧数基准 | 新增 history raw | 新档案 raw 估算 | 最大新 chunk raw |
|---|---:|---:|---:|
| Go2 parent | 84,602,880 B | 199,774,080 B | 7,839,360 B |
| Go2 Gen8-A | 93,312,000 B | 220,337,280 B | 7,839,360 B |
| G1 reference | 55,987,200 B | 197,527,680 B | 7,028,352 B |

最大 chunk 远低于 64 MiB；全部 raw 约 188–210 MiB，存储仍受既有 writer 的 2 GiB 上限约束。该计算不是新档案压缩率、GPU rollout 时间或 RSS 的测量，后者必须用同版 on/off 实测。缓冲仍按既有 64 控制步分块，未改成整回合 GPU 驻留。

最终本地 CPU：

1. `python -m unittest tests.unit.test_stair_numeric_capture tests.unit.test_stair_transition_materialization tests.unit.test_stair_morphology_projection tests.unit.test_stair_p5_runtime tests.unit.test_stair_phased_evaluation tests.unit.test_stair_phase5_contract tests.unit.test_stair_numeric_dev_replay -q`：56 PASS，20.898 秒。日志 `verification_stair_contact_history.log`。覆盖 Go2/G1 profile、真实 recorder 方法 off 不调用 history snapshot、terminal 保存与 reset padding、向量方向/角色顺序/clone、v1 缺失语义、v2 缺数组拒绝。模拟输入不是真实训练/评估成果。
2. 六个修改生产模块 `py_compile`：PASS。
3. `python PLAN/feedback_sys/P2/test_verify_stair_pair.py -q`：4 PASS，5.547 秒。首次发现独立工具只重建旧动态字段，已按授权最小补入 v2 history 成组字段并保留 v1 读取；原事件、统计与验收阈值未改。最终日志 `verification_stair_pair_regression.log`。
4. `python PLAN/feedback_sys/P2/audit_response/test_verify_contact_history.py -q`：1 PASS，0.968 秒。fixture 中 864 个 manager contact terminal 均可由 history 重算，瞬时力均为零；故意清空历史检出 864 个 mismatch；v1 明确 NOT_RECORDED。日志 `verification_contact_history_probe.log`。

独立真实档案核验命令：

```text
python PLAN/feedback_sys/P2/audit_response/verify_contact_history.py --project-root SOURCE --on-dir ON --output OUTPUT.json
```

该工具 hash 校验后逐 valid 帧按同运算顺序核 `history max norm > threshold` 与 `base_contact`，报告瞬时零/history 非零的 manager terminal 数、support 两种操作数的差异和分母、dtype/磁盘/chunk 容量。这里 terminal 是非互斥 manager term + done，不冒充 primary_reason 的互斥分类。工具不替代 `verify_stair_pair.py` 的完整 outcome/summary 对照及时间/RSS 验收；不增加新的接受规则。
