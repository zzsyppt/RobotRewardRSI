# P6 Push 新物理协议设计

2026-09-12 更新（初稿 2026-09-11）。状态：**正式协议、资格门和资源声明已于北京时间 00:32 签发冻结；完整训练及三相已结束，两格 reference 均 NO-GO（G1 未恢复、Go2 饱和）。** [formal_declaration_01.json](formal_declaration_01.json) 在首次 holdout 消费前固定了参数与源码身份，`active` 只表示协议冻结，不表示 reference 已通过。声明解释见 [FORMAL_DECLARATION_ZH.md](FORMAL_DECLARATION_ZH.md)。本轮按声明执行，不根据 accept/sealed 成绩回调参数，也不改旧 G1-29 legacy/reference_v1 的解释。

## 机器人、权限和唯一接口

新 Push 复用既有 G1-37 与 Go2-12 RobotProfile/RuntimeAdapter/PPO，使 Stair/Push 两列使用相同机器人行。G1-29 保留历史读取及当前后端回归。Push 只消费 base、joint/action、observation 和本体尺度，不引入逐足支撑语义。

源码 `tasks/push_recovery/protocol.py` 拥有不可变 `PushRecoveryProtocol`：`defaults()`、`from_mapping()`、`to_dict()`、`protocol_hash`、`resolve(robot_scale_m)`。绑定给出 `robot_scale_m=L`，取 RobotProfile 已有默认根高度事实；`resolve` 用 `v_scale=sqrt(9.81 L)`、`t_scale=sqrt(L/9.81)` 生成 SI 参数。新运行配置与 manifest task projection 都使用 `push_protocol`、`push_robot_scale_m` 两字段，协议和派生实物须一致。

## 施扰与已冻结参数

唯一机制为 **world-frame planar velocity increment**：读取 articulation 根速度，仅在 x/y 加本次抽样的 Δv，保留 z 及三个角速度分量，再写入仿真。它既不是绝对速度重置，也不是外力/冲量。事件保存 requested/observed Δv、完整前后速度、事件序号、episode 序号和控制步；写入后的 tensor 读数用于即时接线检查，下一次物理步证据另存，不能把 read-after-write 当动力学成功。

| 项目 | 冻结值 |
|---|---|
| 站立命令 | x/y/yaw 均为 0；平地，关闭地形 curriculum；任务不修改机器人资产或 PPO |
| 重置 | nominal joint pose、zero root/joint velocity、root x/y偏移0、yaw uniform[-π,π]；机器人现有startup质量/材质随机化仍受adapter/source身份绑定 |
| episode 时长 | `30 t_scale`，向上取整为控制步 |
| 第一次施扰 | `5 t_scale` |
| 第二次施扰 | repeated stratum 在 `17 t_scale`；与第一窗口不重叠 |
| 恢复窗口 / 连续驻留 | `8 t_scale` / `1.5 t_scale`，均向上取整到控制步 |
| 速度幅值 low / medium / high | `0.15 / 0.30 / 0.50 × v_scale` |
| 恢复速度 / 角速度 | 水平速度≤`0.08 v_scale`；角速度范数≤`0.5 / t_scale` |
| 恢复姿态 / 高度 | abs roll、pitch≤0.20 rad；base高度≥`0.70 L` |
| 跌倒 | base高度<`0.45 L` 或 abs roll/pitch>0.80 rad |
| 越界 | 水平位移>`3 L`；以 reset 时根位置为原点 |
| 注入实物容差 | requested Δv 与立即读回增量的逐分量最大误差≤`1e-5 m/s`；未成功注入使技术资格失败 |

训练按 episode 在 low/medium/high/repeated 四类均匀抽样，两个事件的方向分别独立在 [0,2π) 连续均匀抽样；施扰源私有地预先抽样，候选只接收已经发生的事件。评测每 seed 的64环境固定分配low/medium/high/repeated各16，最后一类使用medium幅值、两次施扰。每类覆盖16等间隔方向；共同相位由该cohort的runner seed确定，第二事件方向加π/2。采样概率、方向范围、各类数量和第二次偏移写入resolved protocol并由runtime直接消费；其hash进入科学身份。各卷使用同一分布，不人为固定某卷更容易；cohort seed隔离仍使用已有seed policy。方向位于训练支持内，不能称全新动力学OOD。no_push只用于独立diagnostic，不能混入正式64环境。P7的独立final域另行预承诺。

## 计数与评分实物

no-push control不进入抗扰恢复成功分母。正式分母是全部预定受扰episode，包括施扰前已经跌倒者；不得因其实际事件数为0而将失败过滤掉。成功要求至少一次实际注入，全部计划事件确实执行，每次事件在窗口内达到速度、姿态、高度条件并连续驻留，终态仍满足恢复条件，且全程无跌倒/越界。任何未恢复事件使该episode失败；连续扰动不覆盖或删除前一事件。恢复时间从事件注入步计到驻留成立步；未恢复按窗口上限给出受限恢复时长，并保留censored标志，不只报告成功子集均值。

主分类互斥且整数闭合：`no_push_control`、`success_recovered`、`not_recovered`、`fall_forward/backward/side/height_collapse`、`out_of_bounds`、`invalid_push_injection`。control另报告站立是否通过。技术无效样本不能静默丢弃或改记成功。完整 per-episode/per-event 记录进入私有实物，public projection 不包含未来计划、原始seed或私有题面。

新 `push_active/push_vec_public` 只来自当前控制步已经执行的真实注入事件；无事件返回零。不再用自主运动速度差推断事件；历史协议继续按历史信号规则解释。训练、评测和视频使用同一个 builder、事件和终态采集。视频只是同进程 PhysX/RTX 渲染，复用公共 NativeVideoRecorder，绑定 checkpoint/seed/episode/event；排除 auto-reset 后的帧，至少取得成功与失败各一例并完整解码。

新reward程序完整替换upstream locomotion奖励。RewardManager只安装一个task-owned program term，在同次termination已计算、auto-reset尚未发生时求各RewardTerm加权rate之和；RewardManager独占一次control_dt积分。新reference必须非空。V2输入除旧公开字段外提供nominal_base_height、base_height_relative、base_lin_vel_scaled、base_ang_vel_scaled和terminated_non_timeout；末项取同次termination_manager.terminated，不消费未来事件或重置后状态。旧G1-29继续按additive历史语义执行。

新Push的公开输入、七项policy observation、奖励、termination和startup/reset事件均不消费接触传感器；policy/输入由articulation本体状态、action和command直接提供。因而新V2任务覆盖层同时移除height_scanner和contact_forces，ABI冻结sensors为空，并检查实际`scene.sensors`集合为空；不改RobotProfile、机器人公共builder、Stair或旧Push的传感器语义。

## 预算、顺序与冻结门

每机器人 reference 只做一次 random-init 完整训练：seed=42、4096 env、24 steps/iteration、500 iterations，合计49,152,000 env steps；不因性能不佳延长或重训。dev/accept/sealed仍为3/3/7 seeds ×64 episodes，即192/192/448。两机器人完整训练合计98,304,000 env steps，首次三卷评测共1664 episodes；这是样本合同，不是实测 GPU 小时。录像重放另记账；P7搜索和复现不消费本池。

开发接线检查标记为 diagnostic，不签发可学性或泛化结论；两格 wiring、生产短 train→dev 和 G1 提前事件探针的结果见 [开发记录](DEVELOPMENT_LOG_ZH.md)。reference 基本可学性门、题池余量门、候选接受与 sealed 下降判据、GPU/墙钟/磁盘上限及重试上限现已在首份 holdout 消费前冻结。旧 reference_v1 的0.05分数下限、恢复率下限0和5m/s速度上限没有继承为新资格门。成功/失败原生视频仍需绑定实际 reference checkpoint 取得并核验。

已签发资格门：dev基本可学性与余量区间[0.2,0.9]、accept/sealed绝对下限0.2、同分布accept→sealed下降上限0.1；候选最小改进参数0.05，仍使用既有噪声感知接受规则。每格资格流程总墙钟上限4小时，由现有进程 wrapper 执行；单作业 runner timeout 为36000秒，0次基础设施重试。资格失败记录 NO-GO，不补训练、不调门、不合成 Genesis。详细资源与停止规则以签发声明为准。

2026-09-11 的本地实施阶段未启动 GPU、API 或物理仿真；后续目标机 diagnostic 单独记账。P6只处理任务、reference、科学录像和回放；15代×width2搜索、至少3训练seed配对复现及final新题归P7。声明签发不是 P7 搜索 GO。

本地实施检查：2026-09-11，`.venv` CPU执行 `test_push_physical_protocol`、`test_push_recovery_v2_metrics`、`test_push_recovery_task` 共75项全部通过、0跳过，包含XYZW姿态转换、真实事件状态张量、连续驻留与auto-reset冻结；compileall及diff whitespace检查通过。这些是当时的源码/CPU接线证据，不替代目标机实际 env 的关节顺序、动作有效scale、观测term配置和维度验证。后续目标机接线及完整本地回归已在 [开发记录](DEVELOPMENT_LOG_ZH.md)追加；本轮实测结果和录像覆盖缺口见 [当前验收](ACCEPTANCE_ZH.md)，不能把协议 active 解释为 reference 合格。
