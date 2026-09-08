# P2 实施计划：统一训练、评测和原生录像后端

日期：2026-09-07；2026-09-08 审计修订并完成恢复验收。**状态：源头修复、有效完整训练、新 checkpoint 闭环、正式部署和交付核验全部通过，工程 GO。** 旧完整训练的 PhysX aggregate-pair 溢出使其资格失效，相关失败证据和用户中断前缀仍保留；本文件维护迁移实施边界，逐项归因见 [审计修复记录](AUDIT_FOLLOWUP_ZH.md)，最终结果见 [验收记录](ACCEPTANCE_ZH.md)。旧/新数值等价与新 reference 资格没有随工程 GO 自动成立。基线为本地/服务器共同的 `1da2effbcc4720fcd8ed216f9fec56b0d757697e`；P1 后续 API 收尾已完成，53 项定向测试通过，见 [P1 记录](../P1/AUDIT_FOLLOWUP_ZH.md)。本阶段遵守 [建设主计划](../NEW_WORLD_PHASED_PLAN_ZH.md) 和任务 × 机器人两轴边界。

## 1. 目标与现状

最终新生产作业只运行已验证的 Isaac Sim 6.0.1 / Isaac Lab 3 / PhysX / RSL-RL 5 路线。训练、评测、原生录像共用任务环境构造、机器人绑定、策略实现和 checkpoint 读取；录像仅增加相机与编码呈现，不复制 terrain、actor 或任务判定。旧完整 15 rounds/P5 实物与最小历史 reader 保留，但旧运行时不能作为新生产作业的备用路径。

实施起点核对时，服务器两个环境均可定位、两块 RTX 5090 无计算进程。目标环境当时已有原生录像材料，但尚未证明训练兼容；其独立视频脚本含另建地形、手工 actor 和展示布局，不能直接提升为正式科学录像入口。后续执行状态见验收记录。

| 项目 | 旧运行时，仅用于冻结迁移对照 | 唯一目标运行时 |
|---|---|---|
| Python | `/data/yuyangliu/envs/isaaclab/bin/python`，3.11.15 | `/data/yuyangliu/envs/isaaclab6_validation/bin/python`，3.12.14 |
| Isaac Sim / Lab distribution | 5.1.0.0 / 0.47.2 | 6.0.1.0 / 6.1.14 |
| Isaac Lab 源码 | `/data/yuyangliu/vendor/IsaacLab` | `/data/yuyangliu/isolated/isaac_sim_6_validation/IsaacLab-v3.0.0-beta2.patch1`，源码 VERSION 3.0.0 |
| RSL-RL / Torch / Gymnasium | 3.0.1 / 2.7.0+cu128 / 1.2.0 | 5.0.1 / 2.11.0+cu128 / 1.2.1 |

以实际安装源码为实现依据，准确文件 SHA256 和包版本随运行时证据保存。官方资料：[RSL-RL 5.0.1](https://github.com/leggedrobotics/rsl_rl/releases/tag/v5.0.1)、[Isaac Lab 3.0 beta2](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0-beta2)。不追随浮动 latest。

## 2. 边界与实现顺序

### P2-A1：运行时和所有权收敛

- 后端统一选择目标解释器、PhysX 配置、依赖与实际执行模块身份；在当前生产准入边界拒绝旧运行时启动新作业。旧 manifest reader 保持原解释，禁止通过改 hash 或兼容开关冒充目标运行时。
- Isaac Lab 3 的 preset 通过其官方解析入口解析，训练/eval 使用同一入口。不引入通用多物理后端调度框架。旧 vendor 原件保持不动；实际探针发现 Unitree G1-29 还依赖被移除的 noise/PhysX API，因此目标运行时使用独立固定的 vendor 适配副本，本项目保存必要补丁和源头校验和，不向公共库注入兼容别名。
- RobotProfile 继续提供机器人事实；Isaac 机器人的 PPO 查找委托已有 runtime spec，只保留 G1-29 vendor adapter 的必要独立配置。Go2 flat 配置从 robot execution source 清单移除，其真实 task reference source 证据保留在 Stair 权威链。
- 逐项比较旧/新实际 train/eval config。当前已发现上游 Go2 质量分布、质心与 joint reset 默认值改变，以及资产根从 5.1 浮动至 6.0：由机器人 adapter 恢复绑定的物理配置，并从 RobotProfile 的 asset closure 取实际 USD 根。不得改 RobotProfile 的历史 hash 来掩盖执行资产漂移。
- Gym ID 改为不含 G1 的任务环境 ID，训练与评测各一个；真实机器人仍由 resolved cell 指定。物理实现变更由 backend execution manifest 表达，任务环境规范的 builder 标识和旧训练 binding 保持原解释。早期探针曾试用新 builder 标识，全量回归揭示其改变了冻结训练身份，已撤回；这些早期结果只作接口诊断，不作为最终配对验收。support-effector 只增加对物理 support-body 集合的包含约束，不强制相等。
- 参考本地 Codex `ExecParams` 的职责划分：参数、cwd、env、输出/超时/生命周期分别绑定；复用本项目既有 job、argv、process group 与 lease。Kit 日志由作业输出路径派生，双 lane 不共写同一日志。不移植 Codex 调度器。

### P2-A2：RSL-RL 5 与 checkpoint ABI

- 现有冻结的 policy/algorithm 合同继续作为科学参数来源，在 RSL 后端一次映射为 RSL-RL 5 的 actor / critic / distribution 配置；任务不维护第二份网络参数表。实际 learner 配置随训练证据保存。
- 新作业只创建 RSL-RL 5 learner，训练、评测和录像共用它的策略实现。旧 checkpoint 只经一个显式的状态字典转换入口映射到相同的当前模型；删除手工重建 actor 的重复实现。
- 逐项核实 observation group/顺序/归一化、joint/action 顺序和缩放、四元数、dt/decimation、contact/reset/RNG、reward rate × control dt、timeout/bootstrap。保持任务公式、seed/预算、PPO 超参数和候选权限不变。
- 旧 actor/critic/normalizer 可转换性逐项验证；旧 optimizer 不凭参数槽位猜测迁移，无法证明等价时记录不支持旧训练续跑。新格式必须完整保存/加载 optimizer、normalizer、iteration 并真实续跑。旧 reference 的策略可读不等于获得新后端的 reference 资格。

### P2-B1：同源评测与原生录像

- 录像接入正式 eval 环境/执行器，在环境创建时选择 native RGB render，以薄 Gym wrapper 采集实际控制步后的 RGB，并用已有 `imageio_ffmpeg` 顺序编码。实测旧 MoviePy 的浮点时间索引会跳帧/重复帧，故退役其时间重采样路径，不增加另一套物理回放。记录选项只包含输出、相机和画面尺寸；禁止另设地形、机器人布局、奖励或成功规则。
- 每份录像关联同一 checkpoint hash、resolved cell、题面/cell digest、seed、控制 dt 与执行环境身份。视频结果复用 episode recorder 的原始计数，不在 exporter 重新判定。
- 默认画面 1280×720，FPS 从控制 dt 推导为实时播放；G1/Go2 当前 0.02 s 对应 50 FPS。原生首帧来自 control step 1，影片止于 terminal 前一帧；旁车明确起点、帧数与时长。实测 SDK 在 reset 时尚未发布完整的机器人画面，因此撤掉手工注入的 reset 图片。极早结束而没有 pre-reset 帧时只记录缺帧事实，不伪造 MP4。录像开/关使用同一个 evaluation profile，训练允许关闭相机；声明原有 train/eval 噪声和事件差异，禁止通过录像 profile 偷改科学条件。
- G1-29 Push 的 vendor 环境也接入目标 Lab/RSL 边界并实际执行。P2 不新增 Go2 Push 任务或通用 generation 编排；但不以留用旧生产后端规避 G1-29 兼容问题。

### P2-B2：真实运行、退役与交付

- 先 CPU/短 GPU ABI 和物理对照，再短训练/保存/加载/续跑，再代表性完整预算训练；出现基础 ABI 或任务语义错误时先修复，不用增加训练预算掩盖。
- 完整训练选择 Go2 Stair reference，从随机初始化执行现有 4096 envs × 24 steps × 2000 iterations = 196608000 control steps。初稿将 rollout 长度误写为 48，已按冻结协议的 `steps_per_env=24` 更正；执行器一直直接使用协议，没有改训练参数。短运行使用明确的诊断预算，不能签成完整 reference 资格证据。全阶段不调用 LLM provider。
- 训练/eval/video 与失败退出均核对实际子进程、GPU 占用和独立日志；完成后切换正式入口与环境说明。临时隔离副本按复用价值保留。
- 在新路径验收后删除项目内旧离线 checkpoint-video renderer、独立 actor/terrain 复制逻辑和对应无有效消费者的测试。原生验收材料与旧正式证据保留；不删除既有 checkpoint，也不为避免清理而建立另一套 production fallback。

## 3. 预先固定的验收

迁移对照只消费已公开的题面/seed 和冻结 reference checkpoint，不使用 sealed holdout，不调整任务指标。比较与结果脚本放本 Phase，产品能力放原所有者模块，不建立新的实验管理框架。

| 验收 | 固定执行与判据 |
|---|---|
| API / 静态合同 | P1 53 项已通过；P2 补实际 RSL 配置映射、旧/new checkpoint、Gym ID、robot source 与日志路径等受影响测试，最后本地全量与目标 Python 专项通过。对构造器 raise 不按数量堆砌反例 |
| 策略 ABI | G1 / Go2 同一保存观测输入比较旧/新模型，维度与顺序完全相同；CPU `atol=1e-6, rtol=1e-5`，GPU `atol=2e-5, rtol=1e-5`。actor/critic/normalizer/optimizer 迁移状态分别记录；不支持的旧 optimizer 不用于新训练恢复 |
| 一步与短程物理 | 相同初态、相同动作，记录 base/joint/contact/reward/done：reset 位置/关节/姿态绝对误差 ≤1e-5；一步位置/关节 ≤5e-3、速度 ≤0.1、reward 误差 ≤max(1e-3, 1%)，非临界终止一致。固定 32 步轨迹报告分叉时点及幅度，定位接触/RNG/资产来源；未解释语义偏移不得 GO |
| 公开分层对照 | G1、Go2 使用各自完整 public_dev cohort 和已有 3 个 seed，对照旧/新运行时的 C0/C1/C2 及 family 计数；同层成功率绝对差 ≤0.10、J_primary 差 ≤0.03。超限先定位环境/ABI 差异，不重挑题面、seed、指标或 reference |
| 原生录像开/关 | 同一 runtime、checkpoint、cohort、seed。固定小样本计数一致；完整 3-seed 对照以同层成功率差 ≤0.05、J_primary 差 ≤0.02 为上限，任何差异都记录原因。至少产生 Go2 C0/C1/C2 与 G1 代表性录像；连续两次录制/结束无崩溃 |
| 真实训练 | G1 / Go2 小预算随机初始化训练、保存、加载、续跑成功；Go2 完整 196608000 steps 完成，checkpoint 有效且能评测/原生录像。不以策略表现提升作为迁移完成的替代证据 |
| G1-29 Push | 目标运行时构造 vendor 环境、真实短 PPO、checkpoint 加载与一次施扰/eval，确认 action/observation 接线与任务计数有效 |
| 并发和失败 | 两块 GPU 同时启动独立短 lane，Kit/stdout/TensorBoard 路径分离；一次受控失败完整收尾，无遗留 GPU 子进程。沿用现有 lease，不清空共享锁根目录 |
| 视频实物 | FFmpeg 完整解码；帧数/FPS/仿真时间与输出时长匹配。发现时间重采样缺陷后收紧为帧差 0，另以带序号图像实际编码/解码验证每一帧顺序；抽查首/中/末帧，画面为同进程 Isaac/PhysX/RTX，非离线轨迹渲染 |
| 历史保护与单一入口 | 71 个正式 checkpoint 按原 P0 保护基线全部核对 SHA256（本轮已将 48 个中间模型从 stat 加强为内容校验）；801 份正式证据和 6 份 P5 reference 文件 hash 不变；新默认入口只指向目标后端，旧 reader 不改历史身份 |

完整训练前再核对 GPU 状态并冻结本次执行源码。最多一份完成完整 Go2 预算的训练；诊断阶段总 GPU 消耗先限定 4 GPU-h。初始 4 h 墙钟估计经真实 4096-env 前缀测量不成立：稳定每 iteration 约 10–14 s，2000 iterations 需约 5.6–7.8 h。`go2_full_train_01` 在早期由受控进程树停止，保留为吞吐诊断，不能作为完整训练证据。完整预算作业墙钟上限据此修订为 10 h，步数、seed 和 PPO 参数不变；禁止将较短训练称为完整训练。

## 4. 记录和阶段退出

`go2_full_train_02` 已停止：实际 config 审计证明它继承了上游新的 Go2 随机化条件，不能用于本阶段的对齐验收。停止后已核对父子进程消失、两块 GPU 无计算进程。只有实际配置与物理/观测对照闭合后，才启动下一份完整训练。

本目录维护实施日志、运行时事实、ABI/对照结果、实际命令及验收记录。P2-A 的通过仅表示迁移/ABI；P2 GO 必须同时具备真实完整训练、目标运行时 G1/Go2/Push、录像/并发/退出与历史保护证据。源码变化后只重复受影响验证；日志、旧试错和正式证据明确区分。

修复前 `go2_full_train_03` 的实际 lease 为 2.4075 h，但其 669 条 PhysX 容量错误使这份训练失去资格，该时长也不能代表有效物理下的吞吐。原始失败证据保留。容量修复后的 `audit2_go2_full_train` 按用户要求停在 iteration 630/2000，不计作完整训练。

用户释放资源后，`audit3_go2_full_train` 以同 seed 从随机初始化完成 2000 iterations，核心 PhysX error 为 0，实际 lease **8472.390263615 s（2.353442 h）**；新 checkpoint 经正式准入完成原生录像和部署后的关录像对照。没有拼接中断前缀、改变步数或用旧 checkpoint 替代。10 h 仍是执行截止上限，不是这次实测耗时；P4 以实测回执重新签发成本合同。开发诊断总量未完整计量的限度仍须明确，不把预算估计写成已验证的实际消耗。

若观察到计划假设与已安装实现不符，先记录根因并更新本计划的对应实现，不扩大任务权限或改变任务语义。P3 新指标/新协议 reference、P4 通用 session、P6 Go2 Push、P7 正式搜索仍各自独立验收。

边界补充：P6 冻结的 generation cost v4/v5 含 13.8 / 68.3 GPU-h 总包络，不能承载新后端的 10 h 单训练截止时间。其 reader 和旧预算保持原解释；相关历史 foundation 测试显式使用原 7200 s 输入。P2 完成单作业执行后，P4 必须依据新实测成本重签会话成本合同，再由 P7 启动搜索。不能只提高 timeout、暗增冻结旧预算，或把旧 generation 配置称为新后端可直接开跑的配置。


## 5. 公开迁移对照与工程 GO 的边界

原 `results/public_backend_comparison.json` 保留为接触历史修复前的结果。最终接触/容量修复后的结果见 `results/audit_final_public_backend_comparison.json`：G1 116/216 → 118/216，Go2 187/216 → 192/216。预设阈值与超限项保留，**不把数值等价检查改写为 PASS**。项目总目标是同一新后端上的科学闭环，旧 checkpoint 继续作为诊断 anchor。

原配置漂移已在机器人 adapter 修正；剩余原生运行差异保持版本身份。Lab 3 的质量扰动从 CPU 采样迁至 asset device，参数仍为 uniform(-1,3) kg，实际同 seed 样本不同；源码与实测范围单独存证。G1 的自然 reset 位姿、关节、命令完全一致，一步对照通过，接触阶段开始出现短程数值分叉；共同配置、资产、动作与观测定义均已核对。新增的 external-forces-every-iteration 配置与旧 USD schema 的隐含默认同为 false，并非漏改的开关。

不为复刻旧随机流另写一层事件系统，也不修改任务指标或 PPO 以追平旧 checkpoint 的分层成绩。旧/new 数值等价未成立与新后端工程闭环分别报告；P3 必须在新后端重训并资格化 reference。本阶段其余实测项和已确认的语义缺陷均已闭合，工程 GO 的证据见 `results/audit_completion_acceptance.json`。
