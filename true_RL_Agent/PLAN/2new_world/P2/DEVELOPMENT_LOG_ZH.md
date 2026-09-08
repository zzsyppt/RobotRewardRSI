# P2 开发记录

2026-09-07。**完整训练及新产物评测/原生录像已完成，P2 工程 GO；旧/新数值等价未成立。** 最终逐项证据和部署状态见 [验收记录](ACCEPTANCE_ZH.md)，范围与诊断阈值见 [实施计划](IMPLEMENTATION_PLAN_ZH.md)。

## 主线实现

- P1 API 收尾已闭合，53 项定向测试通过；统计版本、LLM 历史描述和构造器不变式分属原所有者，未扩充通用防御层。
- 新生产入口固定 Sim 6.0.1 / Lab 3 / PhysX / RSL-RL 5。冻结 PPO/Policy 合同通过一个后端映射执行；旧模型状态只有显式 inference reader，旧 optimizer 不迁移。新模型、normalizer、optimizer 和 iteration 的保存/加载/续跑均真实验证。
- Isaac 的数组和四元数变化收敛在后端适配边界；任务继续拥有 reward、题池、episode outcome。Gym ID 不再误标 G1；RobotProfile 的 PPO 查找委托既有运行时配置来源。
- G1-29 的 vendor API 在固定独立副本中适配，URDF 一次性转换为固定、自包含的 USD；新执行不依赖旧导入器，不向上游全局注入 alias。
- 原生录像接入正式 eval 环境和 episode recorder，只改变相机与编码呈现。旧独立 actor/terrain 导出栈及对应专属测试、父层旧 Conda 训练 smoke 已退役；详见 `retired_video_sources.json`、`retired_outer_tools.json`。
- 实测完整训练超过旧截止时间。执行超时归 composition/placement，默认 10 h，切换 cell 不覆盖显式值；协议的历史字段按旧解释保留，预算、seed 和 PPO 不变。旧 P6 cost v4/v5 不能承载新时长，由 P4 重签会话成本，P2 不改写其旧预算。

## 实测发现及源头修复

1. **实际物理配置漂移。** Lab 3 的 Go2 默认质量事件改为乘法 log-uniform，base COM、joint reset 和 armature 也改变，默认资产根由 Isaac 5.1 浮动至 6.0。由机器人 adapter 恢复绑定的参数，并从 RobotProfile 的 asset closure 取得真实 USD 根；接触采样周期恢复为物理 dt。4 份 train/eval 配置逐项审计通过，没有改 profile/task hash 掩盖漂移。
2. **RNG 变化与数值迁移边界。** 同分布不等于跨后端的同一随机样本：上游质量采样从 CPU 移至 GPU，Go2 实测同 seed 的质量最大相差 3.373 kg，但都落在原 uniform(-1,3) kg 范围内。名义质量、同初态动作时，Go2 32 步本体与关节状态完全一致。G1 自然 reset 的位姿、关节、命令一致，一步通过；接触阶段短程有分叉，新旧 PhysX 共同设置及额外开关默认值已核对。公开分层对照的超限项如实保留，不宣称旧 checkpoint 数值等价或新 reference 资格成立。
3. **录像首帧、区间和编码。** 人工追加的 reset 图像不是有效原生帧：SDK 此时尚未发布完整机器人画面，多次纯渲染不能补出该状态。撤掉这段追加逻辑，采用 control step 1 的原生首帧；从任务 outcome 的 E 步计数导出 1…E-1，排除 terminal auto-reset。初始化纹理仅做渲染预热，不多走物理步。随后帧序号复验发现旧 MoviePy 在当前 NumPy 下会跳帧和重复帧，已移除它的时间重采样，改为薄 Gym wrapper 采集实际 RGB、现成 FFmpeg 逐帧编码。469 帧实际 H264 编码/解码序号完全一致，包括实际 RGB 的非连续内存输入；最终四组真实影片及新训练产物的三部影片均完整解码，帧差为 0。没有 pre-reset 帧时记录缺帧事实，不虚构文件。
4. **历史 checkpoint 准入。** 正式 SubprocessRunner 正确拒绝缺少当前 cell 绑定的旧 anchor。P2 不伪造 metadata 来绕过它：历史对照明确经诊断 CLI 执行；完整新训练及其产出评测走正式 runner/attempt scope。均不生成正式 Session Genesis 或更新正式 ledger。
5. **P0 F39 误读。** 原问题只是 `support_effector.targets ⊆ support_body_names`，已在机器人层实现。更正 P0 对力作用点的附加推断，不新增无关约束。F59 的 dt × weighted rate 已实测；prompt 单位说明仍属 P5。

## 运行与验收事实

- 本地全量：1608 项，0 failure/error、32 skip，302.524 s；最终源码目标 Python 专项 20 项通过、0 skip。冻结旧 P6 foundation 测试显式使用原 7200 s 截止时间，原断言未削弱；新默认超时和实际传递由 composition 测试覆盖。
- `g1_resume_final`、`go2_resume_final` 各完成 384 control steps 的真实训练、精确恢复与续跑；`push_smoke_06` 完成 29-action、policy 480 / critic 495 维的同类检查及 100 步施扰执行。
- `runner_timeout_cleanup_01` 经正式 runner 在 30 s 截止，记录 infrastructure_failure / timeout / retryable=false；对应进程清空，GPU lease 可重新取得，未中断完整训练。
- `g1_resume_concurrency` 与 `go2_full_train_03` 的 TensorBoard 标量时间范围重叠 1.195 s，Kit/stdout/TensorBoard 路径均分离。该证据来自真实事件时间，不冒充同时刻的 nvidia-smi 截图。
- 部署后按 P0 清单核对全部 71 个正式 checkpoint：48 个中间模型大小/mtime、23 个权威文件大小/SHA256 均不变；801 份正式 JSON/JSONL 与 6 份 G1/Go2 reference 实物哈希一致。全阶段未调用 LLM provider、未消费 sealed holdout、未修改正式 ledger。

## 完整训练与运行位置

`go2_full_train_01` 因原墙钟估计不成立而受控停止；`go2_full_train_02` 因上述物理配置漂移而受控停止，均是诊断前缀，不能算完整预算证据。

`go2_full_train_03` 在 `/data/yuyangliu/isolated/new_world_p2_train_final/true_RL_Agent` 的冻结源码上，经 SubprocessRunner、pre-Genesis attempt scope 和单 GPU UUID 绑定执行。预算为 **4096 × 24 × 2000 = 196608000 control steps**。冻结文件清单为 `training_source_freeze.json`，运行中不修改该树。随后录像呈现层的修复在开发副本进行，其源码差异须与训练身份分别记录，不重写训练回执。

该训练已正常完成 196608000 steps，GPU lease 8667.011 s；其新 checkpoint 通过正式 runner 评测，159/216、J_primary=0.8754822146364883，随后原生进程正常退出。这个单训练 seed 的结果没有提升声明，也不替代 P3 reference 资格化。

两次 SSH 新连接中断造成一次录像两文件同步未完成，`go2_native_step_c0` 仅作诊断，终止帧不计最终验收；后续同步改为整包校验后发布。完整训练通道未因此退出。正式 checkout 同步的逐文件前置检查、删除清单和最终源码核对统一记在 `results/deployment_verification.json`，不根据本地文件推断服务器状态。

正式 checkout 已同步并复验通过，未创建 commit；两个 GPU lease 可重新取得，无 GPU compute 子进程残留。全量测试后产品仅改录像呈现，最终 20 项目标测试与 15 部真实影片覆盖该变化。复审同时纠正了计数口径：71 个 checkpoint 的保护依据是 48 个 stat 加 23 个内容 hash；最终保护回执按这两类明确报告。开发探针总 GPU-h 的计量限度在验收记录中单列，未将缺失时间包装成预算合规。
