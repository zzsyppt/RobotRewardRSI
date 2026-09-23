# Push 外力评价统一：实施与验收边界

2026-09-22。用户批准将 G1/Go2 的当前 Push 评价统一为同一套任务规则，机器人差异由已有 RobotProfile、runtime adapter 和薄绑定解析。本轮只统一评价，不重写历史训练域，不训练，不调用 provider，不读取 accept/sealed，不据此宣布新的 reference 资格 GO。

## 1. 唯一活跃评价语义

- 新 cell：`push.g1_37dof.recovery_v3`、`push.go2_12dof.recovery_v3`。两行共享一个外力评价协议；旧 cell、协议及 checkpoint 保留历史含义，退出新会话的默认评价入口。
- 尺度来自绑定 RobotProfile 的 `morphology.default_root_height_m`，记作 L；G1 为 0.74 m、Go2 为 0.40 m。`V=sqrt(gL)`、`T=sqrt(L/g)`，g=9.81 m/s²。不能用 Go2 的物理米值替代 G1 参数，也不将尺度来源改为 Stair 的 support-center 长度。
- low/medium/high 的等效比冲量为 `(0.6, 1.2, 2.0)*V`。这完整对应 J 的 D 协议幅值再乘 2，不是在旧默认幅值上仅乘 2。repeated 每推为同环境 high 的 80%，第二推方向相对第一推正交。
- 每次施力为绑定 base link 的 COM 处世界水平短脉冲，直接施加的 torque 为零；不写根速度。质量 M 为 reset 后实测的全 articulation 各 link 质量之和。`F=M*等效比冲量/实际脉宽`，不承诺接触约束下根速度变化恰等于该等效比冲量。
- 脉宽为 `ceil(T/control_dt)*control_dt`；当前 Go2 为 0.22 s、G1 为 0.28 s。两推之间的无力空档固定为 **0.22 s**，不随机器人 T 缩放。第二推起点由首次量化起点、实际脉宽和固定空档相加，不能再令空档等于脉宽。
- 恢复从最后一次力脉冲结束后的下一 control 样本重新累计；在缩放后的恢复窗口内完成连续驻留，窗口末样本完成有效。第一推之前或两推之间的稳定驻留不能借用。
- 全程保留姿态/高度失败，所有计划脉冲须完整且验证通过，回合末步还须稳定。完成驻留后并不要求每一步持续满足恢复阈值，但任何时刻触发保留的失败门仍失败。位移仅记录，不导致失败或提前终止。
- 高度、恢复平面速度、角速度、恢复窗口、驻留和回合长度依统一无量纲规则及各自 L/T 解析；不将 Go2 的 81/16/303 步直接写进 G1。角度阈值为共同任务语义。

## 2. 身份与产品边界

原训练 manifest、training binding、reference program identity、checkpoint SHA 和旧训练环境身份分别保留。新评价 cell manifest、评价协议、解析参数和执行环境单独记录；不得称现有 checkpoint 已在新外力任务上训练，也不得把新评价结果回填为旧正式资格。

新 `PushRecoveryTaskConfig.for_cell` 保留源 `push_protocol` 训练条件及 reference ID，另以 `push_evaluation_protocol` 表达单一活跃外力评价。机器人 ABI、角色和尺度来自既有 registry；力注入、状态采集、失败/恢复计分属于 Push 任务。共享 Runner/Genesis/资源管理不新增 Push 判定。

旧配置从活跃入口退役时，先处理真实消费者；历史证据、冻结源文件、下载清单及回放身份不删除、不重标。旧训练配置仍是已有 checkpoint 的来源合同，不因本次评价迁移变成新协议。

## 3. 有界实际验收

`force_evaluation_acceptance.py` 只组合生产的 config/builder、checkpoint loader、runtime 和 `run_physical_push_rollout`，不复制 PLAN probe 的力注入、原始条件或评分函数。

每机器人在新评价基准 seed **1500000000** 下运行 public_dev 3×64，共192首回合，四层各48例。该批只做新评价实施验收，不消费 holdout；不能用 J 的旧 seed 冒充独立新样本。若以后另做 J 原 seed 复现，必须单独命名，不混入本批分母。

no-push 与 zero-action 工程控制分别使用首个 dev seed 的64环境、独立输出，既不进入正式192例也不用于资格分数。no-push检查零事件、控制状态和末态；zero-action如产生真实失败，保留提前失败/partial pulse，不假造坏注入或强制姿态来取得失败样本。

原生视觉在数值结果后按真实成功/失败记录选例，保留选定 seed 之前的运行历史。只使用生产录像接入；记录 seed/env、完整 episode outcome、同 checkpoint/ABI/评价身份。核对前复位帧、完整解码数量和同回合数值结果，录像重放不增加统计样本。若某机器人没有某类结果，如实记录缺失，不修改规则制造该类。

driver 不获取第二层 GPU lease、不自动重试。目标执行使用既有 `run_owned_probe.py` 持唯一 lease、墙钟上限和过程日志；远端执行、传输和资源收尾由主任务负责。本地仅做必要 CPU 接口与身份测试。

## 4. 完成条件

两行当前 ABI 与 inference checkpoint 加载通过；实际任务外力的施加、持续、清零及事件证据闭合；192例计数与分层一致；控制组与正式分母分开；末态/失败录像可核对；旧训练和新评价身份没有混写。保留历史证据原字节。真实失败率不构成工程失败，工程验收通过也不自动构成 reference 资格 GO。
