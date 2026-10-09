# Push Recovery 任务指南

核对日期：2026-09-22。当前 G1/Go2 统一为 v3 外力评价，任务规则与机器人适配分层。此次实施只改评价，使用原有 checkpoint 做新 public-dev 验收，不训练、不读取 accept/sealed、不调用 provider。工程通过不代表新 reference 资格 GO；历史 G1 的资格仍只在原速度增量域有效。

## 活跃行与身份

| Cell | 评价与机器人 |
|---|---|
| `push.g1_37dof.recovery_v3` | 默认行；G1-37 grouped runtime，L来自其RobotProfile，评价使用统一外力协议 |
| `push.go2_12dof.recovery_v3` | Go2-12 runtime；同一评价协议，经Go2尺度解析 |

[任务配置](../../src/true_rl_agent/tasks/push_recovery/config.py)、[外力评价协议](../../src/true_rl_agent/tasks/push_recovery/force_protocol.py)和[robot binding](../../src/true_rl_agent/tasks/push_recovery/robot_binding.py)是活跃入口。`push_protocol` 保留训练条件；`push_evaluation_protocol` 单独拥有评价语义。新评价 base seed 为1500000000。已有 checkpoint 验收须显式带入原训练协议、reference ID、manifest及checksum，不能把旧模型声明为在新外力域训练。

旧 G1-37/Go2 `recovery_v2` 退出新会话 CLI，保留冻结 manifest 的历史读取和回放。G1-29 的显式历史审计入口仍保留，不属于当前矩阵。原 G1 1.5档训练配置归档到 [历史训练协议](../../PLAN/2new_world/P6/history/push_g1_37dof_velocity_training_v3.json)；原资格结果与声明见 [P6 验收](../../PLAN/2new_world/P6/ACCEPTANCE_ZH.md)。冻结历史不使用新默认重新解释。

## 评价成功与失败

- 在绑定base link的COM处施加世界水平短脉冲，使用reset后全机器人实际质量M。low/medium/high等效比冲量为`(0.6,1.2,2.0)*sqrt(gL)`，力为`M*比冲量/实际脉宽`；repeated两推均为high的80%，方向正交。
- L来自各RobotProfile的`default_root_height_m`。脉宽为`ceil(sqrt(L/g)/control_dt)*control_dt`；无力空档固定0.22秒，不随机器人尺度改变。高度、速度、角速度、恢复窗口和驻留用各机器人尺度解析，不移植Go2的物理阈值到G1。
- 全程不得触发保留的倾倒或高度失败；位移仅记录，不再导致失败或提前终止。早跌回合保留在计划分母内。
- 所有计划力脉冲完整且验证通过后，从最后受力样本的下一步开始，在恢复窗口内完成一次连续稳定驻留，截止样本完成有效。第一推前或两推之间的驻留不能借用。
- 回合末步还须稳定。完成一次驻留后不要求每一步始终满足恢复阈值，但任何时候触发倾倒/高度失败仍失败。no-push控制组单列，不进入正式恢复分母。

## 训练与组件权限

本轮未改变历史训练分布、reward source或checkpoint。原训练的速度增量协议由[训练协议实现](../../src/true_rl_agent/tasks/push_recovery/protocol.py)解释；旧checkpoint原训练环境与新评价执行环境分别保存。

候选reward仍在所选cell冻结的训练合同内执行，`PushRecoveryV2RewardInputs`只公开当前/已发生事件信息，不提供未来推搡计划。完整reward由task-owned单一term执行，RewardManager仅按control dt积分一次；候选不能改变PPO、预算、初始化、动作/观测ABI、评价场景或成功门。机器人模块不拥有Push判定，公共Runner不根据机器人名称推导阈值。

[PhasedEvaluationAdapter](../../src/true_rl_agent/tasks/push_recovery/phased_adapter.py)继续由共享宿主注入；任务拥有分数、资格和公开反馈，宿主拥有生命周期、成本、身份和ledger。正常资格流程保存dev receipt后先检查任务dev门，再决定是否继续holdout；此次有界实施验收不进入该资格流程。

## 本次验收与历史证据

范围和预算见[实施计划](../../PLAN/2new_world/P6/FORCE_UNIFICATION_IMPLEMENTATION_ZH.md)，结果见[统一验收](../../PLAN/2new_world/P6/FORCE_UNIFICATION_ACCEPTANCE_ZH.md)。[验收driver](../../PLAN/2new_world/P6/force_evaluation_acceptance.py)只调用生产builder、checkpoint loader和rollout：每机器人新dev3×64；Go2为141/192、G1为33/192。no-push、zero-action和原生录像分别输出，不混入正式192例。目标使用已有owned-process/GPU lease wrapper，0自动重试；本地CPU预检不能替代目标物理证据。

2026-09-22用户后续修订已接入当前Push新reference默认准入 `observed_outcomes_v1`：dev须有完整有效证据，并至少出现一次成功和一次失败；public_accept与sealed须观察到成功。固定成功率区间不再用于该新准入；相对改进、统计判据及accept→sealed最大0.1下降门保持。两台已有dev测量复核均PASS，说明满足这项最小正反例要求；近饱和仍根据分层与实际结果判断，不把1/192或191/192通过自动门写成余量充分。新域三相完整资格仍待验证。旧manifest沿原门解释，不修改Stair冻结题池或评分，见[准入修订](../../PLAN/2new_world/P6/REFERENCE_ADMISSION_REVISION_ZH.md)。

历史G1原域资格为dev96/192、accept106/192、sealed245/448；Go2原域C标定及D匹配训练超过余量上界，仍按原NO-GO解释。J取消位移门后的142/192属于固定旧checkpoint的公开诊断，不能替代两机器人新外力评价的正式reference资格。P6资格闭合前，不据注册、录像或本次dev检查宣布P7已就绪。
