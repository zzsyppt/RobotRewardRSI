# Push R1/R2：过程派生与默认报告

本轮只修改任务内 `behavior.py` 及行为测试。奖励输入 ABI、任务成功标准、训练逻辑均未改变；生产代码不导入 PLAN。Phase/runner 的 CPU finalizer 由并行任务负责。

## 版本与事实边界

- 新完整工件与默认视图：`true_rl_agent.push_recovery.behavior.v2`，规则 `push_process_v2`。
- 历史 v1 完整工件、已封存 public 仍按原 v1 规则读取。不同 schema/rule 的 parent diff 返回不可比，不能把新旧语义混成趋势。
- 数值归档 v1 缺 `planar_motion` 时，速度、姿态、条件与时间段仍可派生；事件净位移的观测数为 0、median 为 null，明确标为缺失，不能填零。

## 完整工件记录什么

`transcripts[*].recoveries[*].process` 保存 recovery anchor、注入开始帧、首个观测响应帧、窗口 first/peak/last、峰值时点、姿态极值、窗口净位移和 post-latch 不稳定区间。

Velocity 的 anchor 是该次实际注入步；恢复窗口为 `anchor < step <= anchor + window_steps`。窗口净位移是窗口末位置减该次 anchor 位置，不能用从 reset 累计的位置代替。

Force v3 只有整段计划脉冲均完成且通过原验证后才有有效恢复窗口；anchor 是最后脉冲结束步。首物理响应仍按最后脉冲开始后的首个观测帧记录，与窗口首帧区分。Force 的默认净位移明确只计最后脉冲结束到窗口末，**不包含脉冲施力期间**。力/冲量仅引用原有物理钩子回执，未重建未记录的子步力。

五项稳定条件沿用 P2 原始精度判断，是 AND 关系。各失败条件可重叠；`condition_violation_spans` 按连续 control step 划分，端点闭区间。`process.stable_spans` 是全五项同时满足的连续段。三个区间视图复用同一段算法。dwell 与中断计数停在首次 latch，整窗条件计数不停止。每事件 post-latch 观察到下次实际注入或本回合末；整回合另保留完成全部恢复后的再失稳及最终稳定情况。

计划但未执行的第二次事件保留身份和缺失 anchor，不能与实际执行后的零稳定窗口合并解释。Force control 的一个 reducer 存储槽不再被描述成一个失败/删失恢复任务。

## 默认报告如何控制大小

固定 12 KiB，保留来源身份、全体结果、四个强度组分母、首个逻辑运行统计，以及每强度主要失败画像的合法成功对比。匹配条件为强度组、协议名义强度、计划/执行/验证/完成事件数；方向可以不同，所以是描述性反例，不是因果配对。

代表例来自实际条件下最大的失败群体，再按窗口稳定占比中位附近选取；成功必须满足同一事件条件，缺失就明说，不拿其它强度或不同实际事件数凑对。每例详述一个尚未恢复的已观测窗口，否则详述最后窗口；其余所有计划窗口保留状态、anchor、观测/稳定步、dwell、latch、删失与再失稳计数，并明确省略过程详情。统计不会被代表例代替。

公开数值显示到小数点后六位，原始完整值仍在 full；阈值及身份不被显示取整改写。对比行使用带标签的短文本，避免大量列索引。极端预算压力只可省略代表例的可选最终 pose，失败条件、主要过程、各组分母和匹配对比保留；核心仍超限会显式报错。

## 验证与限制

离线入口 `verify_push_behavior_r1.py` 用原 manifest SHA 绑定本地 locator，不改写原 phase。独立旧提取结果逐项比对 192 回合、240 计划恢复单元、227 个实际观测窗口；另直接从 NPZ 复算 1,811 个条件区间、485 个全条件稳定区间、542 个 post-latch 区间。历史 v1 full 投影与封存 public 完全一致。

结果见 `results/push_behavior_r1_verification.json`；unit log 见 `results/verification_push_behavior_r1.log`。这只是既有开发档案的源代码/证据一致性验证。R3/R4 的新来源、事实覆盖与作者改进价值必须由新冻结数据另行判定；不能把本轮 PASS 当作新策略能力或因果结论。
