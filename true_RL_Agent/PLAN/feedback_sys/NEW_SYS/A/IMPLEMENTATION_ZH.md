# A 阶段：行为语义与作者内容

本阶段以 10-03 审计和 `CRITICAL_REVIEW_AND_MINIMAL_PLAN_ZH.md` 为范围，从 `5bcbe35d` 开始；代码提交 `f37f9fa4`，R1b 暂停。10-04 新材料未读取。先在原数组上建立[独立行为预期](BEHAVIOR_EXPECTATIONS_ZH.md)，再核[实际最终作者文字](FINAL_AUTHOR_EXAMPLES_ZH.md)。R4、K1、K2、m50、m300 均已用于开发，不能称 unseen。

## 改动及取舍

- Stair 当前生产报告改用带 `label/unit/value` 的行为事实：总体运动、各结局末窗、成功对照、逐足接触/世界与身体相对点速/相对位置/周期、名义关节上下边带，以及地形参考差。不再按旧18/23开发案例槽拼作者内容；必要内容由实际结局和实体决定，没有固定句数资格。
- 核心事实先覆盖全队列、结局和每足；条件细节按回合覆盖、过程差异排序，姿态和地形对照按已有事实加入。接触组合由实际实体数及频率选出。archive ID 改变不改变作者事实选择；Push 的代表回合也从哈希最小改为接近组内速度中位数。
- 保留10240字节上限。R1a 来源验证仍在正常阶段边界完成，作者只接收简洁 report 和原有来源关系，长哈希留私有产物。一份定义解释统计与边界，避免每句重复限定。
- 点运动统一用相邻连续位置差分；世界漂移与身体相对变化分开。记录速度通道不能未经验证代替位置运动，也不将点运动命名为迈步。末0.5s速度只用两端均在窗内的差分；完整进度窗的确认时刻另计。
- 初始历史不足记不适用；内部缺步或非有限值记未知。低正向进度允许倒退，另报有符号末速及反向位移。姿态使用身体上轴、前轴主值角和相邻四元数短弧，不以全局 Euler unwrap 推断翻转。
- 接触和周期共用任务阈值。名义关节上下边带报告全队列占用；容差只吸收名义边界转为记录精度的实际量化误差，原始signed margin保留。K1右后calf的−0.0856rad未被吞掉。地形是记录参考点相对名义表面的高度，可按结局末窗比较，不叫sole clearance。
- Push 高度/速度列直接由 `POSE_FIELDS` 命名索引，label/unit随数值进入聚合和文字；删除固定20条资格。Push扰动恢复语义继续由任务算子负责，没有硬合为通用平台。CLI预算默认值由composition配置提供。

删除的是 tracked 文件中的旧生产 `DEFINITIONS`、固定槽 `_extension_claims`、旧槽到新槽映射及其固定数量验收；原实现可由上述基点恢复。未删除整个历史文件、用户审计、原档、checkpoint或运行数据。没有新增registry、历史白名单或工作区换行哈希要求。

## 验证结果

五档各864回合由原NPZ重算，经当前生产报告投影；独立 NumPy 检查不导入项目测量/聚合/渲染代码。具体定义、原路径和run/env/step锚见预期文档及 `raw_probe.py`。[独立文字复核](FINAL_AUTHOR_TEXT_REVIEW_ZH.md)记录检查范围及发现的修正，不将同源重放算作独立行为证据。

例如在 `true_RL_Agent` 下执行 `.\.venv\Scripts\python.exe PLAN/feedback_sys/NEW_SYS/A/raw_probe.py outputs/trajectory_p0/raw/numeric_evidence --nominal` 可独立复算R4；m50/m300使用预期文档中的本地目录，省略 `--nominal`。脚本只读NPZ并输出JSON。

| 开发档 | 公开报告 bytes | 作者 profile bytes | 代表性实际区分 |
|---|---:|---:|---|
| R4 | 10204 | 8747 | height_ood接触标签38回合末速0.668m/s，仅3回合曾上轴朝下；区别于c1前沿倒置较多的过程 |
| K1 | 10097 | 8668 | 左后接触18.5%；前沿接触标签末速0.594，窄踏面−0.253m/s且22/25曾上轴朝下 |
| K2 | 10230 | 8801 | 低参与足为右后48.4%；主组合为四足及FL+FR+RL；c2末速0.411m/s |
| m50 | 6548 | 5136 | 864超时，净位移0.13m、末速0、足末窗相对点速约1e−4m/s、无合格周期 |
| m300 | 8135 | 6722 | 198接触标签1.46s结束/末速0.501m/s，与666超时12s结束/末速0分开 |

字节为UTF-8紧凑JSON；公开报告包含验证信封，作者profile不含该信封。接触百分数为逐回合占比的中位数，组合另为回合均值。以上标签均来自存档结局，不直接等同运行时终止触发器。

受影响回归共170项通过：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_trajectory_behavior tests.unit.test_trajectory_measurement tests.unit.test_trajectory_cohort tests.unit.test_trajectory_temporal tests.unit.test_trajectory_cycles tests.unit.test_stair_trajectory_production tests.unit.test_stair_numeric_dev_replay tests.unit.test_push_trajectory tests.unit.test_push_trajectory_semantics tests.unit.test_push_trajectory_pipeline -q
```

其中6项行为测试覆盖静止但速度通道偏置、整体平移、原地腿动、高举单足、倒置/跨角度边界、N/A与真实缺失、倒退、两足5N契约及精度边界。一个Push高度/速度公共列映射破坏被独立预期捕获。没有扩展全字段mutation矩阵。额外复用现有Push内存夹具，1/5 strata分别产生17/21条required，最终投递8061/10115 bytes，均完整通过；原4 strata测试中的20仅是该夹具结果。

正常R1a CPU生命周期采用原历史harness，仅适配作者profile缩减后的比较：composition → finalizer → phase → generation → provider HTTP截获。最终版本通过 `PASS_CPU_COMPOSITION_ARCHIVE_GENERATION_HTTP`，三个来源均完成，下一轮 `g2-lane-a` 精确收到 `generation-001-lane-b-eval-public-dev` 的report；各作者profile仅有kind/report，最终标签和精度措辞确已进入HTTP payload。原始CPU执行产物在 `D:/humanoid_agent/.tmp/a_lifecycle_final/verification.json` 及同目录 `requests/`，未另增冻结收据。训练、非公开结局和作者草稿是夹具；原NPZ未改；模型调用、HTTP网络调用和GPU运行均为0。不验证真实checkpoint加载、Isaac执行或模型理解。

## 阶段边界

A没有新增GPU、模型API、训练或采集，没有读取不相关K5数据。m50/m300只读取得现有档案并核对60块文件；缺名义关节/地形绑定时明确缺口，不扩历史白名单。10-03审计原SHA仍为 `831679f465d4e993df896ec8a6f51a61154b9b75c4272fcbdf84214d5dda62c4`，审计目录保持untracked。Windows深路径表观删除已在正常权限下只读确认并非真实删除，没有提交删除项。

B仍需删除旧证据/固定案例生成链及大量无人消费的区间关系，统一Stair/Push正常接线、合理description失败降级、明确父代匹配。本阶段生产文字已使用新的事实摘要，但私有来源验证仍经过旧base产物；尚未声称完成B的生成链瘦身。新事实结构可直接移入统一路径，无需保留旧槽兼容框架。

m50摘要没有单列“2s后”这一精确持久时段；现有全程/末窗点速和周期足以修正主要行为，未为单档添加专用算子。成功对照未按条件实例配对，实际物理失败原因和作者收益仍未证明。完成提交推送后停在A独立验收，不继续R1b或B。
