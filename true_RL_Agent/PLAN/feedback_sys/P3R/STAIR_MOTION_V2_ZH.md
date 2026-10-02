# Stair R1/R2：有方向的数值过程与默认视图

日期：2026-09-29。状态：**工程及旧开发档案独立数值复算通过；独立语义确认和作者价值门未完成。** 此处没有新增仿真、GPU 作业或 API 调用。R0-b 的 Stair 纯数值整理臂尚无达标提案，因此本次交付不构成楼梯任务的价值证明。

## 实现边界

正式实现只有任务内 `src/true_rl_agent/tasks/stair_climbing/behavior.py` 及针对性测试。已有采集字段足够，不增加采集器、姿态本体、检测服务或公共层任务语义。成功、终止、ABI 和原四类条件阈值不变。

新派生版本为 `stair.observed_motion.v2`，完整档案为 `behavior.full.v2`，有界视图为 `behavior.public.v3`。旧 `public.v1/v2` 读取原样、幂等；旧 `full.v1` 仍按已有字段投影为 `public.v2`，不能凭空补新量。新旧 rule version 比较明确不可比；同版本比较仍按具名列、完整 stratum 分母。没有复制一套旧生产检测算法，也没有回写历史证据。

完整规则与字段说明另存 [stair_behavior_r1_rules.json](results/stair_behavior_r1_rules.json)，供另一实现从原始数组独立核对，不以本实现的预期输出作答案。

## 增加了哪些可测事实

1. **articulation root 的方向与变化**：记录首末 `[x,roll,pitch,up_z]`；记录 x、signed roll/pitch、up_z 的最小/最大值和最早出现步；记录最大前进回退的峰值步、谷值步、距离。`up_z=1-2(qx²+qy²)` 表示机体正 z 轴在世界 z 方向的分量，另记首次小于零的步。Euler 角可能跨 ±π，不把跨界当连续旋转量。
2. **同一 support body 的未检测到瞬时承载且低运动区间**：瞬时力范数 ≤ 任务 contact_N，并且该参考点世界坐标 3D 速率 ≤ 0.05 m/s，连续 `ceil(0.5/dt)` 个采样才确认。它不是物理离地、静止姿态、脚底冻结或滑移的证明。历史力缓冲单独记录，不代替瞬时谓词。每段保留具体身体、起止/确认步、段内最大速度及同时段 root x 净变化。
3. **重叠而非互斥阶段**：low_progress、tilt、同体低运动可以同时存在；每段保留自己的确认时点。核心保留一个正式终止区间、一个已确认 tilt、一个已确认 low_progress；有新同体低运动时，再保留一段优先与低进展重叠的已确认区间。终止处截止不等于状态自然结束。

low_progress 仍为旧定义：满 0.5 秒窗口（含两端，共 `ceil(0.5/dt)+1` 个采样）内最大 x 减窗首 x < 0.02 m。机器人身体可以同时快速运动或后退，不等于全身不动。新低运动按连续采样数确认，不能将 `N` 个样本误写为首末时间差恰好 `N*dt`。所有 before_terminal 计数要求确认步 **严格小于** 终止步；终止当步才确认不会记入。姿态/回退的全回合极值包含终止步。

## 8 KiB 内怎样保留信息

默认报告仍为同一个报告，不根据实验问题手选窗口：

- 全部 stratum 的 n、success、原四条件与新同体低运动计数保留；原因细行可省略并登记覆盖数量，不缩分母。
- 每个有失败的 stratum，保护主导失败原因中有同 cell/run 成功、失败样本较多的一个组。不同终止原因不混合；缺成功明确为 missing。
- 每侧给 n、已确认条件数、出现 up_z<0 的回合数、组内最大回退、signed roll/pitch 范围。范围是全组极值，**不是典型姿态，更不是所有回合共同经历的一条时间线**；组内不同极值可以来自不同回合。
- 只保两条核心代表时间线：先在主要失败组中选最常见 `(tilt,low_progress)` 条件组合，再在组合内部按时长中位和稳定编号选例；成功侧独立应用同样选择。`pattern_n/group_n` 只声明这一条件组合的覆盖率，不冒充完整轨迹模式覆盖率。

为腾出组级有符号范围，`public.v3` 不再展示 last-1s 任意出现列（没有持续要求）；它们仍在完整档案。删去已省略的上升/下降规则专用阈值、未展示规则目录，以及每条时间线重复的覆盖对象；公共时间定义统一说明观测为 step 1 到 terminal，完整档案的 coverage 保持。没有扩大预算。

## 真实 864 回合开发验证

原始输入为 [manifest.json](results/stair_numeric_dev_r1/numeric_evidence/manifest.json) 及其 30 个 NPZ chunks，manifest SHA256 为 `22fb749620c4e614cbb4df4fbfdb82b70d76c8f1cd81a3eda50de465d9fb3e4a`。chunks 逐个核对原 manifest hash。完整生产派生一次约 155 秒 CPU；输出 [stair_motion_v2_full.json](results/stair_motion_v2_full.json)，26,718,265 B。后续投影调整没有重复派生或重新采样。

[verify_stair_motion_v2.py](verify_stair_motion_v2.py) 用 NumPy 直接从原始数组另算四元数方向、所有极值/时间、累计峰值回退、同体谓词的布尔连续区间，与生产结果核对全部 **864 回合、3,312 段**，通过；保留对照的每侧新计数和范围也逐项复算。源 manifest、旧档案、冻结 prompt 均未改动，原终局、steps 和旧四条件计数逐回合相同。

固定速度阈值 0.05 下有 34 回合在终止前确认新低运动条件；仅做 ±10% 敏感性检查时为 33 / 34 / 36。没有据此调阈值。当前默认保留的五个禁触对照组内这一新条件都是 0；34 回合的发生范围在全 stratum 计数中可见，**没有为此手选 timeout 个案替换主导失败，也没有声称默认时间线展示了新条件的具体身体过程**。

最终 [stair_motion_v3_view.json](results/stair_motion_v3_view.json) 与 [stair_motion_v2_validation.json](results/stair_motion_v2_validation.json) 记录数值、来源/代码 hash 和实际尺寸：

| 项目 | 结果 |
|---|---:|
| 默认行为视图 | 8,073 / 8,192 B |
| 完整 stratum 分母 | 8 × 108 = 864 |
| 保留失败 stratum 对照 | 5，全部有同 cell/run 成功 |
| 省略原因细行 | 24，全部仍进入 stratum 汇总 |
| 实际 provider user JSON | 70,543 B |
| 实际 HTTP request JSON | 76,933 B |
| task sections | 40,599 / 57,344 B |

五个有失败的层为 frontier、rough/c2、height、joint、tread；phase 在此旧池为 108/108 成功，不能制造失败对照。各对照失败/成功分母为 9/2、3/6、7/5、6/4、4/8。完整报告仍保留所有层，不按问题反向缩分母。

自动失败代表为 `r1.e64`：主要组 9 个禁触回合中 7 个共同满足 tilt+low_progress，从这 7 个中按确定规则选取；不是按编号指定。观测 low_progress 369–426，tilt 385–426（394 确认），up_z 首次负值在 418，formal 禁触在 426；最大回退 0.647620 m 从 348 到 426。成功代表 `r1.e60` 属于同 cell/run，仍出现低进展和倾斜区间，保留成功反证。不能把代表的一整条先后关系推给 7/9 个回合。

实际序列化尺寸使用冻结旧首包替换视图、重新生成生产 frame/预算 receipt 后，经实际 provider request serializer 计算，**未发 API**。此临时尺寸探针的 `source_public_result_hash` 没有重新绑定，因此不是新合格作者证据、没有保存成可投递实验输入；正常生产路径仍由 phase 派生封存来源。另有真实 CPU fixture 经 phase finalizer → public projection → PublicContextBuilder → serializer 的回归，验证绑定与投递通路。

## 回归与仍然存在的边界

命令：

```powershell
$env:PYTHONPATH='src'
.venv/Scripts/python.exe PLAN/feedback_sys/P3R/verify_stair_motion_v2.py
.venv/Scripts/python.exe -m unittest tests.unit.test_stair_behavior tests.unit.test_p3r_stair_projection tests.unit.test_prompt_capacity_projection tests.unit.test_stair_phase_d_prompt_preflight -q
```

最终 **42 tests PASS**。覆盖有/无历史、身体顺序与姿态镜像、方向和终止当步、低速阈值与持续边缘、交替身体不能拼成长持续、已加载历史不能冒充瞬时承载、代表选择、六失败层容量压力、具名列重排、缺同组成功、旧 public 幂等、生产 source-followup/static-repair/reasoning-exhausted/verifier 等最终序列化回归。没有增加无关安全测试。

这些结果证明代码提取和压缩忠于所查原始数组，**不证明这些量已足够回答所有行为问题**。8 KiB 下只有一组长时间线，其余层只有同作用域计数和范围，无法恢复所有层的时间顺序、受力机制或多模态过程；后续 R3 对某层的必要顺序提问仍须按原问题判不可答，不能缩减题目分母。未知现象只描述已测量事实；既有条件组合不能代替新语义确认。正式 R3/R4 仍需冻结新独立样本、核对原始曲线/视觉和作者正确引用，未过门前不进入 P4。
