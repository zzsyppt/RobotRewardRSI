> 后续验收更正：本首轮交付的数值检查通过，但父线程因离线 preview 语义绑定缺口暂不放行。新增最小修复、反例及复验记录见 [SEMANTIC_BINDING_REPAIR_ZH.md](SEMANTIC_BINDING_REPAIR_ZH.md)。原数值冻结与历史日志保持不变。

# P2c1：局部接触标记周期资格交付

基线 `ca140cfd1e6084118e0c36a4c28f8fe97a2dc294`，同一分支
`codex/trajectory-translator-upgrade`。规则先以 `c7aa200b62f697d1b16327c1278e2f89c067f60d`
提交，再读取本阶段原 NPZ；[冻结定义](FREEZE_ZH.md)不按真实结果修改。
本轮完成离线周期和相位资格、独立原数组核查、报告与预算预览，**开发自测通过，root 独立门待验**。
提交/推送后停写，不进入 P2c2/P2c3。没有读取 K5 行为或 P3 pilot 结果。

## 有界实现与定义

新增 `evaluation/trajectory_translation/cycles.py` 和 task-owned `trajectory_cycles.py`，
以及 `scripts/build_stair_cycle_supplement.py` 离线入口。catalog 显式为
`trajectory.p0.measurements.v3`、base 为 `trajectory.p0.measurements.v2`；新产物 schema
为 `stair.cycle_supplement_manifest.v1` / `stair.cycle_supplement.v1`。
旧 v2 测量/卡片/普通 public.v5 入口不变，不给旧卡悄悄加周期资格。

完整读取周期是同足已知连续岛内相邻 0→1 onset 的区间。所有原始 onset、完整周期、
截尾部分与去抖变体转换均保存。未资格原始间隔名为 `raw_inter_onset_interval_s`；
只在合格窗填 `median_period_s`，否则为 null。每四个连续完整周期滑窗：

```python
cv = std(period_steps, ddof=0) / mean(period_steps)
deviation = max(abs(period_steps - median(period_steps))) / median(period_steps)
# cv <= 0.15, deviation <= 0.20
# 原序列资格，1/2样本非迭代去抖敏感性必须不变；前3off、末on起3条上下文可见。
phi = (target_onset_step - reference_onset_step) / reference_period_steps
R = abs(mean(exp(2j * pi * phi)))  # 四周期一对一配对、双方周期资格；R >= 0.9
```

记录步号读数的节律，不叫物理落地、支撑、承重、腾空或步态名称。history 不参与；没有传感器
实际更新时间，不能补造更高时间分辨率。缺步和 unknown 切断序列。合格窗重叠时唯一周期及
覆盖边集合去重；读取跨度分母是有效相邻边×dt，不是样本数×dt。无资格部分不叫非周期运动。
周期分布先取每回合的合格唯一周期中位数，再按回合等权；池化覆盖比例单列。

相位严格按左闭右开参考周期分箱，保留每个目标 onset、对应目标 cycle index 和完整资格窗 ID。
近同相跨零抖动可能落成零/两个事件，返回配对不确定；不据此否认双方节律。独立 review 发现
窗左边界目标 onset 判别依赖前一条读数，现将其纳入覆盖，合成反例固定回归。
另修正了初期 `np.bool_` 截尾标志不能 JSON 编码的问题。二者均在最终测试/真实构建前修复。

共享复用现有 `NumericEpisode.reference`、`cohort_binding` 与 `distribution`，但旧 shared 六文件
属于 Push 22 文件冻结，**没有修改**。新 shared 文件会进入 Stair 的 code-hash glob；未完成构建
重试 key 因代码闭包变化而变化，既有 sealed replay 保留。Push 漏共享 distribution 的 cache
依赖问题本轮没有触发：旧 shared 内容不变，因此不扩修缓存框架。

## 原数组与真实例

只用既有 [P2a source_lock](../P2a/source_lock.json) 的 K1/K2/K3，SHA
`18f585ac6c10e87a0cd8aa61299ab06adac636fc4494fd3e1fb2fdd651265dcf`。
三个来源各 864 回合；这是已知、有关联 lineage 的开发材料，不能宣称新 holdout/E2 泛化。
新 oracle `audit_raw.py` + `tests/unit/cycle_oracle.py` 不导入生产 cycle/adapter/renderer；
直接验证原 manifest/87 NPZ SHA、valid/step/seed，并用标量范数、独立 run 分组、总体方差、
集合覆盖和 sin/cos 公式复算 25,920 条测量记录。生产与审计不调用同一数值函数。

| 来源 | FL / FR / RL / RR 有合格局部窗的回合数（各分母864） | 相位合格窗（六对之和；重叠窗不是独立样本） |
|---|---|---:|
| K1 | 726 / 738 / 526 / 189 | 7092 |
| K2（current v1，history未采） | 723 / 764 / 749 / 759 | 14141 |
| K3/R4 | 640 / 638 / 225 / 309 | 4501 |

没有把成功率用于阈值选择，也未按终局筛选窗口。报告仅做全已记录回合的描述性汇总。
候选/资格窗口、完整/唯一合格周期、有效/unknown rows、有效边、覆盖边及拒绝原因全部单列。
phase summary 只汇总资格数；每个局部相位数值在 `cycles.jsonl`，不合并成全局相位均值。

[REAL_EXAMPLES.json](REAL_EXAMPLES.json)收录完整 raw current XYZ、step、dt、manifest/chunk/row/env
引用和未舍入结果；其余各类首个见证在 `outputs/p2c1/real_01/{k1,k2,k3}/raw_examples.json`。

- K1 r1.e0 FL：onset `[25,50,79,107,135]`；间隔 `[25,29,28,28]` 步，CV
  `0.05454545454545454`、最大偏差 `0.10714285714285714`，两去抖变体不变，局部周期中位数 `0.56 s`。
- K2 r1.e0 FL：onset `[22,48,76,103,130]`，CV `0.026189140043946207`，中位数 `0.54 s`。
  缺 history 没有使 current 周期失去能力，也未伪造 history 槽时间。
- K3 r1.e0 FL：onset `[8,28,50,77,106]`，CV `0.1485736712098065`、最大偏差
  `0.1836734693877551`，中位数 `0.49 s`。贴近固定变异门但仍合格，没有为此调整门槛。
- K1 r1.e0 RL 反例：onset `[42,70,98,125,152]`，CV 和最大偏差均 `0.01818181818181818`；
  step42–43 仅两条 on，≤2 样本变体会删除这次原始触发，因此 `debounce_sensitive`、周期主值 null。
  不能仅凭间隔整齐就称合格节律。
- K3 r1.e0 FL→FR 相位：参考 `[28,50,77,106,134]`，目标 `[28,50,78,106]`，
  相位 `[0,0,1/29,0]`，R=`0.9956067036693166`，圆周均值=`0.008595323139881974`。
  原例保留目标自身资格窗及所有依赖读数。这是读取序列相位，不能叫真实落地同步。

K2 未找到预声明“完全没有候选窗”的回合/足例；记为空类，未调整规则凑例。单周期不足、缺步、
invalid、截尾、去抖1/2样本及同占比不同次序等边界另由合成 known truth 覆盖。

## 报告与预算

新离线报告含完整来源和 cohort，候选英文句全部 optional；普通 public.v5 不直接接收 v3。
可选句预览保留原 base 内容，再尝试在**合计8192字节**内装入整句；若有新句，envelope 自带
`stair.cycle_offline_preview.v1` 和离线状态。没有句子能容纳时输出原 base 的规范化内容。
实际 `preview.json` 使用紧凑排序 UTF-8、无结尾换行，磁盘字节与 receipt 数字一致；
保留的是旧规范化内容/claim，不声称 pretty JSON 原文件格式字节相同。

| 来源 | base / 实际preview bytes | 新句纳入 / 候选 | 原必保槽 |
|---|---:|---:|---:|
| K1 | 8086 / 8086 | 0 / 4 | 18 |
| K2 | 8032 / 8032 | 0 / 4 | 18 |
| K3 | 8127 / 8127 | 0 / 4 | 18 |

这是真实的预算省略，不称新周期事实已经进入下一作者请求。完整新报告/原数组证据都已交付；
今后生产投递需独立选择旧 optional 与新 optional 的优先级，仍不能挤掉必保槽。
Push 12288B、20必保槽保持原代码。合成中文预览额外验证恰8192通过、8193整句省略及来源互换拒绝。

## 实际检查与成本

```powershell
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c1/run_validation.py --mode tests --output outputs/p2c1/tests_01
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c1/run_validation.py --mode real --output outputs/p2c1/real_01
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c1/verify_compatibility.py
```

复跑须用全新输出目录；最后一条是只读检查。stdout/stderr/真实命令/退出码/峰值内存及 SHA
由既有 P2b1 CPU wrapper 捕获，完整索引在 [verification.json](verification.json)。原输出未改。
关键生产 diff 原文为 `outputs/p2c1/key_diff.patch`（26755B），SHA256
`38798ec15d9f873f5538794a127e4dbf010cdf325e987023e6a2022624929174`，也可审阅本阶段提交。

```text
cycles exit 0 seconds 5.695        Ran 15 tests in 5.068s — OK
compatibility exit 0 seconds 51.205 Ran 139 tests in 49.969s — OK
build_k1 exit 0 seconds 11.310     oracle_k1 exit 0 seconds 8.031
build_k2 exit 0 seconds 10.919     oracle_k2 exit 0 seconds 8.033
build_k3 exit 0 seconds 12.087     oracle_k3 exit 0 seconds 8.168
old_artifacts exit 0 seconds 3.394
```

本阶段154项测试，0 FAIL/ERROR/SKIP。另有3个原数组oracle PASS、3个旧 full/public 实际重放
一致、P2a 33文件/P1b三文件 SHA 一致、Push 22冻结文件一致。没有把历史247/309重新称作本次新执行。
运行 `ruff --version` 时当前 venv 无该模块，未安装；不是测试失败或产品阻塞。

三源构建内计墙钟约10.844/10.485/11.652s、CPU10.453/10.188/11.484s；实际 Python峰值
297,869,312 / 275,374,080 / 317,857,792 B。全部新证据约222MB，未复制GB级旧数值账。
回归进程峰值439,775,232B。实际API/功能LLM/训练/GPU/服务器写入均0。

## 勘误和剩余阶段

本次用户明确 P2b2 独立 CPU 接线及 root 亲核已 PASS，已更新阶段门和P2b2说明；没有代替K5审查。
P2b2 `verification.json.unresolved_or_skipped_tests` 的5个键重复尾部方法名；本轮新索引保留
旧键→正确ID的映射，并核原日志及AST测试方法存在，**不改原 verification 或历史日志**。
唯一ERROR仍是缺omni的原测试，4个SKIP仍是缺目标Isaac source closure；这不是本轮154项中的失败。

- **P2c2输入门**：Go2 profile `configs/robots/unitree_go2_12dof_isaaclab_v1.json` 的USD root SHA
  `ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d`（8142B）与引用mesh SHA
  `2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb51`（18802187B）
  对应用户给的 server cache 文件。本轮只核源码链，不访问SSH。
  后续CPU parser需给每个RevoluteJoint的prim/body0/body1/localRot0/1/axis/limits、角度单位及零点。
  runtime `find_joints(...,preserve_order=True)` 绑定名字，recorder原样取 `joint_pos[:,resolved.joint_ids]`；
  默认站姿不是q零点。archive没有效限位，因此只允许名义裕量代理；随机化/有效限位需补采。
- **P2c3输入门**：C0/C1/C2分别核 `isaaclab_terrain.py`、`isaaclab_irregular_terrain.py`、
  `isaaclab_rough_terrain.py` 的网格；origin为`[start_platform/2,size_y/2,0]`。
  独立oracle需从vertices/faces算三角面，验证平台、阶沿、粗糙对角线、world平移和四侧域外。
  现有query的ceil/bisect/clamp不能自己证明几何真值；未资格类型unknown。
  输出仅参考点到声明表面的高度差，不叫脚底/碰撞距离，不以supporting_surface修正冒充几何。
- 历史link位置/COM速度参考映射仍按原通道措辞；本轮不新造simulation。父代比较与
  action/reward profile、真实clean6.1sol/DeepSeek、训练闭环均另阶段。

AGENTS检查未发现适用于project的文件；读了可用 `ponytail` 技能并采用增量离线方案，无新增依赖。
Windows sandbox的长路径假D在正常权限下核为干净，未恢复/删除任何历史文件。唯一repo writer
为本线程，两个辅助代理仅只读review。最终以本阶段提交为独立验收对象，不能将开发PASS当root放行。
