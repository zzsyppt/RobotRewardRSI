# A 阶段独立行为预期（先于新卡）

2026-10-03。本说明在未读取任何 A 阶段新卡之前建立。依据原始 NPZ、manifest 中任务结局/实体/时钟契约和审计问题；没有调用项目测量、聚合、渲染代码。原审计 ground_truth 是待核草稿。R4/K1/K2/m50/m300 现在全部是开发数据，不再称 unseen；结果不能证明泛化或作者收益。

## 原档及复算物

本地根目录 `D:/humanoid_agent/true_RL_Agent/`：
- R4：`outputs/trajectory_p0/raw/numeric_evidence/manifest.json`，manifest SHA 开头 `8c5f2d68`。
- K1：`outputs/trajectory_p1d/raw/k1_numeric_evidence/manifest.json`，`0389a10d`。
- K2：`outputs/p2a/raw/k2/manifest.json`，`fe488cb8`。
- m50/m300：只读 SSH humanoid-server 下 `/data/yuyangliu/isolated/audit_translator_20261003_claude/{m50,m300}/numeric_evidence/manifest.json`，分别 `0a80fb93`、`09ee8662`；随后已原样只读下载到 `C:/Users/lover/AppData/Local/Temp/behavior_expectations_20261003/raw/{m50,m300}/numeric_evidence/`；目录关系及原manifest内容保留。

同目录 `raw_probe.py` 为独立 NumPy 脚本；上述 Temp 目录中的 `local_results.json`、`remote_results.json` 含每回合原数组派生值和统计。远端只以 stdin 执行 Python CPU 运算，stdout 保存本地 Temp，无远端文件写入、GPU、模型或采集。上述 5 档全部有效记录均有限、步号和 episode_length 连续；每档 864 回合、dt=0.02s。

记录索引由 manifest `chunks[].metadata.run_index/step_start/steps` 定位，NPZ 第一维为块内步，第二维 env（0-based）。实体顺序来自 `metadata.runtime_projection`：足 FL/FR/RL/RR；小腿关节列8/9/10/11。`valid=false`不参与统计。结局由 `metadata.seeds[].outcomes[env].primary_reason`给出，不由运动学重分类。

## 本说明统一定义

净前进位移=`base_pos_w[-1,0]-base_pos_w[0,0]`；末0.5s signed vx 为最后点减前25步的位置再除0.5s，保留方向。不能叫速度通道读数或物理结束原因。全程上轴朝下指归一化 wxyz 四元数的 `1-2(x*x+y*y)<0`；未出现该条件不等于完全直立。

每足接触占比：`norm(support_contact_forces_w)>任务契约1N`，下文默认所有有效记录池化；它不是接触面积、承载比例或三足步态标签。相对足位置为 `R(q).T @ (support_pos_w-base_pos_w)`，明确使用机身坐标；世界差分和相对差分分开。2s后 span 指从 step100 到结束逐轴 max-min 后的三维模长，再取四足/十二关节中最大值；不是累计行程。分位数是每回合等权分位数。

低正向进度只在完整0.5s历史可用的窗口评价，公式为过去窗口最大x减窗口首x<0.02m；起始25点不适用。下文各活动比例分母均为这些完整窗口，与旧三值短路统计分母可能不同。足位置差分仅是记录参考点运动代理。

名义关节对照仅使用已有 R4/K1/K2 资格与 `go2_nominal_usd.json` 中 calf[-155.9992218,-48.0001068]度，下/上边带各为名义范围5%。边带数只吸收各名义边界转换为float32的实际量化误差：下界6.80289598e-8rad、上界6.84396628e-9rad；不使用任意1e-6rad容差，不能据此吞掉0.0856rad真实差。这里不证明环境有效 runtime 限位。`--nominal`仅对本地三个已资格档开启；m50/m300结果不附名义统计。

## R4 必须覆盖的 5 点

1. 总体向前移动，但成功/失败过程不同：682成功、182失败；全队列净x中位4.2545m，成功4.3029m，失败1.8896m。可等价概括，不能只有结束账目。
2. 全程56回合曾上轴朝下，其中40个 forbidden_contact、16个姿态标签；27/56在 c1_conditional_frontier。成功682个均未出现上轴朝下。原数组锚：run1/env50，step276首次为负，最后step588。
3. forbidden_contact 的125个回合中，有68个全程未上轴朝下且末0.5s仍向前>0.2m/s；该子组终止时刻中位4.74s、末窗vx中位0.5661m/s，其中33个在 height_ood。与上一条不同，不应合称“全部触台前已翻倒”。锚：run1/env54，末step132，末窗vx0.676088m/s、up_min0.864091。
4. 39个超时中35个末0.5s |vx|<0.01m/s；超时组末窗vx中位-4.77e-7m/s。反例：另4个超时仍>0.2m/s，不能把所有超时叫静止。
5. 四足接触占比70.32/71.73/42.85/59.35%；左后参与明显较低，组合FL+FR+RR占35.98%。左后calf上边带池化58.86%，下边带0.0164%；另18.31%超出名义上下界及其实际float32量化误差。应描述全队列侧别/占用，不让单点极值代表全部，不将全部outside称浮点噪声。

## K1 必须覆盖的 5 点

1. 785成功、79失败；净x中位4.2867m，末0.5s vx中位0.8118m/s。不能与m50的几乎少动超时文本雷同。
2. 左后足接触池化19.44%，其余FL/FR/RR为71.74/69.07/69.82%；组合FL+FR+RR占54.12%。若采用回合均值，左后为19.00%而不是19.44%，必须明确权重；审计“18%”不作强制事实。
3. 左后足机身相对z的“回合中位数再取队列中位数”为-0.12231m，另三足为FL -0.20250、FR -0.20687、RR -0.25355m；结合其低接触参与可说相对位置更高，不能直接认定异常三足步态。左后calf名义**下**边带占47.43%，上边带0.252%，有持续侧别差异。
4. 24回合曾上轴朝下，其中23在 tread_ood；23个结局为 forbidden_contact、1个timeout。锚：run1/env278，step179首次上轴为负，step214结束（4.28s），末窗vx=-0.086106m/s。不能全局Euler unwrap后宣称真实转过几百度。
5. forbidden_contact 72个中35个全程未上轴朝下且末0.5s vx>0.2；29/35在 c1_conditional_frontier，中位结束5.76s、末vx0.59037m/s。成功785个无上轴朝下，末vx中位0.81349m/s。对照是这些队列的描述，不是失败因果。

## K2 必须覆盖的 4 点

1. 822成功、42失败；全队列净x中位4.2877m。成功组末0.5s vx中位0.80043m/s。
2. 较低参与足改为右后：FL/FR/RL/RR池化65.70/67.95/63.81/48.38%，不能复用K1“左后低参与”主句。
3. 主接触组合为四足全触发26.63%及FL+FR+RL触发26.47%；0/11固定组合不能占据核心描述。
4. 34个 forbidden_contact 中25个全程未上轴朝下且末vx>0.2m/s，20/25在 c2_mechanism_ood，中位末vx0.48827m/s、结束5.50s；全队列只有3个曾上轴朝下。原审计“21个c2带速”来自不同混合分类，不作为此明确定义的真值。锚：run1/env109，step197终止，末vx0.44904m/s、up_min0.93761。

## m50 必须覆盖的 5 点

1. 864/864超时，净x位移中位0.129589m；末0.5s vx中位0，全部|vx|<0.01m/s。没有成功对照时仍应给出此绝对过程。
2. 大部分回合2s后参考点/关节几乎不变：最大world足span中位9.90e-6m，body-relative足span中位2.37e-5m，最大关节span中位7.55e-5rad。world span<1mm为859/864、relative<1mm为856/864；反例最大world span9.82mm，不能断言每回合精确冻结。
3. 同期记录足XY速度每回合中位数的队列中位为0.06046m/s，而world位置差分XY中位4.77e-5m/s、relative位置差分XYZ中位1.04e-4m/s。这证明两通道不等价；不能由速度通道说“93%持续迈步”，不猜求解器原因。
4. 接触标记四足全触发占98.672%；四足各自占98.675/98.673/99.008/98.968%。全程无上轴朝下；末root z相对terrain origin中位0.159689m。这个z未独立重建局部表面，不能写sole clearance或身体离当前面高度。
5. 独立反例锚：run1/env0 step100..600（2..12s），所有足world span最大1.91874e-5m；FL速度通道中位0.077224m/s，FL接触力模中位41.0218N；root只变化约2.46e-5m。完整原数值在remote_results.rows[0].witness_after2s。

## m300 必须覆盖的 5 点

1. 666超时与198 forbidden_contact 必须分开，不能一条全队列中位数抹掉早期前冲子组。
2. 超时组末0.5s vx中位0，最大0.00029755m/s；终点相对origin的x中位0.406272m，结束中位12.0s；支持末段几乎停住。
3. 超时组2s后world足span中位0.5128mm，body-relative中位1.3334mm、关节最大span中位0.006563rad；仅622/666的world span<1cm，不能把666个都写成2s后全程静止。
4. 接触终止组结束中位1.46s，末0.5s signed vx中位0.501144m/s；196/198末vx>0.2，其余2个也向前（最小0.15656）。71个在c1_id、77个在phase_composition_ood，共148/198。锚：run1/env0 step65结束（1.3s），末vx0.476665m/s，上轴最小0.91612。
5. 所有864回合均未上轴朝下。完整窗口中低进度且记录速度任足>=.05占87.446%，改为world位置差分仅1.166%、body-relative XYZ差分2.306%；这些是记录点运动代理的差异，不能叫步态真值。

## 少量反例验收（不是分类平台）

- 静止位置但记录足速度0.08：应描述位置少变；不能引用速度通道判迈步。
- 整体刚性平移0.2m/s、足在机身系固定：world足差分非零、body-relative为零，作者文字要区分整体漂移与腿部相对变化。
- 固定root且腿相对位置周期变化，与全体位置固定应能区分；只检验可观察描述，不命名任意物理原因。
- 单足相对z增加并保持0.5m：哪一足/相对高位必须进入最终文字，不能和同样净进度的对照完全相同。保持正常姿态与上轴朝下亦需区分。

最终验收不要求每条数字逐字齐全或固定句数；要核对上面的关键过程和跨队列差异已进入最终作者文本、数字/实体/单位可独立重算。一个数字或实体/列映射破坏被独立预期抓住即可；不再扩全字段mutation矩阵。

未证明：真实步态类型、接触原因、运行时有效关节硬限位、动作选择因果、实际sole clearance，以及新卡对作者收益。没有读取不相关K5数据、没有改任何仓库文件。

边界反例：K1右后calf全队列最小signed margin=-0.0855956758rad，K2左后=-0.0957334722rad，均远超上述实际float32量化误差；边带与outside必须分别报告，不能给所有outside同一物理解读。

本地只读副本已验证：m50 30块，NPZ 157554654 bytes，全部文件158111379 bytes；m300 30块，NPZ 128557826 bytes，全部文件129127319 bytes。两份原manifest SHA和全部60块的大小/SHA均匹配原manifest；可直接给生产链输入上述本地manifest路径。
