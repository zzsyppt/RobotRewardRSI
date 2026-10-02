# P0 测量目录冻结 v2

测量语义版本 `trajectory.p0.measurements.v2`：父线程复验后统一M04为水平速度、M05为角色索引低位编码、M02为归一化四元数；替代P0 v1对应声明及错误参考事实，不修改任何生产schema/ABI。旧提交及其账hash可追溯，新旧事实不得混用。

本目录是 P1 的实现输入，不是已交付的卡片。依据主计划 v1.1 §5–6、§8、§13 及相邻 readiness。适用范围为 `source_lock.json` 中的公开 Stair Go2 档案；R4 为首个全队列，旧档仅作两个已知反例，绝不合并频率。所有列出的常数在 P1 输出前冻结；开发规则未经统计校准，不能声称误报率保证。

## 共用定义

- 单位：位置 m、时间 s、速度 m/s、力 N、角 rad。`k=1..N` 是本回合控制步，`t[k]=k*dt`，本档 `dt=0.02`。每步是 post_step/pre_reset 的观测；没有 t=0 状态。通用实现必须读取 dt，不能硬编码 50 Hz。
- 完整来源为 `(manifest_sha256, run_index, env_index, chunk_sha256, array, sample_index, entity_index)`。`r1.e54` 不跨档案唯一。
- 世界系地形前进方向 +X 仅由 Stair 契约赋予；身体前向 +X、左向 +Y、上向 +Z 来自 hash 匹配的 Go2 body-frame 资产记录。世界前进不等于身体前进。
- `support_pos_w`来自`body_pos_w`位置通道，`support_lin_vel_w`来自`body_lin_vel_w`速度通道。link位置与COM速度的历史资产参考点/质心偏移映射未完全验证；当前只解释所记录通道，不将速度称为该位置的精确导数，也不以差分不一致推断数据或运动异常。
- 原始 `valid[k,env]` 为 first-scored-episode 覆盖。各量还要求依赖组件有限且 shape/实体正确；缺字段为 missing、坏值为 invalid、不可判定谓词为 unknown，不能填零或算阴性。无接触槽级 valid；history 的零不等于该槽确曾观测到零力。
- 力通道的k/t是recorder读取时刻；current与history的实际sensor更新时间/延迟均未逐样本保存。跨根部/力通道可说“同条记录中”，不能将capture对齐升级为已证明的物理同时性。current也不补造sensor timestamp。
- 连续的 **记录序列** 可以计算 run；不因此证明采样间的物理持续。不得跨 reset、缺步、失效数据求差分、续接或否定。无插值填补接触。所有原始转换保留，无 debounce 或短段合并。
- 一个 true run 的有效样本为 a..b（含端点），存半开索引 `[a,b+1)`；观测跨度 `(b-a)*dt`，采样格时长代理 `(b-a+1)*dt` 分别命名。仅在两侧有有效 false 样本时为完整 run。开始边界 `(t[a-1],t[a]]`，结束边界 `(t[b],t[b+1]]`；首/末记录已为 true 时分别 left/right-censored，真实边界 unknown，不补 t=0 或终止后记录。
- 默认先每回合计算，再在全部有效回合上等权汇总，随后按结果及8个 stratum 分面。池化样本统计单列为采样格时长加权。本批 dt 相同才可将 count 比视作此代理时长比。每项保存分子、有效分母、unknown、未处理、总数、条件、时间窗及截尾计数；完整 run 时长分布不掺截尾 run。
- 数值输出保留未舍入值，显示才舍入。原始 float32 坐标独立重算容差：位置/速度 1e-5、力 1e-4、比例 1e-6；步骤/计数/实体/截尾必须精确相等。边界在上述一个控制步区间内核对，不能用宽容差掩盖错误。

## 首批十组描述量

每组继承上述时钟、有效性、实体与统计规则。`enabled` 表示 P1 必须实现可测部分；附加量标记 deferred 时不能在首批报告里假装已经具备。

### M01 根部运动、进度与低进度（enabled）

输入 `base_pos_w, terrain_env_origins_w, episode_length`；实体 articulation root。`p_task=p_world-origin`，进度 `x=p_task.x`；净进度 `x[N]-x[1]`，水平折线路径 `sum(norm(diff(p_world.xy)))`；相邻差分速度 `diff(p_world)/diff(t)` 是区间平均代理，非记录的瞬时 root velocity。分别保留 x 的范围、回撤、首次重返原峰值或终点截尾；回撤不能命名尝试次数。

低进度：每个完整 0.5 s 窗 `[k-25,k]`，`max(x[k-25:k+1])-x[k-25] < 0.02 m`。dt 不整除窗口时取覆盖至少 0.5 s 的最短完整窗口，显式报告实际跨度。来源类别 `development_calibration`，具体来源 `behavior.py:32-33,314` 的已有开发常数，**非已标定概率保证**。窗口不足 unknown；确认时刻为窗尾，不能把它写作真实开始。P1 若更改旧窗端点定义须版本化、说明差异。

允许：“记录区间净前进…m，水平路径…m；某窗内最大新增进度不足2cm”。禁止：“完全静止/拒绝前进/卡死”。反例：原地往返路径大但净进度小；旧开发成功回合 r1.e60 回退后再返回。

### M02 根部姿态与上轴（enabled）

输入 `base_quat_w`，wxyz。验证范数误差≤1e-4 后归一化计算；超限 invalid。`up_z=1-2(qx²+qy²)`；`roll=atan2(2(qw*qx+qy*qz),1-2(qx²+qy²))`，`pitch=asin(clip(2(qw*qy-qz*qx),-1,1))`。单位 rad；up_z 无量纲。保存范围和原始短暂负值，`up_z<0` 的阈值0是几何定义，类别 `robot_scale_rule`。姿态任务阈值 `abs(roll)>0.65 OR abs(pitch)>0.65`、10步，仅在复述任务信号时使用，来源 `task_spec: manifest.metadata.semantics.episode_outcomes.signals`；不覆盖 evaluator 的终局。

唯一规范为归一化后的q：独立事实的主值、负值区间及M08旋转均采用此口径。`raw_body_up_diagnostic`和`raw_vs_normalized_up_z_max_abs_delta`仅记录未归一化诊断；raw负值不得代替主谓词，尤其不能用数值容差放过零附近的符号变化。原始wxyz操作数不重写。

允许：“身体上轴的世界竖直分量为负/姿态超任务阈值的记录”。不从 Euler 单轴直接命名头部运动；不用翻转自动解释原因。必要反例：`legacy_dev.r1.e54` 先翻转，再出现前足低速未触发，不能写成直立举足。

### M03 每足接触阈值标记（enabled）

输入 **current** `support_contact_forces_w[k,env,foot,xyz]`；FL、FR、RL、RR 顺序由 runtime projection 绑定。`f_i=norm(F_i)`，`c_i=(f_i>1.0 N)`，严格大于。阈值来源 `task_spec`，路径 `metadata.semantics.episode_outcomes.signals.slip.contact_force_threshold`，不是摩擦/承重阈值。占比 `sum(c_i*w)/sum(w)`；恒 dt 下 w=dt。有效力分量不全则该足 unknown。

允许：“已记录净法向接触力超过1N的标记占…”。固定术语定义后简称“接触标记”；禁止“承重占比/完整物理接触时长”。零力是有效 current 观测时可计 false；missing/invalid 不可计 false。history 最大值不能悄悄替代 current。

### M04 单足未触发区间、低世界速度组合（enabled）

输入 M03 及 `support_lin_vel_w`。`off_i=NOT c_i` 的 run 按共用边界输出；完整 run 和左右截尾分开列数量、跨度和采样格时长。`v_xy=norm(v_world.xy)`；`off_i AND v_xy<0.05 m/s` 仅表示标记未触发且所记录速度通道的水平世界速度低，XYZ速度仅作额外诊断，不参与谓词。0.05 来源 `development_calibration: behavior.py:35`，未经误报标定。长 run 的显示起点0.5s也来自旧开发规则；全部短 run仍保留。2s与5s仅是必要事实的预先查询窗，不改变 detector、不排除更短反证。

辨别例：`r4.r1.e60` FL的XY最长区间214..637，424样本、格时长8.48s、观测跨度8.46s且右截尾；XYZ口径错误地得到475..625、151样本、非右截尾。step626所记录速度为`[0.0415688157,0.0166678950,-0.0320286751] m/s`，XY小于0.05而XYZ大于0.05。旧e54两种口径偶然相同，不能单凭它验收。

允许：“该足在连续有效记录中标记未触发…（观测跨度/格时长代理），区间末仍未触发”。禁止“腾空/一直无物理接触/举腿/未承重”。反例：轻触不足1N、reset零history、首帧已off、终点off、1步重触发、世界运动但相对躯干姿态维持。

### M05 多足接触组合（enabled）

输入同时间的各足 c；任何足 unknown 则完整组合 unknown。按绑定实体顺序保存布尔向量，唯一编码 `code=sum(int(marker[i])<<i)`，实体`[FL,FR,RL,RR]`、权重`[1,2,4,8]`，版本`role_index_lsb.v1`。解码`marker[i]=bool((code>>i)&1)`；不以四足常量充当其他形态绑定。每组合 counts/有效联合样本数及原始转换。两足状态一致率 `mean(c_i==c_j)` 另列，包含共同 false。

本次修正目录v1反序声明，保留原参考账代码及生产ABI。真实`r4.r1.e35` step14为`[true,true,false,true]`，code=11（不是13）；该回合counts[11]=135。16种组合须双向核验。

允许：“FL/FR/RR触发、RL未触发的记录比例”。禁止以全false推全身离地，以布尔一致率推落地同步或齐跳。正/反例固定为相同占比不同转换次序、全部足false但base受力。

### M06 标记转换时差与受限周期相位（转换 enabled；周期命名 deferred）

输入相邻有效 c。0→1 的转换区间 `(t[k-1],t[k]]` 表示buffer读取序列的变化区间；实际sensor更新延迟未知，不能将20ms读取边界当物理接触事件的边界或真实落地时刻。两足事件在同一控制步仅“同一步出现转换”；不能在区间重叠时强判先后。若比相邻事件，保留所有事件及配对规则：每个参考足事件取最近目标足事件，等距取较早；同时报告未配对范围，禁止为了得到同步删除短重触发。

周期相位暂不启用：R4存在短重触发，历史探针丢弃0.1s内转换。P0未验证可靠周期判据或事件配对的语义，故不沿用0.97/0.08/0.79为正式事实。P1先交付原始转换/时差；周期阶段须先冻结最少周期数、稳定性、去抖敏感性与完整周期分母并单独复核。本项调整依据计划6.5的“周期足够且稳定才说节律”，不阻塞其余目录。

### M07 净法向力模、竖直分量及采样积分代理（enabled）

输入 current support/base force；单位 N、世界系。`f_i=norm(F_i)` 与 `F_iz` 分别保存，后者可能带符号，不剪裁成承重。模份额 `S_i=sum_k(w_k*f_ik)/sum_jk(w_k*f_jk)`，分母0则 unavailable。总体默认每回合S_i等权；池化S另列明确限定。世界竖直分量求和/份额另名，分母为0或近0不报告份额（绝对阈值1e-12 N仅数值零保护）。

相邻有效点的梯形采样积分 `sum((F[k]+F[k+1])/2*(t[k+1]-t[k]))`，单位 N*s，范围 t1..tN；是插值假设下的 **采样积分代理**，不是已观测物理冲量。固定采样矩形格权重的模份额不混同梯形积分份额。history 不能添造额外dt积分。

允许：“所记录净法向力模的份额/世界竖直分量均值”。禁止“重量份额/全部外力/摩擦力/身体某部位承担残差”。实际质量、全身动量及切向力缺失，动量对账 unavailable；151N旧参考不是体重。

### M08 足参考点相对根部的位置与运动（enabled）

输入 base p/q、support p/v；`r_i=R(q).T*(p_i-p_base)`，使用完整旋转，不以只去yaw冒充body frame。r 的 XYZ、相邻变化和范围；世界速度与相对根部变化分开列。无默认“高位异常”阈值；只报告相对位置变化/范围。默认站姿参照未冻结时 unavailable，不能将旧探针0.18m直接当已核资产姿态。

q按M02验证并归一化；位置通道与COM速度通道参考点映射未完整核实，禁止称所记录速度是足位置或r的精确导数。

允许：“足刚体参考点在根部后方/相对根部升高…m”。禁止“脚底位置/拖地/主动抬腿/保持整条腿姿态”。近似静态r且世界v大是关键反例；身体翻转时body-z的意义必须与M02联合给出。

### M09 关节姿态范围（enabled；限位邻近 deferred）

输入 `joint_pos`，顺序按 manifest joint_names，单位 rad，输出每关节 min/max/变化幅度。RL calf 索引10。没有默认异常阈值。关节组按角色，不用数组最后三列猜后腿。

真实有效硬/软限位未记录。名义URDF −0.8378 rad 未在本冻结中建立与实际USD关节坐标/轴的一致证据，因此“距名义限位0.02rad”的旧探针结论暂不启用。后续取得hash匹配资产限位后，公式 `min(q-q_min,q_max-q)` 并标名义代理；它依旧不等于卡死/力矩饱和。此降级是输入依赖不满足，不是把阈值设零。

### M10 地形参考面、任务位置与终点（任务位置 enabled；几何间隙待独立验证）

输入 terrain origin/type、suite public cell mapping、base/support p；任务spec给 riser/tread/起始平台、goal/deadline。进度/终止离目标余量 `goal-x_terminal` 和离deadline余量 `horizon-t_terminal`；禁止将未到goal直接解释机制。`outcomes.primary_reason` 的 authority 始终为 archived task evaluator；translator不重新分配原因。

投影足参考点高度差 `p_iz-origin_z-h_cell(p_ix-origin_x,p_iy-origin_y)`；现有 terrain query 可在 P1 复用，但每个启用几何类型必须独立核对台阶边界、粗糙面和坐标轴后才输出。base clearance为已有“supporting surface”代理，非root处碰撞几何最短距离；绝不把2cm写成脚底贴面或身体碰撞。P0不把旧粗糙面近似当独立真值，不定义新的碰撞阈值。

保存终点、terminal有效样本、base current force及manager `base_contact` 原始mask；mask可能依赖history，不能要求等于 current norm>1。短终止事件必须有独立保留槽。`r4.r1.e129` 为预选短事件例，终点后的无效padding不参与任何分母。

## 奖励、Push及补采依赖

逐步实际奖励、action、qvel、torque、command、sensor air time/first contact、质量、有效限位均 missing。训练稀疏奖励仅在其实际采样范围内解释；Push评估reward=0不作为学习信号。真实奖励与重算结果必须是不同字段；未触发时段超过0.5s的代理不能冒充 `feet_air_time` 分项正负。

后续最小补采按能力解锁：物理上下文（实际质量/限位/随机化/命令）；必要肢体与接触（完整语义及时间/valid）；动作与执行（raw/scale/target/torque）；真实奖励（分项、权重、dt、gate、air-time/first-contact）。P0仅登记，不改采集器。后续短录制实测字节/步、丢帧、对齐、仿真耗时后再定预算，旧档不能因新profile通过而升级能力。

周期相位、限位邻近、地形参考面距离仍是主计划第一版目录的后续义务，不因P0 deferred而永久移除。周期需可靠周期/完整分母验证，限位需资产坐标绑定或补采，几何需逐类型独立核对；几何待办是尚未完成验证，不是声称现有通道缺失。历史link位置/COM速度映射同样须在推出同参考点运动结论前完成。

Push维持not_collected接触边界，首版限根部/扰动/恢复；不创建本轮Push实现。跨策略3–5个独立lineage、强错误率、新形态和闭环收益均未验证。

## P1交付与语言决定

作者侧采用现有链路的英文，内部审阅稿可中文。在固定gaps/定义句中一次定义“contact marker = recorded world-frame net normal force magnitude > 1 N; samples do not establish continuous contact”，其余句可简称marker但数字/范围/分母不得省略。限位、奖励、几何暂不支持项明确unavailable。

P1须新建task-owned public schema/lexicon及与新目录一致的comparison键；旧public.v4与旧四词比较原样可回放，新旧父代明确不可比或从同原档同版本重算，不静默兼容。沿 `TaskPromptAdapter → PublicContextBuilder → build_prompt_payload → serialize_prompt_payload → provider user.content` 检查实际JSON。卡片不是第二份附件输入。

Stair报告8192 UTF-8 bytes、总task sections及authoring预算分别核查。保留槽顺序：结果/范围、总体行为、失败过程、短终点事件、成功与翻转反例、missing/unknown；其它事实省略须列ID和理由。最终逐句核对statement ID、数字、条件、反例、缺口、内部路径与来源权限。serializer仅检查标记，不替代public_dev来源准入。
