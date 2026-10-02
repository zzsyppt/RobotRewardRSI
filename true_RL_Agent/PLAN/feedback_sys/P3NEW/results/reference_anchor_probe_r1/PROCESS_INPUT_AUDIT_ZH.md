# N.5 过程输入只读审查

2026-09-30。范围：本次旧开发 `episode-01/stair-0357` 的实际 HTTP 请求、15 项事实原答、当前材料生成代码和任务定义。只新增本说明；未改源代码、原答、旧分数或失败记录，未调用 API/GPU。原开发调用仍未通过，R2 仍为 INCONCLUSIVE。

**结论：当前输入保留了过程所需的许多原量，却把可重建的直接量标成笼统的 `not_supplied`，并让模型逐列完成大量算术。F7/F8 把“直接列未投递”退成不可判断，F6 在直接姿态列已经投递的情况下仍未建立区间。问题不止四处引用错误；但一次原答不能证明删列是唯一原因，也不能证明补列后语义验收会通过。**

## 1. 实际送达与代码

- [实际请求](./received/prepared/request.http_body.json) SHA256：`e47723d634a67a12f69d057f7d95dde6dfc366f2c8edb4823e3d739b77dc9ceb`；[原答](./received/prepared/raw_answer.txt)：`8ff1a4a8ebfcb133aa714f3d1d4e951967fa9a735a0736681a09515f5b2d62a5`。本审查只读原答事实字段，未读取 reasoning 文件。
- `related-raw` envelope 的 `/data/clock` 为 1–606、stride=1。`raw_columns` 含 `base_pos_w/base_quat_w/support_pos_w/support_lin_vel_w`，合计 18,786 个原始标量；`direct_columns` 含完整 roll/pitch、up_z、各足 surface height、instant force norm、buffer max force norm，以及 forbidden 两类 norm。另有原始离散列、地形原点、身体顺序、单位、时钟和阈值定义。不是只给终局。
- [prepare_confirmation_materials.py](../../prepare_confirmation_materials.py) `related_episode`（94–128 行）在 Stair 保留上述固定列，删除重复直接表示，再以集合差生成 `not_supplied_direct_fields`。当前函数与本次 `executed_source/prepare_confirmation_materials.py` 的同名函数逐字相同；两个完整文件不同，不能因此宣称整个执行源码等同当前源码。
- `/data/scope_limit` 已说明重复 pose/geometry 曲线是 omitted 而非 missing measurement；但它没有逐量给出可用性与重建关系。`source_direct_fields` 是 dtype/shape/hash 元数据，不是列值。引用块呈现的修复不能自动修复这种含义混淆。

## 2. 三类情况必须分开

| 对象 | 实际证据与原答 | 审查结论 |
|---|---|---|
| F6 姿态区间 | roll/pitch 与 up_z 直接列已送达；原答明确写没有重算连续 10 步区间 | **已供应、未分析完成**，不是观测未知。措辞只指终局附近还遗漏了更早的多段变化。 |
| F7 平面速度/带载点运动 | `support_lin_vel_w` 完整 xyz 与 `support_force_norm_n` 已给；速度直接列列入 not_supplied | **直接表示省略、完整可重建**：每脚每步 `sqrt(vx²+vy²)`，与 instant norm>1 N 联合。向量力方向和历史槽顺序不是该阈值查询的必要操作数。 |
| F8 参考点净空 | foot world-z、terrain origin-z、surface-height 列均已给；原答称阈值不能独立重算 | **直接表示省略、完整可重建**：`foot_z - origin_z - surface_height`。F8 的不可计算理由不成立。它仍只描述任务几何下的参考点净空，不是脚趾皮肤、真实接触面或碰撞真值。 |
| 其他八项 omitted direct 的成员 | root/support terrain xyz、signed-vz、3D speed、drawdown、0.5 s window gain 均有原操作数与相应定义 | 本包八项 `not_supplied_direct_fields` 全部可重建；不同于未发送且无法从当前包恢复的关节角和完整力向量。不能把八项列表本身当作八个观测缺口。 |
| 真实不可恢复范围 | 本包无 joint_pos、力方向/各历史槽向量、媒体或其他回合；历史槽实际时间戳/有效位未记录 | 分别保持 **recorded but not supplied**、**not recorded** 与 **outside this packet scope**。norm 不能反推完整向量；无媒体不能生成姿态形态或接触视觉结论。单回合不能答组级频率/成功反例。 |

八个被省略直接字段逐项如下。`t` 是 1-based control step，`j` 顺序为 FL/FR/RL/RR，`o` 是 `/data/static_values/terrain_env_origins_w`，`p/v` 分别取 `/data/raw_columns` 的位置/速度。原始 float32 必须按声明 dtype 恢复后再计算。下述均仅有控制步采样含义，不承诺连续物理轨迹。

| 被省略字段 | 已供应操作数与精确定义 | 适用范围/限制 |
|---|---|---|
| `root_xyz_terrain_m` | `base_pos_w[t,xyz] - o[xyz]` | 1–606，world轴方向的terrain-origin相对坐标；不是机身坐标或从初始位置的位移。 |
| `support_xyz_terrain_m` | `support_pos_w[t,j,xyz] - o[xyz]` | 1–606、全部四脚；绑定刚体参考点，不是 toe/sole surface。 |
| `support_planar_speed_mps` | `sqrt(v[t,j,x]^2 + v[t,j,y]^2)`，`v=support_lin_vel_w` | 1–606、全部四脚；world-xy参考点速度。带载条件另取 `/data/direct_columns/support_force_norm_n > 1`，不能改成历史最大norm。 |
| `support_signed_vz_mps` | `support_lin_vel_w[t,j,z]` | 1–606、全部四脚；world-z有符号速度，不等同沿局部surface法向速度。 |
| `support_speed3d_mps` | `sqrt(vx^2+vy^2+vz^2)` | 1–606、全部四脚；3D参考点速率，与planar speed不可互换。 |
| `support_reference_clearance_m` | `support_pos_w[t,j,z] - o[z] - h[t,j]`，`h=/data/direct_columns/support_surface_height_terrain_m` | 1–606、全部四脚；已发送task-owned解析几何高度，因此重建无需再推导几何。低净空联合条件另取 instant norm≤1 与 clearance<0.018；不表示接触对、足端皮肤或真实碰撞。 |
| `root_x_drawdown_m` | `max_{u≤t}(x[u]) - x[t]`，`x=base_pos_w[:,x]-o[x]`；原点相消也可直接world-x | 1–606；每步相对先前峰值的回撤，非净位移/单调下降。最大见证按最早最大回撤终点、再最早前缀峰值；值为0时首步两次。 |
| `root_x_window_max_minus_start_m` | `w=ceil(0.5/0.02)=25`；`max(x[t-w:t] inclusive)-x[t-w]` | 步1–25为undefined/null；26–606有完整26样本闭窗口。不是末值减首值；query<0.02描述该窗口内最高前进量，不能再加一个25步驻留或叫全身静止。 |

算术出处：[export_confirmation_raw.py](../../export_confirmation_raw.py) 159–208、351–369 行明确生成上述直接量/条件与定义；生产任务 [transition_decoder.py](../../../../../src/true_rl_agent/tasks/stair_climbing/transition_decoder.py) 559–579 行也以 instant norm、xy speed、terrain-relative z 减 surface height 构造操作数。任务 [behavior.py](../../../../../src/true_rl_agent/tasks/stair_climbing/behavior.py) 186–203 行区分 any-support OR 与同一脚持续；[process_report.py](../../../../../src/true_rl_agent/tasks/stair_climbing/process_report.py) 189–228 行规定语义及不可扩大解释的边界。不能用 `slip_any` 连续替代“同一只脚连续打滑”。

## 3. 用实际已发送列核验可回答性

本次在 CPU 内存中独立解析 HTTP envelope：按 C-order/身体顺序展开所有 606 点，float32 原列先回转到 float32 再转 double，float64 直接列按原值解码；对 surface-height 的闭区间无损展开。直接按公式逐步比较并合并连续区间，没有调用生产 reducer、读取生产事件答案或复用模型原答生成预期。以下只是少量可回答性见证，**不是替评委制定必要事实清单，不是完整形式责任验收**：

- `max(abs(roll),abs(pitch)) > 0.65` 的完整区间为 120–126、295–302、357–403、409–434、448–488、543–572、580–606；前两段不足 10 步，后五段满足，首个 10 步确认点为 366。`up_z < 0` 为 596–606。因此“只有终局旗标、区间不能说明”不是本包数据能力的上限。
- 全四脚参考点净空的最小值为 **0.020960763096809387 m**，大于 0.018 m，故该参考点净空阈值全程为 false；仍不推出无物理碰撞或真实足端接触完全正常。
- 带载点 xy 速度阈值可定位，例如 RL 在 37–44、FL 和 FR 各自在 117–122 满足逐步联合条件。它们只证明已定义的测量条件，不证明接触面的相对滑动，更不证明它导致最终失败。
- 从完整 root world-x 得到最大先前峰值到后来值的回撤为 **0.5429516434669495 m**，539→606；两个 world-x 为 0.20792627334594727 和 -0.3350253701210022 m。这不表示 539–606 单调回退。

原答 15 项中大量项目说明终局、合同、范围和缺口；F6–F8 没有完成这些可算过程区间，未形成主要阶段及其重复/重叠关系。其自报 `complete=true` 不能作为过程语义完成的证据。以上见证也不能反过来给原答补事实或回判召回率。

## 4. facts 输入的另一个源头问题

实际 `/data/episode/original_terminal_record/diagnostic_flags` 含 `base_orientation_instability_detected`、`slip_detected`、`goal_not_reached_detected`。原答 F4 原样依赖旗标，F6/F7 再以旗标替代过程定位，F8 又依据没有旗标来描述阴性。`related_episode` 只剔除 `direct_measurements`，保留整个 `original_terminal_record`，因此此路径现存。

[N5_FACT_REVIEW_PROTOCOL_ZH.md](../../N5_FACT_REVIEW_PROTOCOL_ZH.md) 第10行允许原始终局，但禁止 facts 含生产规则阳性/阴性标签。记录容器名为“原始终局”不使其中诊断旗标变成独立测量。后续新版本应按实际语义投递必要终局字段，把这些诊断旗标留在 report 被评内容/原始归档；不能向 facts 再加生产事件区间或必要事实答案来掩盖本例缺口。原包和原答保留，不回改。

源头最窄改动位置是 `prepare_confirmation_materials.related_episode` 返回对象的 `episode` 投递投影（当前114行），而非改写原始终局归档；上游 `export_confirmation_raw.py` 第508行把 `terminal` 原对象存入 `original_terminal_record`。为防其他输入路径继续泄漏，还应核对 original-profile、focus和媒体的 facts 投递投影：共享层可以检查“facts禁止生产诊断对象”的结构边界，但任务特有允许终局字段应由 task-owned 定义负责。不能全局删除任何含 flag/diagnostic 字样的真实原始测量，或在公用层硬写两任务的行为词汇。

## 5. 最小改进顺序

1. **先修可用性表达。** 保留原始 omitted 列清单作为传输事实，另用小型字段映射明确 `supplied_direct / reconstructible_from_supplied / recorded_not_supplied / not_recorded`，逐量给公式、输入 locator、单位/坐标/身体顺序与有效时域。`not_analyzed` 是评阅完成情况，不是数据状态；不能以它冒充缺观测。模型依旧可以诚实报告未算完，但包应保持未完成。
2. **让中立数值服务过程阅读。** 优先恢复 full-time planar speed、reference clearance 等最常用且无行为标签的直接量；原始向量/全部时间点同时保留。由 task-owned 定义或独立重算生成并绑定原量、公式、hash，不调用生产过程报告来生成 facts。不要机械恢复所有冗余列；新增列的真实 token/字节容量须实测。仅修状态名仍不能消除 F6 的跨数百行阅读/计算负担。
3. **如仍需要阅读帮助，增加原量曲线视图。** 对全程位置/姿态/足速/力/净空用固定、事前冻结的全程曲线及清楚的身体、单位和步轴；保留全量数据定位，不按生产阳性挑段，不添加“跌倒/恢复/碰撞/原因”等答案性标注。曲线是同一数值的阅读表示，不冒充原始相机媒体，也不能顶替精确原量。是否需要它须由后续版本容量和过程评阅结果决定。
4. **职责与所有权不扩张。** 全量精确区间/集合继续独立程序验算；模型审主要变化、并行/先后、局部/整体边界与未知。原量变换、任务坐标及语义由 Stair/Push 各自负责，共用层仅搬运、编码、引用与校验；不把 Stair 的足力/净空含义塞进公用框架。Push 当前 `related_episode` 已发送完整 `pose/planar_motion` 和离散条件，`behavior.py` 89–139 行从同一步样本组织窗口关系；没有足/contact测量就继续明确缺失，不能照搬 Stair 补足。生产不依赖 PLAN。

这些是新版本准备建议，尚未实施或获得能力验证。[职责修订](../../REVIEW_RESPONSIBILITY_REVISION_ZH.md) 要求原始全程、必要覆盖及语义独立保留；[R2 focus 独立补证](../n5_confirmation_execution_r2/focus_operand_audit_r1/README_ZH.md) 的已供应数值零差异，仅证明其登记范围内的工程一致性。本例既不能证明完整过程语义已通过，也不能证明架构必须复杂化、必须增加相机/关节，或任何过程条件导致了终局失败。
