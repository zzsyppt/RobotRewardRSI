# P1a T1/T2 修复记录

起点 `0960aa5161bd06d3dd992ddf7f4984f5ab2e8c96` 未通过父线程独立阶段门。
依据：`C:/Users/lover/AppData/Local/Temp/p1a_acceptance_0960aa_20261002/P1a_ACCEPTANCE_ROOT_ZH.txt`
及同目录 `root_key_evidence_originals.json`。本次只修 T1 引用依赖与 T2 验收门，
不进入 P1b，不改 P0 冻结事实、任务终局、奖励、ABI，不修既有相机/作者测试。

## T1：最近 onset 搜索依赖

旧实现搜索整个共同有效连续区间，却只引用获选两事件之间的行。因此配对数值正确也
不能证明已排除更近候选；无配对及回看谓词同样缺少依赖。保留最近距离、等距取早、
目标可复用、不跨 unknown/缺步的原规则，仅修补引用和搜索范围说明。

共同有效区间为索引 `[a,b)` 时，记录其索引/step范围与 candidate_count；引用
`[max(0,a-1-max(left_lookback,right_lookback)), min(n,b+1))`，覆盖全部候选原行、
两侧阻断行及两轨回看。若参考 onset 的两行没有共同有效覆盖，search_island=null，
只需引用该边两行和回看以证明不能在共同覆盖内搜索。operator_version 改为
`nearest_readout_onset.v2`；配置和值参与稳定 record ID，因此修复关系取得新ID。
source/episode身份和未受影响数值记录不变。

最小反例均先在旧实现失败，再在修复后通过：

| 判别 | 引用缺口与修复 |
|---|---|
| steps1..7，L在4上升，R在6上升 | 旧ref `[3,7)` 未包含step2；只改step2可因等距取早改选2。新ref `[1,8)` |
| L在2上升，R全false | 旧ref `[1,3)` 未包含step7；只改step7即可变配对。新ref `[1,8)` |
| 真实low_progress，28步，p=[0,.02,…]，marker在27上升 | 旧ref `[26,28)` 未含原progress step1；修改step1可移除窗口onset。双方位置均扩为 `[1,29)` |
| 左右unknown边界、LHS独有unknown、缺步 | 包含形成搜索阻断的相邻行；阻断以外改值不能改变既有参考事件关系 |
| 新增非获胜候选 | winner可以不变，candidate_count必须变；其搜索行仍被引用 |

新增7项测试方法覆盖以上及无共同覆盖分支。最终只读审阅另外报告：240组三值/缺步轨迹，
192关系、8334次引用外扰动；40组真实low_progress输入，149关系、7023次引用外数值/NaN
扰动均通过。该随机检查为reviewer报告，根线程实际执行的固定回归日志另列在verification。

## T2：库存表达与严格门控

旧比较器 `necessary==247pass && full.fail==0` 会把新增unsupported也放行。
本次补齐原18项库存映射，并要求247必要事实、309完整事实均全部PASS；不保留allowlist。
删除mean_body_xyz_m、删除/改错库存或locator的变异均须使最终门失败。

Stair v2 source.input_inventory列出真实逐步数组名称，明确缺失的物理量，以及实际力
覆盖的support和non-support实体。显式声明全部非足端力未记录；本档仅四足与base，
不能升级为全身接触结论。

P0的step_index/source_chunk_index/source_chunk_row是导出辅助数组；比较器将其语义
映射至原episode_length控制步、manifest中file+SHA身份、零基half-open row定位。
这些是derived_locators，不加入raw_temporal_channels。库存异形映射不改变冻结事实，
不伪造原始数组，不从oracle替代缺少的生产字段。完整309通过只表示冻结账本检查通过，
不代表P1a父门、物理因果、跨策略泛化或P1b已通过。

比较器同时按冻结full ledger的60个chunk核file/SHA/run、起始step与零基row映射，
拒绝越界、断裂和等长错位；run/env及step/row均要求精确整数。该门不替代原数组数值重算，
也未新增trial/checkpoint与冻结identity的逐字段核对；相关证据来自生产适配器、原独立验收
及本轮header新旧核对。

## 验证和交付

结果、命令、日志/产物SHA及代码匹配见 `repair_verification.json`。
原 `verification.json` 和 `DEVELOPMENT_ZH.md` 保留0960aa开发自测历史，不能作为本次
修复结果。所有新产物放在忽略目录 `outputs/trajectory_p1a/repair_t1_t2`，不覆盖旧证据。

`t1_examples.json` 保存3个最小反例旧/新完整record及原数组扰动，也保存真实r4.r1.e0
FL step8→FR step5的旧/新完整引用和独立原NPZ计算。`all_pair_search_audit.json` 从60个
原NPZ独立重算全部真实nearest关系，并检查完整候选范围及chunk/row引用，无生产或oracle导入。

本轮最终结果：

| 检查 | 实际结果 |
|---|---|
| 相关单元及边界 / 原专项 | 107项、4.830s、OK / 201项、49.238s、OK；均零skip |
| 冻结对比 | 必要247/247、完整309/309全PASS，4168/45157标量；无unsupported |
| R4全队列 | 864回合、298375行；measurement31104/state250376/event232232/relation79417 |
| legacy独立来源 | 全验864回合300105行，选定2回合1113行；未混入R4分母 |
| 原NPZ独审 | 60chunk SHA通过；R4 75397＋legacy226条nearest全查通过 |
| 新旧产物逐条比较 | 519359条非nearest记录完全相同；nearest原有数值字段均相同，header仅新增inventory |
| 真实产物门变异 | 删除均值/库存→308pass+1unsupported且门失败；缺非足力声明/错row语义/伪造raw step_index→308pass+1fail且门失败 |

差异均有原值/新值和理由：必要账1139、完整账2380项仅冻结容差内舍入；必要账另72项为
18例各4个档案终局扩展字段（diagnostic_flags/evaluation_cell_id/max_progress_m/
progress_fraction），这些不在必要账紧凑投影内，但在完整账中核验。无失败或unsupported差异。

真实r4.r1.e0仍为FL step8→FR step5，12个FR候选；旧引用`[4,9)`改为完整`[1,311)`。
原数组FR step9/10力模分别74.3545150756836/77.56150817871094，均已on，因此step10
不是新onset；新引用包含排除此近候选的原行。完整record、五个chunk的SHA/row及原数组
见`t1_examples.json`。其新ID为`evidence:be0350989fe9d066ca876b4de6cb37af53f019d1fb9454082bad5155e4a770af`。

R4新JSONL为1289750110 bytes（旧1218026903，增加71723207 bytes）；完整依赖引用会增加
体积，本次没有改存储架构。legacy为4147480 bytes。两manifest生产代码SHA逐项匹配本次
最终代码；最后精确整数修复仅改验收器，随后再次执行107+201及完整对比/真实变异。

按父验收确认，不重跑已隔离的18分钟广域测试，不将其2125项/1failure/8error/19skip
改称通过；新相关检查与原201专项已最终重跑。周期、限位、地形、历史link/COM映射、
新增捕获、GPU/Isaac、作者链路仍未验。commit/push专用分支后停止写入，等待父线程复验。
