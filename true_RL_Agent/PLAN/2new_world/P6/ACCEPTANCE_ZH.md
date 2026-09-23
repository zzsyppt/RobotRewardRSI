# P6 验收：当前 Push 外力评价与历史资格分开记录

2026-09-22 更新。**P6 完整资格仍未闭合，不能据工程验收直接启动 P7。** 当前G1-37/Go2活动行已推进至 `recovery_v3`：共用无位移限制的有限外力评价，重复每推为high的80%，无力空档固定0.22秒。新公开dev实测Go2为141/192、G1为33/192。用户同日后续要求已接入Push新reference默认准入 `observed_outcomes_v1`：完整有效dev证据须同时出现成功与失败，不再以固定成功率区间拒绝reference。两台现有dev复核均PASS，完整三相资格仍待验证；近饱和需看分层证据，自动门通过不等于科学GO。Stair既有P3题池、mask及评分维持。见[准入修订](REFERENCE_ADMISSION_REVISION_ZH.md)、[外力统一计划](FORCE_UNIFICATION_IMPLEMENTATION_ZH.md)及[外力统一验收](FORCE_UNIFICATION_ACCEPTANCE_ZH.md)。以下旧G1 PASS和Go2 NO-GO均按当时冻结规则保留原义，不能移植到新评价域。

## 1. Stair 当前证据与 Push 原速度域历史资格

| 格子 | 当前实测 | 资格与边界 |
|---|---|---|
| Stair × G1-37 | 当前 dev 618/864，J=0.7436874183847737；逐 episode 与 P3 一致 | 当前执行回归通过，reference 资格沿用 P3 |
| Stair × Go2 | 当前 dev 689/864，J=0.7466970595216049；逐 episode 与 P3 一致 | 同上 |
| Push × G1-37（原速度域） | 当时 dev 96/192、accept 106/192、sealed 245/448 | 原速度域reference资格PASS，真实Genesis、同checkpoint成功/失败录像闭合 |
| Push × Go2 | 原速度任务D为192/192；固定D策略的2×外力：G为152/192，H短间隔60%为149/192，I短间隔80%为121/192；J取消位移门后142/192，重复27/48 | C/D未通过原余量门；E–J均为固定checkpoint公开诊断，没有新的正式外力任务reference资格 |

此处 G1 历史资格的 cell ID 为 `push.g1_37dof.recovery_v2`，当时科学身份由 v3 reference、完整 ×1.5 任务配置及当时 cohort 共同冻结；没有把旧 0/832 改写成该次结果。当前 `recovery_v3` 的外力评价另立身份，原 checkpoint 训练条件仍保留。

2026-09-22按用户授权完成[外力推搡诊断F](EXTERNAL_FORCE_RESULT_ZH.md)：固定现有checkpoint、不重训练，先Go2再G1，在绑定躯干link质心施加名义冲量匹配的短时世界水平力。Go2仍192/192；G1为87/192（历史速度扰动96/192）。这是独立外力条件的公开开发测量，不改变上表原资格；Go2余量问题仍未解决。

随后按独立预声明完成[外力强度与视觉诊断G](FORCE_STRENGTH_RESULT_ZH.md)：仅增加Go2的1.5×、2×两档，分别185/192、152/192；2×的high为14/48，表明固定旧checkpoint在更强外力下存在余量。原始轨迹独立复算、8个案例原生视觉与精确重放核对通过，未发现持续满分由成功判别错误造成。正式外力任务及新reference资格尚未签发，因此P6整体仍NO-GO；不能把这次公开开发诊断直接变成GO。

随后按两次明确授权完成[短间隔外力H/I](SHORT_FORCE_RESULT_ZH.md)：2×外力下，repeated每推60%、空档0.22秒为42/48，总149/192；固定该时序提高至80%为14/48，总121/192。旧G按同序列评分仍45/48、总152/192；三个单推层标签不变。三组原始数据独立复算与四例原生回放通过。80%组34例失败中18例越界、16例姿态/高度失败；未调整阈值、重训或消费holdout，P6正式资格状态不变。

用户随后明确取消位移限制，[J复测](NO_DISPLACEMENT_RESULT_ZH.md)保留2×外力、high80%重复力和0.22秒空档，实时终止与评分同步移除空间失败，恢复/倾倒/高度/末态条件不变。总体142/192、高强度22/48、重复27/48；原27个越界例在新运行中21个恢复、6个仍触发姿态或高度失败。独立原始状态审计通过；该目标澄清没有重写旧资格或消费holdout，P6状态不变。

## 2. G1 原速度域正式资格（历史）

执行前冻结[声明](g1_fresh_qualification_declaration_01.json)，[独立预检](g1_fresh_preflight_audit_01.json)核对源码、实际训练 argv 与 seed 派生。随机初始化，训练 seed=42，4096×24×500=49,152,000 env-steps，最终 iteration=499；没有复用 B checkpoint，没有自动重训。训练与评测同时采用 ×1.5 协议，不是仅把已训练策略放到更强评测中。

| 相位 | 成功 / 总数 | 成功率 | 各 seed 成功数（每 seed 64） |
|---|---:|---:|---|
| public_dev | 96/192 | 50.00% | 29、36、31 |
| public_accept | 106/192 | 55.21% | 30、32、44 |
| sealed_validation | 245/448 | 54.69% | 33、40、40、28、34、33、37 |

| 相位 | low | medium | high | repeated |
|---|---:|---:|---:|---:|
| public_dev | 33/48 | 30/48 | 20/48 | 13/48 |
| public_accept | 29/48 | 33/48 | 21/48 | 23/48 |
| sealed_validation | 89/112 | 55/112 | 57/112 | 44/112 |

原 dev 余量门 [0.2,0.9] 未变；通过 dev 后才消费 accept/sealed。accept 与 sealed 均高于原 0.2 下限，public→sealed 下降为 0.005208，小于原 0.1 上限。三个新 cohort 实际 seed 分别从 1000010000、1000020000、1000030000 起；与预检中已核实的旧集合不重叠。

[离线审计](g1_fresh_qualification_audit_01.json)重算 832 条 episode、1,016 次实际扰动和分层计数。24 次未发生的第二扰动均有提前终止记录，失败样本仍在原分母内；没有将未受扰 episode 算作成功。逐事件验证了方向、幅值、XY 速度增量、z/角速度保持、注入控制步与保存的恢复窗口。训练保存 166,301 次 attempt/verified，实际 reward 执行 12,000 个 control steps、60,000 次 term 调用。

科学结论限于本次单训练 seed；未证明多训练 seed 稳定复现。完整逐步轨迹未为这次资格保存，不能把保存的恢复时间、hash 或录像一致性写成对所有连续 dwell mask 的完全独立重算。原件见[50 文件下载清单](g1_qualified_01/download_manifest.json)。checkpoint 二进制留在目标目录，最终资源审计重新计算其 hash：`91c00d4e61492dcfb6511aa0ae9ef85121c5bb08dafde7eabf29c10281ad1749`。

## 3. 同 checkpoint 原生录像与只读重载

从已完成的真实 Genesis 通过现有 load-only 路径取得原 dev receipt，再执行一次完整 3×64 public-dev 回放。没有训练、accept/sealed 或新 Genesis。固定 env_id=48 的 repeated 场景，三个原 seed 恰好覆盖侧跌、成功恢复和前跌；全部三段都实际发生两次扰动。

- [成功恢复](g1_native_01/native/video/public_dev-02.mp4)：411 帧，8.22 秒。
- [侧跌](g1_native_01/native/video/public_dev-01.mp4)：298 帧，5.96 秒。
- [前跌](g1_native_01/native/video/public_dev-03.mp4)：263 帧，5.26 秒。

[独立录像核对](g1_fresh_native_audit_01.json)对完整 192 条记录做 JSON 全字段比较，含类型表示和嵌套事件，0 差异；逐条核对 sidecar 的原 outcome、seed、checkpoint 和协议。三段 MP4 均逐帧完整解码，1280×720、50 fps；主任务已检查终止前关键帧。最后一帧在 reset 之前，不是倒地后的静止姿态。录像是解释性证据，不增加资格样本。完整原件见[21 文件下载清单](g1_native_01/download_manifest.json)。

首轮旧 reference 的六段录像与旧正式 dev 也逐条一致，见 [原录像审计](native_summary_01.json)；其中 Go2 全成功、G1 全失败，不将它们包装成当前 G1 的资格材料。

## 4. 开发过程、阴性结果与停止边界

| 阶段 | G1-37 | Go2 | 如何使用 |
|---|---|---|---|
| 原 v2 正式资格 | dev 0/192，accept 0/192，sealed 0/448 | 192/192、192/192、448/448 | 两格 NO-GO；原 holdout 已消费，不回调参数 |
| A：同原 checkpoint 的原扰动 / no_push | no_push 0/64 达到完整 dwell | no_push 64/64 | 独立原状态重算未发现所查时序/评价错误 |
| B：只改 stationary 的独立开发训练 | dev 190/192，no_push 64/64 | dev 190/192，no_push 64/64 | 开发条件通过，但都超过资格上界；不等于资格 |
| C：固定 B checkpoint 有限标定 | ×1.5 为 167/192，首点入带后停止 | ×1.5 为 190/192；×2 为 176/192，网格耗尽 | 仅公开 dev；不自动加档、换 seed、放宽门或挑 holdout |
| 新 G1 正式资格 | 本文件第 2 节 PASS | 未执行 | 新训练、新 cohort、独立冻结 |
| D：Go2匹配训练分布 | 未运行 | 同×2新训练500轮，原dev为192/192 | 工程PASS、余量门NO-GO；0holdout，停止该假设 |

原 G1 无扰动时自旋明显；B 仅改变 stationary 项后，逐 episode 平均角速度范数的跨 episode 均值从 5.616 降至 0.097 rad/s。支持该修改改善已观察的静止行为，不把它写成 PPO 失败唯一成因。原 TensorBoard `Metrics/success_rate` 是上游零指令速度跟踪统计，不是本任务的恢复成功率。

A 的完整原轨迹、B 的 256 条完整跟进轨迹均独立复算了稳定/dwell/终止；对应[原诊断](independent_timing_summary_01.json)、[B 身份与训练](stationary_v3_summary_01.json)、[B 独立原轨迹审计](stationary_v3_independent_audit_01.json)。C 的 576 条 episode/715 次事件保存事实通过[独立审计](calibration_independent_audit_01.json)，没有完整逐步轨迹，不夸大该证据的范围。

G1 的 C 点 87% 与匹配新训练分布后 fresh dev 的 50% 不可混为一次固定 checkpoint 评测：训练分布和 cohort 都发生了事前声明的改变。固定 checkpoint 的强度标定不能保证重新训练后的准确成功率。Go2 的 C 失败结论保留；若继续研究，必须另行明确新的公开开发问题、预算与停止条件，不能无声明地延长 C 或直接启动 P7。

D已另行声明在原Go2三个dev seed与已有×2条件下匹配训练分布；实际192/192，四层均48/48，240次扰动全部发生并恢复。初始模型、PPO、奖励及所查ABI与B一致，但执行源码版本不同，不宣称严格单因素因果。工程及[离线审计](go2_matched_dev_audit_01.json)PASS，原[0.2,0.9]资格门NO-GO，0holdout/重试；已结束该假设。具体见[D结果与停止记录](GO2_MATCHED_TRAIN_RESULT_ZH.md)。

D 后另行冻结的一次原生姿态诊断已完成：原 192 条 dev 记录全字段一致，三个 high 成功案例共 906 帧完整解码，所选图像未见通过倒地或异常姿态取巧。105.740 秒、0 训练/holdout/重试，D/G1/c6 原件及 373 源码保护通过。见[姿态结果与任务边界](GO2_POSTURE_RESULT_ZH.md)。它不补充资格样本；恢复窗口内再次受推是可选新研究，不能从总计划“连续扰动”一词推导为必须加做的缺陷修复。

2026-09-21用户授权连续双推试验后，E使用D同一checkpoint，将repeated第二推从第172步提前至61步，第一推仍第51步，原幅值和正交方向不变。原时序/短间隔各3×64，两arm以相同“最后一推后完整窗口恢复”定义比较：双推均48/48、总计均192/192；两arm各144个不变对照与原D记录一致。连续组第二推前40/48尚不满足瞬时稳定条件，最终仍全部恢复。只新增PLAN诊断入口，0训练/holdout/重试，53.939秒；源码及D/G1/c6保护通过。见[E实测结果](CONTINUOUS_PUSH_RESULT_ZH.md)。这次有限试验已完成，未解决余量不足，不据此晋升P6或追加强度/间隔点。

## 5. 工程、矩阵与历史兼容

P5 公开证据修复通过[真实 c6 回放](shared_history_c6_01/result.json)：五个公开历史消费者使用同一 phased adapter；G3-A 的 scheduler/prompt 同源显示实测分差，G3-B 未测保持未知；公开 reason 归宿主所有，minimum-core 升 v6，旧 8→9 补列与重复算术退役。P4 的完整冻结工程回放见[原回执](frozen_p4_replay_01.json)。逐项处置与必要 legacy reader 的消费者见[审计处置](AUDIT_DISPOSITION_ZH.md)。

Push 拥有施扰、恢复、分层、奖励与资格语义；机器人提供尺度/资产/关节/物理与 PPO 事实；共享宿主只调用任务的 phased adapter 和普通训练/评测/receipt 入口。dev 早门由任务判决，共享层不放 Push 阈值或机器人名特判。完整协议、reference ID 和 eval seed 从同一 composition 入口进入 manifest；显式旧参数和旧 manifest 解码含义仍保留，未显式冻结旧参数的命令不承诺沿用新默认自动恢复旧科学身份。

Stair [当前 dev 审计](stair_dev_summary_01.json)及[实际 ABI 日志审计](stair_current_abi_log_audit_01.json)核对真实关节名/顺序、37/12 维动作、310/235 维观测与 8 项 term 顺序，以及 physics .005/control .02/decimation4。不是只比较静态配置；未展开完整 action/observation/sensor 实例参数，不声称全部 phase0 facts 已逐项实测。

旧 G1-29 [当前执行审计](g1_29_regression_audit_01.json)证明 actor480→29、critic495→1、现后端真实推理与旧 decoder 可用，旧 checkpoint/结果未改。208 个短诊断 episode 全失败，原预算/行为失败门保留；它不是当前 G1-37 或 physical V2 的资格证明。

## 6. 原速度域历史回归、保护与资源收尾

本节源码和测试数字属于原速度域资格及随后D的历史收尾，不是外力实现快照。外力统一实施验收结束时的373文件内容闭包、本地/目标各1696项回归及保护结果见[外力统一验收](FORCE_UNIFICATION_ACCEPTANCE_ZH.md)；其后6个产品文件的reference准入变更及两端100项验证另见[准入修订](REFERENCE_ADMISSION_REVISION_ZH.md)。各批冻结源hash保留原义。

- 最终产品源码为 `b73c055c54ecbe445c59e632f928712a43a163e6494d4cd0ab0406b79dc450f2`，373 文件，本地/目标/冻结环境一致；物理运行期间未修改。活动文档和测试的收尾不改变该源码快照。
- 本地 [全量 07](local_regression_07.json)：1677 项、19 skipped、2 个旧 seed 断言失败。仅修正该测试的 G1 默认期望，并保留 Go2=42、显式 G1=42 和 lineage 拒绝检查；[本地专项](local_probe_default_delta_01.json)及[目标专项](target_probe_default_delta_01.json)均 2 项 PASS。**没有把原全量 07 改写成 exit 0，也未因只改测试期望而重跑整套本地全量。** 前一版本[本地全量 06](local_regression_06.json)为 1672 项/19 skipped/0 failure/error。
- 目标 B 源码[全量 03](target_regression_03.json)：1667 项、2 skipped、0 failure/error；后续 canonical 输入/dev 早门增量在目标另跑[66 项](target_calibration_delta_01.json)，最终 G1 默认/显式参数/native 缺省增量在本地及目标各跑[41 项](target_g1_default_delta_01.json)。Ruff F 检查通过。
- [最终执行收尾](execution_closure_02.json)：G1 新资格 1102.780 秒（0.30633 GPU 分配墙钟小时），原生回放 145.055 秒（0.04029 小时），均低于各自事前预算；新训练 1 次、native dev 1 次/3 段、provider 0 次、重试 0 次。
- 同一收尾确认：G1 正式目录 49 文件/77,977,952B 在 load-only 与录像前后 hash/size/mtime 全不变；首轮两格原训练导入源和 B 的两份训练/参考 checkpoint hash 不变；完整 c6 的 1271 文件/619,640,092B/71 checkpoint 全不变。
- 收尾时本任务进程已退出，GPU compute 进程为空，已有四把 lease 均可非阻塞取得并释放；没有删除 checkpoint 或清空共享 lease 根。此前各批成本仍按各自声明记录：首轮训练及续接见 execution_closure_01.json，B 约 0.490 小时，C 约 0.03474 小时；不将不同范围的累计数字相互替代。
- 后续D的[目标收尾](go2_matched_dev_closure_01.json)另计679.348秒（0.18871 GPU分配小时）；373源码、B/G1/c6保护均再次通过，任务进程退出、GPU compute为空。其42文件已下载核验，未新增产品代码或整套回归。

## 7. P6 尚未达到的要求

G1和Go2的新外力域reference三相完整资格均待验证。当前固定旧checkpoint的公开dev分别33/192与141/192，按已实现的 `observed_outcomes_v1` 均通过最小正反例准入；不再以G1低于20%本身作为能力不合格的理由。困难层、失败类型与近饱和情况仍须如实报告。两台已有本次实现验收的成功/失败原生案例，但这些不能代替完整三相、身份和证据验证，因此当前P6不能直接宣布科学GO。原区间判定和历史回执保留；本轮未改变Stair冻结评分域，也不将已消费holdout重新作为未见数据。
