# P2b1 Push 根部离线迁移冻结

基线：`70f0c51176134b63e637414641dcf981456d8da8`。2026-10-03。
用户确认 P2a 三个 Stair 来源已独立和 root 放行，仅是描述性案例；本轮不扩张 E2 声明。
普通权限 `git -c core.longpaths=true status --short` 为空。沙箱内长路径 D 是假删除。
已读用户级 AGENTS、主计划 v1.1、P0/P1/P2a 契约；CodeGraph 用于源码定位。
核对可用 skills：本轮为现有仓库 CPU 代码开发，无需文档生成、模型或 GPU skill。
仅本线程写文件；其他审查线程只读。

## 阶段和边界

P2b1a：冻结来源、规则、模板、样本选择；P2b1b：适配/测量/时序/队列/离线卡片和有限 public；
P2b1c：384 回合全量回放、独立原数组 oracle、真实中文逐句亲核、反例和回归。
开发自测完成后提交并停写，交委托父线程独立验收。没有该放行不进 K5/P3。

本轮 CPU CLI 是正式可调用的离线建卡入口。它不改 composition、runner、generation、
prompt 或 provider；不把 sidecar 称为正常任务已接线。P2b2 负责 profile/配置身份、
phase/来源绑定、正常 runner CPU finalizer、最终 payload 本地捕获与 legacy 双路回放。
Push 使用独立 schema、catalog、模板和 12288 字节预算。Stair 所有生产文件及冻结常量不改。

## 来源与用途

只用 K4 reference，checkpoint `1495390b192b913432b8472cb150e2db07fdc2eb811eeabe4ab87c577b24e0ff`。
六个用户给定 manifest SHA 在 `source_lock.json`；逐个下载并核验原字节及全部 chunk。
force numeric v1 与 velocity numeric v2 各三 run、192 回合。两者是同一权重、不同协议；
分别建卡，不能合池或当作两个训练 lineage。各卡显式保留三份 manifest 与 seed/run 身份。
来源用途仅 development/regression、descriptive_audit。未读 K5 行为/summary，未运行 K5。
P0 inventory K4 的已公开结果只能作历史描述，不是留出证据。

K5 后续需新采 192ep force，单独任务在代码/规则最终冻结后盲核一次；称为
“本次规则冻结后未用于 translator 调规则案例”，不称全历史完全 unseen。
失败如实保留；任何改规则/改实现后重跑是修复回归，不回称首次 holdout PASS。

## 规则（在生成新测量、卡片和查看 NPZ 行为前固定）

不拟合新的行为阈值。原生 dtype 比较沿用档案 parameters：
speed≤recovery_velocity、angular norm≤recovery_angular_velocity、|roll|/|pitch|≤recovery_tilt、
height≥recovery_height；stable 是五条件同记录合取。任一已知 false 足以 false；
否则缺失项为 unknown。阈值按 pose dtype 转换，避免 float32 边界改变。
本批两协议 dt=0.02s、window=ceil(1.6154200875076872/dt)=81、
dwell=ceil(0.3028912664076914/dt)=16；配置保存实际协议值而非硬编码本批常数。
fall height<0.18m、|tilt|>0.8rad，velocity drift>1.2m；force drift 仅诊断。
任务终点和分类由原任务契约验证，不以描述器重定义成功。

- v1 只有六个 root pose/norm 通道。v2 另有 world XY velocity 与 reset-relative XY position。
  v1 根部 signedXY、接触/足部/关节/动作/力矩/逐步奖励缺失，不填零。
  metadata 中扰动向量可有记录，这不赋予 root 运动方向能力。
- 全程六量分别报告有效数、unknown、min/max/mean；终点值只有真实终点行和该分量有效才存在。
  speed×dt 求和只是控制记录格积分代理，非精确路径；报告有效格数/格时长和观测跨度。
  signed XY 只用 v2 原列，不把世界 X/Y 命名成机器人前/侧向。
- 重用通用 NumericEpisode 身份/引用、PredicateTrack 和 interval_rows，以及 distribution；
  Push 自有 record envelope/catalog，避免 Stair 四足模板和全局 DSL 改动。
  真/假/unknown 全部连续记录段保留，含单格事件、gap 和两端截尾。
- 窗口不按结果选择。每个计划扰动：pre 是 [max(1,start-ceil(0.5/dt)+1),start]；
  velocity 的 start 行是写入前快照，during 无控制步持续值；post 是 (start,start+window]。
  force 的施力排程 [start,end)，受该物理子步影响的读数 during 为 (start,end]；
  单事件 post=(end,end+window]，若有下一次扰动只保留到下一 start（含该注入前行），
  避免把后一次扰动归因给前一次。force 序列恢复另用最后计划 pulse end。
  未发生事件窗口 unavailable，缺行/提前结束保留未观测格数，不能说后续未发生。
- 扰动向量引用 runtime_episode_records 的精确 JSON pointer：force_world_n 与 delivered_impulse_world_ns
  分别为记录的力和 physics-hook 冲量，velocity observed_delta_v_world_mps 为写入差；
  不换算成接触力，不宣称子步原始全轨迹。v2 post root velocity 可投影到该世界 XY 扰动单位向量，
  零向量/未验证扰动/缺方向时 unknown。数值投影不说明策略意图、足部补偿或因果。
- velocity 每个计划事件独立恢复；force 全部计划 pulse 完成/验证后的一个序列恢复。
  只在 0<step-anchor≤window 累积 stable，连续 dwell≥required 才锁存首次恢复；
  缺行/unknown 打断可证明 dwell，保留 unknown；锁存后不重新计为未恢复。
  分开报告 recovered、完整窗无 dwell、早终止截尾、采样截尾、anchor 未观测、注入无效。
  recovered 不是整回合 success；终点 stable、failure flags、任务 outcome 分开。
  恢复后到真实终点仍可再次不稳定；缺终点不能声称 final stable/unstable。
- 以每回合为统计单位：主卡报告逐回合均值的分布/均值；另列 pooled 稳定格占比及分子/分母。
  recovery-unit（velocity 可两项）统计另标单位；回合 any recovery 和 final stable 不与之混算。
  全队列、success/failure/unknown、low/medium/high/repeated 分面；不足/零成功明确空分母。
  聚合前强制 checkpoint/protocol/mechanism/schema/evaluation binding/source program/runtime identity 相同，
  run 与 manifest 唯一；不做跨协议或父代比较。父档未提供为 unavailable。

## 冻结模板与选择

所有句子程序成句，数字 6 位有效数字；记录原精度证据。范围/分母/unknown 不省略。
固定槽：来源和样本、全程 root 速度/漂移/高度、stable 回合等权与池化、恢复单位状态、
恢复后终点不稳定、任务 success/failure、各扰动强度、v1/v2 能力、父代和奖励缺口。
pre/during/post 与 signed root 方向在完整证据/代表回合句子中保留。
模板不得产生足接触、支撑、步态、动作力矩或 reward 因果事实。
eval reward=0 是旧评测 stub 的实现语义，不是训练 reward；本轮不重算训练分项。

代表回合在查看新测量前按 metadata 的 stratum×success/failure/unknown 分层；
每个非空格选 SHA256(`p2b1|manifest_sha|run|env`) 最小者，跨三个 run 选一例。
不因结果好看而换样。另按 (run,env,slot) 最小项保留 recovered、完整窗未恢复、
终止截尾、恢复后终点不稳定见证；缺类明确 no_witness，不挑其他条件替换。
公开投影先保留固定总览/边界槽，然后完整代表句按固定顺序装入；不截断句子，
省略项保存 claim ID+budget 原因；必要槽超预算明确拒绝。只含 public identity/句子/ID，
不含本地路径或 private/full artifacts；此投影仍待 P2b2 正常 payload 接线。

## 验收冻结

每版本全部 192 回合：独立直接 np.load 从原始 pose/事件/valid/control_step 复算六量、
stable、dwell、恢复状态、终点、窗口、分母及两种加权；不得导入新测量函数作为 oracle。
记录原数组行→测量 record→中文句→public ID，预选样本逐个保留，不以 schema 通过代替亲核。
反例至少：signed XY swap；condition mismatch；协议/source 错配；sealed 拒绝；
v1 无方向；缺足接触拒绝误述；终点缺测；dwell =required−1/=required；
任务成功与受扰恢复区别；短事件/unknown/gap；budget；零成功；两协议拒绝合池。
回归旧 Push/Stair 相关，新代码独立测试；检查 frozen801/247/309 和 P2a 三来源既有产物哈希。
保存原始命令日志、性能、冻结 hash/代码版本和未覆盖项到 ignored `outputs/p2b`。
只保存一份最终完整产物，小型合成 fixtures/测试与冻结、核验脚本/结论入 Git。
