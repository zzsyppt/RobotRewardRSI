# CPU 行为选择与作者投递修复：有限 CPU 验收通过

本轮基于 `cc36e0f`，仅处理新审计确认的内容选择、数值差分、编辑基准主卡和五处既有回归。未运行模型 API、GPU、训练或新采集；未恢复 run02，未改评分实现。所有原审计、支持目录、原始档案与 checkpoint 保留。2026-10-04经父级亲审与独立复验获得有限CPU通过，获准提交本阶段；不代表全量测试全绿或旧会话可恢复。

## 实现取舍

- 关节下侧占用 = 下侧名义 5% 带内 + 名义下界之外；上侧同理。文字与排序使用同一分子，排序取两侧较大占用/全队列有效点数。界外比例和有符号余量仍单独报告；不解释为运行时软限位或机械锁死。
- 条件集中按 `该条件结局数 − 条件回合数 × 全队列结局比例` 排序，显示条件分母和总体基线。全集中不会被跳过；第二名同分均保留。取消终止速度相对差权重；保留选中条件内部的不同运动过程。条件详情的投递优先级为 `.1 + .1/(1+合计超额回合数)`，使单例条件详情低于覆盖较大条件。
- 差分使用绝对尺度：10 个百分点、0.5m、0.1m/s、10°，再乘有效回合覆盖。过滤小于显示分辨率的变化。比较候选包含全队列结局计数及各足可分辨接触差，其余选主要差异；最终作者片段按统一预算选取，不承诺全部差分候选都展示。不以近零基线制造最大相对差。
- 主卡按真实 `current_best` 的 anchor、checkpoint、accepted trial 匹配跨代历史；初始参考匹配原始 `evaluation_subject.kind/id`。身份一致的 incumbent 父代新评估可以使用，不要求旧 job 相同；不与自身作差。缺少匹配时明确不可用，其他候选仅作为具名旁证。
- 最终作者行为片段的 source、diff、secondary、profile 一起计入 10240 字节。内部可选句携带已有语义 priority，与差分概况、差分行和每条旁证统一排序选择；来源身份及其观测限定、差分状态和主卡 required 留在核心。差分行自动携带对应定义/比较范围且只计一次。旁证不固定占用预算，不发送 job/hash/候选句收据。必需核心超预算时，仅在最终边界捕获已有 `ReportBudgetExceeded` 并保持原基准身份。
- 选例使用 Go2 摘要参数：成功回撤至少 0.02m、翻转后低速未触发段至少 0.5s；不是物理失败判据。先筛翻转后的合格段，再取最长，避免被翻转前更长片段遮住。相对 z 只称端点净升，不称单调上升或世界抬脚。

公开报告 schema 升为 v7，因为内部投递收据增加了最终预算实际使用的候选句和预算参数；比较操作数升为 v2，新增全队列及各足统计。当前报告若两侧比较操作数定义不一致则明确不可比，不伪造历史数据。旧的已持久化 v6 报告不直接兼容新的收据，需要从其原 numeric archive 重新生成；本轮没有迁移或重写历史 ledger，旧报告的通用降级仍属于下一阶段，恢复旧会话之前须处理。

## 先由原数组确定，再核最终文字

这些队列均已成为开发数据，不是 unseen 验证。独立脚本 `scripts/check_stair_behavior_delivery.py` 不导入生产测量或渲染代码；先从 NPZ 数组重算预期，再读取最终 payload。必要句缺失即失败。

| 队列 | 独立依据与最终文字要点 | 最终行为片段字节 | 数值核对 |
| --- | --- | ---: | ---: |
| R4 | RL calf 上侧边带或界外 230244/298375=77.166%；接触失败 125 回合中 40 已见上轴向下、68 未见向下且末速>0.2、17 其余；条件 height 38/108、frontier 36/108，基线125/864 | 10234 | 46 |
| K1 | RL calf 下侧 138488/276259=50.130%；RL 足逐回合接触占比中位数18.495%，最终18.5% | 10037 | 19 |
| m50 | 全864回合超时；低进度455513点中非翻转活动826、低活动454687；四足末窗相对点速均0；各条件均108/108，不虚报条件集中 | 5020 | 8 |
| m300 | 超时666/864；c2、height、joint各108/108并列保留；接触失败198/864，phase 77/108、c1 71/108 | 6434 | 19 |

另按实体、结局、过程分组和总字节检查，不仅比较数字。R4 必需 `down_before_low_off` 句删除后，独立检查报 `required fact missing`。

代表性最终作者句：

> 上轴向下后出现持续低速未触发段15回合；最长例r1/e60：步156首次观察到上轴向下，随后FR_foot最长低速未触发段在步200–637（438记录点，世界XY点速最大0.032m/s，当前力模≤1N）。不能由后段推断直立举足；先后不证明因果。

> 成功组回撤≥0.02m后返回63回合；最大回撤例r3/e281：步270→324回撤0.343m，步396返回此前进度，最终成功；实例不代表发生率。

> m50超时组：低进度455513点/864回合：上轴z≥0活动/低活动826/454687点。

m300 r1/e8 最后步103：base 当前力0N，history槽0/483.55/0N；原数组当前力全程为0，仅最后一步 history ANY 为正。没有推断槽内物理时刻。

## 差分与投递证据的边界

原数组档案经正常 `build_prompt_sections → build_prompt_payload → serialize_prompt_payload` 导出；其历史上下文是显式 CPU fixture，分数参考使用 fixture 常量，不作为新会话或重新资格验证。行为 profile 和比较操作数来自未改动原数组。

R4/K1 的 `evaluation_binding_hash`、`seed_namespace` 不同，正式差分返回不可比；没有改写绑定。独立跨队列观察可看到 RL 足接触中位数18.495%与41.776%，但不能称同一评估域的父代改善。

m50→m300 原始比较域一致，显式 CPU 数值对照保留四足接触约−1.7个百分点、净位移0.130→0.429m，以及条件结局差。包含非空 diff 和旁证的最终片段在选择补修后为7823字节；独立必要事实仍通过。此对照不宣称二者在历史 ledger 中互为父子。

单元测试另以真实测量操作数（位置差分速度1→2m/s）经过已接受候选、adapter 和最终序列化，验证合法父代差分、选中身份与实际字节预算。跨代参考、跨代 accepted trial、同身份父代新 job、无法匹配时不可用均覆盖；不是新增完整会话资格测试。

## 回归与删除

修复旧文字断言两处（改核实体/结局和数值）、不生成空 `parent_public_dev`、诊断夹具补真实 manifest 身份字段。删除已 tracked 的 `tests/unit/test_p2a_retrigger_selection.py`：其唯一目标是已经删除的旧重触发脚本，引用核查无现存调用，Git 可恢复。没有删除 untracked 审计或运行数据。

相关 CPU 回归60项全部通过（26.5秒）。全量单元测试运行2063项（513.5秒），原始结果为1失败、8错误、15跳过：唯一失败是另一关系测试仍断言旧“相对z升”字串，已改为核实体、起止步和端点净升数值；与投影模块共5项定点复测通过（0.48秒）。8个错误分别为7个本机缺 `omni` 的媒体测试实例，以及既有 `test_stair_body_projection` 导入搜索路径错误；后者补 `tests/unit` 搜索路径后通过，未改生产代码。没有再次运行全量并宣称全绿。

两个既有集成模块 `test_generation_session`、`test_fixed_width_generation` 运行12项（612.9秒）：8通过、4按原环境条件跳过。均使用既有CPU fake runner，没有启动GPU或模型API。原数组验证及实际作者出口已通过，`git diff --check` 通过。只读独立复核发现的最终预算溢出、翻转前片段遮蔽均已修复。完整日志位于下述输出目录的 `focused.log`、`unit.log`、`relations_projection.log`、`integration.log`。

亲审输出位于 `D:/humanoid_agent/.tmp/cpu_behavior_fix/`：`review.diff`、各队列的 `author.txt`/`payload.json`/`comparison.json`、`independent_results.json`、m300的`diff.txt`与`paired_payload.json`。原数组位置由独立脚本的 `--archive` 参数明确提供；不绑定本机换行或历史哈希白名单。

通用 description 异常降级、冗余链删除、新 Q 导入和性能工作留到后续阶段。本轮没有据有限过程事实猜失败原因。

## 独立验收后的单点补修

独立 reviewer 发现：合法同域的 G3 跨代 incumbent 场景中，第二条旁证固定增加209字节，挤掉230字节的超时条件详情，却容纳较短的单例条件详情。数值和身份正确不能掩盖这个内容丢失；初次版本因此未通过验收。

本次只修最终选择。所有原 required 保持不变，不增加专用队列槽或预算。主卡继续使用其语义 priority；差分概况为 `.12`、全队列差分行为 `.13`、条件差分行为 `.2`、旁证为 `.25`，与主卡过程、条件、地形等共同排序。上述数字仅是明确的内容取舍顺序，不是物理阈值或全局最优优化器。

沿原独立脚本构造的相同输入做写端重放，经过实际 adapter、payload 序列化和离线 HTTP 请求构造器（未发送请求）：G2/G3均10220字节，增加第二条旁证后主卡没有丢句。超时39/864、joint17/108内14近零/3前进、frontier9/108内9近零/0前进完整保留；四足20个百分点对照保留，另3条差分和旁证裁掉。稀有偏离/滑移结局本体仍在，条件详情省略。长job改名不改变可见行为内容。

两个G2/G3最终片段各通过独立R4原数组46项数值及必要事实检查。四队列选择重投后仍通过原独立核验。新增一项通用30回合测试：8回合超时条件（6近零/2前进）优先于额外旁证及低优先级差分，1回合偏离本体保留、其条件详情可裁。相关5个模块共37项通过（15.1秒）；没有重复全量/GPU测试。

写端重放文件位于 `C:/Users/lover/AppData/Local/Temp/cpu_selection_budget_fix_20261004/`，包括 `writer_replay.json`、`g2_accepted_a_rich_diff.user.json`、`g3_old_incumbent.user.json`、`long_internal_job.user.json`。原独立review目录未改动。

最终独立复验记录为 `C:/Users/lover/AppData/Local/Temp/independent_selection_recheck_20261004_074202/ROOT_RECHECK.json`，状态 `LIMITED_CPU_PASS`：G2/G3/长job请求中的行为片段均10220字节，与写端逐字节一致；39项原数组数值检查、45项内容保留检查通过；独立副本中report模块9项测试通过。确认保留超时条件、四足主要差分、稀有失败本体和过程见证，没有增预算、required槽或队列专用分支。首次FAIL与修复过程如上保留；全量原始15项skip和剩余7个缺omni环境错误不重新归类为通过，旧文字/导入路径仅有上述定点修复证据。v6兼容和一般描述异常隔离仍留下一阶段。


## 后续小阶段 1：纯描述计算失败隔离（有限 CPU 验收通过）

基线为已推送的 `329a155d8a3b9520bf940f3d021cde8c0e3f625f`。本节只处理两份 2026-10-04 修复审计仍指出的“普通描述异常使完成的评估失败”，不代表其他审计项已完成。父级亲审及独立验收已通过，2026-10-04 获准提交本阶段；未恢复 run02，未改历史 Genesis/账务，未运行 GPU、训练、模型或新采集。

实现边界：

- 原档身份、完整性、必需通道、名义来源资格继续严格校验。地形 geometry/坐标/实体/轴检查单独在计算捕获外执行，缺原点或非有限观测仍保留既有 unknown 含义。
- 首个纯测量异常后停止测量，但继续耗尽同一 lazy iterator，并继续每回合地形资格检查；后续读取或绑定失败仍阻断。正常耗尽才抛 `DescriptionComputationError`。聚合和新生成报告渲染采用同一出口；预算和 source/domain 检查保持各自边界。
- finalizer 输出 `description_computation_failed`，保留原科学 summary、评估 pass 和已完成的比较操作数，不发布半成品 behavior/full。原始 cause 经标准 logger 输出 job/stage/traceback；本次只验证日志发出，不保证宿主日志持久化策略。
- 相邻真实投递测试发现 diagnostic 投影只转发 behavior、丢失不可用状态，已补转发经过 canonical phase 校验的 description_status。正式 unified public 投影也接受该原因。作者文字为：“档案完整性已检查；行为描述计算失败，本次保留评估数值，描述不可用。”

验证均在 `true_RL_Agent`、现有 `.venv`、CPU 下执行（PowerShell 设置 `$env:PYTHONUTF8='1'`、`$env:PYTHONPATH='src;.;tests/unit'`）：

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_stair_trajectory_pipeline tests.unit.test_terrain_reference -v
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_stair_trajectory_production tests.unit.test_stair_trajectory_adapter tests.unit.test_stair_trajectory_report tests.unit.test_stair_diagnostic_evaluation -v
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_stair_trajectory_pipeline tests.unit.test_stair_diagnostic_evaluation -v
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_stair_phased_evaluation.StairPhasedEvaluationTests.test_current_robot_rows_enforce_candidate_sealed_consistency tests.unit.test_stair_phased_evaluation.StairPhasedEvaluationTests.test_reference_does_not_consume_candidate_public_to_sealed_guard -v
```

结果：首组 23/23 通过（19.274s）；第二组 47 项中 46 通过，诊断出口丢状态被新增测试发现；修复并加入异常日志后，第三组相关 17/17 通过（20.881s）。正式投递测试构造先修正漏导入和未经过 runner 补齐 sealed 字段的问题，最后第四组 2/2 通过（2.515s，覆盖三个 protocol 子例）。覆盖“测量只调用一次而 iterator 读完”“晚到 NumericEvidenceError 优先”“第二次 full replay 渲染失败不附半成品”“缺 joint_pos 与后续地形轴错误仍阻断”。未重跑全量 unit/integration，不宣称全仓库全绿。

另用同一现有 CPU archive fixture，直接加载 Git 基线的 production/terrain 源码与当前实现比较：summary、comparison、full/public report 逐字段相同。正常行为内容未变。独立代理只读审查确认异常边界和四项关键测试有效；没有新增通用异常平台、历史迁移层或旁路报告。

父级亲读核心 diff 后接受本项。独立 `FINAL_REVIEW.json` 位于 `C:\Users\lover\AppData\Local\Temp\independent_description_isolation_20261004_143417`，结论 `LIMITED_CPU_PASS`：30 项测试通过、1 项因 Temp 快照未包含旧 mesh 产物而跳过。独立验证还覆盖三 run 临时档案的实际第三 chunk 在懒读取阶段损坏：测量先失败且仅调用一次，后续 chunk hash/size 错误仍阻断；正常 summary/comparison/full/public 与基线相等；正式和诊断状态均到达最终作者 HTTP 请求构造，但未发送请求。此结论限定于有限 CPU 合成档案，不证明旧会话可恢复、GPU 运行或训练收益。


## 后续小阶段 2：旧 v6 会话不做迁移（明确接受的边界）

经原 ledger、Genesis reference 和当前 public 投影接线核查，本轮选择不实现 v6 兼容或窄 reference 迁移。旧会话需使用原版本读取；可从保留的 numeric archive 做 CPU 重生成，但这不等于原 ledger 可继续运行，也没有修复旧 run02 恢复。保留现有版本拒绝错误，不改写历史账本。原因是会话资格依赖原始 public receipt、guard 和持久化结果的一致性，不能仅替换文字便声称迁移完成。本轮没有新增兼容路径。

## 后续小阶段 3：三个算子直接返回消费量（独立 CPU 验收通过）

起点 `880e2e03ceb07eb8bb0a5764085d829862cc1e2f`。本次只清理每回合构造后立即丢弃的 cycles/joints/terrain 包装；不改 cohort 聚合、比较操作数、报告选择或版本。正式生产消费者只有 `trajectory_production.build_summary`，`scripts/` 无直接调用；受影响的 unit 调用和独立算子 oracle 已迁移。

- cycles 返回每足合格的唯一原始周期（秒）；保留四周期局部窗口、变异/边界/debounce 资格、未知和缺步分段、重叠去重。`episode_periods` 仍先取每回合中位数，再等权汇总回合。
- joints 返回每实体的全队列计数输入和最小有符号余量（rad）；保留名义坐标/单位/界限/形状检查、上/下侧区分及精度边界容差。删除无人消费的极值见证、范围、sample-cell 统计和 asset_id 包装参数。
- terrain 将资格检查与纯计算保持分离：`prepare_terrain_inputs` 仍在描述失败捕获外；`measure_terrain` 接收已检查输入，直接返回名义高度、参考点相对高度和状态数组。删除重复几何/source/hash/record/extrema 包装及旧薄 wrapper。实体顺序、世界轴、geometry digest、原点未知和边界规则保留。
- 删除三个算子仅用于包装的版本/哈希/配置文字、未使用 phase 常量和过时注释；保留真实行为事件和显著见证使用的 `NumericEpisode.reference` 及共享 contracts。测试删掉四项只验证废弃包装/重签名的用例，保留独立标量周期/关节 oracle 和历史 mesh 数值 oracle；新增实际回合等权断言。未删除整个文件，删除内容均为已 tracked 代码，可从起点 Git 恢复。

历史审计目录中的离线探针是原始记录，原样保留；它们调用旧算子接口，不能宣称可在新接口下原样重跑。没有新增兼容包装，也未改动 untracked 审计、raw、checkpoint 或运行数据。

CPU 验证（同一 `.venv`，`PYTHONUTF8=1`，`PYTHONPATH=src;.;tests/unit`）：

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_trajectory_cycles tests.unit.test_joint_limits tests.unit.test_terrain_reference tests.unit.test_trajectory_behavior
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_stair_trajectory_production tests.unit.test_stair_behavior_relations tests.unit.test_stair_trajectory_report tests.unit.test_stair_trajectory_pipeline tests.unit.test_stair_trajectory_adapter tests.unit.test_stair_diagnostic_evaluation
```

以上分别 36/36（11.335s）、60/60（26.420s）通过。最后去掉测试 oracle 自身不再读取的窗口元数据后，周期/关节 16/16 定点复核通过（0.050s）。最初一次迁移脚本因相对工作目录错误未执行，旧接口测试报 4 个预期不匹配；纠正路径后得到上述通过结果，无生产失败遗留。

R4/m300 使用同一 CPU 导出脚本、原数组和作者上下文 fixture，清理前后 `full.json`（含 audit.summary）、`comparison.json`、`payload.json`、`author.txt` 全部逐字节相等。已进入开发数据的这两组没有作为 unseen 验收。独立脚本 `scripts/check_stair_behavior_delivery.py` 从原数组重算，R4 46 项、m300 19 项数值断言以及结局/实体/过程/正文预算要求全部通过。没有调用模型或发送 HTTP。

| 队列 | 作者可见行为字节 | CPU 秒：前 → 后 | 墙钟秒：前 → 后 |
| --- | ---: | ---: | ---: |
| R4 | 10234 | 18.469 → 11.406 | 19.071 → 11.559 |
| m300 | 6434 | 12.000 → 9.906 | 12.317 → 10.067 |

耗时是同机各一次串行运行的观测，计时从导入后读取 manifest 开始，到写完 payload 为止；不作一般性能保证。前后原始产物和计时位于 `C:/Users/lover/AppData/Local/Temp/trajectory_operator_cleanup_20261004_150333/{before,after}/{r4,m300}/`，同目录 `export_delivery.py` 可核执行范围；此导出 fixture 不证明新会话资格或旧账本恢复。

代表性最终文字保持不变：

> RL_calf_joint名义限位：全队列有效298375点，下侧5%带内或界外0%，上侧77.2%；其中超出名义下/上界0%/18.3%；最小有符号余量-0.043rad。

> 上轴向下后出现持续低速未触发段15回合；最长例r1/e60：步156首次观察到上轴向下，随后FR_foot最长低速未触发段在步200–637（438记录点，世界XY点速最大0.032m/s，当前力模≤1N）。不能由后段推断直立举足；先后不证明因果。

父级亲读实际 diff 后批准本项提交。独立验收 `C:/Users/lover/AppData/Local/Temp/independent_operator_cleanup_20261004_152151/FINAL_REVIEW.json` 为 `LIMITED_CPU_PASS`：68 项相邻测试全通过、无 skip，双岛周期/关节分母/地形数组顺序与未知值独立核查通过，R4/m300 的 46/19 项原数组检查及 baseline/writer 全产物一致。该结论限定于已有开发档案与 CPU fixture；聚合瘦身留作后续独立小阶段，不代替整轮审计处置清单或用户亲审。


## 后续小阶段 4：按消费范围聚合（独立 CPU 验收通过）

第 3 项已提交并推送为 `15c469c179262a7255323d2fe9e5ab581ab1ed29`，远端分支核对一致。第 4 项以此为起点，限定 Stair 聚合；经父级亲审和独立验收批准提交。没有改测量、比较定义、来源资格、正文预算或见证选择算法。

实际调用核查：`aggregate_behavior` 的生产入口只有 Stair `trajectory_production._aggregate_summary`；其余当前调用是四个行为/报告测试模块。共享 `distribution` 仍由 Push 消费，本项保留其实现；Stair 的周期/关节聚合改为直接中位数/最小值，不再调用全分布函数。历史审计探针保留原文。

- `_aggregate_facts` 在计算前选择总体/结局及全程/末窗所需字段；接触组合只求回合均值，其余成句量只求回合中位数，保留事实 label/unit 与有效、未知、不适用分母。姿态出现性只统计三态回合数，duration 只汇总正文使用的缺步数。
- 不再生成持久化 members、positive_episodes、无人读取的分位数、条件组全通道 facts、重复 outcomes、condition/success 组及低进度负向/多余全队列过程统计。保留条件总回合数作为集中度分母、失败条件过程计数及总体/结局所需低进度信息。
- `_aggregate_process` 仅为结局组实际展示的过程分段求速度/结束时刻中位数，条件组只计回合数。一个短函数 `process_partition` 同时供聚合、结局成句和条件成句使用，避免分支资格双维护；未引入注册表。普通 fact 字段与成句布局仍各自明确列出，本项未扩大到模板框架。
- 删除 production 的“先全量计算再裁剪”循环。真实事件见证的来源定位、代表性选择和相对高度过程保留，仅删除未消费的关系实体计数。

`behavior.comparison_operands` 完全未改，仍直接消费每回合事实：末窗均值要求完整共同窗口且无未知，端点位置/倾角保留短回合资格。没有拿正文的部分观测中位数替代比较。扩展已有一项边界测试，以完整、短、末窗含 NaN 的三回合明确验证：正文速度有效 3 回合、中位数 0.8m/s；比较速度有效 1 回合、0.2m/s，高度有效 1 回合、端点位置有效 3 回合。

最终有限 CPU 回归：

```powershell
& '.\.venv\Scripts\python.exe' -m unittest tests.unit.test_trajectory_cycles tests.unit.test_trajectory_behavior tests.unit.test_stair_behavior_relations tests.unit.test_stair_trajectory_production tests.unit.test_stair_trajectory_report tests.unit.test_stair_trajectory_pipeline tests.unit.test_stair_diagnostic_evaluation
```

57/57 通过（23.423s，无 skip）。新增边界断言首跑有测试自身未导入 `np` 的 NameError，已改用 `float("nan")`；最终回归无遗留失败。未重复全量 unit/integration，没有 GPU、模型或新采集。

沿用已验收第 3 项的 R4/m300 产物作为 before，使用同一导出脚本和原数组生成 after；二者都是已有开发数据。结果：

| 检查 | R4 | m300 |
| --- | --- | --- |
| comparison / payload / author.txt | 逐字节相同 | 逐字节相同 |
| public 正文及选择收据（除派生 content_hash） | 相同 | 相同 |
| 保留的 summary 字段与值 | 原 summary 的相等子集 | 原 summary 的相等子集 |
| 紧凑 JSON summary 字节：前 → 后 | 109813 → 77352 | 39971 → 24893 |
| 作者可见行为字节 | 10234 | 6434 |
| 独立原数组数值断言 | 46 通过 | 19 通过 |

full report 的内部 summary 变小，因此 card_id/content_hash 按原规则更新；作者文字和选择不变。清理前后两种 v7 full 都由当前 `replay_production_report` 成功验证，未新增历史兼容分支。已有独立检查脚本还核对结局、实体、过程和预算。没有据本次与测试并行运行的计时主张提速。

复核材料位于 `C:/Users/lover/AppData/Local/Temp/trajectory_aggregate_cleanup_20261004_153550/`：`export_delivery.py`、`before/`、`after/`，各队列包含 full/comparison/payload/author 及 equality 结果。原数组路径与上一小阶段相同。父级亲读实际 diff；独立 `C:/Users/lover/AppData/Local/Temp/independent_aggregate_cleanup_20261004_154812/FINAL_REVIEW.json` 为 `LIMITED_CPU_PASS`：50 项相邻测试全通过、无 skip，非空全未知、条件分母、严格比较末窗独立确认，R4/m300 的 46/19 项原数组检查、作者字节一致及旧新 v7 replay 均通过。此结论限定于已有开发档案和 CPU fixture。Push 最小必要修复与整轮两审计处置清单仍未完成，不进入下一升级计划。


## 后续小阶段 5A：Push 单次测量与父代说明（独立 CPU 复核通过）

第 4 项已提交推送为 `2dfbf3e05ea0d6e014113fb0efd8620b6d61108c`，远端分支核对一致。本项仅收束 Push 正式构建路径及错误的父代说明；作者视图短 ID、实际可见预算和终止时间/同窗对照另作下一小步。

- 单策略卡将“未提供合格父代档案”改为“本卡未计算父代配对差分”，内部原因同为 `not_computed_for_single_policy_card`。这既适用于没有父代的卡，也不会否认 prompt 层已经绑定的父代。扩展已有 selected/sibling 测试，通过有效父代身份运行真实 `_v2_prompt_record` → `_v2_feedback_summary` 链，检查父代卡引用与最终文字同时正确。
- 正式 phase 原来每回合测三遍，CLI 构建测两遍；现在都从当前调用刚得到的 records 直接构卡，每回合测一遍。保留外部 records 加载器的原数组核验，以及显式 full/card 重放核验；这两条有实际消费者，不是正常构建的冗余调用。
- 删除已 tracked 的正式链缓存分支：哈希目录、complete marker、重复 evidence/card/public 写盘、复用校验及对应缓存重试测试。当前 finalizer 仍写原来的 `behavior_full.json`。用一次/重试各测每回合一次的测试替代缓存测试；没有增加旁路输出或兼容框架。原 v1 receipt 字段名 `cache_key` 暂保留为 inputs 摘要，现有 receipt reader 使用它验证绑定；它不再指向缓存，也不读取/写入缓存。
- 删除内容可由 Git 恢复；核过当前生产和测试调用方，移除 `output_dir` 参数后仅有的正式调用和测试调用均已更新。未改 capture、恢复窗口、截尾定义、force/velocity 单位、科学评价或身份资格，也未触碰 untracked 审计和运行数据。

CPU 回归：`test_push_trajectory`、`test_push_trajectory_semantics`、`test_push_trajectory_pipeline`、`test_trajectory_behavior` 共 58/58 通过（19.765s，无 skip）。其中原有外部 records/显式 replay 测试仍验证列映射、时间窗角色和原数组绑定。本步未增加安全矩阵。

从原有 `outputs/p2b/raw/k4_force_v1` 和 `k4_velocity_v2` 各三个 manifest、各 192 回合，以其原始 identity/reference 重建正式 CPU 投影；context 按已有 Go2 档案分别使用 recovery_v3/v2。此为已有开发数据的投递检查，不是新采集或新会话 runtime 资格认证。两组改前/后 card 的 cohorts、selection、examples、parameters、timing、domain、config 完全相同；全部 claims 仅 reward_parent 的文字/ID 改变，最终 public 仅 parent_diff 一条改变，其他 section 逐值相同。来源、身份、phase binding、输入摘要不变；receipt 删除三个缓存/计时附属字段，内容派生 hash 按文字变化更新。public UTF-8 字节分别为 force 12083→12065、velocity 12056→12038。

复核材料：`C:/Users/lover/AppData/Local/Temp/push_direct_build_20261004_160318/` 中 `export_k4.py`、`compare_k4.py` 与 `before|after/{k4_force_v1,k4_velocity_v2}/{full,public,receipt}.json`。旧缓存仅作为改前临时材料留在该 Temp 目录；没有删除原运行数据。

legacy 默认/new opt-in profile 暂不切换：本项纠正已选 trajectory profile 的内部构建和内容；自动切换旧会话配置会改变其既有输出契约，超出这次修复。后续新运行是否切换，应在本轮独立验收与用户亲审后决定。已定位的原档案仅 K4 两组；历史计划把 K5 留作后续，尚无可用的已授权 K5 路径，本轮未搜索隐藏 K5，也不新采集。长 hash 作者视图、真实可见预算、第一推至终止及同窗成败事实仍待 5B，当前不宣称 Push 全部审计项完成。

父级已亲读实际 diff 并准许提交。独立复核 `C:/Users/lover/AppData/Local/Temp/independent_push_5a_20261004_161358/FINAL_REVIEW.json` 为 `LIMITED_CPU_PASS`：49 项测试全通过；K4 两组正式构建和 CLI 均每 192 回合测量 192 次，finalizer 12 回合测量 12 次且只写 full；外部重签 Temp 数值仍被语义核验拒绝；原数组、选择和正文除父代句外保持不变。此结论仅限已有 K4 开发数据和 CPU fixture，不包括 K5 或新运行。


## 后续小阶段 5B：Push 作者视图与首推过程（独立 CPU 复核通过）

5A 已提交推送为 `f933e577cca42d4ed48f74cf2090f1e24a34d8dd`，远端一致。5B 在此基础上工作，目前未提交；不读取隐藏 K5，不启动 GPU、模型或新采集。

时间定义先于新卡确定：首推至终止时长使用首个已验证扰动的实际开始步与已观测任务终点之差乘 dt；不把恢复锁存步当终点。成败同窗另取首推结束后 ceil(0.3/dt) 个记录格，不含锚点格；force 锚定完整已验证脉冲结束，velocity 锚定已验证速度写入。K4 dt=0.02s，因此均为15格、0.3s。窗内三项量（高度、平面速度模、角速度模）必须全部有限且连续完整，按回合等权；首脉冲不完整、任务提前终止、采样截尾、内部缺步/非有限值、窗内计划后续扰动分别排除，不补零。计划下一扰动在窗口末步之后才开始影响下一次读数，因此仅在其计划步严格早于窗口末步时排除。1秒只是本报告明示的时长分段，不是新增任务成败阈值。

本轮没有重定义恢复窗、锁存恢复、终点稳定或结局。已有全程均值保留为全程描述，新对照不再使用整回合均速来描述固定推后过程。各结局的完整窗子集可能有不同强度构成；文字明确它们不是配对或因果比较，也不代表被排除的早终止回合。

- 新增一个小的首推测量结构及按结局的直接汇总，量与作者文字共用 label/unit/阈值。用实际结局与存在的见证决定必需事实，删除固定20条测试和空 `no_witness` 槽；原数值算子、代表性选择及恢复事实保留。
- 正式内部 v5 report 保留所有候选事实与来源摘要，作者投影在身份验证之后才生成短编号、短卡引用。按最终可见的整个 `public_dev.behavior` 对象（正文及 selected/parent/sibling/current-best 来源）计算 UTF-8 JSON 字节，整句选择，保持原12288上限。内部 hash 不再消耗作者正文容量，已有身份/phase/trial/父代核验保留。预算不再套在内部封装上；receipt 检查配置预算与内部声明一致。最终共享 serializer 使用相同的紧凑 UTF-8 JSON 形式。
- 正式构卡不再生成无人消费的离线 public_report/projection_audit；只在原 CLI 构建和外部 records 验证入口附加离线投影。这避免正式链仍被另一套带 hash 的离线包装预算提前截断；CLI 外部验证契约与一次测量路径保留。
- 代表事件文字区分观测有限格均值、内部缺步、非有限值与终止后不适用；采样截尾不伪装成任务终止。原数组均值未改变。外力世界向量仍用 N，速度写入向量仍用 m/s，未合并两种机制。
- 新结构使用 measurement v2、中文模板 v3、正式 full/public v5 与 delivery v3，并更新当前维护的识别入口。没有旧 v4 迁移或恢复兼容层；旧运行仍需原版本。本步 CPU 重建不证明旧会话可继续。legacy 默认与 trajectory opt-in 不切换。

在生成新卡之前，`raw_expectations.py` 直接读取原 manifest/NPZ，按原 pose_fields 的高度列0、速度列3、角速度列4计算预期，不导入生产测量、渲染或格式化函数。独立预期与最终量和序列化文字的78项检查通过，包括一次故意错换高度文字可被预期发现；旧所有分面的 pose/stability/recovery/final 数值逐值相同。两组均为已有开发数据，各192回合，未称 unseen。

| K4组/结局 | 首推至终止中位数 | ≤1秒终止/可判 | 完整推后窗/该结局 | 高度均值m | 平面速度模均值m/s | 角速度模均值rad/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| force成功 | 5.04s | 0/141 | 114/141 | 0.370953 | 0.790734 | 1.39235 |
| force失败 | 0.48s | 49/51 | 7/51 | 0.325594 | 2.255 | 3.69569 |
| velocity成功 | 5.04s | 0/191 | 191/191 | 0.39314 | 0.567064 | 1.15924 |
| velocity失败 | 5.04s | 0/1 | 1/1 | 0.376005 | 1.04165 | 1.20655 |

force成功排除27个窗内计划后续扰动回合；失败排除同类14个、提前终止30个。最终作者原文示例：

> 失败51回合：首推开始至实际终止可判51回合，中位数0.48s，其中≤1s终止49/51；其余0回合无合格首推或实际终点。相同首推后窗完整7/51回合，根部高度均值0.325594m、根部平面速度模均值2.255m/s、根部角速度模均值3.69569rad/s；排除：窗内计划后续扰动14、提前终止30。

最终 K4 投递从未改动的原档案经正式构建、reference 身份检查、prompt adapter、共享 payload builder 和 `serialize_prompt_payload`；force作者行为对象11870字节，velocity9350字节，均≤12288且无64位 hash。整个最终 payload 分别55257/52166字节，其中通用代码来源/任务契约的身份字段不计入行为正文预算。另在现有合成 sibling 测试中检查已绑定父代/所选兄弟来源及最终 serializer，并通过完整 `PublicContextBuilder` 检查行为包装未经 public_history 再次暴露；未调用 HTTP 或模型。

CPU 回归：初轮79项通过；移除正式链离线投影并改事件截尾文字后70项通过；最终增强完整共享投递检查后73项通过（27.182s，无 skip）。这些集合有重叠，不相加。新测试只补完整/短窗/内部缺测/非有限值/后续扰动与最终作者投递，并替换过时槽数断言。另一次相邻四模块39项运行有38项通过、1项在原有 `test_two_native_snapshot_streams_close_at_each_env_terminal` 因本机缺少 `omni.replicator.core` 报错；native_video、rollout及该测试文件均未修改，不为本轮安装模拟器或改变环境。

材料目录：`C:/Users/lover/AppData/Local/Temp/push_author_5b_20261005_002813/`。内含 `raw_expectations.py/json`（先于新卡）、`export_delivery.py`、`check_delivery.py`、两组各自的 full/public/receipt/behavior/payload JSON 与 author.txt。没有新增 tracked 验收框架、兼容平台或哈希收据文档；既有审计/raw/运行数据保留。5B 尚待父级亲审与同一独立复核；其后还有最终全相关回归、两份审计逐项处置和相邻审计，完成后停在用户亲审点。

父级亲审实际实现及最终文字后准许提交。独立复核 `C:/Users/lover/AppData/Local/Temp/independent_push_5b_20261004_165321/FINAL_REVIEW.json` 为 `LIMITED_CPU_PASS`：59项测试全通过；384个原数组回合复算一致；最终 Urllib Request 截获但不发送，行为片段11870/9350字节。生产事件钩子确认下一计划步严格小于窗口末步的判定及首槽累计44子步正确；dt=0.07时文字显示实际0.35秒窗长。此结论不含训练、模型或K5资格，仍须完成本轮处置清单与最终回归后停待用户亲审。


## 后续小阶段 6：Push 描述失败隔离与退役 P0 清理（独立 CPU 复核通过）

5B 已提交推送为 `44384c2796df94a6e1021761b317e56627713979`。最终文档复核发现，Push trajectory 主线的普通测量/成句异常会经 finalizer 变为 runner 的 `SUMMARY_INVALID`，这是静态错误耦合，并非 K4 已触发的故障。

本步在所有档案完整读取/验证及跨档案身份、参数、时钟兼容检查之后，隔离纯测量和成卡异常，沿既有投递链给出 `description_computation_failed`，保留科学结果，不留下半张报告。预算保留原状态；身份、原始块损坏、I/O 继续报错，最终落盘位于 catch 之外。legacy 链未扩大修改，也未改正常测量、成句、预算或版本。

删除 CLI manifest 中没有读取者的 `integration=offline_only_P2b2_pending`。另删除两个 tracked 文件：`tests/unit/test_stair_trajectory_reference_facts.py`（227行，验证退役P0参考生成器的7项测试）及仅其消费的 `PLAN/feedback_sys/NEW_SYS/P0/recompute_reference_facts.py`（634行）。删除前已核消费者与 HEAD 可恢复性；历史 README/JSON、raw、checkpoint、用户审计均未改动，没有另建历史区。

writer 的5个相关模块70项通过；新增两项测试方法覆盖纯计算失败的最终序列化投递、科学量不变、实际临时块损坏和落盘 I/O 仍阻断。删除后的 unit 收集为2061个方法、无 import 错误；收集不算执行通过。两份K4各192回合重出，full/public/receipt/behavior/payload JSON与author.txt共12文件与已审5B产物逐字节相同。

父级亲读实际diff后准许提交。原独立审查者在 `C:/Users/lover/AppData/Local/Temp/independent_push_minimal_20261004_175943/FINAL_REVIEW.json` 给出 `LIMITED_CPU_PASS`：14项pipeline测试通过；测量/成句故障到最终HTTP请求构造的状态正确（截获、未发送）；真实晚到的第二次读档损坏在测量前阻断；384个K4回合的12个正常输出文件不变。writer材料在 `C:/Users/lover/AppData/Local/Temp/push_closeout_minimal_20261004_175352/`。这不是服务器全量、GPU或模型验收；最终处置文档留待服务器结果，不宣称全绿。
