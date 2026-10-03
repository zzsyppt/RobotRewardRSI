# P2a：Go2 Stair 三源描述性迁移开发验收

基线 `aaf67aa23bd7e4aa8aae7a9ec9539f3aa45c6c3e`；专用分支
`codex/trajectory-translator-upgrade`。开发自测、只读 Astra ultra 代码/语义审查及
6.1 sol 身份/原数组抽核通过；**仍待委托父线程干净独立验收，不是完整 E2 放行**。
本轮没有 LLM/API、GPU、训练、服务器写入、Push 新实现或 P3/P4。

**2026-10-03 父验收补正：**独立数值/三源生产/兼容性检查通过，但原审计“长 off 中单步
重触发”选例未检查长 off，故该选例门原先未通过。下文 17/19/18 样本及“0 遗漏”是原账
内的数值核对，不能解释为该反例类别已覆盖。现按审查后明确资格定向补足，保留旧短 on
三例，样本并集为 18/20/18；生产结果不变。详见
[补正说明](./RETRIGGER_CORRECTION_ZH.md) 与 [新证据索引](./RETRIGGER_CORRECTION_VERIFICATION.json)。
原 `verification.json` 保持 e79fa7e 时的历史索引；其中代码/本文/冻结文档哈希应与该提交版本
比较，不应拿补正后的现文件冒充原快照。新索引明确绑定前后版本及未改动的原证据。

## 冻结与实现

预声明见 [FREEZE_ZH.md](./FREEZE_ZH.md)、[source_lock.json](./source_lock.json)。
冻结目录 `trajectory.p0.measurements.v2`、阈值、既有 v2 模板及其选择结果未调整。
历史 v1 由显式 archive capability 表分发：共同 channel/shape/clock/identity 验证继续执行，
v1 不允许混入 history metadata/header/数组，v2 缺 history 仍拒绝，未靠删除版本检查放行。

共享数值层缺 history 返回 unknown 标记轨迹，不能生成 history 转换事件；引用只列实际存在
channel，另列 missing_channels 及 inventory 缺失依据。终止 manager/current 仍独立处理。
共享 cohort 保留每指标有效/未知分母，v1 history 的均值和池化值是 null，不能变成零。
v1 卡片/公开文句说明缺失，并用真实 endpoint 保留缺 history 槽的终止见证；全成功且没有
manager 触发时同样保留合格 endpoint 与 missing 说明，不发明触发。

没有修改已启用测量语义，无需提升全局 catalog/template 版本。此前不受支持的 v1 由源 profile、
新增 `stair.trajectory_archive_capabilities.v1` 和 manifest 中生产代码 SHA 明确区分。
v2 原 source/record/card 输出保持不变；新 manifest 的代码 hash、路径、实测成本按实际构建记录。
没有新父代算子、跨环境 bridge、泛化 fallback 或无关重构。

## 来源、身份与可比范围

| 源 | 权重 SHA 前缀 | profile | NPZ | 回合 / 有效记录 | 原档终局 |
|---|---|---|---:|---:|---|
| K1 | fca6a59c8142 | v2 | 27 | 864 / 276259 | success785，forbidden72，timeout6，off_course1 |
| K2 | bcd8de3a9d0e | v1 | 30 | 864 / 276949 | success822，forbidden34，timeout4，off_course2，orientation2 |
| K3/R4 | 7b582643510c | v2 | 30 | 864 / 298375 | success682，forbidden125，timeout39，orientation16，off_course1，slip1 |

完整 manifest/权重/各 chunk SHA 在 source_lock；独立核验 87 个 chunk 全部匹配。
K2 是 Gen8A 完整训练，非 K1 中间 checkpoint；K1/K2 共享初始化及 seed、reward 不同。
K1/K3 同 reward、初始化不同。K3 是既有规则开发材料，三权重不是三独立随机实验。
训练关系以 `PLAN/feedback_sys/P0/inventory_manifest.json` 为准，而非 NEW_SYS/P0。

[comparability.json](./comparability.json) 保存完整实际差异。K1/K2 binding `ff0b754f…`、
seed cohort 相同，但 K1 preflight `ef434773…`/backend `c472e507…` 与 K2
`dc4f199c…`/`5a5358ba…` 不同；不能因为 binding 一致就逐 episode 配对。
K3 binding `1db7a52e…`、seed cohort 另异。

共同 Go2 实体/公开条件格/current 测量定义允许带各自分母的历史档案描述并列。
history 对 K2 不可比；整回合长度、成功组成和历史环境差异未消除，不能推断策略改善、显著性或因果。
严格父代后续接入同一当前环境、同协议条件、明确父子身份的档案，核实际随机化/命令和共同描述量，
明确整回合或共同时间窗及组成；代码变化解释相关性，不永久机械要求所有 source hash 相同。

## 原数组、事实与最终句子

[raw_to_sentence_examples.json](./raw_to_sentence_examples.json) 保存三例完整来源、原 NPZ/row/env、
原向量、独立事实、实际 claim operands/ID/最终文字。它们是 public 交付例，另于预声明必要样本：

- K1 r2.e66：base current 全回合越阈 0，history ANY 仅步228为阳性；最终句保留“1条，
  格时长代理0.02s、观测跨度0s”，没有解释成一段物理持续碰撞。
- K2 r2.e198：步271 current `[0,0,0]`，manager base_contact/done/wrapper 为真，history
  channel 不存在；最终句明确“history未采集、事件不可判定；manager终止信号独立保留”。
  独立全 K2 核查的34个禁触终点中10个 current 为零、24个非零。
- K3 r3.e64：归一 quaternion 推得身体上轴步154首次负；FL 所选低速 off 从步508记录。
  最终句明确前后顺序“不解释机制，低速off本身不能证明直立举足”。

独立 oracle 不导入生产 translator/renderer；使用原 NPZ 重算所有回合的36项测量主要事实字段、
转换/最近 onset、80项 compact metrics 与全部分面聚合。预声明按8 stratum×成功/失败及终局
分层选例，再独立补反例，最终17/19/18例。脚本/原始输出索引见 verification。

| 核查 | K1 | K2 | K3 |
|---|---:|---:|---:|
| 必要例测量事实 | 612 | 684 | 648 |
| 全队列测量 | 31104 | 31104 | 31104 |
| 转换事件 | 199188 | 91345 | 232232 |
| nearest onset 对 | 67893 | 68660 | 75397 |
| compact metric | 69120 | 69120 | 69120 |
| 聚合 metric | 8240 | 8320 | 8400 |
| 源引用复核 | 31968 | 31968 | 31968 |
| 核查数值错误 / 必要事实遗漏 | 0 / 0 | 0 / 0 | 0 / 0 |

这里的零错误仅限明确检查的档案与事实，不是统计泛化误差率。缺失能力不计成“物理事实正确”。
K2五实体的history及history-on/current-off共10个指标：有效记录0、未知276949、未知回合864，
mean/pooled=null、presence有效分母0。history-only反例单列 capability_unavailable_cases，
不可说成未发生。K1/K2未找到合格“先翻转后长低速off”见证；K3找到。空类不调规则补齐。

924个卡片 claim 的 operands 均解引用核查；公开61条语句（含6条投递元信息）逐句只读复核，
另审各模板代表句，没有发现新增错误或夸大。不是对全部924句人工独立复算的宣称。

## 公开遗漏与能力差异

| 源 | 公开 claim / 全卡 | 预算省略或旧离线投递声明省略 | mandatory槽 | canonical UTF-8 |
|---|---:|---:|---:|---:|
| K1 | 19 / 250 | 231 | 18 | 8086B |
| K2 | 18 / 274 | 256 | 18 | 8032B |
| K3 | 18 / 400 | 382 | 18 | 8127B |

预算是紧凑完整 public 对象的8192B，不是 pretty JSON 文件大小，也不是完整 LLM request。
K1/K2成功长低速off、对应失败对照和翻转顺序槽没有合格卡片见证，保留显式 unavailable，
并非假造阴性。三源互换 source 拒绝、各源 mandatory预算减1B拒绝共6个真实负例通过。

公开摘要的槽完整不等于全部事实覆盖。例如 RL current marker 回合均值
K1=0.190005、K2=0.637008、K3=0.424713，分足均值及接触组合差异仍在全卡/全队列指标，
但冻结摘要不逐项展示。本轮记录这种信息遗漏，不为新源修改冻结模板/优先级。

## 回归、兼容与成本

- 完整相关41模块执行648项：639 PASS、8环境SKIP、1独立审计新断言失败。该进程启动时已加载
  旧审计的缺失ratio整数0期望；实际合法值为0.0。修正独立审计后完整重跑两受影响模块13项全PASS，
  含新增全成功v1用例。合并唯一649项＝**641 PASS、8 SKIP**；没有生产失败被白名单忽略。
  原失败日志及最终日志均保存，未声称一次649全绿。
- K1/K3分别515628/593129条数值记录与基线旧代码逐字节一致，250/400条card claim及ID全等。
  compat输出 `record_bytes_compared` 字段表示逐字节比较的记录数，不是字节大小。
- P1b冻结801声明卡片的 evidence.jsonl/policy_card.json/policy_card_zh.md 三文件 SHA 完全不变；
  当前代码 manifest 下重新原数组审计通过。冻结必要247/完整309门全部PASS。
- 六个v1测试覆盖缺失/unknown/NaN、current短事件、manager独立、版本/数组/clock混入拒绝、
  v2不能降级、source swap、预算不足、全成功无manager触发；共享独立审计缺失篡改拒绝。
- 三源正常构建墙钟112.33/86.24/127.22s；最终独立审计60.43/48.89/63.37s。
  并行CPU执行的实测值，不是隔离性能基准。card构建记录进程峰值432361472/287408128/479465472B。
  输出目录约1.187/0.774/1.362GB（十进制，含card/公开审计），每源仅一份最终完整numeric证据；
  v2兼容使用旧代码内存重算，没有另存GB重复档。成本0 API/0 LLM/0 GPU/0训练。

审计工具开发过程中修复过ratio浮点期望、聚合dict传参、public层级及原记录link读取问题；失败原日志保留。
最终审计增加了K2不可观测反例分类，重新跑全部三源；预声明necessary_raw选择/事实不改。
生产测量没有为结果调参，也未出现需重定义冻结通用语义的bug。

P1d文档勘误：633→634 PASS，8 SKIP；82证据文件是81唯一引用文件加索引本身。
独立从多行docstring测试日志重计，原日志不改。Windows长路径假删除按正常权限只读复核，
使用短 `outputs/p2a`，未更改系统设置。所有 outputs/raw 均被Git忽略。

## 门与后续

已通过：三真实权重身份、有限v1能力、三源描述性迁移、预声明/全队列原数组复算、正常公开投递、
来源/预算负例、v2冻结兼容、相关CPU回归、只读独立开发评审。

未通过/未实施：**完整E2的unseen资格**（K3是开发材料，K1/K2相关lineage）、三独立随机实验、
严格父代描述差分、视觉理解、真实功能LLM理解或闭环收益。后续功能模型按用户要求 clean 6.1 sol，
本轮未调用。K5 Push只保留未见候选，需另确认资格；不在本轮接入。
周期资格、有效/名义限位、逐类地形及历史link/COM参考映射仍是后续义务。
交付专用分支commit/push后停止，等待委托父线程干净验收，不继续Push/P3。
