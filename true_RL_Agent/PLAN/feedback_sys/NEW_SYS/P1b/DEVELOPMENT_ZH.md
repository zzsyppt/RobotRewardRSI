# P1b 离线队列与行为卡片

2026-10-03父线程后续验收：初版提交 `c0cafa0d` 因唯一B1（聚合claim指标/标签语义绑定）暂不放行；真实固定产物数值及801句通过。以下保留初版开发记录，B1修复及重新验收证据见 `REPAIR_B1_ZH.md`、`repair_b1_verification.json`。父线程最终放行仍待执行。

起点 `fe1f5c3775d402a5c245dd754c3041f3be92b53c`，分支
`codex/trajectory-translator-upgrade`。2026-10-02 UTC 正常权限只读确认工作区干净；
沙箱长路径假删除不作恢复、不提交。仓库内无 AGENTS.md，`.agents`/`.codex` 为空；
用户级 AGENTS 要求先用 CodeGraph，已执行，其索引未命中新加 trajectory 文件后读取实际文件。

## 前序放行与 P1 拆分

按本轮父线程明确交接：P0 语义冻结 `2d888a84` 已独立及父线程放行；P1a 初版
`0960aa51` 未通过，T1/T2 修复 `fe1f5c37` 已独立及父线程验收放行。前序开发文档
保留当时“待验收”历史，本记录不覆写历史。P1a 证据入口是
`../P1a/repair_verification.json`：107+201 项通过，必要247及完整309事实全部通过，
nearest 完整搜索域与库存修复已核查。广域 CPU 既有1失败/8错误/19跳过由父线程
独立基线复现；不是全仓通过。本轮不修无关路径。

| 小阶段 | 交付与门 | 状态 |
|---|---|---|
| P1a | 全数值测量、短事件、状态、时序与原数组引用 | 父线程已放行 |
| P1b | cohort、独立分母、确定性卡片、中文视图、逐句证据和真实离线核查 | 自测及两路只读审查通过；父线程验收待执行 |
| P1c | task-owned schema/lexicon/comparison 显式迁移，最终 serialize_prompt_payload 验收 | 未开始，仍是 P1 必须完成的义务 |

## 冻结的实现选择

依主计划 v1.1 §8–13，新增 `cohort.py`、`report.py` 和离线入口。
一个 cohort 只绑定一个 source/trial/checkpoint/任务/机器人/评测域；全队列、结果、
stratum 和精确 cell 分面。legacy 两例仅是另源开发反例，不进入 R4 分母，不伪作父代。
无父代/成功参考时明示 unavailable。成功对照只在同一精确 cell 内描述，并显示
整回合时间范围和组成，不能解释为改善或因果。

逐回合流式读取已验收 P1a JSONL，保留紧凑数值与原 record ID、字节定位、源 hash；
不把全部事件字典同时读入内存，不复制 GB 底账。输入 manifest/hash、P0 预选和必要事实
不修改。测量/配置版本不兼容或来源污染直接拒绝，没有旧口径回退。

每个描述量独立保存总回合、有效、unknown、未处理；有效采样数、unknown 采样数、
分子/分母及权重另列。根部两端位移与完整路径分开判定；力模和 signed Fz 分开有效性；
完整 run 分布不混截尾 run。低进度初始不足窗保持 unknown。存在模式只对已观察记录
表述，阴性要求相关检测覆盖；不把早终止外推到未观察未来。

卡片由确定性模板生成，所有动态句有 claim ID、模板、操作数、适用范围和证据指针。
主视图保留结果/范围、总体行为、失败过程、短终点、反例和能力缺口；18 E1 预选在
可读展开层。省略有明示依据与可寻址证据。内部路径仅在 manifest/证据索引，中文
句子只用不透明引用；本阶段不是最终作者公开协议，也没有实际投递。

四产物：`manifest.json`、`evidence.jsonl`、`policy_card.json`、`policy_card_zh.md`。
新增验证报告与日志放 ignored `outputs/trajectory_p1b/`。本轮不调用功能内 LLM、
付费 API、训练、安装、补采、视觉或 P2/P3；模板已满足当前确定事实表达。

## 验收顺序

1. cohort 与独立分母构造/变形测试；只读独立审查，根线程复核后继续。
2. 卡片模板、选择和逐句证据测试，含多实体、短事件、截尾、unknown 与语言反例。
3. R4 全864回合/298375有效行及另源2例实跑；冻结18例/247必要事实覆盖账，最终
   所有句子逐项核数值、实体、范围、限定和引用；独立原数组重算，不以 schema 代替。
4. 相关测试+107+201回归、hash、时间/峰值/大小、关键 diff 和样例原文交付。
5. 仅提交明确文件并 push 专用分支，停止写入，等待父线程独立验收；不进入 P1c。

周期资格、限位绑定、逐几何地形资格、历史 link/COM 映射仍保留后续义务。它们在卡片
中是未启用能力，不宣称不存在或永久删除。精确奖励、全部非足力、动作/力矩等缺采能力
仍不可用；不会从关联推因果或把低速度写成拒绝落脚。

## 小阶段结果与实际边界调整

1. cohort 小阶段先自测并独立审查，修复两个边界：P1a单回合力份额资格使用未乘dt的sum_N；存在控制步缺口时没有阳性见证不能确认整回合阴性。补反例后审查者另做240个随机三值组合、16252行比对通过，本线程亲核tiny-force和缺步例后进入卡片阶段。
2. 初版卡片审查发现持续区间被计数隐藏、实际逐足贡献缺失、队列组合0/11未表达、展开例导语含混；补必要字段和6句队列事实。增加确定性成功长lowoff及同cell失败见证，不改变原18 E1或阈值。
3. 逐句门发现scope、实体顺序及组合模板角色约束遗漏；用有限的source/episode/group/key/entity核查和反例测试修补，未引入DSL/配置框架。姿态改为首末有效记录值并列步号。manifest锁定8项实际生产依赖。
4. 按父线程补充要求，新增15句主视图引用原claim；其余786句保留33个折叠审计区。作者紧凑卡、字节预算和最终request留待P1c。离线Markdown即可读，不新增UI或服务。
5. 独立脚本 `verify_policy_behavior_card.py` 仅依赖stdlib/NumPy，读取锁定原NPZ重算指标、分面与引用；开发时发现的NumPy标量类型、unknown区间双截尾口径均在验证脚本自身修正，未调整生产事实迎合验证。另有变异测试检查分母、unknown、current/history、实体贡献和份额错置。

只新增P1b模块/脚本/测试/本目录文档。P1a、冻结P0、训练reward、胜负判断和旧report ABI均无修改。只有本线程写文件，两路审查均只读；父线程阶段放行待完成。

## 最终产物、成本和复验

最终目录：`outputs/trajectory_p1b/final_card/`。入口 `scripts/build_policy_behavior_card.py`；完整命令、原始日志路径/hash、冻结输入与代码hash集中于本目录 `verification.json`。

| 产物 | 字节 | 用途 |
|---|---:|---|
| manifest.json | 见verification.json | 输入/代码/产物hash、成本、范围、版本 |
| evidence.jsonl | 61332614 | 866回合底账、105分面、P1a byte locator与claims |
| policy_card.json | 1233288 | 801句claim、27展开例、15句主视图引用 |
| policy_card_zh.md | 295251 | 15句主视图、786句折叠审计 |

R4主队列864回合、298375有效行；legacy2例单独来源，两个原始来源全部60个chunk hash复核。流式读取约1.29GB已接受P1a证据；新构建20.5506s，Windows PeakWorkingSetSize=242462720 bytes（231.23MiB）。四产物合计约60MiB；未复制P1a全部event/relation。API/model/训练调用0；没有安装。

最终原始结果：

```text
新cohort/card/raw-audit + boundaries: Ran 60 tests in 16.519s, OK
P1a专项及边界: Ran 107 tests in 5.135s, OK
原有相关回归: Ran 201 tests in 48.257s, OK
冻结门: necessary pass247/247 (4168 scalar fields), full pass309/309 (45157 scalar fields), no unsupported
P1b raw audit: passed=true, raw_episodes=866, metric_checks=69280,
  aggregate_metric_checks=8400, references_checked=32042,
  claim_count=801, necessary_fact_count=247, wall_seconds=19.3081
```

60与107均含同8项project boundary测试，不能相加冒称368项独立测试。更广CPU既有失败按前序独立基线记录，不声称全仓通过。

两路最终只读审查均通过：

- requirements reviewer：801句、999外链、8生产依赖、四产物、15主视图、105分面/8400指标算术；跨source、force实体顺序、current/history、success/failure和inverted实体交换均拒绝。
- audit reviewer：不调用生产renderer，逐句8629数字/658布尔一致；750 operand/scope记录、648 P1a seek；原60chunk核查27例的698活动区间、33最长逐足诊断、69非空lowoff段；18固定例必要表达复核。最终801 claims对象与已审版相同，证据JSONL字节相同，每句恰好一次、33折叠区配平。

本线程亲核最终主视图全文，并直接查看e129 current/history原数组、e35编码、e60 XY/XYZ及逐足贡献、legacy倒置四元数与成功回撤原位置。`REVIEW_EVIDENCE_ZH.md`含主视图全文、3张展开样例全文、原数组定位及分母表；`outputs/trajectory_p1b/raw_spotchecks.json`保存原锁定NPZ值与引用；完整逐句/247覆盖记录在 `final_independent_audit.json`。

覆盖意为核心事实进入主视图或展开层且底账可追溯，不是每个P0标量都已写入中文。原四元数诊断、全部转换行、力累加、逐关节范围与nearest搜索细节保留底账；未展开理由明确记录。不以计数或schema通过替代真实句子与原数组核查。

## 关键新增原文

流式读取只保留一个回合的measurement对象，并记录原文件SHA/byte offset：

```python
def _episode_groups(path: Path, manifest: dict):
    """Keep only one episode's measurements, not the million event dictionaries."""
```

阳性、unknown和力份额边界：

```python
def observed_presence(true_count: int, unknown_count: int) -> bool | None:
    """A witness proves observed presence; an unknown region cannot prove absence."""
    return True if true_count else (None if unknown_count else False)

metrics[f"{entity}.{field}"]["value"] = v[field][i]
metrics[f"{entity}.{field}"]["episode_share_eligibility"] = "P1a abs(sum_sample_force_N)>1e-12"
```

混入其他回合与组合角色错置拒绝：

```python
if record.get("episode_id") != scope["episode_id"]:
    raise ValueError("claim operand outside episode/source scope")
if slots[role]["group"] != scope["group"] + "/" + role:
    raise ValueError("comparison outcome role mismatch")
```

## 停止点

专用分支commit/push后停止写入。自测和内部独立审查通过不表示父线程已验收；不进入P1c/P2/P3，不宣称E1全项目通过、泛化、因果解释或闭环收益。最终提交/远端一致证据在交付消息给出，以免把提交hash写入自身造成循环。
