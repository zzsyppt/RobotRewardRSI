# Stair 公共投影 v2：无条件修正与真实档案验证

日期：2026-09-29。状态：**表示、计数与投递容量检查 PASS；新语义门、作者价值门未测试。** 本记录不把离线重新投影当作新采集或新实验。

## 改了什么

生产入口仍为任务目录中的 `project_stair_behavior`。`full.v1`、检测阈值、规则版本与任务结果定义不变；只将新生成的公共视图标为 `public.v2`。已有 `public.v1` 原样读取且幂等，不重写历史证据。

- 代表过程优先选主导失败原因，再选有同 cell/run 成功资料、失败分母较大的组，取其中中位长度回合；不再取字典序第一条失败。没有匹配成功则明确 missing，不用其它条件下的成功充数。
- 比较按 **失败原因 / cell / run** 分开，保护主要组以及至少一个存在的匹配组。超预算可以省略其它比较，不能把全部比较删掉。
- 对外将 `slip_any/body` 表述为 `loaded_point_motion_any/body`：有瞬时接触力时参考点快速平移。它不能证明接触面滑动；全局 any 可以由不同身体交替触发，单身体持续确认另有条件。
- 默认画像去掉饱和的 `window_slip_any` 列；保留已确认条件计数。使用具名列和行内文字标签，分母直接写 `n`，不用读者再拼多层标签索引。
- 全量 `stratum_totals` 保留所有回合的分母、成功数及可比条件计数；详细原因行按预算省略时，同时登记省略行数和这些行覆盖的回合数。比较器按版本、列名读取完整汇总；缺列不填零。
- 区间优先保留 formal 终止、已确认 root tilt、已确认 root low_progress 三类中的各一个区间，初选与容量裁剪复用同一核心集合；同类优先延伸至终止、更长、再更早的区间，其余名额补充已有观测。同起止步的 forbidden history 与 manager 不占两份公共区间，完整档案不变。明确 `end=terminal` 是观察截止，不是状态自然结束；未增加姿态类别、检测器或采集字段。

保留 8192 B 行为视图上限，没有新增公共层任务语义或另一条行为派生主线。父任务对 `_Episode._close` 的持续条件单源修改保持。

## 真实 864 回合包

完整输入为历史评阅资料中的 [evidence-stair.json](../P3/results/author_final_bundle/author_blind_review/reviewer/evidence-stair.json)：`full.v1`，**864 回合、50,890,373 B**。旧视图来自同一包对应的 [冻结 Stair 首包](runtime_probe_r1/prompts/stair-01_T.json)。两者的完整 domain 一致。

新视图：[stair_projection_v2_real_view.json](results/stair_projection_v2_real_view.json)。源文件、生产投影文件和结果的 hash，以及逐项检查结果，见 [stair_projection_v2_real_validation.json](results/stair_projection_v2_real_validation.json)。

| 检查项 | 旧 v1 | 新 v2 |
|---|---:|---:|
| 紧凑序列化行为视图 | 8,187 B | 8,118 B |
| 同首包形状的完整 prompt JSON | 70,657 B | 70,588 B |
| 新视图下的 task sections | — | 40,644 / 57,344 B |
| 完整回合分母 | 864 | 864 |
| 全量 strata | 8 | 8 |
| 展示的详细原因组行 | 24 | 2，另 22 行明确省略 |
| 展示的匹配比较 | 0 | 1 个核心组 |
| 失败代表 | `r1.e48`，timeout | `r1.e71`，forbidden_contact |
| 同 cell/run 成功代表 | `r1.e57` | `r1.e60` |

**完整 prompt JSON 的 70,588 B 是离线表示尺寸探针。** 使用相同冻结首包，仅替换行为视图并重新生成通用 frame/预算回执，再经过实际 provider request serializer 计量；这个值对应 user 消息中的 JSON，不含 system 消息及 HTTP 包装。未调用 API。原 `source_public_result_hash` 不因换视图自动成为新投影的来源证明，所以该临时请求没有作为新的合格作者输入保存或投递，也不用于 R.0/R.4 的结果比较。真正上线的新视图须随正常 phase 派生和来源封存产生；R.0 冻结材料继续保持原内容。

旧 v1 的比较只统计 forbidden_contact 对 success：共有 30 个匹配 cell/run，最终全部省略。新 v2 按每种失败原因分别分组：共有 69 个失败原因组，其中 66 个有同 cell/run 成功、3 个缺成功。这两个总体计数的定义不同，不能把 30→66 当作新增观测或改善幅度。

保留的核心比较为 `public_dev.c1_conditional_frontier_level_5`、run 1：**forbidden_contact 9 回合，对 success 2 回合**。其中倾斜条件为 7/9 对 2/2；已确认的参考点快速平移、低进展均为 9/9 对 2/2。该对照直接说明，这些条件在本组成功中也存在，不能仅因失败中出现就认定它们是失败原因。样本较少且回合暴露时长不同，不用于因果或显著性结论。

## 完整性与代价

从完整 864 回合的已有字段逐项独立计数，核对了每个 stratum 的全部汇总、所保留比较组两侧的分母与条件数；每条保留区间也逐条匹配完整档案中的相同 episode、身体和起止/确认步。旧新视图共同指标的比较差值全为零，缺失的旧窗口列单列为未比较。

为了保住可读标签、比较和过程，24 个详细原因组中只展示 2 个，另 22 个组涉及的 **774 回合没有展示其原因组细节**；这些回合仍计入完整 stratum 汇总，不是被排除或计成零。新两个代表分别只保留 4/478、5/618 条区间。因而不能声称默认报告保住所有过程细节，也不能用区间保留数代替语义召回率。后续是否按具体焦点调整必要组，须由 R.0-b 及独立语义核对决定。

独立审查发现，原先仅按确认、是否延伸至终止和持续长度排序，会让重复脚部片段挤掉 root 过程。修正后，同一真实样本在相同 8 KiB 上限下保留 `r1.e71` 的 low_progress 110–141 与 formal 终止 407；其同 cell/run 成功代表 `r1.e60` 保留 low_progress 105–131、336–347，以及已确认 tilt 423–436。这些区间逐项匹配原档案，没有增加检测结果。每类仅保护一个核心区间，其它重复过程仍可能被省略。

此次核验对象是已有 full 档案的表示，不重新计算 NPZ、不新增仿真、也没有用录像确认新代表的高层物理名称。它证明投影与原档案一致，不能证明原检测词义已经充分、也不能证明作者会正确利用新视图。

### 已知选例局限：时长中位例不等于最常见过程

对上述核心组的冻结 full 档案做只读复核：9 个 forbidden_contact 回合按时长依次为 `e66=332`、`e63=338`、`e70=358`、`e61=399`、`e71=407`、`e64=426`、`e67=455`、`e65=456`、`e62=476` 步，均为 run 1，步长 0.02 s。当前选择的 `e71` 确是时长中位数，但属于仅 2/9 个没有 confirmed tilt 的回合；它只有第 123 步的单步 tilt，未达到连续 10 步条件。它也是这 9 个失败中唯一尾部窗口没有 low_progress 的回合，已确认 low_progress 区间仅为 110–141。成功对照 `e60=525`、`e69=556` 步均有 confirmed tilt 和 low_progress。

因此，这不是区间裁剪漏项或数字错误，而是按时长选例不能保证过程条件具有代表性。当前 `index_example` 只能解释为同组的时长中位实例，不能宣传成“最常见失败过程”。群体对照仍完整呈现 tilt 7/9 对 2/2，但这个计数 PASS 不能覆盖过程代表性的缺口。组内也并非同一过程：7 个出现 confirmed tilt 的失败中，只有 `e64/e65/e62` 的 tilt 和 low_progress 区间同时延伸至终止；若简单改选常见条件组合内的时长中位例，会选到 `e64`，仍不能称其末段过程代表全部失败。

本次不修改冻结 R.0-b 的选例、源码或既有验收结果。若 R.0-b 支持继续扩展数值过程，在 R.1/R.2 中再决定采用已有 root 条件常见组合加时长的稳定规则，或继续收窄实例声明，不手工指定更有戏剧性的回合，也不引入聚类或新行为类别。后续必要事实覆盖验收必须检查“对主导过程的声明是否有群体和实例证据共同支持”，并区分群体计数正确、实例轨迹忠实与实例能否代表常见过程。

## 字段改名审查与回归

实际生成的 v2 `semantics` 中不存在 `slip_mps`、`slip_steps`、`slip_body`、`slip_any` 悬空引用。两个阈值分别迁移为 `loaded_point_speed_mps`、`loaded_point_steps`，值与完整档案严格相等；规则、actionability 和事件代码同步映射。源码中仍有 full.v1 定义与投影过程中会被覆写的旧规则字面量，不等于 v2 输出仍引用旧阈值。画像的说明仍有意提及已省略的旧 `window_slip_any`，这是省略说明，不是可读取字段。此次事件选择修正只改任务内的选择与裁剪；`source_r0b` 必须在新探针启动前按新的源码 hash 重新冻结，不能沿用之前的源码回执。

相关组合 37 项通过：新增 projection 测试、Stair 行为测试、prompt 容量与启动预检；最终文字调整后的针对性测试通过。独立测试覆盖主导原因选择、原因不混组、缺匹配成功、持续区间、预算压力下保留比较、v1 幂等，以及重排列后仍按列名比较。另核实了两份真实 v1 文件幂等不变。

事件选择修正后重跑 projection 与 Stair behavior 两个模块，**16 项通过**。新增可重跑的 [真实冻结案例回归](verify_stair_projection_v2.py) 对完整 864 回合核验 8 KiB、全量分母、核心对照、区间来源、root 过程保留、新旧 schema 幂等与共同指标零差值；它只写本目录的离线结果，不启动仿真或 provider 请求。当前 `behavior.py` SHA-256：`01623ec03a280da8713c8451b633ca7279c8cca7e6337b67a33d83fd911803c6`。

```powershell
.venv/Scripts/python.exe -m unittest tests.unit.test_p3r_stair_projection tests.unit.test_stair_behavior tests.unit.test_prompt_capacity_projection tests.unit.test_stair_phase_d_prompt_preflight -q
.venv/Scripts/python.exe -m unittest tests.unit.test_p3r_stair_projection tests.unit.test_stair_behavior -q
.venv/Scripts/python.exe PLAN/feedback_sys/P3R/verify_stair_projection_v2.py
```
