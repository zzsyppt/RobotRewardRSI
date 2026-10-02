# 视觉整句容量处理：开发工程回执

**CPU 工程验证，不是视觉语义验收；0 API、0 GPU、没有新 N5 数据。**

文案复审修订：容量缺口改为 `omitted X optional statements from Y statements`，明确 Y 是全部原句数；统计和省略选择未改。旧 result、回执和 10 份公开投影已保存在 `before_capacity_wording/`，r9 冻结材料未改。

原媒体准备失败需分开解释：冻结 r3 的旧数值文案加 3 条合法最长中文观察，投影为 8,343/8,192 字节；在内存使用现有最新数值文案后为 7,969 字节。原失败材料仍保留在 `confirmation_media_capacity_failure_r1`，没有改旧 full、冻结源码或历史结果。不能由该旧缓存直接声称最新模板中文同构必坏。

另已复现当前源码的真实合同边界：文本/uncertainty 上限是字符数，200+120 个合法四字节 Unicode 字符仍通过 observer schema，却使当前 Stair 投影超过 8,192 字节。Push 当前正文可放下，但加入真实生成路径的合成父差分后也可能越界。

## 最小修复

- 只把 `visual_process_sentences` 自己生成的单条 `visual.observation` S3 原话设为可整条省略；没有把所有 S3 或任何数值句子改为可选。
- S5 覆盖统计仍必需：保留全部输出数、回合分母、机械无效数；存在有效数值冲突时明确保留冲突计数。即使原话因容量省略，冲突提示也不会随之消失。
- 原有共享容量算法提取为 `fit_process_report_capacity`；公开投影收据保留可省略 ID 次序，供后续 S4 更新使用同一策略。两任务只调用共享函数与共享视觉计数更新器。
- 超限时移除整条原话，`omitted_statement_ids` 记录 ID，保留显式 `report.capacity_omissions` 缺口，`shown` 改为实际 0。原始全部观察、引用和 full 证据不变，不截断句子，不提高预算。
- 其他必需句子、反例、残差、视觉 S5、父差分与 actionability 不因容量被删。它们及缺口说明合计仍放不下时，继续明确报 `process report exceeds its byte budget`。旧公开视图没有可省略 ID 收据时不推断新策略。

已有非视觉报告字节不变。有视觉报告新增可省略 ID 收据；有冲突时 S5 新增冲突统计，新增字节均计入原预算。observer 提示、schema 验证器和字段上限未改。并行代理核对当前 `OBSERVER_INSTRUCTION` 与 r9 冻结字符串完全相等，SHA-256 为 `01c7cd26bba019c5d917f18adf5671bb0153377ebcbf166a828c6ccb9c18d259`。

## 当前真实投影边界

所有观察均 synthetic capacity-only、3 条、每条 12 个引用、200 字符原话和 120 字符 uncertainty；数值冲突由明确标 synthetic 的 callback 制造，仅检验提示保留。

|任务/输入|正文投影字节|加合成 S4 后|原话投递|
|---|---:|---:|---|
|Stair，三字节中文|8,083|8,115|均完整投递|
|Stair，四字节 Unicode|7,144|7,176|整条省略，0 shown|
|Stair，正文恰近上限|8,191|7,176|正文完整；S4 后整条省略|
|Push，三字节中文|11,672|11,404|正文完整；S4 后整条省略|
|Push，四字节 Unicode|11,992|11,404|正文完整；S4 后整条省略|

对照把同一个合法四字节 Stair S3 强制恢复 required=True，真实投影验证器仍报预算错误，证明修复处理的是整句容量。实际数值正文、反例、原有 gaps 和原始 observer audit 均逐对象比对保持不变；省略时只新增公开容量缺口与实际 shown 计数。重复投影不会累加同一缺口。

前轮 95 项相关 unittest 通过，含新增合法 Unicode 整句省略、冲突/分母保留、父差分后再次收紧、原文不截断、输入不变、旧视图无收据仍报错，以及必需核心无法容纳仍失败的边界。本次仅修正文案后重跑原 9 项共享测试，全部通过。

独立开发复审代理报告原 10 份保存投影幂等通过；该复审者是开发代理，不是正式 N5 评委。本次更新文案后，这 10 份投影也逐份再次验证幂等，除容量缺口原文外的对象与统计完全一致。

```text
.venv/Scripts/python.exe -m unittest tests.unit.test_process_report tests.unit.test_stair_process_report tests.unit.test_push_process_report tests.unit.test_task_visual_observation tests.unit.test_visual_observer
.venv/Scripts/python.exe -m unittest tests.unit.test_process_report
```

精确源码 SHA、各输入对应完整 audit/public/S4 投影与字节数见 `result.json` 及邻接 JSON。未执行 N4 全套或新的正式评阅。后续 N4 投递账遇到 capacity gap 或 shown 校正，应把它们记录为投影派生事实，不能要求其文字逐字存在于未投影 full，也不能误称省略原话已经送到作者。
