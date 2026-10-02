# 首包引用失败：只读诊断与最小设计审查

2026-09-30。只读诊断；未调用模型 API、未改生产代码、未修原答或重评分。此次审查者参与执行控制与开发，不能担任正式评委。

**结论：19 条均为原答案的真实定位错误，没有发现这些失败由 validator 错误造成。建议在现有 prepare/build/coverage 链中加入确定性证据锚点，保留原始证据和语义门。工程修复无需补采缺失测量；正式新确认不能直接重用已揭示的 R2。**

## 已独立复核的范围

从原始请求重建实际可见材料，使用独立 JSON pointer 遍历器检查 coverage 指出的 19 个引用。19/19 的材料 ID 与 SHA 匹配、原路径解析失败；下表列出的现存路径 19/19 均能解析。只验证结构存在，不评价断言是否得到支持，也没有把诊断路径写回答案。

| 类别 | 数量 | 原路径前缀 → 实际存在的前缀 |
|---|---:|---|
| forbidden_force_norm_n | 2 | `/data/raw_discrete/forbidden_force_norm_n` → `/data/direct_columns/forbidden_force_norm_n` |
| geometry | 8 | `/data/geometry` → `/data/definitions/geometry` |
| contact_history | 7 | `/data/contact_history` → `/data/definitions/contact_history` |
| success | 1 | `/data/definitions/protocol/success` → `/data/definitions/protocol/episode_outcomes/success` |
| failure_priority | 1 | `/data/definitions/protocol/failure_priority` → `/data/definitions/protocol/episode_outcomes/failure_priority` |

逐条诊断和源 SHA 在 [reference_path_structural_audit.json](./reference_path_structural_audit.json)。原模型顶层 `complete=true` 只是自报；receipt 的 `complete=false` 来自严格引用校验。19 条被拒绝，另外 36 条引用可解析；后者也不等于语义正确。

当前本地 runner、coverage 的 SHA 与首包 prepare 登记的冻结文件完全相同。实际来源关系也与打包代码一致：`related_episode` 将 definitions、raw_discrete、direct_columns 分开，未把这些字段放在原答案声称的位置。

## 推荐的最小变更

建议收敛为 **结构块 anchor_id + 可选块内 JSON pointer**。保留现有材料与一包一请求流程。在 `run_confirmation_review.build` 已完成来源校验和 envelope 包装后，由代码从这些**实际可见材料**机械生成中立的 `reference_catalog`，例如 `A017 → {material_id, sha256, base_json_pointer}`。模型看到与块相邻的短 ID、块的类型/键/数组形状和原始内容，答案只存 `{"anchor_id":"A017","within_anchor_pointer":"/history_length"}`。局部定位为空表示整块；它仍使用 RFC 6901、从已选块开始，不能越过块边界。ID 不含“正确、成功、必要事实”等判断标签，不能按此次答案或生产报告挑候选。

| 方案 | 改动与收益 | 仍在承担的风险 |
|---|---|---|
| 附录列全三元组，原样照抄 | 不改引用 schema/validator；材料生成和提示即可接入 | 每条重复 ID/SHA/长路径；模型仍可抄错或自行拼接；细粒度目录易变大 |
| 结构块 ID + 可选局部 pointer（推荐） | 机器维护来源身份和完整前缀；目录按块增长；原数据不删 | 需要一个引用契约增量；局部字段/索引仍可能选错，必须严格失败，不能保证语义正确 |

ID 只在 `packet + catalog SHA` 命名空间中稳定；不能裸 ID 跨包复用。代码把块基址与局部 token 序列确定性组合为完整 locator，并另存解析记录；不得按原答案含义搜索最相似位置。

facts→report 尤其不能重解释 ID：own facts 封存必须绑定其原 catalog 身份，report 带入并核验该绑定，不能用加入报告后的新目录顺序重新编号后解释旧答案。report 新增文本锚点复用既有 inventory_id/locator。解析后的标准三元组仍须通过当前的 report-role 限制和 exact inventory locator 比较，不能因新增 ID 入口绕过这两项检查。这是 seal/报告准备契约需同步登记的少量元数据，不需要另建封存系统。

只需要现有链条内三个小改动：

1. 材料序列化生成可复制 ID 与单一映射；材料、catalog、生成器版本及请求各自绑定 SHA，双方看到相同字节/顺序。原文件 SHA 与 envelope 的 `/data` 包装规则都必须保留。
2. 输出协议明确新引用形式；`prepare.json` 登记 catalog 身份。`build` 重建检查也重建同一 catalog，不能由独立手写表维护第二份路径真相。
3. `coverage.resolve_reference` 按冻结版本精确查 ID、解析块内 pointer，再走现有来源 SHA 与 pointer 校验。未知 ID、跨包 ID、来源漂移、重复 ID、非法/越界局部索引一律失败；禁止猜测父层、按字段名搜索、模糊纠错或隐式退回另一来源。原答案保持原样，解析结果只是可审计的派生记录，不能回填 `complete`。

这不需要新的调度、平台、采集接口或评分类。**它确实需要引用 schema 增量，但无需升级全部采集格式。** 在 manifest/prepare 中显式冻结 `reference_contract` 版本即可；旧完整三元组规则走原有版本，不能静默改变旧答的解释。最小生产改动位置是 `confirmation_review_coverage.py`（确定性 catalog 与精确解析）及 `run_confirmation_review.py`（build、提示、prepare 身份、解析记录）。`prepare_confirmation_materials.py`、`prepare_confirmation_media.py` 已调用 runner prepare，`run_sol_review.py` 已调用同一 build/audit：原则上复用这些接线，仅在公共返回结构变化时同步适配，不能复制第二个 resolver。协议、定向测试和容量回执须一起更新；本次没有实施这些修改。

## 必须保留的粒度与限制

- 定义可按 geometry/contact_history/protocol 等中立结构块给 ID，但具体断言仍用局部 pointer 到相应叶字段；不能以块可解析自动认可全部下属主张。定义说明含义，实例列提供实际值，二者不得互代。
- 数值锚点按列/结构块生成，保留 episode、C-order component、单位/dtype、时钟及区间边界。已有常值区间用局部数组索引定位完整 `[first,last,value]` 行，无需每帧每 scalar 建 ID。逗号时间序列不是 JSON 数组：局部 pointer 只能定位 component 字符串，不能假装 `/17` 可取第 17 个时间点；具体 step/区间仍须明确并由既有独立算术验证。暂不引入新的字符串选择语言。不要用摘要替代完整原数据。
- 相同数值不能合并为同一锚点；数组位置绑定不可变材料。已有稳定记录 ID 应同时校验。复合断言要能引用多个锚点，派生数值仍走既有独立算术职责。
- catalog 是定位辅助，不是事实清单、正确答案或正式必要事实分母。保留“缺证据/无法判断”的出口，保留模型自主发现必要事实与语义切分；合法 ID 不证明引用内容支持断言。
- 图片锚点按原图生成，只标识原图、帧/顺序和已有几何定位，不预写视觉结论。文本锚点保留上下文和否定条件。report 的文本锚点直接复用既有 `build_inventory` 的 inventory_id/locator，保留原覆盖分母与切分职责，不重新维护一份文本目录；不得把 report inventory 塞入 facts。

最小离线验收应覆盖五类失败路径的锚点生成、合法 ID 引错事实、定义冒充实例、同值不同来源、数组/来源 SHA 漂移、复合断言漏支持，以及新 catalog 的最大输入容量。机械测试不替代独立语义验收；一次首包错误也不能证明锚点方案能解决其余 39 包的全部问题。

## 采集复用与新确认边界

从修复所需信息看，原材料已有这些数据；旧 R2 可作为结构回归、容量预检和开发样本，不必为找回这些字段重新跑采集。

但 [N5_RUN_CARD_R2_ZH.md](../../N5_RUN_CARD_R2_ZH.md) 第 41 行与 [N5_FACT_REVIEW_PROTOCOL_ZH.md](../../N5_FACT_REVIEW_PROTOCOL_ZH.md) 第 13 行明确：根据确认结果改变表示/提示等规则，整包转开发，保留原失败/INCONCLUSIVE，另冻未用 cohort。因此采用此改动后，应将 R2 本次失败原样留档；原轮也继续 INCONCLUSIVE。新正式确认必须先冻结新契约，再使用真正未用的独立 cohort；同池改 ID 或同一 checkpoint/namespace 重跑不算未揭示储备。若只复用 R2 重包验证，只能登记为同样本工程开发，不能作为新确认通过。

源码依据：`run_confirmation_review.py:121–234,315–333,388–409`；`confirmation_review_coverage.py:10–32,40–96,166–214`；`prepare_confirmation_materials.py:50–80,94–128,491–540`。本文件是设计审查，不是修改或执行授权，也不是模型或全包资格结论。
