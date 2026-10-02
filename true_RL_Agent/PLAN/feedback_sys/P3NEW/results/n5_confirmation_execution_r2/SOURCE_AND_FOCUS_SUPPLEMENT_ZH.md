# R2 只读来源与焦点确定性补证

2026-09-30。本补证只读真实 R2 原始归档，不导入生产 reducer/exporter/packer，不发模型、不训练/仿真、不写远端源文件。保留原冻结文件、24 episode 算术票、正式原答和原评分。三个执行回执均 exit 0、stderr 为空、输入 SHA 前后不变；中断后先核已存在回执与具体进程，没有重复执行或覆盖原回执。

**结论：已补齐 Push 12 包的记录字节与身份链；已复算本轮全池的一部分确定性对象。不能据此报告完整责任清单 100%、语义准确率或 Phase 3-NEW PASS。**

## 回执与复用接口

| 回执 | 结果 | 范围 |
|---|---|---|
| [来源链](./source_link_audit_r1/receipt.json) | PASS_RECORDED_BYTES_AND_IDENTITY；12 包、144 字段链、0 差异 | 原始 NPZ → 导出 NPZ/全部 JSON 分片 → related 解码；匿名身份 → 完整派生定位 |
| [全池/焦点](./focus_deterministic_audit_r1/receipt.json) | PASS_DECLARED_SUBSET；1223 项、0 差异 | 12 focus 分母/成员，完整池结果计数、逐失败残差及数值采样 |
| [代表/对照与焦点计数](./focus_representative_audit_r1/receipt.json) | PASS_DECLARED_SUBSET；120 项、0 差异 | 冻结选样规则、成功对照、Stair protected contrast 数字元数据、12 focus 结局/时钟 |

执行代码、输入路径/SHA、比较对象及未核项均随回执保存。三个目录下的 `execution.json` 记录脚本 SHA、命令、耗时、结果 SHA；脚本由 SSH stdin 送入远端 Python `-B -`，仅 stdout/stderr 落本地 E2。

冻结独立接口：

- `P3R/confirmation_raw_material.py::stair_rows(manifest, rules_path)`、`push_rows(manifest_paths)`：原 NPZ 读取、连续有效步与基础区间/条件；Stair 使用 full 中 `semantics.thresholds` 作为已冻结定义，full 的派生结果仅作比较对象。
- `P3NEW/recompute_residuals_from_raw.py::signature_stair`、`stair_residuals`、`push_residuals`、`coverage`：相容域、签名、残差和哈希数值抽样。
- 代表核对脚本只重述冻结选样排序，操作数来自上述独立 reader；生产选择函数未调用。它验证“按该规则选了谁”，不证明这种代表性足够表达所有必要事实。

独立 reader SHA：`27892ab7f1d9c544bc5da69e89e77600a704ddfc2e9e71dee75b2448cdbdf4ba`。残差/覆盖 helper SHA：`329596cf06d5cba42dfc36162b632bdb59fb1c13e3e10051a607dc4739682d3a`。比较的 R2 full、原始 manifest/chunk、private 匿名映射、packing、raw focus、related focus 的 SHA 均在 `input_refs`。

## Push 12 包原件链

原始根目录为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20260930_r2`。每包从对应 run 的原 archive 按 0 基 env 和 valid 行切出；12 包均取实际 archive 行 0–411，对应控制步 1–412，未以匿名 ID 推测源 env。分块行、原 NPZ SHA、导出 NPZ SHA、JSON 每分片 SHA、每字段 dtype/shape/C-order 字节 SHA、full 指针与 source index 均存于来源回执。

| 包 | 匿名 ID | 原逻辑 ID | run / env（0 基） | full 指针 | source index | 字段链 |
|---|---|---|---|---|---|---|
| episode-13 | push-0011 | r1.e10 | 1 / 10 | `/transcripts/10` | 0 | 12/12 一致 |
| episode-14 | push-0001 | r1.e0 | 1 / 0 | `/transcripts/0` | 0 | 12/12 一致 |
| episode-15 | push-0082 | r2.e17 | 2 / 17 | `/transcripts/81` | 1 | 12/12 一致 |
| episode-16 | push-0087 | r2.e22 | 2 / 22 | `/transcripts/86` | 1 | 12/12 一致 |
| episode-17 | push-0043 | r1.e42 | 1 / 42 | `/transcripts/42` | 0 | 12/12 一致 |
| episode-18 | push-0044 | r1.e43 | 1 / 43 | `/transcripts/43` | 0 | 12/12 一致 |
| episode-19 | push-0178 | r3.e49 | 3 / 49 | `/transcripts/177` | 2 | 12/12 一致 |
| episode-20 | push-0177 | r3.e48 | 3 / 48 | `/transcripts/176` | 2 | 12/12 一致 |
| episode-21 | push-0086 | r2.e21 | 2 / 21 | `/transcripts/85` | 1 | 12/12 一致 |
| episode-22 | push-0085 | r2.e20 | 2 / 20 | `/transcripts/84` | 1 | 12/12 一致 |
| episode-23 | push-0184 | r3.e55 | 3 / 55 | `/transcripts/183` | 2 | 12/12 一致 |
| episode-24 | push-0189 | r3.e60 | 3 / 60 | `/transcripts/188` | 2 | 12/12 一致 |

核对的 12 原始字段为 conditions、control_step、episode_index、event_steps、event_verified、failure、observed_dwell、observed_recovery_step、planar_motion、pose、timeout、valid。字段四阶段为原 archive、导出 `original_values.npz`、所有原值 JSON 分片、related 解码。还核了原 runtime terminal 复制、定义元数据、private/packing 身份绑定、完整派生 source/env/episode index、source_pointer、original_logical_id 与 source binding。

原 NPZ 分块 SHA（每个 run 的所有本轮被选 env 共用这些分块）：

| run | NPZ | 全归档行范围（0 基闭区间） | SHA256 |
|---|---|---|---|
| 1 | `chunk_000000.npz` | 0–127 | `ff556a7393b69b3f0813a887690dbc5e7ac0fd9dde8371913ab2a6e508626051` |
| 1 | `chunk_000001.npz` | 128–255 | `a7463c3bcbf10cfe583ec34f7d1e3865e01ee5dcae1055db5735a08dbc135be8` |
| 1 | `chunk_000002.npz` | 256–383 | `1c88d63bf8a88db4314ae924ca04e7e787e0d73805fad64727610cd3c8acf349` |
| 1 | `chunk_000003.npz` | 384–411 | `2e656b3107121a8efec656fc79a67741c38ccf94e6e29a793b9ac94db2021220` |
| 2 | `chunk_000000.npz` | 0–127 | `719c1a5dc1036abc00ec4bfdcd9eee4606b744bfb6a0422235d189e13ba35c51` |
| 2 | `chunk_000001.npz` | 128–255 | `d543a60be6564d96b8d15a82604596f8612d1c444a304dc90c1a2cbd1017387e` |
| 2 | `chunk_000002.npz` | 256–383 | `78ee84e1ad2f838c4b9f9d9c1b5425d06ff838369a651d93d4406a9a8fa216ae` |
| 2 | `chunk_000003.npz` | 384–411 | `0d56a3379089c5f3ebcaedf0fe6537d1c36a2dc1fbe8646af6268c5ee60d8a6e` |
| 3 | `chunk_000000.npz` | 0–127 | `03d148429b310695bbc03febee4b7f457de92baa5bae4171a058ae59a7ea93c7` |
| 3 | `chunk_000001.npz` | 128–255 | `d11ff379a721b3801ef397581889bb89b24bd78445be04820d8acc54bb4d225c` |
| 3 | `chunk_000002.npz` | 256–383 | `801e7e0ef1e656824150c54cc1ee6ab1633a883354d71d53b7eae87a75a01298` |
| 3 | `chunk_000003.npz` | 384–411 | `6564f4bd9e2e2acd9eb8b5de83cd9fead9d6b60419e11a26224074aba4e4e6ce` |

这闭合的是上述 12 包已经记录的操作数与派生身份来源；pose 是原记录值并不等于已从原 quaternion、完整角速度和 absolute/reset root position 独立重建。动作向量未供，不能复算 action norm；原 terminal 含义/优先级也未作物理重分类。其他未选回合的所有 raw JSON/related 字段并未因此获得同等逐字节认证。

## 12 focus 与全池已核子集

| focus | task / 范围 | N | 原成功数 |
|---|---|---:|---:|
| focus-01 | push / all | 192 | 95 |
| focus-02 | push / high | 48 | 19 |
| focus-03 | push / logical_run_01 | 64 | 31 |
| focus-04 | push / low | 48 | 32 |
| focus-05 | push / medium | 48 | 26 |
| focus-06 | push / repeated | 48 | 18 |
| focus-07 | stair / c1_conditional_frontier | 108 | 58 |
| focus-08 | stair / c2_mechanism_ood | 108 | 88 |
| focus-09 | stair / height_ood | 108 | 59 |
| focus-10 | stair / joint_ood | 108 | 72 |
| focus-11 | stair / phase_composition_ood | 108 | 108 |
| focus-12 | stair / tread_ood | 108 | 90 |

十二份 raw/related scope 成员、分母、逐成员 run/cell/steps/dt/原始 outcome 均与原 archive 独立 reader 一致；related complete_pool_groups 展开后的成员也一致。每份 raw `scope_outcomes` 与 `scope_successes` 已核。终局类别是原 runtime 记录，未借此重新定义终局。

| 全池项 | Stair | Push |
|---|---:|---:|
| 原始回合 | 864 | 192 |
| 失败 | 173 | 97 |
| 数值抽样回合 | 16 | 10 |
| 数值窗口 | 48 | 30 |
| 规则阴性/对照缺失合计 | 97 | 65 |
| 其中实际抽到 | 4 | 5 |
| 抽样时间覆盖中位数 | 0.32990488489842773 | 0.25 |
| 最小覆盖 | 0.18933823529411764 | 0.25 |

逐失败身份、签名、状态、相容成功分母/同签名数均已核；Stair 的 8 strata 分母/成功/四项终局前谓词计数已核；Push 全池终局计数已核。选中、补抽、残差未抽 ID 及每个数值窗口、时间覆盖中位/最小值均一致。数值抽样不能代替视觉覆盖。

代表与对照：Stair 主域为 run 2 / height_ood level 5 / forbidden_contact，11 个失败中四项终局前规则皆未确认的组合有 7 个，按冻结下中位时长及 ID tie 选择 `r2.e173`（129 步）；同域仅 1 个成功 `r2.e177`（370 步）。Push 主域为 medium、计划/执行/完成/verified 均 1、not_recovered，19 个失败的 modal 条件组合 planar_speed+angular_speed 有 17 个，选择 `r1.e18`（412 步），相容成功共 26 个，同 run 对照 `r1.e16`。保存的完整件选择结果一致。Stair 各 protected contrast 的谓词、分子/分母、左右对照数也一致；本补证未把自由文本自然语言称为已验证。

## 必要的最小剩余清单

下表是已供应材料内可以确定性核对的对象清单；不是新增协议、全新物理真值或大型验证框架。若要声称形式化责任轴 100%，仍须逐项映射冻结责任清单，不能用以上 1,343 次比较或原 24/24 零差异替代未核项。

| 类别 | 已核范围 | 尚未由本补证覆盖的最小对象 / 可复用接口 |
|---|---|---|
| 来源/身份 | Push 12 episode 的原件→操作数→full 全链；focus 原池成员与时钟 | 12 focus 其余 packed operands 与原 NPZ 的逐项对应仍未全核；复用 reader，按 scope_row 顺序绑定，不能拿生产导出器结果自证。 |
| 范围/分母/原结局 | 12 focus 全成员、N、success、outcome；完整池分母 | 原 terminal 的物理含义和优先级、成功/失败机制仍归来源定义与语义责任；这些不由算术自动推翻或自动确认。 |
| Stair 原始区间与极值 | 全池四项 signature 与残差；代表/对照与 protected contrasts | `predicate_component_stats`、`any_body_or_stats`、极值及最早并列步、最大回撤见证、首尾 xyz、termination intervals 仍未全量核每个 focus packed 单元。`stair_rows` 的 spans/extrema、原曲线与 `signature_stair` 可复用；foot clearance/表面、buffer force history/完整 termination 需要原字段及冻结定义，旧 reader 当前未返回全部这些曲线。 |
| Push 原始区间与响应 | 全池 pose→condition、dwell latch reader 内核；逐失败窗口签名/对照；事件条件组数 | `whole_condition_false_stats`、每事件 count/runs/maxstreak/first、pre-latch/dwell interruptions、anchor/first/peak/last pose-motion、逐事件与全 required latch 后统计、terminal sample/conditions 的所有 focus packed 单元未逐项比。可复用 `push_rows` 的 good/pose/motion/events/spans；原 12 包 arithmetic helper 只覆盖所选 episode，不能直接扩大到 192 行。 |
| 区间压缩公共列 | 签名/选择所需区间已独立复算 | `true_steps/run_count/longest/first/first_required/runs_at_least_required/present_in_final_1s_before_T` 必须按各自定义对每个供应组件比较，保留 null、终局同一步与最后窗口截尾。当前补证未声称全部覆盖。 |
| 代表/成功反例/残差/数值覆盖 | 本轮全池固定选样规则、逐失败域/签名/状态及全窗口已核 | “代表所需过程充分”与“所有必要事实被报告”属于独立语义判定；raw/related 焦点中自由发现的行为组合不由固定四/五项签名穷尽。 |
| 报告短句及权限 | 原工程核验确认报告完整序列化、未审视觉句拒收机制、数值/视觉来源区分 | 文本中的所有数值舍入、定义解释、ABI/actionability、禁止因果外推、必要过程召回、未显示视觉输出的整体分母仍未在此评分。 |

旧 `numeric_readonly_review.json` 保持其原先 source-link NOT_AVAILABLE 与 explicit limits；上述新增证据是独立补充，不是改写旧票。R2 预先声明未核的 pose 物理构造、动作指标不能写成 PASS；已供应但尚未逐项覆盖的 focus 形式对象也不能计作 100%。双家正式验收、完整视觉支持、必要事实分母与实际作者送达仍未由此补证解决。
