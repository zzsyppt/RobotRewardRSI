# P2 审计逐项处置（2026-09-24）

对象：[独立审计原件](../Audit_P2_2026-09-24_ZH.md)。本次按照[实施卡](IMPLEMENTATION_ZH.md)修复；历史档案、回执、审计原件不改写。当前源冻结与新增真实验收另见本目录最终验收记录。以下区分事实、改动和明确延期项。

## 逐项裁定

| 原审计项 | 核实与处理 |
|---|---|
| 5.1 Stair 历史接触操作数缺失 | 成立，必须在 P3 词表前补齐。新 `stair.unified_first_episode.v2` 保存现有传感器的支撑/禁触身体三帧完整力向量，保留原瞬时字段与原 decoder 判据。旧 v1 的历史力为 `not_recorded`。没有增加传感器、全身体采集或接触对象推断。 |
| 5.1 历史三帧等于三个物理子步 | 不能据当前源码这样下结论。Isaac Lab lazy sensor 更新使配置周期不等于实测槽间距；记录缓冲更新顺序、实际配置和时间戳缺失，禁止推定每槽相差 0.005 s。不会强制刷新来改变原行为。 |
| 5.2 Push 逐步方向丢失 | 成立，补 `planar_motion[4]`：世界平面速度和 reset 相对位移。保留实际 GPU 范数操作数并离线检查向量范数一致性，避免改成 CPU 重算后出现阈值末位漂移。numeric profile 升为 v2，扰动协议 v2/v3 不变；旧档案显式缺方向。 |
| 5.3 训练采样覆盖窄 | 成立，但“正式训练没有分辨力”证据不足。短 smoke 的展平相邻步相关性不是有效样本量，正式三段本就跨全程。当前采样 v2 在三等分阶段各分散最多八点，总预算仍 24 × 16；保留第 1 步启动锚点，另外 23 点分散，避免长训练很早失败却完全无原始样本。 |
| 5.3 新增全程周期聚合 | 本次不做。分散样本仍不等于全程均值或精确首个非零时刻；周期聚合等 P3 定义真实诊断需求后决定。现有累计调用/非零/最大值不应被误述成只保存 last。 |
| 5.4 磁盘、padding、int32 | 容量允许，暂不改压缩、padding 或整数宽度；不引入额外存储服务。保存关节位置供下一阶段依据词表决定是否降采样。 |
| 5.4 作者/记忆体积 | 新原始历史力和方向留在私有档案，训练公开投影只增加对新 schema 的识别，字段预算不扩张；正式 generation 长期记忆仍为 14 KiB 的逐 lane 短状态。旧 width-one 8 KiB recent history 重复完整覆盖簿记属实，记录其可能挤掉旧轮次，不扩建非正式主线。 |
| 6.1 Push 判据重复 | 部分采纳。在现有任务 `failure_classifier.py` 共用最终分类，保留 velocity/force 不同优先级；连续条件、dwell、恢复时序继续独立复算。不建立通用 Torch/NumPy 状态机；共用分类不再宣称是独立分类 oracle。 |
| 6.2 TrainingBinding 重复构造 | 成立。已有 `RslRlAgentConfiguration.training_binding()` 统一固定 seed 的执行绑定，backend 与 Push 准入共用。任务协议/历史祖先域仍由调用者决定；不把来自 argv 的合同重建误合并成默认绑定。 |
| 6.3 失败 lane 采样桥接复杂 | 保留功能：训练失败恰是作者最需要证据的场合，删除会重新造成诊断空白。只收敛重复 I/O：完整验证成功后输出本次局部已验证 execution records，公开投影复用；formal 有/无两条路径都避免再次整体验证。公开 formal 入口仍完整验证，无全局缓存、skip 参数或新持久格式。 |
| 6.4 两套 Stair 回放入口 | 当前唯一入口是 `scripts/stair_numeric_dev_replay.py`。旧 P6 工具增加历史用途说明，保留旧实验可解释性。新入口的训练完成/预算/迭代/程序来源检查与 runner 祖先准入职责不同，不能按“80 行重复”删掉。 |
| 6.5 PLAN 内行为原型 | 作为只读验收工具保留；P3 生产词条必须进入各任务 src，不可 import PLAN，也不可未经验收就把验收器事件描述当作者诊断。 |
| 4 / 7 两条手工运行来源 | 成立。collector 从实际配对报告确定角色，删除文件名前缀推断；重新读取远端 manifest 与 phase 核 on/off 执行环境和 source preflight 相同，补到总账。启动回执仍为 `not_recorded`，没有伪造 argv 或完整源码证明。 |
| 7 RSS 描述 | 成立。修正生成器及重新生成的本地派生报告；明确 GNU time 单进程高水位含 waited children 的代理量，不是同时进程树总和。远端历史 full report 原字节/hash 保留，数字和成本不变。 |
| 7 `__pycache__` | 已忽略，不需要为本次做清理工程。 |

## 同类复查追加修复

1. Stair 历史力最初在 capture off 时也复制/搬运，会低估新增采集成本并扩大 accept/sealed 采集范围。现从既有数值开关传入任务 recorder 的单一布尔门控：仅 unified public-dev 的数值采集开启时保存历史。
2. generation 有正式结果时，公开 history 曾通过公开 formal helper 再做一轮完整验证。将已验证后的 formal 派生提成私有 helper；公共入口继续验证，内部不会把重复验证当作额外可信度。
3. 旧 Stair 验收器自行组装 transition 时遗漏 v2 新字段。同步其字段转发，保持原事件统计；生产 reader 同时保留 v1 的缺失语义。
4. Push phase 引用校验同步识别 v1/v2，并要求 identity 与 metadata 版本一致。没有因为新生产者升版就拒绝重查旧封存证据。

## 证据与限制

- [Stair 处置](STAIR_CONTACT_HISTORY_ZH.md)、[Push 处置](PUSH_DIRECTION_FIX_ZH.md)、[训练与回执处置](TRAINING_AND_RECEIPT_REVIEW_ZH.md)解释具体字段与测试。
- [旧总账修正核验](historical_metadata_check.json)：11 份原报告、27 次原启动，成本仍为 2995.89 秒（0.8321917 GPU 容量小时）。两条缺启动回执运行的共同 source preflight 为 `dc4f199cf0c1fcca156291aa32f0747f99134c216572ed3c7d28399abf7a6520`，环境 manifest 为 `5a5358ba6672f5d3ec6d69e01e1adbcffe13c56e7212130d6ef79b62470b7949`。
- 补齐数据不等于已经向 LLM 提供高层姿态或视觉。当前作者公开边界、接受权威、长期记忆预算都不据新原始数组改变。P3 仍需定义有证据支持的词条、默认工作集与价值验证。
- 不声称恢复全部物理子步、定位碰撞面、识别迈步、证明失败因果或证明策略改进。新增真实验收及资源数据以最终验收记录为准。
