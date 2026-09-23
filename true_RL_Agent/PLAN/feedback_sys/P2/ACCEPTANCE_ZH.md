# P2 验收：可重算过程证据

> 本页保留首轮 v1 档案与 r2/r5/r6 运行的历史验收事实。2026-09-24 审计修订后的字段、源码与新增真实验证见 [审计逐项处置](audit_response/AUDIT_RESPONSE_ZH.md) 和 [修订验收](audit_response/ACCEPTANCE_ZH.md)。不要把下表 v1 体积/来源当成当前 v2 的测量。

执行：2026-09-23 开始，2026-09-24 验收完成。起点 `b24ddfe9e54c7cc93318a4c6a2675eb9fb799d20`。

状态：**完整 P2 的声明范围已实现并通过验收**。覆盖 Stair/Push、G1/Go2 的当前基础过程及真实短训练采样；每项真实覆盖与尚未观测的细分情况如下，不扩展为视觉或学习收益结论。

## 实现结论

本轮按用户确认的完整范围实现 Stair、Push 基础过程与训练采样。新增共享层只负责数值块、固定采样窗口、容量和内容 hash；机器人、事件、恢复窗口、终局、物理单位及复算仍由任务拥有。使用同步有界 NPZ 块，最后发布 manifest，绑定原有 phase/训练回执；没有另建数据库或日志平台。

正常 composition 已开启 public-dev 数值采集及训练采样，底层 backend 仍允许明确关闭。accept/sealed 不新增作者可用时序。原始数组留在档案，当前作者只新增有界训练覆盖摘要；长期记忆仍使用原 14 KiB 容量和逐 lane 短状态。数值发布完整与训练成功、媒体启用是三个不同事实。

采集明细与含义见 [字段目录](FIELD_CATALOG_ZH.md)。来源准入见 [祖先准入记录](ANCESTRY_ADMISSION_ZH.md)，两项前置修复见 [P1 小修](P1_CLEANUP_ZH.md)。

## 真实评估配对

全部使用隔离的 `source_r2`，每对 on/off 的 policy、程序来源、协议、seed、评估 cohort 与执行源码相同；仅采集开关、输出目录不同。每个存储档案在离线 CPU 中重新读取并核 hash，再重算逐回合结果和现有统计。

| 配对 | 每侧首回合 | on 侧成功 / 失败 | 数值档案字节 | 采集时间差 |
|---|---:|---:|---:|---:|
| Stair Go2 parent `fca6a59c…` | 864 | 785 / 79 | 52,498,933 | +1.76% |
| Stair Go2 Gen8-A `bcd8de3a…` | 864 | 822 / 42 | 53,072,205 | +0.45% |
| Stair G1 reference | 864 | 618 / 246 | 68,170,876 | +2.22% |
| Push Go2 velocity v2 | 192 | 191 / 1 | 1,670,379 | +1.32% |
| Push Go2 force v3 | 192 | 141 / 51 | 1,614,364 | +0.96% |
| Push G1 velocity v2 | 192 | 107 / 85 | 2,188,514 | −3.00% |
| Push G1 force v3 | 192 | 33 / 159 | 2,096,821 | −0.27% |

Stair 时间为 rollout 内测量；Push 为 phase 内测量，均包含采集、编码和发布成本。单对有调度噪声，负差不是加速结论。所有配对的旧任务结果、分母及主统计完全一致；Stair 完整 task summary 相等，逐 episode outcome 复算相等；Push 独立 reducer 核对姿态条件、dwell、恢复、终局及分母。没有因不利结果放宽容差。共保存 3,360 个首回合；on/off 实际评估共 6,720 回合。

Stair 三组分别覆盖 99–544、105–600、88–600 个控制步的短/长回合，终止帧为 pre-reset，闭合后的 padding 不参与持续统计。Go2 保存四个支撑角色及 base forbidden 角色的力向量；没有声称保存完整 15-body 接触，更不能识别接触对象。

Push 四个 profile 均完成完整 public-dev cohort，含成功与失败；本批 G1 velocity、G1 force、Go2 force 分别有 35、145、50 个真实提前终止回合，Go2 velocity 没有提前终止。后者的独立形态运行已核验，velocity 早终止实现由真实 G1 velocity 覆盖，Go2 环境与共享掩码由 Go2 force 覆盖，另有 reset/有效终止帧的 CPU 反例；源码不存在 Go2 velocity 专用闭合分支。不能把这种互补实现验证写成“Go2 velocity 已观测到真实短回合”。详见 [Push 证据与覆盖限制](PUSH_EVIDENCE_FINDINGS_ZH.md)。

## 首个问题已经能查到什么

Stair 档案能定位带符号姿态、支撑条件、禁触、首次立面跨越与终局的控制步，保留未到锚点的分母。离线报告按同 cell/同 seed 分组成功和禁触回合，并保留完整逐回合证据。一个直接反例：parent 的 seed 1、`c1_conditional_frontier_level_5` 中，10 个禁触回合和 2 个成功回合都曾满足持续滑移判据；这个旗标本身不能解释为何失败。该结果是可观察条件对比，不是碰撞根因或奖励改动效果。

Push 已能区分三种不同的 `not_recovered`：曾有 dwell 后被打断、始终没有形成 dwell、曾完成恢复却在终局再次不稳。G1 velocity 的 50 个 `not_recovered` 中有 25 个属于最后一种，G1 force 为 7/14。Go2 两种协议各 1 个 `not_recovered` 都从未形成稳定 dwell；velocity 可定位 pitch 条件连续 71 步不达标，force 可定位高度条件连续 76 步不达标。成功组也有中断或短暂越界，因此不能将任何一次越界当作失败原因。详细分层、锚点、时间区间和分母保存在 [Push 机器报告](PUSH_EVIDENCE_FINDINGS.json)。

上述派生分析是 P2 的离线证据可用性验收，尚未作为自动高层诊断发送给作者；该产品链由 P3 继续实现。

## 真实训练采样配对

| 配对 | 冻结源码 | 每侧真实训练 | 保存控制步 | 档案字节 |
|---|---|---:|---:|---:|
| Stair Go2 | r6 | 16 × 8 × 1 = 128 transitions | 8 | 53,437 |
| Stair G1 | r6 | 16 × 8 × 1 = 128 transitions | 8 | 87,368 |
| Push Go2 | r5 | 16 × 8 × 3 = 384 transitions | 24 | 115,231 |
| Push G1 | r5 | 16 × 8 × 3 = 384 transitions | 24 | 115,343 |

四对均实际执行所有预定窗口，off 明确为 `not_enabled`；on 的来源、窗口索引、环境子集、实际/计划分母及保存值算术均通过独立回读检查。初始 policy hash、实际 reward/runtime 观测以及训练结束的全部模型、优化器状态相等。完整状态用 CPU tensor 比较，不用 checkpoint 容器字节差代替参数比较。

这些是 reference 短训练验证。Stair 的早/中/晚在同一短 rollout 的三个区间内；它证明真实训练 hook 和时钟接线，不等于三次长训练阶段，更不是新策略 qualification、完整预算训练或学习收益。长训练仍使用同一固定调度器，最多保存 24 个控制步 × 前 16 个环境；相应未到达/未采样边界有 CPU 检查。没有为补历史训练数据重训。

raw 指 guard/clip 后、乘权重前的 task term；保存实际 weighted 值及实际 total。Stair 和 physical Push 保留 rate→乘一次 dt 的合同，未另行观测 PPO 的 post-manager/timeout-bootstrap reward。详见 [训练采样说明](TRAINING_NUMERIC_CAPTURE_ZH.md)。

## 验收边界与资源

- 每个数值档案均低于 2 GiB，单块原始数组低于 64 MiB；最大实测档案为 68,170,876 字节，最大 Stair 块 5,037,696 字节。
- 最大正向配对 RSS 差为 22,654,976 字节，低于 512 MiB；它是 GNU time 进程高水位及 waited children 的代理观测，不是同时进程树 RSS 总和，也不是隔离的内存分配分析。
- 评估采集时间差最大 +2.22%，低于预登记 25% 上限。训练报告完整 driver wall，包含启动/退出，未将其冒充独立训练吞吐测量。
- 总共 27 次计时启动，包含所有前置对照、失败及重跑；按整 driver 占用单 GPU 的保守代理口径共 2,995.89 秒，即约 **0.8322 GPU 容量小时**，低于本轮 8 小时分配。模型请求 0、模型费用 0。逐次来源、命令、日志 hash、返回码及时间见 [运行总账](results/run_inventory.json)。
- 数值 complete 与媒体分开；本轮未启用视觉输入或媒体采集。

## 真实发现与源头修复

1. 旧 width-one 零填充不能证明 term 未调用：必须同时有正 step 数及正 term 数才能解释为 `terms_not_called`，否则保留未知。删除仅服务于审计中间字典格式的迁移分支；真实旧 9 列与当前短状态仍可回放。
2. Stair 历史父策略训练 manifest v1 与当前评估 v3 不同：任务拥有的祖先准入核训练合同、完整物理 ABI 与逐级合法继承，不改旧 receipt、不放宽成任意 hash。
3. 首次训练 driver 将 venv 路径解析成基础 Python，导致 `isaaclab` 导入失败；改为保留解释器启动路径，r5 重新准备并运行。
4. Stair checker 错把历史训练协议等同于当前执行协议；按已有 manifest loader 分别核验两种身份，r6 重新准备并完成 Stair 两组配对。原 r5 失败 receipt 保留；其实际已写档案另经只读复算通过，未篡改原失败状态。
5. 训练采样覆盖原先只留私有 summary，补入现有训练公开投影与 runtime 反馈。候选训练失败没有 `TrainJobResult`，因此还须沿已有封闭错误合同与已绑定 execution receipt 传递同一份有界采样摘要。接受、分数和源候选选择规则不据此改变。
6. phase producer 在封存前核 attachment 身份；历史 ledger 回放只校验已封存引用结构，不要求旧绝对路径仍在线。实际读附件时再核字节 hash。避免将新采集接入变成历史回放的本机路径依赖。

## 检查记录与复现

最终合并 CPU 回归 **317 项：316 通过、1 项因本机无法创建文件软链接而跳过**，87.476 秒，见 [最终日志](verification_final.log)。此外完整代际集成 **2 项通过**，包括两个训练失败 lane 的回执重放，见 [代际日志](verification_training_sampling_generation.log)；验收工具自身 **12 项通过**，见 [工具检查](verification_acceptance_tools.log)。这些集合不与此前重叠检查相加夸大数量。

检查覆盖数值存取/损坏/未发布、预重置与缺失语义、合法祖先准入、backend/argv、任务投影、记忆容量、正常与失败训练消费链、generation 回执重建。失败场景使用真实本地非零子进程及合成训练摘要来验传递链，不能冒充 GPU 候选崩溃实验。日志中故意构造的 `FAIL` 输出是通过的负例，最终以 unittest 汇总与进程退出码为准。

真实报告集中在 [results](results/)：7 份评估配对、4 份训练配对均为 PASS。Stair 本地保留含完整检查、分组统计的紧凑报告，逐回合明细位于各报告 `full_report` 指向的远端文件，原 hash 可核。原始 NPZ 和历史权重留在服务器隔离输出；Push 的少量数值副本用于本地离线首问题复现。

配对验收脚本分别是 `verify_stair_pair.py`、`verify_push_pair.py`、`verify_training_pair.py`，仅读取现成结果；生产重放入口在仓库 `scripts` 中，不导入 PLAN。`acceptance_cases.json`、`training_cases.json` 与总账保存实际命令。当前源码与 r2/r5/r6 的差分及可复用范围见 [源码审计](source_equivalence_audit.json)：采集/评估/训练热路径保持受测字节，r6 后仅改变 7 个回执消费及反馈文件，已由最终 CPU 与代际集成验收覆盖。没有将不同源码的整条运行笼统视为同一次验证。

新模块无 PLAN 执行依赖，隔离 GPU 快照也未包含 PLAN。已有 Push upstream 路径只是来源元数据；旧 Stair Phase0 历史接口仍有读取 PLAN 报告的能力，本轮未迁移该历史接口，不能声称全仓从来不存在这种引用。

后续 P3 可以使用这些已保存的数值建立有限的姿态/行为词表和默认作者工作集，再验证改进质量。当前结果不证明视觉诊断、高层行为解释、自动因果归因或 LLM 修改收益已经实现。
