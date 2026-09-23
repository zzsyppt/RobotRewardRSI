# 反馈系统扩展研究与开源工程核查

日期：2026-09-22。对应计划：[FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.md)，v1.1。

**2026-09-23版本说明：**本页保存v1.1的调研及当时审查记录；其报告数量、阶段顺序和预算描述不是当前实施合同。当前计划已修订为v1.2，新增早期行为反馈价值门、明确预测裁决与阶段验收卡，并收窄了部分采集/统计推断。详见[逐条审计处理记录](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/AUDIT_RESPONSE_AND_PLAN_REVISION_2026-09-23_ZH.md)；[v1.1原计划](D:/humanoid_agent/true_RL_Agent/PLAN/feedback_sys/archive/FEEDBACK_SYSTEM_UPGRADE_PLAN_ZH.v1.1.md)保留供对照。研究方法依据继续有效，工程选型和当前provider能力仍须按对应阶段重新核验。

本记录解释新增设计从哪里来、实际代码提供了什么、哪些做法没有采用。不是软件选型采购单，也不代表已经复现论文效果。

## 1. 调研范围与结论

本轮围绕四条线并行检索：机器人失败与时序视觉、轨迹/媒体工程、记忆与上下文、实验评测与持久执行。19次Exa搜索请求的`numResults`合计125，**该数为请求的检索结果量，包含重复，不等于精读125篇论文**。随后直接读取原论文、官方文档、仓库树、许可和关键源码，重点核查下文22组来源；另复核已有AHA与Video2Reward引用。检索命中不等于全部可用，也不声称穷尽所有相关研究。

工程核查采用三个层次：有方法论文；有真实实现文件；实现语义是否适合本项目。只有README、演示视频或仓库名称不足以确认可复用。这里的“已核查实现”表示读过源码，**没有安装或运行外部项目，没有执行模型API评测、统计仿真或RL训练**。快照日期以本页为准，实施时仍须锁版本。

结论是保留原计划主架构，同时补强七个位置：

1. 完整低维时序负责事实，事件前后短窗口负责细节；媒体和数值用同一episode及control step关联。
2. 用数值变化引导选帧，同时保留覆盖帧；避免只看预设失败标签而漏掉陌生行为。
3. 把身体部位、动作方向、起止时间及并行动作变成可核查对象，独立检验时间定位。
4. 记忆区分原始记录、确定性事件、模型笔记；同时保存事件发生时间与证据首次可用时间。
5. 默认给短摘要和目录，必要时回查；视觉与代码修改使用分开的工作集。
6. 增加只读人工核查视图，并让调用、重试、恢复和费用全部可追踪。
7. 冻结案例选型与独立搜索确认分开；查不到、看错、时间错、跨源错和没收益分别报告。

不新增Phase，不整体迁移数据库或Agent框架。计划仍为0–11：先可信采集，再诊断/视觉/记忆，最后验证诊断质量和真实搜索收益。

## 2. 轨迹、媒体与人工核查：4组工程来源

### A1. LeRobot Dataset v3：用索引关联数值和视频

- 依据：[官方数据格式文档](https://huggingface.co/docs/lerobot/main/en/lerobot-dataset-v3)、[作者仓库](https://github.com/huggingface/lerobot)。核查SHA `7af6936589215ea0d2457de43530bc6322c7cea6`，代码Apache-2.0。
- 真实实现：[dataset_reader.py](https://github.com/huggingface/lerobot/blob/7af6936589215ea0d2457de43530bc6322c7cea6/src/lerobot/datasets/dataset_reader.py)、[dataset_writer.py](https://github.com/huggingface/lerobot/blob/7af6936589215ea0d2457de43530bc6322c7cea6/src/lerobot/datasets/dataset_writer.py)。低维Parquet、视频和episode元数据分别保存；读者用边界及时间戳找到同源数据，writer负责异步写入结束与finalize。
- 采用：Phase 2/6的episode/时间索引、分片、异步写入完成后发布manifest；不根据文件名猜对应关系。
- 不照搬：reader会把越界请求夹到episode端点，同时给出`is_pad`。训练数据加载可这样做，但本项目不能把填充点当真实持续姿态，更不能用于连续dwell。也无需把现有所有数据迁到LeRobot或上传Hub。
- 验收：跨episode、短回合、终止边界、异步未完成、损坏文件、填充mask均有反例；原件与索引可互相核对。

### A2. Isaac Lab Recorder：录制必须理解reset生命周期

- 依据：[官方manager文档](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.managers.html)。核查的是上游`main`快照 `b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8`，代码BSD-3-Clause；不代表项目实际安装版本。
- 真实实现：[manager_based_rl_env.py](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/envs/manager_based_rl_env.py)、[recorder_manager.py](https://github.com/isaac-sim/IsaacLab/blob/b0542fe2d45bf91c4e1d9ef6952b9c709c80b4e8/source/isaaclab/isaaclab/managers/recorder_manager.py)。终止/奖励计算、post-step记录、pre-reset、reset、post-reset以及最终返回的观测有明确顺序；还存在success-only导出选项。
- 采用：Phase 0冻结本地版本，Phase 2用调用顺序哨兵和真实边界案例验证terminal state；失败诊断必须覆盖失败记录。
- 不照搬：不能只看环境`step()`返回值就认定它仍是失败前状态。本地已有task-owned pre-reset路径，应复用和核验，不把已有修复重新描述成缺失功能。

### A3. Rerun：曲线、图像与事件共用时间轴

- 依据：[时间轴文档](https://rerun.io/docs/concepts/logging-and-ingestion/timelines)、[Blueprint文档](https://docs.rerun.io/dev/concepts/visualization/blueprints/)。SHA `2def8f3239ba22515b20cfedd4c18a4d1ae2b2e2`，Apache-2.0。
- 真实实现：[time.py](https://github.com/rerun-io/rerun/blob/2def8f3239ba22515b20cfedd4c18a4d1ae2b2e2/rerun_py/rerun_sdk/rerun/time.py)区分sequence、duration、timestamp；记录数据与查看布局分离。
- 采用：Phase 3的只读数值核查视图，Phase 6加入原生帧。用户点击“左脚支撑时躯干倾角持续增加”，能跳到同一时间的接触、倾角和图像。可以只做薄导出层，甚至沿用现有HTML工具。
- 边界：它是查看与QA工具，不是自动诊断器。使用control step/simulation time作事件轴，不能把日志到达时间当物理时间；人工注释与原始值分开。

### A4. MCAP：按时间和主题读取已有证据

- 依据：[格式规范](https://mcap.dev/spec)、[官方仓库](https://github.com/foxglove/mcap)。SHA `1686e8a927031680786d90d3a48606ab1c5db937`，MIT。
- 真实实现：[reader.py](https://github.com/foxglove/mcap/blob/1686e8a927031680786d90d3a48606ab1c5db937/python/mcap/mcap/reader.py)通过chunk/message索引进行topic和时间范围读取。
- 采用：Phase 2/4的schema、索引和窗口读取思想，使查询局部证据无需扫描全部历史。
- 边界：首版NPZ/Parquet＋JSON仍可满足要求；不引入ROS依赖，也不因为已有文件格式就认为数据身份和时间自然正确。

## 3. 机器人诊断与时序视觉：6组研究和实现

| 来源、实现与快照 | 采用内容与Phase | 不能外推的地方 |
|---|---|---|
| **B1 REFLECT，CoRL 2023**：[论文](https://arxiv.org/html/2306.15724)、[exp.py](https://github.com/columbia-ai-robotics/reflect/blob/0ac7458735fba606151df851491124486dc8b489/main/exp.py)。SHA `0ac7458`，MIT | Phase 3–5/8：回合结果、事件目录、局部证据三级摘要和按需回查 | 实验为AI2THOR/UR5e等场景；模拟感知还使用深度、instance masks及环境metadata。不要把特权输入当普通RGB能力，也不照搬完整scene graph |
| **B2 ProTracer，2026-09-18预印本**：[论文](https://arxiv.org/html/2609.21369)、[changepoint.py](https://github.com/Chang-AIML/ProTracer/blob/63dafa9b703be0347d80f63d049304269f8d000f/code/protracer/changepoint.py)、[narrative.py](https://github.com/Chang-AIML/ProTracer/blob/63dafa9b703be0347d80f63d049304269f8d000f/code/protracer/narrative.py)。SHA `63dafa9`，未发现明确LICENSE | Phase 3/6/9：本体信号引导关键窗口，方向、净运动与累计运动分开，独立检验失败相关起点定位 | 很新的预印本，机械臂/夹爪阈值不直接用于腿足；事后定位不等于在线报警；只借设计，不直接复制许可不明确代码 |
| **B3 KITE，ICRA 2026**：[论文](https://arxiv.org/html/2604.07034)、[keyframes.py](https://github.com/m80hz/kite/blob/afead048974e5267087ef72ac2c54950995d6c5c/kite/video/keyframes.py)、[contact.py](https://github.com/m80hz/kite/blob/afead048974e5267087ef72ac2c54950995d6c5c/kite/perception/contact.py)。SHA `afead04`，未发现明确LICENSE | Phase 6/7：带时间、机器人形态、空间参照和必要ROI的紧凑证据包 | 框重叠/距离只是接触启发式；非度量俯视图不证明力或距离。已有仿真几何应优先使用，不额外引入单目深度栈 |
| **B4 RAPT，2026预印本**：[论文](https://arxiv.org/html/2602.01515v1)、[saliency.py](https://github.com/humphreymunn/RAPT/blob/d745668ffdb0e26877b1de4c9b2126af7bda5b03/rapt/saliency.py)、[诊断prompt](https://github.com/humphreymunn/RAPT/blob/d745668ffdb0e26877b1de4c9b2126af7bda5b03/rapt/diagnosis/prompt.py)。SHA `d745668`，BSD-3-Clause | Phase 3/7：时间×身体部位的多信号证据；以后若做学习式监视，另标detector_attribution | 有G1、Isaac Lab和真机脚本，但检测器需要训练/校准。其IG解释的是监视模型末步NLL，不等于策略敏感度或物理根因；首版不训练此网络 |
| **B5 ActionFormer，ECCV 2022**：[论文](https://arxiv.org/abs/2202.07925)、[metrics.py](https://github.com/happyharrycn/actionformer_release/blob/61ea7eb9308a568b0cf45e3804830836e30061de/libs/utils/metrics.py)。SHA `61ea7eb`，MIT | Phase 9：动作区间的tIoU、precision/recall、边界误差与一对一匹配 | 借评价方法，不借人类动作模型充当机器人能力；点事件用时间容差，不能硬套区间IoU |
| **B6 Q-Frame，ICCV 2025**：[论文](https://arxiv.org/html/2506.22139)、[GPT-4o接入](https://github.com/xiaomi-research/q-frame/blob/1b08494c1d91237d3bfdddaff561ff3fd77da2a6/lmms_eval/models/gpt4o_w_qframe.py)、[Qwen多分辨率接入](https://github.com/xiaomi-research/q-frame/blob/1b08494c1d91237d3bfdddaff561ff3fd77da2a6/lmms_eval/models/qwen2_vl_w_qframe.py)。SHA `1b08494`，Apache-2.0 | Phase 6/9：相同媒体池和实际成本下比较选帧/分辨率，首先保证事件前后覆盖 | 普通长视频问答收益不证明能看到短暂碰撞；第一版不引入Long-CLIP和新的语义检索模型 |

源码核查直接转成了以下拒绝规则：

- KITE在找不到主要对象时可返回`CONTACT:STABLE`；便利抽帧失败可返回黑图，FPS可回退30。本项目一律显式缺失，不能把工程fallback解释成行为事实。
- Q-Frame含Gumbel随机步骤及异常时`random.sample`。因此training-free不等于可重放；必须确定性或显式冻结随机性及fallback回执。
- ProTracer的[pipeline.py](https://github.com/Chang-AIML/ProTracer/blob/63dafa9b703be0347d80f63d049304269f8d000f/code/protracer/pipeline.py)实际包含caption→diagnosis两轮及解析重试。计划首版坚持单基础分析调用与统一重试预算，不照搬调用链。
- ProTracer的[evaluation.py](https://github.com/Chang-AIML/ProTracer/blob/63dafa9b703be0347d80f63d049304269f8d000f/code/protracer/evaluation.py)中onset误差只覆盖可用预测。我们同时报告全部失败中的漏检和不可定位比例，防止只在容易定位的少数样本上报好成绩。
- RAPT的模型归因、KITE的接触启发式、VLM的解释，都必须保留各自来源与不确定性；不能因输出有结构就被提升为物理测量。

已有引用也作了更新：[AHA仓库](https://github.com/NVlabs/AHA)确有FailGen与评价代码，但[根许可](https://github.com/NVlabs/AHA/blob/main/LICENSE)含研究/评估非商业限制；[Video2Reward作者仓库](https://github.com/Alvin-Zeng/Video2Reward)本轮快照HEAD `9989639`仅两份文件，未提供可直接接入的实现源码。二者的方法启发与工程可用性分别记录。

## 4. 记忆和上下文：6组研究与实际实现

### C1. MemGPT / Letta：外部档案与有限核心工作集

[MemGPT论文](https://arxiv.org/abs/2310.08560)提供外存和工作集思路。当前实现核查于[letta-code](https://github.com/letta-ai/letta-code)，旧Letta server已归档；SHA `f5c5bbce6e9394b30c626909315c2b3acc665672`，Apache-2.0。[memory-format.ts](https://github.com/letta-ai/letta-code/blob/f5c5bbce6e9394b30c626909315c2b3acc665672/src/agent/memory-format.ts)、[memory-constraints.ts](https://github.com/letta-ai/letta-code/blob/f5c5bbce6e9394b30c626909315c2b3acc665672/src/agent/memory-constraints.ts)包含核心区、分层索引与大小/深度限制。

Phase 5借鉴“短核心＋可回读原文”，但保持现有授权链，不开放自改系统指令。文件Git历史能证明写入过程，不能证明内容为真；字符限制也不能代替实际token和图片预算。

### C2. Mem0：归属、范围过滤与历史，模型笔记不等于事实

[论文](https://arxiv.org/html/2504.19413v1)、[仓库](https://github.com/mem0ai/mem0)，SHA `a39a802bbc93e85b820078cd3c4dbaf53af25dbe`，Apache-2.0。核查[memory/main.py](https://github.com/mem0ai/mem0/blob/a39a802bbc93e85b820078cd3c4dbaf53af25dbe/mem0/memory/main.py)、[storage.py](https://github.com/mem0ai/mem0/blob/a39a802bbc93e85b820078cd3c4dbaf53af25dbe/mem0/memory/storage.py)以及[抽取prompt](https://github.com/mem0ai/mem0/blob/a39a802bbc93e85b820078cd3c4dbaf53af25dbe/mem0/configs/prompts.py)。当前自动抽取为ADD-only，明确允许assistant建议/计划进入记录并标记归属。

Phase 5借鉴追加、归属和scope过滤。我们的assistant建议只能是model note/hypothesis，不能成为observed fact。论文LoCoMo只有10条长对话，且其报告排除了adversarial/unanswerable类；大量问题不是大量独立历史。当前README高分包含托管平台专有优化，不能视为OSS SDK复现成绩。

### C3. Graphiti / Zep：事件时间与获知时间分开

[论文](https://arxiv.org/html/2501.13956v1)、[Graphiti仓库](https://github.com/getzep/graphiti)，SHA `16cdf7045378c8d53ae01f94e2fa60d238cb0f68`，Apache-2.0。[edges.py](https://github.com/getzep/graphiti/blob/16cdf7045378c8d53ae01f94e2fa60d238cb0f68/graphiti_core/edges.py)保存来源episodes及有效/失效时间，[search_filters.py](https://github.com/getzep/graphiti/blob/16cdf7045378c8d53ae01f94e2fa60d238cb0f68/graphiti_core/search/search_filters.py)提供时间过滤。

Phase 4/5采用双时间和失效关联，新增例子：G3才补入的G1诊断不能帮G1决策。索引仍由ledger重建，不引入图数据库。抽取和矛盾识别本身仍使用LLM，有来源不等于正确；当前Zep托管产品也不能与Graphiti OSS混称。

### C4. ACE：小步更新经验，不反复重写整份摘要

[论文](https://arxiv.org/html/2510.04618v1)、[官方仓库](https://github.com/ace-agent/ace)，SHA `82709de050e1db6e6ef2f07bcb0393560b94992a`，Apache-2.0。[curator.py](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/ace/core/curator.py)有局部操作与diff；但[playbook_utils.py](https://github.com/ace-agent/ace/blob/82709de050e1db6e6ef2f07bcb0393560b94992a/playbook_utils.py)实际实现ADD，UPDATE/MERGE/DELETE仍列TODO。

Phase 5只借鉴稳定ID、增量、反证与来源保留，不声称可直接取得完整curation能力。helpful/harmful是反思标签，不是因果效果量。ACE式答题后学习必须另设在线协议，不能把选择集答案写入下一题记忆。

### C5. LongMemEval：分开“找回”与“读懂”

[ICLR 2025论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf)、[仓库](https://github.com/xiaowu0162/LongMemEval)，SHA `9e0b455f4ef0e2ab8f2e582289761153549043fc`，代码MIT。

Phase 5/9借鉴indexing、retrieval、reading分开评测；覆盖更新、时间、跨代与无法回答问题。增加同预算oracle-evidence探针：给正确证据仍答错，才是阅读/推理问题。[evaluate_qa.py](https://github.com/xiaowu0162/LongMemEval/blob/9e0b455f4ef0e2ab8f2e582289761153549043fc/src/evaluation/evaluate_qa.py)使用LLM judge并允许部分日期误差，不能直接用于control-step或dwell精度。代码许可不能代替数据许可。

### C6. LongMemEval-V2 / AgentRunbook：原始状态、事件与策略笔记分池

[2026-05论文](https://arxiv.org/abs/2605.12493)、[官方仓库](https://github.com/xiaowu0162/LongMemEval-V2)，SHA `2cc8c540bdb87fe6761629b585e727e1c4704520`，代码Apache-2.0。[agentrunbook_r.py](https://github.com/xiaowu0162/LongMemEval-V2/blob/2cc8c540bdb87fe6761629b585e727e1c4704520/memory_modules/agentrunbook_r.py)体现三类信息；[harness.py](https://github.com/xiaowu0162/LongMemEval-V2/blob/2cc8c540bdb87fe6761629b585e727e1c4704520/evaluation/harness.py)将题目和可选图片交给memory，并独立校验证据和预算，不暴露gold、题型和原始question ID。

Phase 5采用类型隔离，Phase 9采用盲接口与冻结快照。该基准451问来自多模态web-agent历史，测经验问答而非再次执行机器人的成功率。不借AgentRunbook-C的开放文件系统，也不因为检索更智能就取消延迟与调用限制。

## 5. 实验与恢复：6组依据

### D1. rliable：独立样本到底是什么

[NeurIPS 2021论文](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)、[library.py](https://github.com/google-research/rliable/blob/3ccd9f4dea577a04d3d2b557f259aac08badbd81/rliable/library.py)，SHA `3ccd9f4dea577a04d3d2b557f259aac08badbd81`，Apache-2.0，仓库已归档。

Phase 11以配对search session计算差值。源码`get_interval_estimates((A,B), ...)`会走`StratifiedIndependentBootstrap`，不能如此传入本项目两arm后假设保留了配对。先算pair差值再重采样；episodes、frames和同候选重复训练都不扩大搜索样本量。借离线分析方法，不改变既定均值终点为IQM。

### D2. 小样本统计设计：n=8只是预算，不是可靠性证明

[How Many Random Seeds?](https://arxiv.org/pdf/1806.08295v2.pdf)、[Empirical Design in Reinforcement Learning](https://jmlr.org/papers/volume25/23-0183/23-0183.pdf)讨论随机性、区间、功效和设计；前者[参考代码](https://github.com/flowersteam/rl-difference-testing/blob/bf7cdb8d2cddbea5d5ee9d06de2d51efdf758c00/test_RL_difference.py)使用独立样本Welch检验，未发现LICENSE，不能直接替代本项目配对分析。

Phase 0登记分布与效应假设，在首个正式模型调用、候选生成或训练启动之前，完成合理情景下覆盖、假阳性和功效检查并冻结研究性质；不照抄“必须多少seed”的万能数字。t与bootstrap一致仍可能共同欠覆盖；小样本正态性检验不显著也不证明正态。CPU仿真是条件于所设情景的检查，不是分布无关证明。若预算支持不了确认性研究，应提前承认探索性质或延期。

### D3. 模型选择偏差：选择集胜者需要独立确认

[Cawley & Talbot，JMLR 2010](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)说明有限选择集本身会被过拟合。Phase 9的48案例和20%门槛用于工程选择，不能同时当被选方案的无偏泛化成绩；多重比较修正也不能消除这种胜者偏差。

若另声称R/V各自有效，需预注册比较族和相应确认方法。Phase 11只对一个事前冻结方案进行独立主要比较，不需要因为前面有5组就机械把它处罚5倍。

### D4. Inspect：计划清单与结果清单必须闭合

[eval-sets文档](https://inspect.aisi.org.uk/eval-sets.html)、[evalset.py](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0248a13837b8d82b8d7ebcd98478c921561616d4/src/inspect_ai/_eval/evalset.py)，SHA `0248a13837b8d82b8d7ebcd98478c921561616d4`，MIT。

Phase 9借鉴样本manifest与完成日志核对：case/config/evidence/model身份匹配才可复用；缺项、重复和旧缓存全部显式报告。运行success代表完成，不代表质量通过。不把Inspect引入为第二个主调度器。

### D5. Inspect日志与默认重试：失败尝试也有成本

同一固定版本的[日志结构](https://github.com/UKGovernmentBEIS/inspect_ai/blob/0248a13837b8d82b8d7ebcd98478c921561616d4/src/inspect_ai/log/_log.py)保存错误、usage和时间等，但`eval_set`默认最多10次重试，并默认清理成功重试之前的失败日志。这些默认值不适合本项目。

Phase 8/9复用journal记录每个实际调用；计划报告数、视觉/修复子调用数、训练机会数和实际费用分别核对。失败日志追加保留；超时费用不明时保守占用额度；provider seed不能承诺完全复现。

### D6. LangGraph：恢复点不等于外部API只执行一次

[functional API文档](https://docs.langchain.com/oss/python/langgraph/functional-api)、[time travel文档](https://docs.langchain.com/oss/python/langgraph/use-time-travel)、[重试源码](https://github.com/langchain-ai/langgraph/blob/49cce0ca852be4cfb567a1cbe0e511ff325a1682/libs/langgraph/langgraph/pregel/_retry.py)，SHA `49cce0ca852be4cfb567a1cbe0e511ff325a1682`，MIT。

已完成任务可以缓存恢复，但开始而未完成的任务可能重跑；本地清除writes不会撤销远端API。Phase 8将逻辑operation与实际attempt分开，支持`outcome_unknown`，在发送前、远端成功未持久化、已收到未晋升、已晋升未返回四个切点注入故障。只读回放不得隐式发新请求。采用边界设计，继续使用现有authoring journal。

## 6. 本版计划实际修改与验收落点

| 修改 | 进入的Phase | 核心验收 |
|---|---|---|
| 同源索引、reset顺序、填充mask、完成发布 | 0/2/6 | terminal不串episode；不完整文件不能complete；缺失不充当稳定或零 |
| 事件驱动选帧与覆盖帧、身体×时间描述 | 3/6/7 | 同预算可重放；净运动/累计运动区分；事件前后及并行动作可核查 |
| 只读同步证据视图 | 3/6/9 | 点击断言回到原区间；盲评隐藏配置；注释不改原件 |
| 三层view、双时间、scope缓存 | 4 | 迟到证据不提前进入回放；跨session不串源；跨层级事实一致 |
| 三类记忆、局部幂等追加、反证可见 | 5/8 | 三轮压缩不把假说变事实；旧结论及来源保留；head冲突不静默换parent |
| 逻辑操作和实际调用分开 | 8/10/11 | 四个崩溃切点不重复晋升、不重置预算；未知费用不视为免费 |
| 独立时间定位、检索oracle及盲接口 | 9 | tIoU/点事件容差、一对一匹配、漏检分母；答案不污染后续记忆 |
| 配对分析、统计可行性、故障分类与总账 | 0/10/11 | session为单位；正式前设计检查；失败不因效果改名；资源可重建 |

本版Phase 9新增12个开发问题的实际检索/oracle配对探针，计划报告由288增至312；CORE-only由192增至216。600/300的provider总调用上限暂不增加，所有子调用在其中核算，Phase 0按实际模型冻结可行预算。Phase 10仍16个候选机会、最多18次实际训练启动；Phase 11仍96个候选机会、最多104次实际启动。研究引用不构成暗中增加实验次数的许可。

## 7. 暂缓项目与证据边界

- 不引入训练式异常检测器、专用失败VLM、自动因果图、通用动作识别网络或完整scene graph；这些各自需要数据、校准和新的成本验证。
- 不将Letta、Mem0、Graphiti、ACE、Inspect、LangGraph整体接入现有主循环。先使用现有ledger、index、builder、journal实现必要机制，避免多份状态和真值。
- 不以公开视频问答、操作任务或web-agent成绩证明G1/Go2 Stair/Push收益；新预印本用于提出待验证方案，不称成熟共识。
- 未明确许可的代码只参考方法；AHA的使用限制单列。引入代码、数据或模型权重前分别核验许可，根目录代码许可不能代表全部资产。
- 本轮没有验证论文性能能在本项目复现。设计可信度来自事实边界和可证伪验收，最终效果仍取决于计划中的真实数据、诊断对照和独立搜索实验。

## 8. 计划复核记录

本节记录文档审查，不等同任何Phase实施验收。已完成两轮分方向交叉审阅及主审复核，结果如下：

- 机器人/视觉方向：修正Phase 6对事件定义的依赖、数值与媒体独立complete、原始池确定可评分母、补查和重试共用累计发图预算；复核B1–B6与固定源码一致。
- 记忆/上下文方向：补齐oracle的相同权限/时间/证据池/reader/预算，隔离同case不同arm的全部派生产物，允许正常并发完成乱序；复核C1–C6与源码一致。
- 评测/恢复方向：质量单位与实际尝试分开，补查开关在选择集之前冻结，统计检查与最终manifest在首个正式操作之前锁定；不再把“正式解封前”当作可以先运行后登记。
- 机械核对：Phase标题恰为0–11；报告数312/216、训练启动上限18/104算术一致；29处本地链接均存在，Markdown代码围栏成对，表格行完整，无冲突标记或替换乱码。
- 链接核对：记录中36个GitHub源码/许可链接通过在线读取检查，均返回HTTP 200；机器人方向另核对6个固定提交。来源状态可随时间变化，实施仍需重新锁版本。

以上范围内未发现仍需修正的实质性矛盾。本轮只修改计划与本研究记录；没有改运行代码、启动实验或把任何Phase标为已经通过。研究支持设计值得验证，不保证实现后必然增益。
