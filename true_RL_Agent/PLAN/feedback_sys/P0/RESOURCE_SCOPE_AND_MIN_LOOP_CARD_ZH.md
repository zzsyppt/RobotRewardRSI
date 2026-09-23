# 资源范围与首个最小闭环

更新：2026-09-23，P0审计修订。机器预算见 [resource_limits.json](./resource_limits.json)。本卡确定下一阶段的目标和范围；实际采集字段、误差容限、样本选择在该阶段动工时结合源码确定。

## 资源与现有材料

项目硬预算保留 **500 GPU容量小时、US$120模型费用**。Phase 1分配0 GPUh/$0；Phase 2-min分配8 GPUh/$0。后续阶段开始前从剩余额度分配，不再叠加早期总额、日历时间、全局输入输出token和全局调用数等平行预算。实验的候选机会/重试上限仍用于保持公平；单次请求和采集器仍需实际容量控制。

费用以现有runner/provider回执累计，失败与重试一起记；GPU容量小时按完整外层运行时间×占用GPU数，模型费用按实际usage和登记单价。历史formal c6为45.926567 GPU容量小时、24个训练槽，其中包含失败和评估，不能直接当每个未来候选的训练单价。

初次P0的8次API计量约$0.030562、GPUh为0；用户审计另提供1次Gen8-A replay，外层约290秒（单GPU约0.0806h），属于复用证据，不能写成此次修复重新执行。本次审计修复只有离线分析与原件取回，无新增模型/训练/模拟器调用。

当前库存已有：

- formal c6：23条成功训练lane父子边、22个唯一候选权重、36份作者rationale。代际关系从ledger解析，Gen8-A的历史parent为官方flat-derived权重fca6a59c…。
- 最新Go2 Push force attempt_08：10个训练候选、10条父子边，Gen1-A和Gen4-B接受。该池和旧velocity D池不同。
- Go2 Stair v3：P3 reference 689/864成功；审计提供的Gen8-A新评估822/864。二者协议/cohort相同，可看描述差分；P3 reference 7b582643…不是Gen8-A历史parent，且两次执行源码不同。

工作前按现有资源/lease机制读取即时占用，复用权重，不复制整个checkpoint池。全量数组、缓冲寿命和压缩率在Phase 2首次capture测量，届时确定本地内存/磁盘上限；不能用粗略float数量替代RSS实测。

## Phase 1离线档（历史范围）

本段保留P0当时的最小范围。当前Phase 1已按 [完整实施卡](../P1/IMPLEMENTATION_PLAN_ZH.md) 升级为离线、live/replay、记忆及最终provider输入全范围；不以本段的离线验收代替当前Phase 1完成。

**目标：**现有公开投影能承载新增行为证据，身份、分母和实际作者输入一致。

**验收：**复用既有summary/public projection/journal；用实际parent/candidate公共事实生成T输入，核查scope、checkpoint、协议、来源和缺失状态。任务定义留在任务投影；共享层只负责传递与预算。已有边界回归能覆盖的路径复用它们，修复仅围绕出现的反例。完整live/replay/memory档在生产接入前完成。

## Phase 2-min：两个真实策略的过程采集

**目标：**解释禁触回合中姿态/支撑/受力如何变化，以及同条件成功回合和另一策略有何不同。

**首批选择：**至少一条真实parent–candidate边，建议formal原parent fca6a59c…与Gen8-A bcd8de3a…；两者在同一当前v3版本、完整public-dev cohort下重评。P3 reference可作为第三个现有对照。按事先登记的失败类别覆盖再扩到2–4策略；22个兼容候选是备选池，不要求全跑。首批结果只用于开发过程描述，不能据已知高分宣布新反馈提升策略。

**原parent启动身份已核：**只读服务器核验 `fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729` 的 `best.pt` 存在，6,882,293字节，文件hash完全匹配；训练attempt=`cae0d47c9b7c4877b9d1c0fab59f5fba`，`model_1999.pt`，196,608,000环境步，seed=1262835026，random init、无resume。训练binding=`c0112a87…` 与formal genesis一致，训练summary的manifest=`54cd180c…` 与取回manifest一致；初始policy hash也与formal相同。原件和远端路径见 [parent核验](./inventory_parent_remote_verification.json)，机器库存已绑定。该核验只证明历史权重/训练身份和可取得性；它尚无此次当前v3重评，Phase 2仍需在统一当前版本执行两策略的preflight与完整public-dev采集。没有复制PT或启动GPU。

**两份历史manifest明确不同：**

| 对象 | 历史训练manifest完整hash | 训练cell |
|---|---|---|
| 原parent fca6a59c… | `54cd180c9f39f65bffd71482ec8a87cda912d0e378e8b434d0c44c839a3ab1c2` | `stair.go2_12dof.unified_v1` |
| Gen8-A bcd8de3a… | `193fa9038a6f46539fb71a7e73edc44bdc024be2b899785394b8f82113c91f37` | `stair.go2_12dof.unified_v2` |

已逐字段比对 [parent manifest](./inventory_remote_raw/outputs/p5_go2_derived_reference_ef6826e/resolved_cell_manifest.json) 与 [formal manifest](./inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba/resolved_cell_manifest.json)：差异在cell/environment/binding版本标识、base/resolved task protocol、protocol state和eval runtime dynamics及其汇总hash。robot profile、asset closure、runtime adapter、动作/观测/传感器/策略I/O、training profile、train runtime dynamics和实际机器人绑定内容相同；`phase0_binding_facts.cell_key`仅版本名称不同。[v2协议](../../../configs/tasks/stair_go2_12dof_unified_protocol_v2.json) 明确以v1为evaluation-only successor：删除原base-clearance fall判据、将平台成功改为首次有效到达闭合，因此评估manifest应发生变化。两份训练回执的training binding均为 `c0112a87a7fb5750ef11d55a6bb50cfdeb0be662380c1387b2d8cab8c977de98`。库存现有检查核parent summary与它自己的manifest一致，不能解释成parent与candidate的历史manifest相等。

**尚未完成的P2启动前置：合法祖先准入。** [现有P6回放入口](../../../PLAN/2new_world/P6/stair_dev_regression.py) 检查完成训练预算、权重hash及training binding，保留原manifest，入口本身不要求旧manifest等于当前manifest。但下游 [SubprocessRunner](../../../src/true_rl_agent/runner/subprocess_runner.py) 的checkpoint检查只接纳“当前manifest或直接training predecessor manifest”；[v3协议](../../../configs/tasks/stair_go2_12dof_unified_protocol_v3.json) 的直接前驱是上述v2 manifest，而fca是v1。训练binding沿祖先链保持一致，不代表下游已允许v1直接跨到v3。因此Gen8-A已有v3 PASS不能替fca证明该路径已可执行；这是现有准入范围的待处理项，不是历史证据不可信或已证明权重不兼容。

P2启动时应沿现有协议/runner机制确认或补齐合法祖先兼容路径，核对真实权重与训练回执、已验证的successor链、policy架构和观测/动作/传感器/机器人绑定合同、训练origin/budget，以及此次统一v3的source/environment/evaluation binding/cohort。两策略通过各自合法准入后再做同版public-dev capture对照。不得重写历史manifest冒充v2、绕开checkpoint检查或简单放宽hash集合；也不新增“两个历史训练manifest必须相等”的错误门。P1不实现此P2准入变更，fca尚未开展当前v3回放。

**P1审计补充的P2设计边界：**

- 现有 generation 重放按 Python 字典键值语义比较回执重建的 `public_summary` 与 ledger 摘要，并非比较 JSON 字节。键顺序、缩进或不改变输出的重构不要求冻结；但改变旧回执重建出的字段或值会触发拒绝。P2先保留旧摘要的语义，用任务自有的公开过程视图承载新增描述。若确需改变已持久化摘要合同，须在实际需求出现时设计并验证明确的演化方案；本轮不提前实现投影版本或通用迁移框架。
- 旧数据的派生视图与新采集分开：前者只能使用已保存、可核对的源事实，记录源工件身份、派生逻辑和生成时点；旧回合没有保存的过程不能补推成观测。后者是同一权重在当前登记协议下的新 capture／新运行，产生自己的回执与绑定。在新回执发布前封存其过程工件并由任务证据登记来源、内容 hash 和覆盖范围；`task_evidence`、`task_evidence_hash` 和完整 phase result 都已参与回执校验，不能把后来工件回填旧回执，或把新回放当作历史原回合过程。独立工件的 public-dev 可见性和实际可得时点仍须通过任务投影处理；派生文件存在不代表已接入作者，也不自动取得接受权限。
- formal c6 与 attempt_08 的历史回执引用包含服务器绝对路径；guard 引用的 `to_dict()`（包括路径）也参与 manifest hash。P0式取回原件可支持本地分析，但不自动满足原路径绑定的生产重放。需要该保证时，沿原服务器路径只读重放并导出有来源的公开结果；不得改旧 ledger/ref 路径后声称身份未变。本轮不建设路径搬迁或 resolver 框架。新采集在新运行自己的输出位置创建绑定，不覆盖旧会话。
- 过程描述不得把运行记录“可用”写成“已执行／已测量所有字段”：现有 runtime `available` 表示存在至少一种已记录信息，包括声明或激活记录；计数 `null` 是未知，加载和声明不是执行证明。P2输出需保留各字段的实际覆盖与缺失，不能由状态标签推成零值。
- 新增逐代证据前，复用P1的真实历史保留基线，并在当前阶段会话上同时量出字节数、保留/省略代数及当前parent和待解释trial的覆盖。保持预算内不等于没有信息退化；需要调整默认工作集时，在阶段卡记录实际取舍，不默认扩大预算或复制完整测量表。

**验收：**

1. 每个策略完整覆盖既有864回合定义，包含成功、禁触及其他终局；capture on/off在启动卡确定的容差内保持主结果。记录outer/rollout耗时、峰值内存、实际文件大小和字段dtype。
2. 保存重算首个问题所需的带符号姿态、支撑/身体身份、事件边界和pre-reset状态；能重算已有终局与统计。新增词条不改变success/termination或reward ABI。
3. 可选保存现有15个接触body合力范数，定位哪个身体受力；base-only forbidden定义照旧。范数不识别接触面，必要方向向量不因压缩方便而丢弃。
4. 对照优先同cell，seed内配对或跨seed分层。已保存的P3 reference（7b582643…）在24个cell均有成功，只有2个seed×cell组合零成功；首批formal parent（fca6a59c…）的v3覆盖由新采集计算。未跨越锚点的失败保留终止前窗口；跨cell同阶段只作明确标记的辅助类比，不能当同条件配对。
5. 8 GPUh内完成首批及必要的capture对照；超出时收窄未启动的扩展。具体启动次数根据首轮实测确定，取消原6次启动的僵硬上限，不把30次或全22策略变成新硬前置。

## Phase 3-min：作者是否得到实际帮助

**目标：**新增过程证据帮助作者减少错误解释、提出更有依据的授权修改；同时纠正对既有事实或奖励实现的误读。

历史A输出基线见 [AUTHOR_EVIDENCE_BASELINE_ZH.md](./AUTHOR_EVIDENCE_BASELINE_ZH.md)。从可比较policy/parent/cell组合中登记12–24个开发案例和共同来源，覆盖至少两个过程问题；不能把同一session中的多个案例当独立训练谱系。

A和T都用同一新数据时点、相同作者/源码权限/预算，各案例各臂至少两次新生成。历史A只作现状素材，不能直接替代匹配实验的A。机器核对引用、数值与实现符号，1名人工按固定rubric隐藏臂名、打乱顺序评阅关键事实和修改建议，保留不确定项及单人评阅局限。事先登记T为腾出上下文删去的冗余内容。

通过门保持：身份/权限硬错误为0，事实支持率至少95%；至少两个登记类别有能影响修改判断的增量，授权和可检验性不退化；若走降成本路线，总费用至少降低20%且质量不退化。过程输入更长或措辞更生动不算通过。

## 后续边界

先完成0→1→2-min→3-min，再按价值建设查询、记忆和视觉。3-live仍为1个T session×2代×2lanes=4机会，只测闭环可用性；正式A/T收益由后续匹配搜索实验判断。既有accept/sealed边界、task-owned语义和固定任务成功标准不变。
