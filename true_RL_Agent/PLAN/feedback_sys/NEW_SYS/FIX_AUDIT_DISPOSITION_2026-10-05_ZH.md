# 轨迹翻译器审计处置与 CPU 收口（本轮有限验收完成，待用户审计）

本轮源码与测试的冻结验收基线为 `ac612244ca720418759d82ff99fb11b2a4069b2d`，该基线已推送至 `codex/trajectory-translator-upgrade`。本轮有限 CPU 验收及最终日志/文档只读复核已完成，源码与测试未再修改；本说明记录修复及未完成边界，不表示训练、真实仿真或模型端验收完成。

已全文重读 [修复情况审计 F](FIX_AUDIT_2026-10-04_ZH.md) 和 [轨迹翻译器修复情况审计 I](IMPLEMENTATION_AUDIT_FIX_REVIEW_2026-10-04_ZH.md)，并回溯 [原始审计 O](IMPLEMENTATION_AUDIT_2026-10-03_ZH.md) §3.11 的 18 项及 [独立审计](INDEPENDENT_IMPLEMENTATION_AUDIT_2026-10-04_ZH.md) 的 D1–D12。F/I 审计的代码基线是 `cc36e0f7`，其“现在”不等于本文件的当前状态。四份原审计及其附件保持原样和未跟踪状态，链接供本机复核；首次行为验收未通过的记录也未覆盖。

## 1. 当前实现及验收口径

作者现在收到总体运动、实际结局与末窗过程、条件集中、各足的接触/世界及身体相对点速/相对位置/合格局部周期，以及有资格时的名义关节与地形事实。固定开发案例槽不再是资格要求。选择依据是覆盖、占用和差异；来源哈希负责内部绑定，不决定说哪件事。Stair 保持作者行为部分 10240 字节上限；Push 保持 12288 字节上限，按实际作者投递结构计算。

本轮主要提交及可重查证据见 [阶段记录](CPU_BEHAVIOR_SELECTION_REVIEW_2026-10-04_ZH.md)：`329a155d` 修源选择/差分/预算和相关回归，`880e2e03` 隔离 Stair 纯描述异常，`15c469c1` 删除算子废弃记录包装，`2dfbf3e0` 只聚合消费量，`f933e577` 让 Push 单次测量，`44384c27` 压缩 Push 作者视图并补首推过程，`c29600e9` 隔离 Push 纯描述异常并删除退役 P0 生成器及测试，`ac612244` 修复 CPU 摄像头夹具接口。此前 A/B 的源头修复在 `f37f9fa4`、`38422bb4`、`882152f9`、`74248972`、`0fb4903d` 等提交中；本次没有重写旧阶段的验证历史。

验收先由原数组确定少数必说事实和反例，再检查最终作者文字。数值自回放只证明保存/投影一致，不能充当独立真值。审计的 `ground_truth` 是粗规则材料，不作为物理标签权威；合成输入只要求静止、平移、原地腿动、翻倒、高举单足等描述可区分，不要求猜失败原因。

## 2. 原始 18 项及 D1–D12 的处置

| 对应项 | 当前判断、必要性和源头处置 | 验证与保留边界 |
|---|---|---|
| O1；I§4.5：足活动/速度通道 | 描述层已改为相邻有效位置差分，分别计算 world 和 body-relative；称“记录点速/活动”，不称迈步。阈值与文字取同一事实/配置。 | 原数组 m50/m300 与静止/平移/腿动反例可区分；改录制速度通道不应改变这类位置运动句。奖励输入仍取原速度字段，见 §4，未宣布其物理语义已修复。 |
| O2、D3：接触组合写死 | 按实体数和实际非零频率排序，取主要组合，去掉 0/11 特例。 | K2、不同实体数 CPU 输入覆盖；不是新机器人生产资格证明。 |
| O3、D9：名义资格只认三档案 | 新采集凭显式坐标/实体/地形契约获得资格；缺元数据则不输出相应事实。 | 历史三档案资格仍保留，未扩名单。m50/m300 缺旧元数据仍不能报名义关节/地形；新契约验证不等于执行过新 GPU 采集。 |
| O4：资格依赖工作区换行哈希 | `74248972` 改为历史胶囊语义绑定，移除工作区源码字节相等要求。 | CRLF/LF 不改变资格；历史 importer 字节及运行时表面仍未证明，不能升级成 runtime 几何。 |
| O5：预算/描述错误毁掉评估 | 超预算、能力不支持已有明确降级；`880e2e03` 再将 Stair 测量/聚合/成句的纯计算异常降为描述不可用，保留科学评估结果。 | 原始身份、块损坏、I/O 和资格矛盾仍报错；发生描述异常后继续排空原始迭代，后续真实损坏不能被吞。最后收口已给 Push trajectory 主线加入同原则隔离，见 §3；不是 catch 整个 finalizer。 |
| O6、D8：队列阴性退化 | 初始历史不足为不适用；内部缺测为未知；presence 分已见阳性、完整记录未见、未知三态。 | 真实低进度与有限性/缺步边界测试；不再把开头不足窗计为全回合未知。 |
| O7、D4：固定 1N | 测量、周期和定义句共用任务接触阈值。 | 非 1N 契约输入可运行，未另造文字阈值。 |
| O8：Push 固定 20 槽 | 删除固定槽数量资格，按实际条件/结局成卡；`44384c27` 加入真实首推过程并删空见证占位要求。 | 真实 K4 两协议各 192 回合及不同分层回归；仍有必要总体/定义等结构，不以任意句数代表行为完整。 |
| O9、D1/D2：CLI 漂移与红测试 | CLI 默认回到同一配置源；旧地形链及其编码缺陷随删除消失。`329a155d` 修复陈旧断言、空父代/身份夹具等五项相关失败，删除已无生产入口的旧 retrigger 测试。 | 摄像头 CPU 夹具的接口漂移另由 `ac612244` 修复；最终回归及明确未测项目见 §6。现存纯 ASCII 文件的默认编码读取是小项，未因本轮扩大修改。 |
| O10/O11：循环验收、旧脚本无效 | 旧自称“真值核查”的重复生成链和 P1a 脚本随替代链删除；新增可重跑的 `scripts/check_stair_behavior_delivery.py`，原数组预期独立于成句器。 | R4/K1/m50/m300 必说事实检查及一个渲染破坏反例能发现错误；Push 独立原数组预期和高度列破坏检查。生产 replay 仍只保证一致性，未宣传为外部真值。 |
| O12、D5：关节按孤立极值选 | 按全队列有效读数的下侧/上侧“5%带内或界外”占用排序；`329a155d` 将界外计入、按占用顺序给后续句分配预算。 | R4/K1 选择高占用关节；全队列分母、上下侧、界外比例和有符号余量分列。没有改称关节锁死或物理因果。 |
| O13：名义越界/精度 | 容差只处理恰在记录精度量化边界的值；真实差值保留。带内和界外既合计“贴近该侧”的占用，又分报界外，避免全解释为浮点。 | 不任意吞掉 0.0856rad 真差，也不按审计建议把未知求解器穿透全部吸收。名义界不等于有效运行时限位。 |
| O14、D10：父代空桩 | 实现显式绑定及可比域检查；`329a155d` 用绝对物理尺度与显示精度选差分，覆盖总体结局/足参与，而非零分母相对增幅。 | 相同条件、末窗还需相同结局及完整窗。R4/K1 的域不一致时明确不可配对；m50→m300 仅 CPU 配对夹具，不声称历史父子。Push 单策略卡明确“未计算父代配对差分”。 |
| O15、D6/D7：哈希决定内容 | 语义优先级、群体频率/差异和稳定实体顺序决定内容；内部 ID 改变不会任意换事实。 | 更换来源哈希、重排回合回归；同信息并列保留确定性顺序，不宣称获得唯一最优摘要。 |
| O16：Euler 环绕 | 用身体上轴倾角、前轴仰角与四元数连续短弧累计，不用全局 unwrap 掩盖奇异。 | “回合最大倾角中位数”和旋转路径量已明确；仍未量化完整翻倒侧向、净转向或横移方向，不把路径长度称净 yaw。 |
| O17：低进度/确认时刻 | 低正向进度允许倒退，另报净进度、反向累计和有向末速；确认点不冒充区间开始，差分归后端且不跨缺步。 | 这是定义澄清加方向事实，不为迎合审计将倒退排除。未由记录点推断连续物理事件发生时刻。 |
| O18：重复属性/关系叉积 | 删除旧逐区间账本及无条件关系叉积；`15c469c1` 再删不消费的证据包装，`2dfbf3e0` 只按作者/比较实际需求聚合。 | 原数组 summary 消费值及最终作者文字前后相等；不再生成巨大无人消费关系。 |
| D11：卡片与编辑对象不一致 | `329a155d` 由 `current_best` 的 anchor/checkpoint、已接受 trial 及 reference 的 subject 身份选择主卡，其他卡明示为次要来源。 | 长 generation、reference、G2/G3 最终请求复核；不再以“最后 lane”冒充编辑基准，自比较也不冒充父代改进。 |
| D12：Push 过时投递文案 | 旧未接线声明及其补充说明均删除，模板版本明确变化。 | 正式投递已核；最后收口核实 CLI 私有 manifest 的 `integration` 无读取者，删除其过时值 `offline_only_P2b2_pending`，保留离线 API。 |

## 3. 修复复审新增问题和复杂度小项

| 复审条目 | 当前处置与依据 | 未做部分及原因 |
|---|---|---|
| F§6.3、I§4.1：差分选句低价值 | `329a155d` 使用 10pp、0.5m、0.1m/s、10° 等绝对尺度，先过滤显示精度以下差异；来源选择与配对分别校验。 | 不是普适价值函数；不能从跨域原数相减推出父代改进。 |
| F§6.4、I§4.2：条件样本数偏置 | 以观测数减同结局全队列率的期望数选择集中条件，同时报 k/条件总数及总体基线，保留边界并列。 | 不声称条件有因果效应；小样本仍明确分母。 |
| F§6.4：单回合结局占预算 | 保留实际结局覆盖，单回合结局仍是一条必投句；来源、差分、次要卡和主卡纳入同一作者预算，其他可选细节按信息价值裁剪。 | 此项预算代价仍在，是保留覆盖的选择，未按建议删除罕见结局；极端多结局必需内容仍可能降级为不可用，不承诺任意队列总能塞入上限。 |
| F§6.4、I§4.3：单点见证/旧补丁 | 见证保留队列发生数，选回撤≥0.02m、连续低速≥0.5s 等有分量的实例，不再选 1 个点当过程。 | 有代表实例不等于完整逐回合时间线；保留三类有实际反例价值的见证，没有为清理一概删除。Go2 参数公开是摘要参数。 |
| F§6.4、I§2.5：相对 z “举足” | 改为“相对 z 端点净升后稳定”，明示非单调、身体运动也可造成；例中保留接触、其他足活动、低进度与姿态。 | 检测器仍会覆盖静止/下蹲，不能叫世界抬脚；承重/其他足活动尚未在队列层完全分组。没有增加专用因果分类器。 |
| F§4、I§2.5：文字/数字细节 | 统一固定显示精度，限位占用用百分比；最大倾角明确逐回合最大后取中位；条件加分母；Push 量与文字共用 label/unit 事实，修正高度/速度列对应。 | 总体活动中位数仍会是 0，必须连同低进度的活动/低活动计数阅读；部分罕见原始结局标签仍回退英文键。整份任务 prompt 非全中文，仅本轮作者行为说明保持中文。 |
| F§6.5、I§4.1：异常边界 | Stair 纯描述异常已隔离；最后收口为 Push trajectory 主线在完整读取/验证原档、检查跨档案来源兼容后，隔离测量和成卡普通异常，送出 `description_computation_failed`，保留科学结果且不留半张报告。 | 身份、块损坏及 I/O 继续阻断；最终落盘不在 catch 中，legacy 生成链未扩大修改。Push 原问题是 reviewer 核实的静态错误耦合：普通描述异常会被 runner 判为 `SUMMARY_INVALID`，并非 K4 已发生该故障。 |
| F§6.5：将来多一个采集数组会失败 | 当前采集 schema 与精确数组集共同限定可解释来源，保留版本边界。 | 动作/力矩等新采集需同次更新 contract/adapter，不能把未来字段默认为已理解。本轮未新采集。 |
| F§6.6、I§4.4：废弃算子包装 | `15c469c1` 周期仅返合格间隔，关节仅返计数/余量，地形仅返所需数组；删对应无人消费包装及旧头注/无用目录版本常量。 | 保留真实消费的原数组引用/绑定，未把所有 hash 都当垃圾。 |
| I§4.4：先聚合再删 96% | `2dfbf3e0` 在聚合前限制消费事实，条件/过程组只计需要的数量，去掉后置 keep-list、成员列表及多余分布。 | 成句与事实 key 仍有直接依赖，未设计 registry/platform。 |
| F§6.5、I§6：Push 重测/缓存/作者 hash | `f933e577` 正式与 CLI 各测一次，删生产 hash 目录缓存及重复落盘链；`44384c27` 仅在外部绑定核验后投短编号，正式链不再额外做离线 public 投影。 | 离线 artifacts API 和外部文件加载核验仍有真实消费者；receipt 的历史字段 `cache_key` 是输入摘要而非现存缓存。默认 legacy/显式新 profile 的产品选择未改。 |
| I§4.3：版本/会话兼容 | Stair v7、Push v5 的行为格式有明确边界；旧会话继续由原 schema 身份限制，避免混入不同定义。 | 明确不迁移旧 Stair v6 或 Push v4 会话 ledger；它们不能在新 schema 下称为已恢复。需要时可由原数组 CPU 重出新报告，但不等于恢复旧会话。没有新兼容平台。 |
| F§2.4、I§4.4：剩余重复/可读性 | 已收敛共同阈值、CLI、当前 Stair profile 和消费聚合；Stair/Push 保留任务入口与各自几何语义，共享适合的算子。 | 末 0.5s 的行为/地形选择写法及文字仍各有表达；长成句函数、局部循环导入、单值 profile 参数和部分静态段仍在。未为形式整齐继续拆层。 |
| F§6.6、I§4.4：只写哈希与历史测试 | 最后收口删除验证退役 P0 参考生成器的 7 项测试及仅其消费的 tracked 生成器，删除前已核当前代码消费者及 HEAD 可恢复性。 | recorder/七个几何源文件 hash 仍是只记录的信息，不能冒充运行时校验。当前内部 card/receipt 绑定有消费者，不能按旧审计一概删除；P0 历史说明/结果材料仍原样保留。 |

最后最小收口只删除两个已 tracked、可从 Git 恢复的文件：`tests/unit/test_stair_trajectory_reference_facts.py`（227 行、7 项旧测试）与 `PLAN/feedback_sys/NEW_SYS/P0/recompute_reference_facts.py`（634 行）。删除前 tracked Python 消费者只有二者；历史 README/provenance JSON 的原记录没有改写，也未新造历史区。原有 raw、checkpoint、用户审计及运行目录保持不动。

## 4. 事实边界：已知行为、奖励与泛化

原审计的真实队列事实仍是当前开发验证材料：R4 的翻倒/禁用部位接触与近零、仍前进超时分支，K1 的左后足参与及关节占用，K2 的右后足接触占比偏低，m50 的静止和 m300 的继续前进后结束。m50/m300 读过之后已是开发数据；R1b 也不等于未见策略。审计里“第几阶失败”、全套翻倒侧向及完整末端时间线未成为本轮首类量，不能用支持的过程事实补猜标签。

原 8-case 独立审查者对本轮相关消费链复核未发现新本轮阻断，已知关键事实/界限没有因清理整删，但证据强度不同：R1b e175 首次上轴向下为步 209，随后低速段 275–600 仍为 326 点，209–274 的活动过渡没有单独展开，不能称完整时间线。e212 步 42→84 相对 z 端点净升 0.068206m、保持 84–544 共 461 点，本次只由原切片及当前模板复核；没有重出 864 回合最终请求。它们不是本轮新采集，也不应包装成全部作者请求重验。当前接触力与 history 槽的同拍差异仍有实例/队列计数；槽时标未记录，不能据缓冲正值倒推物理接触起点或持续时长。

地形量始终是记录参考点 z 减名义高度，非 sole clearance。失败末窗和成功末窗处于不同终点分布，文字虽能作队列对照，但没有逐实例、同位置配对，不能说“净空改善”。当前参考几何是已有 Go2/+x 任务范围，不能由 2/3/6 足等核心 CPU 测试推出 G1 或任意地形已完成生产资格。

奖励消费链已只读追到：采集的 `support_pos_w` 来自 `body_pos_w`，`support_lin_vel_w` 及奖励 `support_lin_vel` 来自 `body_lin_vel_w`；legacy reward 继续转交后者。本地线索提示前者是 link 位置、后者可能是 COM 速度，二者不是当然同一点的导数；历史安装版本与对应实现尚未闭环。因而 m50 的“位置几乎不变、原速度非零”足以否定直接用后者描述该参考点运动，却不足以宣布仿真传感器有偏、所有奖励滑移量错误，或擅自把奖励速度换成位置差分。本轮没有改奖励物理语义。若今后要解决此项，应先核对该历史安装的字段定义、实体原点与 COM 偏移、采样相位，再独立评估奖励影响。

新采集代码已有有效关节限位和地形配置快照入口，但当前摘要仍主要使用经过资格检查的名义界/几何；不能拿名义越界占用直接当 soft-limit 奖励惩罚，更不能据 CPU 契约测试声称有效限位或运行时碰撞面已经验证。旧档案缺失的元数据不会靠扩大历史白名单补齐。

真正未见策略、模型是否理解作者文字、同回合视觉核验和 GPU 运行均不在本轮完成范围。合成 turn/sideways 等输入只证明已有量的可区分性，不证明具备完整净转向/横移语义。本轮 CPU 收口没有调用模型或重启 R1b/run02，没有读取 K5 隐藏数据。

## 5. 可亲审的最终文字与原数组证据

Stair 的 R4/K1/m50/m300 原数组预期及作者导出入口见阶段记录和 `scripts/check_stair_behavior_delivery.py`；`D:\humanoid_agent\.tmp\cpu_behavior_fix`、`C:\Users\lover\AppData\Local\Temp\cpu_selection_budget_fix_20261004` 保留来源与输出。`329a155d` 的有限独立复核在 `independent_selection_recheck_20261004_074202/ROOT_RECHECK.json`；其后算子/聚合清理按原数组重出并比较真正消费值与最终文字，没有仅比较废弃壳。

Push 5B 的可复查材料位于 `C:\Users\lover\AppData\Local\Temp\push_author_5b_20261005_002813`：`raw_expectations.py/json` 先直接读 NPZ，`check_delivery.py` 校验作者内容；`k4_force_v1`、`k4_velocity_v2` 各含 `full/public/receipt/behavior/payload.json` 和 `author.txt`。原数组在 `outputs/p2b/raw/k4_force_v1/public_dev-01..03` 及对应 `k4_velocity_v2`。以下是这些最终文字的关键可核数字，不是新生成的模拟结果：

- force：成功 141、失败 51；首推开始至已观测任务终止中位数 5.04s/0.48s，1s 内终止 0/141 与 49/51。合格首推后窗仅 114/141 与 7/51；其余有后续计划推力干扰或提前结束，不能按 0 填充。
- force 的合格子集首推后根部高度约 0.371/0.326m、平面速度 0.791/2.255m/s、角速度 1.392/3.696rad/s。velocity 成功 191、失败 1，均有完整后窗；平面速度 0.567/1.042m/s。不同扰动协议/强度不混作同量，也不由子集差异推断因果。
- 首推后窗按实际 dt 向上取整，dt=0.07 时是 0.35s，不硬写 0.30s；真实 controller hooks 验证下次计划步 `< window_last` 才排除，等于最后读出步仍可用。

这两份 `behavior.json` 按正式规范序列化、包含 sources 的作者行为部分分别 11870/9350 字节（纯 `author.txt` 为 10561/8167 字节，整个任务请求不受这个行为分项上限约束）；独立 reviewer 还截获最终 `urllib Request`、未发送网络请求，并从 384 个原始 K4 回合重算。结论为 `LIMITED_CPU_PASS`，见 `C:\Users\lover\AppData\Local\Temp\independent_push_5b_20261004_165321\FINAL_REVIEW.json`。这证明请求构造与内容，不证明 DeepSeek 已接收、正确理解或产出奖励改进。

## 6. 回归记录与停点

最终回归统一使用已审源码 `ac612244ca720418759d82ff99fb11b2a4069b2d`。本地在原工作区执行完整 unit/integration；服务器只补五项本地缺 Isaac 安装源码而跳过的 fake runner/provider integration，在新目录 `/data/yuyangliu/isolated/trajectory_final_ac612244_20261004_184020/source` 检出同一提交。没有新安装、改变依赖环境、启动 Kit/GPU/真实训练/模型调用，也没有触碰 run02。服务器指定五项现已全部通过；本地和服务器执行前后的 SHA 均一致，tracked 源码无变化。

| 执行环境 | 实际 Python 与依赖 | 测试范围 |
|---|---|---|
| Windows 本地 `.venv/Scripts/python.exe` | Python 3.12.14；NumPy 2.5.3；Torch 2.11.0+cpu；无 IsaacSim/IsaacLab 安装 | 完整 unit 和 integration |
| 服务器 `/data/yuyangliu/envs/isaaclab6_validation/bin/python` | Python 3.12.14；NumPy 2.3.1；Torch 2.11.0+cu128；IsaacSim 6.0.1.0；IsaacLab 6.1.14 | 仅五项安装源码依赖 integration；Torch 的安装版本不表示使用了 GPU |

两边均设置 `PYTHONUTF8=1`、`PYTHONDONTWRITEBYTECODE=1`、`TRUE_RL_AGENT_ISAAC_DYNAMIC_TEST=0`、`TRUE_RL_AGENT_ISAAC_TEST_CHILD=0`、`OMP_NUM_THREADS=2`、`MKL_NUM_THREADS=2`。本地 `CUDA_VISIBLE_DEVICES=-1`，服务器为空；`PYTHONPATH` 仅含各自检出目录的 `src`、项目根和 `tests/unit`。实际完整 argv、cwd、版本、环境覆盖项、前后 SHA、退出码与 skip 原因分别保存在同一证据目录的 `local_results.json` / `server_results.json`；命令也写在每份日志头部。

| 回归 | 结果 | unittest 耗时 | 退出码 |
|---|---|---|---|
| 本地 `python -B -m unittest discover -s tests/unit -t . -v` | 2061 项：2046 pass、15 skip，0 failure/error | 418.306s | 0 |
| 本地 `python -B -m unittest discover -s tests/integration -t . -v` | 96 项：91 pass、5 skip，0 failure/error | 664.629s | 0 |
| 服务器 `python -B -m unittest <下列五个测试 ID> -v` | 5 项：5 pass、0 skip，0 failure/error | 1760.695s | 0 |

服务器五项使用真实 composition、准备/恢复、证据与父代接线，但 train/evaluation runner 及候选作者均为现有 fake；名称中的 training/session 不表示本轮启动物理训练。完整 ID 以 `tests.integration.` 开头：

- `test_completed_training_session.CompletedTrainingSessionTests.test_stair_go2_historical_train_current_q_and_two_generation_pause_resume`
- `test_generation_session.SharedGenerationSessionTests.test_stair_accepted_head_resumes_and_closes_final`
- `test_generation_session.SharedGenerationSessionTests.test_stair_reference_head_uses_shared_preparation_session_final_and_replay`
- `test_generation_session.SharedGenerationSessionTests.test_stair_early_rejection_sealed_rejection_and_no_selection`
- `test_generation_session.SharedGenerationSessionTests.test_stair_runtime_and_infrastructure_failures_keep_distinct_evidence`

本地 unit 的 15 项仍未测，没有通过临时安装、提权或放宽 skip 条件消除：

| 未测数量/原因 | 对应测试 |
|---|---|
| 8 项：Windows 无创建 symlink 权限 | `test_candidate_package`、`test_fs_contract`、`test_live_invocation`、`test_reference_artifact`、`test_subprocess_runner` 各 1 项；`test_stair_local_paths` 3 项 |
| 5 项：Linux procfs/进程组/父进程退出契约 | `test_generation_lifecycle` 1 项、`test_p6b_attempt_chain_fault_probe` 1 项、`test_runner_p1_reliability` 3 项 |
| 2 项：未启用 Kit 动态 opt-in | `StairIsaacLabRuntimeEnvTests.test_real_recorder_payload_decodes_tracks_and_aggregates`、`test_train_and_eval_builders_install_only_the_declared_runtime` |

这 13 项操作系统相关测试及 2 项 Kit opt-in 未列入本次服务器补验；它们不因其他 CPU 测试通过而成为已验证。完整测试方法名与原始 skip 文案保留在 `local_results.json`。本地五项 integration 的安装依赖 skip 仍记录为 skip；已核其完整测试 ID 与服务器实际执行的五项完全一致，并由服务器补验通过，结果单列。

本次日志与命令集中在 `C:\Users\lover\AppData\Local\Temp\trajectory_final_ac612244_20261004_184020`：`unit.log`、`integration.log`、`local_results.json`；服务器原日志在上述隔离目录的 `server_tests.log` 和 `server_results.json`，已回传同一本地目录。服务器五项只执行这一批，未触发 1800s 驱动上限，也未追加或重跑。首次远程启动时，Windows 文本 stdin 将 LF 转成 CRLF，Bash 在测试开始前退出；`server_driver.log` 保留该失败，随后把原 LF 脚本按字节重传，记录在 `server_retry_driver.log`，没有为此修改任何产品代码或测试断言。

前次 `44384c27` 的本地完整 unit 曾有 4 个测试方法产生 7 条含 subtest 的 error，全部在相机接口调用处首先暴露，历史完整日志保留于 `trajectory_closeout_20261005_010925`。根因是 CPU 夹具未跟上现有采集接口：缺少 Replicator、延迟创建的 RGB product 和 link pose 数组，模块替身作用域也只覆盖构造器。`ac612244` 只补测试夹具，保留原 35 个断言并新增 9 个；修前/修后同一 13 项命令由 7 条 error 变为 13/13 通过。独立审查实际执行原生产 attach/render/转换/PNG/manifest 路径，单个矩阵转置负例被新增预期捕获。证据位于 `camera_fixture_20261004_181144` 及 `independent_camera_fixture_20261004_182635/FINAL_REVIEW.json`；不据此声称真实 Replicator 或相机几何已获资格。

Push 最小收口的 70 项直接相关 CPU 回归及独立 14 项 pipeline 复核已通过。新增投递夹具首跑缺少既有 phase 身份字段，补全夹具后通过，未放宽生产校验。删除退役 P0 的 7 项并新增 2 项后，unit 方法数从 2066 变为 2061；如今已有上述真正完整执行结果，未把 import/收集计作通过。两份 K4 各 192 回合的 `full/public/receipt/behavior/payload.json` 与 `author.txt` 共 12 个文件，均与已审 5B 产物逐字节相同；作者行为仍为 11870/9350 字节，材料在 `push_closeout_minimal_20261004_175352`，独立复核在 `independent_push_minimal_20261004_175943/FINAL_REVIEW.json`。

最终只读核验见 `C:\Users\lover\AppData\Local\Temp\independent_final_logs_ac612244_20261004_192404\FINAL_REVIEW.json`，状态为 `FINAL_LOG_AND_DOCUMENT_PASS`：计数、完整测试 ID 对应、命令/版本/环境、冻结 SHA 与范围说明均一致，无阻塞。核验未重跑测试；本地五项 integration skip 与服务器五项属于同一组测试，不能重复增加独立测试数。

全部源码改动已按阶段独立审查后提交。最终日志及本文已获父级与原独立 reviewer 只读核验通过，完成本轮有限验收；本次仅提交处置文档，工程工作停止，等待用户审计。训练升级、DeepSeek 端理解及真正未见策略验证尚未完成，不自动继续后续计划。
