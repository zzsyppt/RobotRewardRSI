# P6 独立开发：先查原因，再修 reference 与难度

2026-09-12。目标仍是 Stair/Push × G1-37/Go2 的可用 reference 和同一共享宿主。首轮正式实验两格 NO-GO 保留；本文不修改其协议、checkpoint、资格判决或已消费留出。P7 尚未获准。

## A. 接线与评价诊断（已完成）

1. 核对原训练冻结源码、实际 env.yaml、500 轮 TensorBoard 与 stdout：分清奖励是否执行、施扰是否接线、训练回合是否达到施扰时刻。没有保存的事件累计数不补造；评测事件数不能替代训练事件数。
2. 阅读目标安装版本的环境 step、EventManager、TerminationManager、RewardManager，确认 termination/reward/reset/interval 的实际顺序和唯一 dt 积分。
3. 同原 checkpoint、首个 public-dev seed、64 env，分别跑原扰动和 no_push，每本体各两次，共四个 rollout。保留真实 pre-reset 原始位姿、速度与实际速度写入；离线独立重算稳定条件、连续驻留、恢复窗口及终态，与产品逐事件结果比较。no_push 只报告站立能力，不计抗扰成功。
4. 同时用已有原生影片解释行为；图像与共享算法的输出一致不等于算法独立正确。无需为本次计数诊断额外追录。

本批不训练、不访问 accept/sealed/final、不改阈值。执行前将脚本、checkpoint、协议、seed、预算写入独立诊断声明；总 GPU 分配墙钟上限 1 小时，每个作业 15 分钟、0 自动重试。复用现有进程与 lease wrapper，不新建调度器。只使用已释放的设备。

验收：独立事件/稳定/dwell/final 逐项相等，原证据 hash 不变；如不相等，定位首个分叉并先修确证的接线或评价错误。若相等，按 no_push 与原扰动的动作区分基础站立能力和受扰恢复能力，不能把全部失败归为缺某项 reward。

A 的实际结果见 `independent_timing_summary_01.json`：四个 rollout 的独立逐事件判据均与产品一致，旧 G1 在 no_push 下 0/64 获得完整稳定驻留，Go2 为 64/64。旧 G1 的基础站立行为存在问题，不能归为只有受推后恢复失败。

## B. Reference 的受控开发（已完成）

只有 A 排除阻断性接线/评价问题后，才决定是否采用现有 stationary_v3 草案。该草案只用已授权的当前速度输入，未引入未来扰动信号；四个其他 reward 函数保持不变。其 CPU 通过不代表训练有效，瞬时奖励排序不证明整段回报正确。

若采用，先独立冻结 reference 源码、训练身份、500 iterations 的固定预算与 dev-only 判据；每本体至多一次 random-init 训练，用原三个 public-dev seed 比较原 reference，同时记录实际训练施扰次数及基本站立诊断。保持机器人 PPO、资产、观测/动作和恢复标准不变。不能从失败中无限换系数重训；失败后发布结果并重新判断主因。

训练施扰计数应由 Push runtime 拥有，写入该任务已有训练证据；不让 Push 事件语义进入共享 Runner 或改动 Stair。旧摘要没有计数只能保持缺失，新 writer 不回填旧证据。

B 已按 `stationary_v3_declaration_01.json` 完成，两格dev均190/192、no_push完整dwell均64/64；初始policy hash和训练条件与原轮相同。G1的自旋显著降低，B预声明开发判据通过。见 `stationary_v3_summary_01.json` 与 `stationary_v3_independent_audit_01.json`。两格仍超过资格余量上界0.9，不能晋升；不改奖励系数重试。

## C. 难度与新的正式资格（有限标定结束，G1正式资格及回放完成）

只在基础站立/恢复能力成立后，依据公开 dev 做事前有限强度标定。机器人层提供尺度事实，任务/格子绑定拥有校准强度；共享方法不等于强行同一无量纲强度。当前 Go2 饱和不通过放宽余量上界解决，G1 基础站立未成立时不靠减轻扰动掩盖。

本批按calibration_declaration_01仅用固定B checkpoint，将原三项强度同时乘1.5或2，原强度1复用B三组dev；两本体用同一选择规则但可选择不同点。从低到高选首个dev落入既有[0.2,0.9]的点，低于下限/无有效证据/全网格仍不合格则停止，不插点、不二分。每点只3×64原公开dev，最多四个新点、0训练/holdout/provider/自动重试、每点900秒。已在目标环境preflight后签发不可变声明再启动；不在本文件隐含扩大扫描。开发确定后，统一签发新科学协议及两格 reference 身份；正式资格采用预先生成且未消费的 accept/sealed，不把旧题换个 hash 当未见。仍按既定基本可学性、余量与独立验证要求判决。

现有 Push 的训练绑定包含完整任务协议及评测 cohort，改变 fresh seeds 后不能通过已有严格续接入口复用 B 训练。B checkpoint 仅作公开开发 anchor；最终条件冻结后重新训练正式资格。Push composition 原先忽略额外传入的 `eval_base_seed`，现已接通该输入及协议文件、reference ID，并完成专项测试；目标部署与实际 preflight 已完成。新题须以实际生成的 seed 集合核对不重叠，不能只改命令参数就宣称 fresh holdout。

最终须补齐两格资格、同 checkpoint 成功/失败原生案例、Stair 当前执行回归、历史读取和共享宿主完整验收。任何单项通过均不能替代 P6 全部完成。

选档后将G1完整协议放入版本化 `configs/tasks/push_g1_37dof_recovery_v3.json`，G1新建装配显式选用该协议、v3 reference 和新 cohort；Go2未选中，不借此修改其默认条件。旧 manifest 的缺省解码含义保持不变。G1最终训练条件已另行冻结。若重新训练后的 dev 仍饱和或低于下限，该次资格失败，在 dev 停止；不自动换档、重训、改变门槛或消费后两相来寻找通过理由。

C已完成3个新增点（G1 ×1.5、Go2 ×1.5/×2），共576个episode，0训练/holdout/provider，0.03474 GPU分配墙钟小时；原始事实独立审计见calibration_independent_audit_01.json。G1为167/192，首点入带后停止；Go2最后176/192仍高于0.9，网格耗尽NO-GO，不自动增档或换seed。G1按g1_fresh_qualification_declaration_01.json执行一次新的500轮随机初始化训练，完整协议同时作用于训练和评测；这不是B checkpoint的新包装。新cohort实际派生为1000010000起的dev、1000020000起的accept、1000030000起的sealed，已对已核实P2/P3/P6集合检查不交。P6总目标仍未完成。

G1新资格已完成：dev96/192、accept106/192、sealed245/448，原门PASS并生成真实Genesis。同checkpoint一次原生dev重放192条全字段零差异，三段覆盖成功/侧跌/前跌；本地完整解码及终止前图像核对通过。详见ACCEPTANCE_ZH.md及g1_fresh_qualification_audit_01.json、g1_fresh_native_audit_01.json。Go2的C网格耗尽结论不变，整个P6尚未完成。

## D. 匹配训练分布（独立声明，已停止NO-GO）

后续独立问题与预算见GO2_MATCHED_TRAIN_DEV_ZH.md，未延长C网格。Go2在同一已有×2和原dev种子下重新随机训练500轮，实测192/192。工程PASS、原余量门NO-GO，停止该假设，未消费holdout或启动新资格。详见GO2_MATCHED_TRAIN_RESULT_ZH.md；P6总体仍未完成。
