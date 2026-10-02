# P0：真实公开输入与测量语义冻结

状态：开发交付；父线程与干净独立验收线程尚未放行。仅完成 P0，不是 E1 卡片验收通过，不进入 P1。

本次是父线程对`d854bc90`未放行后的P0修复，测量语义版本为`trajectory.p0.measurements.v2`；新建后续commit，不amend已推送历史。旧事实中的M04 XYZ口径已被撤回，不能继续用作M04验收真值。

依据 [主计划 v1.1](../TRAJECTORY_TRANSLATOR_IMPLEMENTATION_PLAN_2026-10-02_ZH.md)、[readiness/勘误](../READINESS_AND_ERRATA_2026-10-03_ZH.md)。工作分支 `codex/trajectory-translator-upgrade`，开始 HEAD `e85d37c2ac855bc366a4c80340f7c2ee6af4ae54`。时间记录使用 UTC；目录中的 2026-10-03 是既有交接日期。本次运行环境为 Windows 项目 `.venv`；服务器仅用既有 SSH 读取公开文件，验证 Python 为 `/data/yuyangliu/envs/isaaclab6_validation/bin/python`。

## 交付及审查边界

| 文件 | 作用 |
|---|---|
| [source_lock.json](source_lock.json) | 两个不同公开档案的身份、manifest/chunk/资产哈希；首批R4全队列与旧反例用途分开 |
| [input_contract.json](input_contract.json) | 37项通道/能力、实体顺序、单位、时钟、sample_phase、validity、公开边界及源码哈希 |
| [MEASUREMENT_CATALOG_ZH.md](MEASUREMENT_CATALOG_ZH.md) | 首批十组量的公式、阈值来源、聚合、unknown/截尾、允许措辞和反例；P1迁移与作者语言决定 |
| [e1_selection.json](e1_selection.json) | 新translator输出前冻结的16个R4回合＋2个旧公开反例；含选择规则与必要事实类别 |
| [necessary_facts.json](necessary_facts.json) | 由原始NPZ独立复算的必要事实；P1文字出现后按此核漏项及错误 |
| [recompute_reference_facts.py](recompute_reference_facts.py) | NumPy直接读源数组的独立复算入口，不调用生产decoder/behavior或新增inspector造真值 |
| [输入审计工具](../../../../scripts/inspect_stair_trajectory_input.py) | 可执行的最小P0输入门：真实chunk读入、哈希/形状/映射/公开权限/时钟/首回合覆盖 |
| [输入门测试](../../../../tests/unit/test_stair_trajectory_input_inspection.py) | 密封输入、腐坏末chunk、实体/时钟错配、reset再入、NaN、padding及终点反例 |
| [测量参考事实测试](../../../../tests/unit/test_stair_trajectory_reference_facts.py) | 直接调用事实生成路径：XY/XYZ辨别、严格阈值、截尾、全部16组合、normalized/raw姿态与真实公开反例 |
| [verification.json](verification.json) | 实际命令、退出状态、日志/产物哈希、范围、失败与未验证项 |

原始下载及样本放 `outputs/trajectory_p0/`（Git忽略）；不引入模型、视频或大型NPZ到提交。不修改既有采集、奖励、训练、报告、提示词或接收协议。新脚本不是生产translator，也未创建 P1 的 `trajectory_translation` 包。

关键新增检查原文（完整diff及命令记录见verification）：

```python
_require(identity["splits"] == ["public_dev"], "identity splits must be exactly public_dev")
_require(set(terms) == set(manager_to_reason), "termination manager terms differ from frozen task mapping")
term_union = np.logical_or.reduce([arrays[f"termination_{i}"][valid] for i in range(len(terms))])
_require(np.array_equal(term_union, arrays["done_mask"][valid]), "done differs from OR(termination masks)")
```

`d854bc90`开发时的192项与17项记录保留在verification的历史区；父线程随后确认该提交最终16模块为194项。本次修复新增7项参考事实测试，最终完整专项17模块、201项的实际命令/耗时/结果见verification，不能累加旧测试充数，也不声称全仓通过。真实档案测试在本机全部执行；缺少被忽略的公开档案时允许普通开发测试显式skip，但P0验收要求零skip。

修复后重新读取2个manifest/60个chunk、生成18个样本与309项完整事实；摘要为246项回合事实＋1项队列事实＝247项（原245＋1＝246项，XY口径只新增`r4.r1.e60.RL_foot.low_speed_off`一项达到既有摘要展示长度）。此处仅更正文档计数，不改事实及原始记录。独立事实复跑与冻结摘要字节一致。原始manifest/chunk及18个样本NPZ的hash保持不变；变更的是测量语义、数值账和对应hash。另用标量math独立读取原始R4 chunk核对e60及e35，不导入参考生成器/生产decoder；7种错误语义的内存变异均被新测试拒绝。

JSON显式输出LF，与仓库`.gitattributes`一致；发现Windows默认CRLF可能使提交后来源锁自身hash变化后，已规范化新JSON并重新生成、逐字节复算。原始manifest/NPZ从未重写。提交前另核暂存blob与工作区字节一致，防止跨检出后的伪哈希不一致。

## 输入身份与用途

首个完整档案为 R4 Stair Go2，manifest SHA-256：

```text
8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83
```

远端位置 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20261001_r4/stair/numeric_evidence/manifest.json`；本地复制至 `outputs/trajectory_p0/raw/numeric_evidence/`。30个NPZ共83,039,745B，manifest 573,879B，chunk原始数组233,923,680B。全部chunk实际读取并与manifest内SHA/shape/dtype匹配；不是只检查文件存在。

还直接读取了服务器 `n5_confirmation_source_r4_final` 的numeric_capture、runtime_projection、isaaclab_runtime_env、transition_decoder四个源文件SHA，均与本地冻结代码相同。当前源码对实体顺序及pre_reset采集的说明因此有对应的历史源字节支撑；未把当前其他runtime实现倒灌成历史能力。

`identity.splits=["public_dev"]`，三个seed也均为public_dev。历史探针的“sealed archive”仅表示已封存，**不等于任务 sealed_validation**。metadata中包含未授权split的规格字样不是读取了该split；准入逐一核对真实identity/seed。冻结与翻译器评测用途为两条独立轴；本批已被既有审计使用，只能作development_regression/descriptive_audit，不能冒充盲确认。

策略checkpoint哈希 `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`，cell=`stair.go2_12dof.unified_v3`，projection=`ea9d42a4012815a8f5ebf6b35b93303111477817feb5a1391fa807efc6167a14`。不将不同seed或同lineage checkpoint计为独立策略。

R4有3×288=864回合，8个stratum各108回合。有效样本298,375，终点之后padding251,993；前者包括每回合终点、后者不参与分母。存档及既有生产decoder重放结果均为成功682、forbidden_contact125、timeout39、base_orientation_instability16、off_course1、slip1。此重放验证旧终局一致性，不充当新量的独立物理真值。

另一个来源是旧开发档 `22fb749620c4e614cbb4df4fbfdb82b70d76c8f1cd81a3eda50de465d9fb3e4a`，本地既有 `PLAN/feedback_sys/P3R/results/stair_numeric_dev_r1/numeric_evidence/`。只选 `legacy_dev.r1.e54`（先翻转）与 `legacy_dev.r1.e60`（成功回退并返回）。它们与R4同名回合不是同一实例；来源锁、事实与引用均加archive命名空间，不合并频率或父代比较。

## 实体、时间和真实数值锚点

足顺序为 FL_foot、FR_foot、RL_foot、RR_foot；关节顺序为四hip、四thigh、四calf，RL calf索引10。base运动是articulation root；forbidden force仅base，不能借robot profile的全禁触部位列表声称记录了小腿受力。Go2 body-frame记录与manifest的profile/资产哈希相符。

位置来自`body_pos_w`、速度来自`body_lin_vel_w`；link位置与COM速度的历史资产参考点/质心偏移映射未完全验证。当前只说所记录的位置/速度通道，不将速度解释为位置的精确导数。M02及M08统一先校验raw四元数范数误差≤1e-4再归一化；主up-z、负值区间和body-frame旋转均用normalized q，raw操作数及raw up-z诊断单独保留。

首块 `chunk_000000.npz` SHA为 `5f5e89aef9bf56a51d638be61c33783abe8473e79337bc2743599fd96b5e0e6b`，support force shape `[64,288,4,3]`、history shape `[64,288,3,4,3]`。控制步0.02s，物理配置0.005s；每步末post_step/pre_reset克隆。首帧k=1、t=0.02，未记录t=0。history三槽newest→oldest，**没有逐槽时间/valid**；不能按配置period补全物理子步接触。

力通道的控制步时间是读取时间，不是已记录的传感更新事件时间；物理同步误差未知。独立复算的 `r4.r1.e129` 在终点step356的`base_contact` mask为true，但current base force全回合为0；必须保留history与manager信号的定义，不能以current零力抹掉短终止事件。

首块索引 `[0,0]` 的root位置 `[-3.3240957260,-46.0897979736,0.4006170630] m`，地形origin `[-3.25,-46,0] m`，故任务x=`-0.0740957260 m`。四足world位置在本近零yaw姿态下的左右/前后符号与绑定一致；完整原向量保存在input audit raw_anchors与独立事实样本，避免仅靠两个程序共用标签。

两个独立可复算反例（详细操作数/定位/容差见necessary_facts）：

1. `legacy_dev.r1.e54` 首次 `up_z<0` 为step278，即5.56s；FL低世界水平速度且标记off的末段step326..588有263个样本，格时长代理 `263×0.02=5.26s`，首末观测跨度 `262×0.02=5.24s`，右截尾。不能写成5.26s连续物理腾空，也不能写成身体直立时举足。
2. `legacy_dev.r1.e60` 在step84峰值到116谷值回撤约0.125878811m，step141首次返回原峰值，仍是成功回合。回退不必然失败，返回同一x也不等于已恢复平衡。

全682个成功回合的全部有效记录池化后，RL净法向力**模**份额为0.1528875367，带符号world-Fz份额为0.1510607564；独立复算两者均保留，不混称竖直冲量或重量份额，也不冒充仅台阶段统计。

父线程指定的修复辨别证据：

- `r4.r1.e60` FL按唯一水平速度谓词得到step214..637，424样本、格时长8.48s、观测跨度8.46s、右截尾；旧XYZ口径仅475..625、151样本、非右截尾。step626原力为0，所记录速度`[0.0415688157081604,0.016667895019054413,-0.0320286750793457]`，XY=0.04478599294137869、XYZ=0.05506015974444246 m/s。
- M05唯一编码`sum(int(marker[i])<<i)`，实体FL/FR/RL/RR、权重1/2/4/8，编码版本`role_index_lsb.v1`。`r4.r1.e35` step14的`[true,true,false,true]`为11，回合counts[11]=135，不能用13。只纠正P0目录反序声明并明确参考账语义，不改变生产ABI。
- 合成`v=(0.03,0,0.08)`、力0必须入选M04；水平速度恰为0.05不入选；力恰为1N仍是marker off，大于1N才on。旧e54专用分支也直接测试，避免仅修通用分支。

## 首批选择、必要事实及反例门

16个R4回合：每stratum一个成功例，按固定salt+run/env的SHA最小值选择；再纳入已知r1.e129短终点事件、全部两个稀有终局、一个姿态终局、四种条件与多个seed的timeout。它是分层工程抽核，不是总体频率估计。旧两反例另列，总计18个实际回合。

在新系统文字出现前固定的必要事实包括：身份/实体/时钟；实际终点与任务authority；根部净进度、路径、姿态；各足标记数和截尾run；短转换及终点事件；受力语义和缺通道。未知机制不计错误，但删除已有短事件、把unknown记阴性、错归实体、把代理升级成物理事实均为关键错误。P1必须对所有最终卡片句子逐条核验，不能只抽数字或拿旧PASS通过。

P1仍须实现并测试：交替左右足与同一足持续的区分、单步重触发、低净进度但仍运动、随身体移动的高位足、地面高度变化与相对抬高的区分、合法越障成功、缺测/遮挡、未知接触对象、左右截尾、零成功/无父代、容量与密封污染。P0输入门测试不能代替这些尚未实现的测量/文字门。

## 依据事实作出的范围决定

- 首批启用原始标记转换和时差；周期/标准步态命名暂不启用。旧探针0.1s去抖会丢事件，可靠周期规则尚未独立验证。详见目录M06。
- 启用关节姿态范围，限位邻近待绑定与实际USD同坐标的名义限位；真实有效限位本档未记录。不把旧URDF常数直接冒充该实例限位（M09）。
- 任务进度/条件/终点可用；terrain高度差及base间隙在P1逐几何类型独立验证后再启用，粗糙面代理不是碰撞真值（M10）。
- 当前HEAD强制更新接触传感器的实现，不能倒灌升级历史R4的slot时间/覆盖声明。
- 作者文本沿现有英文链路，固定定义句解释marker等限定；P1显式升级Stair public schema/lexicon/comparison并保留旧回放。最终输入为serialize后的JSON，完整档案、路径和typed slots不自动到达作者。

这些是P0的输入依赖和实现顺序决定，不改变任务评分或计划的P1最终目标。周期/限位/几何仍属第一版目录的后续义务，不能永久移除；几何是独立验证未做而非输入不存在，历史link/COM映射也需在更强运动结论前补齐。阶段推进由父线程决定。

## 未验证与实际失败记录

没有P1卡片、自然语言事实支持率/召回、实际作者读法、DeepSeek真实调用、视觉检查、补采、跨lineage迁移、新形态或训练收益验证。既有decoder重放及合成测试不是这些能力的替代。

启动时`rg`不可用，改用PowerShell/Python搜索；沙箱SSH未读取别名而解析失败，使用既有授权SSH在正常权限下读取后成功；没有修改认证。Git在沙箱对长路径返回访问拒绝而显示D，正常权限只读status确认没有实际删除/既有改动；未恢复、清理或覆盖任何文件。早期manifest检查先于传输完成而报FileNotFound，待字节原样下载后成功。新增测试开发中的一次测试import拼写错误已修正；最终命令/退出状态另存verification。上述启动/开发错误没有包装为通过。

真实inspector首次复核命令误填checksum，被正确拒绝；错误日志 `outputs/trajectory_p0/input_inspector_bad_checksum.log` 保留，按冻结正确hash重跑通过。`ruff`在现有环境不可用，未安装依赖；语法、专项unittest及实际数据入口已检查。
