# P1a 数值测量与时序证据

起点：`2d888a846636ffb0e3a732d9d3423288ff3b2cac`，专用分支
`codex/trajectory-translator-upgrade`。正常权限只读工作区干净；沙箱长路径 D 为假删除。
仓库根及项目内无适用 AGENTS.md，`.agents`/`.codex` 正常权限枚举为空。

范围：contracts、measurement、temporal、Stair 公开数值适配器及离线证据入口。
按 P0 v2 启用量覆盖全 R4 队列；legacy 两反例单独输出，不混频率。
不实现 P1b 卡片/作者、查询 AST、视觉或训练，不修改任务终局、奖励或 ABI。
周期资格、限位绑定、地形几何及历史 link/COM 映射仍为后续义务。

小阶段门：契约/时序合成判别 → 测量/适配判别 → 真实全队列、独立事实对照及回归。
原始 source_lock、e1_selection、necessary_facts 与 full_reference_facts 均保持冻结。
开发自测不替代父线程独立验收；本轮结束停止写入。

## 实现及边界依据

`evaluation/trajectory_translation/{contracts,measurement,temporal}.py` 只接收已绑定的
实体、时钟、任务参数与数组，不导入具体任务；Stair 的身份、角色、结果及采集格式在
`tasks/stair_climbing/trajectory_adapter.py` 校验。适配器复用 runner 数值读器：先校验
全部 public_dev seeds/chunks，再逐 run 拼接全部有效首回合，不能因选了两个 legacy
反例而跳过其余源文件的准入检查。档案 v2 自称完整首回合，因此缺 chunk、reset、
valid 重入、schema/角色/时钟漂移直接报错；一般 core 支持显式缺步及逐通道 NaN→unknown。

episode ID 绑定 manifest SHA、trial、run、env；record ID 再绑定配置、实体、通道、
单位、参考系、区间和值。显示名不是身份。每个记录包含原 chunk SHA、行区间、env、
原通道；关系保留 operand ID 或直接原读数引用。归档 outcome 保持任务评价器权威。

启用 M01–M10 的可测部分：根部进展/路径/回撤、归一化姿态、各足原始接触段及转换、
低 XY 速度且当前 marker off、组合编码、最近 onset 配对、法向合力模与有符号 Fz、
身体坐标相对位置、关节绝对范围、任务终点及其完整前情索引。M06 配对是方向性的最近
目标、等距选较早、允许复用，不伪称周期相位。全足均 off 不代表整体腾空。

未平滑或删除单采样事件。未知样本、缺步、左右边界截尾均显式标注；记录连续不等于
物理连续。严格 `XY norm < 0.05`、当前力模 `> 1`、FL/FR/RL/RR 权重 1/2/4/8，
raw wxyz 在容差内才归一化，raw up-z 只作诊断。history 是控制端点读到的 buffer ANY，
无逐槽有效性/时间戳，不拼成物理子步，不覆盖 current marker。

样本个数、`n*dt` 格时长代理、`(n-1)*dt` 观测跨度、相邻有效边梯形积分代理分别保留。
占比同时报告回合等权及样本格时间权重和分母；力模、signed Fz 的有效分母分别计算。
只有诊断积分代理，没有真实冲量、承载份额或动量平衡判断。历史 link position 与
COM velocity 映射未核清，不声称两者精确导数关系。

入口 `scripts/build_stair_trajectory_evidence.py` 产生 episode/measurement/state_interval/
event/temporal_relation JSONL；完整校验并写完后才生成 complete manifest，拒绝覆盖既有目录。
manifest 的小队列摘要只服务算术验收，不含策略卡片、父代选择、作者投影、查询 AST。
每个 source 单独运行。约 1.22 GB 的 R4 底账放在 ignored outputs；未为展示压缩删事件。

## 开发验收记录

冻结输入和范围确认后依次通过：core/项目边界 33 项；测量集成及新增适配器、入口逐步
扩展至 80 项。最终只读审阅发现合取算子丢失回看窗口依赖，修复为继承两 operand 最大
lookback，新增嵌套合取的状态/转换引用判别，相关 81 项通过（4.201s）。此前失败的开发
日志保留在 outputs，均不作为最终通过记录。生产 R4 路径未使用该 helper，仍按最终代码
重建产物，保证 manifest 代码哈希绑定准确。

合成判别覆盖：同占比不同顺序、一次短接触打断否定、同足与交替足、unknown 和缺步、
左右截尾、reset、重复显示名异来源、严格阈值、编码、归一化/反向旋转、积分与占比单位、
current/history 分离、源损坏/后续 seed 非公开、拒绝半成品发布。不是仅检 schema。

独立比较器 `scripts/verify_stair_trajectory_evidence.py` 仅使用标准库，读取冻结事实及
生产输出，不 import 生产测量或 P0 oracle，不从原数组补造缺失产物。18 个冻结例按
manifest SHA+run+env 对齐，整数/布尔精确比较，浮点只用冻结的容差；逐字段保存舍入差异、
映射及不支持理由。P0 事实、选样和源锁均未更改。

最终命令、原始结果摘要、输入/产物/日志 SHA 见 `verification.json`；真实数组原文及对应
完整证据记录见 `outputs/trajectory_p1a/raw_spotchecks.json`，简要说明见下节。专项 201 项
与更广 CPU discover 分别报告，不把专项结果当全仓或 GPU 通过。

最终自测：相关 93 项（4.457s）、原 201 项（50.775s）均 OK、零 skip。
R4 全部864回合/298375有效行：31104测量、250376状态、232232事件、79417关系；
legacy 源全864回合均校验，单独输出冻结两例/1113行。最终重建 JSONL 与首次产物
SHA完全相同，manifest的代码哈希与交付文件逐项匹配。

冻结必要事实247/247通过（4168标量）；完整事实291通过、18项库存表达 unsupported，
0数值/时序不符（44365标量）。浮点差异均逐条在冻结容差内；半开区间、实体ID、稀疏零计数
等字段映射有明示依据。最初72项阈值来源文本未映射，最终比较器改为核对生产
`development_calibration`、0.05及明确XY公式，没有修改冻结值或生产以迎合oracle。
剩余18项各有 `missing_channels` 与 `existing_temporal_channels` 两条未自动验收路径：
产品使用 limitations、coverage、channel refs、locators，未提供P0导出样本的同形库存表。
其中全非足接触力缺口只由绑定 bodies=[base] 暴露，没有独立同名库存项；不声称库存覆盖
已完整验收。阈值、数值、全部短转换、on/off段、history诊断与截尾已独立比较。

更广 CPU `unittest discover -s tests -t . -v` 实跑2125项/1087.209s，结果为
`FAILED (failures=1, errors=8, skipped=19)`，不能作为全仓通过。7条error来自既有相机
路径缺少 `omni`（含4个subcase），另1条是既有 `test_stair_body_projection` 裸import：
补测试搜索路径后其2项通过。1条FAIL是既有Push static_repair测试要求省略
`public_feedback_summary`，而基线required_sections包含此项。两项代表失败单独及fresh
进程复现均相同，已加载旧Python文件与accepted P0内容一致，未加载任何新trajectory模块。
本轮不安装Isaac依赖、不改旧作者/相机契约。19 skip限平台权限、Linux、Isaac等条件，
逐项原文在verification中。discover启动早于最后lookback修复；最终93相关项及201专项
在代码冻结后重跑，补足最后改动验证；未将两轮测试数相加宣称覆盖。

## 真实输入到证据的抽核

以下均从源 manifest 核 SHA 后直接读取原 NPZ，用独立 numpy/标量计算核对输出记录。

| 源/原行 | 原数组与计算 | 输出证据 |
|---|---|---|
| R4 r1 e60，step626，chunk9 row49，FL | 速度 XY norm=0.04478599294137869；XYZ norm=0.05506015974444246；当前力=0 | FL low_speed_off 214..637 共424格，8.48格秒/8.46跨度，右截尾；XYZ 不参与判别 |
| R4 r1 e35，step14，chunk0 row13 | 四足力 `>1` 为 `[true,true,false,true]`；1+2+8=11 | 整回合 code11 计数135，保留全部原始转换 |
| R4 r1 e129，step356，chunk5 row35 | 当前 base 力=0；history 有一次越阈；termination_1=true | current 触发0、history 触发1、manager终点356，彼此不替代 |
| legacy r1 e54，step278，chunk4 row21 | 原 wxyz 归一化后 up-z 首次负 | inversion 先于同足 FL 326..588 的263格低速 off；只确认读数时序 |

同名 R4 e54 与 legacy e54 使用不同 manifest/episode ID。legacy 两例不进入 R4 队列分母。
验收抽核不把 P0 的 `confirmed_by_raw_operands` 收据当作本轮新读取原数组的证明。

## 后续补采登记（proposed / not_collected）

| 最小增量 | 解锁条件 |
|---|---|
| 每环境质量、随机化、有效限位、逐步 command | 物理上下文、限位邻近、指令偏差 |
| 刚体参考点/COM 偏移与同点速度；sensor 更新序号、时间、逐槽 valid | 同参考点运动与接触覆盖；切向力/接触对象仍须另采 |
| raw action、scale/clip、target、qvel、实际 torque/饱和标记 | 动作到执行的数值对照 |
| reward 总量/分项、权重、dt、gate、air-time/first-contact | 实际奖励对账；真实与重算分字段 |

后续短录制须用冻结身份的 public_dev checkpoint，关联 action 所作用区间、reward 与
post_step/pre_reset state，核对 reset/首帧/terminal/缺步/未填满 history 槽；实测逐组
字节每步、缓冲峰值、写入和仿真耗时、丢步/重复、对齐误差后再定预算。本轮仅设计，
未采集或解锁。新 profile 不升级旧档；周期、限位、逐类型地形及历史 link/COM 映射
各自通过独立资格门后再启用更强表述。
