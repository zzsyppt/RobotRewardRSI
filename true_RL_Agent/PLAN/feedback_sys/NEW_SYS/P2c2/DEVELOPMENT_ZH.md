# P2c2：关节名义 USD 限位诊断交付

起点 `71c9cac31664ce32bb58c0804d33efa2ff8c0dd1`，分支 `codex/trajectory-translator-upgrade`。
先以 `a67988fc` 提交[冻结判据](FREEZE_ZH.md)，再提取资产与读取原 q。开发自测及内部只读
审查 PASS，**等待父线程独立验收**。用户已确认 P2c1 限定离线 PASS；本阶段不改其旧证据。

入口 `scripts/build_stair_joint_limit_supplement.py` 接受三份已核历史 raw manifest reference，
复用旧 Stair adapter，生成 M09 名义限位证据、汇总、12条可选句及合计8192B离线预览。
catalog v4/base v2 独立增量；旧 v2/v3、Push冻结代码未改。正常作者入口未接新句。

## 真实资产与历史 q

[ASSET_AND_SOURCE_EVIDENCE.json](ASSET_AND_SOURCE_EVIDENCE.json) 保存三原 job/source 路径、
SHA、逐行采集片段和独立组成资产结果。生产包 `tasks/stair_climbing/go2_nominal_usd.json`
保存完整 Sdf 字段、root USDA及36条引用，其固定 SHA：
`b8a1ab9846cca1b282258cd9b8c871d5f6746bd9bbb85e3d38a34921d90614bb`。

root 8142B SHA `ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d`；
mesh 18802187B SHA `2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5`。
root原始36个reference均为`./Props/instanceable_meshes.usd`，目标prim全存在，无sublayer或
进一步外部引用。三archive的名义closure逐项匹配，包括完整URL和字节数。独立Usd/UsdPhysics
在CPU上组成原二进制：只在匿名内存层映射缓存引用，重读12关节，与Sdf的名字/prim/body0/
body1/上下限逐项相同。没有写服务器。给定root缓存名缺了`f2`，实际路径见冻结文件。
P2c1 DEVELOPMENT 的mesh SHA末尾多写`1`，正确值为以上64位；未改历史文件。

| 关节组（组内逐名核验） | USD lower / upper，degree | 名义 lower / upper，rad |
|---|---|---|
| 四 hip | −60.0001335144 / 60.0001335144 | −1.04719988146 / 1.04719988146 |
| FL/FR thigh | −90.0002059937 / 200.0023651123 | −1.57079992206 / 3.49069978299 |
| RL/RR thigh | −30.0000667572 / 260.0025024414 | −0.523599940731 / 4.53789973103 |
| 四 calf | −155.9992218018 / −48.0001068115 | −2.72270005099 / −0.837759905170 |

USD limits的单位为度，原float32字段转double后乘π/180，依据
[OpenUSD schema](https://openusd.org/release/api/class_usd_physics_revolute_joint.html)。全部joint
axis为X；hip localRot identity，thigh/calf localRot绕Z的90°使轴在父体中为+Y，不能把90°加到q。
两侧localRot相同、body原orientation identity，代入`G0·L0·J=G1·L1`得J零旋转；anchor最大差
`3.725290298461914e-9m`。base是唯一articulation root，body0沿树为父链接，无反向链/mimic。
因此本资产无额外零点/反号，默认站姿并非零坐标。依据
[Joint Frames与limits](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/joints.html)
及[articulation拓扑](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/rigid_bodies_articulations/articulations.html)。

三原job的`metadata.work_dir`、result numeric reference、原manifest SHA现场闭合：

| 源 | 历史source（前缀`/data/yuyangliu/AGENT/`） | recorder / 保序resolver行 |
|---|---|---|
| K1 | `feedback_p2_audit_20260924/source` | `isaaclab_runtime_env.py:117` / `runtime_projection.py:899` |
| K2 | `feedback_p2_20260923/source_r2` | `isaaclab_runtime_env.py:115` / `runtime_projection.py:863` |
| K3 | `feedback_p3new_20260929/n5_confirmation_source_r4_final` | `isaaclab_runtime_env.py:117` / `runtime_projection.py:899` |

三源关键原文：

```python
"joint_pos": robot_data.joint_pos[:, resolved.joint_ids].detach().clone(),
ids, resolved_names = method(list(names), preserve_order=True)
if tuple(resolved_names) != names or len(ids) != len(names):
    raise RuntimeError(f"Stair {context} schema drifted")
```

历史projection声明rad；normalizer原样复制，capture原样stack，仅清零invalid padding；无
减默认站姿、归一化或角度转换。冻结IsaacLab revision的
[ArticulationData.joint_pos源码](https://raw.githubusercontent.com/isaac-sim/IsaacLab/3c6e67bb5c7ada942a6d1884ab69338f57596f77/source/isaaclab/isaaclab/assets/articulation/articulation_data.py)
直接取PhysX DOF position。archive列序为四hip、四thigh、四calf，各组FL/FR/RL/RR；USD
遍历顺序不同，必须按名字取限位；RL calf为零基第10列。

这证明相对于绑定**名义资产坐标**的含义。profile closure是声明绑定，不是每步运行场景
快照；有效硬/软限位与随机化后边界没有记录，始终unknown。

## 最小实现和有据调整

共享新增`joint_limits.py`，任务新增`trajectory_joint_limits.py`，未改原六个冻结共享模块。
新共享文件会进入Stair未完成构建的code-hash glob，已sealed旧replay不重写。

```python
lower_distance, upper_distance = q - L, U - q
margin = min(lower_distance, upper_distance)
u = (q - L) / (U - L)                 # 不clip
range_occupancy = (max(q) - min(q)) / (U - L)
near = L <= q <= U and (q <= L + .05*(U-L) or q >= U - .05*(U-L))
outside = margin < 0
```

每关节保留total/finite/unknown、near/outside分子及有限分母、sample-cell seconds、q/u范围，
首个极值q/step/chunk SHA/row/env/column。回合等权分布与池化样本/格时间分别保存；缺步不
插值，格时长不称物理连续持续。缺映射/无有限q为unknown、比例null；错单位/退化限位拒绝。

内部审查修复两项：直接margin比较会在真实USD精确5%端点因约1e−16舍入漏计，故用等价q
闭区间比较，无容差扩张。其次旧public自哈希挡不住改文后重签，现将本阶段入口限定到已核
K1/K2/K3 raw SHA及P2a完整public规范JSON SHA。新句只由raw内部生成，入口不收自由records/
report/text。换源/未资格源/改句重签全部拒绝。此为有意缩小task adapter资格，避免重造旧卡
验证器；未来新档案和生产投递需另做动态资格，不能将白名单称为通用投递能力。

## 原数组与实际句子

独立`tests/unit/joint_limit_oracle.py`不导入生产测量/asset loader，从USDA重读float度字段和
框架，用标量公式复算全部87个SHA锁定NPZ。`audit_raw.py`逐条核31,104记录、全部极值定位、
等权/池化统计、36条句子和receipt。[REAL_EXAMPLES.json](REAL_EXAMPLES.json)保存预声明
三类别首个见证、全12维原q、原定位、完整关节记录及全部句子，不按终局或好看输出挑例。

| 源 | 首个内侧近带 run/env/step/列 | q / 裕量，rad | 首个名义界外 run/env/step/列 | q / 裕量，rad |
|---|---|---|---|---|
| K1 | 1/0/2/11 RR calf | −0.871483206749 / 0.033723301579 | 1/0/50/10 RL calf | −2.722701072693 / −1.02170328e−6 |
| K2 | 1/0/2/10 RL calf | −0.871974527836 / 0.034214622666 | 1/0/3/10 RL calf | −0.837753057480 / −6.84769018e−6 |
| K3 | 1/0/2/10 RL calf | −0.871963500977 / 0.034203595807 | 1/0/38/10 RL calf | −0.837759613991 / −2.91179257e−7 |

表中独立`deg*pi/180`与生产`math.radians`末位差约4e−16，低于冻结1e−12；不是不同物理量。
三类均有见证；首个普通内侧均为r1.e0 step1 FL hip。微小负裕量是相对名义边界的读数差，
不能判为运行时违反约束。K3 RL calf实际新句（完整报告中，未投递）：

```text
RL_calf_joint: 175661/298375 finite recorded q samples lie inside the nominal USD 5% edge band; 54635/298375 lie outside nominal bounds. Episode-equal median minimum signed margin -0.0052905rad; median range occupancy 0.406794. Unknown samples 0/298375. Effective runtime limits unknown; no saturation or cause claim.
```

| 源 | 回合 / 有效样本行 / 关节记录 | preview字节 | 新句纳入 / 候选 |
|---|---|---:|---:|
| K1 | 864 / 276259 / 10368 | 8086 | 0 / 12 |
| K2 | 864 / 276949 / 10368 | 8032 | 0 / 12 |
| K3 | 864 / 298375 / 10368 | 8127 | 0 / 12 |

合计2,592回合、851,583样本行、10,218,996关节读数。真实preview保留整个旧public，18必保槽
未挤掉；36新句全部因预算省略。合成短base能装入；精确8192B接收，第一句导致8193B则整句
省略。**生产投递仍是后续专门阶段。**

## 精确执行、成本和兼容

在`D:\humanoid_agent\true_RL_Agent`执行（复跑须新目录）：

```powershell
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c2/run_validation.py --mode tests --output outputs/p2c2/tests_02
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c2/run_validation.py --mode real --output outputs/p2c2/real_02
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c2/assemble_verification.py
```

assemble只索引已有结果，create-once，不覆盖现索引。只读SSH argv、stdin脚本SHA、真实stdout
及墙钟见[verification.json](verification.json)。没有安装或改变服务器环境。

```text
limits           exit 0  9.087s  Ran 13 tests — OK
compatibility    exit 0 61.858s  Ran 158 tests — OK
frozen_artifacts exit 0  3.668s  PASS
build_k1         exit 0  5.598s  oracle_k1 exit 0 8.948s PASS
build_k2         exit 0  5.567s  oracle_k2 exit 0 8.993s PASS
build_k3         exit 0  5.851s  oracle_k3 exit 0 9.051s PASS
```

171项实际测试，0 FAIL/ERROR/SKIP。P2a 33产物、P1b三卡文件、Push22冻结文件SHA不变，三源
旧full/public实际重放一致；包含旧周期/语义绑定回归，没有将历史247/309门称本次重跑。
最终9条本地命令合计118.6215s、进程峰值460,210,176B。三构建内部wall5.186/5.129/5.429s，
CPU5.141/5.094/5.328s，峰值184,610,816 / 165,793,792 / 192,352,256B；四类最终产物合计
127,838,624B（不含manifest/audit）。CPU资产组成核查0.509s、峰值120,373,248B；历史源码
核查0.014s。SSH含连接约7–9s/命令。API/功能LLM/训练/GPU/服务器写入均0。

探索与开发失败不计入118.6215s。保留tests_01反例异常类型期望错误（产品已拒绝）、real_01
oracle遗漏quantile_method字段的原日志；更早临时测试修过mappingproxy deepcopy、fixture
288回合期望、远端Gf四元数API和历史源码精确断言。未删除旧产物或改写原日志。内部只读
审查分别核历史source/零框架和实现反例，初审两问题已复核关闭，不能替代父级阶段门。

## 停止点和剩余边界

commit/push核远端后停写，等父线程独立验收。不进入P2c3、K5/LLM结果、训练或新GPU采集。
名义坐标无映射阻塞；有效硬/软限位继续unknown。后续最小采集：同joint_names/IDs、env、
时钟相位下的实际硬限位与soft_joint_pos_limits、资产/版本；限位随机化或写入时记录新值
和生效步（不变时每环境一次加版本）。未来资产零位/拓扑静态不可证时，才需要同步DOF q
与两侧joint frame的受控校验。当前名义代理无需新采集，本阶段未执行采集。

已读用户级`C:/Users/lover/.codex/AGENTS.md`并使用CodeGraph；仓库内无AGENTS，`.agents`/
`.codex`正常权限为空。采用ponytail最小增量技能，无新依赖/框架。本线程唯一repo writer；
Windows深路径假删除用正常权限核实，不恢复/删除，不纳入提交。
