# P2c3：局部名义地形参考高度，限定离线交付

起点 `68a60710a9d13534460a7f53755106189ad20bd3`；先提交预冻结 `574b841c`。
开发自测和只读 Astra ultra 审阅通过，**等待父级独立验收**。没有生产投递。
用户已确认 P2c1/P2c2 原代码、原数组及独立结果仅限定离线 PASS；本阶段不提升其结论。
P2c2 的 5% 是浮点构造端点闭区间，不声称精确实数逐位等价。

交付入口：[测量实现](../../../../src/true_rl_agent/tasks/stair_climbing/trajectory_terrain.py)、
[离线构建](../../../../scripts/build_stair_terrain_supplement.py)、[原数组审计](audit_raw.py)、
[独立三角面 oracle](../../../../tests/unit/terrain_mesh_oracle.py)、
[原始例子](REAL_EXAMPLES.json)、[来源与代码片段](TERRAIN_SOURCE_EVIDENCE.json)、[执行证据](verification.json)。

## 证据支持的范围与一项收窄

历史 K1/K2/K3 job 的 `metadata.work_dir`、result numeric ref、原 manifest SHA 均闭合。
只读 SSH 提取历史 task source，在单次 CPU Python 进程内执行历史 mesh 函数的原 AST，
以各自 manifest 中的 training/evaluation 配置重建 public_dev 的全部 24 个 cell：
12 个 C0、9 个 C1、3 个确定性 C2。每 cell 12 env，共 288 env/seed、3 seed/source。
未启动 Isaac Sim、GPU 或 rollout；这不是新采集。

源任务定义的 `size=[8,4]m`，起始平台 `S=1.5m`，floor depth `0.5m`，8 级台阶；
具体 riser/tread/C2 顶点参数在来源证据逐 cell 保存，不能用 scalar 高度替换不规则序列。
ordered evaluation 的 C0/C1 均走 irregular-flat mesh，C2 走指定三角 mesh。
原 generator 为一行 24 列、border=0、use_cache=False，明确忽略 difficulty。
历史网格 origin 原代码：

```python
return meshes, np.array(
    [start_platform_length / 2.0, size_y / 2.0, 0.0], dtype=np.float64
)
faces.extend(((v00, v10, v11), (v00, v11, v01)))
```

源文件实际位置如 `/data/yuyangliu/AGENT/feedback_p2_audit_20260924/source/src/true_rl_agent/tasks/stair_climbing/isaaclab_rough_terrain.py:106`、`:223`。
三个 snapshot 的生成函数、配置展开/映射均与当前复用文件 raw SHA 一致；runtime recorder
文件有历史接触历史/画面采集差异，地形构建路径一致。历史源原文、SHA、行号及实际生成的
全部 vertices/faces 在 `outputs/p2c3/historical_02/stdout.json`，9,961,316B，SHA：
`84aecf9e412be7578ee567ca65cc1c6ff48351f8f45c1fb789a1a61913e8a48f`。

**预冻结后的证据驱动收窄：** `/data/yuyangliu/vendor/IsaacLab` 不是 Git 仓库，
`git -C ... rev-parse HEAD` 原输出 exit 128，无法核实 composition 声明的历史 revision
`3c6e67bb5c7ada942a6d1884ab69338f57596f77`。当前 vendor 的三次平移与预期 layout 相符，
但不能称其为已验证的历史 importer。当前 importer 把原 vertices/faces 写成 USD Mesh，
没有在该路径明确固定 cook/approximation/welding；contact/rest offset 的 `None` 是未写，
不是零。没有历史场景或 cooked mesh 快照。

因此本次证明的是：**历史任务配置定义的名义上表面，按名义 world 轴和记录 origin 放置后的
参考高度及相对竖直高度**。它不是历史物理 runtime 表面的重建证明。报告 binding 明写
`nominal_expected_layout_with_recorded_origins; historical_importer_bytes_unverified`，
runtime surface 全部 unknown。未来需先获得历史安装源字节/版本与原场景/collision 参数，
或在另行授权的最小采集里记录生成器版本、资产/mesh SHA、最终 world transform、cell/type/
origin 与实际碰撞配置；若要主张脚底/接触距离，还需真实足底碰撞几何和相应距离/接触记录。
本阶段未请求或进行新采集。

## 计算、边界与最小集成

新增独立 task 模块及离线 runner，catalog v5/base v2；不改旧 `terrain_evidence.py` 的
越界延拓/边界选侧，不改 P0/P2a/Push、已有生产开关或投递优先级。
每点用自己的 XY；base 来自已绑定 articulation root，四足只称记录的 rigid-body reference point。
实际实现片段：

```python
local = points - origin
local[:, :2] += [geometry['start_platform_length_m']/2, geometry['size_m'][1]/2]
boundary |= (x >= edge-guard) & (x <= edge+guard)
height[known] += origin[2]
return height, points[:, 2]-height, status
```

flat 依历史 mesh 的逐次浮点累加；C2 复用历史已核的三角线性公式，正弦只生成顶点。
foot/reference z 减的是本点 XY 下的名义面，不是所有足取最低面、base 支撑代理或距离范数。
patch 外 unknown；四周与所有 tread 接缝的 `1e-4m` 闭带 unknown；此带是排除规则，非
安全阈值或误差保证。端点按浮点构造坐标区间比较；不宣称精确实数判断。
三角形内共享边连续可算；立面及台阶多高度交界被排除，任意未知曲面不猜平面/零。
越出所属 patch 时不自动改用相邻 cell 的地形，所属域之外仍为 unknown。
缺原点、缺点坐标、NaN/Inf、未资格化几何均保留 unknown，JSON 写 null；畸形几何或
明确单位/frame/entity/spec/type/source 错配拒绝。负高度差保留，不称穿透。

每回合 5 个记录含 full trace 对应范围、counts/unknown 原因、sample-cell 秒数、统计量、
首极值/首状态见证和原 chunk/row/env/channel/column；origin/type 均有静态原数组引用。
汇总按 terrain regime×实体；回合等权分布、池化样本及池化 sample-cell 时间分开。
终点计入、padding 排除；没有把 count×dt 当连续物理持续时间。

离线 runner 只接受原 archive，由测量生成句子；历史 qualification 和 P2a public 完整 SHA
固定在该离线 runner。它们**不是最终生产准入设计**。生产后续必须以真实采集契约与
资产/配置资格替换专用历史锁，不能永久依赖这三个 manifest 或 P2a 文件路径。
本次未扩建历史白名单框架，也未借修改旧 public 文字释放预算。

## 独立验收、真实覆盖与缺口

oracle 直接核原 manifest/87 个 chunk SHA，读 NPZ，不导入生产 translator、surface helper
或新测量代码。以历史 CPU 实际生成的每个 upward triangle 做 2×2 重心坐标解，取名义外部
上表面；独立比较所有 trace、12,960 条记录、汇总和 45 句。当前 51,840 个首极值索引
也与独立面计算完全一致；另外验证 trace 首索引排序并拒绝“同高后行替代首行”的重签反例。
与另一种 3×3 代数相比，平面并列可出现 1 ulp 的首索引变化；不把不同代数称逐位等价。

| 来源 | 回合/有效行 | 参考点样本 | 名义 known | 边界带 unknown | patch 外 unknown | 新句进入 preview |
|---|---:|---:|---:|---:|---:|---:|
| K1 | 864 / 276259 | 1381295 | 1380447 | 319 | 529 | 0/15 |
| K2 | 864 / 276949 | 1384745 | 1382654 | 264 | 1827 | 0/15 |
| K3 | 864 / 298375 | 1491875 | 1490227 | 300 | 1348 | 0/15 |

合计 4,257,915 点，4,253,328 nominal known、4,587 unknown。
三源的最大公式/三角面差均 `1.5543122344752192e-15m`，小于预冻结 `1e-10m`。
这里 known 只指名义模型，实际 runtime 面不是 known。
真实三源没有缺 origin、非有限坐标、未知生成器样本；这些只由合成测试覆盖。
预声明平台/tread/top/unknown 三种 regime 均找到；C0/C1 未找到负差真实例，明确空类。

18 项新测试覆盖全 24 cell 的全部顶面中心/顶点/边中点、非零 XYZ origin/平移、负差、
不同实体落不同阶、闭带端点及 nextafter、四边外部、NaN/Inf/缺原点、缺几何、错单位/frame/
实体、坏spec/type/origin/source、重签 metadata/文字/首见证和不等长回合分母。
另 171 项相关兼容测试通过，0 FAIL/ERROR/SKIP；冻结检查 P2a 33 文件、P1b 3 文件、
Push 22 文件 SHA 不变，三源旧 full/public 实际 replay 相等。
没有把既有 247/309 计入本轮重跑；没有全仓、盲测或完整 E2 声明。
普通开发环境缺历史大文件时相关测试可显式 skip；本阶段 runner 在缺文件时拒绝启动，
本次真实执行无 skip。

## 可以亲查的原始例子

预声明首 boundary 例：K1 r1/e0/step228，RL_foot（support column 2），
`chunk_000003.npz` row35，SHA `a0e509c79f8e7a2bf67d044f3adea1a9b7b6fd3ec5c1334856d8373b4845eeb0`。
原位置 `[-0.10009574890136719,-46.00204086303711,1.0632357597351074]`，
origin `[-3.25,-46,0]`，mesh x=`3.899904251098633`，落在 x=3.9 的闭带内。
高度及差值为 null，不能沿用旧低侧/高侧约定。完整五点原向量在 REAL_EXAMPLES。

另外只读审阅者独立选择的非平面 C2 例（非预声明首例）：K1 r1/e111/step41，FL_foot，
type9，`public_dev.c2_mechanism_ood_level_2`，`chunk_000000.npz` row40，SHA
`5518421701fb1c13944be8a6c9296d415a3493ca35394b8e0a6f502ae05fe4c0`。

```text
p_world = [-2.4434642791748047, -9.795625686645508, 0.313984215259552]
origin  = [-3.25, -10, 0]
mesh_xy = [1.5565357208251953, 2.204374313354492]
mesh1 face345 vertices =
  [1.5333333333333334, 2.142857142857143, 0.2105411021450036]
  [1.5666666666666667, 2.2142857142857144, 0.1716303397842156]
  [1.5333333333333334, 2.2142857142857144, 0.2104358300812905]
barycentric = [0.13875961303710888, 0.6960716247558674, 0.1651687622070237]
h_nominal = 0.18343903691158198 m
relative_z = 0.13054517834797003 m
```

生产 trace 对应全源 row33843/列1，h=`0.1834390369115823`、差=`0.1305451783479697`，
误差 `3.33e-16m`。可运行 [independent_spotcheck.py](independent_spotcheck.py) 重复该独立
3×3 面解；原 stdout 在 `outputs/p2c3/final_audit_01/independent_spotcheck.stdout.txt`。
真实负差例 K1 r1/e108/step247 FL：h=`1.0241138227609154`，差=`-0.001470137420034101m`；
它仍只是参考点相对名义面的竖直差。

实际新增候选句之一：

```text
c2_rough / robot_0.support.FL_foot: nominal terrain height qualified at 40870/40905 recorded reference-point samples; episode-equal median mean vertical difference 0.0625019m. Unknown samples 35. Reconstructed nominal surface only; runtime surface unknown. Not sole clearance, contact distance, or a cause claim.
```

三源 preview 仍为 8086/8032/8127 UTF-8 canonical bytes，45 新句全省略；receipt 逐句留 ID
和理由。这是离线集成及预算拒绝证据，不是可用的正常生产投递功能。

## 实际命令、成本与开发失败

在 `D:\humanoid_agent\true_RL_Agent`，`PYTHONUTF8=1`，最终相关命令：

```powershell
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/capture_remote.py PLAN/feedback_sys/NEW_SYS/P2c3/read_historical_terrain.py outputs/p2c3/historical_02
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/capture_remote.py PLAN/feedback_sys/NEW_SYS/P2c3/read_vendor.py outputs/p2c3/vendor_01
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/run_validation.py --mode tests --output outputs/p2c3/tests_03
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/run_validation.py --mode real --output outputs/p2c3/real_03
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/run_validation.py --mode final_audit --output outputs/p2c3/final_audit_01
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/run_validation.py --mode raw_audit --output outputs/p2c3/raw_audit_03
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/run_validation.py --mode terrain --output outputs/p2c3/tests_04
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c3/assemble_verification.py
```

目录均 create-once；复跑须换新目录，不能覆盖这些原输出。raw_audit 模式只读 real_03
产物重核；精确 logical argv、原 stdout/stderr、SHA、墙钟、实际 Python 峰值均在 verification。
构建用时 K1/K2/K3 为 8.652/7.714/8.684s；最终 oracle 为 11.540/11.402/12.000s；
相关兼容套件 73.567s。最终命令成本精确汇总见 verification，不把重复重核再计为新增测试。
峰值 460,607,488B，三源最终 build artifact 合计 99,526,286B；原始大文件留 Git 忽略目录。
历史网格成功提取内部 CPU 1.914s、wall 1.916s、峰值190,345,216B；SSH 进程30.745s。
任务 API/功能 LLM/训练/GPU/新采集/服务器写入均 0。

保留的开发失败：tests_01/real_01 发现 read_text 规范化混合换行使源码文本与 raw SHA 不同，
改为原字节解码并重取 historical_02；real_02 在末尾 manifest 记录中遇 runpy 相对 __file__，
已 resolve；tests_02 的 hash 变异 mock 未命中实际读取函数，修正测试夹具。
没有改判成功或覆盖失败日志。初期只读审阅指出非有限见证/松散资格/畸形几何门，均修正后
才出最终产物；最终 oracle 额外实测独立首索引差异为0，移除借 trace 覆盖独立索引的旧检查。

本地用户 AGENTS 已读；CodeGraph 先定位，ponytail 限制实现规模，无新依赖或通用框架。
Windows 沙箱假报32深路径删除，正常权限 status 起点干净；只暂存明确文件，未恢复、删除或
提交假删除。授权范围内提交推送核远端后停止写入，等待父级独立验收，不推进后续阶段。
