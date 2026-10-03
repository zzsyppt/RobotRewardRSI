# P2c3：局部名义地形参考高度预冻结

起点 `68a60710a9d13534460a7f53755106189ad20bd3`，唯一 writer 为本执行线程。
先核历史源码/配置，再按此规则计算 K1/K2/K3 全部 public_dev 首回合；不筛 outcome。
这是已见材料的离线 qualification，不是生产接入、盲测或运行时物理形状验收。
旧 v2/P0/P2a/Push 与 P2c1/P2c2 均保持；P2c2 的 5% 是浮点构造端点闭区间。

## 几何资格与坐标

新增独立 M10 supplement，catalog `trajectory.p0.measurements.v5`，base 仍 v2。
必须闭合 archive SHA → 原 job/work_dir → 历史 generator/config/placement/capture 源码；
suite 原配置 → split 内 cell/spec/geometry_digest → terrain type → env 顺序 → 原 origin。
仅启用已核的 ordered evaluation、平移放置、米制/world +Z，禁止把机器人 yaw 当地形旋转。
现有 adapter 验证坐标、类型和 origin；补充证据另保留 type 静态原数组引用。

C0/C1 的 ordered 路径都是明确 riser/tread 序列的 flat mesh；已知 C2 是
`bounded_piecewise_linear_v1` 的确定性三角网格。只有历史顶点/face 可重建且独立验证
才资格化 C2；任意粗糙/未知生成器/非唯一曲面不能替换成平面或零。畸形契约拒绝，
有效但缺地形资格的输入报告 unknown。未记录的运行时碰撞烘焙、形变、接触偏移始终 unknown。

给记录参考点 `p_w`，原 terrain origin `o_w`，起始平台长 S、宽 W：

```text
x_mesh = p_w.x - o_w.x + S/2
y_mesh = p_w.y - o_w.y + W/2
h_world_nominal = o_w.z + h_mesh(x_mesh, y_mesh)
relative_z = p_w.z - h_world_nominal
```

origin 是起始平台中心，不是整个 patch 中心。沿历史 mesh 原浮点累加 riser/tread，
不用旧 query 的 round/ceil 延拓。flat 区间高度为累计 riser；C2 正弦只生成顶点，
面为 `(v00,v10,v11)` / `(v00,v11,v01)`，按三角形线性插值，不用连续正弦或 bilinear。
选取名义外部上表面；叠在 tread 下方的 floor 顶面不是另一个参考高度。

## 采样域与 unknown

预声明 `boundary_guard_m=1e-4`，是保守排除带，不是误差上界、碰撞或安全阈值。
patch 外严格 unknown；patch 四边、所有 tread 接缝（含最后 tread/top 接缝）距离
≤guard 的闭带 unknown，不猜边界归属。立面/多高度相交处由此排除。
共享三角形对角线/网格边若是同一连续高度则可计算；若证据表明多值则 unknown。
缺点坐标/NaN/Inf、缺 origin、缺/不支持几何资格分别记 unknown，数值输出 null。
只支持已证明的无旋转平移；未知/旋转坐标约定不默认为 identity。
参考点低于名义面时保留负值，绝不称穿透、接触距离或脚底 clearance。

## 事实、文字与验收

每回合每实体（记录 base、FL/FR/RL/RR 参考点）保留 total/known/unknown、原因计数、
known 的高度/relative_z 范围、均值和首个极值见证；见证有原点坐标、原位置、step、
chunk SHA/row/env/entity column、cell/spec/digest/type。unknown 不算零；终点纳入、padding
排除；不跨缺步积分。样本 count 与 count×dt 的 sample-cell 时间代理分列，非物理持续时间。
聚合按 geometry regime×实体，回合等权分布与池化样本/时间分母分开。
英文句只陈述 recorded reference point / reconstructed nominal surface / vertical difference；
明确 runtime surface unknown，以及无 sole/contact/causal 主张。

先提交此冻结再实现。原数组 oracle 不调用新增测量或旧 surface query；独立读取原 NPZ，
从历史 mesh 顶点/face 做重心插值，核三源全部有效点、全部实体事实/摘要/句子/原引用。
数值容差 1e-10 m（float64 名义公式/面插值），离散状态/实体/单位/frame/来源完全一致。
合成反例覆盖不同阶面、非零 XYZ origin、平移、单位/frame/entity/type/spec 错配、边界
及紧邻内外、NaN/Inf、unknown、C2 diagonal/vertex/两三角形、原点变化与相对高度辨别。
真实例按 run/env/step/entity 顺序取每 regime 的平台/tread/top/unknown 首例和负差首例；
找不到的类别明确空缺，不改规则造覆盖。

离线 preview 与已核原 public 合计 8192B，整句放入或整句省略并留 receipt；
不得接收自由文本作新事实。历史 public/manifest 锁仅用于本次离线审计；未来生产必须
改用真实采集契约与资产/配置资格，不能把本阶段历史 runner/白名单当生产终态。
原始命令/stdout/hash/耗时/内存、成功和开发失败均留档。0 API/LLM/训练/GPU/新采集，
SSH 只读。历史证据不足时收窄资格并记录最小补证条件，不伪造 runtime 验收。
commit/push 核远端后停写，等父级独立验收，不推进后续。
