# P2c2：关节名义 USD 限位诊断预冻结

起点 `71c9cac31664ce32bb58c0804d33efa2ff8c0dd1`，分支
`codex/trajectory-translator-upgrade`。用户已确认 P2c1 限定离线 PASS；新周期句仍全部被
8192B 预算省略，不能称为已投递。此文件先于本阶段原 q 数组读取、USD 数值提取和实现保存。

## 数据和资格门

仅用 P2a/source_lock.json 已冻结 K1/K2/K3 的全部 public_dev 回合，已知开发材料，非 holdout。
沿用旧 adapter 的 archive SHA、实体投影、有效首回合、pre-reset 终点与原数组定位校验。
资产取 profile 的 used-layer closure：root SHA
`ba171c972b987d8c8fb7157ccad2ba9c0c1fed105755d2d8af46bef96cc11c6d`，mesh SHA
`2902646d0f4c13c9ecae3ac9046e32c7d18902eef9679de9829f642a15c93fb5`。
实际只读 SSH 已核 SHA；用户 root 缓存路径缺少 f2，真实路径为
`/home/yuyangliu/.cache/ov/client/https/f1ee52af0de7caec907cf4a6317543f0b852f5f2a2741d46d3f39dc0a781225c.usd`。
文件名不作为匹配依据。必须核 root 引用、layer closure、RevoluteJoint prim/body0/body1、
axis/localPos/localRot、度→弧度、零坐标及 archive q 的名字/列/单位/是否减默认姿态。
默认站姿不能作零位。不能证明的映射须阻塞或显式 unknown，并列最小补采项。
archive 的声明资产身份只证明所绑定的名义资产，不证明每环境实际运行限位。

## 冻结公式和语义

增量 catalog `trajectory.p0.measurements.v4`，base v2，独立 M09 supplement，不修改 v2 或 v3。
对每个已绑定关节的有限 q（rad），由已核 USD 得 L/U（rad），W=U-L>0：

- 下/上有符号距离 `q-L`, `U-q`；最近名义边界有符号裕量 `min(q-L,U-q)`，外侧为负。
- 名义区间位置 `u=(q-L)/W`，不裁剪；0/1 是名义边界，越界保留 <0/>1。
- 回合范围占用率 `(max(q)-min(q))/W`，不是时间比例，也不裁剪。
- 名义近边界采样带：`0 <= margin <= 0.05*W`（闭边界）；外侧 `margin<0` 单列，
  不把所有外侧样本混进近边界。5% 是预声明描述分带，非安全阈值/运行限位判断。
- 每关节独立 total/finite/unknown rows、近边界/外侧分子和有限分母、比例、`count*dt`
  采样格时长代理；缺步不补值、不推物理连续持续。无有效 q 或映射不明返回 null/unknown。
- 极值保留首个见证 q、step、原 chunk/row/env 引用；所有有效样本参与，包括 pre-reset 终点；
  padding 不参与。跨回合先等权分布；另报池化 count 与格时间分母，不能混为一个权重。

文句只称 nominal USD bounds / sampled band / recorded q，不称实际触限、卡死、饱和、故障或原因。
有效硬/软限位、随机化后限位、执行器动作/力矩没有记录，始终 unknown。

## 验收预期和交付

先从资产独立 CPU 解析原始字段，再以不导入生产数值函数的原 NPZ oracle 重算三源全部关节。
预声明每源按 run/env/step/关节序取首个内侧近带、外侧、普通内侧见证；空类如实记未找到。
合成例覆盖多实体不同范围/列重排、度/弧度错配、未知/缺失/重复名字、错误资产/来源/句子重签、
L/U 相等或反向、精确 L/U/中点/5% 边界、边界外、NaN/Inf、缺步、单样本与全 unknown。
对极值/公式误差上限 1e-12（float64 从原 float32 输入计算），计数/实体/引用严格一致。
资产值的独立解析与度转换单独验，不拿生产常数当真值。

离线报告和证据必须有来源/资产/规则绑定；预算预览仅由已绑定测量重建句子，不能接收任意文字。
保留整个旧 public.v5，再在合计8192B内尝试整句；可全部省略，记录 receipt，不声称生产投递。
原 P0/P1/P2 v2 产物、Push 22冻结文件和12288B预算不改；独立核 SHA 和实际旧 full/public 重放。
记录实际 argv/退出码/日志SHA、墙钟/CPU/峰值/存储成本。只读 SSH、无新采集，无API/LLM/训练/GPU。
若名义映射门失败，交付可审 unknown 与最小补采方案，不猜数值。commit/push 核远端后停写，
等父线程独立验收；不进入 P2c3/K5/模型结果。

## 实现审查补记（原冻结后，最终验收前）

只读合成审查发现真实USD浮点边界的减法舍入可把精确5%带端点排除。因此保持数学定义，
实现用等价闭区间 `L<=q<=U AND (q<=L+.05W OR q>=U-.05W)`，不新增容差，外侧仍严格分开。
第二项调整：本阶段只资格化上述三份历史档案及 P2a 冻结 public 的完整规范JSON SHA。
入口只接收 raw manifest ref 并自己生成测量/句子，不接收 caller 自报 supplement/records/text；
旧 public 也核外部冻结内容hash，拒绝改写文字后重签自hash。新源明确报未资格，后续生产
投递须独立实现动态来源语义资格，不能冒用本阶段白名单作为通用能力。
