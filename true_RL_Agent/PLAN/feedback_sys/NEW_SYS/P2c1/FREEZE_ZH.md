# P2c1：读取序列的局部周期资格（先冻结，后原数组核查）

起点 `ca140cfd1e6084118e0c36a4c28f8fe97a2dc294`，2026-10-03。
依据主计划 §6.5、P0/M06 deferred、P2a 重触发补正。开发前只读了契约、源码和既有文档；
尚未读取本阶段真实 NPZ。使用 P2a source_lock 的 K1/K2/K3 全部 public-dev episodes，
不读 K5、P3 pilot，不按结果/成功率调规则。数据是已知开发材料，不叫新 holdout。

## 范围与版本

显式增量 catalog `trajectory.p0.measurements.v3`，base catalog 是冻结 v2。
新入口只输出 M06 周期/相位的离线补充证据和报告，不改 v2 数值/卡片/模板/正常投递入口。
复用 NumericEpisode 的时钟、实体、source reference 和 Stair adapter；复用 cohort distribution。
Push 22 个冻结文件（包含旧 shared 文件）不改。新 shared 文件会改变 Stair 未完成构建的 code key，
已封存 replay 不重算。新规则有自己的 oracle，不重写 P0 truth 或旧审计证据。

## 冻结定义

输入只取 current `support_contact_forces_w`，有限 XYZ 的 Euclidean norm 严格 > task 的 1 N。
单位 N/s；实体顺序来自 adapter；时钟为 recorder 控制步读取，不是物理落地时间。
history 绝不展开为子步。任一 NaN/Inf 或缺步切断该足的已知连续岛，不插值。

1. **完整读取周期**：同足、同已知连续岛内相邻两个已观测 0→1 的 onset。
   onset 使用第一个 on 的 step，并保存前一 off 到此 on 的读取边界 `(t[k-1],t[k]]`。
   周期跨度为两 onset 的步数差乘 dt；首帧已 on 不补 onset，首/尾未闭合部分单列，不能算周期。
2. **局部节律候选**：每连续 4 个完整周期（5 个 onset）为一个窗口，逐个滑动枚举全部候选。
   不删异常周期、不按 outcome 挑窗口。只对这个窗口说节律，不合并窗口后声称整回合周期稳定。
3. **稳定性**：以整数步周期计算，population CV ≤ 0.15，且
   `max(abs(period - median(period))) / median(period) ≤ 0.20`，两者均需通过。
   最少 4 周期为有限重复证据，CV/偏差防局部变异和单个离群；不是置信度或生物步态标准。
4. **去抖敏感性**：原标记不改；分别从原始 runs 一遍替换有两侧已知同状态的完整
   1 样本、≤2 样本短 run，两个变体互不串联，不跨岛、不删边界截尾 run。
   记录两变体全部转换与改变的样本。主值始终来自 raw。
   候选 `[a,b)` 的资格另要求 a 前 3 条连续 off、b 起 3 条连续 on 已可见；
   并且两变体在 `[a-3,b+3)` 与原标记逐条相同。末 on 只有 1/2 条仍有完整读取周期，
   但缺去抖边界上下文，不能获得稳定节律资格。3 条是读取分辨率条件，不是物理接触持续 60 ms。
5. **相位**：角色序较低者为参考。只对合格的四周期窗口计算；每个半开参考周期必须恰好
   一个目标 onset，目标对应的起始周期也须属于目标的某个合格窗口。所选目标 onset 所启动的
   cycle 及其资格窗 ID 均保留，未知/缺步不得跨越参考窗。`phi=(target-ref)/(next_ref-ref)`。
   保存全部 4 个相位、`R=abs(mean(exp(2πi phi)))`；R≥0.9 才报告圆周均值 `[0,1)`。
   零/多个目标事件、目标周期无资格、覆盖不足或低集中度分别记录拒绝；不使用 nearest-onset
   的可重复配对，不把相位改称落地同步、步态名称、物理接触或机制。
6. **分母**：完整 cycle 全保留（即使不规则）；窗口计数和唯一合格 cycle 计数分开。
   重叠窗口的 `[onset_a,onset_b)` 覆盖按边并集计算，不累加窗口时长。
   输出每足 total/known/unknown rows、missing steps、known adjacent edges、覆盖边数和拒绝原因。
   完整周期不等于已资格周期；无资格不能说“无周期运动”。跨回合先等权，池化边数另列。

上述常数来源统一为 `development_calibration: predeclared_resolution_and_variation_rule`；
不是从成功率拟合，也没有误报概率保证。比较采用数学闭边界，浮点计算仅容许 1e-12 舍入误差。

## 独立验收与交付

合成 known truth：周期 20 步、同占比异次序、不规则、仅 1/3 周期、CV 与 median deviation
单独边界、1/2 样本 on/off 重触发、缺步/NaN/Inf、首尾截尾、dt 改变、多实体重排、相位 wrap/
多目标/缺目标/跨 gap。独立 oracle 从原向量计算 marker、run、cycle、资格和圆周统计，不导入
生产 cycle 公式。真实三源全队列独立重算；预声明取遍历序第一条合格/不合格/短重触发/
边界不足/相位见证，空类明确未找到，不改规则补齐。

新增句全部 optional。离线候选投影先完整保留旧 public.v5，再在同一个 8192 UTF-8 byte
预算内加入整句；空间不足可全省略，省略理由与来源进 private receipt。不把独立补充报告
称作已进入正常作者 payload；Push 12288 和原有必保槽不变。

P2c1 自测、commit/push 后停写，待 root 独立 gate。P2c2 再核 USD 轴/单位/零点/关节顺序及
archive q 映射，只允许名义裕量代理；实际随机化/有效限位仍需补采。P2c3 再逐类核 terrain
origin/XY/Z/边界及 root/foot 参考点高度差，不叫脚底间隙或碰撞距离。历史 link/COM 位置映射
未证明时继续按原通道措辞。父代比较、action/reward profile、LLM/API/train/GPU 均不在本轮。
