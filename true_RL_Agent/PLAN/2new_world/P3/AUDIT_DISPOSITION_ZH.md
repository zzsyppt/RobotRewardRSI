# P3 复审逐项处置

2026-09-09。以当前任务 × 机器人边界判断，不直接把建议当需求。验收结果及实物引用统一写入 [ACCEPTANCE_ZH.md](ACCEPTANCE_ZH.md)；本表保留前一轮问题的根因与处理。其后修复目标全量/default fixture、C2 卷间比例和版本表达，见 [回归复审](REGRESSION_FOLLOWUP_ZH.md)。当前比例修订后的资格是 G1 pass、Go2 NO-GO；此前两个本体均通过的记录属于旧 suite hash，不能为新套卷签发资格。

| 审计项 | 判断与源头处置 |
|---|---|
| 天花板主分、easy anchor、phase 饱和 | `fixed`：当前研究 J_primary=J_OOD；ID/phase 单独守卫。目标选转折附近，控制选各 pattern 最难且可通过的档位。并不永久排除未来机器人的 ID 改善；改变难度必须保持 ID/OOD 声明真实。 |
| 9 episodes 的薄标定 | `fixed`：每 cell 36 episodes，公开粗定位再一次细化；正式每 seed 12 环境，共 288 envs。二项概率原说法不准确，复算与独立性限制见复审计划。 |
| 亚毫米/微米 cohort 差异 | `fixed`：采用 L 比例的有实际尺寸差异的高度/踏深与 surface phase；平面对照和 C2 共用变换。实际尺寸与难度用新实测报告，不能仅凭 hash 宣称泛化。 |
| tread 端点和新机器人需手写旧 suite | `fixed`：从 d0 有界收窄，工厂直接消费 RobotProfile、训练 profile 与八个任务 stratum；合成 L 范围反例通过。训练绝对尺寸保持训练身份，不暗改已有 checkpoint 条件。 |
| 一次性脚本与缺 onboarding | `retired/fixed`：三个旧 P3 prepare/calibrate 脚本退役，保留的源码包能解释历史运行。新的两个 scripts 入口负责生成/修订和生产固定 reference 标定；任务指南新增接入步骤。 |
| 4.1 匹配 C2 平面控制可消失 | `fixed`：选择时强制包含实际选中的控制；suite 冻结和 private round-trip 使用实际几何检查。较低 ID 基底反例覆盖。 |
| 4.2 C1 level-5 特例 | `retired`：移除已撤回基底方案分支，正式 C1 按 pattern 保留覆盖。 |
| 4.3 裸 15 mm 守卫 | `retired`：删除额外 legacy 常量门，统一使用表面规范和相对 riser 的合法性；旧文件身份保持原样。 |
| 4.4 公式多源 | `fixed`：历史少 mean 的拼写只留具名 legacy 常量以保持摘要字节；新摘要读 suite.score_contract。 |
| 4.5 成功进度硬编码 | `fixed`：episode facts 校验读取任务 semantics 的 minimum_progress_fraction；0.9 阈值反例覆盖。 |
| 4.6 private dict/cells 不一致 | `fixed`：删除特殊 suite 子类，实例化普通冻结 suite；实际 geometry/hash/cells/to_dict 可 round-trip。source_suite_hash 明确表示公开来源，私有实例不会伪装成公开 suite。 |
| 4.7 mask qualified 含义冲突 | `fixed`：当前 family 局部证据名为 has_success_transition，列出 learnable strata；总体签发仅在每个目标 stratum 有转折时通过。历史 mask 按原语义读取。 |
| 无消费者的 completion 派生值 | `retired`：删除 mean_efficiency、censored mean 和 range；保留实际消费的 episode facts，步数不进分数。 |
| 多版本 mask 与投影身份 | `retained_with_required_consumer`：旧 mask 由历史 summary/decision reader 使用；当前 mask 仍由 public_result 投影及 qualification 复算消费。全有全无是资格结果，不能替代来源/分层证据与不可变投影身份。为删除少量字段重写公共结果协议不符合本轮复杂度边界。 |
| V3 binding 近似重复 | `retained_with_required_consumer`：版本 binding 是 ResolvedCellManifest 冻结输入，继承训练条件由 lineage 等价验证。保留约 5 KB 的显式版本数据；不为节省配置字节新增继承 DSL 或修改历史 hash。 |
| lineage 加载成本与环检测 | `retained_with_required_consumer`：仅三层 lineage，当前成本可接受；环检测给出编程错误原因。未引入缓存和新的配置状态。 |
| 三个默认入口仍 V2 | `fixed`：唯一 DEFAULT_STAIR_CELL_VARIANT_ID 由 catalog 提供，默认协议从该行派生；新入口选 V3。历史常量显式命名 V2，既有 fixture/reader 继续选旧协议。 |
| P2 九项复核 | `retained`：已有独立处置与验收，本轮不重新打开已解决分支。必要视频/运行时专项随 P3 当前源码复核。 |

## 复审的结论边界

尺度只初始化搜索，不等于运动学可达范围。当前机器人事实缺少可靠足底碰撞尺寸和最大步长，未增加虚假的机器人参数；同一任务方法分别用各机器人实测标定。宽度、平台与训练踏深等明确属于训练/任务条件，新机器人的训练条件须独立设计并训练 reference。

增加难度发生在公开标定，保持成功/progress 的简单目标。正式冻结后不看 candidate 或 sealed 调题；若独立资格没有转折则 NO-GO。旧 15 rounds 的所有保护 checkpoint、原协议与 ledger 不作为本次清理对象。

## 扩大审计的补充发现

- 粗梯度与固定 cohort 位移同网格时，留出题可能回到 calibration 几何。冻结检查真实拦截了三处，正式运行前修订共用位移规则并重新冻结；没有看 formal 成绩改题。规则不对任意未来训练配置承诺自动成功，新机器人冻结仍须通过同一分离检查。
- 默认入口切到 V3 后，一个旧 final 入口测试仍只切 namespace、不创建私有计划。测试已改为经过真实 generation final plan 的承诺路径，产品的“缺计划拒绝”边界保持有效。此前 1631 项通过仅指本地完整回归，不能替代目标 Python 全量；目标失败及后续修复统一见回归复审，不再以手选专项声称完整覆盖。
- 实际位移按几何约束解析后重新计算，不以名义系数代替实际值。相对 calibration 的 max(高度、踏深) 变化：Go2 2.624–10.570 mm，G1 4.507–17.942 mm；Go2 某 joint 题高度受 OOD 边界折返后仅 0.423 mm，但同题踏深变化 10.569 mm。这是实际题面差异，仍不单独证明难度可比或普遍泛化。
- 任务指南仍残留“默认入口交给 P4 切换”的旧描述，已与 catalog 的实际 V3 默认收敛；README、runbook、系统概览和主计划同步当前验收状态。P4 尚待完成的编排/成本边界继续明确保留。
