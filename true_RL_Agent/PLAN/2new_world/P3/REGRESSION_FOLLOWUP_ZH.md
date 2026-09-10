# P3 目标运行时回归与卷间比例复审

2026-09-09。此项是对已有 P3 的修正，不启动 P4 编排重构或新候选搜索。任务语义继续归 Stair；机器人事实、共享 runner 和成本合同不新增地形分支。

本文记录较早一次粗糙度比例修复及其原始验收。随后发现的 C2 基底卷间偏移及修订见 [卷间构造审计](COHORT_ALIGNMENT_ZH.md)；活动声明与资格状态以 [当前验收](ACCEPTANCE_ZH.md)为准，不将本文的旧数值带入新 suite hash。

## 实施与验收顺序

1. 在目标 Python 复现 generation session 的三个失败。fixture 从当前默认行的科学成本基础派生上限；同步按比例构造合成 episode，保留接受、拒绝、硬退出、resume 和双 final 的原断言。增加无需 Isaac 即可执行的默认行成本检查。不能改成强选旧 V2 或放松产品成本相等性门。
2. 收敛 suite 版本判断的名称；episode facts 能力只服务事实记录，mask 的版本字段映射集中一处。补上 final 缺计划/用途不符拒绝的直接断言。缺平面对照只补具名错误，不建恢复或回退系统。manifest-free 入口保留原 G1 V2 历史契约；新入口必须使用 resolved cell；默认行只从 catalog 导出。
3. C2 位移保留 `amplitude / min(realized risers)`，涵盖规则与 C1 基底。沿用已测公开标定的选档结果，重新冻结当前 G1/Go2 套卷并生成新 hash；不改变波形、频率、reference、训练、评分或资格门。final 生成规则显式升版，已承诺的旧 final 仍按其原规则重建。旧套卷和测量保留原义，不能用旧资格冒充修订后通过。
4. 对新的公开/接受/留出实例重新执行两本体固定 reference 资格及 final 探针；结果未达原门即 NO-GO，不根据 accept/sealed 再调题。运行前固定新套卷及生成规则，保存源文件清单和原始回执。
5. 本地全量与目标 Python 全量均执行；目标显式开启 Isaac 动态测试，必须包含 generation session 三个场景且不能跳过。随后复查同类默认值/fixture、版本字段和几何变换，核对历史保护与资源收尾，更新当前验收正文。

## 判断边界

- 服务器实测已复现三个失败，成本相等性门正确，错误位于陈旧 fixture。专项集合通过不等于目标全量通过。
- A/riser 漂移属生成规则缺口，但比例对齐本身不证明难度相等，更不证明波纹形态符合自然楼梯。刚才渲染揭示的形态问题不在本次修复中改动。
- 旧 15 rounds 的协议、ledger、checkpoint 不修改。当前未启动搜索的 P3 套卷修订形成新的科学比较域，旧 P3 成绩不混入新资格。
- 不新增通用版本框架，不将轻量错误提示扩成防御工程；必要历史 reader 只为已有证据服务。

## 逐项处置

| 项目 | 根因与处理 |
|---|---|
| 三个目标集成失败 | fixture 的 caps 从默认行的科学成本基础构造，产品相等性门不变。继续跑到末端后发现同一 fixture 的另外两处旧约定：final episode 与 phase summary 使用不同配置、终态 replay 写死 V2。前者统一解析并传递实际 eval config，后者从唯一默认行取得预期身份；原接受、拒绝、硬退出、resume、双 final 和只读 replay 断言保留。V3 的完整 facts 使执行及报告复算超过原零散等待上限；两个 worker 统一采用原连续十五代场景已有的 900 秒上限，删除按动作加时的特例，产品预算与超时不变。 |
| 合成 reference 随样本量失真 | 原固定 2/3 个成功 episode 改为 2/3 的比例；当前能力控制保持成功，目标 stratum 承担候选选择信号。不是降低真实门，也不以旧 V2 回避默认行。 |
| 4.1 C2 比例漂移 | 按实际最小 riser 缩放 amplitude，规则和 C1 基底均覆盖。原选题及顺序、非 C2 几何、macro 尺寸、频率和波形不变。新 protocol/suite hash 与 final generator 版本明确分离；原 payload 和实际比例存于 geometry_revision.json。 |
| 4.2 final 缺计划 | 现有 final 入口正例旁新增两条直接拒绝断言：缺 plan、带 plan 但缺 final purpose。 |
| 4.3 裸 KeyError | 只增加具名 ValueError，指出缺失平面对照的 template，保留原异常原因；补一个直接反例。 |
| 4.4 版本开关 | calibrated suite 判断改用 is_calibrated；records_episode_facts 只留在实际 episode facts builder/validator。mask schema 与字段名由一个私有映射提供，未新增版本框架。 |
| 4.5 默认值与重复导出 | manifest-free fallback 有意保持原 G1 V2 历史契约并写明原因。新入口有 resolved cell；删除 robot_binding 的无消费者重复导出，默认行由 row_catalog 单源提供。 |
| 验收范围失真 | README 与两份 P3 实施计划同步要求目标 Python 全量并开启动态用例，三个 generation session 场景实际执行。49/32 项专项和本地环境跳过不能签发目标完整回归。 |
| 扩大审计：final 结果容量 | V3 完整结果为约 785,000 字节，系统自己的 final 原子发布被旧 512 KiB 上限拒绝。将 final result 既有的共用读写上限改为 2 MiB；不改变 plan/reservation、结果内容或通用文件层。一个本地测试用 G1/Go2 完整 V3 结果复现旧限额，再验证发布/重读，旧历史测试保留。 |
| 扩大审计：终态启动检查 | final 发布修复后，实际十五代终态检查进一步暴露 launch readiness 将公开 config 传给私有 final reader。只在这一个调用点复用既有 final 绑定；公开资格 mask 仍用公开 suite，其余生产 reader 已正确绑定。保留真实集成反例，不新增包装层或 mock 恒真测试。 |

## 验收状态

证据归档到 `results/regression_followup_20260909/`。工程修复验收完成：本地全量 1636 项通过、19 项环境跳过；目标 Python 全量 1636 项通过、0 跳过。完整 discovery 的用例 ID 逐项匹配，三个 generation session 场景实际完成，包括连续十五代 final 硬退出/恢复、reference 保持及双 final、三代暂停/恢复和终态审计。回执分别为 `local_verified.json` 与 `target_verified.json`，前面各轮失败回执保留。

相对进场快照，产品只改动 9 个 Stair 模块与 4 份 Stair 配置，公共层未变。本地/服务器的 478 个源码、配置和测试文件字节一致，另一个资格审计工具也一致；71 个历史 checkpoint 和 884 份保护产物未变。任务进程已结束，GPU compute 查询为空，四个既有 lease 均可获得；详见 `source_consistency.json`、`history_preservation.json`、`resource_closure.json`。

比例对齐后的独立 reference gate：G1 pass，Go2 fail。Go2 的 public-accept C2 三题均没有 20%–80% 成功转折，因此当前修订套卷为 NO-GO。保留结果与原门，不根据 accept/sealed 增大 amplitude、重抽几何或重复测量直到通过。Go2 后续需要明确的新公开地形设计/标定，再重新冻结独立资格卷；应同时处理此前渲染暴露的表面形态问题，不能把接近 riser 一半的波幅自动视作合理楼梯。

这项资格限制与 fixture 回归修复分别判断。P4 的 GPU 小时和输出空间仍须实测重签：step 算术对齐并不等于已证明新的 15 代预算可执行。
