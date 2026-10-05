# B：直接生成、明确父代与有界描述降级

B 接续已通过独立复验的 A（基线 `f23dae47`）。本阶段不改变 A 的行为定义和正文选择；五个开发队列 R4/K1/K2/m50/m300 的最终正文、数字及顺序逐项保持一致，仅把原父代占位句改为“父代匹配与数值差分见本次反馈；本卡描述当前队列。”实测及原始依据仍见 [A 修订作者全文](../A/REVISION_AUTHOR_EXAMPLES_ZH.md) 和 [A 原数组探针](../A/raw_probe.py)。五队列均为开发数据，不再称 unseen。

## 最终实现与取舍

Stair 当前路径为：已校验原始 archive → 数值算子 → 被消费的队列摘要/选中事件/父代比较量 → 同一组带 label、unit、统计口径的事实 → 作者正文。删除旧 base/card/supplement/每回合账目生成依赖及无人消费的区间关系。逐回合计算只在本次计算内暂存；持久化保留可重放正文的紧凑摘要、选中原数组定位和必要比较量。`behavior_full.json` 重放无需重新读取 raw，archive 的原始来源引用仍在内部。

Stair 仅保留当前 profile；CLI、composition、finalizer 从同一源读取 `trajectory_numeric_v2` 和 10240 bytes。该上限约束公开行为报告，不是整个 HTTP 请求。旧版本生成器不永久兼容；历史文档和运行结果原样留存，需要重现旧生成器时使用对应 Git 提交。

量纲、坐标系、实体次序、语义 contract、底层资产闭包继续校验；工作区源文件字节和 EOL 不再决定已有证据是否合格。历史 capsule 及关节边界元数据按解析内容绑定，没有新增历史白名单。首次 clean R4 检查仍发现 capsule 文件字节 hash 影响卡身份；`74248972` 将该处改为内容 hash 后，干净源码 checkout 的完整公开报告与工作区逐字相同。

`ReportBudgetExceeded` 与已明确的不支持 capture contract 可以降级：评估科学数值继续保留，作者收到来源绑定的 `description_status`。没有通用异常兜底；身份错配、原始 chunk 损坏仍报错。初始参考和候选反馈都接通。最后的初始参考实际请求检查曾被公共 metrics 白名单阻断，修复后重新通过，未把首次失败写成成功。

父代只取明确 guard 所指来源。比较要求相同任务 protocol、descriptor、unit 和完整条件 cell；末窗过程再要求同结局。绝对父/子值、差值、有效回合分母、unknown/不适用计数、双方结局组成一起保留。初始缺父代、缺比较量与不具可比条件分别说明。短末窗不当作完整 0.5s 均值；终点位置/倾角与完整末窗速度分开。报告选择最多三项按相对差异乘有效回合数排序，语义键打破平局，不按 archive ID 或 hash。它是组分布比较，不能解释为逐实例配对或因果改善。

Push 复用整条事实的预算选择、格式化和预算异常类型，保留任务自身恢复窗口/扰动/序列语义及正在使用的显式 profile。动态 strata 与 A 的高度/速度列修正保留。没有强行统一两任务 schema，也没有声称本阶段删除了 Push 的全部 task-specific card 链。

## 实际作者内容与原数组依据

正常 CPU 集成的下一次实际作者 HTTP 请求中，父代 guard 为 K1，当前已测行为为 R4；24 个相同条件均匹配。父代 864 回合（成功785、接触72、超时6、偏离1）；当前864回合（成功682、接触125、超时39、姿态16、偏离1、滑移1）。没有把最近兄弟试验当成父代。

> c1_conditional_frontier/public_dev.c1_conditional_frontier_level_5/forbidden_contact：正向差分速度0.594→-0.339m/s，差-0.933m/s；有效回合29/29→28/28，缺测读数0→0。

对应未格式化的原数组统计：K1 `0.5937145352363586`，R4 `-0.33933985233306885`，差 `-0.9330543875694275 m/s`。均为相同条件、禁用部位接触标签、完整终止前0.5s窗口的逐回合统计后中位数；已独立同原数组读数核对。内部比较操作数不进入作者 HTTP。

A 的结局内区分仍在：R4 接触125回合分为已见上轴向下40、完整记录未见上轴向下且末速>0.2m/s的68、其余已知17；对应结束时刻6.83/4.74/5.40s、末速 -0.467446804/+0.566051722/+0.104366541 m/s。超时39中35回合末速绝对值<0.01m/s，另4回合>0.2m/s（中位0.635192618）。这些是支持的过程描述，不给接触标签强加单一失败原因。

> m50：超时组864回合，记录结束时刻12s；末窗正向差分速度0m/s，低正向进度确认占比100%。
>
> m300：禁用部位接触标签组198回合，记录结束时刻1.46s，末窗正向差分速度0.501m/s；超时组666回合，记录结束时刻12s，末窗正向差分速度0m/s。

m50 全程仍有初始位移（净正向0.13m），不能把全回合写成绝对静止；位置差分识别的是后段几乎不动。m300 接触组原数组末速中位 `0.501143694 m/s`。原速度通道偏置不再被当作足点运动证据。

预算128的初始参考实际请求明确包含：

```json
{"source_kind":"bound_public_dev_observation","source_evaluation_job_id":"p6d-production-shaped-fixture-eval-public-dev","description_status":{"status":"unavailable","reason":"budget_exceeded","required_bytes":7110,"max_bytes":128,"text":"必需行为描述超出配置容量，本次保留评估数值，描述不可用。"}}
```

原输入位置（均为已有档案）：R4 `outputs/trajectory_p0/raw/numeric_evidence/manifest.json`；K1 `outputs/trajectory_p1d/raw/k1_numeric_evidence/manifest.json`；K2 `outputs/p2a/raw/k2/manifest.json`；m50/m300 为临时只读审计下载目录 `C:/Users/lover/AppData/Local/Temp/behavior_expectations_20261003/raw/{m50,m300}/numeric_evidence/manifest.json`。没有新 rollout。

## 验证与规模

| 实际产物 | R4 | K1 | K2 | m50 | m300 |
|---|---:|---:|---:|---:|---:|
| 公开报告 bytes | 10222 | 10062 | 10089 | 5310 | 6529 |
| 完整紧凑报告 bytes | 90526 | 72625 | 75481 | 30696 | 42092 |
| 正文条数（观测值，非资格要求） | 29 | 31 | 31 | 14 | 19 |

同样的三次 CPU 档案注入/正常 composition→generation→下一请求链，A 描述相关产物为 4,335,231,147 bytes；B 三个完整报告共235,883 bytes，加比较操作数44,985 bytes共280,868 bytes。不含原始 NPZ；旧历史输出并未删除。一次运行墙钟由489.740s变为131.050s；这是同环境运行观察，不是严格性能基准。受改动的30个生产源文件总大小905,983→614,849 bytes。

验证分组有重叠，不累加成独立测试总数：

- 18个相关 unit 模块196测试通过（87.085s）：行为/时间/周期/关节/地形、Stair生成/报告/raw适配/v1/直接pipeline/CLI、Push生成/语义/pipeline/process/phased/behavior及共享选择。
- phased evaluation + prompt capacity 42测试通过（8.391s）；最终参考接线后 pipeline/report/phased/capacity 共55测试通过（14.229s）。
- 删除周期旧补充层后的相关16测试通过；Push预算与phase19测试通过。保留关键真实行为和边界，未建立任意物理因果标签矩阵。
- 独立稀疏源码/config/test checkout：`882152f9` 的29测试通过（34.682s）；`74248972` 的真实R4完整报告与工作区相同（18.644s）；最终参考修复 `0fb4903d` 的报告与phase32测试通过（7.781s）。稀疏 checkout 使用同一已安装 Python 环境，没有复制训练数据；不宣称完整仓库/部署/GPU验证。
- 五队列完整正文由只读 reviewer 与 A 冻结前基线逐项对照；parent 句之外完全相同。最终代码再次重放五份完整报告及正常集成的三份 phase 报告通过。
- 正常实际 HTTP 三次捕获、已完成phase复用及配置漂移拒绝通过；预算128初始参考实际HTTP通过。unsupported capture、身份错配和原 chunk 损坏的边界由现有直接pipeline测试验证。

可重跑的正常集成入口：[capture_lifecycle.py](capture_lifecycle.py)：

```powershell
true_RL_Agent/.venv/Scripts/python.exe true_RL_Agent/PLAN/feedback_sys/NEW_SYS/B/capture_lifecycle.py --output <新临时目录> --r4 true_RL_Agent/outputs/trajectory_p0/raw/numeric_evidence/manifest.json --k1 true_RL_Agent/outputs/trajectory_p1d/raw/k1_numeric_evidence/manifest.json
```

本地复核证据：`.tmp/stage_b/{r4,k1,k2,m50,m300}/{public,full,comparison}.json`；`.tmp/b_lifecycle_final/verification.json` 及 `requests/g2-lane-a/{user_payload,request.http_body}.json`；`.tmp/b_budget_lifecycle_final/requests/g1-lane-a/`。预算检查复用同一集成流程，将 profile预算设128并在首次作者请求后停止。它不调用网络；training/provider边界为明确fixture，原NPZ不改，只为CPU集成重新绑定manifest身份。此结果不证明checkpoint真实加载、Isaac执行或模型理解。

## 10-03 原审计逐项处置

| 项 | 处置与边界 |
|---:|---|
| 1 | A：世界/身体相对位置差分替代未经证明的速度通道解释；m50/m300检验后段静止与速度偏置。 |
| 2 | A：contact组合由实际实体及频率生成。 |
| 3 | 有界：v3语义contract可声明名义能力，缺元数据仍不可用；未添加历史白名单或新runtime证明。 |
| 4 | B：去工作区源码/EOL资格门槛；capsule与资产元数据按内容绑定，clean R4复现。 |
| 5 | B：预算/明确不支持能力降级保留科学数值，候选及初始参考作者可见；损坏不吞掉。 |
| 6 | A：初始不足历史记不适用，内部缺失记未知，修复阴性退化。 |
| 7 | A：阈值由任务contract传给计算与文字，不复制定义。 |
| 8 | A：Push移除固定20槽，B共享整条选择机械。 |
| 9 | B：Stair CLI/profile/预算单源，默认10240。 |
| 10 | A/B：先由原数组建立行为要求，再核文字；Push独立高度/速度期望发现列映射问题。未声称所有字段均独立证明。 |
| 11 | B：删除无人消费的旧证据生成/自证verify链，保留当前原数组行为验证。 |
| 12 | A：名义上下边带与越界按全队列有效点统计，不靠单离群样本。 |
| 13 | 有界：仅扣已证明的记录精度边界量化误差，保留真实约-0.0856rad差值；名义不叫runtime，不把outside一概说浮点。 |
| 14 | B：明确父代guard，条件/结局/单位/定义匹配，给绝对值、差值及分母组成。 |
| 15 | A/B：按覆盖、频率及差异选择，语义排序，archive ID不决定内容。 |
| 16 | A：身体轴倾角与分段四元数短弧，去全局Euler unwrap解释。 |
| 17 | 有界：低正向进度可含倒退并另报方向；确认时刻不是区间起点，不强行回推物理起因。 |
| 18 | B：删除逐步对象持久账本和无人消费区间交叉关系，保留正文消费的汇总。 |

原始10-03审计和目录保持 untracked 原样；历史验证文档留存，未继续堆冻结hash收据。10-04独立审计内容未读取。R1b、新GPU、模型API、训练、奖励profile扩展与新采集均未启动。阶段完成后等待独立验收。

## 已删除/合并项

以下均是本阶段前已 tracked、可从 `f23dae47` 恢复的实现或脚本。删除前检索调用/导入；当前调用者已移往直接summary或保留算子。没有删除 untracked 审计、raw、checkpoint、运行数据。Windows受限视图的假深路径删除未进入提交。

`P1d/capture_lifecycle.py` 合并原 P1c/R1a 层后移至本目录，保留正常CPU投递验证；Push P2b2现有调用同步更新。下表路径相对 `true_RL_Agent/`。

| 路径 | 删除依据/替代 |
|---|---|
| `PLAN/feedback_sys/NEW_SYS/P1c/audit_delivery.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P1c/capture_delivery.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P1d/assemble_verification.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2a/assemble_verification.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2a/audit_replay.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2a/check_real_guards.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2a/compat_v2.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2a/replay.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2a/trace_examples.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2b1/check_stair_compat.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c1/assemble_verification.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c1/run_semantic_repair.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c1/run_validation.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c1/semantic_repair_replay.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c1/verify_compatibility.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c2/assemble_verification.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c2/run_validation.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c3/assemble_verification.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/P2c3/run_validation.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/R1a/capture_lifecycle.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/R1a/compare_budgets.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/R1a/inspect_delivery.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/R1a/package_evidence.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `PLAN/feedback_sys/NEW_SYS/R1a/run_validation.py` | 旧生成/兼容/收据包装脚本；历史文档与结果保留，当前投递由B单一CPU入口覆盖。 |
| `scripts/build_policy_behavior_card.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `scripts/build_stair_cycle_supplement.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `scripts/build_stair_joint_limit_supplement.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `scripts/build_stair_terrain_supplement.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `scripts/build_stair_trajectory_evidence.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `scripts/verify_policy_behavior_card.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `scripts/verify_stair_trajectory_evidence.py` | 旧card或supplement生成/verify CLI；当前stair_numeric_dev_replay直接生成紧凑报告。 |
| `src/true_rl_agent/evaluation/trajectory_translation/report.py` | 旧通用card/证据拼装生成器；不再有当前消费者。 |
| `src/true_rl_agent/tasks/stair_climbing/process_report.py` | 旧Stair固定槽生成器；由当前行为summary与production取代。 |
| `src/true_rl_agent/tasks/stair_climbing/trajectory_artifacts.py` | 旧base/card缓存及大账目编排；由直接pipeline取代。 |
| `tests/unit/stair_process_development_oracle.py` | 绑定旧固定槽结构的oracle；A独立原数组行为要求和当前事实验证替代。 |
| `tests/unit/test_p3r_stair_projection.py` | 旧projection兼容测试；当前phase/source绑定测试替代。 |
| `tests/unit/test_stair_behavior.py` | 旧检测卡测试；有效阶段/参考投递由当前unit及真实CPU请求验证。 |
| `tests/unit/test_stair_cycle_supplement.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_stair_process_report.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_stair_trajectory_evidence_build.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_terrain_offline.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_trajectory_card.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_trajectory_card_verification.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_trajectory_cohort.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_trajectory_evidence_verification.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
| `tests/unit/test_trajectory_measurement.py` | 被删除的旧生成器/补充账目/自证card测试；保留当前原算子、边界和行为要求，fixture移入trajectory_fixtures。 |
