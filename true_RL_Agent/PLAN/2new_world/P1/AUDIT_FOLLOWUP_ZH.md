# P1 复审与修复

日期：2026-09-07。状态：**复审修复与验收完成，P1 GO**。2026-09-06 的证据保留；其 GO 没有覆盖“已兼容的旧 reference 证据 + 新候选”的比较形态，本记录补充该缺口。机器记录见 [audit_validation_results.json](audit_validation_results.json)。

## 核对与实施顺序

1. **混合统计版本阻断：已复现。** 同一比较域的当前 screen producer 统一选择派生统计契约，旧 summary 先按原版本验证，再形成当前比较投影；不重跑 reference、不修改旧回执、不删除 selector 的版本一致性检查。历史 selection 回放仍使用已封 screen 的统计版本。
2. **上游均值：缺口成立，但未证明支持中的 Python 今天会因此漂移。** 当前 ordered/unified writer 从 episode 到 cell、stratum、family 和 paired vector 的均值统一使用明确算术；旧 summary reader 保持原解释。补验区分“从相同 episode 重新生成”和“读取已存 summary”。
3. **统计二源与重复公式：成立。** 收敛至现有 `evaluation/derived_statistics.py`；任务模块只保留评分结构、协议字段与版本选择，不新增统计框架或任务专用共享逻辑。ordered schema 由定义模块提供，未知版本不能静默降级。
4. **入口可诊断性：成立。** 从实际 backend 配置记录 train/eval entrypoint 到 placement；Stair 已拒绝非规范入口，不能把只改 `task_defaults` 的实验当成活动入口已改变。保留科学身份与位置的现有归属。
5. **硬失败覆盖：按行为而非报错字符串数量判断。** 补有实际代价的混用、父进程源码漂移、已封 reference 损坏与 fresh/resume boot 分界；复用现有夹具和测试，不搭故障注入框架。
6. **命名与版本债务：修正误导名称，写明现有旧版本最后消费者与退役条件。** 当前仍被冻结 15 rounds/P5 消费的 reader 保留；P4 的通用 session 改造不在此抢做。

## 验收要求

- 新旧 reference/candidate 组合经实际 phase → screen → selector 路径通过，原 phase hash 不变；旧 selection 与 formal assembly 仍可回放。
- G1、Go2 从相同原始 episode 在 Python 3.11/3.12 生成的当前摘要、投影身份一致；真实历史摘要保持原 hash。
- 新旧统计与边界专项通过，再完成本地全量回归；仅对未覆盖或受影响的 Linux 路径补测。
- 对照冻结配置、当前 source diff 和所有统计消费者复审，必要证据留在本目录。正式 checkpoint、历史 ledger 和回执不删除或重写，临时产物按复用价值保留。

## 逐项复核结果

| 审计项 | 判断与实际处置 |
|---|---|
| G1 / Go2 科学身份未退化 | 重新解析两个 active unified V2 cell，manifest、suite、196608000 steps / 2000 iterations / 4096 envs、3/5/7 seeds 均与冻结值一致，见 `audit_matrix_identity.json`。不以测试通过替代该检查 |
| 旧 reference 与新 candidate 的统计版本混用 | 缺陷成立。修复前真实 selector 可复现 `{None, derived_statistics.v1}` 拒绝。当前 task-owned screen producer 对所有参与者选择同一比较算术；旧 summary 先按旧格式验证，投影不改 phase bytes/hash。selector 的一致性检查保留；历史 formal assembly 从已封 screen 取统计版本，避免回放时擅自升级 |
| 上游均值未收敛 | 成立。当前 ordered/unified 的 episode → cell → stratum → family → seed 聚合及 qualification、prompt projection 统一使用明确的精确均值；selector guards、探索候选 guard floor 和 final 配对统计也使用同一数值实现。旧摘要依旧按旧算术解释 |
| ordered v3 字面量二源和未知版本降级 | 成立。ordered 定义模块提供唯一 schema → statistics resolver；unified 从该 resolver 选择格式，未知 ordered schema 明确拒绝 |
| 多份 legacy 统计公式 | 成立。完整 mean/std/SE 计算集中到现有 `evaluation/derived_statistics.py`，任务内 helper 只装配字段；删除 `_screen_score_stats`，同时消除其变量遮蔽。共享模块没有任务或机器人分支 |
| entrypoint 丢失 | 位置可诊断性缺口成立。Stair / Push composition 从实际 backend config 记录 train/eval entrypoint。Stair 原有非规范入口拒绝仍保留；只改未生效的 CLI 默认值不能证明实际执行入口发生变化 |
| 22 个硬失败缺少覆盖 | 不采用“报错字符串零命中 = 行为未覆盖”的推断。补最有实际代价的边界：混用比较、父进程源码漂移、reference 实物损坏、fresh / resume probe boot 归属、final 配对统计单 ULP 篡改；复用现有测试结构，没有另建故障注入框架 |
| readiness 命名误导 | 成立。入口 validator 改名 `_validate_subject_receipt_mapping`，报错说明支持的 subject receipt 格式。v3 引入且格式未变的 CLI/config 与子结构名称保留，不进行无意义的全文件改名 |
| 版本面与大文件债务 | 保留必要历史 reader，退役条件见下表。本次没有增加 schema 版本。P4 的 session 拆分仍按任务 × 机器人边界推进；本次不通过搬动大文件或增加共享层任务分支假装完成通用编排 |

审计中的两项推断需要收窄：

- “下一次必然在两条训练结束后崩溃、浪费约 4 GPU-h”不是当前入口必然行为。真实导入还检查 Genesis snapshot、evaluation binding 和 backend 环境；旧 P5 v3 与新 v6 composition 并不自动兼容，可能在训练前就被拒绝。本次解决比较投影缺陷，不跳过这些边界，也不将旧 checkpoint 自动授予新后端资格。
- 原 `numeric_identity_probe.py` 确实从构造的 episode 生成摘要，并非只重新读取存储摘要。但其 G1 使用旧 unified V1 协议，且上游算术未全面固定；新探针覆盖两个 active unified V2 cell。本次没有复跑审计给出的 100000 / 20000 次随机统计实验，不将其归因数字写成已独立验证的结论。

## 实际验收与证据边界

| 检查 | 结果 | 证据 |
|---|---|---|
| Windows Python 3.12 全量 | 1615 项，0 failure/error，31 skip，287.630 秒 | `audit_full_regression.log` |
| 混合版本、readiness、final 等专项 | 57 项通过 | `audit_mixed_version_tests.log` |
| 最后加入 final 单 ULP 拒绝断言后的复核 | 14 项通过；产品源码未再变化 | `audit_final_confirmation_tests.log` |
| Linux Python 3.11.15 / 3.12.14 | 各 250 项通过，0 skip，84.065 / 86.916 秒 | `audit_linux_py311_tests.log`、`audit_linux_py312_tests.log` |
| Linux 完整 15 代与 final 中断恢复 | 1 个完整长用例通过，835.849 秒；terminal audit 后 ledger/report 不变、实际 job 不重复 | `audit_linux_formal_tests.log` |
| 删除无用导入后的最终源码复核 | Windows 3.12、Linux 3.11/3.12 各 22 项通过，3.250 / 4.626 / 4.593 秒；源码与数值身份一致 | `audit_final_source*_tests.log`、`audit_final_source_identity.json`、`audit_final_source_py311.json`、`audit_final_source_py312.json` |
| 原始 episode → 当前摘要与投影 | Windows 3.12 与 Linux 3.11/3.12 的 G1、Go2 原始输入、summary、projection hash 完全相同 | `audit_numeric_local.json`、`audit_numeric_py311.json`、`audit_numeric_py312.json` |
| 真实旧 P5 public-dev summary | G1 / Go2 完整验证、保留原 hash；通过 phase → journal → bootstrap → screen → selector 混合版本用例 | `tests/fixtures/stair_summaries/p5_legacy_public_dev.json`、`test_stair_phased_evaluation.py` |
| 历史 formal assembly | 已封 legacy screen 序列化、重读、重组 formal evaluation 后仍保持旧统计表示 | `test_historical_screen_statistics_replay_through_formal_assembly` |
| 24 份几何实验旧摘要统计 | stored seed vectors → legacy score_stats，24/24 一致；与当前统计表示共 7 个字段不同 | `audit_legacy_statistics.json` |
| 历史实物保护 | 71 个正式 checkpoint 大小/mtime 不变，801 份正式证据及 6 份 reference 文件 SHA256 不变 | `audit_protected_artifacts.json` |

上述混合版本用例使用真实 P5 摘要和真实 phase/journal/bootstrap/selector；Genesis 已验证对象的定位由夹具提供，backend 不训练、不评测。它验证比较入口的恢复形态，不声称旧 P5 已通过新运行环境的完整 Genesis import 资格化。24 份几何实验只复核存储 seed 向量的统计，没有宣称重建其各自缩放协议；两个 P5 摘要则按各自真实 active V2 协议完成完整验证。

当前生成链另有测试将 `statistics.fmean/stdev` 替换为拒绝调用，确保从原始 episode 构造当前摘要不落入 legacy 算术；这个测试与跨 Python 实际结果互相补充。旧 ordered v1/v2 的兼容测试从真实历史摘要构造，没有把新摘要换一个 schema 标签冒充旧格式。

完整 15 代用例使用既有 fake runner：第 1 代走双 lane、选择与接受，后 14 代合法 authoring-incomplete 关账，final 在实际子进程硬退出后恢复并重复审计。它覆盖编排和证据闭合，不是 15 代真实训练或科学收益验证。首轮 P1 已通过的 reference-head 15 代与 3 代崩溃用例未机械重跑；本次用受影响的 phase/final/恢复专项和 candidate-head 长用例补验。

首次 `audit_targeted_tests.log` 保留了一次 62 项中的夹具错误：修改 probe JSON 后未使用规范序列化，先触发了原有 JSON 校验。修正夹具后，fresh boot / 两个 probe 不同 boot 的目标拒绝断言通过；最终结果以之后的专项、全量和 Linux 日志为准。

- 全量、双 Linux 250 项和完整 15 代测试时：323 个执行文件，snapshot hash `c2ce4f9f8fd50002e464ff47f10ea97a243d86efd7ecc51246982c8447516a0e`。
- 最终交付：323 个执行文件，snapshot hash `d0c45575636c7a30d87443069b110e36bd9085ca45641b012ed92d581da784ee`。相对上述实测源码，仅从 `phased_evaluation.py` 删除一个已有的无消费者 import，函数体不变；已对照源包逐文件确认差异，并在三种环境各补 22 项相关测试，没有重跑长用例。
- 原始 episode 数值 probe 的合并身份：`0366825ca7a2cb73ce42694b42d1e1bd38b76aff83310fb200495c89cab69c92`。
- 未修改 protocol / suite / budget / seed / PPO / acceptance threshold；没有新增矩阵轴或持久化 cell epoch。

## 旧版本的必要消费者与退役条件

此表描述现有边界，不建立新的版本注册表。今后新增活动格式，在对应 Phase 同时说明旧消费者和退役条件；历史不可替代的 reader 不因版本数量而删除。

| 身份族 / 当前主线 | 仍需保留的消费者 | 可删除条件 |
|---|---|---|
| composition snapshot v3 / Genesis v6 | `SessionGenesisArtifact`、reference import 和恢复仍读取 P5、正式 15 rounds 的旧 Genesis v2–v5 及原 snapshot hash；低层兼容调用仍可形成旧结构 | 当前生产入口不再写旧结构、无可恢复会话依赖后，移除旧 writer；冻结历史须先有受支持的最小 reader 或其原 commit 审计路径，不能通过改写旧 hash 清除依赖 |
| backend environment manifest v2 | 旧 Genesis 环境身份及 `test_isaac_lab_execution_environment.py` 的兼容证据仍消费 v1 | P2 完成后端迁移后删除无人使用的旧生产路径；必要的冻结 v1 reader 单独保留，不作为第二个生产后端 |
| launch readiness v4 | v3 的同 boot 历史 receipt 仍需读取；v1 validator 是复用的基础校验。v2 已退役，通过其绑定的历史 commit 审计 | 不恢复 v2 主线 reader；v3 在无在途/可恢复会话依赖且冻结审计有明确入口后删除。v1 作为仍被调用的内部校验不能按版本号误删 |
| ordered summary v3 / unified summary v2 | P5、完整 15 rounds 和几何实验的 ordered v1/v2、unified v1 原始摘要；acceptance/final 的历史配对统计回放 | 当前 writer 统一后，旧生成脚手架可删；旧 reader 在受保护证据仍有消费者时保留，公式通过共享 legacy 分支收敛 |

当前 derived statistics v1 在 P1 尚未部署的实现内补齐上游算术，没有创造一个“兼容旧 P1 临时新版摘要”的新生产格式。首轮 P1 探针及 fake-runner 日志作为当时验证记录保留，不用它们冒充本次源码身份。

## 同类问题复审与交付范围

复核了 ordered/unified builder 和 validator、screen/selector、formal assembly、reference qualification、exploration、acceptance、final 的统计消费者。受影响的当前 writer 均使用明确算术；`acceptance_protocol` 剩余的 `fmean` 只用于容差验证存储值，不生成新的身份字段。旧 grid/generalization 协议不因名字相似而改写数值。当前比较版本由 Stair 投影入口拥有，共享模块只计算数字。

这里没有声称所有 Stair 均值都已 canonical：`tasks/stair_climbing/memory.py` 直接导入的 `fmean` 计算给 LLM 的历史分数描述，不进入 summary、receipt 身份或接受门，因此保留。原始 episode 数值测试覆盖摘要构造链，不覆盖全部提示/历史汇总；其 patch 也不是全仓库静态禁用 `statistics` 的规则。

本次删除的是已被统一实现替代的统计包装逻辑，没有删除正式 checkpoint、ledger、P5 anchor 或原始摘要。服务器只使用 `/data/yuyangliu/AGENT/.tmp/p1_audit_fix_20260907` 隔离副本；源包、隔离副本和必要诊断日志保留以便复用。正式 checkout 未部署，真实 RL / provider / 录像作业数为 0。P2 的后端资格化与 P4 的矩阵编排解耦仍按主计划实施，不由这次 P1 GO 代替。

## P2 前的 API 与目录收尾（2026-09-07）

后续建议复核成立的最小改动：三个公开统计 helper 默认统一为当前 derived statistics v1；`paired_standard_error` 删除没有活消费者的空向量返回零分支，复用已有非空校验；私有 `_factor_diagnostics` 和 ordered/unified `_score_stats` 要求调用者显式选择算术。当前三个 acceptance 调用本来就显式传参，历史计算值不变。11 个构造器不变式检查保留，不为错误消息计数机械增加测试。

父 Git 根加入 `.tmp/` 和 `/p6d-experiment-demo/` ignore；同时将已有 outputs 子目录规则收敛为 `/true_RL_Agent/outputs/`。核对时 outputs 中 Git 跟踪文件为 0、未跟踪文件 581 个/29047494 字节，属于运行产物；正式可版本化报告继续放 PLAN，必要小夹具放 tests。ignore 不删除磁盘产物或改变独立 Site 仓库。未独立重算“29 GB”这一审计估计。

定向验收日志为 `api_cleanup_tests.log`；这些 API 收尾属于后续源码修改，上文两次 P1 验收的源码 hash 保持历史值，不能当作当前 P2 工作树身份。
