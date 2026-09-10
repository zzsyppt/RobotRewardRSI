# P3 当前验收：C2 固定基底修订与资格复测

2026-09-09。**当前 G1、Go2 的 reference 资格均按原 public-accept 规则通过；本轮工程验收通过。** 这不代表三卷难度完全相等，也不代表 P3 的全部泛化目标已经完成。未运行新训练、provider/candidate 搜索或产生新的 accepted 结论。

本次核实了 C2 同组模板继承同一位移的问题，但“sealed 一致性参数没有执行”的判断不符合当前源码：候选已有真实门，reference 明确只记 sealed 表现。预先固定的修订、逐项辨析和防止依结果改题的边界见 [卷间构造审计](COHORT_ALIGNMENT_ZH.md)。较早的比例修订见 [历史复审](REGRESSION_FOLLOWUP_ZH.md)及[原始结果](results/regression_followup_20260909/qualification_analysis.json)，其旧 Go2 NO-GO 和 G1 PASS 不改写。

## 1. 修改范围与冻结身份

C2 的 dev/accept/sealed 保留公开 fine calibration 原先选中的基底、振幅、频率和网格，仅留出不同表面相位与种子。Go2 基底为 0.1972 m，G1 为 0.335 m；没有重选模板、提高 roughness 或重标定。匹配的 ID 平面对照固定复用、在主目标之外；其他模板几何不变。final 继续使用原先声明的新基底外推，不能与仅相位留出的三卷混称。

成功/progress、评分权重、reference checkpoint、训练条件和资格门不变。`J_primary=J_OOD`，cell score 为 `0.5 success + 0.5 progress`。`[0.2,0.8]` 仍在 public-accept 判，候选 sealed 的绝对下限与 `0.05` 容差均保持原规则。

| 本体 | 当前 suite hash | 未变的 checkpoint SHA256 |
|---|---|---|
| Go2 | `0ceb7ad4f7bf0091e62808c2542aca5af3d7c83d61020b85c6ab5762c5b28997` | `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c` |
| G1-37 | `d78f19feeaeb8a7baf725e56acb178d7aaf58f8931f9b8e7e69ede9aa9aca73f` | `4f62db0e487cfc39c3cef61777bef9e4160b64f91f92fcca1cc795585ab8236c` |

相对本轮进场快照，只改两个 Stair 产品模块、两个测试模块、四份评测声明；478 个源码/测试/配置文件中其余 470 个未变。公共层、机器人事实和训练/接受合同没有承接地形特例，见 [范围核验](results/cohort_alignment_20260909/scope_audit.json)。工作区还包含此前 P3 改动，不能把整个 Git diff 都归入本轮。

## 2. 当前协议的实际结果

两台 reference 均来自既有完整训练 `4096 × 24 × 2000` steps。每 cell 每 seed 12 episodes；dev/accept/sealed/final 为 3/5/7/7 seeds。以下成功计数包含控制题，不能与主分 J 混写。

| 本体 | public-dev | public-accept | sealed-validation | private final 探针 | reference 资格 |
|---|---:|---:|---:|---:|---|
| Go2 | 689/864（79.75%） | 1165/1440（80.90%） | 1613/2016（80.01%） | 1502/2016（74.50%） | PASS |
| G1-37 | 618/864（71.53%） | 1040/1440（72.22%） | 1383/2016（68.60%） | 1472/2016（73.02%） | PASS |

| 本体 | dev J | accept J | sealed J | final J | accept F_scalar | accept F_c1 | accept F_c2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Go2 | 0.746697 | 0.779666 | 0.788654 | 0.735123 | 0.755303 | 0.726826 | 0.856867 |
| G1-37 | 0.743687 | 0.733584 | 0.676049 | 0.753947 | 0.662561 | 0.780372 | 0.757818 |

| accept 目标 stratum | Go2 成功数 /60 | G1 成功数 /60 |
|---|---|---|
| height_ood | 38、14、44 | 32、21、45 |
| tread_ood | 49、48、47 | 29、29、44 |
| joint_ood | 51、28、48 | 29、12、38 |
| c1_conditional_frontier | 47、28、40 | 37、52、28 |
| c2_mechanism_ood | 48、48、47 | 40、37、33 |

两个 accept 的五个目标题层均有带内转折，控制能力通过，资格 mask 均被真实 reader 接受。Go2 C2 的 80%、80%、78.33% 接近上界，不能描述为余量很大。新旧题面与种子不同，不将 J 差写成策略改进或退化，也不用于本体排名。

原始 [Go2 回执](results/cohort_alignment_20260909/go2_12dof_qualification/result.json)、[G1 回执](results/cohort_alignment_20260909/g1_37dof_qualification/result.json)及[独立复算](results/cohort_alignment_20260909/qualification_analysis.json)保存完整计数、mask、身份和 final 计划绑定。

## 3. 卷间差异与泛化限制

| 当前 public-accept J − sealed J | Go2 | G1-37 |
|---|---:|---:|
| 整体分差 | −0.008989 | +0.057534 |
| 其中 C2 贡献 | −0.006991 | +0.001147 |
| 其中 C1 贡献 | −0.009239 | +0.050519 |

C2 基底混杂已消除，但 G1 的整体分差仍超过候选 `0.05` 容差，主要来自 C1。具有同样两卷分数的候选会被该门拒绝；reference 不是候选，不能把它的差值理解成预先扣掉的候选预算。该门真实存在且保持不变，不能删字段、放宽阈值或把判定移到恰好通过的卷来掩盖结果。

G1 dev 的 C1 为 35/36、35/36、6/36；sealed 的 tread-OOD 为 70/84、71/84、5/84，这两处没有带内转折。因此当前资格通过只说明既定 accept 规则满足，不能扩展为每一卷都有同等区分力。C1 三模板轮换位移、平均高度/踏深基本一致，没有 C2 的整组共享位置错误；几何均值一致仍不保证策略响应一致。本轮按测量前声明保留这些结果，不再根据 accept/sealed/final 改参数。

该修订发生在旧 holdout 已被人工审阅之后。新声明和新种子不让复用的旧几何重新成为“未见”；新测量是修订协议的资格验证，不是严格盲测的策略改善证明。当前两个 final 探针各自消耗了其 study，后续搜索必须生成并预先承诺自己的 final。

大振幅 C2 仍可能呈现崎岖坡的形态，本次未改波形，也未证明自然粗糙楼梯、跨训练 seed 或任意新本体泛化。L 是初始搜索尺度代理；每个新机器人仍需要公开实测标定。

## 4. 工程验收与资源

| 检查 | 证据 |
|---|---|
| 本地全量 | 1640 项，0 failure/error，19 项环境跳过；[回执](results/cohort_alignment_20260909/local_verified.json)。第一次全量中的 localhost HTTP 用例返回 network_error，其独立复跑与完整复跑均通过；原失败记录保留，没有修改 provider 或削弱测试。 |
| 目标 Python 相关回归 | 47 项通过、零跳过，含当前两本体候选 sealed 拒绝及伪造通过回执拒绝；[回执](results/cohort_alignment_20260909/target_focused.json)。 |
| 目标 Python 全量 | **1640 项，0 failure/error，0 跳过**，三个正式 generation / 恢复 / 终态场景均实际执行。完整 discovery 的实际用例 ID 与本地完全相同，无遗漏或重复；耗时 1274.5 秒，[完整回执](results/cohort_alignment_20260909/target_verified.json)。 |
| 真实评测 | 八个 typed attempt 均 completed，failure_kind=null；三卷、资格 mask、私有 final/seed 身份独立重建通过。 |
| 真实网格 | 两本体三卷及 private final 的最大振幅 C2 共 8 个 cell、59,392 个顶面三角形，网格闭合；高度查询最大误差约 3.29e-10 m，[回执](results/cohort_alignment_20260909/formal_geometry_replay.json)。 |
| 原生录像 | Go2 423/393/408 帧，G1 231/294/330 帧；六段实际解码数均等于 episode_steps−start_control_step，文件 SHA256、旁车和几何身份匹配。来自实际 public-dev 同进程 PhysX/RTX。 |
| 历史兼容与保护 | 旧 P3 三卷、资格和 final 按原声明复算一致；旧 V2 J=0.883938290781893。原完整 15 rounds 的 71 个 checkpoint、884 份受保护产物无变化，[回执](results/cohort_alignment_20260909/historical_replay.json)。 |
| 部署与进程收尾 | 部署前逐字节核对正式工作区的 478 个原文件，再同步 8 个产品/测试/声明文件及当前文档。本地、资格目录和正式工作区的 478 个文件一致，337 个产品源文件的完整快照 hash 一致，[源码核验](results/cohort_alignment_20260909/source_consistency.json)。本轮控制进程均已退出，无遗留 GPU 计算进程，[资源回执](results/cohort_alignment_20260909/resource_closure.json)。 |

首次目标全量的 2 个 failure、7 个 error 来自隔离测试装配：源码包缺 Git，子进程缺项目导入路径。[原失败全量](results/cohort_alignment_20260909/target_full.json)保留；在真实 Git 检出和显式 `PYTHONPATH` 下，[9 项复核](results/cohort_alignment_20260909/target_checkout_probe.json)及随后完整回归均通过，未修改产品或断言。原始六份三卷摘要回传 Windows 后逐字段复算一致，[跨位置复算](results/cohort_alignment_20260909/local_formal_replay.json)通过；当前文档及架构边界专项 15 项也通过。交付前再次核对原 15 rounds 的 932 个文件（含 884 份受保护产物、71 个 checkpoint），均无变化，[最终库存核验](results/cohort_alignment_20260909/history_preservation.json)。

本次资格共 **12,672 episodes，GPU lease 合计 1.115479 小时**，低于预先声明的 2 小时。该数字仅计资格作业，不冒充包括完整测试的总成本。完整运行材料在服务器 `/data/yuyangliu/isolated/p3_cohort_alignment_20260909/true_RL_Agent/PLAN/2new_world/P3/results/cohort_alignment_20260909`；本地保存必要 JSON 和验证记录。

默认入口仍为 G1 V3，Go2 显式选择对应 V3 行。reference 资格通过不等于新正式搜索可以启动：P4 的编排及实测 GPU 小时/输出空间合同仍需完成；卷间区分力和地形形态的限制也不能从研究声明中省略。Go2 × Push Recovery 仍由后续任务 Phase 闭合。
