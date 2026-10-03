# P2a 长 off 单步 on 选例定向补正（2026-10-03）

此补正基于 `b8d4c1401c676d472795b4b80aeff12276fceb54`，只修改审计代码、测试和验收记录。
父线程已通过生产实现、三源数值和 v2 兼容性检查；剩余阻塞是反例选择器只检查一条完整 on，
未落实 FREEZE 原文“长 off 中单步重触发”的长 off 条件。原三个指定例的相邻 off 时长
加总最大值也仅为 K1 0.14s、K2 0.22s、K3 0.36s，因此撤回它们的长 off 分类。
三例保留为有效 `isolated_current_on` 回归材料，不删除旧样本、原日志或原审计结论痕迹。

## 本次澄清与范围

用户在审查后明确：**前侧同足 off 已观察格时长 ≥0.5s，紧随 1 条完整 on，后侧至少
1 条 off 且无缺步/unknown**。这是后续资格澄清，不声称原先已精确冻结此谓词。
同足 finite current 力模 `>1N` 为 on；unknown/实际步间断切分区间，不能跨段凑足时长。
前 off 可左截尾，后 off 不必达到 0.5s；两侧相加或只后侧够长都不合格。
全队列按 `(run, env, on_step, stable entity ID)` 排序取最小项。

冻结的生产测量、阈值、模板、source lock、三源 numeric/card/full/public 均未改。
`audit_replay.prepare` 若将来完整运行会另写 `necessary_raw_retrigger_corrected.json`，不覆盖旧账；
本次使用 `repair_retrigger_audit.py` 仅读取 current 必要数组、复核原产物 SHA，并 seek 三个既有
timeline。没有重跑数百万项全量测量，也没有创建另一份 GB 级生产证据。

`verify --selection <账本> --output <新报告>` 现在绑定选例账本 SHA。默认旧账的 counterexamples、
sample reasons、absent_cases 同步更名，并明确 `not_established_by_original_ledger`，不误报长 off
不存在。新账须匹配本次资格；默认报告改为 `*_audit_selection_corrected.json`，文件存在即拒绝，
所有写出仍用 create-once。此接线通过单测；本次没有再次执行全量 `prepare` / `verify`。

## 全队列最小见证

每源 864 回合、四足全部扫描；有效行分别 276259、276949、298375，所有 dt 均为 0.02s。
下表区间为实际步半开区间；格时长为 `N×dt`，观察跨度为 `(N−1)×dt`。

| 源 / 最小键 | 全部合格事件 | 前 off：区间 / N / 格时长 / 跨度 | on：区间 / N / 格时长 / 跨度 | 后 off：区间 / N / 格时长 / 跨度 |
|---|---:|---|---|---|
| K1 `(1,96,41,RL_foot)` | 54 | `[1,41)` / 40 / 0.80s / 0.78s | `[41,42)` / 1 / 0.02s / 0s | `[42,66)` / 24 / 0.48s / 0.46s |
| K2 `(1,111,312,RR_foot)` | 18 | `[285,312)` / 27 / 0.54s / 0.52s | `[312,313)` / 1 / 0.02s / 0s | `[313,319)` / 6 / 0.12s / 0.10s |
| K3 `(1,50,107,RL_foot)` | 48 | `[77,107)` / 30 / 0.60s / 0.58s | `[107,108)` / 1 / 0.02s / 0s | `[108,119)` / 11 / 0.22s / 0.20s |

K1 前 off 左截尾、右侧不截尾，不能宣称完整 off 持续时间；其已观察 40 格足够资格。
其余表内 off 区间左右均不截尾，三例 on 均左右有紧邻 off，属于完整单条 on。
这些是采样 current 状态与读数转换，不是连续物理接触时刻或持续时间断言。
父审其他“任一侧够长”的 101/33/130 计数属于不同谓词，不与本表混用。

全部 120 个有序候选和三个最小项存于 `outputs/p2a_retrigger/correction.json`；新索引另保留完整
候选键及前后区间供持久复核。旧样本关系如下，完整 ID 列表见新索引：

| 源 | 旧短 on 指定例 | 原样本数 → 并集数 | 新最小项与原样本关系 |
|---|---|---:|---|
| K1 | r1.e0 | 17 → 18 | 新增 r1.e96；原样本另有合格 r1.e196，但不是全局最小 |
| K2 | r1.e2 | 19 → 20 | 新增 r1.e111；原样本另有合格 r2.e278，但不是全局最小 |
| K3 | r1.e0 | 18 → 18 | r1.e50 已在原样本内，补正确分类与区间证据 |

## 原数组到既有事实

三例分别记录 66/36/44 条原数组锚点（完整前/on/后区间及可观察的相邻边界），含 NPZ SHA、
零基 row/env、实际 step、force_xyz、力模与 marker。on 的定位及力模如下；完整向量见新索引。

| 源 | NPZ / 零基 row / env | on 力模 | chunk SHA256 |
|---|---|---:|---|
| K1 | chunk_000000.npz / 40 / 96 | 18.4740104675293N | 5518421701fb1c13944be8a6c9296d415a3493ca35394b8e0a6f502ae05fe4c0 |
| K2 | chunk_000004.npz / 55 / 111 | 35.43811140536757N | 344e3447aef8ec53d3e9b4c9791420bc0f900c721be4b43c813f764277001b6e |
| K3 | chunk_000001.npz / 42 / 50 | 14.22898197174073N | c60f501d8570e604f6191aae1c43d96ba4dc190c7c72eeea9a573d62ff7850ec |

每例通过既有 episode_summary 的 timeline_ref 定位原 numeric JSONL，核对同足 marker measurement
中的三个区间、三个 state_interval、on/off 两次 transition，以及来源 SHA、channel、chunk/row/env
引用。confirmation_time 为实际 step×dt，physical_event_range 仍为空；既有事实已正确保留该单步
on。33 份原生产产物 SHA 全部不变，包括三份完整 numeric JSONL、卡片和 public；本次未新增或
改写生产文句，原“数组→事实→句子”核查继续引用原记录，不给新例虚构一句已输出的文字。

## 验证、成本与结论

在仓库根目录执行（PowerShell；`PYTHONUTF8=1`）：

```powershell
.\.venv\Scripts\python.exe -B PLAN/feedback_sys/NEW_SYS/P2a/repair_retrigger_audit.py
.\.venv\Scripts\python.exe -B -m unittest tests.unit.test_p2a_retrigger_selection tests.unit.test_trajectory_temporal -v
```

补证命令 exit 0，原始结尾 `PASS targeted retrigger correction 5.1640755999833345`，约 5.16s。
首次边界/时间测试 40 PASS；新增旧账和 CLI 接线测试后最终 **43 PASS，0 FAIL，0 SKIP**，0.102s。
这 43 项包含先前的 40 项，不把两次相加成 83 项。边界覆盖恰好 0.5s、短前侧、仅后侧长、两侧
相加才够长、unknown、缺步、多条 on、未观察到后 off、稳定实体排序、原账不改和另名拒覆盖。
日志及 SHA 见 [RETRIGGER_CORRECTION_VERIFICATION.json](./RETRIGGER_CORRECTION_VERIFICATION.json)。

只读 6.1 sol 另写 NumPy 全 current 扫描，未调用生产或本选择器，实现独立复算：120 个候选的
排序/实体/on 步/前后区间逐项一致，146 条 raw anchors 精确一致；样本并集与旧三例撤销分类均通过。
只读 Astra ultra 审查选择器与补证：区间、事件、引用、33 份产物保留及边界测试通过。

原 `verification.json` 与原 `necessary_raw.json`、三份 `*_audit.json` 保留为 e79fa7e 的历史证据；
前者引用的旧代码/文档应在 e79fa7e 快照验证，不能要求补正后文件仍等于旧 SHA。新索引同时记录
历史锚点、修复后代码、原证据和本次输出 SHA；没有回写原审计掩盖选择器错误。

此补正完成长 off 见证的定向审计修复，待父线程干净复验放行；不能补造原预声明资格或未见资格。
三策略仍是相关 lineage 的开发材料，完整 E2 未通过。严格父代比较、周期/限位/地形资格及 K5
Push 未见候选仍按原计划后续处理。本次提交后停止，不推进 P2b、Push 或 P3。
