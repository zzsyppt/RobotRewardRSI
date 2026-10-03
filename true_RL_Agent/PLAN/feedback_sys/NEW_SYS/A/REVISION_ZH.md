# A 首次独立验收 FAIL 后的定向修正

本轮从 `c277a7588bf9c07b47ebc45c12dc67e08d5c020a` 继续，只修首次验收指出的作者内容缺口和缺测措辞。原结论原样留在 [ACCEPTANCE_FIRST_FAIL_ZH.md](ACCEPTANCE_FIRST_FAIL_ZH.md)，与 Temp 验收文件逐字节相同；此前实现说明和首次作者全文保留为历史记录。本说明不替代同一审计线程的复验结论。

实现提交：`38422bb4baf1cc96a703576909fffad5ecfad775`。

## 源头改动与取舍

- `measure_behavior` 增加三类实际存在才生成的小事件：成功回撤后返回、终点 current/history 差异、首次观察到上轴向下先于所选低速未触发段。复用回撤、位置差分、谓词和区间算子；current/history 共用任务阈值。每个事件保留局部来源通道和区间，没有恢复旧区间关系全排列或固定案例槽。
- `aggregate_behavior` 在结局和条件组内保留少量联合频数及分组末速、结束时刻。Go2 Stair 的 `0.01/0.2m/s` 仅是摘要描述分段，由任务层传入，不当成运行时触发阈值或物理分类器。上轴向下分为已观察到、完整记录未见、缺测未定；末窗速度仅在完整连续窗口上参与联合分组。阳性姿态可保留在末速缺测的回合中，文字另报末速有效回合数。
- 删除本轮替代掉的逐条条件候选循环。每个失败结局按条件覆盖数、结局比例差、联合分布差和末速差选择最多两个有信息条件，合成一句；与整个结局重复的条件不再占位。失败/成功末窗地形参考差作为同一内容单元选入。零占用且无越界的关节候选不占空间。10240 字节上限不变，无固定句数资格。
- 事件选择不看 archive ID：成功反例取最大的已返回回撤；history 先选完整且 current 无阳性、history 少量阳性的模式，再按 run/env 取普通实例并报告该模式频数；先后例取最长可核实低速段。没有按最大力或哈希挑异常点。实例均明说不代表发生率或因果。

本次替换/删除均在已 tracked 源码内，可从修正基点恢复；没有删除历史文档、原审计、NPZ、checkpoint 或运行数据。私有旧 base 生成链仍留给 B，不声称这次已完成其删减。

## 独立原数组与最终文字

以下预期来自首次独立验收及只用 NumPy/stdlib 的原数组复算；不是把新候选字段当事实标签。五队列均为已使用的开发数据。

| R4 必说内容 | 原数组依据 | 修订后的实际作者内容 |
|---|---|---|
| 接触标签内相反末段运动 | 125 回合内 40/68/17；末速中位 −0.4674468/+0.5660517/+0.1043665m/s，结束中位 6.83/4.74/5.40s | 同一结局句逐组报告 40/68/17，末速 −0.467/+0.566/+0.104m/s；不再只靠总体 +0.241m/s |
| 超时不是全部停住 | 39 回合中 35 的末速绝对值<0.01，另 4 >0.2m/s；后者中位 0.6351926m/s | 末窗分布 35/4；joint 条件 14/3，tread 条件 0/1 |
| 条件内部也混合 | c1 的 36 回合为 19/13/4；height 的 38 回合为 3/33/2 | 与上述联合分组共用顺序，分别报告两个三元组 |
| current 无阳性不等于 history 无信号 | 全 R4 有 50 回合 current 全程 0、history 仅终点 1 拍；普通例 r1/e54 step132，槽力模 [0,0,112.801593]N，base_contact=true | “50回合……history ANY仅终点1拍”，展示 r1/e54 的 0/0/113N；明确槽时标未知 |
| 成功可伴随回撤并返回 | r3/e281 的世界 x 在 270/324/396 为 −0.8367239833/−1.1797779799/−0.8276581168m | “步270→324回撤0.343m，步396返回此前进度，最终成功；实例不代表发生率” |
| 后段低速未触发不能推出直立举足 | r1/e60 step156 首次上轴向下；FR 在 200–637 连续 438 点低速未触发，世界 XY 点速最大 0.0323217m/s | 报告该先后次序、实体、区间和点速，明确先后不证明因果 |

审计原锚 r1/e127 的 [0,84.2553996,137.2681591]N 和 r3/e64 的 FL 513–637 段均再次独立核对，保留在首次 FAIL。最终普通 history 例和最长先后例自然变为 r1/e54、r1/e60 FR；不是为了复现旧编号硬选。旧 r3/e64 FL 的 508 起点来自原速度通道，未恢复。

关键原数组位置：`outputs/trajectory_p0/raw/numeric_evidence/manifest.json`，R4 的 30 块、864 回合联合频数独立全量复算。完整独立依据仍在首次验收引用的 Temp 文件：`trajectory_audit_readonly_20261003/A_R4_MIXED_PROCESS_CHECK.py/.json`、`codex_a_raw_spotcheck_20261003/r4_joint_facets_results.json`、`codex_A_content_20261003/r4_old_new_probe.py` 和 `r4_old_new_result.json`。没有把这些大结果复制成新永久测试或冻结收据。

[修订后的五队列全文](REVISION_AUTHOR_EXAMPLES_ZH.md) 由实际投影导出：

| 开发队列 | public bytes | 作者 profile bytes | 保留的关键差异 |
|---|---:|---:|---|
| R4 | 10228 | 8771 | 上述联合过程、三个短例；接触标签 FL 末窗名义地形差 0.0875m 对成功 0.0281m |
| K1 | 10068 | 8639 | 左后接触 18.5%；接触标签内 23/35/14 及相反末速 |
| K2 | 10095 | 8666 | 低参与足仍为右后 48.4%，主要接触组合与 K1 区分 |
| m50 | 5316 | 3904 | 864 超时；末速 0，末窗足相对点速约 1e−4m/s，无合格周期 |
| m300 | 6535 | 5122 | 198 接触标签在 1.46s 结束且末速 0.501m/s；666 超时在 12s 结束且末速 0 |

K1/K2 所选先后例的低速未触发段只有 1 个记录点，正文明确写出 1 点；它们仅支持该采样点在已见上轴向下之后，不能说明持续停滞。没有将其与 R4 的 438 点过程等同。

## 有限验证与停点

64 项受影响单测通过（18.070s）：

```powershell
.\.venv\Scripts\python.exe -m unittest tests.unit.test_trajectory_behavior tests.unit.test_trajectory_measurement tests.unit.test_trajectory_cohort tests.unit.test_stair_trajectory_production -q
```

只新增一个最终 public 投影测试：八个小合成回合穿过真实投影、public validator 和预算，检查联合过程、短 history、成功回撤返回、先后反例、缺步/姿态未知与阳性组的末速有效分母。它不冒充原数组或 HTTP 独立验收。既有 Push 测试将高度/速度标签及完整单位片段一起断言，避免 `0.4m` 错配 `0.4m/s` 仍通过。没有扩展安全矩阵。

五份已有测量用当前版本逐项重放来源、实体和局部区间，并重聚合一致；最终公开报告全部在同一预算内。

只读代理另按原 NPZ 核实新增分组、history 频数和所选实例，确认五份全文逐句一致、m50/m300 没有退化；随后核实 g1 的 K1 31/31 句及 g2 的 R4 29/29 句完整进入实际 user。其有限结论为本轮指定内容通过、无阻塞，保留上述单点先后例限制；不代替父级最终 A 验收。原文位于 `C:/Users/lover/AppData/Local/Temp/behavior_expectations_20261003/REVISION_CONTENT_REVIEW_ZH.md`。

复用原 R1a CPU harness，composition → finalizer → phase → generation → provider serializer 截获通过 `PASS_CPU_COMPOSITION_ARCHIVE_GENERATION_HTTP`，489.740s。三个请求的 HTTP user 与保存的 `user_payload.json` 逐字节一致，作者 profile 仅有 kind/report。g1 两个请求收到 K1；g2-lane-a 收到 `generation-001-lane-b-eval-public-dev` 的 R4，正文与该 phase report 完全一致，40/68/17、35/4、history50、成功回撤返回、r1/e60 FR 先后例和 c1 的 19/13/4 均到达实际作者输入。没有把 K2/m50/m300 的离线投影说成它们已走过 HTTP 全链。

| 截获请求 | user bytes | HTTP body bytes |
|---|---:|---:|
| g1-lane-a | 70998 | 77180 |
| g1-lane-b | 71670 | 77886 |
| g2-lane-a | 91096 | 98738 |

本地执行产物：`D:/humanoid_agent/.tmp/a_lifecycle_revision/verification.json` 和同目录 `requests/`。历史 NPZ 未改；元数据、训练结果及作者答复仍是 CPU 夹具。`external_model_calls`、`http_network_calls`、`gpu_runs` 均为 0；不验证真实训练、checkpoint 加载、物理原因或模型理解。

保留 10-03 原始审计及目录原样 untracked；原审计 SHA 不变。正常权限只读 git 检查没有真实 tracked 删除。本轮没有读取 10-04 审计或 K5，没有模型 API、GPU、训练或新采集。B/R1b 仍暂停；提交推送后停止，等待同一审计线程复验。
