# Push 代表回合完整骨架：旧开发离线新投影

状态：`PASS_OFFLINE_PROJECTION_ONLY`。0 API、0 GPU、0 SSH、0 新 rollout；没有新的作者请求或实际送达。旧 `delivery_offline_r1`、r10 和 r11 冻结源保持原哈希。

当前 `build_push_process_behavior(legacy_full=measured_v2, seed_summaries=旧 phase seed_summaries)` 从原 hash-bound v2 测量重新生成完整 v3，再由 `project_push_process_behavior()` 生成 public。并非只修改旧 full 的文字。原 v2 canonical SHA 与旧 full 的 `source_derivation_hash` 完全相同；全部 transcripts、数值定义、profile、残差、采样、代表例选择、source_bindings 均未变。与已有 `n1n2_completion_r1/push_full_units_current.json` 比较，只有 `sentences` 改变；相较更早 delivery r1 的 lexicon units 差异是既有元数据刷新，本次没有新增。

`behavior_full.json` 按 canonical JSON 写入，其文件 SHA 与 `public_report.json.full_derivation_sha256` 相同。新的 `public_dev_result.json` 同时指向新 full 文件的真实 SHA/字节数，并包含新 public。这里是明确离线的新视图，原数值结局与历史来源未重新采集或改写。

## 最小模板变化

- 代表实例逐一概述全部已有 recoveries；按原身份保留单次 event 或 force sequence，不新增事件框架。计划未执行与已执行但缺少有效恢复观察分开；缺测不记为零失稳。
- 每个单位保留 anchor/观察窗口、稳定步数/最长 dwell、latch 状态、恢复后失稳首次/末段/总步数及观察截止。最后一段到达截止只声明观测边界，不断言此时已经结束。
- 最多一个详细窗仍按既有规则选择。条件组合的 10/14 范围与单个实例的时间序列明确分开；gaps 如实说明全部单位已概述、一个窗口详述。
- 骨架属于 required 的 `push.process.example`。不改阈值，不删数值对比、反例、残差或缺口；容量仍用现有整视觉句省略策略。

## 验证

`source=0, env_id=51` 对应 `/transcripts/51`：event_0 未 latch；event_1 在324步 latch，之后366–374、385–412两段共37步失稳，观测截止412。终局 planar speed 为0.24674086272716522 m/s，原阈值0.21554618994545 m/s；终局不稳没有被写成从未恢复。原数值对象逐项相等。

| 场景 | 字节数 | 结果 |
| --- | ---: | --- |
| 新默认报告 | 10937 | 无省略 |
| 最大4字节 Unicode 视觉文本 | 11222 | 整句省略 visual.observation |
| 同时附 synthetic parent 差分 | 11893 / 12288 | 必需骨架、数值、反例、残差、覆盖与冲突计数均保留 |

压力材料为3条合成观察，各含200字正文、120字不确定性、6显示步×2视角的12个图像证据引用；不代表真实视觉判断。父差分前/后附加顺序均验证，重复投影一致，所有原始观察仍在 pressure full 中。

执行 `.venv/Scripts/python.exe -B -m unittest tests.unit.test_push_process_report tests.unit.test_push_behavior tests.unit.test_process_report`：**34 tests，OK**。新增针对性案例覆盖首窗未恢复且后窗恢复后再次失稳、计划未执行/有效窗缺失、无后续采样与观测零失稳的区分、父差分及最大 Unicode 视觉句整句省略。`verify_projection.py` 另验证真实旧开发 env51、新 full/public/phase 哈希绑定、12KiB组合压力与旧封存哈希不变。

## 精确新代表例文字

> Root episode r1.e51 has the condition combination found in 10/14 compatible episodes (most frequent combination); conditions confirmed strictly before final step: planar_speed, angular_speed. This episode's sequence only: event_0: anchor 69, observed window 70-179; all-five stable 11/110 steps, maximum pre-latch dwell 10 steps, latch not observed. No latch observed; post-latch interval unavailable. event_1: anchor 234, observed window 235-344; all-five stable 42/110 steps, maximum pre-latch dwell 21 steps, latch 324. Post-latch instability: 37 observed steps in 2 interval(s), first 366-374, last 385-412; through 412 (last_observation, inclusive); last interval reaches observation cutoff. Detail event_0: planar speed violated in 2 interval(s), first 70-112, longest 114-169. angular speed violated in 7 interval(s), first 71-71, longest 139-150. All-five stability had 2 interval(s), longest 170-179. First response step 70; window speed first/peak/last 1.16/1.16/0.0975 m/s, peak step 70. Window net xy displacement magnitude 0.344 m from this anchor, not reset drift. Final observed step 412, outcome not_recovered, failed bounds planar_speed. Intervals can overlap; a state at the final step is not evidence that it preceded termination.

gaps 对应末句：

> The example outlines all 2 recorded recovery units, with 2/2 executed/planned events; 1 window detailed. Other interval details stay in the full archive.

SHA-256：

- 当前 task-owned process_report：`1a4609d679abc9135832c37289022a7d9c77c0dd2a7e6d95d6bb8838f68506cf`
- 新 full：`225fec5f4db4af3b141e3ca10a1abb590550a326ff27f3c8708cfc3587cdcee1`
- 新 public：`26675e0af826e37a8802c97dac4f06b6395bab1914e19ee12335700da20a4fce`
- 新 phase：`9d6e1a82b965b037eb508bdc156f785d397e162f66bd1c615bf393b4d2ded08b`

机器回执与完整来源哈希见 `verification.json`。复现脚本只采用排他新建写入，不能覆盖已有结果。
