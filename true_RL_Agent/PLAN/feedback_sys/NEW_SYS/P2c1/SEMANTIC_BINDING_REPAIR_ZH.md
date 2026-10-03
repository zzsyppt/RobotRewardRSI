# P2c1 离线预览语义绑定修复

父线程对 `1831182aa06a73c1e4af07d585c4a65997c0a7f0` **暂不放行**：数值、154 tests、
25,920 条独立测量已通过，但 preview 接受调用者自报 archive hash 和任意 `{id,text}`，
oracle 也没有固定校验 unit/frame。本次只修这一个边界，新 commit，不 amend。
当前状态为修复自测 PASS、等待父线程复验，不能视为 P2c1 放行。

## 有限修复及证据边界

唯一生产文件为 `tasks/stair_climbing/trajectory_cycles.py`。入口改为：

```python
preview_with_cycles(base_public, supplement_report, episode_records, *, max_bytes=8192)
```

取消自由 statements 和 archive_hash 参数。由 supplement 的 manifest 路径与 hash 读取真实
numeric manifest，复用已有 `_load_contract` 验证 hash、公开 split、runtime projection 及实体顺序。
从实际 manifest 取得 source/domain；其声明 evaluation_plan_hash 时也核对 base 的 plan。
不重读所有 NPZ，不重新执行整条数值 pipeline。

每个 episode 核对完整 run/env 顺序、seed、dt、episode ID；每条记录核对 metric/entity 顺序、
catalog/config、固定 `unit="s; phase turns"`、`frame="recorder_readout"`、current 通道和原始
manifest/chunk/row/step 定位、原 operand ID；phase 引用必须对应两足记录。
由这些记录重新聚合 summary，与 supplement.summary 比较，再由唯一模板生成句子，
与 supplement.optional_statements 全量比较。省略预算判断发生在语义校验之后。

选择**显式 validation 覆盖语义字段**，保留 v1 record ID 算法、数值、原句、catalog v3、
extractor v1 和已有 schema。原 ID 未包含 unit/frame 的事实不被掩盖；只验 ID 仍然不够。
元数据校验先于 ID，所以重新签 ID 也不能放行错误单位或坐标含义。
独立 oracle 使用冻结字面量，未调用生产 validator。

输入仍是已审 adapter/build 的测量 operands。此次真实重放先验原 artifact SHA，再校验语义；
hash 不是签名，本校验不声称能证明任意自行改写并重签的原始数值。
原数组数值资格沿用已通过的独立审计，本轮另核 K1/K3 各一个完整原数组 episode。

## 父反例前后

父证据原件未改，副本及 SHA 收入新的 [verification](semantic_repair_verification.json)。

| 反例 | 首轮 | 修复后 |
|---|---|---|
| K1 supplement 用于另一合法 source base | 接受 | source/domain 不同，拒绝 |
| FL → RR，保留 statement ID | 接受 | 与记录导出的模板不符，拒绝 |
| `0.585s` → `0.585ms`，保留 ID | 接受 | 与记录导出的模板不符，拒绝 |
| FL → RR，group/key 重算 statement ID | 接受 | 与记录导出的模板不符，拒绝 |
| 同时改 report.summary 的 group/key/entity/ID | 未阻止的扩展路径 | 与实际记录重新聚合不符，拒绝 |
| 全 10 records 的 unit/frame 错标 | oracle 接受 | preview 与独立 oracle 均拒绝 |
| unit/frame 改后再签 record ID | 原入口无固定语义验证 | preview 与独立 oracle 均拒绝 |

实际 replay 共 13 个拒绝断言：前五项，另 unit/frame × 原 ID/重签 ID × preview/oracle 八项。
单测另覆盖 source identity/projection/plan、record metric/entity/locator、dt 与缺 episode。

K1 合法完整句原文及 ID **未变**：

```text
cycle:1eeef901ec908ba1d12aa5a7f915fa3792a9ad4afc118775a76f739a6c4836b9
robot_0.support.FL_foot: 3751/8122 four-cycle readout windows qualified; 5991/10714 unique complete cycles; 726/864 episodes contain a qualified local window. Median of per-episode median qualified periods 0.585s. Current >1N marker, no physical contact/gait claim; uncovered spans remain unqualified.
```

这里 `0.585s` 是 cohort 内每回合合格唯一周期中位数再取回合中位数，不是下面局部窗口的周期。
K1 r1.e0 FL 首个合格窗仍为 onset `[25,50,79,107,135]`、间隔 `[25,29,28,28]`、dt `0.02s`、
CV `0.05454545454545454`、相对偏差 `0.10714285714285714`、局部周期中位数 `0.56s`。
本轮完整原数组/所有记录/句子保存在 `outputs/p2c1/semantic_repair_01/k1_real_example.json`，
K3 对应 `k3_real_example.json`；路径和 SHA 均在新 verification 的 `after.raw_episode_checks`。
原 [REAL_EXAMPLES.json](REAL_EXAMPLES.json) 未改。

## 实际执行和兼容

```powershell
.venv/Scripts/python.exe -B PLAN/feedback_sys/NEW_SYS/P2c1/run_semantic_repair.py --output outputs/p2c1/semantic_repair_01
```

最终原输出：

```text
tests exit 0 seconds 9.009
build_k1 exit 0 seconds 11.944
semantic_replay exit 0 seconds 17.025
compatibility exit 0 seconds 3.366

Ran 19 tests in 8.402s
OK

{"status":"PASS","source_previews":3,"metadata_records_checked":25920,
 "raw_episodes":2,"rejected_counterexamples":13,"fresh_build_artifacts_unchanged":4}
```

逐命令 logical argv、stdout/stderr 原文件及 SHA、真实 Python peak working set 均收录新 verification。
最后一块 JSON 仅在此压缩排版，原 stdout 文件完整保留。开发首测曾发现新 validator 误取 chunk
顶层 run/step，已修为 `chunk.metadata` 后才产生上述最终记录；不改写旧日志。

四条最终记录命令累计 wall `41.34382689988706s`，峰值 `640716800B`，API/LLM/GPU/train 均 0。
这个时间不包含探索性读取和两次手动开发测试。未重跑不相关大套件。

K1 新构建的 `cycles.jsonl/report.json/preview.json/preview_receipt.json` 四个 SHA 与旧构建全同。
新 manifest 记录修复后的 code hash/本次成本，SHA：
`87e9435d0463fadad4a0a54879e7b75f563ef7ca02904a7f886e45bf4a2760c7`。
K1/K2/K3 preview 字节仍为 `8086/8032/8127`，receipt 未变；四句均因合计8192B预算省略，
没有因此跳过语义校验，也不宣称已投递模型。

生产精确 diff：`outputs/p2c1/semantic_repair_01/production_diff.patch`，12509B，SHA
`525af19b80674fadb7460f121c57c470b2ff66ae420513ce65dccf4a5a602666`。
新 verification SHA：`83a6a833ee415abd2b0d6ec6be9123be27683755992310d83962b46e44714f64`。
完整代码、测试与文档变更以本次新增 commit 的 `git show` 为准。

[FREEZE_ZH.md](FREEZE_ZH.md) 的周期/变异阈值及其预冻结 commit 不变；原 verification 不覆盖、不改写。
P0 文件未动；P2a 33 文件 SHA 与三源 full/public 实际重放通过，P1b 三个冻结产物通过，
Push 22 文件 freeze 全同、12288B 预算不变。没有修改共享 distribution 或共享数值模块，
因此本轮不触碰 Push cache 依赖问题。旧 v2 重放维持原版。

## 停止位置

提交并推送现有分支后停止写入，等父线程独立复验。P2c2 名义 USD 限位、P2c3 地形参考点高度、
生产投递选择仍是后续有界阶段；本次未扩张。K5/P3 行为及结果未读，未交换结果调规则。
