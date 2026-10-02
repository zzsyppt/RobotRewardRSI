# N5 两层数值材料容量验证：related_v1

状态：**旧开发池离线工程验证；NOT_FROZEN；未正式评阅；0 API / GPU 调用。** 主计划、协议、生产源码和旧工件均未修改。原始 NPZ、完整原始 JSON 与直接曲线继续留档；模型实际收到的内容由 manifest 列出，不能把未供应字段说成已经审阅。

## 两层固定范围

| 层 | 实际供应给两名评者的数值 | 明确不覆盖 |
|---|---|---|
| 24 episode：Stair | 每个有效步 1..T 的 root xyz、wxyz quaternion、四脚参考点 xyz 与 xyz 速度；有符号 roll/pitch、up_z、实际地表参考高；每脚即时力范数和历史缓冲最大力范数，禁接触角色同类量；全部原始终止标记、有效性、terrain origin、定义、身体次序及来源 | 关节策略、力向量方向、历史槽的物理时间顺序、脚尖皮肤/接触对真值。所有原始字段仍留档；这些缺口不等同于原本未记录 |
| 24 episode：Push | 每个有效步 1..T 的 pose 六量、world vx/vy 和 reset-relative x/y；完整五条件、原始 event/verified/dwell/latch/failure/timeout、终局及定义 | 未采集的全身姿态、足端/接触量、连续施力传感真值 |
| 12 focus：Stair | 全 pool 成员和全 scope 分母/原始终局/匹配属性；每个固定谓词和身体的全时段区间算术：步数、次数、最长连续段、首次位置、首次规定连续长度、满足长度的段数、末 1 s 且严格 T 前是否出现；根部 x/有符号姿态极值、最大回退原值与步、脚位净空/速度关键极值；原始终止区间 | 未选回合的完整未知脚过程、任意跨身体重叠和任意时序关系。不得据这些摘要声称看过全部未选轨迹 |
| 12 focus：Push | 全体分母、原始事件/匹配条件；全时段及事件窗五条件区间算术、dwell 中断时刻与失败条件、每次/全部 latch 后不稳定分布；每个实际事件 anchor、首响应、首次窗内速度最大值、窗末、T 的完整同一行 pose+motion | 未选回合其余步的 pose/motion 没有供应；不凭极值拼接不存在的同一行状态 |

字段集合按任务与事实类型固定，不按自动报告阳性筛选，不截时间。Stair 的同脚未加载低速保持逐脚计时；不能用左右脚交替的 OR 代替。滑移/低净空的全局 OR 则先合并逐步 mask 后求连续段。低前进量的 0.5 s 窗已经在原查询内，确认只需一个满足窗，不能再额外添加 0.5 s dwell。所有组层数据只是直接算术，不是必要事实答案或行为真值标签。

五束问题对应：Q1 使用完整成员/终局；Q2、Q3 使用全 scope 分布与实际匹配属性，成功反例不从报告阳性挑；Q4 的完整过程关系主要从 24 个全时段 episode 建立，组层只承诺列出的分布操作数；Q5 逐字段区分 recorded-but-not-supplied、not-recorded、未定义时段与实际覆盖。若最终 full 包含超出已供应操作数的命题，保留该命题分母并标证据不足，不能靠摘要判真或静默删除。

## 实际离线序列化容量

数值列用可读的逗号分隔十进制字符串，字段/身体/步轴定义只写一次。float32 使用 9 有效位，逐值还原为原 dtype 后比对原字节；派生 float64 保留可精确往返的十进制。没有用四舍五入阈值附近的数值来节约容量。

| 实际 facts 请求 | 官方 DeepSeek tokenizer 文本 token 范围 | 最大 HTTP body |
|---|---:|---:|
| Stair episode ×12 | 52,770–168,220 | 345,632 B |
| Push episode ×12 | 16,914–38,304 | 91,655 B |
| Stair focus ×6 | 167,549–168,981 | 371,305 B |
| Push focus ×6 | 48,525–165,942 | 379,828 B |

计数覆盖实际保存请求的所有 system/content 文本，使用 [DeepSeek 官方 tokenizer](https://api-docs.deepseek.com/quick_start/token_usage/)，无截断。它是离线估计，**不是实际 usage，也不是 SOL 的精确 tokenizer**。对本机 272,000 × 95% = 258,400 的可用上下文，只能作为预检代理：最大 facts 包 168,981 + 65,536 输出 + 1,024 封装 + 8,192 额外余量 = 243,733，剩余 14,667。官方 API 的 1.05M 不能代替本机限制。

当前保持 **36 logical units = 每名评者 36 份 facts 请求**，不分时段、不递归摘要。两名评者合计 72 份 facts 载荷；实际发送数为 0。report 仍是不可执行草稿，不能把 facts 的容量 PASS 当作 report 的容量 PASS：完整自己的 facts 答案、所需 full 切片、有界报告及实际请求材料加入后，必须再次按最终包核算；必要时由 root 采集前冻结阶段输出预留。预计每名评者再 36 份 report 请求，但尚未验证或放行这一阶段。

## 验证与复现

- 旧 24 回合选择及顺序、全 pool 成员不变，旧 217 份可读 JSON 的哈希全部保持。
- 新旧选定回合 540,493 个原始/静态值逐字段、dtype、shape、字节一致；新增导出 540,445 个非静态原值自身往返检查通过。
- 实际供应的 260,955 个 raw 值、64,005 个 direct 值回转通过；组层 1,096 行（包含 scope 重叠）保持分母，Stair 区间统计以独立布尔 mask 重算一致。
- Push 全 192 回合的 227 个实际事件峰值直接复核原 NPZ，使用窗内最早最大值并保留该步完整 pose/motion；组包中含 scope 重叠的 527 个峰值引用及 dwell 中断重算一致。
- exporter 7 项 CPU 单测通过；额外检查 float32 次正规值/负零/极值、同身体非连续段及 terminal-only 末窗极性。无开发语义打分。

从仓库根运行：

```powershell
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/export_confirmation_raw.py --stair PLAN/feedback_sys/P3R/results/stair_numeric_dev_r1/numeric_evidence/manifest.json --push PLAN/feedback_sys/P3R/results/push_numeric_dev_r1/numeric/public_dev-01/manifest.json PLAN/feedback_sys/P3R/results/push_numeric_dev_r1/numeric/public_dev-02/manifest.json PLAN/feedback_sys/P3R/results/push_numeric_dev_r1/numeric/public_dev-03/manifest.json --output <new-raw-output> --purpose development --profile related_v1
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/prepare_confirmation_materials.py --raw-export <new-raw-output> --output <new-packed-output> --profile related_v1 --context-tokens 258400
```

默认 `--profile original` 继续旧材料布局。新 exporter 的 `related_v1` 只补额外峰值定位与样本；packer 执行固定的相关字段/分布投影。两个输出目录必须全新。本次使用 `raw_export_development_related_v1` 与本目录。`verification.json`、`tokenizer_check.json` 和 `packing_receipt.json` 分别保留原值证明、完整请求容量和逐包来源哈希。

root 仍须冻结：正式协议中的这两层审阅范围；N5 源与选择结果；双方真实可用 context/输出预留；媒体供应清单；report 切片及所有真实 facts seals。此处没有生成任何人工答案、正式 PASS 或 release。
