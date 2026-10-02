# N5 数值材料离线打包开发验证

2026-09-30。**NOT_FROZEN / NOT_RELEASED**。仅用既有 `raw_export_development_focus_v2`；没有新 N5 数据、API、GPU 或语义评分。编写者参与开发，不是正式评委。

已用现有 `run_confirmation_review.prepare` 生成 24 episode 和 12 focus 的 facts manifests，以及每包 DeepSeek / GPT6-SOL 两份准备件。两评者实际材料字节及顺序完全一致。GPT6-SOL 准备件是供全新 agent 阅读的同材料载体，不是 GPT API 调用。全部请求均标记 **offline production serialization，未真实发送**。

每个 episode 的 header、全部连续 values 分片（合计 144 片）在同一请求中；12 focus 保留全部 scope rows、全任务成员分母、原始结局和新增直接算术操作数。问题只按 episode/focus 的资料范围区分，不按报告阳性挑事实。原始 NPZ 的选中回合切片保留在材料目录作来源审计，不代替模型实际收到的可读数值。私有身份映射、生产报告、旧评分均未投递；本包没有媒体，不能将 not_supplied 写成 not_recorded。

| 任务/轴 | 包数 | 原始 JSON UTF-8 字节/包 | 实际 DeepSeek HTTP body 字节/包 | 保守所需上下文估计/包 |
|---|---:|---:|---:|---:|
| Stair episode | 12 | 383,064–1,366,645 | 429,860–1,478,269 | 502,082–1,549,733 |
| Push episode | 12 | 69,936–227,394 | 97,528–268,872 | 170,636–341,336 |
| Stair focus | 6 | 1,153,374–1,377,215 | 1,396,897–1,664,133 | 1,382,517–1,649,455 |
| Push focus | 6 | 369,781–1,475,608 | 452,270–1,735,368 | 502,398–1,725,252 |

每包估计按实际模型可见文本 UTF-8 字节数 × 1 token/byte，再加 1,024 对话包装、65,536 输出预留及 8,192 余量。**这是保守工程估计，不是供应商 tokenizer 的实数或上下文能力保证。** 当前卡没有冻结两评者可用上下文，所以可执行性结论仅为“数值内容齐全，既有 runner 可离线序列化，全部 body <48 MiB”；模型上下文仍未核准。HTTP 字节通过不能替代上下文通过。需要拆分时应在正式采集前冻结跨片时序与汇总方式，不能事后丢分片。本轮没有自行拆分或压缩数值。

`packing_receipt.json` 保留全部 36 包两种载体的实际字节、请求 SHA、文本字节、输出预留、保守估计和来源 hash；`packet_sizes.csv` 可直接比较。`verification.json` 验证 204 份复制文件与来源字节一致，72 份请求重建字节一致，36 份 facts manifest 可构造，144 数值分片齐全，双方材料相同。

36 份 `report.manifest.draft.json` 已绑定相应 facts SHA，并列出依赖，但刻意使用 **draft schema**，现有 runner 拒绝执行它们。不是完成的 report 请求，原因如下：

1. episode 尚须按真实来源映射切出完整 full 语义记录及共享定义；不能按默认报告引用窗口截断。
2. focus 尚须整份有界报告、相应组级 full 主张与最终请求。若用现行生产源码构建，须记录其来源/源码 hash 和 raw 输入绑定；离线序列化不能称“作者已实际接收”。
3. 双方真实 facts 答案及各自封存件尚不存在。不能伪造占位答案绕过 runner；root 需在所有 facts 完成后核准全包屏障，才允许 report 揭示。

本次只读现有草案，没有冻结或修改 protocol/capture card。root 仍需冻结正式匿名问题/规则、上下文容量与输出预算、必要分包方案、report 来源与对应关系、角色与放行；媒体由独立准备工作处理。

离线复现（使用一个不存在的新输出目录）：

```powershell
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/prepare_confirmation_materials.py --raw-export PLAN/feedback_sys/P3NEW/results/raw_export_development_focus_v2 --output PLAN/feedback_sys/P3NEW/results/NEW_OFFLINE_DIRECTORY
```

单包沿用现有调用入口重新准备（仍不放行、不发送）：

```powershell
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/run_confirmation_review.py prepare --manifest PLAN/feedback_sys/P3NEW/results/n5_numeric_packing_development_r1/packets/episode-01/facts.manifest.json --judge-id deepseek --output PLAN/feedback_sys/P3NEW/results/NEW_SINGLE_PACKET_DIRECTORY
```

已有目录中 `release.draft.json` 全为 `PREPARED_NOT_RELEASED`，费用/usage 均未发生。不能将 36 logical units 或 72 份离线准备件直接当成已批准调用数。
