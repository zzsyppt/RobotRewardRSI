当前结果：离线重建及独立输入算术核验已完成，随后两例开发检查也已完成；见[两例结果与载体记录](./probes/RESULT_ZH.md)。以下无模型/API的说明只指`attempt_01`离线重建，不包括单独的probes。当前生成器另修正了历史槽顺序的范围措辞，旧attempt及原答保持。

本目录将真实旧 R2 数据重新制作成 `related_v2` 开发材料，完成全部 40 个 facts 包的离线工程检查与文字容量测量。最终有效尝试为 `attempt_01`，状态 `PASS_ENGINEERING_ONLY_NOT_RELEASED`；没有调用模型、使用 GPU、重新采集数据或生成 facts seal。

`attempt_01` 实际冻结的 packer 已包含最后一次 `related_focus` 入口终局投影修正，SHA 与当时当前文件逐字一致。消息先后曾造成“可能需要 attempt_02”的推断，实际快照消除了该疑点，因此没有重复执行或创建 attempt_02。

| 工程检查 | 结果 |
| --- | --- |
| 真实数字材料 | 24 个 episode、12 个 focus，共 36 包 |
| 真实媒体材料 | 4 包，由各自旧 header、完整 JSON parts 和 NPZ 重建 |
| 实际 prepared requests | 80 个，40 包 × 2 位评者，逐包 user blocks 一致 |
| 工程断言 | 1,456 项通过，0 项失败 |
| 原始输入 | 1,510 个文件引用，执行前后 SHA 全部一致 |
| task_definition | 原 4 份合约与 Stair/Push 数字来源绑定保留，36 包原 routing 不变 |
| 原图片 | 48 张 PNG 原字节、ID 与顺序保持；原 media metadata 直接引用且 hash 不变 |
| 最大完整 HTTP body | 31,806,278 字节，低于 runner 的 48 MiB 边界 |

执行时完整复制原 `n5_confirmation_source_r2` 源码快照，再只覆盖本次固定的 PLAN helper 与协议副本。旧 `numeric_raw` 也复制为新的 `raw_development_view`：所有非 manifest 文件保持原字节；manifest 是独立新文件，将 `purpose` 改为 `development` 并记录原 manifest 路径、SHA 和原 purpose。其底层 `material_profile` 仍为 `related_v1`，因为这表示旧 raw archive 的实际导出格式；新 packer 的交付 profile 是 `related_v2`。旧确认原件没有被重新标记或覆盖。

Stair 原供给的完整 raw/discrete/direct 时间列保持原值；仅新增已记录的 `support_planar_speed_mps` 与 `support_reference_clearance_m` 两条直接曲线，并对每个值回解到原 JSON parts 的 float64 数据核验。新的 `direct_arithmetic` 来自原 header 的固定原始算术字段，终局投影去除 `diagnostic_flags`；观测可用性明确区分已供给、可由本包重建、仅未传输、未供给和未记录。该变化覆盖 12 个 Stair 选中回合及 2 个 Stair 媒体回合。

Push 的完整原位姿、平面运动、离散量和允许的原终局/事件值保持；原字段逐值回解核验，不新增 direct 曲线。`direct_arithmetic` 保留原 extrema/events 等中立算术，不包含 `arithmetic_latch_equals_original_record`。12 个 focus 包的完整组分母、成员顺序和原分布操作数与旧包一致；这不代表额外提供了未选中回合的完整轨迹。

| 此批真实 `related_v2` 的文字容量 | 结果 |
| --- | --- |
| 计数器 | 服务器已缓存的 DeepSeek V4 tokenizer，CPU，逐完整 system/user text block 计数 |
| 请求数 | 80 |
| 最大文字量 | 263,606 tokens，`media-02 / gpt6-sol`；其中 12 张图片未计 token |
| 额外系统/工具、输出、代理余量预留 | 32,768 + 32,768 + 8,192 = 73,728 tokens |
| 最大文字量加预留 | 337,334 tokens |
| 360,000 文字输入代理线的差额 | 96,394 tokens |
| 450,000 自动压缩比较线的差额 | 112,666 tokens，尚未加入图片等未测开销 |
| 512,000 上下文比较线的差额 | 174,666 tokens，尚未加入图片等未测开销 |
| 容量状态 | `STATIC_TEXT_MEASURED_NOT_RELEASED`；`capacity_release=false` |

本次没有复用 `reference_blocks_development_r1` 的旧包 token 数冒充新材料结果。这里重新计数了实际新生成的 80 个请求。DeepSeek tokenizer 对 SOL 仍是代理；图片 token、provider framing/special tokens、实际 SOL tokenizer 与工具运行时上下文峰值未测。因此上述比较不自动释放多模态请求，不证明模型会正确解释过程、确定完整必要事实或正确处理未知。新增算术的独立验证属于另行审计，不由本工程回执替代。后续模型探针登记文件也不构成本脚本的 API 授权或执行记录。

固定执行源码的关键 SHA-256：

| 来源 | SHA-256 |
| --- | --- |
| `prepare_confirmation_materials.py` | `996411dc3e2cb3e00faf47d5ce5af5731ba71f6ddf0a74e296e483b1bbfa5800` |
| `export_confirmation_raw.py` | `ff40c9332d5d4cd6235f7d417e0ce27bc87218753f7a207c7713b287b7410478` |
| `N5_FACT_REVIEW_PROTOCOL_ZH.md` | `c0304873b03a9de202d3efb47307315cf4d73bf8b3e202418e03d2bca94a774f` |

本地入口：

- `attempt_01/execution.json`：实际 SSH/Python 命令、stdin 脚本 hash、7 个覆盖源码 hash、退出码 0、约 71.4 秒总时长。
- `attempt_01/executed_source/`：本次覆盖使用的 7 个文件的原字节副本。
- `attempt_01/artifacts/receipt.json`：完整工程回执，含原输入 hash、逐包 manifest/request 身份、检查及容量摘要。
- `attempt_01/artifacts/numeric_verification.json`：36 包原量、新增曲线、终局投影、算术字段与定义绑定检查。
- `attempt_01/artifacts/text_capacity.json`：80 请求的实际逐块文字 token 结果。
- `attempt_01/artifacts/numeric_materials/materials/related/`：36 个真实新数字材料 JSON。
- `attempt_01/artifacts/media_materials/`：4 个真实新媒体相关数字材料、开发 manifest 与各包验证结果。
- `attempt_01/retrieval.json`：只读取回的 91 个 artifact 的 SHA；原 prepared 请求没有被改写成本地路径。

完整远端输出根目录为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/semantic_input_development_r1/attempt_01`。数字 prepared 请求位于 `numeric_materials/prepared/<packet>/<judge>`，媒体请求位于 `media_materials/<packet>/<judge>/prepared`；完整源码副本在 `source_snapshot`，旧 raw 的独立开发副本在 `raw_development_view`。本地 `run_development.py`、`rebuild_remote.py` 与 `retrieve_artifacts.py` 是本次实际工作流，拒绝覆盖既有 attempt 或已取回的 artifact。
