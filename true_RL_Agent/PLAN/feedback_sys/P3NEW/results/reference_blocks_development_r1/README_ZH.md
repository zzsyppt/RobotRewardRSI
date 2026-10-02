本目录完成旧 40 个 R2 facts 包的 v2 输入呈现工程验证与静态文字容量测量。工程检查通过，文字代理比较通过；未释放任何模型调用，也未形成语义验收结论。

原始材料来自 `n5_confirmation_execution_r2`。新开发 manifest 使用 `p3new.reference_blocks.v2`，目录仍使用 `p3new.reference_anchors.v1`。同材料 v1/v2 对照共同使用本次固定的协议文本；历史协议、HTTP、原答案和旧计量均未修改。此结果不适用于后续新增或重写的 `related_v2` 语义材料。

| 检查 | 结果 |
| --- | --- |
| 原 facts 包 | 40 个，旧 HTTP 重建逐字一致 |
| 重建材料 | 全部值、JSON 类型、原键顺序一致 |
| 目录与引用 | 3,910 项历史 v1 锚点保持不变；3,862 个完整值块和 48 个图片锚点 |
| 图片 | 48 张原 PNG 字节、显示顺序、紧邻前方的原 metadata 文本均保持 |
| 两位评者 | 80 个 prepared request，逐包 user blocks 一致 |
| 工程检查 | 4,413 项，0 项失败 |
| v2 最大 HTTP body | 31,796,660 字节，`media-03` |
| v2 最大全部消息文字 | 442,421 UTF-8 字节，`media-02` |
| v2 最大单个文字块 | 223,790 UTF-8 字节，`focus-01` |

文字计量通过 SSH stdin 在 `humanoid-server` 上执行一次纯 CPU 脚本，读取已缓存的 DeepSeek V4 tokenizer，没有下载 tokenizer、写远端脚本文件、读取 API key、调用模型或使用 GPU。对每个真实 prepared request，先核对本目录工程回执绑定的 SHA-256 和字节数，再逐块统计完整 system/user 文本；不把 PNG 的 base64 当作文字计数。tokenizer JSON SHA-256 为 `89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614`，`tokenizers` 版本为 `0.22.2`。

| 文字计量 | 结果 |
| --- | --- |
| 实测请求 | 80 个，40 包 × 2 位评者 |
| 全部文字最大值 | 209,157 tokens，`media-02 / gpt6-sol`；该请求含 12 张未计 token 的图片 |
| 目录文字最大值 | 13,017 tokens |
| 单文字块最大值 | 115,472 tokens，`focus-01` 的一个原值块 |
| 所有请求的输出上限 | 32,768 tokens |
| 额外系统/工具预留 + 输出预留 + 代理余量 | 32,768 + 32,768 + 8,192 = 73,728 tokens |
| 最大文字加上述预留 | 282,885 tokens |
| 与 360,000 文字输入代理限制的差额 | 150,843 tokens |
| 与 450,000 自动压缩比较线的差额 | 167,115 tokens，尚未加入图片等未测开销 |
| 与 512,000 上下文比较线的差额 | 229,115 tokens，尚未加入图片等未测开销 |

以上计量状态为 `STATIC_TEXT_MEASURED_NOT_RELEASED`。DeepSeek tokenizer 对 SOL 只是代理；图片 token、provider chat framing / special tokens、实际 SOL tokenizer、工具运行时上下文峰值和 provider 实际 usage 均未测量。文字代理余量不能作为多模态容量释放凭证。材料可重建、引用可解析或总量低于文字代理比较线，都不证明必要事实完整、任务含义正确、模型会选对引用、未知量处理正确或有界观察者合格。未生成任何 facts seal。

文件入口：

- `receipt.json`：40 包工程检查、逐包大小、所有读入来源的 hash 和检查结果。
- `packets/<packet>/comparison.json`：同材料 v1/v2 对照；两位评者的 prepared request 位于同包的 `deepseek/prepared` 和 `gpt6-sol/prepared`。
- `executed_source/`：本次工程验证使用的 runner、coverage 与协议快照。
- `text_capacity.json`：80 请求的逐文字块 token 结果、代理比较与明确的未测项。
- `text_capacity_execution.json`：实际 SSH 命令、退出码、运行时长、stdin 脚本及输入/输出 hash；本次退出码 0，约 49.8 秒。
- `verify_reference_blocks.py`、`measure_text_capacity.py`、`count_text_remote.py`：实际执行脚本；两个本地入口拒绝覆盖已生成的对应结果，远端计数脚本只向 stdout 输出。

工程验证使用 bundled Python（PNG 校验需要已存在的 Pillow）；文字调用器使用仓库 `.venv/Scripts/python.exe`。两个工作流仅处理本目录的新开发产物，没有执行正式评审或训练。
