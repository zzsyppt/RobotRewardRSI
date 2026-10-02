# R2 正式 SOL facts 执行准备

当前只完成离线准备，**没有执行 `run_sol_review.py`、没有调用模型、没有生成 facts seal**。准备回执是 `sol_facts_execution_preparation.numeric.json`。这里不读取模型原答或评分，也不把控制文件加入评委上下文。

## 已核实

- 36 个数值 facts 的本地 prepared 在 `sol_facts/<packet>/prepared`，顺序来自真实 packing receipt：`episode-01` 至 `episode-24`，随后 `focus-01` 至 `focus-12`。每份请求 SHA 与服务器对应 SOL 原请求相同。
- 已下载完整原计量回执 `numeric_prepared_text_counts.remote.json`（SHA `ac9ebe2fe418e668b604a3ccc6e8dbde35aabead9d1e2a45e7072aa837954a95`），把其中 36 条 SOL 实际请求 SHA 与本地逐一配对。最大输入文字代理为 174,126 tokens，加登记预留为 247,854；静态文字比较全部满足卡中边界。计量使用 DeepSeek tokenizer，**不是实际 SOL token 计数**，没有证明模型能评完。
- 本机实际 CLI 为 `C:\Users\lover\AppData\Local\OpenAI\Codex\bin\faa963e871dd422c\codex.EXE`，只执行了 `--version` 和 `exec --help`，返回 `codex-cli 0.158.0-alpha.2.1`；冻结入口使用的具名 flags 均在 help 中。
- 固定入口仍为本轮 frozen `run_sol_review.py`，SHA `ec6c2b00c2f1a082476c54f498ce7e644d1f39b4c9c803fe8e45b38ab78b4276`。没有新模型调用器或调度平台。

## 当前尚未由本准备核实的前提

1. 两任务 root 模块核对、全部原观察及附件的实际封存。任务交接时 Push 仍在封存中；主线应引用真实回执，不能以本文件代替。
2. 4 个媒体 facts 的实际 corrected pack、原图字节/顺序、Windows 迁移和完整请求计量。尚未收到材料就绪通知；36 个纯数值单元不能代表全 40 单元。
3. 每个实际调用仍应核新输出路径、完整输入及适用容量边界；未发生实际模型阅读，不能登记“容量执行通过”。媒体的图像 token 与后端缩放仍不可精确测得。

待媒体生成后，复用 `relocate_sol_facts.py` 的 `export` / `prepare`；从真实 media receipt 读取 manifest、packet 和对应 prepared，不猜文件名，不覆盖既有 36 个数值目录。另写真实媒体迁移回执，并将四份原图材料的 id、SHA、相对顺序与请求中的 image block 逐项核对。

## 唯一实际执行命令

下列为**待主线满足前提后调用的接口**，本准备没有执行。`$packet` 必须来自最终 40 单元清单，不能根据内容选择。准备 JSON 已给每个数值单元登记精确 argv。

```powershell
$repo = 'D:\humanoid_agent\true_RL_Agent'
$control = "$repo/PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r2"
$source = "$repo/PLAN/feedback_sys/P3NEW/results/n5_confirmation_source_r2/source"
$packet = '<真实 packet_id>'
& "$repo/.venv/Scripts/python.exe" -B "$source/PLAN/feedback_sys/P3NEW/run_sol_review.py" `
  --prepared "$control/sol_facts/$packet/prepared" `
  --output "$control/sol_facts_execution/$packet" --execute-one-call
```

入口实际为每次新建 ephemeral CLI，`gpt-6-sol`、high、上下文 512000、自动压缩阈值 450000，memory 关闭，无 resume，`project_doc_max_bytes=0`，忽略用户配置，不改全局设置。给出的正文和原图保持冻结顺序；CLI 附件保留图像相对顺序，但不宣称复现 API 图文块交错布局。

**“无工具”必须按真实实现表述：**冻结 prompt 禁止使用工具，`-s read-only` 并不技术禁用读文件等工具。入口记录 `non_message_items`、`compaction_events` 和原事件流，没有自动将所有违规事件变成失败。主线须核实际事件与覆盖回执：使用工具、压缩、截断、漏评或 `complete=false` 按冻结规则保留未完成，不能只看原答一句 `complete=true`。32768 输出只作预留和文字要求，CLI 没有该硬上限。

## Facts → report 的屏障

两家都必须完成并封存 **24 episode + 12 focus + 4 media，共 40 个 facts**，才可发送任意 report。缺失或未完成的包保留在分母中；不自动重试，不裁掉难段或换载体重算同轮。

每个 SOL seal 绑定实际本地 facts manifest SHA 和该评委完整原答的真实 SHA，不使用其他评委答案，也不预造 seal。Report 通过冻结 `prepare_report_manifest` 从实际 local facts 构造，追加原报告/派生/原观察等已登记材料，随后 `prepare(..., 'gpt6-sol', ..., own_facts_seal)`；服务器与本地的控制路径变换由迁移回执解释，不能假装 manifest SHA 相同。

Report 必须包含**本人完整事实原答**再核全请求容量。Facts 目前的文字计量不能替代 report 容量检查，也不能补成必要事实召回或语义通过。
