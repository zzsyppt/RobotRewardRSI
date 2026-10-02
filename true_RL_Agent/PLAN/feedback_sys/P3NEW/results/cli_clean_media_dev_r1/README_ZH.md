# 旧图文包的独立 CLI 运行检查

2026-10-01。开发工程检查，不是 N.5 独立验收，不运行作者、DeepSeek 或训练，不启动新采集。

固定复用 R2 已归档的 `media-04`：4 份完整 JSON 与 12 张原始 PNG。先逐项核对原 SHA，仅改本地文件定位和开发用途标记，然后用当前入口生成一份 `gpt6.1-sol` 请求。完整 HTTP body 为 30,774,007 B；转换后原文与 PNG 可完全重建，stdin 含完整消息载体，不把原图作为前置附件。

主线入口为 `P3NEW/prepare_desktop_review.py`、`run_sol_review.py` 和既有 `run_confirmation_review.py`。3 项 CPU 检查全部通过，覆盖全文/图像顺序、本人 facts seal、事实阶段隔离和实际看图回执。CLI 政策使用 `N5_SOL_CLI_POLICY_R4.json` 的 512000 窗口、450000 自动压缩线、GPT-6.1-SOL/high；32768 是输出预留，不冒称 CLI 硬输出上限。

实际一次运行于 UTC 17:50:25 登记，UTC 18:08:59 结束，退出码 0，线程 `01a0f370-6981-7391-9972-63dbb21a0371`。使用独立临时 home，清除继承的桌面任务身份，按进程参数关闭记忆/技能/插件，无 resume。终态已核实临时复制的 `auth.json` 被移除；原 stdout 与 rollout 留在私有运行目录，仅导出选择性访问元数据，不复制私有推理。完整输入、实际执行脚本和政策 SHA 见 [登记](./execution/registration.json)。

运行时 CLI 通过 code-mode 的 `exec` 包装原生 `view_image`。原自动提取器只识别直接 `view_image`，因此原 [回执](./execution/receipt.json) 保持 `TRANSPORT_NOT_VERIFIED`；该状态是提取器缺口，不能据此断言未看图。最终主线提取器只额外接受按当前白名单序号逐字匹配的固定包装，不解释任意 JavaScript，其他包装或多余工具仍拒绝。

[最终离线复核](./final_access_reaudit.json) 为 `PASS_ENGINEERING_ACCESS_AND_ANSWER_STRUCTURE`，26 项机械核验全部通过：12 个看图包装、12 个同 call_id 返回和 12 个 ImageView 事件均按 `image-0000` 至 `image-0011` 顺序匹配；每个返回均为原 PNG 字节 SHA、`detail=original`、1920×1080，未见额外工具调用。原输入及 12 张原 PNG 的执行后字节未变，实际模型/effort 为 `gpt-6.1-sol/high`，未观察到压缩。复核单独绑定原执行脚本、原回执、stdout、最终 rollout 和当前提取器 SHA，没有改写任何执行原件或再次调用模型。

实际 CLI 累计用量为 input 1,049,554、cached input 947,584、output 21,794、reasoning output 2,932。输入累计包含初始请求及 12 次看图后的续请求，不能与单次上下文窗口比较；rollout 记录的单次输入最大值为 95,816，观察到的窗口为 486,400。配置请求仍是 512000/450000；没有独立证明后端完整配置、图像内部处理或隐藏重试次数。

原答为 69,742 B，SHA `da3224a949d696f77fb17fab71773133750811c4af468307f429bdf88d700c42`。当前开发检查确认身份/packet/stage/complete envelope 正确，129 项引用全部可解析，`mechanical_valid=true`、`review_complete=true`，没有结构错误。该检查不证明事实语义正确，不封存正式 facts 或计入 N.5 分数。

已执行 wrapper 要求直接原生看图；未来 wrapper 明示这一个规范 code-mode 包装，原文字块与原图不变。已执行脚本 SHA 为 `c861f4dc82b63e248e5b4a5aab5706e1c3d6577e90a0f28c41fe831ab63631fe`，最终提取器 SHA 为 `771ef1ab53fc4483233865bb766ecedb1890f25d2b2421ab54f907bec478bafa`，最终私有 rollout SHA 为 `da7196fec445ace8941920945a07e6cc7cb01ec601f869b20dfd342a695c0a58`。

这一个旧 media-04 包的执行、原图访问和答案机械结构检查已闭合，没有发现阻塞。它不代表全部包容量或正式语义验收；工具访问核验也不等同于 OS 文件读取隔离。返回的旧资料事实仅作工程输出，不据它调高分数或选例。
