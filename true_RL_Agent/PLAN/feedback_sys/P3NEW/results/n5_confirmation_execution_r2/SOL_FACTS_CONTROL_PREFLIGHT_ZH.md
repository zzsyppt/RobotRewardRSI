# R2 SOL facts 控制预检

本控制助手只核执行和材料元数据，不是正式评委；未阅读正式评委原答、原观察语义或旧评分。SOL 模型调用、facts seal、report 均为 0。主线尚未发出正式 SOL 启动，随后明确要求保持暂停。

`sol_facts_control/preflight.final.json` 是最终控制源码的实际离线检查回执；早一份 `preflight.json` 原样保留。40 包按 `episode-01..24 → focus-01..12 → media-01..04` 固定顺序，重新核 SHA、材料原字节、冻结 builder 生成请求一致性、原图顺序与尺寸、全新输出路径。四份媒体来自实际清单中的 `sol_facts_media_bundled`；旧缺 Pillow 的准备失败保留。没有修改冻结文件、prompt 或已有准备清单。

当前状态为 `ALL40_INPUT_PASS_ENVIRONMENT_BLOCKED_NO_MODEL_CALL`。原登记 CLI 是 `faa963e871dd422c/codex.EXE`（`0.158.0-alpha.2.1`），该路径现已不存在；当前 `shutil.which("codex")` 得到 `c6fe824d725f02d7/codex.EXE`（实际 `--version` 为 `0.159.2`）。新版 `exec --help` 返回 0，冻结入口使用的具名 flags 均存在。配置键在实际模型调用中是否被接受尚未验证；这不是原 CLI 不变的证明。首次预检在发现此差异时以 1 停止，另存真实失败回执；后续只继续离线输入检查，并保留环境阻断，未释放调用。

两任务 root 模块回执元数据和审查文件 SHA 已核验，封存分别为 UTC 08:46:05 与 09:31:13，回执登记封存前 facts/report 调用均为 0。没有打开审查正文或原观察正文供本助手评判。

容量的已知边界是：最大文字输入代理 184,862，加登记预留为 258,590；代理输入阈值 360,000。每份媒体 12 张 1920×1080 原图，合计 48 张；最大 prepared HTTP body 为 31,735,746 字节，低于冻结 48 MiB 载体限制。此 body 限制不是 SOL CLI 上下文 token 保证。文字代理使用 DeepSeek tokenizer；SOL 实际输入 token、图像 token 和后端缩放仍未知，未套用 DeepSeek 每图 1024 公式。冻结入口配置为 SOL/high、512,000 context、450,000 auto compact；是否无压缩地完整完成只能由实际调用事件、usage 和覆盖结果支持。

`sol_facts_control.py` 默认只做离线预检。执行分支仍调用唯一 frozen `run_sol_review.py --execute-one-call`，每包 fresh ephemeral/no memory/no resume/no inherited history；不改 prompt。除主线显式启动记录外，当前 CLI 漂移阻断也必须处理，脚本现在不会越过该阻断。它保存原入口的 event、raw、usage、coverage、exit，以及控制 stdout/stderr、dispatch 和结果。reasoning、agent_message、todo_list、plan 不自动计为工具；实际工具类型单列，未知类型单列并停止待人工控制分类。非零退出、工具调用、压缩、complete 不为 true 或冻结机械覆盖未完成即停止，不自动重试。全部 40 仍留在分母中。

即使将来机械检查通过，本控制也不作语义通过声明、不封 facts、不准备 report。两家全部 40 facts 封存前不得进入 report。
