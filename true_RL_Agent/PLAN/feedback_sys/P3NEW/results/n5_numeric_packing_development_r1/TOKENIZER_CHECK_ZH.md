# 乱码核查与完整数值包容量补充

2026-09-30。仅旧开发材料，离线 CPU；无 API、GPU、模型权重下载或软件安装。旧导出、请求和 `packing_receipt.json` 均未改写。

**没有复现源文件乱码。** `focus/push_all.json` SHA 为 `b4071ee5f2b57ae51b19513a142dff1a29d052d2c2c320f9dd6fadc588885a69`；五束问题直接 `json.loads(bytes)` 后均为正确中文码点。原导出 220 个 JSON、原打包 507 个 JSON，以及 72 个请求的模型可见文本均无 U+FFFD 或其转义。导出器自身也没有替换符。`unicode_check.json` 用 ASCII 转义和 UTF-8 十六进制保存证据，不依赖终端能否显示中文。当前证据不足以定位展示端为何出现替换符，但足以否定本地源文件包含这些替换符；无需修改导出器或重导数值。

**先不压缩或拆分数值。** 服务器既有 `tokenizers 0.22.2`、`transformers 4.57.6` 可用；常规 Hugging Face 缓存和已查 AGENT/isolated 目录没有现成词表。使用 [DeepSeek 官方 Token Usage 页](https://api-docs.deepseek.com/quick_start/token_usage/) 当前直接链接的 [tokenizer zip](https://cdn.deepseek.com/api-docs/deepseek_v4_tokenizer.zip)，仅在内存读取分词器数据，以现有 CPU 库计数全部原始 system/content 文本块。没有执行下载包的 Python 代码，没有加载模型。

压缩包为 1,911,504 B，SHA `e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495`；其中 tokenizer JSON SHA `89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614`。数字预切分为每组最多三位，故不再使用 1 token/byte 作为主要容量判断。计数禁用 padding/truncation，未省略任何文本块。

| 任务/轴 | 完整文本离线 tokenizer 计数 | 加包装、输出与余量后的所需上下文估计 |
|---|---:|---:|
| Stair episode | 204,481–730,258 | 279,233–805,010 |
| Push episode | 38,767–117,117 | 113,519–191,869 |
| Stair focus | 557,274–690,095 | 632,026–764,847 |
| Push focus | 169,666–678,929 | 244,418–753,681 |

每包额外保留 1,024 包装、65,536 输出与 8,192 余量。最大为 `episode-02`，并非 HTTP 字节最大的 `focus-01`。按 [当前模型页](https://api-docs.deepseek.com/quick_start/pricing/) 的 `deepseek-flash` 1M 上下文，全部数值 facts 包在此离线估计下可一单位一上下文；最大合计 805,010，即使按 1,000,000 计算仍有 194,990 token 余量。输入与输出共同受上下文上限约束，见 [Chat Completions](https://api-docs.deepseek.com/api/create-chat-completion/)。

这是**官方当前链接 tokenizer 的离线估计，不是实际 usage**。未验证该词表与 API fingerprint 的精确一一对应，服务器消息编码也可能另有开销，因此不能写成真实发送/精确计费。逐包计数和原请求 SHA 在 `tokenizer_check.json`；旧字节上界 receipt 保留作历史记录。SOL 的可用上下文与分词方式仍应独立核准，不能拿 DeepSeek 计数替它担保；report 阶段增加的 full/report/own-facts 也须重新测量，当前仅 facts 容量。

最小下一步是 root 核准 SOL 可用上下文，并据这些实际离线计数冻结每包 reservation。若任一评者确需缩短，才考虑只对已明确声明 float32 的原始字段使用可逐值还原 float32 位模式的短十进制；需从原 NPZ 验证 roundtrip，且不能顺带舍入 float64 派生量。全组分母和每回合全部时序不能删。当前没有容量证据要求引入跨请求评阅或改写数值。
