# 完整事件表显示验证（旧开发单包，未放行容量）

本次只改变新 `related_v1` Stair 评阅材料中完整 episode 的 `events` 显示：普通列名 `body, code, start, confirmed, end` 加原顺序 `rows`，存在的 `measured` 按原零起始事件 index 补充。行 i 对应原 `/events/i`；完整文件定位为材料的 `source_pointer` 加该路径。body/code 字符串、所有原值、null、原索引及顺序均保留，没有整数词典或自创等号语法。Push 和原 profile 不变。

入口为 `prepare_confirmation_materials.py::pack_capacity_reports()`，调用 `review_episode_derivation(row, *, task, material_profile)`；`expand_review_episode_derivation(row)` 仅供离线重建完整原对象。未知事件字段会报错，不会丢弃。

验证使用已有 `n5_review_capacity_development_r2` 的 `episode-02`（525 步、620 条事件）。从旧材料的原完整文件和 `source_pointer` 重新取得对象，重建后的完整 episode **canonical JSON 字节相等**。旧完整源、材料、manifest、合成 facts 压力占位和旧请求的哈希前后相同；实际请求中仅 `full_derivation` 文本块变化，其余块和输出设置相同。4 个小测试通过，覆盖顺序、可选 measured/null、空表、profile/task 边界和未知字段拒绝。

| 测量 | 旧 r2 | 新 event table r3 | 减少 |
| --- | ---: | ---: | ---: |
| events 紧凑 JSON 字符数（本例全 ASCII） | 50409 | 26059 | 24350 |
| full_derivation 代理 token | 23762 | 14570 | 9192 |
| DeepSeek 请求输入代理 token | 205153 | 195961 | 9192 |
| SOL 请求输入代理 token | 205160 | 195968 | 9192 |
| 每个实际序列化 HTTP 请求字节 | 447877 | 414263 | 33614 |

计数使用与 r2 相同且 SHA 固定的公开 DeepSeek v4 tokenizer，对实际请求的每个 system/text 块计数（`add_special_tokens=False`）。旧基线已重现。此数值是 SOL 的代理计数，不是真实 SOL 分词或 provider usage；也未计入实际 Codex 运行外壳。只在远端 CPU 中读取公开静态 tokenizer 文件并计数，没有模型 API 调用、GPU 工作或新确认材料读取。

**容量仍未放行。** 保持原 180000 材料目标及原预留：system/tools 32768、输出 32768、额外余量 8192；SOL 新总量为 `195968 + 32768 + 32768 + 8192 = 269696`，仍比有效上下文 `258400` 多 **11296**。新输入本身仍超过 180000 目标 **15968**。本次未改变目标、预留或任何质量门槛。合成 own-facts 仅用于压力测试，不是正式评委回答，更不是运行时自动核对者结果。

机器回执为 `verification.json`、`tokenizer_check.json`；两个新请求卡均为 `CAPACITY_ONLY_NOT_EXECUTABLE`。`verify_capacity.py` 记录整个复现流程并采用排他写入，避免覆盖已有结果。

关键 SHA-256：

- preparer：`085b99cadbc69b124bdf48d000d615cc8b5c06695574136969fe480588bc9317`
- 完整原始源：`0e99375ca17ef8d21fcf673e12827c4ed5d4972621ffe9e38efe0cfa81b9061a`
- 原 events canonical：`71e618d2b6d40ef68537ba000637fe683fed6aef8ccff9c9055daf6f2326b580`
- 完整 episode canonical：`ee2bbfc6f3214d0c1c76c6283c97616b2d0e7646e26516f480d4e5cfb3a796dc`
- 新 SOL 请求：`37ac085b598bc311c40603398e8b6eade8a2ebbdc8e3058b96414aa2cf4589de`
- 未修改的 runner：`c09ef80569d1649a42c397285e12c91a9dabd8bdff9ab87f6c902f733bd5d038`
