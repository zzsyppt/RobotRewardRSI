# 单个旧开发 episode 的真实原答与报告长度核算

## 结论与范围

这次只证明：将一次实际返回的完整原答接入现有 report 序列化流程后，**这个请求**的静态文本长度满足原先的保守容量算式。它不证明 GPT6-SOL 已完整阅读原始证据，不证明 facts 符合完整评阅要求，也不证明 report 能完整返回。本次不是 N5，不登记语义 PASS，不放行 report 评阅。

原答有 24 条 `necessary_facts`，41,030 bytes，原始 SHA 为 `312365777c99b3e7666a86d914da759ca1fc54e3de9c3529a2da3186f163c2f7`。原答和 access receipt 均保留原样；未要求原评阅者补答或修答。

## 阅读与完整性的四种不同含义

| 项目 | 这次实际证据 |
|---|---|
| 原评阅者的完整性声明 | 原答与 access receipt 自报 `complete=true`；完整保留该声明。 |
| CPU 对源材料的遍历 | 独立展开 525 步的 11 个数值字段，共 25,200 个值；按原 dtype、shape、C-order 重建的字段 SHA 全部一致。5 个白名单文件的 SHA/bytes 与材料清单一致。 |
| 模型实际可见材料 | access receipt 记录了 25 步间隔的扫描及若干事件窗口。不能把程序完整遍历等同于 143,114 个 raw numeric 代理 token 全部进入模型上下文。 |
| 真正的完整上下文容量与语义完整性 | **未证明**。实际上下文峰值、是否 compaction 均为 unknown；没有该 agent 的精确 thread/rollout 定位，未扩大搜索其他线程。facts 程序合规与语义完整性未判 PASS。 |

原答内容以单个成功 Stair episode 为范围，没有把它升格为群体失败比例，也明确没有图像材料与因果机制证据。这里只做边界阅读，**没有对 24 条事实逐条给出语义正确性评分**。

## 封存与真实序列化

复用 `run_confirmation_review.py` 的 facts seal 与 `--own-facts-seal` 路径；seal 仅封存原答及其来源身份，不表示事实已合格。原答作为 JSON 数据完整接入，无删条、截断或改写。原来的排版空白会按 runner 的标准 JSON 序列化改变，但反序列化后的原答与输入数据逐值完全相等。

报告仍采用 `capacity_event_table_r3` 的同一个 episode-02：525 步、620 个事件。除 own facts 块外，system、完整数值原始材料、协议、问题、完整派生报告及所有请求参数与 r3 逐值相等。没有删样本、改分母或改变报告判分项。

新请求 `prepared_report/episode-02/gpt6-sol/request.http_body.json` 为 381,417 bytes，SHA：

`74c3bd9209ecc3de048aa89398ac4db8fdbfaea6b058a38b3a2b62b7ffdfa5a3`

状态仍为 `PREPARED_NOT_RELEASED`，只有既有 runner 自动生成的 `release.draft.json`，没有真实 release、report 模型调用或 GPU 运行。通用准备器的 HTTP 兼容载体仍写作 `deepseek-flash`；它不是实际发送给 SOL 的载体，也不能作为 SOL 的实际 usage 记录。

## 静态分词结果

使用与 r3 相同的官方 DeepSeek v4 静态 tokenizer，逐个计完整 text block，`add_special_tokens=False`。服务器已有 CPU tokenizer runtime；没有匹配缓存，因此仅下载公开静态 tokenizer ZIP，严格核对原冻结的 ZIP 与 tokenizer SHA；没有访问模型 API。该 tokenizer 对 SOL 仅是**代理计数**。

| 完整文本部分 | r3 人工容量占位 | 本次实际原答 |
|---|---:|---:|
| system | 148 | 148 |
| review protocol | 4,857 | 4,857 |
| focus questions | 378 | 378 |
| raw numeric | 143,114 | 143,114 |
| full derivation | 14,570 | 14,570 |
| own facts | 32,901 | 10,176 |
| 合计 | 195,968 | **173,243** |
| 加原预留后的总量 | 269,696 | **246,971** |

原预留保持不变：system/tools 32,768 + output 32,768 + margin 8,192，有效容量 258,400，材料目标 180,000。本次距材料目标余 6,757，距原保守总量上限余 11,429。32,768 wrapper 是原规则的保守假设，**不是实测 Codex 开销**。原答文件自身的含缩进文本为 12,995 代理 token；runner 紧凑序列化后的 own facts 块为 10,176，差别来自序列化形式，没有删掉数据。

`tokenizer_check.json` SHA：`4e013174a3628bf37d711c1fcf3135deae9b1e6b77dd66fc6328ba06411d1ba7`。同时重算的旧 r3 基线仍为 195,968，与旧登记完全一致。

## 仍未解决的验收条件

1. 这份答案的大小不能作为任何其他包的上界；其事实完整性尚未建立。
2. 完整证据实际进入 SOL 上下文及过程不被 compaction 改变，仍未证明。
3. 620 events 与完整派生字段所要求的 `all_claims` 真正输出长度、是否能在 32,768 输出上限内完整返回，仍未测试。不能用输入能放下代替输出验收。
4. 这一份 tool-assisted 原答不能被直接计入正式 N5。正式流程仍需保持完整 episode、完整判分分母与原定独立评阅边界。

本目录新增的两个脚本都是这一次 CPU 证据核算的复现材料，未进入生产模块。`prepare_actual_report.py` 复用既有准备器；`measure_token_capacity.py` 读取 `tokenizer_materials.zip` 可复算静态计数。旧 r2/r3 材料与原始原答均未修改。
