# R2 真实 40 facts 包：引用契约离线工程核验

2026-09-30。**PASS_ENGINEERING_ONLY**：旧协议分支保持请求字节兼容，新 anchor 契约未改变原证据、图片或其顺序。没有模型调用、GPU、正式评分、facts seal 或调用放行；这不是新一轮冻结或正式语义通过。

## 已执行范围

- 24 episode + 12 focus 来自 E2/sol_facts，4 media 来自 E2/sol_facts_media_bundled。sol_facts 中另有较早 media-01 副本，未混入本次40包。最初清单检查因该副本停止，尚未写任何 packet 输出；脚本与停止记录保留在 attempts/01_inventory_selection。
- 当前 runner 对全部40个不带 reference_contract 的原 manifest 重建，HTTP body 与旧 prepared 原件逐字相同。旧 manifest、材料、prepared、原审计回执、原答均未改。
- 新 manifest 全部绝对绑定原文件；仅 review_protocol 换为当前 packet_protocol 输出（保留模态附加条款，并显式标记 development/not frozen），新增 reference_contract=p3new.reference_anchors.v1。
- 每包保存 deepseek 与 gpt6-sol 两份 prepared，两者 user 消息完全相同。剔除 protocol 与新 catalog 后，所有原材料块、label、PNG 数据与排列逐字相同。48张原PNG各自逐字匹配，48个独立SHA均保留；没有降分辨率、裁图或重编码。
- 每个 catalog 条目只有 anchor_id 与 locator，ID由结构定位确定。3,910个ID全部唯一、确定且精确解析到原 envelope 子树；不加入支持/必要性/机制等语义标签。
- 原19条诊断的 actual_existing_locator 均能由新 anchor_id + within_anchor_pointer 精确表达并解到同值。这仅证明现存地址可表达，没有替换原错误引用、修原答或重新评分。

## 规模与容量

| 包类型 | 包数 | anchor总数 | 单包最大anchor数 | 单包最大catalog canonical字节 | 单包最大HTTP字节 |
|---|---:|---:|---:|---:|---:|
| episode | 24 | 2604 | 122 | 25,469 | 397,136 |
| focus | 12 | 786 | 86 | 18,882 | 387,887 |
| media | 4 | 520 | 142 | 29,036 | 31,764,148 |

最大HTTP为 media-03/deepseek：31,764,148字节，低于既有48 MiB边界（50,331,648字节）。最大catalog为 media-01：142项，canonical 29,036字节，保存的 catalog JSON 30,317字节，实际HTTP中的catalog文本块32,951字节。不同表示法的字节数不能混用。

使用原缓存 DeepSeek V4 tokenizer 在远端内存加载，仅传输压缩的完整文本，不传PNG、不下载/解压落盘。逐块计数覆盖80请求的完整 system 与每一段 user text，add_special_tokens=False。缓存ZIP与tokenizer JSON SHA均核对。

| 文本指标 | 实测/计算值 |
|---|---:|
| 80请求最大DeepSeek文本token | 198,055 |
| 单catalog最大文本token | 13,017 |
| 保留量：system/tools + output + margin | 32,768 + 32,768 + 8,192 = 73,728 |
| 最大文本+保留proxy | 271,783 |
| 既有input proxy / auto-compact / context | 360,000 / 450,000 / 512,000 |
| episode-01/deepseek 全文本 / catalog | 185,912 / 11,435 |
| episode-01/deepseek 文本+保留 / 图片 / 输出cap | 259,640 / 0 / 32,768 |

全部文本proxy比较通过。**未计图片token、provider chat framing/special tokens、SOL真实tokenizer、agent/tool/runtime峰值或provider实际usage。** SOL表中的数字只是同一DeepSeek tokenizer代理；不能把这些文本比较升级成多模态容量放行。

## 文件与可复查边界

- [机械回执](./receipt.json)：16,391项工程检查；425个输入文件SHA前后不变；逐包输入绑定和输出hash。
- [文本容量回执](./text_capacity.json) 与 [执行记录](./text_capacity_execution.json)：80请求逐块文本SHA/token，缓存tokenizer身份，43.69秒CPU执行，exit 0、stderr空、远端源写入0。
- [19地址表达](./diagnostic_address_representations.json)：原诊断地址、新anchor与局部pointer、归一化原地址、目标值hash。只读诊断用途。
- 每个 `packets/<packet>/` 包含新 `facts.manifest.json`、`review_protocol.json`、`comparison.json`；两评者的 `prepared/` 保存完整HTTP、指令、prepare、catalog与未放行release草稿。
- [核验脚本](./verify_reference_anchors.py)、[文本提取脚本](./measure_text_capacity.py)、[远端计数脚本](./count_text_remote.py)；原协议与原材料仍在E2原目录。

这次检查范围是facts输入载体。report阶段对自己封存facts及其原catalog的连续绑定、真实模型能否正确引用、语义判断和必要事实完整性均不由本次机械核验建立。旧19引用错误仍保留其原失败状态。

当前coverage SHA：`7548da8dcf1e43fad256ce993923589b90d00bcab8733ef6bc36177b3a28957f`。tokenizer ZIP SHA：`e7310d1dafe0a86d8a5629fe78a7c763760f651db9b8682718a1781dcd6fe495`；tokenizer JSON SHA：`89085f12ef79460ac5f66d1119325ddfc694b4ab209d80bbd81d35f081dc9614`。
