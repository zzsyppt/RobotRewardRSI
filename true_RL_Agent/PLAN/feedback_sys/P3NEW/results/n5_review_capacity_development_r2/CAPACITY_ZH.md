# 旧开发材料：数值两阶段容量收尾

状态：**NOT_FROZEN；离线工程准备；0 次 API；没有新 N5 数据或语义评分。**

本轮仅改 `prepare_confirmation_materials.py`，保留历史 r1 输出。原始资料来自 `raw_export_development_related_v1`；默认报告与作者请求来自同原始数值池的 `delivery_offline_r1`。请求是生产格式的离线序列化，不是历史真实发送或模型 usage。

## 保留与整理的边界

- 24 个回合保留完整 1..T 的相关原始 root/foot 位置、速度和姿态数据；Stair 四元数与直接 roll/pitch/up-z 均保留。所有投递原始浮点按原 dtype 逐值往返；直接 float64 不降精度。
- 仅 surface height 和两种 forbidden force norm 在更短时使用直观 `[首步,末步,同一原值]` 区间。没有时间抽样；瞬时 support force norm 与历史缓冲最大范数仍分开保留。
- focus 的全池成员表只把重复 run/scope/cell/dt 属性集中列出，仍保留 1,056 个成员与全部操作数。完整成员表可以逐行重建，原行序与操作数对应不变。
- focus 完整派生附件仅留必要共享定义。此前额外附上的非引用聚合派生行不属于新的评分对象；完整派生轴仍是 24 个选定回合的完整记录，各回合 full 行没有删字段。
- 支持率轴仍覆盖完整默认报告及最终请求中的全部相关声明，不以 focus scope 过滤。默认报告对象已完整存在于同包的原作者请求中；现在用双 locator 和 hash 指向该对象，删除的只是第二份相同文本。两份原请求字节完全不变，完整报告另留档。
- 每个 focus report 继续提供全池计数/分组操作数和全部被引用代表例、反例的整回合相关原量。facts 必要事实的分母不会根据报告回写。

## 实际容量

用官方 DeepSeek v4 tokenizer 对每个完整请求的所有实际文本块计数。下表为 DeepSeek carrier，另一家使用相同材料；该 tokenizer 对本地 SOL 仅是代理估计，不能冒充 SOL 精确分词或实测 usage。

|阶段|任务/逻辑单位|最小输入 token|最大输入 token|最大 HTTP 字节|
|---|---|---:|---:|---:|
|facts|Stair episode|47,705|148,487|308,319|
|facts|Stair focus|147,804|149,228|308,528|
|facts|Push episode|16,941|37,385|88,874|
|facts|Push focus|43,133|149,084|310,199|
|report|Stair episode|88,272|205,153|447,877|
|report|Stair focus|162,507|162,511|412,927|
|report|Push episode|54,622|78,175|176,404|
|report|Push focus|160,177|160,190|387,358|

36 个数值逻辑单位 × 2 阶段 × 2 carrier，共 144 个真实序列化文件，均未发送。facts 最大为 SOL carrier 149,234；report 最大为 SOL carrier 205,160。两阶段输出均预留 32,768，report 另实际包含约 32,900 token 的 synthetic own-facts 压力文本；这不是虚构事实答案或封存结果。

本地 SOL 的约束按 `272000 × 95% = 258400`，另预留 system/tools 32,768、输出 32,768、额外余量 8,192；文本代理目标为 180,000。**仅 `episode-02`（Stair，525 步）报告仍不满足：205,160 比代理目标多 25,160，总预留 278,888 比有效上下文多 20,488。**其余 35 个逻辑单位的 facts/report 文本代理均进入目标。HTTP 48 MiB 限制不是这些数值包的阻碍。

最长报告的组成（DeepSeek carrier）：system 142、协议 4,857、问题 378、完整相关原量 143,114、完整本回合 full 23,762、自身 facts 压力文本 32,900，共 205,153。相对于本回合 facts 的新增为 56,666，主要是 full 与 own-facts；不是 81k。

这轮常量区间将最长回合表面高度 10,179→792 token，两种全零 forbidden norm 各 2,124→39。support 瞬时/历史范数用区间反而更大，因此保持 CSV。替换成原始三维力也更大：support 原力 31,880 对标范数 13,154；原历史力 95,320 对标历史最大范数 13,331。未继续尝试编码或跨片系统。

## 已核对

`verification.json` 记录：

- 投递的 163,165 个原始浮点值和 64,005 个直接 float64 值均展开到完全相同 dtype 字节；所有离散字段同源相等。
- 24 个 full 回合行逐字段与 source JSON pointer 完全相等。
- 1,056 个池成员及其操作数、hash 顺序与整理前一致；引用例原量不变。
- 两任务完整作者请求原字节不变；报告 locator 取回对象与 phase 中完整默认报告完全相等。
- 144 个请求与 prepare 卡 hash/字节数一致；report 压力卡全部 `CAPACITY_ONLY_NOT_EXECUTABLE`，facts 卡全部 `PREPARED_NOT_RELEASED`。
- exporter 7 项 CPU self-test 通过；本轮没有修改生产代码。

## 当前可执行材料与停点

facts manifests、完整相关原量、report manifests、报告定位/原量验证材料以及 144 个离线请求均已存在；不是“未来 driver 再接”的空接口。真实 report 仍须用对应评者本人封存的 facts，通过已有 `run_confirmation_review.py prepare --own-facts-seal ...` 重新生成；当前 synthetic 压力卡不能发布或调用。

本轮在此停止。根代理需决定唯一超限长回合的固定资料方案，并处理独立 4 个媒体逻辑单位的两阶段容量；本目录不包含媒体容量，也不据此宣布 N5 可放行。协议/采集卡仍未冻结；诊断模块开发验收与正式独立 N5 保持区分。

复现命令（输出目录必须不存在）：

```text
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/prepare_confirmation_materials.py --raw-export PLAN/feedback_sys/P3NEW/results/raw_export_development_related_v1 --output <新输出目录> --profile related_v1 --context-tokens 258400 --max-output-tokens 32768 --report-capacity-root PLAN/feedback_sys/P3NEW/results/delivery_offline_r1
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/export_confirmation_raw.py --self-test
```

源码 SHA-256：packer `b43dcdb9d518f5150045457fcd9e64f4d3fd2fff4135129a3869aa7b93dd9e38`；exporter `eda4edb84dc15fe52bcb5607bc2adfba448e0e1d32eb6172bfc121f675757592`；runner `c09ef80569d1649a42c397285e12c91a9dabd8bdff9ab87f6c902f733bd5d038`。精确请求 hash、分块 token、原值验证与边界在邻接 JSON 中。
