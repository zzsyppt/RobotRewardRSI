# 真实数值 facts/report 序列化容量：旧开发工程验证

**NOT_FROZEN；不是正式评阅；未读新 N5 材料；0 模型/API/GPU 调用。当前 report 容量未通过，不可放行。**

本目录的 144 份请求载荷（36 单元 × facts/report × 两名评者）均由现有 `run_confirmation_review.py` 的实际序列化函数生成。每阶段显式配置 32,768 输出。report 使用 `synthetic_only=true, complete=false` 的纯占位文本压力夹具，约 32,900 token，既不含事实答案也不生成伪造的 SEALED 文件。对应卡为 `CAPACITY_ONLY_NOT_EXECUTABLE`，即使手工改 release，runner 也在读取密钥和调用 transport 前拒绝执行。

## 实际材料闭合

- 24 个 episode report：保留该回合全部相关原量曲线、完整原始 full 回合对象的所有字段及共享定义；未用有界报告替代 full。
- 12 个 focus report：保留整份任务默认报告、原生产序列化 author request、全任务 pool 的固定计数/匹配/采样操作数、所有默认报告代表例与成功反例的原量，以及有关 full 组事实。支持率的材料范围是全任务，不能按当前 focus 过滤默认报告声明。
- facts 仍各自供应完整相关回合轨迹或全 focus 原始分母；report 补充原量只服务逐句验证，不能回写必要事实清单。所有 facts 包在读取任何自动报告之前生成。
- 默认报告原样保留。旧 author request 标签明确是 `delivery_offline_r1` 的离线生产序列化、未实际发送；其原字节逐任务一致。phase/full 引用哈希及原数值 archive 绑定逐任务一致。
- 24 份 full 回合切片逐对象等值；全部 4 个报告引用回合均覆盖，32,604 个所供原值按 dtype 还原后与原始 NPZ 字节一致；报告计数层保留完整 864+192=1,056 成员。Stair 首次确认和 Push 条件计数均从原始区间/条件重算核对。

## 容量结果

使用[官方 DeepSeek tokenizer](https://api-docs.deepseek.com/quick_start/token_usage/)对保存的完整请求文本计数；不是实际 usage，**不是 SOL 的精确 tokenizer**。本机 SOL 的 272,000 × 95%=258,400 容量还须容纳系统与工具引导。按 root 指定的预检，输入目标 ≤180,000，另留 32,768 系统/工具、32,768 输出；本账另外保留 8,192 余量。此前仅用 1,024 JSON/chat 封装余量的测量不能证明本机 SOL 可执行。

| DeepSeek 载荷代理计数 | facts 最大 | report 最大，含 synthetic 自 facts |
|---|---:|---:|
| Stair episode | 162,001 | 218,667 |
| Stair focus | 149,228 | 225,739 |
| Push episode | 37,342 | 78,134 |
| Push focus | 149,084 | 215,681 |

两种 carrier 计数略有差异，最大 facts=162,007，最大 report=225,756。13 个逻辑 report 单元（Stair 最长 episode-02 与全部 12 个 focus）超过 180k，共 26 份双方 carrier；其余 episode report 最大为 episode-04 的 176,206。实际 HTTP body 最大 583,631 B，数值包的 48 MiB HTTP 限制不是当前阻碍。

主要占用（DeepSeek）：

| report | 原量 | full 切片/组事实 | 默认报告+原请求 | 自 facts 夹具 |
|---|---:|---:|---:|---:|
| 最长 Stair episode | 156,628 | 23,762 | 不额外重复任务报告 | 32,900 |
| Stair focus | 119,933 | 43,883 | 23,665 | 约 32,900 |
| Push focus | 103,569 | 52,526 | 21,325 | 约 32,900 |

协议、问题与 system 文本另计，完整逐包/逐角色账在 `tokenizer_check.json`。没有为满足容量删时点或事实。root 尚须选择并预登记最小固定分包；候选是 focus 按全 pool 统计声明与代表例/反例过程声明分两份，最长 episode 按 root/foot 事实类分两份、各自保留全时间。必须对声明归属做并集/无重计核对；此处**尚未实施分包或合并**，不建立递归摘要平台。

媒体尚未供应。本目录只有数值，不声称已检查媒体评者容量。最小接口是既有 runner 的 JSON/原 PNG materials 清单：facts 只含实际 observer 所见全部原图、step/view/alias/相机信息及同回合数值；report 再加 observer 输出、full 视觉条目、默认报告和实际/离线请求。媒体单元归属与必要事实分母由 root 冻结，不能由打包器自行改变。

## 复现

```powershell
.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/prepare_confirmation_materials.py --raw-export PLAN/feedback_sys/P3NEW/results/raw_export_development_related_v1 --output <new-output-directory> --profile related_v1 --context-tokens 258400 --max-output-tokens 32768 --report-capacity-root PLAN/feedback_sys/P3NEW/results/delivery_offline_r1
```

`packing_receipt.json`、`report_capacity_receipt.json`、`verification.json`、`tokenizer_check.json` 分别保存 facts 来源、report 构建、原值/跨 scope 验证与容量。假 transport CPU 测试证明：预算使用配置的 32,768 输出，40,000 completion 被判超过上限；synthetic report 在密钥读取前被拒。未生成任何模型答案或语义分数。
