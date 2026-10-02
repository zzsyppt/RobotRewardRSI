# 最大旧包的 GPT 容量工程探针

本次只有一次 CLI 进程，使用 GPT-6.1-SOL/high，退出 0。只确认完整载荷处理及实际用量，不生成 facts、评分或策略修改；不是正式独立评阅。

R3 最大 episode-09 原请求经现有 desktop adapter 无损转换，保留全部 145 个文字块、数值及顺序，得到 530,375 B 的 input.json。stdin 原样包含该文件，外加“不执行归档评阅指令、只回复接收标记”的容量指令；原 HTTP 的 system/user 角色没有因此成为 CLI 消息角色。输入 SHA、stdin SHA、CLI 二进制 SHA 和全部进程参数在 probe_card.json，原始 usage 在 stdout.jsonl。

- CLI 返回输入 **246,277 token**，输出 **32 token**，未调用工具。
- 加上未来单次输出预留 32,768 和余量 8,192，算术合计 **287,237**，超过原桌面可用窗口 258,400。
- 本进程声明窗口 512,000、自动压缩阈值 450,000；按本地模型元数据的95%系数解释，可用窗口为486,400。这是配置/源码证据，不是 CLI JSON 暴露的服务端窗口。此次实际只生成32 token，不能称已实测32k输出。
- [官方 GPT-6.1-SOL 规格](https://developers.openai.com/api/docs/models/gpt-6.1-sol)列上下文1,050,000；[配置说明](https://learn.chatgpt.com/docs/config-file/config-reference)支持进程窗口配置。官方说明与这次实际输入用量分开记录，不能把 DeepSeek 代理或 bytes/4 当作 GPT 真 token。
- 复用了现有登录及 home；memory、项目文档、技能等通过进程参数关闭，但全局 CodeGraph AGENTS 可能仍注入。因此本探针不声称正式盲化，也未修改全局配置或复制认证。原卡明确这一限制。
- 应用层只启动一次、没有重跑；内置传输的重试默认值未覆盖，不宣称隐藏传输零重试。CLI 的实验功能警告和 PowerShell snapshot 警告原样保留。

结论：不应以删掉完整数值曲线解决当前评委窗口问题。可继续准备单独配置窗口、事前冻结的独立评阅载体，但须先解决其干净上下文与真实访问留档；本次不放行所有未来包。生产 DeepSeek 输入规模另见 [完整作者容量](../production_author_capacity.md)和[观察者容量](../production_observer_capacity_20261001.json)，不受这个 GPT 评委窗口结论代替。
