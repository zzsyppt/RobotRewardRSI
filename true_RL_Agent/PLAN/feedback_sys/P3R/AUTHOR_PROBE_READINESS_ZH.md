# R.0 作者探针准备核查

日期：2026-09-29。下表为当前实际结果，其后保留最初准备与独立 r2 授权过程。离线检查本身没有 API 调用、GPU 作业或训练；两轮真实探针的费用单独列示。

## 当前状态：r2 运行门通过，R.0-b 已启动

| 独立轮次 | 输出请求上限 | 首次完整响应 / 实际请求 | 直接候选 | 运行门 | 独立静态准入 | 按实际 usage 及登记价格计费 |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| r1 | 32768 | 6 / 8 | 6 | NOT_PASSED | 未在本索引中验收 | US$0.236300256 |
| r2 | 65536 | 8 / 8 | 8 | PASS | 8 / 8 PASS | US$0.201673776 |

共 **16 个实际请求，US$0.437974032**。r2 获选使用自己的 8/8 结果，不从两轮中为各包挑最好回答，不把 16 次合并成运行门成绩。r1 的两次失败分别是 `stair-01/A` 推理耗尽且 final 为空，以及 `stair-03/T` 的 rationale 不满足静态可解析字符串合同；原回执完整保留。

逐条 journal 复算费用与两份汇总一致。两轮响应均为 `deepseek-flash`、请求名 `deepseek-v4-flash`、effort `high`，16 条回执的 fingerprint 均为 `aeb56401ca74e127821c4f9126dcb669`。r1 输出量为 14,185–32,768，中位数 22,065.5；r2 为 13,697–29,166，中位数 22,974.5。**r2 每条实际输出都低于旧上限 32768，不能据这两轮宣称扩大上限导致成功。**本次 8/8 也不是长期可靠性的置信保证。现存汇总、启动记录和逐次回执未记权威请求时长，因此不使用解压后文件时间推算延迟。

结果 JSON 索引：

- [两轮核对与来源 hash](results/runtime_two_round_index.json)。
- [r1 实际汇总](results/checkpoint_evidence_r1/runtime_probe_r1/summary.json)。
- [r2 实际汇总](results/runtime_r2_evidence/runtime_probe_r2/summary.json)与[独立静态准入](results/runtime_r2_evidence/runtime_probe_r2_static.json)。运行汇总中的 `static_admission=NOT_EVALUATED` 保持原样，静态证据在独立文件；此次静态验收不包含 CPU smoke、语义或训练效果。

[阶段预算 r2](stage_budget_r2.json) 将未使用的辅助 API US$3 转入 R.0-b，R.0-b 分配为 US$15，阶段总额仍为 US$30，项目总额仍为 US$120。R.0-b 的 16 个首包已由根代理在服务器完成正式 preflight，32 episode 运行已启动；该状态来自根代理的运行核实，本索引不替代其后续实际回执。每 episode 仍为最多 5 attempts / 2 次源码请求，65536 输出下 160 次最坏费用 US$14.502912。R.4 为 `DEFERRED_NOT_RELEASED`，其余 US$12 不足以覆盖现设置下 48×5 次的最坏费用，须待先前门和实际账目闭合后重新核定，当前无调用放行。

## 最初已核实的运行前提（历史准备记录）

- 生产 `OpenAICompatibleProviderConfig` 支持 `reasoning_effort=high`、`max_tokens` 和 `empty_final_reasoning_retry`。旧 P3 脚本没有传递最后这个布尔参数；它只改变空 final 后的下一次请求，不能提高“首次响应完成率”。本次仅将原 `max` 改为 `high`，保留 32768 输出上限，关闭自动重试。
- R.0-a 的 8 包为 `stair-01`（conditional frontier）、`stair-03`（height OOD）、`push-02`（medium）、`push-04`（repeated），每包 A/T 各一份。已核对 P3 的全部 24 份首包原文件及 canonical user hash，再按固定种子 2026092901 排列这 8 份。不是按历史回答质量筛选。
- 冻结 P3 `source_r7` 上实际 `_system_prompt()` 与当前生产函数输出相同：SHA-256 `8a231efd7f369af55c3a156e55b8325f355c09dedc04fb9251006e4adca03b98`。远程 provider 文件 hash 也与 P3 `source_snapshot_r7.json` 一致。新的请求只改变运行参数；共有修复提示更新留给 R.0-b，不混入这 8 个原首包。
- 服务器 `/data/yuyangliu/secrets/deepseek.env` 已只读核实为可读；未输出内容、未验证余额或凭证。一次 SSH 握手 reset 后重试成功。正式执行仍需在可读取该文件的服务器环境中进行。
- [官方价格](https://api-docs.deepseek.com/quick_start/pricing/)于 2026-09-29 核实：Flash 峰价每百万输入 cache miss / hit 为 US$0.30 / 0.006，输出 US$1.20。旧请求名 `deepseek-v4-flash` 仍被接受并映射到 `deepseek-flash`；保留原请求名，并记录实际响应模型与 fingerprint。
- 按每请求 40000 输入预留、32768 输出上限，8 次最坏峰价为 **US$0.4105728**，不等于账单。阶段上限 US$3 不自动允许再试其它设置或增加 8 个机会。

## 工具与冻结入口

[author_runtime_probe.py](author_runtime_probe.py) 仅实现 R.0-a。复用生产 provider、`RoundAuthoringJournal`、`CostGuardedProvider` 和序列化；不复制作者循环、不查询新源码、不修复、不训练。

```powershell
.venv\Scripts\python.exe PLAN\feedback_sys\P3R\author_runtime_probe.py prepare --output PLAN\feedback_sys\P3R\runtime_probe_r1
.venv\Scripts\python.exe PLAN\feedback_sys\P3R\author_runtime_probe.py inspect --stage PLAN\feedback_sys\P3R\runtime_probe_r1\stage.json --output <真实运行目录>
```

`prepare` 已执行，原目录不得覆盖。准备卡为 [runtime_probe_r1/stage.json](runtime_probe_r1/stage.json)，8 份 user payload、完整 request hash、system hash、请求大小、设置、价格及顺序都已登记。最初状态为 `PREPARED_NOT_RELEASED`；后续根代理核对阶段资源分配与实施卡，设为 `FROZEN_API_RELEASED` 并填入 `resource_allocation_ref` 后执行了上表真实运行。冻结后的实际请求必须逐字节匹配；公共提示或源码变动导致 request hash 改变时不能静默续跑。

`run --stage <准备卡> --output <新的运行目录>` 使用原 journal。已有尝试（包括未知回执）不重发；服务拒绝、未知 usage、输入超预留或模型异常会保留记录并停止。实际输出超请求上限按真实 usage 全额计费并记录偏差，不截断费用。正常生成失败继续消耗该一次机会，不找替补输入。

完整响应定义为：非空 final、未因 length 截断、满足生产 `candidate_draft` 或 `source_request` 合同。合法源码请求单独计数，也算一次完整响应；不算候选已写完。8 包的 90% 门在整数上等于 8/8。候选物化、完整响应、静态准入、语义正确和训练效果互不替代。

离线 fake transport 已跑通生产 journal/预算链的 8 次合法 `source_request`，再次运行没有重发，结果为接线 PASS。未调用真实 provider。当前工具的 `static_admission=NOT_EVALUATED` 明示独立静态验收待补；不能据运行门作候选质量结论。8 个历史丢反馈末帧的重建由 [rebuild_repair_frames.py](rebuild_repair_frames.py) 独立负责，不修改这里的旧首包。

## R.0-b 最小复用边界（最初 32768 预算准备）

- P3 `author_experiment.py` 的 `reference_context`、任务 composition、真实 `PublicContextBuilder`、`FrozenFeedbackAdapter` 和生产 `CandidateAuthoringLoop` 可复用；不直接重跑整个 P3 schedule。旧 `build_case` 会验证 A/T 旧投影并读取当前语义，不能拿它重建 R.0-a 的冻结首包，也不能不加区分套在四个新变体上。
- 新实验只需一个 4 案例数据文件，分别绑定 A、T0、Tc-num、Tc-vis 的明确证据视图；不把 10 条人工评语或四个案例的特殊规则写进公共 provider。Tc-num 的每条测量及 Tc-vis 的每条视觉观察需有来源、单位/时窗与可见性，不能夹奖励建议、评阅答案或未经证实的接触位置。
- Tc-vis 是额外的带来源视觉观察条件，不能说它与 Tc-num 同模态，也不能称“作者看了图像”除非实际发图。人工整理文字可先测语义上限，真实视觉传输是另一项验收。
- 32 个 episode（4 案例 × 4 变体 × 2）继续复用原最多 5 attempts、2 次源码请求、48 KiB 源码返回限制。按当前 40k/32768 峰价上限，160 个请求最多 US$8.211456，含 40 个 episode 的绝对机会上限则为 US$10.26432；均只在当前报价与输入预留成立时有效，真实回执超量全额计账。
- 正式启动前还需：R.0-a 实测结果、四变体内容冻结与来源核查、共同修复底座与代码快照冻结、model_review 两项判分口径及预算分配。没有这些资料时准备脚本不意味着可以执行 R.0-b。

## r2：用户批准的独立 65536 输出验证

用户随后明确批准相同模型、相同 8 包和顺序，只把输出上限改为 65536，再独立验证一次。r1 脚本原字节仍在 `runtime_probe_r1/source_upload.zip`，旧结果不重写；r2 准备卡为 [runtime_probe_r2/stage.json](runtime_probe_r2/stage.json)。两版实际 request hash 均已重现，唯一请求差异确为 `max_tokens`，见 [逐包等价检查](results/runtime_r2_request_equivalence.json)。

根代理核实 r1 实测 US$0.236300256，因此 r2 从原 US$3 分配中最多使用 US$2.763699744；8 次最坏峰价 US$0.7251456。当前脚本从每张卡自己的预算读取输出上限、未知预留与超量检查，默认准备值仍为 32768。旧卡检查和新卡“完整响应后未知用量”的关键离线核验通过，见 [检查记录](results/runtime_r2_stage_budget_offline_check.json)。这不构成 r2 真实运行结果。

若 r2 通过并获选，R.0-b 与 R.4 必须重新登记同源输出设置及预算。65536 时 32×5 次的最坏峰价为 US$14.502912，超过原 R.0-b US$12；48×5 则为 US$21.754368。不得保留旧 32768 未知预留，也不得悄悄改变尝试次数或放宽阶段额度。
