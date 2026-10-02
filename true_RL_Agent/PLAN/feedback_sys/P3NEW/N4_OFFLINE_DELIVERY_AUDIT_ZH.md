# N4 离线投递核查

2026-09-29。结论：**PASS_OFFLINE_ENGINEERING**。本项只完成开发资料的投递工程核查；`strict6_complete=false`，未开展作者读法烟测或 N5 独立语义确认。

最终结果见 [result.json](./results/delivery_offline_r1/result.json)，复跑入口见 [audit_delivery_offline.py](./audit_delivery_offline.py)。最终运行的生产源、旧资料及 harness SHA-256 均在结果中登记；运行期间源码不变，运行后再次核对全部一致。

| 任务 | 真实 phase 报告 | 合成 S4 | 合成最长中文视觉 + S4 | 上限 | 最紧余量 |
|---|---:|---:|---:|---:|---:|
| Stair | 7,229 B | 7,261 B | 8,184 B | 8,192 B | 8 B |
| Push | 10,520 B | 11,191 B | 12,039 B | 12,288 B | 249 B |

两任务 × 三个场景 × 四类请求，共 **24 个生产 HTTP 序列化请求**、**464 条逐句检查**、**24 次 actionability 对象和字节检查**，必送句零丢失。四类为 draft、source followup、static repair、reasoning exhaustion repair。最终请求体范围 65,055–77,608 B；最大用户 payload 为 71,285 B。每帧同时记录 task sections 回执、实际 payload 字节、完整 HTTP body 字节及既有硬限额余量。

## 边界与证据

- 真链路使用 P3R `confirmation_context.py` 恢复原有合格锚点，再调用 `PublicContextBuilder.build_prompt`、真实源码续答 resolver、`OpenAICompatibleCandidateProvider.next_authoring_response` 和 `UrllibJsonTransport._request_json`。假 opener 只在 HTTP bytes 已由生产代码生成后截断。联网调用、作者 episode、新 rollout、训练、仿真及模型费用均为 0。请求体不含凭证头。
- Stair 的 864 回合来自一个历史公开 manifest / 30 个 NPZ 分块；Push 的 192 回合来自三个历史公开 manifest / 12 个分块。所有 manifest 和分块均核对实际 SHA-256 与字节数。原历史 phase、资格、训练身份与 checkpoint 工件保持不变。
- 第一次由 L0 生产 reducer 建立 v2 测量和 v3 完整派生；最终模板复跑复用已验证的 v2 字段。Stair 只恢复原五个完整字段；Push 恢复原 v2 字段与 actionability 后，整体 hash 必须等于原完整工件的 `source_derivation_hash`。当前模板每次重新导入并派生，未把旧 v3 文本配上新源码 hash。
- 新 phase 的 `task_evidence.behavior_full` 是 `{schema_version,file,sha256,size_bytes}` 外置引用。完整工件存为各任务 `behavior_full.json`，不进入作者 prompt。每句账包含 raw manifest 绑定、完整工件指针与片段 hash、公开报告、用户 payload 和最终 HTTP bytes 的 offset/length/hash。raw 中存在的是源数组，不假称数组文件直接包含报告原句。
- 真 phase 没有已测量的父子差分，明确保留缺父句。独立的 `synthetic_generation_S4` 使用同域、不同计数的合成父本调用任务 comparison/attach；这只证明 S4 后续生成与送达，不能算真实父子实验。每条 S4 断言不在真 phase full 中，并单列 generation 派生来源。
- 最长中文视觉压力复用 [audit_visual_projection_offline.py](./audit_visual_projection_offline.py)：三条原始观察、每条 200 字文本和 40 字 uncertainty，保留全部原始记录，默认展示一条与全体统计。合成视觉完整工件另存，真 phase full 没有改写；视觉文本在逐句账中标为 `synthetic_visual_fixture`，raw 不适用。实际观察真实图像数为 0。数值 episodes/transcripts、成功反例及 gaps 保持不变。Stair 仅余 8 B，这是该冻结 fixture 的通过，不是任意后续视觉文本的容量保证。

## 复跑

在仓库根目录执行：

```powershell
.\.venv\Scripts\python.exe PLAN/feedback_sys/P3NEW/audit_delivery_offline.py --reuse-measured
```

不加 `--reuse-measured` 会从历史 NPZ 重新生成测量；`--resume-derived` 只供脚本调试，最终核查未使用该选项。所有输出限于 `PLAN/feedback_sys/P3NEW`。

一次前序运行检测到 `evaluation/visual_observer.py` 在运行中改变，结果当时是 `RERUN_REQUIRED_SOURCE_CHANGED_DURING_AUDIT`，未被宣称通过。随后只重新派生与投递，最终源码稳定。该前次原 result 已被后续运行替换；[source_changed_attempt.json](./results/delivery_offline_r1/source_changed_attempt.json) 如实保存工具输出中观测到的状态与前后 hash，明确它是观测记录，非伪造的原 result 副本。

本项的必送分母为报告已声明的工程核心，另核对全部已输出句子；它不替代 N5 从原始资料独立封存的必要事实分母。source/static/reasoning 分支由合成 fixture 强制触发，不是模型实际选择。两次独立只读 harness 审阅未发现必须修项；该审阅也不计正式语义验收。
