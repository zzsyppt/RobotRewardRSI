# P5 开发日志

## 2026-08-18

### 入场与范围

- 重读主计划、P0–P4 交接和当前代码，冻结目标为：同一 generation session 内可审计续行、故障不掩盖且不重置成本、ledger 派生探索索引、高价值 loser 有界重访。
- 采用 Codex/成熟 workflow 中“稳定幂等键、外部副作用 receipt、append-only event、派生 view”的适用思想；明确不引入通用 Agent runtime、动态 DAG、Ray/Temporal/Graph DB 或第二套 event store。
- 只读复核 `D:\humanoid_agent\codex` 的 `56b82e67`：借鉴 unified-exec 对明确 process state、序号缺口检测、取消/终止与输出闭合的分层；没有复制其通用 thread/event/runtime 体系。
- 开发基线：`96e03abb792ac6525cf4f562ec2e2a8b27a60773`；保留工作树中所有无关 outputs、临时包和用户产物，不做清理、移动或批量 stage。

### Lifecycle 与 attempt

- 新增 generation continuation policy/identity/segment journal/audit/cursor/terminal binding。
- 为 controlled exit 和 hard crash 建立不同 receipt；unclean recovery 在 lease/PID/ledger/cost 审计后结算未闭合 reservation，且 result 绑定 recovery receipt hash。
- 扩展 P1 attempt journal，使下一 segment 可从同一 reservation authority 恢复并消费唯一 retry。
- 审计发现窄崩溃窗口：runner 已完成并发布 attempt result，但 P2/P3 receipt 尚未发布。新增 completed output 只读重验路径，禁止重新启动已完成 job。

### Exploration 与 verifier

- 新增 ledger-derived evidence card/index/frontier/lane intent/prompt projection；Stair observed-effect 与价值规则留在 task-owned projector。
- 将 generation parent、lane intent、guard reference 加入公开 allowlist 投影，为卡片重建提供单一事实来源。
- experiment memory 增加可选 generation exploration 投影，同时保持 width=1 默认路径不变和14 KiB硬上限。
- 初次审计发现“负向 exact key”仍会被 parent/source/name 绕过。改为复用 verifier single-component operation delta，派生跨 parent/改名/参数归一化的 equivalence key。
- 再审计发现 bounded prompt 负向卡不能代表完整阻断集合。将完整 exclusion set hash 绑定到 lane intent/authoring gate，prompt 只作为有限可读摘要。
- 再审计发现 `verified_not_run` 会错误消费 source card，accepted-equivalent loser 仍可入队。改为只有重访产生 public evidence 后才消费 source 所属的 verifier-equivalence class，并对 incorporated/duplicate equivalence 生成明确 ineligible reason。
- 将新 schema 的处置名从误导性的 `suppress_exact` 收敛为 `suppress_equivalent`；exact implementation key 只保留审计用途。
- 用户审计后，以当前生产 projector 亲自重放真实15轮：缺失 parent public-dev guard reference 的 R2 从错误 high 收敛为 unknown；最终为 high=1、unknown=1、low=13。确认14个等价类中只有 R1/R8 重复，同一位置/进度输入签名有6个结构变体。
- 保留严格 high 判定，不用稀疏 width=1 历史反推 width=2 的精确重访概率。自动重访改为 priority-only；调度以 verifier-equivalence class 服务次数、head transition距离、task-owned公开增益和新近度排序，同一等价类最多2次、任一等价卡取得public evidence即消费整类，超过1次head推进过期。
- 最终对抗复审发现 frontier 只校 `index_hash` 可被伪造为空，以及重复等价卡可轮换绕过消费/次数上限。新增 canonical frontier/lane-intent 重算校验，并把服务与消费统一到 verifier-equivalence class；没有增加第二套身份规则。
- 完整 verifier exclusion set 继续唯一阻断；prompt 增加最多2张 recent verifier-equivalent avoidance 和2条 current-head repeated-negative input-signature advisory。advisory 非阻断且不创建机制家族身份。
- frontier 容量外卡片新增逐卡 deferred reason/queue position，避免只剩 backlog_count。
- lifecycle start/recovery schema 绑定 process-incarnation token，消除 PID 复用导致 session 永久卡死的编号歧义；Linux 使用 boot ID + procfs starttime，Windows 使用进程创建时间。
- 最终字节边界复审复现合法大卡使 exploration prompt 无限自旋，并发现12 KiB卡片、6 KiB prompt、2 KiB lane用途混淆。源头改为：完整卡留在索引；prompt统一由一个有限且每步严格变化的收敛器生成，无法表达时整卡省略并记账；lane复用generation contract唯一2 KiB门，只携带有界preview和summary hash/省略标记。80字段合法卡与4 KiB内大summary均加入最小回归，不增加任务语义或通用安全框架。
- 真实15轮收益表述同步纠偏：唯一high为accepted/incorporated的R3，priority loser实际为0。R1/R8只证明“在同一generation session内可于训练前阻断”的反事实能力，不冒充P5已实际节省；同输入advisory与high-loser重访的运行收益都留待P6观测，不为提高触发率放宽high。

### 验收

- Runner/P1、exploration、lifecycle、authoring 定向组合：`Ran 130 tests; OK (skipped=1)`。
- P4 generation/P3 phase/P2 width=1/Stair/Push 相关组合通过。
- 全量最终复跑：`Ran 1256 tests in 216.915s; OK (skipped=21)`。
- `python -m compileall -q src tests` 通过。
- `git diff --check` 通过。
- 全部P5生产源码与测试文件的SHA-256冻结在`P5_SOURCE_TEST_SHA256.txt`；P5文档不纳入以避免自引用。
- 未调用 provider、GPU、服务器；未启动 P6 或任何正式实验。
- Windows 本机无法替代 Linux killpg 分支；新增 Linux-only 非 GPU 子进程用例并明确留作 P6-D 前服务器/CI 强制门，不把本地 skip 写成通过。

### 交接

- P6 只能在用户验收后开始。
- P6 需要做真实双卡资源/墙钟/RAM/IO/磁盘测量、正式成本合同冻结、generation CLI/runbook 与 generation-aware final-confirmation 接线。
- continuation 只限同一 immutable generation session；跨完整15-generation实验的重试不在本版范围。
