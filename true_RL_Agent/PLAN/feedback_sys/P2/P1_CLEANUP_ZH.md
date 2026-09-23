# P2 开始前的两项 P1 小修（2026-09-23）

起始提交为 `b24ddfe9e54c7cc93318a4c6a2675eb9fb799d20`。本轮只修改 `candidates/mechanism.py`、`agent_core/experiment_memory.py` 及直接相关的 `test_runtime_evidence_projection.py`，没有重写 P1 的历史验收、冻结 payload 或原日志。先查询 CodeGraph；它多次返回相邻符号而未覆盖目标函数，随后针对缺失的具体函数读取当前源码与 Git 历史。

## 1. 旧 width-one 零填充存在真实歧义，已修

`fe96455d` 的 `RuntimeActivationEvidence` 默认 `reward_term_calls=0`、`diagnostic_calls=0`、`nonzero_reward_contribution=False`，`to_public_dict()` 会输出这些默认值；当时没有 `step_calls`、`reward_term_count`。因此旧记录中的调用数 0 不能单独证明“存在 reward term、adapter 已执行、但 term 没被调用”。

修复后，只有同时满足 `step_calls > 0`、`reward_term_count > 0`、`reward_term_calls == 0` 才返回 `terms_not_called`。若旧记录仅有零填充计数，或缺少上述任一正值证据，则返回 `unknown`。显式的 `step_calls == 0` 仍是 `not_executed`；显式的 term 数 0 仍是 `no_reward_terms`；正调用数及贡献证据的现有状态保持不变。

新反例覆盖：缺 term 数、缺 step 数、只有零调用数、完整旧版零填充 width-one 输出。其中前三项穿过逐 lane 记忆路径，最后一项穿过真实 `PublicContextBuilder` 的 round 路径，并验证原零计数仍保留、缺失的新字段仍为 null，只有记忆解释改为 `unknown`。修改生产源码前，这四项均出现 `terms_not_called != unknown`；修改后均通过。

同类审计中，`context._candidate_runtime_activated_in_training` 对只有旧零计数的记录已返回 `unknown`，没有同样的“未调用”推断；`MechanismEvidenceGate` 对零或缺调用数返回 `attribution_unclear`，也没有作这种推断。源码中没有其他 `terms_not_called` 发射点，故修复留在状态解释的所有权层。

## 2. 删除审计中间字典格式的迁移特例

“旧字典从未持久化或提交”不是准确事实：以下三个 P1 审计输出已经保存在仓库，并随 `b24ddfe9` 提交；它们的第 10 列分别含 10、12、14 个字典。这些文件作为历史证据原样保留。

| 已保存审计输出（均位于 `P1/audit_replay/`） | SHA-256 |
| --- | --- |
| `payload_overlay_push_before_g06.json` | `e6aa841083dea160e6089864b1335bee7908ffa77080ce1e02e602897a5080fc` |
| `payload_overlay_stair_before_g13.json` | `550773157e707a1140e4c81a68131968719bb142cb0b8aa0e5788d6685264c56` |
| `payload_overlay_stair_before_g16.json` | `1f063373e537f555555cf378d067aed1807d8ca1f111083f1cc59beacc7932c1` |

删除兼容分支的依据是其**没有生产格式迁移需求**：可达 Git 历史中，`experiment_memory.py` 的 `runtime_observation` 首次在 `b24ddfe9` 出现，该提交的生产发射器已经使用短字符串；没有查到发射字典列的已提交生产版本。本地 `outputs/`、`PLAN/2new_world/` 的 JSON 搜索也没有查到这种字段。`build_public_history_section` 对由记录重建的公开记忆做再次压缩，并未从上述审计输出恢复生产状态。这个判断限于已审查的提交历史、本地产物和消费路径，不声称所有远端文件均被穷举。

因此删除 `_runtime_observation_state` 和它的 dict / `status: available` 特例：普通 round 顶层仅保留短字符串；generation 的实际旧 9 列行追加 `not_available`，当前第 10 列短字符串原样保留。未认可的非字符串不再作为已支持的迁移格式解析。新增正常回归测试锁定旧 9 列、当前 10 列、普通 round 短状态与再次压缩幂等，删除了专门维持审计中间字典格式的手写测试。

## 3. 检查结果与边界

- [修复前日志](./p1_cleanup_before.log)：7 tests，4 个反例失败，证明问题可复现。
- [本地环境受限日志](./p1_cleanup_local_torch_missing.log)：bundled Python 跑 120 tests，119 通过；唯一 ERROR 为既有 reward peak frame 测试导入不到 `torch`，没有断言失败。
- [最终检查日志](./p1_cleanup_after.log)：使用项目 `.venv/Scripts/python.exe`、`torch 2.11.0+cpu`、`CUDA_VISIBLE_DEVICES` 为空，六个相关模块 **120 tests 全部通过，5.352 秒，exit 0**；包括运行测量投影、冻结真实记忆 fixture、历史记忆容量、机制 gate、Push runtime adapter 与 exploration frontier。
- `git diff --check` 对上述三份变更文件通过。原 14 KiB 预算与保留门槛未调整。此次没有重跑 P1 服务器真实会话，也没有把旧回放宣称为本轮新回放；已有真实记忆 fixture 通过本次源码检查。

未启动 GPU、模拟器、训练或模型调用。曾为缺 Torch 准备并传输一份 58,881 字节临时测试包，确认项目本地 CPU 环境可用后没有执行远端测试，也没有创建 P2 远端 scratch；仅删除了这份本轮未使用的传输包。原 P1 远端隔离目录未修改。

本轮受测文件指纹：

| 文件 | SHA-256 |
| --- | --- |
| `src/true_rl_agent/candidates/mechanism.py` | `246e9941caa285d6b6f9f11cf1bdf381e145f4bd06b5d474479c7f1621ae3475` |
| `src/true_rl_agent/agent_core/experiment_memory.py` | `00f55007a60ca5e5c1172cf42460444cda59f0bc1b6eb4cbbd3c629274ec98a9` |
| `tests/unit/test_runtime_evidence_projection.py` | `d889bc9a24f60dffe6a0d9d90b0777c4284ff19e60dbad188c6ee5832d17d30c` |
