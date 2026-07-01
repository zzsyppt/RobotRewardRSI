# Phase 8E-6-3 provider 到 candidate package 自动落盘

日期：2026-06-27

## 目标

实现 Phase 8E 中的 provider ingestion 环节：让 LLM/provider 输出能被落成真实 code-level candidate package，并在进入训练前先通过本地包约束与 Phase 8D verifier。

本阶段不启动 formal training、trusted eval、decision 或 ledger inheritance，避免把 8E-6-3 误写成完整自演进闭环。

## 设计结论

8E-6-3 的合理边界是：

1. runner/orchestrator 层构造 public-only prompt；
2. provider 层可以联网、可以读取本层凭据；
3. provider 输出必须是结构化 JSON 文件包；
4. 文件包只允许写入 staging candidate dir；
5. 写入前检查路径逃逸、必备 manifest/training_code、public-only 禁词与已知敏感值；
6. 写入后再交给 Phase 8D verifier；
7. 训练/eval/accept 仍留给后续闭环编排。

这样可以保持 true code-level 目标：LLM 写的是候选 Python reward/loss/schedule/diagnostic 组件，而不是修改 runner、预算、eval 或 checkpoint 逻辑。

## 实现内容

- 新增 `src/humanoid_agent/cold_start_lab/phase8e_provider.py`
  - `run_phase8e_provider_candidate(...)`
  - `stage_candidate_package_from_response(...)`
  - `Phase8EProviderError`
- 新增 CLI：`scripts/phase8e_generate_candidate.py`
- 新增 selftest：`scripts/phase8e_provider_ingest_selftest.py`
- 更新 `configs/cold_start/phase8e_loop.yaml`
  - 增加 `provider` 配置；
  - 默认 `dry_run`；
  - 支持 `replay`；
  - 支持 env-based `glm` 与 `custom_openai_compatible`。
- 更新 Phase 8E prompt package
  - 明确 provider JSON 输出格式；
  - 要求输出完整 candidate files；
  - 禁止请求 runner/eval/budget/checkpoint 改动。

## 安全边界

- 不保存 raw provider response；
- provider summary 不输出 API key 环境变量名；
- candidate staging 目录必须为空；
- 只允许相对 POSIX 路径；
- 禁止 `..`、绝对路径、反斜杠路径和非白名单顶层目录；
- candidate 文件内容进入 public-only assertion；
- public-only assertion 失败会被 provider ingestion 层转成 reject，不以 traceback 形式外溢，也不回显禁词细节；
- 已知 provider 凭据值若出现在 response 或 candidate 文件中会直接 reject；
- 8D verifier 仍是进入训练前的强制门。

## 本地验收

本地 Windows 不具备 Phase 8C Linux namespace/Landlock，因此本地验收只要求 8D static pass；完整 8D dynamic pass 需要在服务器验收。

已新增回归覆盖：

- 路径逃逸被拒绝；
- 缺 `candidate_manifest.yaml` 被拒绝；
- response 中出现敏感值被拒绝且错误不回显敏感值；
- dry-run provider 可以落成 LLM-authored no-op candidate，并通过 8D static；
- replay provider 可以复用同一接口落成 candidate package。

## 后续衔接

8E-6-3 完成后，下一步仍需要实现 full round driver：

```text
provider candidate package
-> 8D verifier
-> fixed-budget isolated training
-> runner-resolved sealed checkpoint
-> trusted public eval
-> decision / ledger / accepted recipe snapshot
```

只有上述链路跑通后，才能称为 Phase 8E 单 Agent code-level cold-start 闭环首轮测试。

## 2026-06-27 补充：provider 小问题收口与 full-round driver

本次补齐了两个 provider ingestion 层面的工程问题：

- `--skip-verify` 不再报告 `status: pass`，而是报告 `status: staged_unverified`，并显式给出 `training_ready: false`。这样 parser/落盘测试仍可使用 skip-verify，但 full-round driver 不会把“只落盘、未过 8D”的候选当作训练绿灯。
- candidate package staging 改为“预校验全部文件 → 临时目录写入 → 原子 rename”。如果 public-only、敏感值、路径或必需文件检查失败，不会在正式 candidate 目录留下半包残留。

同时新增完整闭环入口：

```text
provider/replay/preexisting candidate
-> Phase 8D verifier must pass
-> unshare + minimal-rootfs fixed-budget train-start
-> runner-owned sealed checkpoint safe-load gate
-> trusted public rollout on that checkpoint
-> trusted eval projection
-> decision / ledger / accepted recipe inheritance
```

新增文件：

- `src/humanoid_agent/cold_start_lab/phase8e_full_round.py`
- `scripts/phase8e_full_round.py`
- `scripts/phase8e_full_round_selftest.py`

关键边界：

- 默认要求 server hard enforcement；如需本地 parser/debug 才能显式传 `--no-require-server-enforcement`。
- 训练预算固定从 `configs/cold_start/phase8b_budget.yaml` 读取，当前为 `4096 envs × 500 iterations × 24 steps = 49,152,000 env-steps`。
- full-round summary 只记录 rollout summary 路径和聚合阶段状态，不回显 raw rollout seed；seed 只存在 runner-owned raw rollout summary 内，随后由 trusted eval 投影成 run-index 聚合结果。
- eval 端只使用 runner-resolved sealed checkpoint host path；候选不能上报 checkpoint 路径给 eval。

本地验收：

- `scripts/phase8e_provider_ingest_selftest.py`：pass
- `scripts/phase8e_full_round_selftest.py`：pass
- `scripts/phase8e_linked_round_selftest.py`：pass（本地无 torch，safe-load 子项按既有逻辑 skipped）
- `scripts/phase8e_trusted_public_eval_selftest.py`：pass（本地无 torch，trusted eval CLI 子项按既有逻辑 skipped）

## 2026-06-27 补充：full-round 顶层诚实字段与失败分类

审计确认两个问题真实存在，并已修复：

- 顶层 `phase8e_full_round_summary.json` 增加权威字段 `formal_training_executed` 与 `formal_training`。当 fixed-budget isolated training stage 通过时，顶层明确记录训练已经发生，并写入 elapsed/budget/summary 路径。嵌套的 `phase8e_linked_round_summary.json` 与 `phase8e_round_summary.json` 仍保持局部语义：这些阶段本身不负责训练，因此不把它们改成训练执行者。
- full-round driver 的失败分类从粗粒度 `fail/reject` 收紧：
  - provider/8D 未通过：`reject`
  - isolated training boundary blocked 或训练进程不可用：`blocked`
  - sealed checkpoint / safe-load / trusted eval boundary 类失败：`failed_safety_gate`
  - 未预期 orchestrator 异常：`fail`

新增 selftest 覆盖：

- 成功 full-round 顶层必须 `formal_training_executed: true`；
- blocked training boundary 必须映射为 `blocked`；
- checkpoint safe-load failure 必须映射为 `failed_safety_gate`。
