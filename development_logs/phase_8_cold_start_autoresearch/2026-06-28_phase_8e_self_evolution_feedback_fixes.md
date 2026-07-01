# Phase 8E self-evolution feedback + public-only hardening

日期：2026-06-28

跟在 8E-6-4(多轮自演进循环)与 live-provider-readiness 修复之后，补两个 review 里发现的真问题。

## Finding A：训练阶段失败也要反馈给下一轮（之前只有训练前反馈）

之前只有"候选过不了 8D / provider 门"时会写一条公开失败反馈进共享 ledger；如果候选**过了 8D 但训练阶段失败**（NaN/发散/超时/checkpoint 没封好），`run_phase8e_full_round` 在到 linked round 之前就 raise，**不写 ledger**，于是下一轮 LLM 看不到"上一轮训练崩了"。

修复：

- 把 `_append_pretraining_failure_feedback` 泛化成 `_append_failure_feedback(..., detail)`，两条失败路径共用；
- 训练失败时也 append 一条公开反馈，`detail.stage="fixed_budget_training"`、`training_status` 为 runner 分类枚举（blocked/reject/failed_safety_gate/fail）；
- **反馈 reason 只用 runner 拥有的分类**（`fixed_budget_training_<status>`），不回显原始 `training["reason"]`——因为原始 reason 可能夹带候选影响的错误文本（如 reject_reason）。原始 reason 仍保留在 full_round summary（落盘、面向运维），但**不进 prompt**。
- 回归：`test_training_failure_appends_public_feedback_without_raw_reason` 断言训练 blocked → 1 条公开反馈、`stage=fixed_budget_training`、`training_status=blocked`、`reason=fixed_budget_training_blocked`，且原始 `cuda_oom_in_candidate_reward` 不出现在 ledger 记录里。

## Finding B：数字禁词正则误命中 sha256 fingerprint 里的 1001/2001

`assert_public_only_payload` 的禁词数字正则用 `(?<![\d.])(?:1001|2001)(?![\d.])`，会误命中 fingerprint 这类十六进制串里被字母包裹的 `1001`/`2001`（如 `...a1001b...`）。`build_ledger_record` 含 fingerprint 又做 public-only 断言，约 0.1% 概率的 fingerprint 会让 ledger append 抛错、那一轮变 `fail`。新的训练失败反馈路径也调 `build_ledger_record`，共享该隐患。

修复：边界从 `[\d.]` 改成 `[\w.]`——eval 种子总是以独立数字出现（`"seed": 1001`），仍会被抓；嵌在字母数字串里的不再误判。回归：`assert_public_only_payload({"fingerprint": "3f9a1001b7c2001de"})` 不再抛错，而独立 `{"seed": 1001}` 仍被拒。

## 边界不变

两条失败反馈都只写公开内容（verifier 投影的 issue_codes/status + 候选公开包元数据 + runner 拥有的 detail），整条记录过 `assert_public_only_payload`；不含 eval 种子、hidden 度量、原始候选错误文本。评测端不被腐蚀。
