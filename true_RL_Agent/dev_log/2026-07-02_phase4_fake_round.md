# Phase 4 FakeTask + FakeBackend + FakeRunner 最小纵向闭环

## 目标

按照计划阶段 4，在接入 Isaac Lab 和 push 真实训练前，用 fake task/backend/runner 跑通一次最小单轮闭环，验证 ledger 和 decision shape。当前阶段不启动真实训练，不连接服务器训练环境，不接真实后端。

## 本次实现

- 新增 `CandidateVerifier` 与 `VerificationResult`，让 fake round 显式经过 verifier gate。
- 新增 `Runner` 抽象接口与 `TrainJobResult`，让 fake runner 通过同一 runner 边界返回 checkpoint。
- 新增 `DecisionRecord`、`decide_candidate()` 与 `reject_without_eval()`，把四类 outcome 落成可写入 ledger 的 public decision。
- 新增 `append_ledger_record()` 与 `load_ledger()`，采用 append-only JSONL。
- 新增 `RoundInputs`、`RoundResult` 与 `run_candidate_round()`，跑通 public context -> replay candidate -> verifier -> fake train -> fake eval -> decision -> ledger。
- 新增 `tests/fakes/` 下的 FakeTaskPlugin、FakeBackendAdapter、FakeRunner、FakeVerifier。
- 新增 `tests/integration/test_fake_round.py`，覆盖 policy accept、parameter-only reject、diagnostic record、engineering record、verifier fail no-train。

## Review 中发现并修复的问题

初版 ledger 记录的 eval checkpoint 是 backend planned checkpoint，而不是 fake train 结果替换后的实际 eval checkpoint。已新增测试确认 `jobs.eval.checkpoint.uri == train_result.checkpoint.uri`，并修复 `run_candidate_round()` 记录实际 eval spec。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest tests.integration.test_fake_round -v
```

结果：5 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：22 个测试通过。

```powershell
rg -n "humanoid_agent|D:\\humanoid_agent|/data/yuyangliu/AGENT|\\.\\.[/\\]src[/\\]humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

```powershell
git diff --check
```

结果：无空白错误。

## Review 结论

- FakeRunner 不启动训练，只生成 deterministic train/eval summary。
- Verifier failure 会 reject 且不会 materialize jobs、不会 train/eval。
- parameter-only candidate 即使 public score 很高也不能 accept。
- diagnostic-only 与 engineering-only candidate 只记录，不冒充 policy improvement。
- ledger 只包含 public context、public result、decision 和反馈安全标志，不写入 hidden/private eval 信息。

## 下一步

进入阶段 5：实现 Runner Job 抽象。阶段 5 应补 artifact layout、budget check、checkpoint seal/safe-load、isolation placeholder、eval seal 等 runner-owned 结构；不能直接跳到真实 Isaac Lab push 训练。
