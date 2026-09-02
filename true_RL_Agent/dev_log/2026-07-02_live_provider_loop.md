# 2026-07-02 Live Provider Loop

## 目标

按 Phase B 之后的缺口补齐真实 LLM 候选生成与 live round loop，避免 15 rounds 只停留在 dry-run audit。

## 实现

- 新增 OpenAI-compatible chat provider，默认可连接 DeepSeek 兼容接口。
- provider 只接收 public prompt payload，使用 JSON object 响应，物化受控候选包文件：
  - `candidate_manifest.yaml`
  - `training_code/components.py`
  - `rationale.md`
  - `operation_diff.json`
- 新增 `run_live_loop`，每轮持久化 prompt payload，调用 provider 生成候选，再交给 runner-owned round 执行训练、评估与 ledger 记录。
- 扩展 `scripts/run_loop.py --mode live`，接入 provider、IsaacLab backend、SubprocessRunner、StaticCandidateVerifier 与 push recovery task。
- prompt payload 新增 candidate component contract，暴露已有 `RewardInputs` 字段、允许 import、构造签名和最小 `components.py` 骨架，降低真实 LLM 生成不可验证代码的概率。
- review 中发现并修复失败候选自报污染问题：当 verifier fail 且没有可信 `derived_change_set` 时，round 不再使用 LLM 自报 changed components，而是记录为空变更与 `no_code_change`。
- 服务器 provider-only smoke 发现 prompt contract 允许 `RewardInputs` 类型标注，但 StaticCandidateVerifier 未放行该 import；已补齐 allowlist 并加回归测试。
- 服务器 live smoke 发现 provider 偶发收到 fenced/prose 包裹的 JSON；已增加严格 JSON 优先、失败后提取首个平衡 JSON object 的解析路径，并覆盖回归测试。
- 参考旧工程 `src/humanoid_agent/cold_start_lab/phase8e_provider.py`：旧实现包含 attempt artifacts、source request budget、JSON syntax repair 和 static-failure repair feedback。本轮只迁移与当前 live smoke 直接相关、且不引入旧工程耦合的 JSON 包裹解析边界；完整 repair loop 留作新项目独立接口增强。
- live smoke 继续发现 4096 token 容易截断完整 `components_py`，且模型可能把 `declared_changed_components` 写成说明字符串。已参考旧工程 provider token 预算把 live 默认提升到 8192，并将无效自报组件降级为空集合；执行真相仍由 StaticCandidateVerifier 的 `derived_change_set` 决定。
- live smoke 显示只完成 verifier reject 也会让旧 summary checks 通过；已加入轻量 provider repair attempt，静态失败先反馈公开 verifier issue 和候选源码给下一次 provider call，最后候选才进入 ledger/runner。同时 live summary 增加 train/public result 计数，避免无训练轮次被误报为成功。
- live smoke 检查发现 eval raw summary 已写出 `candidate_code_loaded_in_eval=false` 和 `candidate_package_loaded=false`，但 `SubprocessRunner` 未投影到 `PublicEvalResult.metrics`，ledger 难以直接审计公开评估隔离。已将这两个 runner-owned 布尔证据纳入 public metrics，并保留类型校验。
- 15-round 实测第 1 轮暴露候选通过静态验证但训练期张量布尔判断失败的问题；训练 summary 已有 `status=fail/errors`，但 runner 原先只报告 checkpoint 缺失。已新增候选运行时错误类型，保留公开错误摘要，并让 live loop 在非最终 attempt 把运行时失败反馈给 provider 修复；最终仍失败时记录为 `candidate_runtime_failed` reject，不让坏候选崩掉整轮测试。
- 15-round 结果审计发现默认 `policy_accept_delta=0.0` 配合 `>=` 会把 public score 与当前 best 持平的候选标记为 `policy_improvement_accept`。已改为必须严格高于当前 best，且仍满足 delta 阈值，避免把无提升结果记成策略改进。

## 验证

- `python -m unittest tests.unit.test_openai_compatible_provider tests.integration.test_live_loop tests.integration.test_dry_run_audit_loop -v`
- `python -m unittest tests.integration.test_fake_round tests.unit.test_push_recovery_task tests.integration.test_live_loop -v`
- `python -m unittest discover -s tests -v`
- `python -m compileall -q src scripts tests`
- `git diff --check`

本地全量测试结果：109 tests passed。

## 边界

- API key 仅从 env 或 env-file 读取，不写入 prompt payload、candidate artifacts 或日志。
- prompt payload 仍由 public-only guard 检查，禁止 private、hidden、trusted eval、checkpoint selector、secret/token 类键和值。
- LLM 只能生成 candidate training components；训练协议、eval 协议、课程、预算、seed、接受规则仍由 runner/task/backend 拥有。
