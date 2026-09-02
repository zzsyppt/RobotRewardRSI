# 2026-07-02 Phase 8 Candidate 与 Prompt Projection 基础能力

## 本阶段目标

按计划完成 Phase 8：建立 LLM proposal 的安全边界，让 prompt 只展示当前真实可用能力，并把 candidate package 限制在受控入口内。

## 实现内容

- 新增 `TrainingComponents`、`RewardTerm`、`DiagnosticTerm`、`ScheduleTerm` 受控组件模型。
- 新增 candidate manifest/loader，第一版只允许 `candidate_manifest.yaml` + `training_code/components.py`，entrypoint 固定为 `training_code.components:build_training_components`。
- 新增 `StaticCandidateVerifier`，限制候选文件白名单、manifest schema、entrypoint、Python AST 顶层形态、导入范围和危险调用。
- 新增 prompt-facing projection：只暴露 `RewardTerm`、`DiagnosticTerm`、`ScheduleTerm` 和公开 `RewardInputs` 字段，不暴露 env、runner、optimizer、checkpoint selector、eval config、success metric。
- 新增 closed source-pack policy，只允许通过 logical id 访问 allowlist source。
- 新增 `MechanismEvidenceGate`、runtime activation evidence schema 和 knob-only shadow candidate 生成规则。
- 明确 parameter-only candidate 不允许进入 policy accept 路径；DiagnosticTerm、ScheduleTerm 只能按 surface policy 进入对应的 record/reject 路径。

## Review 结果

- 修正了 prompt payload 中通过 `excluded_fields` 泄露 `eval_config` 等 runner/eval 控制面名称的问题，改为只发布正向白名单字段。
- 修正了 `RewardTerm`/`DiagnosticTerm` 只保存字符串引用的问题，改为保存真实 callable，避免退化为不可加载的标签。
- 补强 verifier：候选代码必须先通过 AST 审计，再允许 loader 导入；entrypoint 逃逸、动态 import、顶层副作用都会被拒绝。

## 验证

- `python -m unittest tests.unit.test_candidate_package tests.unit.test_prompt_projection_and_mechanism -v`
- `python -m unittest discover -s tests -v`
- `rg` 旧工程路径和父项目 runtime 依赖扫描无命中。
- `rg` 危险 shell/subprocess 调用扫描无命中。
- `git diff --check -- true_RL_Agent`
- `python -m compileall -q src tests`
- `python -m pip install --dry-run -e .`

## 结论

Phase 8 已按当前计划完成。没有开放任意 repo patch、env/runner/eval 控制面、未接线 capability 或纯调参 accept 路径。
