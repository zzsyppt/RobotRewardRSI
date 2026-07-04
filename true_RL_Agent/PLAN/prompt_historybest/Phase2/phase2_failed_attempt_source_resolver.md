# Phase2: Failed-Attempt Source Resolver 收敛

## 目标

把 verifier/runtime repair 中的失败候选源码读取，从 `previous_components_py` 这条临时字段迁移到 Phase1 已建立的 `source_catalog` / `source_payloads` / `authoring_context_frame` 体系。

Phase2 只处理 failed-attempt source 归一化，不改变 provider 输出协议，不开放 source request，不修正 runtime failure 同轮 retry 语义；runtime failure 边界留给 Phase3。

## 实施计划

1. 在 `true_rl_agent.prompting.source_catalog` 内扩展统一 resolver。
   - 继续使用 logical id `attempt/latest_failed/training_code/components.py`。
   - 复用 candidate package root 解析、path containment、symlink 拒绝、path/secret scrub、hash、bytes、summary、inline/read budget。
   - current-best 和 latest-failed 共用 source payload 构造与 validator，不在 live loop 再读文件。

2. 让 source context 可以携带 latest failed attempt。
   - `build_prompt_source_context()` 增加可选 `latest_failed_candidate` 输入。
   - catalog 中 failed-attempt item 从 `pending` 变为 `inline` 或 `omitted_over_budget`。
   - `source_payloads` 可以同时包含 current-best 和 latest-failed payload，但必须通过同源 validator。
   - `candidate_lineage` 仍只描述 current-best，不把 failed attempt 混成 accepted lineage。

3. live loop repair feedback 改为 source ref。
   - 删除 `_candidate_components_source()` 直接 path reader。
   - `_build_repair_feedback()` / `_build_runtime_repair_feedback()` 不再写 `previous_components_py`。
   - repair feedback 只写 `previous_candidate_source_logical_id`、source delivery、hash/bytes/summary 等 public-safe metadata。
   - `_build_attempt_prompt_payload()` 把 latest failed candidate 传给 `PublicContextBuilder.build()`，使 failed source 进入 canonical `source_catalog` / `source_payloads`。

4. frame 与预算保持不爆炸。
   - `static_repair` frame 可以包含 latest failed source payload metadata。
   - source payload 全文只在 inline budget 内出现。
   - 超 read budget 只暴露 raw bytes hash、bytes、粗 summary，不暴露源码内容。
   - `public_context` transitional alias 不重复携带 source sections。

5. 回归测试。
   - verifier repair 的第二次 provider payload 包含 failed-attempt logical source payload，不包含 `previous_components_py`。
   - failed source 中的本地/服务器路径被 scrub，不泄露真实路径。
   - oversized failed source 不进入 prompt content，但 catalog/frame 仍有 hash、bytes、summary。
   - runtime repair 也使用同一 latest-failed source path，不再走第二套 reader。
   - non-push 任务不引入 push-specific 公共语义。

## 边界

- Provider 仍是 candidate-only：`CandidateProvider.next_candidate(prompt_payload=...) -> CandidatePackageRef`。
- 不新增 `source_request` 或 provider union response。
- 不让 provider 访问真实文件系统。
- `candidate.path` 只用于 runner-side private source resolution，不能进入 prompt。
- 失败候选源码和 current-best 源码共用 source validator；不允许 live loop 维护平行 path reader。
- Phase2 不改变同一 round 内 runtime failure retry 行为；这属于 Phase3。

## 验收

- 代码中不再存在 `_candidate_components_source()` 或 `previous_components_py` 的 live-loop reader 路径。
- failed-attempt source payload 的 logical id 固定为 `attempt/latest_failed/training_code/components.py`。
- prompt payload 通过 public-only 和 source section consistency 校验。
- direct repair prompt 不泄露 Windows path、server path、secret、raw seed、checkpoint。
- full unit/integration/compileall/diff-check 通过。
