# P6-A2 正式 author-only paired-ready 探针实施计划

> 状态：正式运行与严格验收已闭合。机器聚合为 `12/15`、人工保守降级后为
> `11/15`，等价漏放为 0，P6-A2 工程门通过；P6-B 仍需用户另行批准。
> 本文件与 create-once `probe_plan.json` 共同记录运行前冻结边界，运行后的权威结论见
> `P6A2_ACCEPTANCE_REPORT_ZH.md`。

## 1. 目标与非目标

本阶段只回答一个工程问题：在 A 的生产 `single_component_semantic_exclusion`
进入 B 的正式 authoring prompt 后，B 的**首个且唯一一次** provider 响应有多少份能经当前
Stair admission、component smoke、anti-degenerate 与 verifier-owned sibling comparison 闭合为
真正不同的单组件候选。

本阶段不训练、不评测、不访问 sealed、不写 primary ledger、不推进 current-best，也不估计
候选的科学效果。它不建立第二套相似性分析器；blocking 判定只调用 P4 已交付的
`SingleComponentSemanticExclusion` 与 `CandidateAuthoringLoop`。

## 2. 冻结输入

- 唯一输入容器以 SHA256 绑定；服务器 create-once 文件名为
  `paired_ready_source_20260817_c285da71.tar.gz`；SHA256
  `c285da71b3c2b6110a8f02f010f61bdb93ae1ac1fb5172e0f236ef84224e2b79`。
- 15 行 source ledger SHA256：
  `1d49e22d0530b40332ced06e1f562d57855ba9ded927d258c32850b5634a95e2`。
- source Session Genesis：
  `744cc76d31704ebf2f5571c9bc8a7261242471d42a604fd86b3b05c0e9a2059b`。
- 固定顺序为 source round 1..15；每轮只使用 `attempt_000001/prompt_payload.json`
  作为 base context，并使用该轮 ledger 封存的 A candidate。R09/R10 虽有第二次历史
  attempt，但首 prompt 的 authoring anchor 与最终 A 的 parent 已逐字节核对一致。
- A exclusion 必须由当前
  `SingleComponentSemanticExclusion.from_verified_candidate(A, first_prompt)` 重算；旧
  `probe_exclusions.json` 的手写语义不是权威输入。
- create-once plan 还绑定运行代码 commit、每个 prompt/A source/A anchor/prefix ledger、
  生产 8 字段 exclusion、完整 provider request、request hash 与真实 wire UTF-8 字节数。

旧 15-call 报告中的 `13/15` 是信息更充分且泄露 A candidate/round ID 的上界；旧 5-call
production recheck 只有 5 个样本且使用 16K 输出上限。两者都不得用于本次 GO 判定。

## 3. 冻结 provider 与调用预算

- API：DeepSeek official `/chat/completions`。
- model：`deepseek-v4-flash`。
- thinking：显式 `enabled`。
- `reasoning_effort=max`；不做 high/max sweep。
- `max_tokens=32,768`。这是项目的单次 reasoning+final 共享输出预算，不是要求生成
  32K 最终 JSON，也不是 provider context 上限。
- thinking 请求不发送 `temperature`。
- `response_format={"type":"json_object"}`，non-streaming，timeout 300 秒，response
  envelope 512 KiB。
- 15 个 case 每个只有一次 durable attempt reservation；总请求恰好 15，无 retry、补样、
  换样本或缩分母。用户已豁免金额门，usage 仍逐 case 留证。

若本次门通过，P6-B 及后续新 generation-session contract 默认沿用上述 provider 组合；
任何变更必须在新的运行合同中另行批准，不能根据本次或正式 session 中途结果调整。

## 4. 唯一执行路径

1. task-owned source loader 复核 archive/ledger/Genesis 与 15 轮 prompt/A/anchor。
2. 当前 Stair admission 重新验证 A；生产 exclusion 原语投影唯一 public 8 字段 payload。
3. `FrozenStairPromptBuilder` 保留历史 public base，只通过当前 `build_prompt_payload`
   重建 candidate contract/frame；A→B 唯一动态变化只能落在
   `required_candidate_package` 与 `authoring_context_frame`。
4. `CandidateAuthoringLoop(max_attempts=1, max_source_requests_per_round=0)` 持有正式
   authoring journal；provider 实际 payload 必须逐字节等于 plan 中的 request。
5. 当前静态 verifier、CPU smoke、anti-degenerate 与 sibling comparison 依次运行；probe
   只记录其正式结果，不自行比较源码 hash、组件名或 candidate ID。
6. 每个 case 保存 create-once request、raw JSON response、authoring evidence 与
   case result；全部 15 个闭合后才创建 aggregate result。

已闭合 case 恢复时只回读 evidence，不新增调用；authoring terminal evidence 已封存但
case receipt 尚未落盘时，可由同一 journal 无 HTTP 重投影 receipt。若 prompt 已 reserve、
但没有 terminal evidence，该 case 的唯一调用预算已消耗，整个 probe 保持 incomplete/no-go；
不得删除目录后重叫。response 文件本身不能替代 terminal authoring evidence。

## 5. 互斥分类与冻结门

- `valid_structurally_distinct`：正式 stop reason 为 `candidate_verified`，smoke/static/
  anti-degenerate 均通过，production sibling assessment 为 pass，且独立复算同一 verifier
  comparison 返回 `None`。
- `equivalent_blocked`：`identical_or_format_only` 或 `parameter_only_delta` 被正式 sibling
  gate 阻断。这证明 exclusion 生效，但不计 paired-ready。
- `equivalent_leak`：候选被标为 verified，独立复算却仍为上述等价类；任意 1 条即 NO-GO。
- `authoring_invalid`：compound/no-single-delta、comparison unavailable、static/smoke/
  anti-degenerate 失败或 source_request 等其他非 ready 结果。
- `provider_failed`：transport、完整响应解析或 materialization 未闭合；仍占固定 15 分母。

工程 GO：`valid_structurally_distinct >= 10/15`、`equivalent_leak == 0`、实际请求数恰好 15。
15 个历史 case 不是独立同分布样本，因此不计算 Wilson interval，也不外推“真实概率”。

## 6. 人工复核与验收

机器聚合之后，主 agent 必须逐份打开并核对 15 个 case 的：source A delta、生产 exclusion、
实际 request 字段、完整 response、物化 `components.py`、static/smoke/anti-degenerate、正式
sibling assessment 与独立复算。人工只能把可疑机器通过降级或发现 leakage，不能把截断、
reasoning 草稿、失败 smoke 或 compound delta 升级为有效候选。

最终交付包括：15 行人工复核表、aggregate 独立重算、调用/usage 汇总、零 GPU/runner/eval/
sealed/ledger/head 证明、focused/full tests、compile/diff-check、local/origin/server commit 与
tracked-clean 证据，以及 `P6A2_ACCEPTANCE_REPORT_ZH.md`。通过只开放“请求用户批准 P6-B”
的门，不自动启动 P6-B。

## 7. 实际闭合与冻结结论

- plan 构建提交：`b5dff4df89ecd43b7cdbe9b7d99d69ebb74c930c`；运行时投影修正提交：
  `bb226275730581ec61d0b5e87dec3e1601c48d16`。后者是前者的直接单提交子代，diff 仅含
  probe CLI、Stair-owned runtime 和对应单测三个预声明路径。
- 唯一正式输出：
  `/data/yuyangliu/true_rl_agent_phase_g/p6a2_author_only_b5dff4d_20260819T165305Z`。
  `plan_hash=548c0c0391006d5d47e62cb17eaf1e34dce4e76ed21ad79ac13c774d9e93cfd7`，
  `runtime_correction_receipt_hash=ac44ec14e61b81aba23ce0facc1df33a5e8aebbd34747225466dc59bf82643eb`，
  `result_hash=45d9bf91ce9ee40601dbfca776f58ffc653116ef2db928177760518c7e0d306e`。
- 首次运行在 R01 provider terminal evidence 已 create-once 后，因 receipt 投影误调用不存在的
  `AuthoringEvidenceRef.to_public_dict()` 退出。原请求、响应、evidence、日志与退出码保持不变；
  修正只把 evidence ref 投影改为既有 private contract，并以 create-once correction receipt
  绑定旧/新提交与原制品。R01 通过 terminal journal 零网络重投影，未产生第二次请求；随后
  R02--R15 在修正提交上各调用一次。旧运行没有被重判、覆盖或换样。
- 机器结果：12 `valid_structurally_distinct`、2 `provider_failed`、1 `authoring_invalid`、
  0 `equivalent_blocked`、0 `equivalent_leak`。主 agent 逐件复核后将 R14 因
  `projected_gravity[:, 2] > 0.85` 把倒置方向误作 upright 而降级，最终保守口径为
  11 valid、2 provider failed、2 authoring invalid、0 leak，仍通过 `>=10/15` 门。
- 本次通过冻结后续 generation 的默认 provider 组合为
  `deepseek-v4-flash + thinking enabled + reasoning_effort=max + temperature omitted +
  max_tokens=32,768`；任何改变都必须进入新合同并另行批准。该结论只证明 authoring
  paired-ready，不授权训练、评测或 P6-B。
