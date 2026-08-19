# P6-A2 author-only 探针开发日志

## 2026-08-20：正式 production-path P6-A2（GO）

- 用户已批准正式 P6-A2，并明确豁免金额门；调用数量和协议预算不变：固定 15 个输入、
  每 case 一次请求、无 retry/替换、单 request <=300,000 UTF-8 bytes、O=32,768、
  response envelope 512 KiB。
- provider 预冻结为 `deepseek-v4-flash`、thinking enabled、
  `reasoning_effort=max`、timeout 300 秒、thinking 模式不发送 temperature。
- 唯一输入容器/ledger/Genesis 已分别冻结为
  `c285da71...e2b79` / `1d49e22d...a95e2` / `744cc76d...a2059b`；15/15 first
  prompt、A source、own anchor 与生产 exclusion 已离线重算闭合。
- 新实现直接复用 `CandidateAuthoringLoop`、当前 Stair admission/smoke 和
  verifier-owned `SingleComponentSemanticExclusion`；不复用旧手写 prompt/exclusion
  脚本，不触碰 GPU、train/eval、sealed、primary ledger 或 current-best。
- 离线 worst-shape preflight：15 份正式 wire request body 为 75,144--107,429 bytes
  （约 73.4--104.9 KiB），总计 1,515,605 bytes，全部低于单次 300,000 bytes。
- plan 构建/第一次运行提交：`b5dff4df89ecd43b7cdbe9b7d99d69ebb74c930c`。R01 的唯一
  request、response 与 terminal evidence 已 create-once 后，case receipt 投影误调用不存在的
  `AuthoringEvidenceRef.to_public_dict()`，进程以 1 退出；R01 final content 为空，合法分类为
  `provider_failed/json_parse_error`。
- 没有删 R01、没有重发、没有换输出。直接子提交
  `bb226275730581ec61d0b5e87dec3e1601c48d16` 只改 probe CLI、Stair-owned runtime 和单测；
  create-once `runtime_correction.json` 绑定原 plan/日志/退出码/request/response/evidence，R01
  通过 terminal journal 零网络重投影。R02--R15 随后各执行一次，commit split 为 `1+14`。
- 唯一正式输出：
  `/data/yuyangliu/true_rl_agent_phase_g/p6a2_author_only_b5dff4d_20260819T165305Z`；
  plan/correction/result 内容 hash 分别为 `548c0c03...e93cfd7`、
  `ac44ec14...82643eb`、`45d9bf91...0d306e`。
- 机器结果：12 valid、2 provider_failed（R01/R10 均为空 final content）、1 authoring_invalid
  （R13 smoke `[2,2]`，期望 `[2]`）、0 blocked、0 leak，`go=true`。
- 主 agent 逐件复核将 R14 人工降级：它用 `projected_gravity[:, 2] > 0.85` 表示 upright，
  与 prompt 明示的 `-z` 直立语义相反。最终保守结果为 11 valid、2 provider failed、
  2 authoring invalid、0 leak，仍通过 `>=10/15`。
- 三路 raw 审计重算 plan、correction、aggregate、15 case/request/response/evidence/prompt/
  candidate 均一致；另从 frozen A source 重建 15 个 exclusion，并对 13 个 B 重跑任务比较器，
  全部 `equivalence=None`。usage 合计 input 367,449、output 141,658、reasoning 113,693、
  total 509,107。
- authority 六项在 plan/correction/result/15 cases 全为 0；结束后无 probe/tmux/GPU compute
  进程，输出内无 train/eval/sealed/runner 目录。
- 最终回归：P6-A2 focused `9/9`、authoring/provider/evidence 相关 `131/131`、全量
  `1279 tests / OK / skipped=21`；compile/compileall 与 diff-check 通过。

正式 P6-A2 工程门为 GO，并冻结后续 generation 默认 provider 合同；旧 `13/15` 仍只保留
为历史高信息上界。本结果不授权 P6-B。

> 2026-08-17 纠偏：本日志记录的是旧的高信息 exclusion 探针，不是 P4 最终生产 payload。`86.7%` 只能作为上界；其中包含最终合同禁止向 B 暴露的 A candidate/round ID 和逐轮手写语义。生产同形状复查及新结论见 `PLAN/2-GPUS/P4/PRODUCTION_EXCLUSION_RECHECK_REPORT_ZH.md`。

## 2026-08-17

### 用户授权范围

- 运行约 15 次 provider 调用，回答 A 的机制被排除后 B 是否还能提出真正不同的有效机制；
- 禁止 GPU；
- 每个样本由主执行者亲自阅读和确认，脚本只作传输、物化和现有合同投影参考。

### 实际执行

- 从正式 session `stair-reference-formal-20260813T125345Z` 只读取得 15 份真实 prompt、候选、own anchor 和 ledger；
- 对每轮 A 使用现有 verifier-owned operation delta 生成 public-safe exclusion；
- 使用 `deepseek-v4-pro / temperature=0.2 / adaptive reasoning / max_output=16384`；
- R01–R15 各发出一次实际 HTTP 请求，严格串行，不重试；
- 每个可解析 B 均人工阅读 A exclusion、B changed function 与组件结构，再在服务器执行 CPU-only static/component smoke；
- 未运行 train/eval/sealed，未写 primary ledger。

### 结果

- 有效且实质不同：13
- provider 传输失败：1（R02）
- output length 截断：1（R05）
- 等价机制漏放：0
- paired-ready：13/15 = 86.7%
- 当时的高信息可行性阈值：OBSERVED（13/15 上界）；不适用于 2026-08-20
  production-path 正式门

### 边界说明

当前树没有 P4 production `single_component_semantic_exclusion` 与 `GenerationCoordinator`。本次是用户授权的一次性 author-only 可行性探针，不得冒充正式 P6-A2 生产接线验收，也不授权 P6-B GPU pilot。

### 证据

- 完整输出：`D:\humanoid_agent\true_RL_Agent\outputs\author_only_paired_ready_probe_20260817`
- 结构化结果：`RESULT.json`
- 逐轮人工表：`MANUAL_REVIEW_ZH.md`
- 正式报告：`AUTHOR_ONLY_PAIRED_READY_PROBE_REPORT_ZH.md`
