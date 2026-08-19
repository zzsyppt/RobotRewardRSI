# P6-A2 author-only paired-ready 正式验收报告

> 验收日期：2026-08-20
>
> 结论：**GO**（机器 `12/15`；逐件人工保守降级后 `11/15`；等价漏放 `0`）
>
> 权限边界：只冻结后续 generation 的默认 provider 合同；不授权 P6-B、GPU、train、eval、
> sealed、primary ledger 或 current-best 变更。

## 1. 结论

正式 P6-A2 已按预冻结的 15 个 production-shaped 首请求完整闭合。15 次 provider execution
均有唯一 request/response、authoring attempt 与 terminal receipt；每个 case 计数恰好为 1，
零 retry、零补样、固定分母 15。该证明来自 runner-owned durable artifacts，而不是独立网络
抓包或 provider 账单。机器聚合得到：

| 分类 | 数量 |
|---|---:|
| `valid_structurally_distinct` | 12 |
| `provider_failed` | 2 |
| `authoring_invalid` | 1 |
| `equivalent_blocked` | 0 |
| `equivalent_leak` | 0 |

主 agent 逐件阅读 A exclusion、B 源码、evidence、smoke 与 comparison 后，将 R14 从机器
valid 降级为人工 `authoring_invalid`：它把 `projected_gravity[:, 2] > 0.85` 当作 upright，
而 task-owned prompt 明示直立方向为 `-z`。最终保守口径为 `11 valid / 2 provider_failed /
2 authoring_invalid / 0 leak`，仍满足 `>=10/15` 且 leak 为 0 的冻结门。

因此，后续新 generation-session 的默认 provider 合同冻结为：

```text
model=deepseek-v4-flash
base_url=https://api.deepseek.com
endpoint=/chat/completions
thinking=enabled
reasoning_effort=max
temperature=omitted
max_tokens=32768
timeout_seconds=300
response_envelope=512 KiB
response_format=json_object
stream=false
retry_policy=none
```

不得根据正式 generation 的中途结果静默切换模型、reasoning effort、temperature 或输出上限；
如需改变，必须创建新合同并另行批准。

## 2. 冻结输入与制品身份

- 唯一 source archive SHA256：
  `c285da71b3c2b6110a8f02f010f61bdb93ae1ac1fb5172e0f236ef84224e2b79`。
- 15 行 source ledger SHA256：
  `1d49e22d0530b40332ced06e1f562d57855ba9ded927d258c32850b5634a95e2`。
- source Session Genesis：
  `744cc76d31704ebf2f5571c9bc8a7261242471d42a604fd86b3b05c0e9a2059b`。
- plan 构建提交：`b5dff4df89ecd43b7cdbe9b7d99d69ebb74c930c`。
- receipt 投影修正提交：`bb226275730581ec61d0b5e87dec3e1601c48d16`；是 plan 提交的
  直接单提交子代，且只改 probe CLI、Stair-owned runtime、对应单测三个预声明路径。
- 唯一正式输出：
  `/data/yuyangliu/true_rl_agent_phase_g/p6a2_author_only_b5dff4d_20260819T165305Z`。

| 制品 | 内容 hash | 文件 SHA256 |
|---|---|---|
| `probe_plan.json` | `548c0c0391006d5d47e62cb17eaf1e34dce4e76ed21ad79ac13c774d9e93cfd7` | `a28aa6d0a74e881b6b54bb4b0278122922bf1f06593108f9d50d4a85aec85c85` |
| `runtime_correction.json` | `ac44ec14e61b81aba23ce0facc1df33a5e8aebbd34747225466dc59bf82643eb` | `3137fc7e6b062f7789402648ab1ce558683012d188f70c39d7a0e47b6946b1c4` |
| `probe_result.json` | `45d9bf91ce9ee40601dbfca776f58ffc653116ef2db928177760518c7e0d306e` | `de8b40da1f07a45e224fd72ce9cb9337bee66513f20abdb5d4ba096edd9ce8e8` |

15 个 plan exclusion 另由 frozen ledger、历史 A candidate 和 own anchor 重新投影，逐项等于
plan；13 个实际 B candidate 均再次调用 task-owned verifier comparison，结果全部为 `None`。
因此“零等价漏放”不是依赖 aggregate 自报字段。

## 3. 唯一运行异常与根修

首次运行在 R01 的 request、response 和 terminal authoring evidence 已 create-once 后，receipt
投影误调用不存在的 `AuthoringEvidenceRef.to_public_dict()`，进程以 1 退出。R01 本身是合法
`provider_failed/json_parse_error`：`finish_reason=stop`，final content 为 0 bytes，只有
reasoning 草稿；草稿从未被当成候选。

没有删除 R01、没有重发、没有 fresh output 替换失败样本，也没有改写原 plan。修正提交只做
task-owned receipt projection 根修：改用既有 private ref 合同，以 create-once correction
receipt 绑定原 plan、旧/新提交、原日志/退出码、R01 request/response/evidence；R01 从 terminal
journal 零网络重投影，随后 R02--R15 各调用一次。最终 provider execution split 为
`{b5dff4d: 1, bb226275: 14}`，trigger replay 新增请求数为 0。

原始失败证据保持不变：

- `run.log` SHA256：`0de89c39c987c3030b89902b5cc3c0f6c04f2a61b2a7d1ebb8f0815dac3f6f0b`；
- `run.exitcode` SHA256：`4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865`；
- resume log/exit 分别写入独立文件，resume exit 为 0。

这是工程协议修正，不是根据结果放宽科学阈值：15 个输入、provider 请求合同、`>=10/15`、
zero-leak 门和所有 authority 边界均未改变；旧 incomplete 运行也没有被重新解释成成功。

## 4. 严格重算

三路只读审计分别从 raw plan/case/request/response/evidence/candidate 重算。最终独立检查
`721/721` 通过，另一路审计 `ERROR_COUNT=0`：

- 15 request、15 response、15 case result、15 attempt、15 evidence，round 1..15 连续；
- 每例 request/response/evidence 唯一，每例 provider request count 恰为 1；
- 15 case hash、prompt ref、candidate source checksum/size、plan/correction/result canonical hash
  全部一致；
- R01 原 request、response、evidence、日志与退出码均未变化，evidence 仍只有一份；
- 正式运行和代码验收时 local/origin/server 的 runtime source identity 均为
  `bb226275730581ec61d0b5e87dec3e1601c48d16`，服务器 tracked worktree clean；后续仅文档
  提交不会改变该运行身份；
- 结束后 probe 进程、tmux、GPU compute process 均为 0，输出中无 train/eval/sealed/runner
  目录。

provider usage 合计（API usage units）：

| 字段 | 数量 |
|---|---:|
| request/input | 367,449 |
| response/completion | 141,658 |
| 其中 reasoning | 113,693 |
| cache hit | 66,176 |
| cache miss | 301,273 |
| total | 509,107 |

用户已豁免金额门；usage 只作事实记录，没有触发样本选择、重试或提前停机。

## 5. 人工复核结论

逐轮表见 `MANUAL_REVIEW_15_ZH.md`。除 R14 外，其余 11 个机器 valid 无需降级；它们都
相对 own anchor 具有恰好一个 canonical RewardTerm delta，static、smoke、anti-degenerate、
sibling gate 均 pass，独立 equivalence 为 `None`，且不是改名、权重或 parameter-only 变体。

R08 对向下 `height_scan` 使用“headroom”命名，rationale 对前方/躯干障碍的解释偏强；但
order-invariant reduction 合法、结果有限且形状正确，相对本轮 A 真实不同，因此记录为质量
风险而不降级。候选集中在净空/地形与近顶端稳定两个近族，说明 provider 能满足局部 sibling
排除，但不能据此声称跨代概念多样性已经充分。

## 6. 权限与旧任务边界

plan、correction、aggregate 和 15 个 case 的六项 authority counters 全为 0：

```text
gpu_jobs=0
train_jobs=0
eval_jobs=0
sealed_queries=0
primary_ledger_writes=0
current_best_transitions=0
```

实现只在 Stair task-owned probe/runtime 中处理 source 与 correction；manual verdict 仅由验收
文档记录，原机器 result 保持 12 valid 不改写。common authoring/verifier/provider 合同被直接
复用，没有引入任务语义、第二套等价判定器、通用恢复框架或新调度器。Push 与 legacy
width=1 路径未改变。

当前关键源文件 SHA256：

| 文件 | SHA256 |
|---|---|
| `scripts/run_stair_paired_ready_probe.py` | `488cb3bab0208648aa8272deae6d7aedaac1ed24ab0524bcfd02c2bcd6c5854e` |
| `tasks/stair_climbing/paired_ready_probe.py` | `4d6e2cd2bff3f2c84080a6d2f7bef47255b72b9473afdbb9f9b76b6d700fa1e2` |
| `tasks/stair_climbing/paired_ready_probe_runtime.py` | `8576fa1312ba495fdec350fcc65fb039054751c292e382270022a861318132c5` |
| `tests/unit/test_stair_paired_ready_probe.py` | `7fcab4cb0df337104b7e38bf9f6548fdb36ab52392fe1ade965375c81860a189` |

验收测试：P6-A2 focused `9/9`；authoring/provider/evidence 相关 `131/131`；全量
`1279 tests / OK / skipped=21`；`py_compile/compileall` 与 `git diff --check` 通过。

## 7. 最终门

```text
P6-A2 machine gate: GO (12/15, leak=0)
P6-A2 manual conservative gate: GO (11/15, leak=0)
provider contract for subsequent generation: FROZEN
P6-B authorization: NOT GRANTED
```

P6-A2 只证明生产 exclusion 下首个 B authoring 响应具有足够 paired-ready 率。它不证明候选
会训练成功，不证明 width=2 端到端 generation 已接入，也不替代 P6-B 前正式 Stair UUID
隔离、lease、train/eval receipt 和 generation ledger 的实现与验收。下一步只有在用户明确批准
后，才进入 P6-B。
