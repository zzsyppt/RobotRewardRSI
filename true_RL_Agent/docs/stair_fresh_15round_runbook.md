# Stair 全新正式 15 轮启动手册

这是一份发布后、且用户明确批准启动后才可执行的操作手册。它不是当前启动授权。旧的
`outputs/launch_genesis_*.sh` 和 `outputs/launch_live_*.sh` 没有新的最终确认计划绑定，
不得用于新主线正式实验。

## 不变量

- 使用一个全新的、此前不存在的 `OUT`，不复用旧 Genesis、ledger 或 runner 产物。
- 四个阶段使用同一个 `SESSION_ID`、`OUT`、`PLAN`、发布 commit 和
  `--execution-device`。
- `PLAN` 必须在 Genesis 之前 create-once；计划路径必须是
  `$OUT/final_confirmation/plan.json`。
- 正式 live 入口显式请求且只允许 15 轮，并且必须由一次 invocation 从空 ledger 闭合全部
  15 轮。ledger 行数或 reward 不能代替该入口生成的闭合 live receipt、ledger replay 和
  eval audit。
- 所有命令使用配置绑定的 Isaac Python。最终确认入口会再次强制检查当前解释器。
- provider 密钥只通过只读 env 文件加载，不复制进输出、命令行、prompt 或 followup。
- 任何失败都先保留现场并审计根因；不删除/覆盖正式目录，不在训练中临时叠补丁。

启动批准前，必须把
`outputs/formal_15round_38818b5_evalv6_followups.md` 中每一项逐条标为已由证据关闭、无需
实现，或明确需要下一轮实验验证；不能带着未处理的权威性 blocker 启动。发布审计还要确认
每项代码升级确有必要、在源头解决、没有把 Stair 语义推入公用层，也没有让旧任务插件退化。

## 发布和环境前置检查

在服务器 Bash 中设置本次发布值。`EXPECTED_HEAD` 与 `RUN_TAG` 必须由已批准发布填写：

```bash
set -euo pipefail

REPO=/data/yuyangliu/AGENT
PROJECT="$REPO/true_RL_Agent"
ISAAC_PY=/data/yuyangliu/envs/isaaclab/bin/python
PROVIDER_ENV=/data/yuyangliu/secrets/deepseek.env
DEVICE=cuda:0

: "${EXPECTED_HEAD:?set EXPECTED_HEAD to the approved release commit}"
: "${RUN_TAG:?set RUN_TAG to a new UTC tag, for example 20260813T120000Z}"

SESSION_ID="stair-reference-formal-${RUN_TAG}"
OUT="/data/yuyangliu/true_rl_agent_phase_g/fresh_formal_15round_${RUN_TAG}"
PLAN="$OUT/final_confirmation/plan.json"

assert_release_state() {
  test "$(git -C "$REPO" rev-parse HEAD)" = "$EXPECTED_HEAD"
  test -z "$(git -C "$REPO" status --porcelain --untracked-files=no)"
  test -z "$(git -C "$REPO" status --porcelain --untracked-files=all -- \
    true_RL_Agent/src \
    true_RL_Agent/scripts \
    true_RL_Agent/configs \
    true_RL_Agent/docs)"
}

test -x "$ISAAC_PY"
test -r "$PROVIDER_ENV"
assert_release_state
test ! -e "$OUT"
mkdir -p "$OUT"
cd "$PROJECT"
```

`assert_release_state` 会在四个正式阶段前重复检查发布 commit、全仓 tracked dirty state，
以及 `src/scripts/configs/docs` 生产表面的 untracked 文件。`OUT` 位于仓库外，正式输出不会被
这些源码状态检查误拦。不要在前置检查后切换 commit、修改 tracked 文件或更换 Python/设备。
若任一检查失败，停止，不创建正式状态。

## 1. 在 Genesis 前创建最终确认计划

```bash
assert_release_state

"$ISAAC_PY" scripts/create_stair_final_confirmation_plan.py \
  --session-id "$SESSION_ID" \
  --output-dir "$OUT" \
  --execution-device "$DEVICE" \
  2>&1 | tee "$OUT/final_confirmation_plan.log"

test -s "$PLAN"
test ! -e "$OUT/session_genesis.json"
test ! -e "$OUT/ledger.jsonl"
```

此步骤将 never-fed 最终 cohort、任务/评估协议和配对规则冻结在计划中。计划已存在时
入口会失败；不能覆盖后重试。

## 2. 用同一计划创建并只读复核 reference Genesis

```bash
assert_release_state

"$ISAAC_PY" scripts/create_stair_reference_genesis.py \
  --mode create \
  --session-id "$SESSION_ID" \
  --output-dir "$OUT" \
  --final-confirmation-plan "$PLAN" \
  --execution-device "$DEVICE" \
  2>&1 | tee "$OUT/genesis_create.log"

test -s "$OUT/session_genesis.json"
test ! -e "$OUT/ledger.jsonl"

"$ISAAC_PY" scripts/create_stair_reference_genesis.py \
  --mode load-only \
  --session-id "$SESSION_ID" \
  --output-dir "$OUT" \
  --final-confirmation-plan "$PLAN" \
  --execution-device "$DEVICE" \
  2>&1 | tee "$OUT/genesis_load_only.log"

test ! -e "$OUT/ledger.jsonl"
```

`load-only` 必须报告 runner/provider 调用数为 0，并证明 Genesis 和 ledger（若存在）未被
改写。新实验在此时 ledger 仍应不存在。

## 3. 从空 ledger 启动单次 15 轮正式 live

```bash
assert_release_state

"$ISAAC_PY" scripts/run_stair_live.py \
  --rounds 15 \
  --session-id "$SESSION_ID" \
  --output-dir "$OUT" \
  --final-confirmation-plan "$PLAN" \
  --provider-base-url https://api.deepseek.com \
  --provider-endpoint /chat/completions \
  --provider-model deepseek-v4-flash \
  --provider-api-key-env DEEPSEEK_API_KEY \
  --provider-env-file "$PROVIDER_ENV" \
  --provider-timeout-s 180 \
  --provider-max-tokens 32768 \
  --provider-reasoning-mode adaptive \
  --provider-reasoning-effort max \
  --provider-max-attempts 5 \
  --execution-device "$DEVICE" \
  2>&1 | tee "$OUT/live.log"
```

最终确认只接受这一次 live invocation 从空 ledger 生成全部 15 条记录。通用 loop 的 resume
能力仍保留，但本次正式进程一旦中断，resume 产物只能作为开发证据；必须使用新的
`OUT`、计划和 Genesis 重开完整正式 15 轮，不能把恢复后的 ledger 送入最终确认。

ledger 已有 15 条时禁止再次运行同一个 `--rounds 15` 命令。该零工作量调用会把根
`live_summary.json` 改成 `15 -> 15` 的 no-op summary，遮住原闭合 receipt 的根绑定；这个
`OUT` 随即失去最终确认资格，不能靠再次运行恢复。

每一轮都检查 `authoring` evidence、`runner/<round>/train|eval/summary.json`、正式 eval、
ledger record、live invocation receipt，以及当前 best/parent/lineage 绑定。跨轮还要检查
prompt/history/memory 的继承与探索关系；不能把进度、reward 或 provider 成功当作接受证据。
对本次改动的计划/Genesis 绑定、authoring anchor、backend/source manifest、provider attempt
evidence、public safety 和 live summary v6 要做同等逐轮重点监察，而不是只看新候选的表现。

训练中发现非阻断问题时，记录到
`outputs/formal_15round_38818b5_evalv6_followups.md`，在下一次正式启动前逐条核实；只有明确
阻断当前权威性的根因才进入“停止并重开全新 15 轮”的发布审计。

## 4. 15 轮闭合后运行单次最终确认

只有 live 命令成功返回、15 条 ledger 均完成逐轮审计、且入口生成的闭合 receipt 与
ledger replay/eval audit 一致后，才运行：

```bash
assert_release_state

"$ISAAC_PY" scripts/run_stair_final_confirmation.py \
  --session-id "$SESSION_ID" \
  --output-dir "$OUT" \
  --final-confirmation-plan "$PLAN" \
  --execution-device "$DEVICE" \
  2>&1 | tee "$OUT/final_confirmation.log"

test -s "$OUT/final_confirmation/reservation.json"
test -s "$OUT/final_confirmation/result.json"
```

该入口会从真实 ledger 重建 acceptance replay/eval audit，验证 15 轮闭合 live receipt、
Genesis 与最终 current-best checkpoint、后端环境和源码 manifest，然后才对预承诺 cohort
按 Genesis -> final 顺序做一次配对确认。结果仅用于最终诊断，不回流到 acceptance、prompt、
history 或 memory。

如果最终确认在写入 `reservation.json` 后失败，reservation 会故意保留。不要删除它或直接
重跑来规避 single-use 语义；保留全部证据，先完成根因审计并决定是否需要新的完整正式实验。
