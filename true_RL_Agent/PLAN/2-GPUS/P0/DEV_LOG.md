# P0 开发日志

> 本页前半段记录 2026-08-15 初次只读基线审计；后半段“判重根因纠偏与最终验收”是后续复审后的增量记录。后者取代前者关于 repeat 根因、测试基线和文档哈希的旧结论，但不改写初次审计事实。

## 2026-08-15

### 初次基线审计（纠偏前）

### 范围

- 完整重读双卡主计划并提取 P0 出口门；
- 使用 CodeGraph 核对 `true_RL_Agent` 当前 round/ledger/provider/repeat/lifecycle 边界；
- 使用 CodeGraph 核对 `D:\humanoid_agent\codex` 的 process ownership、首错保留和 fault-injection 实现模式；
- 只读检查服务器正式 session、ledger、15 个 train/eval summary、live invocation、GPU/process 和 final-confirmation 目录；
- 生成 P0 实施清单、基线/协议冻结报告，并更新主计划基线状态。

该初次审计阶段未做：生产代码修改、服务器写入、GPU/API 调用、final-confirmation 执行、P1–P6 实现、commit/push。随后根据用户提供的真实反例，仅对 verifier 判重根因进行了 P0 级源修，见下方增量记录。

### 保护现有工作

P0 开始前已有且未被改动：

- `scripts/run_stair_final_confirmation.py`
- `tests/unit/test_stair_final_confirmation_entrypoint.py`
- 用户已有 outputs、压缩包和 `uv.lock`

### 关键结论

1. 本地、origin、服务器代码均为 `e8b1e9e80f4b2a12af993b691cb8a76692cc3dfd`。
2. 正式 width=1 live 已 15/15 闭合，唯一 invocation completed，最终 head 为 R3；没有 infra failure。
3. final-confirmation 尚未执行；服务器入口的 checkpoint 路径缺口与本地用户修复严格分开记录。
4. 当前 runner 没有 train/eval job retry，仍由 blocking `subprocess.run` 执行；属于 P1，不在 P0 表面加补丁。
5. Stair value 只按公开、context-bound evidence 派生；repeat identity 继续由 verifier 单源拥有。本条的旧 repeat 粒度结论已由下方根因纠偏取代。
6. A-first/B-second、B 只看 verifier-owned exclusion、temperature 0.2、reasoning adaptive 已作为未来 generation 合同草案冻结；exclusion 的精确语义以下方根修后的单组件比较为准。
7. 历史产物没有持久化 GPU/RAM 峰值；该缺口留给 P1 receipt/P6 pilot，不用估计值补写。

### 验证

```text
Python 3.12.13
python -m unittest discover -s tests
Ran 1153 tests in 244.314s
OK (skipped=20)

python -m compileall -q src scripts tests
PASS

git diff --check
PASS
```

此前同一当前代码工作树的 P0 定向集合：

```text
Ran 215 tests
OK (skipped=1)
```

测试中的 SimulationApp close、summary publication、backend/source drift 文本来自预期故障注入，最终退出码为 0。

### 文档冻结哈希

| 文件 | bytes | SHA256 |
|---|---:|---|
| `../dual_gpu_generation_upgrade_plan_zh.md` | 85,669 | `683979fbf07fd40721dea2b1b12df92039ae0306e2b48cc477985434aa68fc21` |
| `P0_IMPLEMENTATION_PLAN_ZH.md` | 5,291 | `d7451b3a64a55cda4f3fb6eb81a3cd8a27e9ba9232dfcab2b61178cb322f7e0a` |
| `P0_BASELINE_AND_PROTOCOL_FREEZE_ZH.md` | 14,955 | `a371db6424f3396450dc3b166bb393779655e754f6c0b081c638a35530871b8b` |

这些哈希冻结的是本次 P0 handoff 内容。任何后续修改必须重新生成哈希并在新日志条目中说明，不能静默覆盖基线。

### 出口状态

P0 PASS，等待用户检查。未经用户明确批准，不进入 P1，不同步服务器，不运行终局诊断或新实验。

## 2026-08-15：判重根因纠偏与最终验收

### 触发证据与结论

用户用生产函数和真实 R1/R4/R8/R11 source 证明了两条独立漏口：

1. 整份 candidate source 对历史 source 比较会因两者 authoring anchor 不同而把单机制重复看成 compound delta；
2. 以组件名为 key 的比较可被 term/function 重命名绕过，即使 source 与 anchor 都已完整提供。

因此“主要是 supplied evaluated sources 有界且不完整”的旧根因被撤回。历史 evidence 的有界覆盖仍是 P5 要解决的信息可达性问题，但不能替代 verifier 粒度修复。

### 源头修改

- `component_source_semantics.py`：复用现有 AST/component inventory 流水线，新增唯一的“各自相对 own anchor 的单组件语义”投影与等价比较；组件/函数名不作为语义身份，保留 kind、变化后存在性、输入依赖、callable/clip/字段结构，并额外形成 numeric-parameter-normalized signature；
- 历史 identical/parameter-only 重复继续由现有 anti-degenerate gate blocking，结果仍为 `historical_mechanism_repeat`，未新增第二个判重系统；
- rejected history 的 same-input structural variant 改为跨 parent advisory，不再用 same-parent 窗口遗漏信息，也不把 advisory 升格为历史 blocking；
- `authoring_loop.py`：从每条已验证、不可变的历史 prompt payload 投影其真实 authoring anchor；旧记录缺 ref 时仍走既有 legacy fallback；
- 暴露同一个 task-neutral `classify_single_component_delta_equivalence`，供未来 P4 A→B blocking exclusion 复用；P0 没有实现 coordinator 或 sibling lane；
- 集成 fixture 原先用“同公式只改数值常量”模拟不同 round，新门正确将其视为重复；fixture 改用三种真正不同的运算结构，未放宽 verifier。

同类路径扫描没有发现第二个 blocking 判重实现：历史 blocking 仍只经 `_assess_against_evaluated_history`；experiment memory/runtime component identity 只做证据投影，不自行判断语义重复；frontier 仍未实现。旧记录缺失可验证 prompt anchor 时保留 legacy full-source fallback，只用于兼容，不能作为 generation sibling exclusion。未新增 Stair 字段、cell 或失败语义到公共 verifier。

生产只读对照结果：

- R1 `goal_progress` 与 R8 `goal_progress_reward`，分别相对 reference/R3 anchor：`parameter_only_delta`；
- R4 force-sum 与 R11 force-norm，分别相对自身 anchor：`None`（结构不同），并保留 same-input advisory；
- R8 是本次可证明的实际重复浪费，约 73 分钟；R11 不是重复。R14/R15 不再在缺少同等级 component-relative 证据时并列归因。

### 计划与成本边界同步

- P4 完成生产 exclusion 后、P6-B 前增加 15-prompt author-only paired-ready 探针；必须用户单独批准 API/金额，不用 GPU、sealed、ledger 或 temperature sweep；
- `64 GiB` 只保留为 P6-A 待测暂定值，P0 未声称已测 RAM 门；
- `I=300,000` 与 `C_generation/C_session` 只是保守 reserve 草案；P1 tokenizer/preflight 完成后按真实 usage 与最大合法 prompt 重新校准；
- 未调用 provider API，未启动 GPU/新实验，未执行 final-confirmation，未写服务器，未 commit/push。

### 最终验证

```text
python -m unittest \
  tests.unit.test_component_smoke_and_antidegenerate \
  tests.unit.test_candidate_authoring_loop
Ran 81 tests in 25.761s
OK

prompt/replay/Stair/Push compatibility set
Ran 127 tests
OK (skipped=2)

python -m unittest discover -s tests
Ran 1155 tests in 176.416s
OK (skipped=20)

python -m compileall -q src scripts tests
PASS

git diff --check
PASS
```

身份说明：

- 干净 `e8b1e9e`：`1150/20`；
- P0 入场工作树（另含用户 final-confirmation 修复及 3 个测试）：`1153/20`；
- 当前 P0 集成候选（再含 verifier 根修及 2 个测试）：`1155/20`。

三者不得互相替代。当前候选尚未 commit，因此仍不是可由单一 commit checkout 复现的 release 基线；等待用户验收后再决定版本化提交与后续同步。

### 当前集成候选文件哈希

下表用于把当前未提交集成候选与基础 commit 分开复核。前两项是 P0 入场前已有的用户 final-confirmation 修复；其余代码/测试是本次判重根修，三份 Markdown 是纠偏后的 P0 handoff。`DEV_LOG.md` 不自哈希。

| 所属 | 文件 | bytes | SHA256 |
|---|---|---:|---|
| 用户已有 | `scripts/run_stair_final_confirmation.py` | 18,445 | `016edc439b3c8d52d0031da1cc9125bf296369f9147681628a52f34828ec6559` |
| 用户已有 | `tests/unit/test_stair_final_confirmation_entrypoint.py` | 33,495 | `11dda25de807d2eb3697ed259a098e550c665d4d3ab3458dd9c27d670dcd5422` |
| P0 根修 | `src/true_rl_agent/agent_core/authoring_loop.py` | 61,105 | `935cecc6cec2fcc227263a08ed8419e2c61b5fcbf482e1ab5817e908eb175cc4` |
| P0 根修 | `src/true_rl_agent/candidates/authoring_checks.py` | 5,329 | `e198fbf2143dd5b1ac116349f2407bd9a6c258e1a86084f75c5b55968490d7ea` |
| P0 根修 | `src/true_rl_agent/candidates/component_source_semantics.py` | 62,310 | `9c6e33b23aa6492f4d644cbc96b9949d209a51872fb344bc5e9138942fc67711` |
| P0 根修测试 | `tests/unit/test_candidate_authoring_loop.py` | 109,771 | `5f02e4f3051594291cd63911d59a4d21b29c14745aa1568d5f3d0393c68d189a` |
| P0 根修测试 | `tests/unit/test_component_smoke_and_antidegenerate.py` | 71,602 | `85081add0b8f34fe77110915756a2ee1be4b6c15e79d5ba4f58a6764f6acd9da` |
| P0 兼容测试 | `tests/integration/test_session_genesis_live_loop.py` | 40,135 | `85d7008c7f31076171a1f21d1ebb48e336c062054d033c8f42f564d15ecb4a50` |
| P0 文档 | `PLAN/2-GPUS/dual_gpu_generation_upgrade_plan_zh.md` | 89,452 | `73348e6f49a50e10bf1605d120c3d3595a153c56b02734f9cf23050e4f7ebd1c` |
| P0 文档 | `PLAN/2-GPUS/P0/P0_IMPLEMENTATION_PLAN_ZH.md` | 7,294 | `823d350a5799e670b6509b18b713e1e4312d24b09b7e0deb92ae5733ec9d3902` |
| P0 文档 | `PLAN/2-GPUS/P0/P0_BASELINE_AND_PROTOCOL_FREEZE_ZH.md` | 18,540 | `90d93834c0b873642d842a52b94cb549d56fd5d118898aa640f9e29314db9c16` |

### 出口状态

P0 根因纠偏与本地验收 PASS。仍等待用户检查；未经用户明确批准，不进入 P1，不执行 author-only 探针、final-confirmation 或任何新实验。

## 2026-08-16：数值结构边界、历史覆盖与 sibling API 收口

### 逐项判定

1. **fn 数值归一化误伤 tensor 语义：真实，阻断级，已源修。** 旧实现把 callable 内所有数值常量都归零，确会把 `[:,2]`/`[:,0]` 和 `sum(dim=1)`/`sum(dim=2)` 判为 parameter-only。修复仍使用同一 AST signature 流水线：保留 subscript/slice 以及 reduction/shape 调用的维度/轴配置；普通算术、`.mul(...)` 缩放、`clamp(...)` 阈值和外层 `weight/clip/value` 继续归一化。没有加入 Stair 字段表或轴含义表。
2. **缺历史 anchor 的 legacy 回退静默：真实，可观测性缺口，已收口。** 每份 anti-degenerate evidence 现在无论命中与否都记录 `evaluated_history_source_count`、`history_sources_with_anchor`、`history_sources_without_anchor` 和 `candidate_authoring_anchor_available`。prompt payload 缺失或不可读时仍保留旧 full-source 兼容回退，但 coverage 降级可直接审计，且该回退不得证明未来 sibling distinct。
3. **公开/内部单组件 API 失败语义不一致：真实，P4 前必须明确，已收口。** 对外唯一入口改为 `compare_single_component_deltas`；只有两侧都可投影为恰好一个 delta 时才返回 equivalence 或 `None`（结构不同），否则统一抛 `SingleComponentDeltaComparisonUnavailable`。P4 已冻结为 B lane fail-closed/repair，不能把 unavailable 当作 distinct。
4. **非-fn 字段签名重复代码：真实但非阻断，已作小型同源清理。** exact 与 parameter-normalized 两条路径统一调用 `_component_field_signature`；没有新建模块、规则表或第二套分析器。

### 同类审计边界

- blocking history repeat 仍只有 `_assess_against_evaluated_history`；future sibling exclusion 复用同一公开原语；frontier/memory 只能引用结果；
- history 比较仍允许 legacy fallback 以读取旧记录，但新的 generation sibling 路径没有 fallback，compound/parse/结构不可用一律不可训练；
- 保留 positional tensor 维度的集合只描述通用 reduction/shape API，不含 Stair 输入名、cell、失败原因或 reward 语义；
- 对未解析的更一般程序等价不作承诺，避免把静态 verifier 扩成全程序证明器；
- 旧 Push、Stair admission、prompt/replay 与当前 round-v7 路径均由全量回归覆盖，没有改 eval-v6、20-step success、acceptance、sealed 或 ledger 语义。

### 验证

```text
python -m unittest \
  tests.unit.test_component_smoke_and_antidegenerate \
  tests.unit.test_candidate_authoring_loop
Ran 82 tests in 29.074s
OK

prompt/replay/Stair/Push compatibility set
Ran 127 tests in 7.002s
OK (skipped=2)

python -m unittest discover -s tests
Ran 1156 tests in 177.563s
OK (skipped=20)

python -m compileall -q src scripts tests
PASS

git diff --check
PASS
```

测试身份仍分三层：干净 `e8b1e9e=1150/20`；P0 入场工作树（含用户 final-confirmation 修复）`1153/20`；当前未提交 P0 集成候选 `1156/20`。本条替代 2026-08-15 条目中的 provisional `classify_single_component_delta_equivalence` 名称与 `1155/20` 候选数字，但不改写那次历史记录。

### 当前集成候选文件哈希（取代上一张候选表）

| 所属 | 文件 | bytes | SHA256 |
|---|---|---:|---|
| 用户已有 | `scripts/run_stair_final_confirmation.py` | 18,445 | `016edc439b3c8d52d0031da1cc9125bf296369f9147681628a52f34828ec6559` |
| 用户已有 | `tests/unit/test_stair_final_confirmation_entrypoint.py` | 33,495 | `11dda25de807d2eb3697ed259a098e550c665d4d3ab3458dd9c27d670dcd5422` |
| P0 根修 | `src/true_rl_agent/agent_core/authoring_loop.py` | 61,105 | `935cecc6cec2fcc227263a08ed8419e2c61b5fcbf482e1ab5817e908eb175cc4` |
| P0 根修 | `src/true_rl_agent/candidates/authoring_checks.py` | 5,401 | `a18f57c25c4d35f5c6b19bdc089435126dd8316ba3e44d1e90967d96196cc722` |
| P0 根修 | `src/true_rl_agent/candidates/component_source_semantics.py` | 65,836 | `101798286d84a4eadacbbb70d82dc321383f0a60feeb719fe2aa03b96e5c3128` |
| P0 根修测试 | `tests/unit/test_candidate_authoring_loop.py` | 110,405 | `e6223c955ace7c1fbd032f869537f17b6536adf71c85f84a01f0487bdf863e67` |
| P0 根修测试 | `tests/unit/test_component_smoke_and_antidegenerate.py` | 80,565 | `535d0ee92a0c7ebcec689b3a53b826b0bb5608c50ba73d60595c01324756cf07` |
| P0 兼容测试 | `tests/integration/test_session_genesis_live_loop.py` | 40,135 | `85d7008c7f31076171a1f21d1ebb48e336c062054d033c8f42f564d15ecb4a50` |
| P0 文档 | `PLAN/2-GPUS/dual_gpu_generation_upgrade_plan_zh.md` | 90,900 | `64b503fe917908a36eb9e2aec408f1f4e9e6c5f5cf66ec990028ebd73b2e0cf3` |
| P0 文档 | `PLAN/2-GPUS/P0/P0_IMPLEMENTATION_PLAN_ZH.md` | 8,134 | `10c7dc23505db70bb44949d09c50dd62ae6c9e9d4e813188da70fe670a58bf73` |
| P0 文档 | `PLAN/2-GPUS/P0/P0_BASELINE_AND_PROTOCOL_FREEZE_ZH.md` | 19,937 | `e67e001fc6ad27e9d71aae931c22e4dd6952cf07672a5f39a8775410787c246e` |

### 出口状态

P0 本地实现与验收仍为 PASS，等待用户检查。未调用 provider API，未启动 GPU、服务器作业、final-confirmation 或新实验，未 commit/push；未经用户批准不进入 P1。
