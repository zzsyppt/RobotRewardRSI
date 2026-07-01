# 2026-07-01 相对归因与课程主执行面修复

## 背景

一次真实 DeepSeek 15-round 试跑暴露出两个相关问题：

- Prompt 仍把最显眼的可执行面写成 `RewardTerm` / `ScheduleTerm` / `DiagnosticTerm`，并明确提示候选可以通过 `RewardTerm` 和 `reward_weight_scale` alone 改进。
- `curriculum_lineage.public_summary.attribution` 使用的是“候选是否包含课程”的绝对视角，而不是“相对 current_recipe 是否真的改变课程”的视角。

因此在 round_004 中，候选实际修改了 reward/schedule 代码，但保留相同课程结构时，lineage 仍可能报告：

- `reward_code_changed=false`
- `curriculum_changed=true`
- `gain_attributable_to=curriculum`

这会让系统把 code-level reward tweak 误当成课程收益，正好对应“课程设计还不是 prompt 里的一等动作面”的问题。

相关试跑路径：

`/data/yuyangliu/runs/phase_8_cold_start_autoresearch/llm_curriculum_v1_deepseek_15round_20260701_121202`

该试跑已停止，避免继续产生误归因样本。

## 修复顺序

1. 给 curriculum compiler 增加相对 current_recipe 的归因层。

   该层比较 candidate 与 prompt 中 current_recipe 的：

   - `training_code_tree_sha256`
   - 规范化后的 `operation_diff.json.curriculum_design`
   - `diagnostics.json`
   - stage capability signature

   如果课程效果未变但组件代码变了，公开 attribution 会报告 `reward_code_changed=true`、`curriculum_changed=false`、`gain_attributable_to=reward_code`。

2. 规范化课程比较。

   `stage_list` / `stages`、`fraction` / `min_fraction` / `min_fraction_of_training`、target push / robustness aliases 会被归一到同一个 effect signature，避免同义 schema 形式导致 false curriculum delta。

3. 把 prompt-facing 主执行面改成 runner-compiled CurriculumDesign。

   Prompt 现在明确：

   - `operation_diff.json.curriculum_design` 是 first-class runner-compiled training surface。
   - `candidate_allowed_controls` 首项是 CurriculumDesign。
   - `reward_weight_scale` 是辅助 schedule key，不是 curriculum design 的替代品。
   - provider JSON contract 要求 `operation_diff.json` 包含 `curriculum_design`，并禁止把 reward-only / reward_weight_scale-only tweak 表述成 curriculum improvement。

4. 加回归测试。

   新增测试覆盖：

   - 同课程 + reward code 变化必须归因到 reward code。
   - 同课程但使用 schema aliases 时不得误判为课程变化。
   - prompt 中不再出现 `reward_weight_scale alone` 作为主机制。
   - prompt contract / allowed controls / provider rules / self-audit checklist 都把 CurriculumDesign 作为一等面。

## 本地验证

已通过：

```powershell
python scripts\phase8e_curriculum_compiler_selftest.py
python scripts\phase8e_selftest.py
python scripts\phase8e_full_round_selftest.py
python scripts\phase8e_provider_ingest_selftest.py
python -m py_compile src\humanoid_agent\cold_start_lab\phase8e_prompt.py src\humanoid_agent\cold_start_lab\curriculum_compiler.py scripts\phase8e_selftest.py scripts\phase8e_curriculum_compiler_selftest.py
git diff --check
```

## 后续试跑审计点

重启 15-round 后必须重点检查：

- round_001 prompt 的 `candidate_allowed_controls` 是否把 CurriculumDesign 放在主执行面。
- `operation_diff.json.curriculum_design` 是否仍由 LLM 真实设计多阶段课程，而不是只改 reward。
- 如果后续 round 只改 reward/schedule 且课程效果签名不变，lineage 必须报告 `curriculum_changed=false`。
- public history 仍不得泄漏 private accept seed、runner path、hidden decision internals。
