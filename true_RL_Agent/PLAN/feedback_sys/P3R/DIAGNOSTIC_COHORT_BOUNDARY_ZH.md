# 独立 Stair public-dev 诊断与重放租约边界

日期：2026-09-29。状态：本地实现与针对性验证通过，尚未采集新的确认资料。测试使用合成数组/环境替身，不作为 R.3/R.4 回合。

## 已实现边界

- 在 `tasks/stair_climbing/seed_policy.py` 增加 `public_dev_diagnostic_v1`。base seed 固定为 170100；新 namespace 同时进入 runner seed 派生和 cohort hash，新 cohort 只承诺 public-dev。
- 新 CLI purpose 为 `public_dev_diagnostic`，只允许统一协议的 public-dev phase 且必须绑定 plan hash。public-accept、sealed、完整评估均拒绝。没有开放任意 seed，没有新 profile，旧 formal/pilot/final 算法和输出保留。
- 任务内 `diagnostic_evaluation.py` 的 `prepare_stair_public_dev_diagnostic` 从已绑定 checkpoint 的正式 eval plan 继承 subject、manifest、源码、training binding、capability、协议及计数，只重建 job/cohort binding；复用 backend materialization 和现有 runner。
- `project_stair_public_dev_diagnostic` 先用 `require_phase_result_matches_plan` 验证新 phase，再按新的预期 namespace 进行任务投影。它返回公开观察，不生成资格或接受决定。正式 assembly 仍要求正式 binding，不能把新诊断和旧 sealed 拼成新资格。
- 原 qualified anchor 的恢复/核验仍由调用方负责。本入口不补写历史资格，不改训练 manifest，也不把 ordinary candidate 改称 reference。

## 同一生产脚本的调用

原 `scripts/stair_numeric_dev_replay.py` 增加 `--cohort formal|diagnostic`，默认 formal 保留原 cohort、seed 与 source 身份。诊断不通过复制旧脚本或直接启动 Isaac 子进程实现。

先替换下列已有输入路径与新输出目录；本记录没有执行命令中的仿真：

```text
python scripts/stair_numeric_dev_replay.py --checkpoint EXISTING_CHECKPOINT --train-summary ORIGINAL_TRAIN_SUMMARY --protocol configs/tasks/stair_go2_12dof_unified_protocol_v3.json --output NEW_OUTPUT_DIRECTORY --device cuda:0 --isaac-python ISAAC_PYTHON --mode preflight --capture-numeric on --cohort diagnostic
```

正式采集沿用相同输入，将 `--mode` 改为 `run` 并使用另一个新输出目录。入口仍核对原完成训练回执、真实权重、祖先 manifest 与当前运行环境。采集启动及预算由本阶段运行卡另行管理，本实现不自动启动。

程序调用可直接取 `prepare_stair_public_dev_diagnostic(...)` 返回的 `EvalPhaseJobSpec`；新 plan 为其 `evaluation_plan`。通过已配置 GPU manager 的 runner 执行后，使用 `project_stair_public_dev_diagnostic(config=..., eval_plan=phase.evaluation_plan, phase_result=result)`。运行阶段需要真实 `attempt_scope`，脚本已构建现有 `PreGenesisExecutionScope`；不能在调用方另持一个覆盖整个 phase 的外层 GPU 租约。

## 发现并修复的外层租约

原重放脚本在环境探测前取得外层 GPU lease，到整个 `run_eval_phase` 完成后才释放。即使 runner 内已将 CPU 派生移到子进程退出后，这个外层 lease 仍会占住 GPU。

现在环境探测使用短租约并立即退出，phase 使用现有 PreGenesis attempt 和 runner 自己的 GPU lease；子进程结束、实际释放 lease 后才执行 CPU finalizer，最终 phase 和 attempt output hash 在派生后封存。不开自动重采，单次 attempt、零基础设施自动重试。report 将环境探测与 phase 租约耗时分开，phase 值取原 attempt receipt；缺失的 phase 耗时为 null，不能伪装为零。CPU 耗时继续由最终 phase 的 execution cost 保存。

## 验证与证据范围

1. 新诊断、原 entrypoint 与 phase 回归：42 项通过。新增检查对 G1/Go2 unified v1/v2/v3 及历史 Stair v2 的所有 formal/pilot/final runner seeds 和 cohort hashes，比对修改前的完整输出摘要，全部保持不变。
2. 新诊断、重放脚本及既有 CPU finalizer 接线：13 项通过。包括新 cohort 的 preflight、禁止改 seed/换 split/冒用旧 binding、保留原训练身份、正式 assembly 拒绝新 cohort。重放入口测试实际运行文件锁、PreGenesis journal 和 runner 调用顺序；环境探测及 rollout 被替身替换。CPU 回调内另一 job 可取得同一设备锁，且完成 attempt 尚未发布。
3. 随后新增的诊断数值档案单项通过：使用真实数组采集/封存函数和两步环境替身，沿新 namespace 生成档案，经过 `postprocessing → CPU finalizer → pass`，再删除临时档案并成功投影已封存 phase。没有另写行为派生规则。
4. `git diff --check` 对本次涉及文件通过。未修改 Stair/Push behavior 或公共 prompt。

可重跑命令：

```text
.venv/Scripts/python.exe -m unittest tests.unit.test_stair_diagnostic_evaluation tests.unit.test_stair_entrypoint_materialization tests.unit.test_stair_phased_evaluation -q
.venv/Scripts/python.exe -m unittest tests.unit.test_stair_diagnostic_evaluation tests.unit.test_stair_numeric_dev_replay tests.unit.test_phase_postprocessing -q
```

源文件 SHA-256：

| 文件 | SHA-256 |
|---|---|
| scripts/stair_numeric_dev_replay.py | b7a0d50b46eb69c389fe2a24a20a61b54c3768c504bc904959c1c4b00a1ce2ec |
| scripts/isaaclab_stair_rsl_rl_eval.py | c897d03436c95fba902adbcbdee064debe69a8487344fd7d36f02e2cc01dcb8e |
| tasks/stair_climbing/seed_policy.py | 366f7198a9f33b9c4fe18b2f8f93ba5082ead9e3d2f58af31c4f61f6c01f5aa8 |
| tasks/stair_climbing/entrypoint_contract.py | 6e7cb917d96aee40a2b4e5c12fe7d93827f09c33adfecd29290409417e273e4d |
| tasks/stair_climbing/diagnostic_evaluation.py | 64de35a99f7b69d2a6ed76de638cbdac0501feb9defd5aad799e291c4a1d5ec1 |

此诊断 namespace 只提供一批固定的新采集。再次使用同 checkpoint 和该 namespace 是重复测量；R.3/R.4 若共用，必须在观察确认结果前共同冻结。新样本只支持同策略、同地形协议的新初态/运行实例，不是新策略谱系或新地形泛化；真实采集后还须核验初态与数据确实变化。
