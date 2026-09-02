# Phase 8E-6-4 capped multi-round self-evolution loop

日期：2026-06-27

## 目标

把单轮 `run_phase8e_full_round` 串成一个有上限的自演进循环：默认跑满 3 轮停止，先观察效果。每轮 LLM 看到累积的公开反馈与已接受 recipe，写新候选，从随机策略重训、可信评测、决策、入账。

## 关键修复：provider 没拿到共享 lab_dir（自演进反馈断链）

实现循环时发现一个真 bug：`phase8e_full_round._prepare_candidate` 调 provider 时**没传 `lab_dir`**，于是 `run_phase8e_provider_candidate` 用默认 lab_dir（= run_dir/provider_candidate，每轮都是空的）构造 prompt。结果：

- ledger / accepted-recipe 累积在共享 lab_dir（linked 步骤写入），但
- 构造 prompt 的 provider 步骤读的是另一个空 ledger，

导致**第二轮起 LLM 看到的 history 为空、没有继承 recipe**——多轮"反馈→改动→重训"会变成每轮从零、互不影响的假自演进。

修复：`_prepare_candidate` 增加 `lab_dir` 参数并透传给 `provider_runner`；`run_phase8e_full_round` 把自己的 `lab_dir` 传进去。新增 full_round 回归：断言 provider 收到的 `lab_dir` == 传入的共享 lab_dir。

## 实现内容

- `src/humanoid_agent/cold_start_lab/phase8e_evolution_loop.py`
  - `run_phase8e_evolution_loop(...)`：顺序跑 `max_rounds` 轮（默认 3），所有轮共享一个 lab_dir；每轮 `append_ledger=True`、`inherit_if_accepted=True`。
  - 轮间天然串行：round N+1 的 provider prompt 继承 round N 的公开反馈 + 最新 accepted recipe；策略权重每轮 cold-start，只继承 recipe 代码。
  - reject/blocked 是正常结果、不终止循环；停机条件为 `max_rounds`，可选 `stop_after_no_improve`（连续 N 轮无新最优 public-accept 即停，best-effort 读取嵌套决策摘要，仅当显式开启时影响控制流）。
  - 聚合摘要 `phase8e_evolution_loop_summary.json`：每轮 status/accepted/exit_code/public_accept_mean/failure_classification + accepted_rounds + best mean + stop_reason。
- `scripts/phase8e_evolution_loop.py`：CLI，`--max-rounds`（默认 3）、`--stop-after-no-improve`、`--provider`、`--allow-api-spend` 等。
- `scripts/phase8e_evolution_loop_selftest.py`：循环跑满轮数、共享 lab_dir、各轮独立 run_dir、append/inherit、no-improve 早停、max_rounds=0 拒绝。
- `__init__.py` 导出。

## 边界保持

循环只是顺序调用 full round，**不改任何 runner-owned 边界**：每轮仍走 8D verifier → 固定预算隔离训练 → runner 解析 sealed checkpoint → trusted eval 写 public_result → 决策。预算/eval/checkpoint/metric 仍 runner 拥有，候选只写训练组件。

## 并行说明（GPU）

轮与轮**必须串行**（round N+1 继承 round N 的 accepted recipe）。单轮内训练是单卡（cuda:0，4096 env）。6 个公开 rollout（dev/accept seeds）目前在 cuda:0 上**顺序**跑、第二张卡闲置；它们相互独立，可后续选做双卡并行以缩短 rollout 墙钟，但相对 500-iter 训练收益有限。多卡训练（DDP）是更大改动，需重新标定吞吐，暂不做。

## 验收

```text
python scripts/phase8e_evolution_loop_selftest.py
python scripts/phase8e_full_round_selftest.py   # 含 provider lab_dir 回归
python -m py_compile src/humanoid_agent/cold_start_lab/phase8e_evolution_loop.py scripts/phase8e_evolution_loop.py scripts/phase8e_evolution_loop_selftest.py
```

## 跑法

```bash
source scripts/activate_environment.sh
source /data/yuyangliu/secrets/deepseek.env
python scripts/phase8e_evolution_loop.py \
  --run-root /data/yuyangliu/runs/phase_8_cold_start_autoresearch/evolution_001 \
  --max-rounds 3 --allow-api-spend
```
（provider 默认 deepseek；`--provider dry_run` 可零 API 验链路。每轮约 500-iter 真训练，建议挂 tmux。）
