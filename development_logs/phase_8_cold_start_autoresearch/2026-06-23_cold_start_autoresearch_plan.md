# Cold-start ENPIRE-like 代码沙盒改造计划

日期：2026-06-23

## 0. 本次重构结论

这版计划的核心变化是：下一步直接进入 ENPIRE-like 的最小代码沙盒，但必须把“代码权限”限定为训练组件权限，而不是 runner 权限。

换句话说：

```text
candidate = components
runner = loop + budget + eval + checkpoint + accept/reject
```

候选可以提供 reward、loss 项、schedule、observation transform、diagnostics 等组件；候选不能驱动训练循环、不能调用 `optimizer.step()`、不能分发 env step、不能选择 checkpoint、不能运行 eval。

原因：

- `config-only` 对当前任务存在隐含 executor 依赖，例如 curriculum schedule / delayed push / magnitude ramp 不是纯 YAML 能自动生效；
- cold-start baseline 可能接近 0，传统 `candidate - baseline > threshold` 的 accept/reject 规则不稳定；
- code-level 权限真正的风险不是“恶意 Agent”，而是 metric hacking、budget hacking、runner 绕行和 eval-detection；
- 静态 grep、AST、canary 只能做次级防线，主防线必须是 OS/进程/文件系统隔离和不可变 runner；
- 如果没有 accepted recipe 继承，系统只是安全 sampler，不是 ENPIRE-like evolver。

本计划仍保留当前 Phase1-7 的成果，但新链路和旧链路严格隔离。

## 1. 实验目标与诚实边界

建立一条独立的 cold-start autoresearch 代码沙盒链路：

- policy 从随机初始化开始；
- 不 warm-start 当前已能站立的 checkpoint；
- 不把 Phase2/Round3 accepted 配置当默认训练起点；
- LLM Agent 可以修改训练侧组件代码；
- 外部 immutable runner 固定预算、评估、seed、checkpoint selection 和 accept/reject；
- public feedback 可进入闭环；
- hidden / iid / audit 数据绝不进入 Agent prompt；
- 先完成 G1 standing push recovery 一个任务，不提前泛化到大量任务。

必须诚实表述实验目标：

> 本实验不是证明 LLM “tabula-rasa 地发现先站立再抗 push”。LLM 预训练中本来就可能知道 locomotion curriculum / standing-first / perturbation training 等常识。实验真正要验证的是：Agent 能否在固定外部裁判下，自主组合、实现、训练并验证一个可复用 recipe，并且这个 recipe 能在后续轮次中继承和累积改进。

## 2. 与现有 Phase1-7 的关系

| 内容 | 当前 Phase1-7 | 新 cold-start 代码沙盒 |
|---|---|---|
| 环境与服务器 | 复用 | 不重装 |
| Isaac/G1 overlay | 可复用环境封装 | 不把成功配置作为默认起点 |
| Phase2 baseline | 作为历史参考 | 不作为 cold-start 起点 |
| Phase3 verifier | 复用“规则文件需从可信 base_ref 读”的教训 | 不能直接承担 code-level verifier |
| Phase5/6 prompt/provider | 可复用 API/provider 管理 | prompt schema 重写 |
| Phase6 public-only 逻辑 | 复用防泄露原则 | 新增 public_dev/public_accept/OOD stress split |
| Phase7 budget calibration | 复用 4000-iteration 经验 | 转换成 env-step 预算候选，不直接固定 iteration |

旧链路不删除、不覆盖、不复写结论。新链路用独立目录、独立 run root、独立候选包。

建议新目录：

```text
src/humanoid_agent/cold_start_lab/
src/humanoid_agent/code_sandbox/
src/humanoid_agent/immutable_runner/
configs/cold_start/
development_logs/phase_8_cold_start_autoresearch/
```

服务器 run root：

```text
/data/yuyangliu/runs/cold_start_autoresearch/
```

## 3. 最重要的设计边界

### 3.1 “不让裁判作弊”不等于“不能作弊”

在仿真里，即使 Agent 完全不能修改 budget/eval/seed/metric/runner，也仍可能学到“裁判可接受但物理上不合理”的策略。例如：

- 关节很硬、能耗极高；
- 姿态卡在几何阈值内但动作不可用；
- 对 public push 分布过拟合；
- 利用仿真实现细节通过 geometry success；
- 用异常高频抖动或死撑策略过 gate。

因此本计划不再声称 “provably no cheat”。更准确的目标是：

> 通过外部不可变 runner、分布外 stress gate、运动合理性 gate 和一次性 hidden audit，显著提高 reward/metric hacking 的难度，并让失败可审计。

### 3.2 裁判分层

评估分为四层：

1. `public_dev`
   - 给 Agent feedback；
   - 用于开发和候选选择；
   - 不能单独决定最终 accept。

2. `public_accept`
   - 不进入下一轮 prompt；
   - 用于正式 accept/reject；
   - 与 public_dev 分离，降低 winner's curse。

3. `ood_stress_gate`
   - 参与 accept；
   - 候选看不到具体分布参数；
   - 覆盖不同 push 幅度、方向、interval、轻微地形/物理扰动；
   - 目标是让 geometry-threshold 的退化策略更难通过。

4. `hidden_audit_once`
   - 永远一次性；
   - 只做最终报告；
   - 不参与多轮 gate；
   - 不进入 prompt、ledger 或候选反馈。

### 3.3 运动合理性 gate

当前 energy / action-rate / foot-slip / oscillation 等指标不能只当 diagnostics。对于 accepted candidate，需要增加运动合理性 gate。

建议 gate：

- energy proxy 不得超过合理上界；
- action L2 / action-rate L2 不得超过合理上界；
- foot slip 不得超过合理上界；
- oscillation proxy 不得异常；
- 轨迹不能出现明显“死撑/抽搐”模式；
- 若能导出短视频或 trajectory summary，应做 report-only 人工抽查。

这些不改变 success metric 本身，但作为 accept gate。也就是说：

```text
success metric = 几何任务成功
accept = success metric 达标 + 运动合理性 gate 通过 + OOD stress gate 通过
```

## 4. 核心架构

### 4.1 不可变外部 runner

runner 是裁判，Agent 不可改。它负责：

- 创建候选运行目录；
- 注入固定 task spec；
- 固定 train seeds；
- 固定 eval seeds；
- 固定 env-step budget；
- 固定 wall-clock safety cap；
- 固定 grad/update/optimizer-step cap；
- 固定 checkpoint selection；
- 固定 public_dev / public_accept / ood_stress / hidden_audit split；
- 收集 TensorBoard 和 JSON summary；
- 执行 accept/reject。

Agent 只提交 candidate package，不直接调用 eval。

### 4.2 候选代码包

候选包是 Agent 的唯一写入目标。

```text
runs/cold_start_autoresearch/<run_id>/candidates/candidate_001/
  candidate_manifest.yaml
  training_code/
    rewards.py
    schedules.py
    observations.py
    losses.py
    diagnostics.py
  rationale.md
  operation_diff.json
```

候选包只能通过固定 API 被 runner 加载，例如：

```python
def build_training_components(task_spec, base_env_cfg, base_agent_cfg):
    return TrainingComponents(
        reward_terms=[...],
        loss_terms=[...],
        schedules=[...],
        train_observation_transforms=[...],
        diagnostics=[...],
    )
```

禁止候选提供：

- runner hook；
- PPO loop hook；
- env-step dispatcher；
- optimizer step hook；
- checkpoint selector；
- eval wrapper；
- success classifier；
- process/network/file-system control。

### 4.3 observation/action 接口冻结

这是一个隐藏执行漏洞：如果候选改了 observation wrapper/history，训练出的 policy 在 eval 时也依赖候选 observation 代码。这样候选代码就会在裁判过程中执行，可能根据 seed/env/profile 识别自己正在 eval 并作弊。

因此第一版代码沙盒采用更保守规则：

- eval observation/action 接口不可变；
- policy 输入维度和语义由 runner 固定；
- 候选不能改变 eval 期 observation space；
- 候选 observation transform 只允许作为 training-only augmentation 或固定、无状态、无条件分支的 transform；
- eval 期不执行 LLM-authored observation code，除非后续已有 OS 隔离 + 条件分支限制 + transform freezing。

第一版优先开放 reward、schedule、loss、diagnostics；observation/history 改动后移或仅做 training-only augmentation。

### 4.4 OS 级隔离是主防线

代码级 verifier 不能只靠静态检查。主防线应是运行隔离：

- 候选代码在独立进程运行；
- repo 以只读方式挂载；
- hidden / iid / audit 数据不挂载进候选进程；
- 候选目录是唯一可写目录；
- 候选进程无网络；
- 候选进程环境变量白名单；
- `DEEPSEEK_API_KEY` 等密钥绝不进入候选环境；
- 限制 wall-clock；
- 限制 total env steps；
- 限制 grad/update/optimizer steps；
- 限制输出文件大小；
- runner 在候选进程外部选择 checkpoint 和执行 eval。

网络分层：

- orchestrator / LLM provider 层可以联网；
- candidate training/eval 进程必须断网；
- secrets 只存在 orchestrator 层，不进入候选进程。

### 4.5 可信规则来自 base_ref

沿用 Phase3 的教训：verifier 自己的规则和 code verifier 规则不能从 Agent 可写 worktree 读取。

要求：

- verifier 规则从可信 base_ref 或安装包读取；
- 候选 diff 即使修改 verifier/config，也不影响当前验证规则；
- protected files 缺失、重命名、删除必须报错；
- code verifier 自身不能被候选代码 import shadowing 或路径劫持。

### 4.6 public-only 输出断言

以下候选产物如果要进入下一轮 prompt 或 ledger，必须过 public-only assertion 和 canary 检查：

- `rationale.md`
- `operation_diff.json`
- `diagnostics.json`
- training summary；
- eval summary；
- feedback summary；
- ledger entry。

这些文件不能被视为天然安全输出，因为它们是候选或候选相关流程产生的文本。

## 5. Phase 安排

### Phase 7A：收口当前预算与 checkpoint 缺陷

工作量：0.5-1 天。

目标：

- 正式修复 checkpoint selection bug；
- 固化 4000-iteration calibration 结论；
- 不改变 Phase6 默认配置；
- 保持本地和服务器 repo 干净。

必须完成：

1. 把 checkpoint selection 抽到 immutable runner 公共函数。
2. 按 `model_<iteration>.pt` 数字 iteration 选最大，而不是字符串排序。
3. 增加测试：
   - `model_900.pt` vs `model_1999.pt` 选 `1999`；
   - 无 checkpoint 返回 fail；
   - 非标准命名不误选。
4. 记录 4000 calibration：
   - seed 101/102/103 全部完成；
   - 约 600-800 iteration 学会基本站住；
   - 约 2000 iteration timeout / bad_orientation 基本稳定；
   - 2000-4000 reward 仍继续提升，但 success 相关指标在约 2000 后已接近饱和。

验收：

- checkpoint 测试通过；
- dev log 完成；
- repo 干净。

### Phase 8A-min：最小代码沙盒 MVP

工作量：1-2 天。

目标：

直接建立最小 code sandbox，不先做完整 config-only loop。

必须完成：

1. 新建 candidate package schema。
2. 新建 sandbox loader。
3. 新建 immutable runner 最小版本：
   - 固定 seed；
   - 固定 env-step budget；
   - 固定并行 env 数，Phase 8A 目标值为 4096；
   - 4096 必须通过服务器 1-iteration train-start smoke 容量验收后才作为默认执行值；
   - 固定 checkpoint selection；
   - 固定 TensorBoard/log 输出；
   - 暂时只跑 no-op candidate 和 train-start smoke。
4. 写一个 no-op demo candidate：
   - 由我们手写，不是 LLM-authored；
   - 不改变训练行为；
   - 只证明候选代码能被加载；
   - 只证明 runner 可以调用候选接口。
5. 候选代码不得覆盖主线文件。

硬门：

- 在 Phase 8C 隔离通过前，不允许运行任何 LLM-authored candidate code；
- 8A-min 只能运行我们手写的 no-op/demo candidate。

暂不做：

- 多任务 registry；
- 多候选池；
- Git evolution；
- 复杂 public_accept；
- 完整 LLM 代码生成。

验收：

- no-op candidate 能通过 sandbox loader；
- runner 能启动一次 smoke；
- candidate 失败时 runner 能 reject；
- 旧 Phase6/7 脚本不受影响。

### Phase 8B：immutable runner 与预算公平性

工作量：2-4 天。

目标：

把预算从“固定 4000 iterations”改为“固定环境交互预算 + 次级计算预算 cap”。

设计判断：合理，但必须限定在 runner 层预算 accounting / cap enforcement。
Phase 8B 不能宣称解决候选代码隔离，也不能把 4096 envs 的 1-iteration
smoke 当作长训容量证明。OS/进程隔离仍是 Phase 8C，LLM-authored candidate
仍不能运行。

主公平单位：

```text
total_env_steps
```

初始 env-step budget 可从 4000-iteration calibration 换算：

```text
env_step_budget ~= 4000 * 512 * steps_per_iteration
```

上述公式来自 Phase 7A 的 512-env calibration，只用于估算同等环境交互量。
Phase 8A 的执行目标是 4096 并行 envs；进入 Phase 8B 后应固定
`total_env_steps`，而不是固定 iteration 数。若 steps/iteration 不变，
4096 envs 下同等 env-step budget 对应的 iteration 数约为 512-env 设置的
1/8。

具体 `steps_per_iteration` 由 runner 从实际训练循环中记录，不靠手写常数猜。

次级 cap：

- wall-clock：只作为安全 cap，不作为公平主指标；
- total grad steps；
- total optimizer steps；
- total policy updates；
- total forward/backward 调用数，如可观测；
- checkpoint 数量和大小。

为什么不能只固定 iterations：

code-level 下不同 recipe 每个 iteration 做的事情可能不同。有的带 aux loss，有的多次 inner update，有的改 batch 结构。固定 iteration 不是公平计算单位。

必须完成：

1. runner 记录：
   - total env steps；
    - update count；
    - optimizer step count；
   - elapsed seconds；
   - GPU id；
   - checkpoint list。
2. 超 cap 直接 fail/reject。
3. eval checkpoint 由 runner 外部选择。
4. candidate summary 中写入 budget accounting。
5. budget 是否足够应以 success 相关指标平台为主，而不是 reward 平台：
   - timeout；
   - bad_orientation；
   - public success；
   - OOD stress success；
   - 运动合理性 gate。

验收：

- 人造 candidate 试图多跑 update 会被 reject；
- 人造 candidate 试图多写 checkpoint 不影响 eval 选点；
- 预算 summary 可审计。

Phase 8B 最小实现落地标准：

- 新增 `configs/cold_start/phase8b_budget.yaml`；
- 以 Phase 7A calibration 估算初始 budget：

  ```text
  env_step_budget = 4000 * 512 * 24 = 49,152,000
  ```

- 在 `num_envs=4096`、`steps_per_iteration=24` 下派生：

  ```text
  max_iterations = 500
  planned_env_steps = 49,152,000
  ```

- runner summary 必须包含 `budget_accounting`；
- Phase 8B 的 timesteps 字段命名为 `config_derived_total_timesteps`，明确它是由 `num_envs * num_steps_per_env * max_iterations` 推导出的 budget accounting 值，不伪装成 runtime observation；
- stdout `Total timesteps` 只能作为显式 legacy fallback，默认禁用；
- `config_derived_total_timesteps` 超 cap 或与 planned env steps 不一致时 reject；
- train summary 中实际 `num_steps_per_env` 与 budget contract 不一致时 reject；
- checkpoint count / size 超 secondary cap 时 reject；
- 不运行长训作为 Phase 8B 必要验收，长训容量留给后续正式 budget/capacity validation。

Phase 8B 仍未完成的非阻断项：

- grad / optimizer / forward-backward 的底层 hook 还未实现；
- 当前只能记录由 RSL-RL 配置推导出的估算值；
- 真正不可绕过的 update/optimizer 计数必须在 Phase 8D/8E 前补齐。

### Phase 8C：OS 级候选隔离

工作量：2-5 天，取决于服务器权限。

目标：

把候选代码放入受限运行环境，避免 hidden leakage 和主线污染。

优先级从高到低：

1. 候选进程独立；
2. repo 只读；
3. candidate dir 唯一可写；
4. hidden/audit 路径不挂载；
5. 网络禁用；
6. 环境变量白名单；
7. secret 不进入候选环境；
8. 输出大小限制；
9. 进程超时和子进程清理。

实现选项：

- 若服务器允许：容器 / namespace / chroot；
- 当前服务器实测：Docker 无当前用户权限，unprivileged bind mount 不可用；`unshare -Urnfp --mount-proc` 可用且 CUDA smoke 通过；
- 因此 Phase 8C 最小实现采用 `unshare` network/PID namespace + Landlock 文件系统限制 + subprocess wrapper + 环境变量清理；
- Landlock 不能授予 whole-repo read：`configs/` 里包含 protected eval profile/seed 定义，候选进程只能读取最小 candidate API 面；
- 候选可写输出目录必须和 runner 权威产物目录分离：`candidate_output/` 可写，`runner_artifacts/` 不可写；
- 若 Landlock 或 `unshare` 在服务器强验收中不可用，Phase 8C 必须 fail closed，不能降级为“看起来隔离”的普通 subprocess；
- 无论哪种，runner 必须在候选进程外部。

验收：

- candidate 无法读取 hidden/audit 文件，因为文件系统内不存在；
- candidate 无法读取仓库内 protected config，例如 `configs/phase2_baseline.yaml`；
- candidate 无法写主线 repo；
- candidate 无法写 runner artifact 目录；
- candidate 可以写 candidate output 目录；
- candidate 无法访问网络；
- candidate 无法读取 `DEEPSEEK_API_KEY`；
- candidate 退出后无残留子进程。

Phase 8C 的边界：

- 8C 只证明候选代码加载/组件构建可以在受限子进程中完成；
- 8C 不直接开放 LLM-authored 正式训练；
- reward/loss callable 在 Isaac/RSL-RL 长训过程中的全链路隔离留到 8D/8E 接入；
- 如果未来训练必须在隔离 namespace 中运行，需要先做 CUDA + Isaac train-start 容量验收。

理由：

这是 code-level 的真正主防线。静态检查不能防字符串拼接、base64、`getattr`、`importlib`、环境变量和动态 import。

### Phase 8D：code-level verifier v1

工作量：2-4 天。

目标：

在 OS 隔离之上增加静态和动态检查。注意：这是第二层防线。

实现边界：

- 8D verifier 只验证 candidate package 和隔离组件加载；
- 不提前运行完整 Isaac/RSL-RL 长训；
- 不宣称 reward / obs / loss callable 已在 PPO loop 内完成 runtime 验证；
- full train-start、reward finite、optimizer/grad hook 必须在 8E 接入真实 callable 后再做。

静态检查：

- 候选目录以外无 diff；
- 禁止候选 manifest 声明 forbidden capability；
- 禁止明显 hidden/iid/audit 字符串；
- 禁止明显 seed 硬编码，但用 AST / token-aware 检查，避免普通浮点指标误判；
- 禁止 subprocess / socket / requests / urllib；
- 禁止 import eval runner；
- 禁止改 success metric；
- 禁止修改或 shadow verifier；
- protected files 缺失视为 error；
- diff filter 包含 delete。

Phase 8D v1 对 candidate package 采用更具体的落地检查：

- manifest parse / entrypoint path；
- file count / file size / total size；
- top-level entries allowlist；
- suffix allowlist；
- symlink reject，且父进程静态扫描不得跟随 symlink 读取目标内容；
- 只读取 regular file；FIFO、device、socket 等特殊文件直接 reject；
- Python AST import/call/token checks；
- 对 candidate 包内允许后缀的文本产物做 forbidden-token 内容扫描，包括 `rationale.md`、`operation_diff.json`、`diagnostics.json` 和 `.txt`；
- public component summary token scan。

动态检查：

- candidate import timeout；
- isolated component load；
- public-only output scan；
- reward finite（8E 接入 callable 后启用）；
- observation finite（8E 接入 callable 后启用）；
- reset/step smoke（8E 接入训练后启用）；
- train-start smoke（8E 接入训练后启用）；
- NaN/Inf 检查；
- canary 输出检查；
- budget accounting 检查。

必须继承 Phase3 教训：

- verifier 规则从可信 base_ref 读；
- 不从候选 worktree 读自身规则；
- 候选删保护文件也必须报错；
- diff filter 包含删除。

验收：

- 攻击样例全 reject；
- no-op candidate 通过；
- 合法 reward/curriculum candidate 通过。

### Phase 8E：单 Agent code-level cold-start 闭环

工作量：4-7 天，主要消耗训练时间。

目标：

让 LLM Agent 在代码沙盒内从 cold-start 目标开始产生训练 recipe，并让 accepted recipe 能被下一轮继承。

prompt 原则：

- 不写“你必须先站立再抗 push”；
- 可以写“你可以自主设计 curriculum/staged training/auxiliary objectives”；
- 明确目标是固定 public eval 下成功率最大化；
- 明确不能改裁判；
- 明确所有提升必须通过外部 runner 验证。

开放能力：

- reward shaping；
- curriculum schedule；
- delayed push / magnitude ramp；
- training-only DR；
- auxiliary loss terms；
- diagnostics；
- 无状态 training-only observation augmentation。

暂不开：

- runner hook；
- PPO loop hook；
- optimizer hook；
- eval observation wrapper；
- eval action wrapper；
- checkpoint policy；
- success metric。

闭环：

1. 给 public-only feedback；
2. Agent 写 candidate package；
3. code sandbox verifier；
4. immutable runner 训练；
5. public_dev eval；
6. public_accept + OOD stress gate；
7. accepted recipe 进入 public-only ledger；
8. 下一轮从 accepted recipe 代码作为当前 best recipe 继续修改。

这一步必须区别：

- policy 权重每轮 cold-start，保证公平；
- recipe 代码可以继承，形成自进化。

如果 accepted recipe 不被下一轮继承，每轮都是独立采样，不是 evolution。

accept/reject 规则需要适配 cold-start near-zero baseline。

两阶段规则：

1. 离地期 absolute floor：
   - 如果 baseline 近 0，候选在 dev 至少 `k` 个 seed 超过绝对最低成功率，标记为 `promising`；
   - 例如 `>=2/3 seeds` 超过 `absolute_success_floor`；
   - 这不是最终 accept，只是避免 pooled SE 在 0 附近失真。
2. 稳定期 noise-aware delta：
   - 当 baseline 和 candidate 都离开 0 区间后，再使用 paired/noise-aware delta；
   - near-threshold 才加 seed 或 public_accept；
   - public_dev 不能同时承担候选池选择和最终 accept。

自进化成功指标：

- 连续 `N` 次 accepted recipe 能形成 lineage；
- best public_accept success 相对 initial minimal recipe 累计提升 `X`；
- OOD stress gate 不下降；
- 运动合理性 gate 不恶化；
- 提升可归因到 recipe diff，而不是单次 cold-start 运气。

保留 report-only 参照：

- 保留一个 config/warm/reference recipe 作为报告参照；
- 不一定做完整 A/B；
- 用于回答“代码级权限相比 config-only 到底多拿到了什么”。

验收：

- 至少跑通 1 个 no-op candidate 和 1 个 LLM-authored candidate；
- LLM-authored candidate 必须在 8C 隔离通过后才允许运行；
- 候选从 cold-start policy 出发；
- accepted recipe 可被下一轮继承；
- 没有 hidden leakage；
- 有可审计 budget accounting；
- TensorBoard 和 summary 可复现。

落地拆分：

- Phase 8E scaffold 先实现 public-only prompt、8D 验证接线、public decision、ledger 和 accepted recipe inheritance；
- scaffold 阶段不启动正式训练，不调用 LLM API，不做候选池；
- 正式训练/eval 执行接线必须由 immutable runner 显式加入，不能由 candidate 或 prompt 侧触发。

### Phase 8E-code：从声明式 scaffold 修正到真 code-level（修正计划）

日期：2026-06-25。

背景与判断：当前 8E scaffold 的候选返回的是**声明式数据 hint**（如 `reward_weight_hint` / `curriculum_hint`），必须由一个 runner 侧 executor 去解释才能生效——这等于 **config 级表达力**（候选只能在人预编的词汇表里点名+调数字），并不是项目立论要验证的「code-level 比 config 多拿到的东西」，而且会让 8C/8D 的隔离对「只传数据」而言过度建设。

**目标改为：候选直接写出可执行的 reward/loss/curriculum 函数，runner 直接调用、直接适配进训练，不再经过 hint→executor 这第二层。**

#### 0. 已先行修复的前置 BUG（开训前必须，已完成并在服务器验证）

- **BUG-1a**：ledger 与不可变快照改为 `lab_dir` 级稳定路径；跨 `run_dir` 多轮可继承（之前 per-round run_dir 会静默重置 ledger → 无继承 = 非进化）。
- **P1b / BUG-1b**：accepted recipe 存为 `lab_dir/accepted_recipes/<idx>_<id>` **不可变快照**，`resolve_current_recipe` 读快照而非原始候选目录（原始目录后续被改不会污染历史 accepted recipe）。
- **P1a**：`public_accept` 不再进入下一轮 prompt（prompt 只保留 `public_dev` 作反馈 + gate 的 pass/fail 状态；`public_accept` 标记为 `withheld_from_prompt_history`）。

#### 1. 安全模型（先立这个，后面所有改动服从它）

不靠「进程内 Python 沙箱」（逃逸防不住），靠两道真边界：

1. **OS 隔离边界（对外）**：整个**训练进程**必须跑在能真实承载 CUDA/Isaac 的 OS 隔离里。2026-06-26 服务器实测显示：`unshare` namespace 与 CUDA 兼容，但 `Landlock + CUDA` 不兼容；因此 GPU 训练边界不能继续写成“unshare + Landlock”。当前已选定并验证 `namespace + minimal rootfs/pivot_root`：只 bind 系统/CUDA/Isaac env、`repo/src`、候选目录和 candidate output，不 bind protected configs/secrets/host repo root。候选 reward 代码即使是任意 Python、即使在进程内完全逃逸，**也只能影响这个训练进程**，拿不到隐藏数据、传不出去、改不了主线。
2. **独立可信裁判边界（对内/正确性）**：训练进程**只产出 checkpoint（纯权重）+ public 训练日志**；**eval 在另一个可信进程**里加载权重、用冻结接口 + 不可变 metric 评分，候选代码**永不进入裁判进程**。

推论：候选能做的最坏的事是训出一个 game 仿真的策略 → 由**独立 eval + OOD gate + motion gate** 兜底。**8D 的 AST 静态检查降为 defense-in-depth，不作主防线；因此候选可以写真 Python，而不必受限于 DSL——安全来自「训练进程的 blast radius 被 OS 隔离」，而不是来自「限制候选代码本身」。**

阶段状态与详细验收记录不写在计划正文中，见：

- `development_logs/phase_8_cold_start_autoresearch/2026-06-25_phase_8e_1_safety_model.md`
- `development_logs/phase_8_cold_start_autoresearch/2026-06-27_phase_8e_4_process_split.md`

#### 2. 组件契约改造（数据 → 带契约的 callable）

- `TrainingComponents` 字段从「数据 dict」改为「可执行 callable」：
  - `reward_terms: [{name, fn}]`，`fn(inp: RewardInputs) -> Tensor[num_envs]`；
  - `loss_terms` 同理（加进 PPO loss）；
  - `schedules: [{name, fn}]`，`fn(progress: float) -> dict`（参数限定在白名单 key）。
- 新增固定、public、**不泄露**的输入契约 `RewardInputs`：`base_height, roll, pitch, base_lin_vel, base_ang_vel, joint_pos, joint_vel, last_action, push_active, push_vec_public, progress_fraction`；**绝不含** eval seed / iid profile / hidden。候选 `fn` 只收这些张量，**收不到 env 对象 / metric / eval 配置**。

阶段状态与详细验收记录不写在计划正文中，见：

- `development_logs/phase_8_cold_start_autoresearch/2026-06-26_phase_8e_2_component_contract.md`
- `development_logs/phase_8_cold_start_autoresearch/2026-06-26_phase_8e_3_training_loop_adapter.md`

#### 3. 三类组件的执行方式

| 组件 | 是否在 loop 里跑候选代码 | 安全执行方式 |
|---|---|---|
| curriculum schedule | 否 | runner 在沙箱里按 iteration 网格**预求值成数值表**，训练按表应用。零 in-loop 候选代码、零成本。 |
| reward / aux loss | 是（每 step 向量化调用） | 候选 callable 在 **`minimal_rootfs_pivot_root` 隔离训练进程**内被 runner 调用，输出做 finite/shape/dtype/clip 守护。`Landlock + CUDA` 已证伪，不能作为这个执行环境。 |
| train-only DR / obs augmentation | 训练时是，**eval 时一定关** | 只允许「eval 时可关、不改 policy 输入空间」的增强；**禁止改 eval 期观测空间** → eval 不需要任何候选代码。 |

#### 4. 训练/评估分进程（解决 F1/F2/eval-freeze 的根）

- **训练进程**（namespace + minimal rootfs/pivot_root + CUDA）：runner 拥有 env、rsl_rl `OnPolicyRunner`、optimizer、checkpoint 选择与预算 cap；候选 callable 仅作**纯回调**；**沙箱外父 runner** 再加 wall-clock / 总 env-step 硬 cap（逃逸也被外部 kill）。当前代码里的 8B 预算仍是 `config_derived_total_timesteps`，不是 runtime observed counter；首轮可先使用 runner 固定的 8B 校准预算，真正不可绕过的 runtime env-step / optimizer-step 计数器作为后续硬化项接入。Landlock 暂退到 component-load/static probe，不再作为 GPU 长训主边界。
- **评估进程**（可信、无候选代码）：只消费 runner 解析并封存的 sealed checkpoint 路径，冻结 runner-owned `agent_cfg/env_cfg`、obs/action 接口 + 不可变 metric，先 `assert_safe_checkpoint` 再 `load_state_dict`；不得使用候选上报的 checkpoint path、arch、obs/action 定义。评估产出 `public_dev` / `public_accept` + 已明确定义的 gate 摘要。

阶段状态与详细验收记录不写在计划正文中，见：

- `development_logs/phase_8_cold_start_autoresearch/2026-06-27_phase_8e_4_process_split.md`

#### 5. 具体文件改动

- `code_sandbox/components.py`：定义 `RewardInputs` + callable 契约（reward/loss/schedule 的签名与返回守护说明）。
- `cold_start_lab/recipe_adapter.py`（新，runner-owned，从可信 ref 加载）：**直接把候选 callable 接进 rsl_rl 的 reward/loss 计算** + 应用预求值 schedule 表 + 逐项输出守护。注意：这是「直接调用候选函数」，**不是解释 hint**。
- 训练 runner（扩展 8B runner）：拥有 loop/计数器/optimizer/checkpoint；整训练进程用 8C `run_isolated_subprocess` 起；父进程加硬 cap。
- eval runner（新增/复用 phase1 eval，标记 trusted）：冻结接口 + 实现 OOD/motion gate 的**真定义**（填上 scaffold 里「gate 只被消费没被定义」的洞）。
- `code_sandbox/verifier.py`（8D 增项）：对候选 callable 做 AST **defense-in-depth**（标记 import / dunder 属性 / subprocess 等明显危险，但**不作硬边界**）；校验 manifest `eval_interface_changed: False` 且无 eval 期 obs/action transform。
- `phase8e_minimal_rootfs_probe.py`：`unshare -Urmnfp` 下构建 tmpfs rootfs，bind 系统/CUDA/Isaac env、`repo/src`、candidate、candidate output 与 runner artifact；`pivot_root` 后 detach oldroot；验证 CUDA 与 hidden/config/repo/network/secret 边界；`/runner_inputs` 只暴露训练专用脱敏配置并做禁词回归；可选 `--train-start-smoke` 真跑 `phase1_push_recovery_train_smoke.py`，其中 `/candidate_output` 只作候选可写输出，权威 train summary 与 sealed checkpoint 必须落在 `/runner_artifacts`，无 checkpoint 或子进程失败统一视为 candidate/recipe reject。
- `code_sandbox/isolation.py`（8C/8E 增项）：保留 Landlock component-load/static probe；GPU 训练隔离走 namespace + minimal rootfs/pivot_root。当前只允许 `/dev/urandom` 读、`/dev/null` 写这类最小安全设备用于 Landlock probe，仍不放开整仓、不放开 hidden/secret。

#### 6. 从 0 自演进首轮测试的后续实现计划

当前已接好并复验的链路是：

```text
候选契约 -> 8D 校验 -> minimal-rootfs 隔离训练 -> 随机初始化 train-start -> reward callable 真进入训练 step -> sealed checkpoint -> trusted checkpoint safety gate
```

这说明 true code-level 的训练侧基础已经成立。后续实现重点不是重做安全边界，而是把“训练侧产物”接到真实公开评测、自动决策、LLM 候选生成和 recipe lineage 中，形成可复现的一轮从 0 自演进实验。

口径约束：

- 当前通过的 `phase1_push_recovery_train_smoke.py --max_iterations 1` 只是 train-start / minimal-rootfs / CUDA runability smoke，不能当作完整预算训练；
- 首轮真正训练必须使用 Phase 8B 校准出的固定预算（例如 4096 env、固定 `max_iterations`、固定 `num_steps_per_env`、固定 wall-clock cap）跑满，并在 summary 中写清楚预计耗时/实际耗时；
- 首轮预算可先以 runner 固定配置推导的 `config_derived_total_timesteps` 执行与审计，不得写成 runtime observed；runtime observed env-step / optimizer-step counter 留作后续硬化；
- `required_pass_gates` 只能列已经由 trusted eval / runner summary 明确定义并产出的 gate；首轮未实现 OOD 真研究量前，应收窄到已定义的最小 gate。

##### 任务 1：实现真实公开评测 metric

将 `phase8e_trusted_eval_probe.py` 从边界探针扩展为可信 public eval runner。它仍保持“不接收 candidate_dir、不 import 候选代码”的裁判边界，但要开始真实加载 checkpoint 并运行 public rollout。

实现内容：

- 从 sealed checkpoint 加载 policy；
- 在固定 public push 任务上运行 eval rollout；
- 产出 `public_dev` 与 `public_accept` 的 success mean / per-seed success / failure labels；
- 产出 motion-quality 指标，例如 energy、action-rate、接触合理性、异常振荡摘要；
- 产出 decision 会消费的 gate 对象，至少包含首轮定义清楚的 `budget_accounting` 与 `motion_quality_gate`，每个 gate 都必须有 `status: pass|fail` 和简短 reason；
- 若 `ood_stress_gate` 还没有真实 OOD 分布与阈值定义，则首轮不要把它放入 `required_pass_gates`，只以可选诊断或后续 9A 项记录；
- 通过 runner 侧 `train_start_paths` 解析 sealed checkpoint，只允许 eval 消费 runner-owned 路径，不接受候选 summary 上报路径；
- 用 runner-owned 固定 `agent_cfg/env_cfg` 重建 policy，先 `assert_safe_checkpoint`，再 `load_state_dict`；不得使用候选派生的 arch / observation / action 空间；
- eval 进程不接收 `candidate_dir`，不 import 候选代码，不读取 hidden/iid；
- `public_accept` 不进入下一轮 prompt，只能进入 decision 与 ledger 的 withheld 摘要。

验收：

- no-op / baseline checkpoint 能跑出真实 public_result JSON；
- 恶意 candidate 目录传给 eval 必须 reject；
- 候选 summary 中伪造 checkpoint path 时，eval 仍只加载 runner 解析出的 sealed checkpoint；
- checkpoint 不安全必须 reject；
- `required_pass_gates` 中每个 gate 都在 public_result / runner summary 中有真实 pass/fail 对象；
- public_result 通过 public-only assertion 和 canary。

##### 任务 2：实现 train/eval/decision/ledger 闭环编排

在当前 `run_phase8e_round()` scaffold 外新增 runner-owned full-round orchestrator。`run_phase8e_round()` 保持为“public_result -> decision/ledger/inheritance”的纯决策层；完整训练与评测链路由 `phase8e_full_round.py` 串联，取消正式闭环对手工 `--public-result` JSON 的依赖。

实现内容：

- `src/humanoid_agent/cold_start_lab/phase8e_full_round.py` 负责自动执行：
  1. provider/replay/preexisting candidate package；
  2. Phase 8D verifier；
  3. `unshare + minimal-rootfs` fixed-budget train-start；
  4. runner-owned sealed checkpoint safe-load gate；
  5. trusted public rollout；
  6. trusted eval public_result projection；
  7. decision；
  8. public-only ledger；
  9. accepted recipe snapshot；
- CLI 入口为 `scripts/phase8e_full_round.py`；
- `isolated training` 显式区别于 1-iteration smoke：正式一轮按 8B 固定预算 `4096 envs × 500 iterations × 24 steps = 49,152,000 env-steps` 跑满；
- training summary 必须写入 `budget_accounting`、`config_derived_total_timesteps`、`num_envs`、`num_steps_per_env`、`max_iterations`、elapsed seconds、checkpoint list 与 sealed checkpoint path；
- public_result 只能由 trusted eval runner 写入，不能由候选或手工 JSON 伪造；
- 失败路径要清晰区分 `reject`、`blocked`、`failed_safety_gate`。

验收：

- 单个手写 candidate 可以从一条命令完成 train/eval/decision/ledger；
- no public_result 手工注入也能闭环；
- 训练阶段不是 `--max_iterations 1` smoke，而是固定 full-budget run；运行时长以分钟/小时记录到 summary；
- accepted recipe 能作为下一轮 current best recipe；
- policy 权重每轮随机初始化，recipe 代码可继承。

当前落地状态（2026-06-27）：

- 已实现 full-round driver 与 CLI；
- 已实现 fake selftest，验证未过 8D/skip-verify candidate 不会进入训练；
- 已验证 full-round driver 的顺序为 provider/8D -> isolated fixed-budget training -> public rollout -> linked decision；
- 真实长训/真实 public rollout 尚未在本地启动，需在服务器 GPU 空闲时运行 `scripts/phase8e_full_round.py` 作为首轮实测。

##### 任务 3：实现 LLM provider 到 candidate package 的自动落盘

在当前 prompt package 机制上接入 provider abstraction，使 DeepSeek/其他 provider 的输出能被落成候选包并进入 verifier。provider 层与 candidate 层继续保持网络和密钥隔离。

当前实现状态（8E-6-3）：

- 已新增 provider ingestion 层：构造 public-only prompt、调用 provider/replay/dry-run、解析结构化候选文件包、写入 staging candidate dir；
- 已支持 provider 可替换：默认 dry-run，本地 replay，GLM/OpenAI-compatible 通过 env 配置接入；
- 已加入路径逃逸、缺 manifest、敏感值泄露、public-only 禁词与 8D static verifier 回归测试；
- provider skip-verify 只报告 `staged_unverified`，不会被 full-round driver 当成训练绿灯；
- candidate staging 已改为“预校验全部文件 -> 临时目录写入 -> 原子 rename”，失败时不留下半包残留；
- provider 输出的 candidate package 已能被 full-round driver 消费，继续接入 fixed-budget isolated training 与 runner-resolved sealed checkpoint eval。

实现内容：

- provider abstraction：GLM 只是一个 provider，后续可换 OpenAI-compatible/本地模型；
- prompt package 明确要求输出文件集合，而不是自然语言建议；
- 生成结果写入 staging candidate dir；
- 写入后跑 manifest/schema/public-only/8D verifier；
- provider 层可联网、可读 API key；candidate 训练/eval 层断网且不含 API key；
- prompt/response 日志脱敏保存，不能泄露 secret、iid、hidden、public_accept 原始结果。

验收：

- 不泄露 API key；
- LLM 生成的最小候选能落盘并过 verifier；
- LLM 输出非法文件、缺 manifest、越权修改时能明确 reject；
- 同一接口可替换 provider。

##### 任务 4：逐步增强可用训练杠杆

这些增强项决定 Agent 后续能否持续提升。首轮最小闭环可以先用已经生效的 reward shaping + `reward_weight_scale` 跑通；随后按收益和风险逐项打开更多训练杠杆：

- `push_active` / `push_vec_public` 仍为 `not_instrumented`，候选看不到真实 push 触发；
- schedule 中 `push_magnitude_scale`、`push_start_fraction`、`push_interval_s`、`domain_randomization_scale` 仍主要是预求值/记录，尚未完全施加到 env；
- `LossTerm` 仍未接入 PPO loss；
- reward shaping 主杠杆已经生效，可作为首轮最小闭环的主要搜索空间。
- OOD stress gate 是独立研究量，首轮只做最小 sanity gate：public success + 基本 motion-quality 合理性；待 trusted eval 稳定后再把 OOD 分布、阈值和 accept gate 推到 9A 或后续增强。

处理策略：

- 首轮最小闭环允许只开放 reward shaping + `reward_weight_scale`；
- push/DR schedule 应作为 8E 后续增强项，或在 Phase 9A 前补；
- `required_pass_gates` 必须与首轮实际 gate 范围一致，不能提前要求未产出的 OOD/motion 复杂 gate；
- LossTerm PPO hook 不应在 public eval/closed-loop 接通前抢先做。

#### 7. 首轮从 0 自演进测试的完成标准

首轮测试的目标是证明“LLM 生成或修改 recipe 代码 -> 从随机 policy 开始训练 -> 外部可信评测 -> recipe 可继承”这一条链路可运行。建议按以下完成标准收口：

1. `trusted_eval` 从 runner-resolved sealed checkpoint 真实跑 public rollout，产出 public_result。
2. `run_phase8e_round` 或新 orchestrator 能自动串起 train/eval/decision/ledger。
3. DeepSeek/provider 输出能落成 candidate package，并经过 8D verifier。
4. public-only prompt 只包含 public_dev / 允许的 gate 摘要 / accepted recipe lineage，不含 public_accept 原始统计、iid、hidden、eval seed。
5. 单个手写真候选完成一轮 fixed-budget isolated training / trusted eval / decision / ledger；不能用 1-iteration smoke 冒充正式训练。
6. 单个 LLM-authored 真候选完成一轮 fixed-budget isolated training / trusted eval / decision / ledger。
7. `required_pass_gates` 只包含已产出的 gate，且每个 gate 有 `status: pass|fail`；未定义 OOD gate 不进入首轮必过列表。
8. eval 使用 runner-owned 固定 policy arch / obs / action 接口，先 checkpoint safety gate，再加载权重。
9. accepted recipe snapshot 能被下一轮继承；policy 权重仍每轮 cold-start。
10. TensorBoard、training summary、eval summary、decision summary、ledger 能复现同一轮。

达到以上 10 条后，即可运行并记录“Phase 8E 单 Agent code-level cold-start 闭环首轮测试”。如果只完成其中一部分，报告时应明确对应的是训练侧接入、评测接入、LLM 生成接入或闭环编排接入，不把局部验收写成完整自演进结果。

#### 8. 诚实口径

候选写的是**真 Python reward/loss/schedule 函数、被 runner 直接调用**（不是 hint，也不是受限 DSL）。**安全不靠限制候选代码本身，而靠「训练进程 OS 隔离的 blast radius + 独立可信裁判 + runner 拥有 loop/预算」。** 这样「能改动的东西」从「调人给的旋钮」升级为「写系统里本不存在的训练逻辑」——这才是 code-level 自进化要验证的对象。

### Phase 9A：候选池 best-of-K

工作量：3-5 天。

目标：

在 code-level 单候选跑通后，加入候选池，而不是马上做 Git evolution。

方案：

- 每轮 K=4-8 个候选；
- verifier 先筛；
- train-start smoke 只过滤崩溃，不排序；
- public_dev 只用于候选池选择；
- top 1-2 进入 public_accept + OOD stress gate；
- public_accept 结果才决定是否进入 ledger；
- accepted recipe 作为下一轮 current best recipe。

注意：

- 不要用短训 success 排序 curriculum 类候选；
- cheap screen 只作为 runability 过滤器；
- 9A 的并行收益通常足够，不急着 9B。

验收：

- 同一轮多个候选能被独立运行；
- winner's curse 被 dev/accept split 控制；
- rejected candidate 不污染下一轮 prompt；
- accepted recipe lineage 可追踪。

### Phase 9B：Git evolution / branch 协作

工作量：5-10 天。

目标：

接近 ENPIRE 的 evolution module。

先决条件：

- 9A 证明候选池确实有用；
- immutable runner 稳定；
- code sandbox 稳定；
- public-only ledger 稳定；
- recipe inheritance 稳定。

设计：

- 每个 lane 一个 branch；
- 每个 branch 只提交 candidate package 和 public-only report；
- 不直接推主线；
- accepted recipe 由 orchestrator cherry-pick；
- raw hidden/private eval 不进入 prompt；
- 自动生成 idea tree。

为什么不提前做：

在本地只有少量 GPU 的情况下，9A 候选池已经能拿到大部分并行收益。Git evolution 的主要价值是长期多 Agent 协作和知识沉淀，先等候选池证明有效。

### Phase 10：跨任务扩展

工作量：每个任务 2-5 天。

这部分后移，不在第一个 cold-start loop 前投入大量抽象。

原则：

- 第一个任务先具体做 G1 standing push recovery；
- task spec 保持薄，不做过度泛化；
- 等完整 loop 跑完一次，再抽象共性；
- 新任务只新增 adapter/spec，尽量不改 runner/verifier。

候选后续任务：

1. G1 standing push recovery cold-start；
2. G1 velocity tracking no-push cold-start；
3. G1 multi-direction push recovery；
4. G1 terrain stability；
5. 其他机器人或 manipulation。

## 6. 对图中问题的处理结论

### P1：仿真中 “不让裁判作弊” 不等于 “不能作弊”

成立，已修改。

处理：

- 删除 “provably no cheat” 式表述；
- 加入 OOD stress gate；
- 加入运动合理性 gate；
- hidden 仍保持一次性 audit，不作为反复 gate。

### P2：计划目前像安全 sampler，不像 evolver

成立，已修改。

处理：

- 明确 policy 权重每轮 cold-start；
- recipe 代码必须继承；
- accepted recipe 作为下一轮 current best recipe；
- 增加自进化成功指标。

### P3：API 边界和预算 cap 自相矛盾

成立，已修改。

处理：

- 删除候选 runner hook / PPO loop hook 权限；
- 候选只提供组件；
- runner 控制外层循环、env step、optimizer step；
- 预算主公平单位改为 env-steps。

### P4：预算公平单位应是 env-steps，不是 4000 iterations

成立，已修改。

处理：

- 4000-iteration calibration 只用于估算初始 env-step budget；
- Phase 8B runner 记录 config-derived env steps；真正的 runtime VecEnv.step counter 留到 Phase 8D/8E；
- grad/optimizer/update steps 作为次级 cap；
- wall-clock 只做 safety cap。

### P5：eval 时候选 observation code 可能成为执行漏洞

成立，已修改。

处理：

- eval observation/action 接口默认冻结；
- 第一版不允许 LLM 改 eval observation wrapper；
- observation/history 改动先限于 training-only augmentation；
- 若未来必须 eval 期执行候选 transform，要先做 transform freezing 和更强隔离。

### P6：8C 前不应运行 LLM-authored code

成立，已修改。

处理：

- 8A-min 只跑我们手写 no-op candidate；
- LLM-authored candidate 必须等 8C 隔离通过。

### P7：网络和密钥要分层

成立，已修改。

处理：

- orchestrator/LLM 层可联网；
- candidate 层 no-network；
- `DEEPSEEK_API_KEY` 不进入候选环境变量。

### P8：candidate 输出也要 public-only 断言

成立，已修改。

处理：

- `rationale.md` / `operation_diff.json` / `diagnostics.json` / summary / ledger 全部过 public-only assertion 和 canary。

### P9：需要保留 config/warm 参照点

成立，已修改。

处理：

- 8E 保留 report-only reference recipe；
- 不一定做完整 A/B，但用于解释代码级权限带来的增量。

## 7. 推荐下一步

下一步直接做：

```text
Phase 7A -> Phase 8A-min -> Phase 8B -> Phase 8C -> Phase 8D -> Phase 8E
```

最小实现目标：

1. 修 checkpoint selection；
2. 建 no-op code sandbox candidate；
3. 建 immutable runner 最小版；
4. 把预算主单位改成 env-steps；
5. 加运行时 budget accounting；
6. 加最小 OS/进程/文件系统隔离；
7. 隔离通过后再跑 LLM-authored code-level cold-start candidate。

暂不做：

- 完整 config-only loop；
- 完整跨任务 registry；
- Git evolution；
- 多 Agent branch 协作；
- 复杂 idea tree；
- eval observation wrapper 修改。

## 8. 最终边界

这条路线更接近 ENPIRE，但底线必须保持：

> Agent 可以改训练 recipe，不能改裁判；policy 每轮 cold-start，recipe 可以继承。

裁判包括：

- env-step budget；
- eval；
- seed；
- success metric；
- checkpoint policy；
- public/accept/OOD/hidden split；
- accept/reject；
- verifier 规则；
- eval observation/action interface。

只要这个边界守住，代码沙盒就可以尽快开放；如果这个边界没守住，code-level 结果即使提升也不可解释。

### 2026-06-27 补充：8E-6-2 linked eval/decision 前置链接层

已新增 `scripts/phase8e_linked_round.py` 作为任务 2 的前置编排入口。它不启动 LLM、不启动正式长训，只消费 runner/minimal-rootfs 已产出的训练 summary 与 sealed checkpoint，并把它们接到 trusted eval / decision / ledger。

该入口的职责边界：

- `--train-summary` 必须来自 runner-owned 训练阶段；
- `--runner-artifacts-root` 必须由编排器显式传入，不能从候选目录或候选 summary 推断；
- checkpoint 只接受 `/runner_artifacts/train_start_smoke/sealed_checkpoints/model_<n>.pt`，并通过 `train_start_paths` 映射到 host sealed checkpoint；
- trusted eval 前必须重新执行 checkpoint safe-load gate；
- `public_result` 只能由 `phase8e_trusted_eval_probe.py` 写出；
- `run_phase8e_round()` 继续负责 8D verifier、decision、ledger 与 accepted recipe inheritance。

后续任务 2 仍需要把 isolated full-budget training 和真实 rollout eval 串进同一条命令；8E-6-2 只解决“已有训练产物如何被安全、可复现地送入 trusted eval/decision”的链接问题，不能替代 full-budget 自演进首轮。
