# 人形机器人运控 Agent 自改进项目计划书 v3

> 项目暂定名：**Humanoid Control Agentic Self-Improvement MVP**
> 当前目标：先做一个**最小闭环**，在仿真中证明 Agent 能基于真实训练/评估反馈，自循环改进一个简单人形机器人运控策略。
> 当前不追求论文完整性，不做复杂画图、不做多 Agent、不做记忆迁移、不接真机。第一优先级是：**环境跑通 → baseline 跑通 → 自动评估跑通 → Agent 单轮改进跑通 → 多轮自循环跑通**。


## 1. 项目核心定义

### 1.1 要做什么

做一个人形机器人简单运控任务：

> **Unitree G1/H1 在仿真中进行 standing push recovery**。机器人站立，环境随机施加外力扰动，策略需要保持不摔倒，并尽量恢复姿态和位置。

系统外层加入一个 coding agent：

```text
训练当前策略
    ↓
固定评估集 rollout
    ↓
生成结构化失败报告
    ↓
Agent 阅读报告、训练日志、配置
    ↓
Agent 修改 reward / curriculum / domain randomization / PPO 参数（仅 config + reward-term 注册表）
    ↓
自动 verifier + smoke test
    ↓
固定预算重新训练多个 seed
    ↓
按统计准则接受或拒绝
    ↓
进入下一轮
```

### 1.2 不做什么

当前第一阶段不做：

- 不接真实机器人；
- 不做视觉输入；
- 不做人形上楼梯；
- 不做人形抓取/放置；
- 不做多 Agent fleet；
- 不做经验迁移；
- 不做论文级画图；
- 不让 Agent 改活源码 / 任意修改整个仓库（MVP 只改 config + reward-term 注册表）；
- 不允许 Agent 增加训练步数或降低评估难度刷结果。

### 1.3 参考 ENPIRE 的地方（只取最小必要部分）

ENPIRE 的核心是构建一个可重复反馈循环：**reset scene → execute policy → verify outcome → refine next iteration**。本项目只取最小必要部分：

```text
Environment：仿真 reset + push disturbance + success verifier
Rollout：训练和评估策略
Policy Improvement：Agent 修改有限 config / reward-term
Evolution：第一版不做多 Agent，只做单 Agent accept/reject
```

> 注意定位：本项目闭环跑在**仿真**里，因此严格说更接近 Eureka/DrEureka 那一类（LLM 改训练信号 → 仿真训练 → 看统计再改），而不是 ENPIRE 的真机闭环。这版不展开论文层面的区分，先把工程闭环跑通。

---

## 2. 当前硬件与开发条件

### 2.1 你的机器情况（已更新）

- 本地 Windows 电脑；
- 可 SSH 连接的 Ubuntu 22.04.5 LTS 服务器；
- 服务器 GPU：2 × NVIDIA GeForce RTX 5090，每张 32GB 显存；
- `nvidia-smi` 显示 driver 595.58.03，CUDA 13.2；
- **vLLM 服务已 kill，两张 GPU 现在都空闲，可用于训练。**

**LLM 后端决定**：Agent 的 LLM 用**云 API**（Claude / Codex / GPT 等），不占本地显存。
- 好处：两张 5090 全部留给训练；云端模型更强、改 config 时更稳。
- 需要：API 访问与预算（按本项目 token 量不大，主要花在每轮读 summary + 出 patch）。
- 如果将来想用本地模型：会重新占用一张卡，双 seed 只能串行，且本地模型要足够强才安全——MVP 不建议。


### 2.2 Windows 开发方式

```text
Windows 本地：VS Code
    ↓
Remote SSH 插件连接服务器
    ↓
服务器上写代码、跑训练、看日志
```

建议安装/使用：VS Code Remote SSH、tmux、git、conda/mamba、TensorBoard（第一阶段够用）、rsync/scp（下载日志和 checkpoint）。

---

## 3. 开源资源选择

### 3.1 首选：Unitree RL Lab

提供 Unitree 机器人在 Isaac Lab 上的 RL 环境和 baseline，当前支持 Go2、H1、G1-29DoF。优先从这里入手。

```text
https://github.com/unitreerobotics/unitree_rl_lab
```

使用顺序：先跑官方已有任务 → 确认 Isaac Lab + Unitree robot + PPO baseline 能跑 → 再改 standing push recovery。

### 3.2 底层平台：Isaac Lab / Isaac Sim

GPU 并行仿真、环境管理、RL 训练接口。安装前一定看官方版本要求，重点核对 driver / CUDA / PyTorch / Isaac Sim 版本匹配。

```text
Isaac Lab 安装:
https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html
Isaac Sim 系统要求:
https://docs.isaacsim.omniverse.nvidia.com/
```

> Blackwell（5090）提示：headless 训练据社区反馈可行，但常需较新的 PyTorch（cu128 系列），且部分版本会出现 **PhysX GPU 管线回退到 CPU**、渲染/IRAY 不可用等问题。务必先按第 4 节做完 smoke test 再投入。

### 3.3 软件 loop 练手（非主项目）：gym-pusht

如果 Isaac Lab 一时装不上，可以先用 `gym-pusht` 把「Agent 改 config → 训练 → 评估 → accept/reject」的**软件闭环**跑通练手。但它是 2D 推块、**不是人形任务**，只能当软件预备，不能作为主项目，也不能当人形退路。

### 3.4 人形退路（如果 Isaac 在 5090 上跑不动 GPU 物理）

万一 Isaac Sim 在这台 5090 上确实做不了 GPU PhysX（见 4.3 的 go/no-go），需要一条**非 Omniverse 的人形栈**，跑在原生 JAX/PyTorch 上：

```text
候选（需自行确认当前是否有现成 push-recovery / 人形 locomotion 环境）：
- MuJoCo MJX + Unitree G1 MJCF（mujoco_menagerie 里有 G1 模型）
- mujoco_playground（DeepMind 的 locomotion 套件）
- loco-mujoco（人形 locomotion/imitation）
```

这些不依赖 Isaac，可绕开 Blackwell 渲染/PhysX 的坑。gym-pusht 救不了人形，这条才是真正的人形退路。

---

## 4. 开工前必须完成的环境跑通与 smoke test

这一部分比设计 Agent 更重要。环境不稳定时，Agent 只会放大混乱。

### 4.1 服务器基础检查

```bash
nvidia-smi
lsb_release -a
uname -a
df -h
free -h
which python || true
which conda || true
```

确认：GPU 空闲（vLLM 已 kill，应显示空）、`/home` 空间足够、RAM 足够、能创建 conda 环境、是否有 sudo / 能否装系统依赖。

### 4.2 GPU 使用约定（已更新：两张卡都空）

两张卡现在都可用，默认安排：

```text
跑单个候选时：可用单卡
跑多 seed 时：GPU 0 跑 seed A，GPU 1 跑 seed B（并行）
当前最大风险是 RL 方差，不是候选数量 → 双卡优先用于「同一候选多 seed」
```

### 4.3 Isaac Lab 安装验证（按顺序，逐个 gate）

#### Smoke test A：torch + CUDA

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

通过标准：`is_available()==True`，能看到 2 张 RTX 5090。

#### Smoke test B：Isaac Sim / Isaac Lab 能 headless 启动

跑官方最小空场景或示例（命令按你最终安装版本文档为准）。通过标准：headless 不 crash、能创建 simulation app、能正常退出。

#### Smoke test C：官方 RL 示例能跑 1–2 分钟，**且物理在 GPU 上**

先不改任何代码，跑官方已有 task。除了「能跑」，**必须确认物理没有回退到 CPU**：

```text
检查项（关键，Blackwell 易踩）：
- sim 配置确实是 GPU/PhysX GPU pipeline（device=cuda，use_gpu_pipeline=True 之类）
- 训练时 nvidia-smi 里该进程有真实的 GPU 计算占用（不只是显存）
- steps/sec 或 FPS 在合理量级；若比预期慢 10× 以上，怀疑 PhysX 在 CPU
- log 无 NaN，能存 checkpoint
```

**Go / No-Go 决策点（重要）**：
```text
若多次尝试后，5090 上仍无法用 GPU PhysX（持续回退 CPU、吞吐塌方），
不要硬扛 →  切到第 3.4 节的非-Isaac 人形退路，整个项目换栈。
设一个时间盒，例如 3–4 天内 C 不通过就切换。
```

#### Smoke test D：Unitree RL Lab 官方 G1/H1 示例能跑

通过标准：G1/H1 能加载、环境能 reset、random/baseline policy 能 step、官方训练脚本能跑至少 1000–5000 steps（且物理在 GPU）。

> 只有 A–D 都通过，才开始写自己的 push recovery 任务。

---

## 5. MVP 任务设计：Standing Push Recovery

### 5.1 任务描述

机器人初始站立。每个 episode 在随机时刻对 base 施加扰动。

**扰动建议用「base 速度增量」而不是裸 N 的外力**：Isaac Lab 的 locomotion 环境通常用 `push_by_setting_velocity` 这类事件，直接给 base 一个速度 kick；这比施加力更可控，magnitude 单位也清楚。

```yaml
push:
  mechanism: base_velocity_kick   # 若用外力，则换算成冲量并按机器人质量归一
  direction: front / back / left / right / diagonal
  delta_v_range: [0.2, 0.5]       # m/s, 初始小一点；按机器人质量考虑，G1 ~35kg
  time_range: [1.0, 3.0]          # reset 后施加时间(s)
  # 若 mechanism 用 external wrench:
  #   用 impulse = force * duration，并除以 base mass 转成 Δv 再设区间
```

> Phase 1 第一步先去 Unitree RL Lab / Isaac Lab 找现成的 push/外力接口，确认它是速度 kick 还是外力，再把 magnitude 单位对齐。数值都是起点，需实测调整。

策略目标：不摔倒；base height 不低于阈值；roll/pitch 不超阈值；xy drift 不超阈值；扰动后尽量恢复直立。

### 5.2 动作空间

第一版沿用现有 Unitree RL Lab / Isaac Lab 的动作定义，通常是：

```text
policy 输出 joint position target 或 residual joint target
底层 PD controller 跟踪
```

不要第一版改 torque control（更难、更易不稳）。

### 5.3 观测空间

第一版不加视觉，只用 proprioception：

```text
base angular velocity
projected gravity / base orientation
joint positions
joint velocities
previous action
command 或 push phase 信息（视 baseline 设计而定）
```

是否把 push direction/magnitude 放入 observation：第一版建议不直接给真实 push magnitude；可以给一个简单 command（保持 standing）；如果训练太难，再考虑加 push phase indicator。

---

## 6. Success metric 与 diagnostics

### 6.1 二值 success metric（保持干净）

二值 success 只判断任务是否完成，不放动作平滑/能耗。**所有高度阈值用相对名义站高的比例**（先测出该机器人站立时的 base 名义高度，再设比例），G1/H1 不同。

```python
def success(traj):
    return (
        not traj.fallen
        and traj.min_base_height > height_threshold     # = ratio * nominal_standing_height
        and traj.max_abs_roll < roll_threshold
        and traj.max_abs_pitch < pitch_threshold
        and traj.final_xy_drift < drift_threshold
    )
```

不要把 `mean_action_rate / energy / mean_torque / action smoothness` 放进 binary success——它们是 secondary metrics，用于诊断和最终报告，不作为门槛。

### 6.2 Secondary metrics（记录但不作 gate）

```text
mean torque / mean power(energy proxy) / action rate
joint limit violation / foot slip rate
recovery time after push / max pitch / max roll / final drift
```

### 6.3 Failure classifier（先人工核对，再喂给 Agent）

每次 evaluation 自动分类失败原因，第一版简单规则即可：

```python
if fallen:
    if max_pitch > pitch_threshold:        failure = "fall_forward"
    elif min_pitch < -pitch_threshold:     failure = "fall_backward"
    elif abs(max_roll) > roll_threshold:   failure = "fall_side"
    elif min_base_height < height_threshold: failure = "height_collapse"
    else:                                  failure = "fall_unknown"
elif final_xy_drift > drift_threshold:     failure = "excessive_drift"
else:                                      failure = "success"
```

`oscillation / high_energy / foot_slip / joint_limit` 单独记录，不作为 failure 主类（否则 success/failure 分类会混）。

> 重要：分类规则有**顺序依赖和标签重叠**（例如「直接瘫下去但略前倾」会被判成 fall_forward 而非 height_collapse）。而 **Agent 是读这些标签来形成 hypothesis 的，标错就直接误导 Agent**。所以信任它驱动 Agent 之前，先拿 5–10 条 rollout 人工核对标签是否合理。
> 注意：Blackwell 上渲染/视频导出可能不可用，若要看 rollout 视频核对，可能需要换机器或 CPU 渲染单独跑评估。

---

## 7. Agent 权限设计：MVP 只改 config + reward-term 注册表

### 7.1 为什么 MVP 不让 Agent 改活源码

让 Agent 自由编辑 `src/rewards.py` 等活代码，看起来更 agentic，但对 MVP 是负担：

```text
- verifier 要挡住任意代码破坏（shape 错、死循环、finite 但语义错的 reward、副作用波及别处）
- 「新 reward 必须过单元测试」无法实现：对临时写出的新 reward 形式，你只能测它崩不崩，测不了对不对
- 需要 Agent 真正读懂 Isaac Lab / Unitree RL Lab 的代码结构（manager-based，间接层多）
- 这块和 Extension C 重复（Extension C 本就把「改 reward 代码结构」列为后期）
```

而「能不能证明 Agent 比 if-else 有用」靠的是 **rule-baseline 这个对照**，不是「在不在改源码」。所以 MVP 收成：

> Agent 只能改 **config（YAML）** + 在一个**预定义 reward-term 注册表**里启用/加权已有项；不写新 reward 代码、不改 observation 结构、不加 aux loss（这些归 Extension C）。

这仍然比固定 typed library 更 agentic：Agent 在完整统计画面下自由决定**改哪些旋钮、改多少、什么组合、给什么理由**——这正是 rule-baseline 难以覆盖的部分。

### 7.2 Agent 可以修改的东西（MVP）

```text
configs/reward_config.yaml         # reward 各项权重
configs/curriculum_config.yaml     # 训练 curriculum 进度/调度
configs/domain_randomization.yaml  # 训练 DR 范围（不动 evaluation 分布）
configs/ppo_config.yaml            # 有限 PPO 标量参数（见下）
configs/reward_terms.yaml          # 在 reward-term 注册表里 启用/关闭/加权 已有项
src/agent_notes.md                 # Agent 自己的笔记/推理记录
```

**reward-term 注册表**：在 `src/rewards.py` 里**由人**预先实现一组带名字的 reward 项（如 `post_push_pitch_recovery`、`post_push_height_recovery`、`upright`、`drift_penalty`、`action_rate_penalty`、`foot_slip_penalty`…），Agent 通过 `reward_terms.yaml` 启用并设权重，但不改这些项的实现代码。需要新项时——记到 Extension C。

可改内容小结：

```text
reward 各项权重 + 启用/关闭已有 reward 项
curriculum 进度/调度
训练 push 分布 / DR 范围（不能改 evaluation 分布）
有限 PPO 参数：learning_rate / entropy_coef / clip_param / gamma / lambda / minibatch_size
observation 的轻量开关（是否加 history / previous action），如果 config 层支持
```

### 7.3 Agent 禁止修改的东西

```text
评估脚本 evaluate.py / summarize.py
public / iid-hidden eval seed 文件
success metric 与 failure classifier 的核心规则
robot asset / URDF / USD
episode length / total training steps / num_envs（凡改变总样本预算的）
最终评估难度 / eval episodes 数 / eval push 分布
任何 metric 计算逻辑
reward 项的实现代码（只能在注册表里加权，不能改实现）
```

### 7.4 Agent 每轮输出要求

```text
1. hypothesis：当前主要失败原因
2. evidence：引用哪些统计支持（要结合多个指标，不是单看一个 dominant failure）
3. patch summary：改了哪些 config 项 / 启用了哪些 reward 项
4. expected effect：期望哪些 metric 改善
5. risk：可能的副作用
6. rollback condition：什么情况下回滚
7. config diff（git diff）
```

示例：

```json
{
  "hypothesis": "Forward falls and height_collapse co-occur after frontal pushes. Energy is not high, so the policy is under-reacting rather than over-acting.",
  "evidence": {
    "fall_forward": 0.34,
    "height_collapse": 0.18,
    "mean_energy": "below baseline"
  },
  "patch_summary": [
    "enable+weight post_push_pitch_recovery reward",
    "enable post_push_height_recovery reward (active only after push)",
    "increase frontal push sampling in TRAINING curriculum (eval distribution unchanged)"
  ],
  "expected_effect": "reduce fall_forward and height_collapse",
  "risk": "may increase torque and drift",
  "rollback_condition": "mean success does not improve beyond noise threshold, or energy increases >20%"
}
```

这样 Agent 不是简单分类器，而是要结合多个统计量做判断。

---

## 8. Baselines：第一版必须有的对照

### 8.1 Baseline A：No-Agent，**两种预算**

```text
A1（起点预算）：固定人工 reward/curriculum，每候选预算，多 seed → 知道任务难度与种子方差
A2（同算力对照，关键）：给 baseline config 与 Agent loop「同等总算力」
    例：若 Agent 跑了 5 轮 × 3 seed × 预算 B = 15B，
        则 A2 给 baseline 配 15B（多 seed / 训更久），报 best-of-N 和 mean-of-N
```

**A2 是核心控制**：回答「同样的算力，不用 Agent、只是把 baseline 多训/多 seed 取最好，能不能达到一样成功率？」。如果能，Agent 就只是个复杂的烧算力方式。所以第 12 节要记 Agent loop 的总 GPU-hours。

### 8.2 Baseline B：Hand-coded failure→intervention rule baseline（必须有）

不用 LLM 的规则系统，输入 summary，输出 config patch：

```python
if fall_forward high and energy low:
    enable/raise post_push_pitch_recovery; increase frontal push curriculum
elif fall_forward high and energy high:
    reduce action scale; increase action smoothness penalty
elif fall_side high:
    raise roll stability reward; increase lateral push sampling
elif excessive_drift high:
    increase drift penalty
else:
    small perturbation around current config
```

目的不是一定要赢，而是回答「LLM Agent 是否比手写规则更有用」。LLM 赢 → Agent 有价值；没赢 → 贡献收缩为「框架先跑通，下一步增强 Agent 权限/反馈」。

### 8.3 Baseline C / D（可选，后期）：Random config perturbation、One-shot LLM。第一版不必。

---

## 9. 种子方差与 accept/reject 策略

### 9.1 先测 baseline 种子方差（必须）

启动 Agent 前，用固定 config 跑 3–5 个 seed，每个 checkpoint 在 public eval 上评估，得到 `mean_success / std_success / standard_error`。不知道种子方差，就不能判断 +3pp 是真提升还是噪声。

### 9.2 每个 candidate 至少 2 个 seed

```text
普通候选：2 seeds（两张卡并行，GPU0=seedA, GPU1=seedB）
准备接受的候选：追加到 3 seeds
最终 best：3–5 seeds
```

### 9.3 接受规则（看分布，不看单点）

```text
accept 若: candidate_mean - best_mean > max(5 pp, 1.0 * pooled_standard_error)
方差大时收紧到: max(7 pp, 1.5 * pooled_standard_error)
```

第一版不用复杂统计检验，但必须避免接受明显落在噪声里的提升。

### 9.4 周期性重评 best

每接受 2–3 次后，对当前 best 重跑一组 eval seeds，检查是否均值回归；若掉回 baseline 附近，说明之前可能是种子运气，回滚或降低置信。

---

## 10. Evaluation set 设计

### 10.1 Public eval（用于每轮 accept/reject）

固定 seeds、固定 push 分布；Agent 能看到 summary，但看不到也不能改 seed 文件。
> **注意：public 数字是被「选择」抬高的（每轮都对它择优），不能直接作为「最终结果」上报。** 最终结果以 iid-hidden / 重评为准。

### 10.2 IID-hidden eval（防 public 过拟合）

与 public 同分布、不同 seeds、不用于每轮 accept/reject，只在 milestone / 最终检查使用。用途：检测 Agent 或搜索过程是否过拟合了 public seeds。

### 10.3 OOD-hidden eval（后期再做）

更强 push / 没见过的方向组合 / 不同摩擦 / 轻微质量扰动。用途是讲鲁棒性，不是防作弊。MVP 不做。

---

## 11. Multi-fidelity 策略

### 11.1 MVP 只做 smoke test，不做短训排序

```text
config verifier → smoke test → 固定预算训练 → 固定 public eval
```

不要用 short training 排序候选（不默认「短训排名≈全训排名」，对 RL 经常不成立）。

### 11.2 smoke test 只判断「能不能跑」

```text
代码能 import；环境能 reset；跑 100–500 steps 不 NaN；reward finite；
observation/action shape 正确；不会一启动就所有 env fall；能写 log/checkpoint
```

### 11.3 后期若要短训筛选，先验证 rank correlation

随机选 8–12 个历史 candidate，每个跑 short 和 full，算 short↔full 的 rank correlation；相关性低就不能用短训筛选。

---

## 12. 训练预算规则与算力记账

### 12.1 固定预算

```yaml
train:
  total_env_steps: fixed
  num_envs: fixed（或按总样本量归一）
  rollout_length: fixed
  num_epochs: fixed   # 先锁死，避免改变 compute 量
  eval_episodes: fixed
  eval_seeds: fixed
```

Agent 禁止改：`increase_training_steps / episode_length / 降低 eval push magnitude / 降低 eval 难度 / 减少 eval episodes`。

### 12.2 可改但受限的 PPO 参数

`learning_rate / entropy_coef / clip_param / gamma / lambda / minibatch_size`。
`num_epochs` 会改变 compute，第一版锁死。

### 12.3 算力记账（为 8.1-A2 服务）

每个 run 记录 GPU-hours / wall-clock / env steps；汇总 Agent loop 的**总** GPU-hours，用来给 no-agent A2 配同等预算，并报告「单位算力的提升」。

---

## 13. 最小代码结构

```text
humanoid-agentic-mvp/
  README.md
  setup/
    server_check.md
    install_notes.md
  configs/
    baseline.yaml
    reward_config.yaml
    reward_terms.yaml          # reward-term 注册表的启用/权重（Agent 可改）
    curriculum_config.yaml
    domain_randomization.yaml
    ppo_config.yaml
    public_eval_seeds.yaml     # protected
    iid_hidden_eval_seeds.yaml # protected
  src/
    task/
      push_recovery_env_cfg.py
      rewards.py               # 预定义 reward 项实现（protected，Agent 不改实现）
      curriculum.py
      events.py                # push 事件
      terminations.py
    eval/
      evaluate.py              # protected
      failure_classifier.py    # protected
      summarize.py             # protected
    agent_loop/
      build_prompt.py
      run_agent.py
      validate_patch.py
      run_smoke_test.py
      accept_reject.py
      compute_accounting.py    # 记 GPU-hours
    baselines/
      rule_based_agent.py
  scripts/
    train_one_seed.sh
    eval_checkpoint.sh
    run_candidate_multi_seed.sh
    run_baseline_variance.sh
    run_no_agent_equal_compute.sh
  runs/
    .gitkeep
```

`evaluate.py / summarize.py / failure_classifier.py / rewards.py 的实现 / seed 文件` 设为 protected；用 git pre-commit 或 validator 检查 Agent 的 diff 没碰它们。

---

## 14. 分阶段实施计划

### Phase -1：服务器与基础环境跑通（含 go/no-go）

任务：SSH 登录 → 确认 GPU 空闲 → 建工作目录 → 装 Isaac Lab → 跑官方 smoke test A/B/C（**含 GPU-physics 检查**）→ 克隆 Unitree RL Lab → 跑官方 G1/H1 示例(D)。
通过标准：A–D 全过，且物理在 GPU。
**go/no-go**：若 C 在时间盒内（如 3–4 天）始终回退 CPU，切换到第 3.4 节人形退路。不通过则不进入下一阶段。

### Phase 0：不改环境，先跑官方 baseline

熟悉训练命令、日志、checkpoint、评估流程。通过标准：能训练 / 能恢复 checkpoint / 能评估 / 能复现命令。

### Phase 1：实现 Standing Push Recovery 环境

任务：找 push/外力接口（确认是速度 kick 还是力）→ 加 push event → 加 standing command → 加二值 success → 加 failure classifier → random policy smoke → 短训 smoke。
通过标准：环境能 reset；push 确实生效；random policy 大多失败；无/小 push 下 baseline 有希望站稳；failure report 能输出 JSON。

示例 `summary.json`：

```json
{
  "success_rate": 0.42,
  "failure_breakdown": {
    "fall_forward": 0.30, "fall_backward": 0.06, "fall_side": 0.18,
    "height_collapse": 0.14, "excessive_drift": 0.10
  },
  "secondary_metrics": { "mean_energy": 12.4, "mean_action_rate": 0.31, "foot_slip_rate": 0.19 },
  "compute": { "gpu_hours_this_run": 0.0 }
}
```

### Phase 2：训练人工 baseline 并测种子方差

手写初始 reward/curriculum → 训 3 seeds → 固定 public eval → 算 mean/std → 调任务难度到 public success 平均 30%–70%（别低于 10% 太难改，别高于 85% 空间太小）。
通过标准：有 baseline_mean / baseline_std / public eval summary / iid-hidden summary（暂不调参用）。**同时记下单 seed 的 GPU-hours**，为后面 A2 估总预算。

### Phase 3：实现 verifier 和 smoke test

verifier 检查：git diff 没改禁止文件；YAML/JSON 能 parse；Python 能 import；reward finite；环境能 reset；随机 action 100–500 steps 不 NaN；eval seeds 未改；training budget 未改。
通过标准：合法 patch 通过；改 eval seed 的非法 patch 被拒；NaN reward patch 被拒。

### Phase 4：实现 hand-coded rule baseline

读 summary → 出 config patch（覆盖 forward/backward/side fall、height collapse、drift、oscillation、high energy）→ 过 verifier → 能启动训练。

### Phase 5：实现 LLM Agent 单轮改进

输入：任务说明 / 当前 config / 可改文件列表 / 禁改文件列表 / 预算约束 / public summary / failure report / secondary metrics / 上一轮 diff 和结果。
输出：`analysis.json` + config patch。
流程：build prompt → Agent 出 patch → validate → smoke → 2 seeds 固定预算训练 → public eval → 接近阈值则追加第 3 seed → accept/reject。

### Phase 6：跑最小多轮闭环（成功判据 claim-aligned）

轮数 3–5；每轮：best config → Agent proposal → verifier → 2 seeds 训练 → eval → accept/reject；每 2 轮重评一次 best。

**MVP 成功判据（重写）**：

```text
机制层（这是 MVP 的核心，必须达成）：
  1. 全流程无需手工改代码即可跑完 3+ 轮
  2. accept/reject 由「多 seed、noise-aware」的结果正确驱动（不是单点）
  3. 没有通过改评估/训练预算/降难度作弊
  4. ≥3 个「通过 verifier 且能跑起来」的不同干预（不是 3 个口头假设）

改进层（证明 Agent 改进了策略，区分于「只是 loop 在择优」）：
  5. 被接受的提升在 iid-hidden / 重评上仍站得住（不只是 public 涨）
  6. 在同等总算力下，Agent ≥ rule baseline，且优于 no-agent(A2)
```

退路框架（不算失败）：

```text
若 Agent 没明显超过 rule baseline / A2：
结论调整为「框架跑通，机制成立；当前 LLM 在受限(config-level)反馈下未明显超过规则表/同算力训练；
下一步：更丰富 diagnostics 或开放更多代码权限(Extension C)」。
```

> 预期管理：3–5 轮、人形 PPO 方差大、config 级编辑，**很可能只拿到 +2–4pp，落在噪声里**。这不代表 MVP 失败——**MVP 要证明的是「带正确统计的闭环机制」**，成功率大涨是 stretch goal，不是及格线。

### Phase 7：MVP 之后的可选扩展（前面都稳定后才做）

- **Extension A：OOD hidden eval**（更强推扰/不同方向组合/摩擦变化）。
- **Extension B：短训多保真筛选**（先验证 short/full rank correlation 再启用）。
- **Extension C：更开放的代码权限**——这才是「Agent 自由写代码」的归属：允许改 reward function 结构、observation history、PPO aux loss、BC regularization(若有 demo)。需要把 verifier 升级到能挡任意代码破坏。
- **Extension D：in-place stepping push recovery**（允许迈步恢复）。
- **Extension E：经验总结迁移**（standing → stepping）。

当前版本不做这些。

---

## 15. 最小链路日程建议（含 buffer）

> Isaac/Blackwell 环境是最大不确定项，环境阶段留足 buffer；若触发 go/no-go 切到 MuJoCo 退路，日程顺延。

- **第 1–3 天**：SSH + Remote SSH；确认 GPU 空闲；装 Isaac Lab；跑 smoke A/B/C（含 GPU-physics 检查与 go/no-go）。
- **第 4–5 天**：克隆 Unitree RL Lab；跑 G1/H1 官方示例(D)；摸清训练/评估/checkpoint 流程。
- **第 6–9 天**：push 环境（push event + success + failure classifier + summary JSON）；人工核对 classifier。
- **第 10–12 天**：手写 baseline；跑 3 seeds；调难度到 30%–70%；记单 seed GPU-hours。
- **第 13–15 天**：verifier + smoke test + hand-coded rule baseline；自动跑一个 rule candidate。
- **第 16–19 天**：LLM Agent 单轮（prompt → patch → verifier → 2–3 seeds → accept/reject）。
- **第 20–26 天**：多轮闭环 3–5 轮；与 no-agent(A2) 和 rule baseline 对比；iid-hidden 确认；整理结果与问题。

---

## 16. 第一版最终交付物

```text
1. 可安装/运行的 repo
2. 一个 G1/H1 push recovery 仿真任务（或 MuJoCo 退路上的等价任务）
3. baseline 训练脚本 + 种子方差结果
4. evaluation + failure summary 脚本
5. verifier + smoke test
6. hand-coded rule baseline
7. LLM Agent config-patch loop（+ reward-term 注册表）
8. 至少 3 轮自动改进记录 + 算力记账
9. no-agent(A2, 同算力) 与 rule baseline 的对比
10. iid-hidden 上的确认结果
11. 简单 markdown 实验记录
```

不需要：漂亮论文图 / 复杂 ablation / 多 Agent / 真机 / 视觉 / 网页 demo。

---

## 17. 最关键的工程原则

1. **先跑通环境，再谈 Agent。** 物理还会回退 CPU 时，先解决环境。
2. **先测 seed 方差，再判断提升。** 单 seed 不能作 accept/reject 依据。
3. **success metric 保持干净。** 二值成功只衡量任务完成。
4. **Agent 不得改训练预算/评估条件。** 否则提升无法解释。
5. **必须有 rule baseline + 同算力 no-agent(A2)。** 否则证明不了「Agent 有用」也证明不了「不是单纯烧算力」。
6. **最终结果看 iid-hidden / 重评，不看 public。** public 是被择优抬高的。
7. **MVP 只改 config + reward-term 注册表。** 自由写代码留给 Extension C。
8. **双卡优先跑同一候选多 seed。** 当前最大风险是方差，不是候选数量。
9. **MVP 证明的是机制，不是大幅提分。** 提分落在噪声里也不算失败。

---

## 18. 当前最推荐的下一步

不要先写 Agent，也不要先改 reward。先做：

```text
Step 1: 确认 GPU 空闲（vLLM 已 kill，应已空）
Step 2: 装 Isaac Lab
Step 3: 跑 Isaac Lab 官方 headless smoke（A/B）
Step 4: 跑官方 RL 示例（C），并确认物理在 GPU 上（关键 go/no-go）
Step 5: 克隆 Unitree RL Lab，跑一个官方 G1/H1 训练示例（D）
Step 6: 保存完整命令、日志、checkpoint 路径
```

只有 A–D 跑通、且物理在 GPU，才开始写 push recovery 环境。若 C 始终回退 CPU，切到 MuJoCo 人形退路。
