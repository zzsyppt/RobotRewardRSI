# Phase 8E：单 Agent code-level cold-start 闭环骨架

日期：2026-06-25

## 设计判断

Phase 8E 的目标不是立刻开启正式训练，而是把 code-level cold-start loop 的安全骨架搭起来：

```text
public-only prompt -> candidate package -> Phase8D verifier -> public decision/ledger -> accepted recipe inheritance
```

当前阶段不启动 Isaac/RSL-RL 长训，原因是：

- 8A/8B/8C/8D 已经分别解决了最小 runner、预算口径、隔离加载、candidate verifier；
- 8E 的第一个风险不是“训练时间”，而是闭环是否会把非 public 信息、候选自报信息或 runner 权限混进 Agent 输入；
- 正式训练应由 immutable runner 显式启动，不能由候选代码或 prompt 侧绕行；
- recipe inheritance 是 ENPIRE-like evolution 的关键，如果没有 ledger 和继承，本系统只是安全 sampler。

因此本次只实现 8E 的必要功能，不提前做 Phase 9A best-of-K，不调用 LLM API，不启动正式训练。

## 实现边界

已开放：

- public-only prompt package；
- LLM-authored candidate package 的 8D 验证入口；
- public result 的 allowlist 投影；
- `pending_training` / `promising` / `accept` / `reject` 决策；
- JSONL ledger；
- accepted recipe 复制为下一轮 current best；
- 一个最小 declarative LLM-authored candidate 示例。

未开放：

- 正式 GPU 长训；
- public_dev 候选池排序；
- Phase 9A best-of-K；
- Git evolution / branch lane；
- runner hook / PPO hook / optimizer hook；
- eval observation/action wrapper；
- checkpoint selector；
- success metric 修改。

## 决策规则

8E 区分三种状态：

1. `pending_training`
   - candidate 通过 8D；
   - 还没有外部 public result；
   - 不允许 accept，也不允许 recipe inheritance。

2. `promising`
   - 只有 public_dev；
   - 至少达到 absolute floor；
   - 仍然不允许 accept。

3. `accept`
   - 8D verifier pass；
   - public_accept 存在；
   - budget / OOD stress / motion quality gate 都 pass；
   - near-zero baseline 下超过 absolute accept floor，或稳定期超过 noise-aware delta threshold。

这样处理是为了避免把 public_dev 同时用于“挑候选”和“最终接受”，也避免 cold-start baseline 接近 0 时 pooled-SE 规则失真。

## 新增文件

```text
configs/cold_start/phase8e_loop.yaml
src/humanoid_agent/cold_start_lab/__init__.py
src/humanoid_agent/cold_start_lab/phase8e.py
scripts/phase8e_loop.py
scripts/phase8e_selftest.py
examples/cold_start_candidates/declarative_curriculum_candidate/
```

## 验收覆盖

selftest 覆盖：

- prompt-only round 能生成 public-only prompt；
- benign float `0.820012` 不会被误判为 protected seed；
- exact protected numeric token 会被 public-only scanner 拒绝；
- LLM-authored declarative candidate 先过 8D verifier；
- 无 public result 时只能 `pending_training`；
- 只有 public_dev 且达到 floor 时只能 `promising`，不能 accept；
- public_accept + required gates 后可以 accept；
- accepted recipe 会被复制为下一轮 current best；
- candidate 文本产物泄露 forbidden token 会被 8D 拒绝。

## 本阶段结论

Phase 8E 的“闭环骨架”已经具备：Agent 可以收到 public-only prompt，产出候选包，候选包必须先过 8D，外部 public result 才能驱动 ledger 和 recipe inheritance。

下一步如果要进入正式训练，应只新增 immutable runner 的训练/eval执行接线，并继续保持：

```text
candidate = training components
runner = loop + budget + eval + checkpoint + accept/reject
```

不能把训练循环控制权交给 candidate。
