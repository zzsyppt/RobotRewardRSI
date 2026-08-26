# P6-C 三条 generation 记录有界验证最终验收报告

> 结论：`GO_FOR_P6D`
>
> 证据日期：2026-08-23
>
> 源提交：`415bfad6eb6e85c6cf0cf94851c6b39d1067116b`

## 1. 权威身份

- session：`p6c-415bfad-20260823T042422Z`；
- formal output：`/data/yuyangliu/true_rl_agent_phase_g/p6c_generation_415bfad_20260823T042422Z`；
- local、`origin/true-rl-agent-refactor`、服务器 checkout 与 formal report 均绑定同一完整提交；
- official report：`generation_session_report.json`，SHA-256
  `5b10f1a8e4d851ae542e2e7fd23d44f7f1e56718b3ee5fd58cd9721946c5afcc`；
- 独立 G4 审计：
  `/data/yuyangliu/true_rl_agent_phase_g/p6c_prelaunch/415bfad_20260823T042422Z/p6c_g4_audit_v3.json`，
  SHA-256 `ad691299c0ca5b08383f984a68f514d987c1831cdd3b24d367a9ca34d7266a93`；
- 审计前后 formal output 均为 706 个目录项，metadata hash
  `8140506496bd3a24684d8a6d68587d3d9a28f663c7a48312b15dbedc60cb66b6`，
  独立审计没有修改正式证据。

## 2. 三条 generation 记录与两代科学结果

| Generation | 结果 | Head 变化 | 说明 |
|---|---|---:|---|
| 1 | `paired_authoring_incomplete` | 否 | lane A 用尽 5 次 authoring attempt，lane B 未启动；零训练、零评测，不计作完成的科学 generation |
| 2 | `accepted` | `0.5386 -> 0.8729` | 两条 lane 均 guard-eligible；selector 按 objective 排序，lane B 胜出并通过 sealed |
| 3 | `formal_rejected` | `0.8729 -> 0.8729` | lane B guard 通过但 objective 从 `0.8729` 降至 `0.7336`，正式门拒绝并保持 Gen2 head |

三条 ledger record 连续，parent/resulting head 逐条可重算。具有完整双公开证据的两代中，
selector material reversal 为 0。negative verifier-equivalent trained repeat 为 0；priority/high
frontier 本次没有到达项，因此相应 service 与 high-card recall 是明确记录的 vacuous pass，不冒充有样本
的有效性结论。

Gen2 是本次唯一真正执行双 lane 排序的 generation：objective delta 为 `+0.3343`，高于 materiality
threshold `0.0923`，`ranking_confidence=outside_estimated_noise_band`。Gen3 则在较高 head 区域重现
guard/objective 对冲：guard delta 约 `+0.068`，但 objective delta 为 `-0.1393`，低于 `0.0254`
门槛。它证明配对设计在两条 lane 都可用时确实比较，而一条 lane authoring 灾难时只会退化为过滤；
不能把 Gen1 写成第三个完成的科学 generation。

唯一 accepted winner 的机制运行门通过：dispatcher 调用 48,001 次，
`max_abs_weighted_contribution=10.0`。一个未入选 lane 的新机制贡献恒为 0，已由候选运行门拒绝；
它没有成为 winner、没有改变 head。这证明运行门实际生效，而非只记录指标。

## 3. 生命周期与恢复

最终链恰好 4 段：

1. Gen1 后 `paused/operator_pause`；
2. Gen2 两条 train 与两条 public-dev receipt 已 create-once 后，对准确 launcher PID 执行一次 active
   TERM；launcher 返回 143，segment exit 按设计不存在，owned process group 与 GPU lease 清空；
3. 唯一 recovery 复用前述四个 job，未创建 retry，完成 Gen2 后再次 clean pause；
4. Gen3 正常闭合，lifecycle terminal。

恢复前后 cost head 相同，17 个 logical job 均只有 `attempt_00`：17 reservations、17 results、
0 open reservation、0 retry。没有并发 segment、分叉、选择性丢弃或跨 session 搬运证据。

## 4. 成本与资源

| 项目 | 实际值 | P6-C cap | 结果 |
|---|---:|---:|---|
| Train jobs | 5 | 8 | 通过 |
| Train env-steps | 983,040,000 | 1,572,864,000 | 通过 |
| Eval seed-units | 64 | 94 | 通过 |
| Episodes | 17,920 | 26,320 | 通过 |
| Simulated control steps | 995,763,200 | 1,591,551,200 | 通过 |
| Provider attempts | 10 | 30 | 通过 |
| Provider request units | 184,264 | 9,000,000 | 通过 |
| Provider response units | 111,939 | 983,040 | 通过 |
| Provider total units | 296,203 | 9,983,040 | 通过 |
| Allocated GPU capacity-hours | 6.281334262245833 | 12.5 | 通过 |
| Accounted output bytes | 1,674,207,212 | 3,758,096,384 | 通过 |

两张目标 GPU 使用不同物理 binding，runner-owned child 均看到 logical `cuda:0`。终态没有目标 GPU
compute process、持有中的 formal/GPU lease、owned process-group residual 或同 incarnation 活进程。
父进程 GPU 环境未被改写。

## 5. G4 判定

official report 的 12 个闭合检查全部通过；P6-D 的 5 个入场门——authoring timing 完整、
high-card recall 完整、priority revisit service 完整、negative equivalent trained repeat 为 0、
selector material reversal 缺失——全部通过。独立审计只具有 observer 权限，明确
`acceptance_authority=false`，最终状态为 `GO_FOR_P6D`。

因此 P6-C 不需要重跑。P6-D 必须新建 Genesis、空 ledger、正式 15-generation 合同和终局确认
Plan；不得续写或拼接本 P6-C output。

## 6. 2026-08-24 独立复核补记

- official report SHA-256 独立复算仍为
  `5b10f1a8e4d851ae542e2e7fd23d44f7f1e56718b3ee5fd58cd9721946c5afcc`；
- 3 条 ledger、4 个 lifecycle segment、17 个 job / 17 个 reservation / 17 个 result、0 retry 与
  `6.281334262245833` allocated GPU capacity-hours 全部复算吻合并位于 cap 内；
- `segment-0001` 没有 `exit.json`，与该段由准确 launcher PID 收到 active TERM 的预注册故障注入一致，
  不是证据缺失；其余异常段的持久 exit receipt 已包含脱敏后的异常类型与消息；
- Gen1 暴露的 `expected_shape=[2]`、`observed_shape=[2,2]` 是 smoke fixture 的环境轴与双足物理轴碰撞，
  属于开发工具缺陷，不推翻已闭合 ledger，但必须在下一主线从源头消除；
- 本报告的 G4 权威来自服务器 formal output 内的不可变审计制品。一次性本地审计脚本不属于验收权威，
  也不是重放 P6-C 结论的运行时依赖。
