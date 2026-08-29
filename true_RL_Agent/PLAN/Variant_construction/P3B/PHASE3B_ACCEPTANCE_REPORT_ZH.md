# Phase 3B 验收报告：C2 roughness OOD 与 failure-feedback 校准

## 1. 结论

**GO。** C2 已作为唯一 G1-37 × Stair 格子中的评测机制 OOD 落地，统一协议已由 `draft`
转为 `active`。训练分布、课程、seed、PPO、奖励候选权限和共享矩阵层未因 C2 改变；没有新增
C2 cell、manifest、reference、Genesis、ledger、current-best 或 LOOP 分支。

本结论只证明考卷、观测、证据、反馈、身份和成本可靠。Phase 3B 没有执行完整 reference 训练、
候选训练或 LLM LOOP，因此不声称策略在 C2 上成功或已有性能提升。

## 2. 最终身份

| 身份 | 冻结值 |
| --- | --- |
| cell | `stair.g1_37dof.unified_v1` |
| protocol state | `active` |
| protocol hash | `3682d316fc1319ae2660d55d2153afb3f17d1003c2f1e541d8e977899d93b2d5` |
| task protocol hash | `5488c5616e85c2846a6206d4dfcb19832486274075267e7003a5855c8c1a848f` |
| resolved task protocol hash | `401a22a8f85b54cc287bc07a38586699f82b2146faa65018cc4653b1a819c242` |
| resolved manifest hash | `0c9156a2cca0a6dbe03bb38a5e441cf4b527f7a4bc6b10d423b6186af595b5be` |
| training profile hash | `d222a1ddc2ed52ed6feae43d8413b6364576880ac4febbc09e6ac7a9122028d1` |
| evaluation suite hash | `e9fb21758761a2d2f38996e8946fdcbd1539f6cb63c3e72fa0f5307f5d9f3153` |
| acceptance protocol hash | `d18b5345c8b3112546af401a8dd1560e6c6e840ad8c5ef4d102bd3e1ea5b4c24` |

`protocol_state` 不进入 task projection，因此 draft→active 是唯一允许的状态跃迁：训练、评测、
acceptance、reward authority 和 task protocol hash 均未改变；manifest 中的状态哈希随之重签。

## 3. 严格验收结果

### G1：单格边界与训练排除

- catalog 只含一个 active Stair cell；C0/C1/C2 只作为 task-owned terrain regimes；
- 统一训练 profile 只允许 `c0_regular`、`c1_irregular_flat`，显式禁止 `c2_rough`；
- 真实 Isaac/RSL-RL 128-step smoke 使用 16 个环境，初始分配严格为 C0=8、C1=8、C2=0；
- smoke 占完整训练预算 `6.510416666666666e-07`，未用正式训练结果调地形或阈值；
- `src/`、`scripts/` 中 `is_irregular_cell` 命中为 0；C2 标记在 Stair TaskPlugin 外命中为 0。

### G2：C2 几何、身份与成本

- C2 幅值阶梯为 4/6/8 mm，硬上限保持 15 mm；start/top platform、riser 边和 nominal goal 未改变；
- public_dev、public_accept、sealed_validation 各 24 道、8 strata × 3 道；三套卷 72 个正式 geometry digest 两两不重合；
- 每套卷恰有 3 道 C2，三套卷使用相同 surface-profile 难度多重集、不同 C1 骨架；
- 每 cell/seed 从 4 episodes 收敛为 3，15 个 formal seeds 共 1080 episodes，低于 P3R 的 1260 上限；
- `surface_profile` 已进入 ordered geometry v3 的 digest，修改 surface 参数会改变 suite/cell 身份。

### G3：真实 mesh 与 surface query

- 9 个正式 C2 cell 均在真实 Isaac 运行时生成；每个 cell 的 10 个 solid mesh 全部 watertight；
- mesh 顶点 oracle 与 task-owned query 的最大绝对误差为 `3.3306690738754696e-16 m`；
- 观测到的最大表面偏移为 `0.008000000000000007 m`，未超过 suite 幅值或 15 mm 硬上限；
- start/top platform 保持 flat，C2 只作用于实际 tread 内部，decoder 输出有限的局部表面 clearance。

### G4：可观测性与 failure-feedback

- 真实 evaluation 场景关闭 observation corruption 后，187 条成对 height-scan ray 全部有限；
- 120 条 ray 的 C1/C2 差异至少 1 mm，最大差异 `4.52423095703125 mm`，超过预注册的 8 条/2 mm 下限；
- 训练期 height-scan 仍使用冻结的 `UniformNoiseCfg(±0.100 m)`（标准差约 57.7 mm），最大
  C1/C2 ray 差仅为该标准差的约 7.8%；因此上述结论只证明 mesh→sensor 物理链可测，不声称
  策略会利用该信号。C2 按 surface/contact mechanism shift 解释，行为有效性留给 Phase 4 reference；
- 保持相同局部表面 clearance 的 C1/C2 成对样本，其 primary reason 与 diagnostic flags 完全一致；
- 保持 world-z 的 896 个边界样本中有 82 次阈值翻转，全部位于 6 mm roughness 幅值带内，带外翻转为 0；
- `insufficient_clearance=0.03 m` 等任务阈值未放宽，LLM 只看到 public_dev 的 stratum/failure 聚合，不得到 public_accept 或 sealed 题面。

### G5：回归与可复算性

- 完工后审计的 P3B 契约/表面/ordered-eval/terrain-evidence/runtime-pipeline 定向回归
  `67 tests` 通过、`2 skipped`；P3B verifier 自测 `4/4`、legacy Push 定向回归 `76/76` 通过；
- 完整非 GPU 回归 `1525 tests` 全部通过，`30 skipped`，无 failure/error；
- active entrypoint 使用真实 resolved manifest 正向验收；资格哈希篡改在程序资格与执行授权绑定处被拒；
- 重复的 Phase 3R draft 期断言已删除，只保留一条使用合成 draft 协议的 catalog fail-closed 负测；
- prompt 的 8 个 strata、逐 stratum claim kind、objective、guards 与 C2 exposure 均从 active
  suite/profile 派生；P3B verifier 自测除运行回执/snapshot 篡改外，也会拒绝 prompt 少报 C2；
- runtime receipt 的源码闭包只覆盖实际参与 mesh/scanner/decoder probe 的 11 个文件；纯
  prompt 与 post-rollout scoring 文件由静态 contract snapshot 验收，不再让非物理修复伪造 GPU
  证据失效；
- 旧 P3R/P3A 回执继续作为其冻结身份下的历史证据，不被重写成 P3B PASS。

## 4. 权威证据链

- [C2 真实运行回执](PHASE3B_RUNTIME_RECEIPT.json)：receipt hash
  `e8a8e50e5c2225d83d76d0f73c466483b28d86fe7d398750daf85025a66c9a12`；
- [训练排除回执](PHASE3B_TRAINING_EXCLUSION_RUNTIME_RECEIPT.json)：receipt hash
  `a51d237864d46d5bf2aacad4a056aef830076e99cb4721528466cb23d0b41140`；
- [契约快照](PHASE3B_CONTRACT_SNAPSHOT.json)：当前 snapshot hash
  `92b94b749e490e45b8ad9daf70079299174d0827a544b3c930d980b22474b861`。P3B 完工时的历史 hash
  为 `ac32b92e592eddd6597b2409993de26b385f6a902dc6e447ad78bd198c13625f`；Phase 4 预实验审计先后因
  provider 评分语义升级为 `public_dev-only`、family mask 收敛为 floor-only、prompt/scoring 闭包、
  calibration 缺族聚合及 stratum 截断真值修复而重签当前静态快照。快照明确列出
  historical/current contract transition，并冻结
  `rollout_geometry_or_training_changed=false`；两个 GPU 运行回执未重写；
- [验收器](verify_phase3b.py)同时复算配置 hash、单格 manifest、三套卷隔离、C2 难度等价、
  运行回执、训练 exposure、prompt↔suite 一致性、task ownership 和最终产品源码闭包。

可复现命令：

```powershell
.\.venv\Scripts\python.exe PLAN/Variant_construction/P3B/verify_phase3b.py
.\.venv\Scripts\python.exe -m unittest PLAN.Variant_construction.P3B.test_verify_phase3b
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

## 5. 其余审计项裁决

- `F_c2` 只有一个 stratum 且 family 等权：真实但属于预注册研究权重。三 family 均合格时 C2 占
  `J_OOD` 的 `1/3`、`J_primary` 的 `2/9`；不改成按样本量加权，Phase 4 必须报告实际 mask/权重和
  paired uncertainty。
- 3 episodes 的二元 success 粒度确实较粗，但 `S_stratum` 的另一半是连续 progress，正式
  candidate/head 比较按 seed 配对，sealed consistency 比较聚合 `J_primary`。在没有行为数据证明
  判定不稳定前，不突破 1260 成本上限、不改 acceptance threshold。
- 三套卷复用同一 C2 surface-profile 难度多重集是有意的难度对齐；hidden 部分是骨架几何和 seed，
  不是未见 roughness generator。当前只 claim 相对 flat 训练分布的机制迁移，不 claim 对新粗糙度
  实现的泛化。

## 6. Phase 4 入口边界

Phase 4 必须在本 active identity 上完成 official reference 资格化，机械产生并冻结
`objective_family_mask_hash`，再启动固定预算的 reward LOOP。现有 legacy equal-height final-confirmation
计划不能冒充 unified ordered suite 的最终结论；相关统一 reference、family-mask、candidate campaign 与
final-confirmation 绑定属于 Phase 4，不是 Phase 3B 的性能结论。若 `F_c2` 未达到 Phase 4 冻结的 reference floor，它在该
epoch 退出 objective 且不得作 C2 泛化 claim；不得据此回头改训练噪声、突破 15 mm 或结果驱动加题。
