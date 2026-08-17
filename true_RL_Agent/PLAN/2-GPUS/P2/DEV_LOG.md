# P2 开发日志

## 2026-08-16

### 目标与边界确认

- 重读双 GPU 主计划、P0/P1 记录和当前代码；
- 使用 CodeGraph 从 `run_candidate_round()`、ledger append、runner 与插件调用链定位真实拆分边界；
- 参考 `D:\humanoid_agent\codex` 的适用思想仅限稳定执行身份、结果与上层提交分离、唯一 commit owner；
- 明确不引入通用 agent/workflow/Graph 框架，不修改任务语义，不启动 P3/P4。

### 基线

- Git 基点：`81565dd970a6726fda6a219704bb9a18651e6bb8`；
- 工作树包含已存在、未提交的 P0/P1 修改，未覆盖或回滚；
- P2 前全量：1183 tests，21 skipped；
- P2 前聚焦：144 tests，全部通过。

### 实施

1. 建立 task-neutral authoring artifact typed-size contract；
2. 让 producer、Ref、bounded reader、resume/verifier 共享各类型独立 cap；
3. 从现有 verifier AST 链投影 mechanism operation delta v2；
4. 新建私有 TrialReceiptJournal，封存 contract/train/eval/final receipt；
5. 将 `run_candidate_round()` 拆成 executor 与唯一 commit，并保留 width=1 wrapper；
6. 在 ledger 现有锁内增加当前 writer record 的 idempotent append；
7. 修正 mixed legacy ledger 下 exactly-once identity 只作用于当前 schema；
8. 增加 train/eval/final 复用、tamper、session/epoch/round 搬运与 typed-cap 回归。

### 验收

- P2/纠偏直接聚焦：101 tests，全部通过；
- prompt/memory/live/Push/ledger 邻接集：221 tests，全部通过；
- Windows 全量：1197 tests，21 skipped；
- WSL 非 GPU 定向：101 tests，全部通过；
- Python 3.10 AST grammar：10/10；
- compileall：通过；
- git diff check：通过；
- 未调用 GPU、API、服务器；未 commit/push。

### 阶段结论

P2 的实现目标闭合：trial worker 可以产出完整、可验证、可复用的私有执行 receipt，但不能独立改变科学历史；旧 width=1 仍由同一 wrapper 单次提交 round-v7。P3/P4 尚未开始，等待用户验收授权。

### 用户复审后的必要纠偏

- 修复 Stair feedback 列表缺失/`None` 时压缩路径抛 `TypeError`；七种已报告形状在容量压力下均稳定进入 minimum core；
- 把 provider task section 清单与 task-only 例外收敛成单向派生关系；
- 让 Trial contract 在 publish/load/commit 三个实际决策边界分别绑定 record 与当前 inputs；
- ledger identity 冲突补充首个差异字段和双方 record hash，identity 仍有意不包含 candidate/parent；
- 同类扫描只额外发现并修复 `detail_omissions` 的同形状列表归一化；未发现第二处 contract 校验后丢弃、第二套 task section 完整清单或同类 ledger identity 写入路径；
- provider attempt 默认值与 artifact 上限、四类 Trial receipt 均确认属于合理现状，不做整理性改动。
