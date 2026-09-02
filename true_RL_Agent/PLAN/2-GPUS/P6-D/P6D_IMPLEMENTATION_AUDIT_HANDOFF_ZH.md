# P6-D 本地实现与开发侧审计交接

> 交接状态：`POST_P6BC_REWORK_COMPLETE_PENDING_USER_AUDIT`
>
> 实验状态：`PRE_GENESIS_REFERENCE_TRAIN_ABORTED`
>
> 实现父基线：`415bfad6eb6e85c6cf0cf94851c6b39d1067116b`

## 1. 结论与停止边界

P6-D 的 15-generation/finalization 实现曾完成开发侧审计；随后 P6-B/P6-C 独立复核发现 smoke 夹具、
authoring 失败证据输入、paired statistics 语义与启动脚手架复杂度仍需在新主线修正。当前修订只做
静态实现、历史重放和 CPU 测试，完成后停给用户审计，不继续真机实验。

此前基于 `16175c152430018469c22d1866d8e1ca7d4149db` 的一次预启动进入 reference train 后约两分钟被
人工停止；provider attempts、generation ledger 和科学 head 均为 0，相关 GPU/process-group 已清理。
由于本轮 source/contract 已变化，该 output 永久禁止 resume。它既不是 P6-D generation 结果，也不能
用作新提交的 readiness/Genesis 证据。

## 2. 已实现的冻结语义

### 2.1 15-generation 与成本/生命周期

- P6-C 与 P6-D 共用同一个 task-owned bounded-session engine；formal wrapper 固定 15 代，不能由 CLI
  自由改变代数、暂停点或 segment 预算。
- continuation policy v2 固定 clean pause 边界 `3/6/9/12`，正常终态为 15；`max_segments=7`，
  仅保留两段 generation recovery，不形成额外 generation 或预算。
- cost v3 冻结 provider、train/eval/control-step、GPU-hour、retry 和 13.5 GiB output hard cap；
  operation 前 admission、失败/partial 保守结算、resume 累计不重置。
- formal cost replay 只允许 generation terminal head 后恰好一个
  `final-confirmation-settlement`，拒绝任意第二条尾记录或分叉。

### 2.2 Runner-owned final confirmation

- Plan v2 在 Genesis 前 create-once；Reservation v2 只在 Gen15 terminal ledger/lifecycle/cost/resource
  replay 全绿后 create-once。
- Genesis 与 final current-best 使用预承诺、与迭代 split 不相交的同一 7-seed cohort；两次 eval
  顺序复用现有 trusted runner、attempt journal、lease、process group 和 lane-A 完整私有 GPU binding，
  child 仍只看到 logical `cuda:0`。
- final candidate checkpoint lineage 从唯一引入 terminal head 的 accepted generation 的严格私有 train receipt 重载；
  checkpoint、subject、training/capability binding 与 terminal head 必须一致。
- Result v2 不含 `accepted`、`passed` 或 head transition；Result 写入前后复验 ledger checksum，
  不 append ledger、不更新 current-best、不进入本 session prompt。
- finalization 中断后按 create-once receipt/attempt result 原字节复用，不重跑已完成 eval。

### 2.3 Readiness v3、CLI 与报告

- readiness v3 严格区分 `formal_fresh_absent`、`formal_nonterminal_resume`、
  `formal_terminal_finalization`、`formal_terminal_audit_only`，并绑定准确 source/session/plan/cost/
  ledger/lifecycle/attempt/final artifact/argv/probe 身份。
- P6-D provider 门只检查本地 credential 边界，不发网络请求；receipt 不保存 secret、secret hash、
  环境文件内容或路径。准确环境文件路径由 formal command hash 间接绑定。
- resume 磁盘门按剩余可信 output 动态计算，且不低于 30 GiB；preflight 真实调用 compute-idle checker，
  不再用 mock 掩盖 stage 白名单。
- readiness receipt 在 launcher 创建任何 attempt/子进程前 create-once 消费；第二次消费 fail closed。
  当前受支持 CLI/launcher 只接受 v3；无入口、无覆盖的 v2 分支已经删除。v1 只保留被 v3 复用的底层
  mapping 校验核心，不再宣称当前提交可加载历史 v1/v2 readiness receipt；历史证据应在其冻结提交上审计。
- report v5 从 ledger、receipt、attempt、lifecycle 和 cost chain 重算；顶层另列纯 ledger 派生的
  `scientific_output`，明确 accepted 数、有效科学代数和 head 是否移动。正式写入前证明报告编码后仍落在
  已结算 `accounted_output_bytes` 内。

## 3. 实现中发现并在源头修复的缺口

1. formal reference task identity 曾使用目录字面值，已改为 task-owned `STAIR_TASK_ID`。
2. 旧 formal cost replay 不认识 terminal 后唯一 final settlement；已在 cost v3 replay 规则中显式表达，
   report/readiness 共用同一规则。
3. generation 终态资源证据曾可能把之后的 final attempt 混入代内闭合；现按证据作用域严格分离。
4. report 生成前只检查当前落盘值，未证明 report 本身仍在 output reserve；现按 canonical 编码长度检查。
5. v3 readiness 虽校验 fresh，却可被同一 launcher receipt 重放；现增加相邻 create-once 消费标记，
   并覆盖真实 launcher 调用链。
6. formal terminal audit 最初仍走 helper context，可能绕开 v3 launcher/readiness 绑定；现按正式
   terminal-audit subject 消费完整启动上下文。
7. 全量测试发现一个用 `SimpleNamespace` 仿造 eval context 的旧夹具缺少新增 `seed_namespace`；
   产品入口的 frozen dataclass/validator 本来始终提供该字段，已把夹具补成结构真实的 formal namespace，
   没有在产品代码增加掩盖合同缺失的 fallback。
8. Stair smoke 的 `num_envs=2` 与双足 trailing dimension 撞维，导致 `[2,2]` 看起来像环境轴错误；现用
   5 并由碰撞回归锁定，不改变 reward 或公共 smoke 合同。
9. 最近一代 authoring incomplete 会遮住上一代两条有效 trial evidence；现由 Stair task-owned public
   projection选择最近一代结构完整的 public-accept-vs-parent 比较，并严格裁剪公开字段。
10. guard-filtered 只有一条 eligible lane 时曾继续输出成对数值；当前 schema 显式
    `paired_statistics.status=not_applicable`，冻结 v3 result/contract 仍严格重放。
11. P6-A2 一次性 paired-ready probe 与当前主线不消费的 P6-C runtime-monitor 证据链已退役；资源真相
    收敛为 prelaunch compute-idle、runner-owned cleanup、generation closure 和最终 G5。
12. 初版 B2 比较在真实 PublicContextBuilder 脱敏后尝试严格重载完整 current-best，组合链正确报
    fields mismatch；现只从 scrub 后仍存在的 protocol-bound public score stats 构造 task-owned 比较
    视图。完整 loader 未放宽，sealed 字段仍不可进入 prompt；只有 lane A/B 两条均 public-complete 时
    才更新“最近完整 pair”，单 lane incomplete 不会遮住上一代完整证据。
13. readiness v1/v2 producer、网络 readiness collector 与低层 `--validation-session` 旁路已经删除；
    v2 的 635 行无入口分支和 11 行无效 runtime-monitor 墓碑也已删除。v1 mapping validator 仅作为 v3
    synthetic-subject 校验的共享核心保留，历史 v1/v2 receipt 由对应冻结提交负责解释。
14. 旧 15-generation production-shaped 用例的 head 从未移动，导致 candidate finalization 分支零覆盖；
    当前用例在 Gen1 生成两条真实候选、完成选择并接受一条，随后证明 terminal reservation 的 checkpoint、
    source manifest 和 final anchor 都来自该 candidate head。没有 Torch 的本地环境在该用例中用进程内
    sealed source qualification 替代 isolated worker，并跳过 callable/determinism；worker 进程握手、超时和
    Torch 执行由独立测试与后续服务器门承担。该用例仍真实执行 source load、双 lane runner-shaped
    训练/评测与 finalization。
15. candidate finalization 曾对可空 `subject_manifest_hash` 使用 `str(...)`；现三个 argv 边界都先显式执行
    `require_sha256_digest`，不会把字面量 `"None"` 送入子进程。

## 4. 三视角静态审计

### Authority / evaluation

- 选择、formal acceptance 与 head transition 仍只来自原 generation ledger 路径。
- final Plan/Reservation/Result 均为 task-owned 严格 schema，显式 `acceptance_authority=false`；
  finalization 前后 ledger/head 不变。报告把 `engineering_status`、`acceptance_authority` 和
  `scientific_output` 分开，工程 GO 不能再被解释为 scientific pass。
- terminal 后普通 generation resume 被拒绝；只剩 finalization 或 terminal audit-only 两条准确模式。

### Runner / resource

- parent 不直接启动 Isaac final eval；仍由 backend materialization + `run_eval_phase` 创建 runner-owned
  child，并复用既有 attempt/retry/lease/process-group/receipt 边界。
- 两个物理 UUID、child logical `cuda:0`、shared retry entitlement、open reservation 和最终资源闭合
  都进入严格 replay/report 检查。
- final settlement、attempt chain 和 output bytes 使用同一 cost v3 真相，不另建旁路计数器。

### Prompt / memory

- `agent_core`、prompting 和 memory 中没有 P6-D/Stair/15/暂停边界常量，也没有 final Result/report
  消费入口。
- sealed/private/final diagnostic 不进入 public context、experiment memory 或下一代 prompt；Gen15 后
  不存在下一次 provider prompt。
- exploration/frontier、two-lane authoring、旧任务 plugin 和 width=1 权限没有扩大。

公共层除 continuation policy v2 外，另增加一个只读、任务中立的 train-receipt loader，与已有
eval-phase loader 对称，用于重启后恢复严格 checkpoint lineage。它不增加持久化字段、不改变旧 record
校验/字节、不含 Stair/final 语义；调用方仍须先 replay generation chain。此项已在冻结计划中显式记录，
建议用户审计时重点确认该边界是否可接受。

## 5. 测试与检查证据

以下只保留为上一轮 P6-D 实现基线的历史记录，不是当前提交门：

- production-shaped fake provider/runner 真实走 15 条连续 ledger、3/6/9/12 pause/resume、一次 unclean
  recovery、terminal finalization 中断复用、两次 final eval、Result/report/readiness 四模式：通过，
  用时 324.608 秒。
- P6-D/P6-C/legacy 聚焦矩阵：173 项通过，2 项按平台跳过。
- 当时 readiness v1/v2/v3 共 20 项通过；该数字对应现已删除的 v2 开发分支，不能用于当前主线。
- 新 seed namespace 相关 execution source/entrypoint/phased/final 回归：49 项通过。
- 全仓 `compileall` 与 `git diff --check` 通过；本环境没有可用 Ruff 命令，因此不把 Ruff 记为已通过。
- 第二轮 clean Windows 全量：`Ran 1504 tests in 580.541s`，`OK (skipped=29)`。

第一轮全量为 1503 项，唯一错误就是第 3 节第 7 项的结构不完整测试夹具；没有产品路径失败。修复夹具并
加入 readiness create-once launcher 回归后，第二轮以 1504 项 clean 全量实际闭合，没有用定向测试
推断全量绿。29 项 skip 是当前 Windows/平台门控；Linux 全量仍明确留在用户审计之后，不能由本结果替代。

本次 P6-B/P6-C 独立复核后的当前证据（替代上述旧基线作为提交门）：

- Windows `.venv` 对 formal integration 模块以外的全仓测试：`Ran 1471 tests in 265.574s`，
  `OK (skipped=26)`。被单独隔离的模块含一个重型 formal 用例和一个 Torch-gated 三代用例；后者在本环境
  明确报告 `OK (skipped=1)`，不是静默遗漏。
- 15-generation production-shaped formal 用例现于 Gen1 训练/评测两条 candidate、接受 lane A 并移动
  head；随后真实走完 15 条 ledger、3/6/9/12 pause/resume、terminal finalization crash/reuse 和两次
  final eval。它单独复跑为 `Ran 1 test in 576.545s`、`OK`，并从 raw private phase receipt 断言 final
  manifest/checkpoint 均属于 candidate head、不同于 Genesis。
- task-focused 矩阵 166 项通过（1 项 Windows skip），legacy/plugin/兼容矩阵 144 项通过（1 项
  Windows skip），launcher 模块 13 项通过（1 项 Linux-only skip）。
- 全仓剩余回归首次暴露 `_formal_args` 测试辅助函数没有携带 `--formal-session`；产品 CLI 的 fail-fast
  保持不变，辅助函数从源头修正后 launcher 模块与 1471 项剩余全仓回归均重新通过。
- 全仓 `compileall` 与 `git diff --check` 通过；本环境仍无可用 Ruff 命令，因此不把 Ruff 记作证据。
- 相对审计父提交 `2d9ca9d`，本轮代码与测试 tracked diff 为 585 行增加、1,424 行删除，净减少 839 行；
  其中删除 readiness v2
  产品分支、其专属 recovery 测试和 runtime-monitor 墓碑。历史 outputs、转移 tarball 与用户恢复材料
  均未删除。

本轮没有执行 Linux、SSH、Isaac、GPU、provider、硬件 preflight/readiness 或 G5；上述 Windows/fake
证据不能替代下一轮真机门。

## 6. 本轮停止边界与下一步

先完成本轮代码/文档 diff、定向与 Windows 全量回归、legacy replay 和三视角复核，然后提交给用户做
新的静态审计。本交接完成即停止；不得运行 SSH/Linux GPU 门、真实 Stair teardown、硬件 preflight、
provider 或 generation。用户明确批准下一轮真机门之后，才可从准确的新提交、新 output、新 v3
readiness 和新 Genesis 重新开始，禁止 resume `16175c1` 的已停止 output。
