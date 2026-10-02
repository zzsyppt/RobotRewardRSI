# P3-R.2：Stair CPU 派生与 GPU 租约边界

日期：2026-09-29。状态：本地实现与最小真实生命周期接线验证通过；本记录不代表 P3-R 语义门、作者价值门或真实仿真收益通过。

## 源头与最终边界

原 `build_stair_eval_phase_summary` 在 Isaac 子进程中全量重读数值档案并生成行为报告。父 runner 的 GPU 租约一直持续到该子进程完成，因此仅删除仿真对象或改变耗时统计不能释放资源。

现在沿用同一 job attempt 和 phase receipt 生命周期：

1. Stair 子进程完成 rollout、封存数值档案并核对来源；统一协议且带 public-dev 数值档案时，输出明确的 `status=postprocessing` 交接摘要。
2. 通用 runner 确认拥有的子进程退出，释放现有 GPU lease；GPU 租约计量在实际释放时截止。
3. composition 注册的任务函数 `finalize_stair_eval_phase_summary` 在父进程 CPU 上调用原有唯一的 `build_stair_behavior` / `project_stair_behavior`，再次按当前 job/档案绑定生成行为内容。
4. 完整摘要校验后原子写回；只有此后才生成 `EvaluationPhaseResult`、完成 attempt output hash，供原有 phase receipt 发布。

通用层新增的是中立的 `eval_phase_summary_finalizer(spec, summary)` 接口，不识别足、台阶、姿态或成功语义。Stair 仅为统一协议注册；实际触发限定于待处理的 public-dev 数值采集摘要。Push、sealed phase、没有数值档案的历史路径不执行行为派生。

待处理摘要不接受为历史完成结果；缺少 finalizer 也不能将其封存为完成。CPU 派生失败按 `SUMMARY_INVALID` 结束本次 attempt，无成功 output hash，不触发 GPU rollout 重试。已完成摘要恢复直接读取已封存内容，不访问原档案，也不把新规则补算进历史回执。

## 计量与部署条件

- 原 rollout `wall_time_s` 保留；该 split 增加 `postprocessing_wall_time_s` 和 `postprocessing_cpu_time_s`，并进入最终摘要与 phase result。
- attempt 保留原完整开始/结束时点；`resource_lease_wall_duration_ns` 不包含释放后的 CPU 派生。不能将少算 GPU 秒数误报为端到端加速。
- 派生在 controller Python 运行，需其可用 `numpy` / `torch`；本地 `.venv` 已执行真实数组派生。后续服务器部署应核对实际 controller 环境。没有引入服务、队列、另一套计算规则或 PLAN 运行依赖。

## 验收证据

在仓库根目录运行，使用现有 `.venv/Scripts/python.exe`：

```powershell
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe -m unittest tests.unit.test_phase_postprocessing -v
.venv/Scripts/python.exe -m unittest tests.unit.test_subprocess_runner tests.unit.test_runner_p1_reliability tests.unit.test_stair_phased_evaluation -q
.venv/Scripts/python.exe -m unittest tests.unit.test_stair_behavior.StairBehaviorTests.test_phase_sealing_portable_replay_and_actual_author_payload -v
```

- 新增生命周期测试 3 项通过。使用真实轻量 Python 子进程及真实 `GpuLeaseManager` 文件锁：在 CPU finalizer 中，另一 job 能获取同一物理设备 identity 的锁；派生之前没有完成 attempt 结果；之后 output hash 对应最终 phase。还核对 CPU 时间不计入 GPU 租约、恢复不重新调用派生或重启子进程、派生失败不重跑 GPU，以及无 finalizer 时拒收待处理摘要。
- 原 runner、可靠性与 Stair phase 回归共 135 项：131 项通过，4 项按原平台条件跳过；跳过项不计执行通过。
- 实际 Stair 数值数组 → 待派生摘要 → 唯一任务派生 → 可移植 phase replay → 最终作者 payload 的单项回归通过。该测试由并行的 prompt 修改同步更新调用边界。
- `git diff --check` 对此次生产修改和新增测试通过。

这些测试没有启动 Isaac 仿真、GPU 运算或付费模型调用。它们证明释放顺序、封存与恢复接线；真实集群时长改善仍需后续现有采集过程记录，不预设节省百分比。

## 重放入口补充检查

独立诊断 cohort 落地时进一步发现，旧 `scripts/stair_numeric_dev_replay.py` 在 runner 外持有 GPU lease，范围覆盖到 CPU 派生结束。原上述 runner 内验收仍成立，但不能据此声称该脚本的外层租约也已释放。该入口现已改为环境探测短租约、随后现有 PreGenesis attempt 下的 runner phase 租约；实际文件锁测试确认 CPU 回调时另一 job 可取得同一设备，完成 attempt 在回调后才封存。新脚本区分探测/phase 租约计时，缺失的 phase 时间记为未知。详情、测试范围和源码 hash 见 [独立诊断与重放边界](DIAGNOSTIC_COHORT_BOUNDARY_ZH.md)。旧实验脚本副本及历史结果未覆盖。
## Push 同源收口（2026-09-29）

准备独立确认采集时发现 Push 的 `build_push_phase_summary` 也在仿真进程内重读数值档案，`push_numeric_replay.py` 外层租约又覆盖了结果投影。现复用同一个 runner finalizer：仅 public-dev 且各 run 都有档案时发布 `postprocessing`，将任务拥有的 behavior context 一并交接；进程退出与租约释放后校验档案/计划身份、生成行为、更新成本与 evidence hash，最后 `pass`。缺档案仍原 missing 表达，accept/sealed 正常路径不改变。不新建回调协议或恢复平台，CPU 失败仍沿既有不可重试失败路径。

Push replay 改为环境探测短租约和 runner 的 `PreGenesisExecutionScope` 租约，成本来自既有 attempt receipt；最终投影不持 GPU。`prepare(args, *, gpu_lease_manager=None)` 在 run 模式要求传入 manager；历史录像脚本副本未改。CPU 预检和运行参数保持同一入口。

实际本地验收：行为 producer→待处理→真实 finalizer→任务投影测试、Push checkpoint admission、phased evaluation、共同 postprocessing 共 **12 项通过**；新增 replay 的两项测试 **2 项通过**。后一组用真实文件锁验证探测及 rollout 时其它 job 不能获得 GPU 租约、CPU completion 时可以，且 attempt receipt 到 completion 后才封存；另核新 seed 进入命令、cohort/plan，CPU task state 的扰动相位确实不同。未启动模拟器/GPU/API。

新 seed 不是新资格。Push 旧 TaskSpec 的 invariant 已包含 cohort，因此新 seed 还会自然改变 task protocol/manifest 派生 hash；物理规则和策略 ABI 相同，源训练 binding 保留。不可为了新诊断回写旧资格 hash，实验入口须重建新 seed 的完整合法 composition，并通过任务原 source admission 核原训练来源。
