# R3 采集准备核查

后续更新（2026-09-30 14:17 UTC）：清理后的独立资源复查为46,171,140,096 bytes，原逐任务门槛均满足；最终card/driver已在 `../n5_confirmation_execution_r3/` 冻结并经 `final_card_readonly_audit.json` 核对，未部署或启动。下文存储不足及等待freeze的描述保留为先前准备时点。

2026-09-30。只读审查 R3 合同、最终命令、R2 card/driver 后，已获 root 授权准备独立 R3 机器卡构建脚本；未运行采集或模型。现有单任务驱动可复用，不需要调度器。最终机器卡须等存储处理后的新资源证据；当前真实时点的卡及驱动保存在 `../n5_confirmation_execution_r3/pre_resource/`，不作为启动入口。

## 必须更新与保留

| 字段 | R3 处理 |
|---|---|
| `source`、`experiment_files`、`static_preflight`、`cohort_verification` | 指向 485 文件的最终清单、`source_final/`、实际 CPU 结果；远端源为 `n5_confirmation_source_r3_final`。 |
| `main_plan_version`、`run_contract`、`review_protocol`、新增 `dual_review_protocol` / `review_assignment` | 绑定冻结树中的 v1.4.23、R3 合同、两份评阅协议和角色表 SHA；原合同 NOT_RELEASED 是原时点记录，无须改冻结源码。 |
| `tasks.*.cohort/runner_seeds/current_evaluation_binding/static_preflight_result` | 使用本轮已验的真实绑定；Stair `n5_r3_20260930`，Push `1700000000`。训练来源绑定按新 preflight 结果记录。 |
| `commands`、`output_root`、`execution_control_root` | 完整采用 `commands.candidate.final.json`；capture=`n5_confirmation_20260930_r3`，control=`n5_execution_20260930_r3`。 |
| `model_roles` | 观察者共4次 fresh `gpt-6.1-sol/high`；A=SOL-a 40单位×两阶段80次；B=DeepSeek 8×2=16次、SOL-b 32×2=64次。删除旧 CLI 载体/旧模型和512000窗口假定；记录校准实际窗口258400，不能把360000输入代理线当容量保证。 |
| `materials` | pack为 `related_v2` / `p3new.reference_blocks.v2`，数值/媒体准备都显式绑定同一角色表。80份facts封存后才开始report，且仅本人seal。raw exporter已有 `related_v1` CLI表示是上游接口；不能把它误当 pack旧版残留。 |
| `acceptance.calibration_*` | 新20项一次校准18/20 verdict、19/20 kind，达到原0.9 gate；原答、载体及工具trace已封存。这是agreement而非准确率或正式N5结论。 |
| `budget` | 继承最新远端完整6条账本，0.723248796美元；新增最多16次API×0.15=2.40，累计预留3.123248796，小于原15总额。桌面144次正式阶段请求和4次观察者另记，美元成本未知。 |
| `resources`、`latest_resource_prestart` | 保留原逐任务门槛与存储估计；绑定新现场资源，启动时再次检查。不可继承旧资源“READY”事实。 |

保持 checkpoint 路径/字节、任务/地形suite、env选择与全部媒体参数、三seed、数值/媒体选择规则、阈值/分母、无自动重试、旧轮INCONCLUSIVE及原答不变。每个旧顶层字段均在新构建脚本中明确继承或替换，并有未处理字段断言；不是全卡字符串替换。

## 实际阻断和旧账本陷阱

13:58:34 UTC 的 `preparation_resource_readonly.json` 记录：新capture/control根均不存在；GPU0/1分别7/6 MiB，无compute进程；磁盘仅 **19,848,667,136 bytes**，低于 Stair **31,162,953,808** 和 Push **28,276,851,520**。root另行处理存储副本，处理结果未由本审查冒称完成。最终freeze须用处理后的新资源JSON，驱动仍会在每个任务前重验磁盘、GPU与源SHA；现有runner负责实际lease。

本地 R2 `budget_ledger.working.json` 只有3条旧记录，不能作为新总账。真实远端最新账本6条，SHA `40250ddaa8f9b68d350b8c208a31a4dfb7289f39017713bb0742e01c3321aa91`，已下载到 `../n5_confirmation_execution_r3/budget_ledger.inherited.json`。保留两个旧episode01尝试及本次开发Stair02；不得重新清零或只带成功调用。新工作账本从该文件逐字复制，单写者顺序使用。

## 驱动与真实启动入口（尚未执行）

R2 `capture_driver.py` 从相邻card/manifest读配置，只执行参数指定的一个任务。R3副本仅更新卡SHA这一常量；保留源SHA/size、checkpoint SHA、fresh output、磁盘/GPU、`--mode run`、独立日志及无重试。Push须先有本轮 `stair.exit.json` 且returncode=0。驱动不调用评阅，也不把进程成功自动记作诊断或语义验收通过。

待最终机器卡、工作账本、485清单和驱动部署到新control根并由root核验后，每次仅执行下面一条；不能把两条串成自动链：

```text
/data/yuyangliu/envs/isaaclab6_validation/bin/python -B /data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r3/capture_driver.py stair
/data/yuyangliu/envs/isaaclab6_validation/bin/python -B /data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r3/capture_driver.py push
```

驱动内部采用机器卡的完整argv；与已跑CPU preflight只差 `--mode run` 和新output/video路径。每个采集退出后另核实际产物；媒体选择、四次盲观察、root模块核对、完整原量算术/来源核验、实际包容量、任务首项映射与单次模型release均是后续独立步骤。全80份facts barrier和同角色本人seal不由capture driver实施，不能从capture PASS推断已评阅。

构建脚本：`../n5_confirmation_execution_r3/freeze_capture.py <新资源JSON路径>`。当前不应运行它：旧资源不足，脚本会在写最终卡前拒绝。最终冻结源码与计划未修改。
