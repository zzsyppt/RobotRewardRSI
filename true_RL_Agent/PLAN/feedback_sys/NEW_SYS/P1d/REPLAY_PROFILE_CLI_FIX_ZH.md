# replay CLI profile 参数入口窄修复

基线 `c8600e206b00b11be4e0225ee84b70546ada97bb`。用户追加授权，独立新commit；
保留GPU binding修复，不修改任何src/P2a数值、阈值、模板、三源产物或P2a验收记录。

## 实证与实现

已只读获取服务器原始失败日志：
`/data/yuyangliu/isolated/p1d_gpu_aaf67aa_20261003_qualification01/cli_fix_c8600e20_20261003/cpu_profile_preflight.log`。
1515B，SHA256 `7bbaaa798d741cf24822b9947e4d7aa52edc48d551e1b770aa5f795a5bb00b9b`，
本地原文 `outputs/profile_cli/original_cpu_profile_preflight.log`。
原CLI报 `unrecognized arguments: --behavior-feedback-profile trajectory_numeric_v1 --trajectory-report-max-bytes 8192`，
CPU解析exit2，尚未启动GPU。

仅给原 `scripts/stair_numeric_dev_replay.py` 添加已有的两个参数，显式传给
`StairClimbingCompositionProvider.resolve(values=...)`。choices/default复用
`trajectory_pipeline.PROFILES/LEGACY_PROFILE`，预算默认8192；有效性检查全部复用现有composition。
与run_generation_session/run_loop相同，profile和budget经provider进入正常runner的CPU finalizer。
未增加wrapper、手动建卡路径或GPU设备设置。没有强开capture：`--capture-numeric off`保持关闭。

默认仍为legacy_process_v1；默认与显式legacy的task_defaults相等，两profile键继续从legacy身份
配置中省略。trajectory/8000实际传入finalizer，训练binding和评价protocol保持相等。
8000 preflight通过仅表示配置合法，不承诺某个实际档案的18必保槽一定装得下。

## 验证

以下三模块最终 **26 PASS、0 SKIP、0 FAIL，34.423s**：

```text
.venv/Scripts/python.exe -B -m unittest
  tests.unit.test_stair_numeric_dev_replay
  tests.unit.test_stair_trajectory_pipeline
  tests.unit.test_gpu_binding_projection_contract -v
```

新增4项入口测试验证：parser与正常provider的choices/default一致；未知profile和非整数budget
解析exit2；budget为0/-1/8193及legacy+8000由既有validator拒绝，并检查错误中明确包含budget、
GPU discovery/probe/launch均未调用；真实provider→finalizer参数；真实子进程help及trajectory preflight。
既有host cuda:1→child cuda:0、UUID/lease、formal/diagnostic、来源和capture on/off测试继续通过。

额外保存原入口真实命令结果，无patch、无生产wrapper：

```text
.venv/Scripts/python.exe -B scripts/stair_numeric_dev_replay.py --help
exit 0，help列出两个新增选项。

.venv/Scripts/python.exe -B scripts/stair_numeric_dev_replay.py
  --checkpoint outputs/profile_cli/fixture/synthetic.pt
  --train-summary outputs/profile_cli/fixture/synthetic_train_summary.json
  --protocol configs/tasks/stair_go2_12dof_unified_protocol_v3.json
  --output outputs/profile_cli/preflight --device cuda:1 --mode preflight
  --capture-numeric on --behavior-feedback-profile trajectory_numeric_v1
  --trajectory-report-max-bytes 8192
exit 0，PREFLIGHT_PASS；planned_public_dev_episodes=864，phases_executed=[]。
```

实际绝对argv、原日志及hash见 [REPLAY_PROFILE_CLI_VERIFICATION.json](./REPLAY_PROFILE_CLI_VERIFICATION.json)。
本地preflight使用明确synthetic训练契约夹具和非策略checkpoint字节，只验证CPU入口，
不是新真实策略评测、产卡结果或GPU资格。没有GPU/训练/LLM/服务器写入。

独立只读审查及提交后停止；修复后原CLI的服务器GPU资格仍需原资格线程实测回执。
