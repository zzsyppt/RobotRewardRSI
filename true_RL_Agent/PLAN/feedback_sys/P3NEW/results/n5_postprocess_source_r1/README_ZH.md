# N5 独立 CPU 后处理源码树

状态：`CPU_SOURCE_READY`。远端根为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_postprocess_source_r1`，控制/验证材料在同级 `n5_postprocess_source_r1_inputs`。正在运行的 `source_n5_preflight_r4` 未打开或修改；本次未读新 N5 结果，0 provider/model 请求、0 GPU/训练/采集。

共445个唯一文件：r4 runtime及四助手400文件＋最终 `experiment_files.final.json` 的46个 `frozen_files_final` 原件－4个相同重合文件＋3个既有依赖。保持repo相对布局，全部原字节复制；无生产代码、方法、提示或协议修订。源码manifest SHA：`9b19b13e42299bc8b78a9a34e332fc99a0dea7e5ac945f8ea077d8425f8b4078`；tar.gz SHA：`90429070f891615a0b18d1c18a34917a0bdd8426f8566c41189286acf76486b5`。

补入 `P3R/confirmation_context.py`、`P3R/author_ceiling_probe.py`、`P3/author_experiment.py`。三者已在最后N4的 `preparation_receipt.json` 于2026-09-30 01:02:39 UTC登记，其所绑定 `stage.draft.json` SHA仍一致；正式release继续登记相同字节。当前本地、旧远端 `text_author_smoke_r1` 原件和独立树的三个SHA全相同，详见 `dependency_supplement_provenance.json`。这是采集前既有依赖的封存遗漏补录，不是采集后的代码修订。

2026-09-30 02:50:53 UTC远端CPU验证结束。14条 `--help` 与1条真实导入检查全部exit 0；三个prepare/export入口、媒体提取/选择/观察者准备、独立算术脚本、评阅准备入口及两个旧作者helper均可导入。143个运行库导入均来自新树src，13个显式模块/动态加载路径具有真实 `__file__`，没有伪造路径。检查前后445文件逐字节一致，旧远端三依赖原件仍一致。原stdout/stderr、argv/cwd/env/exit与哈希见 `remote_verification/remote_deployment_receipt.json`；可复现命令见 `verified_cpu_commands.sh`。

环境固定为新树 `src:PLAN/feedback_sys/P3NEW:PLAN/feedback_sys/P3R:PLAN/feedback_sys/P3` 的绝对路径PYTHONPATH，`PYTHONDONTWRITEBYTECODE=1`；验证进程局部 `CUDA_VISIBLE_DEVICES=''`。使用 `/data/yuyangliu/envs/isaaclab6_validation/bin/python -B`。运行阶段由root保留这一真实路径绑定，不修改已在用的采集源码。

没有未解决的代码导入依赖。离线最终请求仍需显式绑定既有r0b/case/context JSON及 `context_dev_originals_r1/source_index.json` 对应资格原件；本次只记录5个入口原件SHA与索引条数，没有重扫全部历史原件或混入源码树。采集卡、实际新phase/job/full/numeric/media等数据由root后续提供，详见 `pending_input_dependencies.json`。不能直接运行旧smoke的development cohort流程作为N5驱动。材料生成须显式 `--max-output-tokens 32768`；本次只验证import/help，不代表材料完整性、语义资格或模型release。
