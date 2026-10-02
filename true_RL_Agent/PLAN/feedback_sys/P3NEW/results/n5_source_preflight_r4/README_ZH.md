# 当前源 r4 静态预检

结论：`STATIC_PREFLIGHT_PASS_CAPTURE_NOT_RELEASED`。独立远端 `source_n5_preflight_r4` / `n5_preflight_r4` 已实际执行两条 `--mode preflight`：Stair `PREFLIGHT_PASS`、Push `PREPARED`，exit 均为 0。没有执行 run、GPU 仿真、训练、模型/API 请求或正式采集；生产源码、主计划和 r3 原件未改。正式协议与评阅角色由 root 另行冻结。

当前 src/scripts/configs（排除缓存）及四既有助手共 400 文件已封存；相对 r3 无增删、399 相同，唯一变化为 `src/true_rl_agent/tasks/push_recovery/process_report.py` 的缺测说明，52136 → 52172 字节（+36）。原 SHA `1a4609d679abc9135832c37289022a7d9c77c0dd2a7e6d95d6bb8838f68506cf`，新 SHA `3e3beaee6754e2b4ec4785f667fdcd72f9a18abc8ca1a6eb90d280f3f2b92e3e`。源 manifest SHA `1137cc48c8c0f9c21787b99e12b5f5cf4e0fdb6cb915c4ae3c2e608b67b0f0ee`。本地复制前后、包内、远端执行前后以及收尾时当前源码均逐文件一致。具体字节差异见 `production_source.r3.diff`。

实际 checkpoint / training summary / Push source manifest 原字节重新核对。`independent_binding_verification.json` 从本次冻结配置及返回 composition manifest 独立重建双方 evaluation binding，并核对实际 result、training binding、job/plan hash、源脚本 / protocol / PYTHONPATH / remote receipt 同一 r4 根；173 个导入模块均来自 r4 快照。Stair binding 为 `cb8551a6ddb57a8e452ce0a3421e5a96d1a6c46e1e8d82bb9a37db43f17d530a`，Push 为 `acde9f4b15abb449366290bdee8d87e1c0179ec07f4a05448e97f089c803d464`，与 r3 一致。Push 原结果的 `checkpoint_bytes_checked=false`、`checkpoint_runner_admission=not_executed` 保留，独立字节核查见 `remote_preflight/source_input_verification.json`。

原 checkpoint、cohort、seed、三个 seed runs、7200 秒 timeout、1920×1080 双视角及每步录制参数完整沿用。Stair env 2 / 63 静态映射仍为 `public_dev.c0_id_level_3` / `public_dev.c1_conditional_frontier_level_5`；Push env 3 为 low / direction 3 / planned 1，env 59 为 repeated / direction 11 / planned 2。两条 argv 与 r3 实际成功命令只有 source/output 路径变化，见 `pre_execution_audit.json`。

资产缓存和 588 个历史 runtime 文件核查继承 r3 原件；精确原件路径、大小与 SHA 在 `inherited_r3_evidence.json`，本次复核这些回执原件，未重新遍历 588 文件或重核资产字节。当前两个 robot profile 未变。继承结论限 r3 明列文件与缓存，既非新的全服务器 seed 未使用证明，也不证明 runtime asset resolver 可加载；运行时资产、GPU lease、checkpoint admission 等生产闸门仍须执行。

远端 2026-09-30 01:47:17–01:47:21 UTC（北京时间 09:47）检查：GPU 0/1 均为 29024/32607 MiB，VLLM PID 3441306/3441307 各占 29012 MiB，未停止或修改；0% 瞬时利用率不能视为资源可用。检查后磁盘余 72,571,514,880 B。当前 GPU 占用仍阻塞采集安排。

精确实际命令在 `exact_preflight_commands.sh`、`*.command.json` 和 `remote_preflight/task_preflight_results.json`；两任务 stdout/stderr、result、job、composition 和远端回执在 `remote_preflight/`。下载包 `remote_results.tar` SHA 已核对，直接用 Windows `\\?\` 扩展路径解包成功，未重跑预检。r3 四脚本复用仅改必要路径、继承回执和独立 binding 核查，改动 diff 随附。初始 `preflight_card.json` 保留原样，完成卡为 `completed_preflight_card.json`，索引为 `completion_receipt.json`。本目录不提供正式 run 采集 release。
