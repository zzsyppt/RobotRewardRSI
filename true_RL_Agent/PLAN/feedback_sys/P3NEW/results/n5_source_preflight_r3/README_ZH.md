# 当前源 r3 静态预检

结论：`STATIC_PREFLIGHT_PASS_CAPTURE_NOT_RELEASED`。已在 humanoid-server 上传独立 `source_n5_preflight_r3` 并实际执行两条 `--mode preflight`；Stair 返回 `PREFLIGHT_PASS`、Push 返回 `PREPARED`，退出码均为 0。没有调用 run、模型、训练或采集。正式冻结和采集由 root 另行决定。

本地 400 文件快照覆盖当前 src/scripts/configs（排除缓存）及四个既有采集助手。相对 r2，只有 Stair `behavior.py`、`process_report.py` 改变；其余 398 文件相同。源清单 SHA：`4c546b9aff291e91b6f8324c207a5f2cb33415e0dce7f3ea500476ca7eebf9cc`。本地复制前后、包内、远端执行前后逐文件一致，收尾时当前工作区这 400 文件也与快照一致。原 r2 快照、manifest 和既有草案未改。

已验证双方 checkpoint 原字节、训练 summary、Push 原 manifest、Stair 当前 protocol；实际 preflight 的 training binding、training manifest、evaluation binding 均与继承的来源一致。Push 原 preflight 字段 `checkpoint_bytes_checked=false` 保留原样，独立字节检查记录在 `remote_preflight/source_input_verification.json`，没有将静态准备冒充 runner admission 或新资格验收。

选取环境在当前源下复核一致：Stair env 2 属 `public_dev.c0_id_level_3`，env 63 属 `public_dev.c1_conditional_frontier_level_5`；Push env 3 为 low / direction 3 / planned 1，env 59 为 repeated / direction 11 / planned 2。详见 `static_env_mapping.json`。

Go2 的 USD 与 meshes、G1 的 USD 均在既有缓存中按大小及 SHA 找到，与当前 robot profile 的 closure entries 精确一致。Go2 在真实 `/home/yuyangliu/.cache/ov/client/https` 下，不在最初查的 `/data/yuyangliu` 缓存根。初次未命中结果没有改写，补核另存 `asset_cache_supplement.json`。没有启动 Kit、解析器或物理环境；运行时资产加载及既有环境/准入闸门仍须执行。

历史核查覆盖回执所列六个目录的 588 个 runtime summary/result/manifest，跳过静态 source/material/reference/preflight 副本，无读取错误或超大跳过。Stair `public_dev_diagnostic_v1` 与 Push base 1500000000 / runner seeds 1500010000–1500010002 未发现精确值使用记录。五个原始字串候选均是旧 Stair 指标的小数尾数，逐 JSON pointer 消歧见 `history_disambiguation.json`。该结论仅覆盖明列文件，不是全服务器绝对未使用证明。

2026-09-30 00:06:25 UTC 的资源状态：GPU 0/1 各使用 29024 / 32607 MiB，现有 VLLM worker PID 3441306/3441307 各占 29012 MiB；瞬时 0% 利用率不等于资源可用。磁盘剩余 72,850,571,264 字节。未停止或修改这些进程。GPU 占用是当前采集安排的明确阻塞，执行前还需再次核对资源与 lease。

实际命令/环境、stdout/stderr、result、生成 job、composition、远端文件哈希在 `remote_preflight/` 与 `remote_results.tar`。下载包 SHA 经验证；本地首轮解包遇 Windows MAX_PATH，使用扩展长度路径补全并逐字节核已有文件，没有重跑远端 preflight（见 `local_extraction_completion.json`）。`preflight_card.json` 保留上传时原件，完成状态另存 `completed_preflight_card.json`，最终目录索引见 `completion_receipt.json`。

## 正式采集命令草案追加（NOT_RELEASED）

`capture_commands_draft.json` 从上述实际成功 argv 生成，两个冻结 CLI 的 `--mode` 枚举均为 `preflight/run`。每任务只将 mode 改为 `run`，将 output/video-dir 换为全新 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_20260930/{stair,push}`；同 r3 源、checkpoint/cohort、env（Stair 2/63，Push 3/59）、三 seed、1920×1080 逐步双视角、7200 秒 phase timeout 均保持。该 JSON 只供审阅，未执行。

入口显式保留 `PYTHONPATH=<r3 source>/src`、`PYTHONDONTWRITEBYTECODE=1`，按两个 run 分支要求在进程局部取消 `CUDA_VISIBLE_DEVICES` 预遮罩，由生产 runner 处理子进程设备遮罩。资产仍使用冻结 profile 的原 URL 与现有 Omniverse 缓存/解析器，不改资产路径。只请求一张可供生产 lease 使用的 32 GB RTX 5090，Stair→Push 顺序运行；未测得本轮精确显存峰值，不能把当前约 3.5 GiB 余量视为足够。

本任务唯一一次远端读取为 2026-09-30 00:25:10 UTC：新根及两个任务目录均不存在，磁盘余 72,850,280,448 B；GPU 0/1 仍各占 29024/32607 MiB，VLLM PID 3441306/3441307 未动。资源原返回、命令哈希、来源及仅三项 argv 差分均写入草案。完整存储预算、protocol/capture 正式冻结与放行仍由 root 完成。此次只新增命令草案并追加本文；既有完成回执内 README 哈希对应追加前的原文，未回写旧回执。
