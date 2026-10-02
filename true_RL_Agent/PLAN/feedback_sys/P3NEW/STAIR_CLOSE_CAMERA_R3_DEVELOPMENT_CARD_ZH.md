# Stair / Go2 背坡侧近景开发试拍 r3

状态：**RECORDED_DEVELOPMENT_ONLY；画质未裁决**。只修正视觉遮挡，不是 N.5 新确认，也不改物理、checkpoint、seed、任务成功判定或旧评估身份。r1/r2 helper 原字节 SHA `64dbf919b7435fc7f9015f03b63f718ce05c3c111443c60ba30c1963103290a9` 已保存在 [`results/stair_camera_probe_r2_quality/stair_close_camera_probe_r1_r2_frozen.py`](results/stair_camera_probe_r2_quality/stair_close_camera_probe_r1_r2_frozen.py)；旧 r1/r2 输出不覆盖。

r2 的预定原帧显示 seed2 中段有前景高台阶大面积挡住躯干与腿。任务源码 `tasks/stair_climbing/isaaclab_terrain.py` 的 `one_way_stairs_terrain` 明确台阶沿局部 **+x** 升高；r2 eye `(2.5,-2.5,1.4)` 从上坡侧回看，会把高处台阶置于身体前方。r3 固定背坡侧 root-relative eye `(-1.0,-1.8,1.0)`、lookat `(0.25,0,-0.15)`、1280×720；仍由 `asset_root_tracking.v1` 跟随根部平移。该候选现已实录；原生取景开发检查、远端最终输入辨读与同步双机位能力须分别裁决，见 r3 质量记录。

沿用远端 `/data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1`；此 staging 与旧 `source_r5` 仅 renderer 不同，renderer SHA `933e76a4b6f6112c74a87b1e06a1f9c76dc0dcbf9f3de1f146fb5d9881bb4919`、Stair eval SHA `904142dd66a2e350d253295f3c6bd7514d52b8237a5c5b615de09759df1bf398`，真实环境 manifest `64ca409c9f968f735d06e7c60e1365496e75b11e3fc9b3034cbd7773db5d2465`。同旧合格 checkpoint `sha256:7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`、formal public-dev 原 namespace/3 seeds/288 env，只录 env282 的全部三段。新 helper SHA `d9cd4b960c64815399493068c20ece6e3fdb28a3522e9fa6f34d3d60e233d95a`，只开放显式相机 eye/lookat 与硬超时参数，其他原 argv 继承；输出全新 `stair_close_camera_r3`。

阶段上限仍 2 GPUh：正式同步 1.0、Push 补录 0.4、开发/失败 0.6。r1 两卡保守 `0.0808889212 GPUh`；r2 lease 435.92666910775006s，两卡 `0.2421814828 GPUh`；已用合计 `0.3230704040 GPUh`。r3 硬上限 470s，两卡最坏 `0.2611111111 GPUh`，总最坏 `0.5841815151 < 0.6 GPUh`，余量约 `0.0158184849 GPUh`。helper 子进程等待上限会给结束清理留 10s，`run_status.json` 记录实际 lease 墙钟及 `2×秒数/3600`；如上限不足即记失败，不自动重试、不越额续跑。运行须两卡无其他作业；项目现有 lease 持有原 argv `cuda:0`，GPU1 若发生附带占用仍保守按两卡计。

唯一执行命令（先不带 `--run` 做 CPU 预检，核对 `READY_TO_RUN` 的源/hash/机位/470s/新输出；再经实时双卡进程与 lease 核查后加 `--run` 一次）：

```powershell
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py --source /data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1 --output /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_r3 --camera-eye -1.0 -1.8 1.0 --camera-lookat 0.25 0.0 -0.15 --timeout-s 470'
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py --source /data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1 --output /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_r3 --camera-eye -1.0 -1.8 1.0 --camera-lookat 0.25 0.0 -0.15 --timeout-s 470 --run'
```

完成条件：核 `run_status`、3 段 MP4 与 metadata/每段 SHA、评估 summary，不能以 exit0 单独代替媒体完整性；三段全保留。每段按其有效帧数 `N` 固定取 1、ceil(N/4)、ceil(N/2)、ceil(3N/4)、N；seed1 额外取 160/190/220（若实际存在）。只解码原 RGB，保留 step→frame 和 SHA；检查全身、足/台阶/遮挡、姿态及最终发送尺寸，不因成功或模型回答挑帧。不宣称画质 PASS，正式机位须后续独立质量核对。

**执行前实测：** 远端 helper SHA 与本卡一致；CPU 预检返回 `READY_TO_RUN`、上述 checkpoint/环境 manifest/eval脚本 SHA、env282/3 seeds、eye/lookat/1280×720、`lease_timeout_s=470` 和双卡上限 `0.2611111111 GPUh`；r3 输出目录仍不存在。随后两卡分别 7/6 MiB、0% utilization，无 compute-app、相关 replay 进程或现有 GPU lease 文件持有者。该状态只在这次启动前有效；启动后仍以实际 receipt 记账。

**实际结果：** `run_status.json` 为 FINISHED/exit0，GPU0 lease 436.74315439909697s，双卡容量 `0.2426350857772761 GPUh`；三段 MP4、metadata SHA/帧数和 evaluation summary 均已核。完整原帧、18 张预定固定帧、发送布局及三类能力边界见 [`r3 质量记录`](results/stair_camera_probe_r3_quality/QUALITY_CHECK_ZH.md)。不从此自动推导正式画质 PASS。
