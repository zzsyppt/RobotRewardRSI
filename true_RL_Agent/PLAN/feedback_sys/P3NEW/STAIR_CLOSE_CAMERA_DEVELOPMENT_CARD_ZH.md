# Stair / Go2 旧开发轨迹近景录制卡

状态：**R1_OPERATOR_INTERRUPTED**，2026-09-29。原运行字节、设备误述纠正及成本见 [`r1 中断记录`](results/stair_camera_probe_r1_failure/INCIDENT_ZH.md)；后续仅依 [`r2 续试卡`](STAIR_CLOSE_CAMERA_R2_CONTINUATION_ZH.md)。本卡以下保留 r1 原预定设置，不代表已取得可用录像。

## 已核实来源及当前条件

- 复用旧 P3 `source_r5` 的 `stair_v3_reference_r5_on/public_dev_job.json` 和 `review_video_20260924/slot282`：formal public-dev 原命名空间、base seed 170100、288 env、3 public seeds，只录 env 282。旧录像的三段均保留；第 1 段 342 个有效渲染控制帧，旧视频 SHA `034f0478…7a5f03`。不改 checkpoint、任务配置、协议/plan hash、成功定义、动作与随机种子。
- 旧合格 checkpoint 文件在远端存在，实际 SHA-256 `7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`，与旧 job 的 `--checkpoint-sha256` 一致。旧 `source_r5` 的 `native_video.py` SHA `168e156a…69a58e6f`、Stair eval 脚本 SHA `904142dd…159df1bf`。
- 用旧 Isaac Python、旧 `source_r5` 在不启动仿真的 CPU 命令实算 execution-environment manifest，得到 `16644058f9b86d58a2f3a1486b9568016fb1d94f9e89807809b93aa05b142630`，与旧 job 相同；staging 后按同法得到新的真实 hash。下列辅助脚本当前 SHA-256 为 `64dbf919b7435fc7f9015f03b63f718ce05c3c111443c60ba30c1963103290a9`。
- 旧 `review_video_20260924/launch_video.py` 固定运行 `source_r5`，旧 renderer 只接受 `--video-dir/--video-env-index`；**不能直接往旧脚本追加新相机参数**。旧 slot282 从 argv 文件 22:13 到三个 seed 的 recordings 文件 22:20，约 7 分钟，非精确 GPU 计时。当前远端 20:51 两卡 0% utilization、显存 7/6 MiB；未见 Isaac replay 进程、锁文件未见持有进程。执行前须重新确认，不能把该快照当保留的 GPU 许可。
- Isaac Lab `ViewportCameraController` 计算 `camera=robot.root_pos+offset`，每次渲染跟随 root 位置；不随 root 朝向转动。候选 Go2 eye `(2.5,-2.5,1.4)`、lookat `(0.4,0,-0.15)`，1280×720。它比旧 eye `(4.5,-4.5,3.0)` 近，目标略在根部下方并朝前；**未实录验证**，不可称已达画质标准。

## 源冻结与一次运行

只复制旧 `source_r5`（约 12 MB）到新目录，再替换唯一文件 `src/true_rl_agent/runner/native_video.py` 为已测试的相机 CLI/metadata 实现；不要把当前本地其它语义开发源码混入。新模块 SHA-256 固定为 `933e76a4b6f6112c74a87b1e06a1f9c76dc0dcbf9f3de1f146fb5d9881bb4919`。新旧源码的其余 `src/scripts/configs` 文件必须逐文件同 hash，辅助脚本会拒绝额外差异。辅助脚本在源码树外，不进入 execution environment 快照。

在本仓库根目录，审阅后可执行下列 PowerShell 命令；前两段仅 staging / CPU 预检，最后一行才启动 GPU：

```powershell
scp ./src/true_rl_agent/runner/native_video.py humanoid-server:/data/yuyangliu/AGENT/feedback_p3_20260924/native_video_camera_r1.py
scp ./PLAN/feedback_sys/P3NEW/stair_close_camera_probe.py humanoid-server:/data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py
ssh humanoid-server 'set -e; base=/data/yuyangliu/AGENT/feedback_p3_20260924; test ! -e "$base/source_stair_camera_r1"; cp -a "$base/source_r5" "$base/source_stair_camera_r1"; cp "$base/native_video_camera_r1.py" "$base/source_stair_camera_r1/src/true_rl_agent/runner/native_video.py"; sha256sum "$base/source_stair_camera_r1/src/true_rl_agent/runner/native_video.py"'
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py --source /data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1 --output /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_r1'
# 重新查 nvidia-smi、Isaac 进程及 GPU lease holder，并核对上行 READY_TO_RUN/hash 后才执行：
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py --source /data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1 --output /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_r1 --run'
```

辅助脚本从旧 job 构造原命令：只移除数字档案参数、改新输出路径、追加 env282 及上述相机参数。旧 job 的 `argv[1]` 已核实为相对路径 `scripts/isaaclab_stair_rsl_rl_eval.py`；helper 仍将其改为 staging 内的绝对路径、逐文件核验脚本 SHA，并在 Isaac Python 中断言实际导入的是 staging 的 renderer。项目源码快照包含 renderer 文件，修改后旧 `--backend-execution-environment-manifest-hash` 必须用 staging 源在同一 Isaac Python 中重新计算并替换；不冒充旧来源 hash。其余 eval/protocol/seed/plan/checkpoint 参数保持旧值。脚本在 `--run` 才创建输出并启动；复用项目 `GpuLeaseManager` 对原 GPU ordinal 全程持锁，`finally` 记录实际租约墙钟与 receipt；保存 `argv.json`、`run.log`、`run_status.json`，子进程硬超时 900 秒并记录失败，绝不自动重试。预检 `READY_TO_RUN` 无仿真、无租约。旧 full-cohort 重放本来一次跑 288 env × 3 seeds，即使只录 env282 也需付整次 eval 成本；删到 1 seed 会改变评估身份，故不做。

**实际 staging / CPU 预检（当时）：** 已复制旧源并只替换 renderer；远端文件 hash 与本卡一致，逐文件 source diff 仅该模块。`READY_TO_RUN` 返回 staging environment manifest `64ca409c9f968f735d06e7c60e1365496e75b11e3fc9b3034cbd7773db5d2465`、checkpoint `sha256:7b582643…b203da11c4c`、staging eval 绝对路径 `/data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1/scripts/isaaclab_stair_rsl_rl_eval.py` 及 SHA `904142dd…159df1bf`、env282、3 seeds、上述 eye/lookat/1280×720。原 job 的实际 device 为 **cuda:0**，r1 lease 也正确持有 cuda:0；此前把旧 slot282 录像命令的 cuda:1 当成此次 device 的说明错误。后续运行和预算状态以 r1 中断记录、r2 续试卡为准。

## 容量和预定画面检查

从阶段上限 2 GPU 容量小时中的 **0.5 GPUh 开发/失败余量预留 0.25 GPUh** 给本次一次性探针；该余量执行后最多剩 0.25 GPUh，正式同步采集 1 GPUh、Push 补录 0.5 GPUh 原分配不动。旧 7 分钟仅作可行性参考；900 秒上限恰为 0.25 GPUh，实际以独立 GPU lease/进程计时为准，失败也计消耗。运行需有足够磁盘，保存三段 MP4、metadata 和日志。

不看新成功结果选帧。三段录像各按其最终 `frame_count=N` 预定取第 `1, ceil(N/4), ceil(N/2), ceil(3N/4), N` 个控制帧；第 1 段额外固定取旧开发已知姿态区间的 `160,190,220` 帧。只解码原 RGB，不均衡、不裁切放大、不非等比拉伸。逐帧检查完整身体、腿/地面与邻近台阶、画面边缘裁切、遮挡、姿态变化、跟随镜头造成的背景漂移，并在最终发送尺寸复核。三段都保留，质量失败也留原样和理由；不挑成功 seed，不据模型观察答案换机位。若该单一候选机位合格，先冻结 Stair 机位和拼图版本，再用于后续正式集；若失败，开发卡记录缺陷、另行审阅新候选，不能把此探针算正式证据。
