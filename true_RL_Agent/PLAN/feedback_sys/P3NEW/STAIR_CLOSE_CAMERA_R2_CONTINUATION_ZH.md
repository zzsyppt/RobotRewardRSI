# Stair 旧开发轨迹近景续试 r2

状态：**PREPARED_NOT_RUN；须先复核双卡空闲与阶段容量分配**。r1 因根端误判设备而发出 SIGTERM，虽回执 exit 0，但没有视频和评估 summary；它是 operator-interrupted 基础设施失败，不能作为画质通过样本。原始文件与计量见 [`results/stair_camera_probe_r1_failure/INCIDENT_ZH.md`](results/stair_camera_probe_r1_failure/INCIDENT_ZH.md)。本次续试只补完同一个旧开发轨迹的机位校准，不是 N.5 新确认、没有新相机挑选。

完全沿用 r1 staging `source_stair_camera_r1`、冻结 renderer SHA `933e76a4b6f6112c74a87b1e06a1f9c76dc0dcbf9f3de1f146fb5d9881bb4919`、helper SHA `64dbf919b7435fc7f9015f03b63f718ce05c3c111443c60ba30c1963103290a9`、环境 manifest `64ca409c9f968f735d06e7c60e1365496e75b11e3fc9b3034cbd7773db5d2465`、旧 checkpoint/288 env/3 public seeds/env282、eye `(2.5,-2.5,1.4)`、lookat `(0.4,0,-0.15)`、1280×720。仅输出目录由 `stair_close_camera_r1` 改为 `stair_close_camera_r2`。冻结 helper 实际从最终原 job 的 `--device cuda:0` 获取并持有 GPU0 现有 lease；不改 helper、不修改源码、seed、评估身份、成功判定或 renderer。r1 远端目录不覆盖。

Isaac/Kit 在 r1 中还让同一进程占用了 GPU1 约 256 MiB；虽然日志把 GPU0 标为 Active 且 viewport `deviceMask: 1`，具体附带上下文成因未知。本次安排两卡同时空闲、不与另一个 GPU1 作业并行，并把双卡整个持锁区间按容量小时保守计费。根端已在**阶段总上限仍为 2 GPUh** 的前提下事前重分配：正式同步采集 1.0 GPUh 不变，Push 补录余量由 0.5 调为 0.4 GPUh，开发/失败由 0.5 调为 0.6 GPUh。r1 已耗 `0.0808889212 GPUh`；r2 沿用同 helper 的 900 秒上限，两卡最坏再按 `0.5 GPUh` 预留，合计 `0.5808889212 GPUh ≤ 0.6 GPUh`，留下约 `0.0191110788 GPUh` 处理结束开销及本行项其他消耗；实际按两卡运行墙钟结算，若将触及行项上限须停止而非静默超支。此处 r2 是根端误判设备并对 r1 发 SIGTERM 后的**基础设施续试**，不是从模型/画质结果挑有利样本。

预算确认并再次只读检查 GPU 进程、lease 持有者、staging/hash、r2 目录不存在后，先做无仿真预检；随后仅一次运行，无自动重试：

```powershell
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py --source /data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1 --output /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_r2'
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_probe.py --source /data/yuyangliu/AGENT/feedback_p3_20260924/source_stair_camera_r1 --output /data/yuyangliu/AGENT/feedback_p3_20260924/stair_close_camera_r2 --run'
```

完成条件先看三段 `video/public_dev_seed_*.mp4` 与 `recordings.json`、评估 summary 是否实际存在及帧数/SHA，不能只看进程 exit code；然后对三段全部取预定首/四分位/中/四分三/末原 RGB 帧，第 1 段另取已冻结的 160/190/220 步。逐帧查完整身体、足与相邻台阶、边缘裁切、自然遮挡、跟随镜头和最终发送尺寸；不按成功/失败或模型回答挑选录像。画质未验证前不得送正式观察。
