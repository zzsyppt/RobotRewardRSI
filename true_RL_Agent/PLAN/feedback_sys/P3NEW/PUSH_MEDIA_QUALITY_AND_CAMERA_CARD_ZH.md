# Push / G1 旧 medium19 原帧质量与近景候选

状态：**旧帧已核验；近景候选 PREPARED_NOT_RUN**。没有启动 GPU、仿真或视觉 API，也不将旧重放媒体冒充与当前数值档案同轨的 N.5 资料。

旧视频在远端 `/data/yuyangliu/AGENT/feedback_p3r_20260929/video_medium19_r2/video/public_dev-01.mp4`，SHA-256 `40b74acedfda45259a0f9b16d701a50fecf52a63b335e5da476b32122bfd1d3e`，与原 `recordings.json` 的 run1/env19/411帧一致。本地保留 [原 MP4、metadata、四张未增强 PNG 与索引](results/push_media_quality_check/manifest.json)。固定控制步 1/95/250/411 对应零基帧 0/94/249/410；逐帧从 MP4 解码 RGB24 后原尺寸 1280×720 写入 PNG，不裁剪、缩放、均衡或调色。第 412 步数值终止帧没有录进视频。

四帧均能看到**完整 G1、双脚及地面网格**，没有画面边缘截断；原色正常。机器人在 1280×720 中约 80–100 px 宽、190–205 px 高，头/躯干/大致腿姿可辨，但足端、双臂遮挡关系及接触细节弱。旧 12 格拼图把每帧缩成 640×360 后，身体约 100 px 高；单凭该拼图作细肢、足接触或短暂恢复过程判断会引入图像质量偏差。若仅陈述四帧可见的全身粗姿态，可继续用原帧并标不可见；若要满足本阶段更细的过程观察，建议**先试一段较近但仍保留全身与地面参照的开发录像**，在实录/最终发送尺寸核验后才冻结任务机位。不能通过对旧图做颜色增强或非等比拉伸来假装新增细节。

近景候选沿用旧 [P3R `record_push_probe.py`](../P3R/record_push_probe.py) 的合格 checkpoint、`push.g1_37dof.recovery_v2`、旧 `eval_base_seed=1000000000`、medium env19、3 public seeds、runner/lease。唯一任务参数候选为 asset-root-relative eye `(3.0,-3.0,1.8)`、lookat `(0.3,0,-0.2)`、1280×720；Isaac Lab 实际目标坐标是 `root_pos+lookat`，不是 world z=-0.2。它意在让全身增大同时保留脚和地面，**尚未实录，不声称可用**。三段视频无论成功/失败都保留；首/四分位/中/四分三/末与旧已知姿态步在新录像前确定，不因结果挑有利帧。

旧远端 `source_r3` 的 renderer SHA `168e156ac541d1ea9acf3d85a98babf75cef86ec74ed60126dd2ff109ea58e6f` 不支持新 camera CLI；不能直接在旧 `record_push_probe.py` 后加参数。候选方案只复制旧 13 MB 源为 `source_push_camera_r1`，替换唯一的 renderer 文件为 SHA `933e76a4b6f6112c74a87b1e06a1f9c76dc0dcbf9f3de1f146fb5d9881bb4919`；旧 eval 脚本 SHA `8b588d668a14515b3e904b43331ca4625cd7df0c9b2d09c88cbc014ac65ba425` 和 replay 脚本 SHA `c899b06f8f702599b0c72bbaf5b1f692d1afa596a391fff1295f03c121218be9` 保持。薄 wrapper [`push_close_camera_probe.py`](push_close_camera_probe.py) 仅把旧 wrapper 的 SOURCE 指向 staging 并追加上述相机 CLI；`--timeout-s 600` 是 0.4 GPUh 双卡容量下的基础设施上限，不更改评估 seed、协议或成功定义。旧 medium19 replay 回执的主 GPU lease 为 142.0888 秒；只是历史可行性参考。

以下命令**仅为后续执行卡，当前未执行**。在 Stair r2 结束、两卡及已有 lease 实时确认空闲后，先 staging 与 CPU 预检 hash/源差异，再由主代理决定是否启动一次；输出目录必须全新：

```powershell
scp ./src/true_rl_agent/runner/native_video.py humanoid-server:/data/yuyangliu/AGENT/feedback_p3_20260924/native_video_push_camera_r1.py
scp ./PLAN/feedback_sys/P3NEW/push_close_camera_probe.py humanoid-server:/data/yuyangliu/AGENT/feedback_p3_20260924/push_close_camera_probe.py
ssh humanoid-server 'test ! -e /data/yuyangliu/AGENT/feedback_p3_20260924/source_push_camera_r1 && cp -a /data/yuyangliu/AGENT/feedback_p3_20260924/source_r3 /data/yuyangliu/AGENT/feedback_p3_20260924/source_push_camera_r1 && cp /data/yuyangliu/AGENT/feedback_p3_20260924/native_video_push_camera_r1.py /data/yuyangliu/AGENT/feedback_p3_20260924/source_push_camera_r1/src/true_rl_agent/runner/native_video.py'
# 核对 staging 除 renderer 外的 src/scripts/configs 全部 SHA 不变，checkpoint 和新 environment manifest 真实匹配；此处未宣称已通过。
ssh humanoid-server 'python3 /data/yuyangliu/AGENT/feedback_p3_20260924/push_close_camera_probe.py 19 cuda:0 /data/yuyangliu/AGENT/feedback_p3_20260924/push_close_camera_r1'
```

阶段 2 GPUh 重分配后 Push 补录余量为 **0.4 GPUh**。600 秒若两卡均被 Isaac 进程占用，保守容量为 `2×600/3600=0.3333 GPUh`，另留约 0.0667 GPUh 给启动/清理；实际计量以 GPU lease 和进程墙钟为准。旧 wrapper 只给主 GPU 加 lease，若第二卡出现附带占用，运行期间安排两卡独占并按两卡计，不使用 CUDA mask。若任何源 hash、预算或设备状态不符，不启动；一旦录制，必须先核三段 MP4、metadata、帧数与最终发送图，不以 exit 0 单独判成功。
