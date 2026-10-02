# Stair r3 固定开发媒体机械核查

状态：**录制与固定提帧完整；画质是否进入正式输入未裁决。** r3 在原 checkpoint / env282 / 3 个旧 public-dev seeds 下只改 root-relative 相机偏移，不是 N.5 确认样本。远端 `run_status.json` 为 `FINISHED`/exit 0、GPU0 lease 436.74315439909697 秒；保守按两卡计 `0.2426350857772761 GPUh`。r1+r2+r3 已知双卡合计约 `0.5657054898 GPUh`，未超过开发/失败行项 0.6 GPUh。

三段 MP4 均存在，metadata SHA 与本地重算逐一相同，frame_count 342/316/290；`recordings.json` 与 `run_status.json` 原字节已复制。相机记录均为 `asset_root_tracking.v1`、eye `(-1.0,-1.8,1.0)`、lookat `(0.25,0,-0.15)`、1280×720。评价 summary 在远端存在。完整 MP4、固定 18 张未增强/未缩放原 RGB PNG、逐张 SHA 和 `step→frame_index=step-1` 见 [`manifest.json`](manifest.json)：run1 `1/86/160/171/190/220/257/342`，run2 `1/79/158/237/316`，run3 `1/73/145/218/290`，无结果后挑帧。

人工总览见 `quality_contact_run{1,2,3}_HALF_SCALE_QA_ONLY.png`：它们仅供核对，原 1280×720 逐帧按 0.5 倍 Lanczos **等比缩小**到 640×360，并在帧外加步标签；工具显示时可能进一步缩放。它们**不作为观察者输入**。实际候选输入顺序见 [`send_layout_plan.json`](send_layout_plan.json) 和 [`send_layout_preview.html`](send_layout_preview.html)：18 张各自原始 1280×720 PNG、每张一步、16,588,800 像素／7,953,035 PNG 字节，图片字节不缩放，HTML 仅 CSS 等比显示，点击打开原帧。尚未调用视觉 API，尚未核 provider 端最终缩放/读回。

固定帧总览中，背坡侧视角较 r2 避开了 seed2 中段的高台阶大面积挡身：选定帧可看到躯干与四足的大致轮廓，地形台阶也仍在画面内。白色机器人与浅灰台阶的低对比仍存在；局部腿脚与台阶/机身会自然重叠，单帧不能推断接触或连续过程。还需在实际发送尺寸逐张复核遮挡、地面/台阶参照与 robot-centring，再决定同任务统一机位；本记录不授予画质 PASS、物理接触判断或正式视觉资格。

三个能力边界分别记录：

1. **原生取景开发检查：已完成固定帧提取和人工开发查看。** 根端查看 seed1 的 1/160/342、seed2 的 158、seed3 的 145 后认为全身和台阶关系较 r2 明显改善，同时仍有自然遮挡。这只覆盖旧开发重放及所看帧。
2. **远端最终输入辨读：未完成。** 虽有 18 张原 PNG 的候选请求顺序，尚未将该布局发送观察者并检查 provider 实际接收尺寸、逐步识别、可见性与输出审计；HTML/CSS 总览不是远端读回。
3. **同步双机位能力：未完成。** r2 与 r3 是两次独立 replay，即使 checkpoint、seed 和 step 标签一致，也无同一仿真状态的双视角同步保证，不能拼成同轨多视角证据。未来最多两互补机位与额外仿真字段须用新的同 step/state 绑定合同；本轮未实现、未采集。
