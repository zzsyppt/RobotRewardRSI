# N.3 旧媒体开发库存（只读，未发送）

截至 2026-09-29。本页只登记本地实际存在的图像、其已保存的录像元数据和对应数值档案。旧图均为**另一次原生重放**的采样帧；现有核查只到 checkpoint／协议／env／终点或少数数值点，没有证明图像第 *s* 步就是 P3NEW 数值档案第 *s* 步。因此可作开发性 `visual_only` 观察，不能把旧图的 `numeric_claim` 与当前 task checker 逐步配对，也不能把 `source_id` 伪填成数值 manifest SHA 后当成同源观察。此库存没有调用观察 API、GPU 或作者模型。

## 可见图像及身份

| 回合与本地图像（均 2560×1080 PNG、4×3 共 12 个离散格） | 图上可读 control step | 录像元数据及数值来源 | 适用性 |
|---|---|---|---|
| Stair `r1.e64`：[g64_r1_zoom.png](../P3/review_video_20260924/g64_r1_zoom.png) | 360, 380, 390, 398, 405, 410, 414, 418, 421, 423, 424, 425 | [slot64_recordings.json](../P3/review_video_20260924/slot64_recordings.json)，run 1、env 64、checkpoint `sha256:7b582643…`；录像 425 帧，数值终止 426。 | 仅重放视觉；终止帧不在图内。 |
| Stair `r1.e156`：[g156_r1_zoom.png](../P3/review_video_20260924/g156_r1_zoom.png) | 170, 185, 195, 205, 212, 218, 222, 225, 227, 229, 230, 231 | [slot156_recordings.json](../P3/review_video_20260924/slot156_recordings.json)，run 1、env 156、同 checkpoint；录像 231 帧，数值终止 232。 | 仅重放视觉；终止帧不在图内。 |
| Stair `r1.e55`：[g55_r1_eq.png](../P3/review_video_20260924/g55_r1_eq.png) | 40, 60, 75, 90, 105, 130, 180, 250, 330, 410, 500, 587 | [slot55_recordings.json](../P3/review_video_20260924/slot55_recordings.json)，run 1、env 55、同 checkpoint；录像 587 帧，数值终止 588。 | 仅重放视觉；终止帧不在图内。 |
| Stair `r1.e282`：[g282_r1_eq.png](../P3/review_video_20260924/g282_r1_eq.png) | 40, 70, 100, 130, 160, 190, 220, 250, 280, 300, 320, 342 | [slot282_recordings.json](../P3/review_video_20260924/slot282_recordings.json)，run 1、env 282、同 checkpoint；录像 342 帧，数值终止 343。 | 仅重放视觉；终止帧不在图内。 |
| Push `r1.e19`：[medium19_run1_grid.png](../P3R/results/checkpoint_evidence_r1/video_medium19_r2/video/medium19_run1_grid.png) | 1, 68, 70, 71, 95, 109, 122, 179, 250, 350, 400, 411 | [recordings.json](../P3R/results/checkpoint_evidence_r1/video_medium19_r2/video/recordings.json)，run 1、env 19、checkpoint `sha256:91c00d4e…`、resolved protocol `db74aed6…`；录像 411 帧，数值终止 412。 | 仅重放视觉；终止帧不在图内。 |

Stair 四张图的步标签、文件 SHA 与回放身份已登记在 [旧视觉评阅记录](../P3R/audit_response_20260929/visual_probe_model_review.json)；该评阅曾接触未过滤的结局字段，标记为探索性开发资料。Push 三次 env19 录像的 checkpoint、协议、槽位、步数、终局及选定数值点通过 [27 项检查](../P3R/results/medium_video_checks.json)，检查自身明确**不证明逐步同轨**。本地在该 Push 视频目录只见 run 1 的拼图 PNG 与 `recordings.json`；元数据列出 run 1–3 的 MP4 文件名，不能把未在本地找到的 MP4 当作可提帧资产。

对应的当前开发数值档案是 Stair [manifest.json](../P3R/results/stair_numeric_dev_r1/numeric_evidence/manifest.json)，SHA-256 `22fb749620c4e614cbb4df4fbfdb82b70d76c8f1cd81a3eda50de465d9fb3e4a`；Push run 1–3 分别为 [public_dev-01](../P3R/results/push_numeric_dev_r1/numeric/public_dev-01/manifest.json) `e724f9fe…`、[02](../P3R/results/push_numeric_dev_r1/numeric/public_dev-02/manifest.json) `b01239d1…`、[03](../P3R/results/push_numeric_dev_r1/numeric/public_dev-03/manifest.json) `164c4310…`。这些路径与 hash 来自 P3NEW `results/delivery_offline_r1/{stair,push}/raw_sources.json`，不是录像同步采样证明。Stair 数值档案的 checkpoint、cohort 与旧录像对应，但仍是新采集；Push 也是独立重放。

## 图像尺度与最小提取

录像记录的相机 eye offset 为 `[4.5,-4.5,3.0]`、look-at offset `[0.8,0,0]`，原视频 1280×720、50 fps、控制步长 0.02 s、从 step 1 开始且不含 reset。保存的拼图每格约 640×360；`zoom`／`eq` 等后处理不能提供世界米尺，图上相对地面高度也不是 `root_pos_w.z-env_origin_w.z`。镜头看不到精确接触、摩擦、力或未展示步。Push 拼图中的机器人占画面较小，细足动作的判断尤其受尺度限制。

最小开发包直接读取上述 PNG 原字节，不重新编码或从缺失的 MP4 猜帧；以图上 12 个递增 step 作为单个 `MediaImage.displayed_steps`，`MediaWindow` 只给确实可见的 body 名称，令观察者在跨间隔时使用 `sampled_frames_only`。同一图的 12 格仍是**一张图、12 子帧**。保存图像 SHA、相应 `recordings.json`、标签清单和回放标记；数值 `s0`、`measurement_claims` 与 task checker 暂不附加。若要将媒体写进当前 process full 并做数值 claim，需新的同次记录及八项 media/numeric 身份绑定，且验证每个显示 step 对应原始有效步数组；现有旧图不满足这一点。

## 预算内候选包

按**每份报告**最多 4 回合／24 图／96 子帧：Stair 可选上述四回合、每回合一张已有拼图，合计 **4 图／48 子帧／11,059,200 像素／3,528,655 PNG 字节**，不超过图数和子帧硬上限；另需核对实际设置的累计像素、字节、费用及 4 次调用限额。Push 现有本地图只有 `r1.e19` 一张，**1 图／12 子帧／2,764,800 像素／2,306,665 字节**，其余 3 个名额明确为无本地媒体，不能补造为已观察。这个旧图包仅是预算可行性示例；尚未核对 N.3 固定的成功反例／残差／缺对照选择规则，不能因此替代正式前四回合的选择结果。两个任务分开计报告与预算；不能把五个回合混成同一报告。旧图已经参与 P3/P3R 开发，以上只适合接口与措辞试验，不构成新盲确认或资格证据。

离线输入已由 [prepare_observer_development.py](prepare_observer_development.py) 固定到 [observer_development_r1/card.json](results/observer_development_r1/card.json)：Stair 四图一次请求、Push 一图一次请求，均为 `PREPARED_NOT_SENT`。五张原图已人工查看，仅见控制步标签，未见结果或分数字样。请求无 S0 与 measurement claims，私有 sidecar 保留重放来源；这仍不是同轨数值验证。脚本通过 `prepare_observer_request(generation_options={"thinking":{"type":"disabled"}})` 生成最终字节，再用无网络假 transport 走 `call_observer`，检验重建与原字节一致。离线回执不冒充真实服务响应。
