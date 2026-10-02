# 旧开发 Push 逐步采集工程记录

本次仅用于验证 v4 状态、相机和图像能被严格的完整轨迹输入入口消费，不属于 N5，不训练、不调用模型 API。

- 使用 r1 原封不动的 401 文件冻结源；manifest SHA256 为 `b406b38f0e5befa647115a41b505474af3c93d207f84951e5b16b31bad448645`。
- 执行卡修订版 SHA256 为 `24d41a6348f1d2c6cc9992c4c5f82916bc6f6e43028d1ff2e8dfb3e8b0cc5843`。修订仅更正 baseline 卡哈希，命令不变；原卡保存在 `control/capture_card.not_released_v1.json`。
- GPU 实际执行 1 次；2026-09-29 21:33:08 UTC 启动，生产状态 PASS，GPU 租约计量 887.1110228755098 秒。API 调用 0 次、训练 0 次。
- 第一次 SSH 启动连接失败，保留了当时未知状态的 `control/launch_transport_failure.json`。随后只读核实没有输出目录、启动记录、资源预检记录及对应执行进程后，才进行了上面的实际首次执行；不是重复启动已有任务。

## 已完成验收

`receipt/verification.json` 为 PASS：3 个旧开发种子、每种子 64 条评估轨迹，共 192 条；12 个数值字段逐 dtype 和有效字节、全部终止记录均与 r1 完全一致。r1 已与旧 r4 对齐。本轮选定 env19 的轨迹长度分别为 412、412、111，共 935 个时刻、1870 张双视角原始 RGB PNG，覆盖各自 1..T，包含终止帧。

所有原图为 1920×1080，文件哈希、样本绑定、全状态渲染前后相等、部位名称排列、双视角相机参数均通过检查；与数值数据的全部重合时刻对齐。实际相机逆视图矩阵原点与采集记录的最大绝对差为 2.384185835424546e-7 米。这不表示已证明独立的屏幕投影误差界。

完整服务器导出：`/data/yuyangliu/AGENT/feedback_p3new_20260929/push_body_grounding_dev_r2_export`；`extraction.json` SHA256 为 `d612bceab5477569c2756329c958fdc949bed136ac9a71d5f2bc9b84982f924d`。Extractor 自身登记 `not_checked_by_extractor`；本轮对齐结论来自上述独立核验记录，没有改写导出器声明。

## 原图复核交接

run1 的 1、44、71、109、206、412 步，两视角共 12 张原图已下载到 `materials/push_run01/video/`。这些时刻的全状态、相机参数与 r1 完全一致，但 12 张 PNG 的字节哈希均不同，因此没有直接继承 r1 的像素评阅结论。具体表在 `materials/selected_image_comparison_r1.json` 与 `materials/selected_state_camera_comparison_r1.json`。

Root 已逐张查看这 12 张新版原图，复核记录为 [root_pixel_review.json](receipt/root_pixel_review.json)，SHA256 `3dda2e1d0faf1faec4817e6fc4fd94c562fb74fabc8580384506437ba03a8ccf`。其余 1858 张仅经过完整性/哈希/分辨率/状态绑定等机械核验，未进行逐像素语义审阅。本次没有进行严格消融，不能把后续差异归因于某一新增输入因素。

本地小包 SHA256 为 `f8f6e069c14ada8ce145b827609cad9f8976e9f3a3f2b20cceecb317818e604e`，24,541,608 字节；26 个索引文件全部重新核验。R11 观察输入准备及后续模型调用不计入本记录，本轮没有执行它们。
