# 旧开发媒体：供 root 开发者验收

本包不是正式 N5 独立双评，没有新增 API、采集或训练，没有语义评分。

| 页面 | 原始回合 | 完整数值步 | 新 selector 图 | 已有真实观察 |
|---|---|---:|---:|---|
| [media-01](materials/media-01/review.html) | Stair r1/env282，success | 1–343 | 6 step × 2 view | r8 原 4 step × 2 view |
| [media-02](materials/media-02/review.html) | Stair r2/env282，forbidden_contact | 1–317 | 6 step × 2 view | 无，未合成回复 |
| [media-03](materials/media-03/review.html) | Push r1/env19，not_recovered | 1–412 | 6 step × 2 view | r8 原 8 step × 2 view |

每页将新 selector 帧和真实 r8 帧分开。新 selector 未调用观察者。`raw/related_numeric.json` 保留完整 1..T 相关曲线；`media.json` 给出逐图 step/view、相机、S0 定义与采样范围；`full_episode.json` 是原 full 的回合记录，不属于 facts manifest。`r8_actual` 保留原答、原始 provider JSON、核查与报告投影原件。

128 个源文件（其中 60 个 PNG）均已按 `materials/download_hash_index.json` 逐字节核验；32 个本地既有 PNG 按相同 SHA 复用，其余 28 图从原出口复制。旧目录名字不改变其与 r8 private sidecar 的真实字节对应关系。

额外容量试跑保存在远端 `confirmation_media_capacity_r1`，它在首个单位的旧 r3 缓存报告文案投影处失败，未完成四单位容量验收。测量与不改历史的当前模板内存对照见旁边的 `confirmation_media_capacity_failure_r1`：旧文案 8343/8192 B，当前已有模板 7969/8192 B。这不是新采集派生必定超限的证据；没有改写旧报告或绕过验证器。

执行源码为只读 `source_n5_preflight_r1`；实验助手封存在独立 `confirmation_media_control_r1`。未修改生产、r8、历史助手或原回执。
