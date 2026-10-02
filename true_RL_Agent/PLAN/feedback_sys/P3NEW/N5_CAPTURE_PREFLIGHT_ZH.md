# N5 采集草案静态预检

状态：DRAFT / NOT_RELEASED；2026-09-30 北京时间。未采集新数值、PNG，未运行 GPU 或 API。

基于已执行 r4 的命令，以 `--mode preflight` 在服务器 `N5_preflight_development_draft` 保留静态 composition/reference/job/result 材料；Stair 返回 `PREFLIGHT_PASS`，Push 返回 `PREPARED`。两 checkpoint 另做真实字节 SHA-256，比对训练收据及 r4 记录均一致；训练 binding 和 source manifest 沿用同一已训练父项。Push 自带 preflight 的 `checkpoint_bytes_checked=false` 是原始语义，本次独立文件哈希补验单独记录，没有改写原始回执。

预选按 SHA256(`p3new-n5-media-v1|task|env`) 在冻结域内升序取首项：

| 任务 | env | 静态映射 |
|---|---:|---|
| Stair | 2 | `public_dev.c0_id_level_3`，regular，首回合上限 600 步 |
| Stair | 63 | `public_dev.c1_conditional_frontier_level_5`，back_loaded，上限 637 步 |
| Push | 3 | low，方向索引 3，计划 1 次推扰 |
| Push | 59 | repeated，方向索引 11，计划 2 次推扰 |

Stair 使用 `--cohort diagnostic` / `public_dev_diagnostic_v1`，288 env × 3 seeds；Push 使用 base seed 1500000000，64 env × 3 seeds。媒体均为预选 2 env × 3 seeds、逐步双视角、1920×1080、各自首次终止 T 裁尾。预选不保证成功/失败类别齐全，不允许见结果后换 env。两个新 cohort binding 都与 r4 开发数据不同，但这不替代历史未用性核查。

2026-09-29 16:43 UTC 快照：数据盘可用 81,081,896,960 bytes；两张 RTX 5090 显存使用分别 7/6 MiB、利用率 0%，无 compute process。此为当前状态，不代表已完成逐步 PNG+数值+元数据的容量验收或锁定 GPU。

真正运行前：完成 observer v3r6 容量修复并重新冻结最终源码/argv；最终源码下再次 preflight；检查 checkpoint/资格绑定与 cohort 未用性；冻结抽样、缺类与先事实后报告规则；核定完整密采磁盘上界及余量、空输出目录和当时 GPU/lease 状态；签署新 release。草案的 `__FINAL_FROZEN_SOURCE_ROOT__` 必须由实际最终冻结目录替换，不能直接运行。

原始静态材料保留在服务器 `/data/yuyangliu/AGENT/feedback_p3new_20260929/N5_preflight_development_draft`，本地镜像及摘要位于 `results/N5_preflight_development_draft`。远端原始摘要 SHA-256：`2d0d420d9ac63a2fe6828e5824dac0e42c03cc2c7c291c2616f3019b48968541`。详见 [草案](./N5_CAPTURE_DRAFT.json) 中逐项路径、SHA-256、binding 与完整预选顺序。
