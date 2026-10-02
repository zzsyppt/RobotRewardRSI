# 历史导出 PNG 同字节副本清理候选（尚未执行）

2026-09-30。只在服务器本任务 `/data/yuyangliu/AGENT/feedback_p3new_20260929` 检查占用、PNG 和进程引用；未删除、移动、改写或建立硬链接。用户已授权确认后清理；本目录交付 root 执行前的逐文件候选，不是清理成功回执。

**15,238 对 PNG 全部一一对应、两侧逐文件 SHA256 相同，若将 export 一侧改为指向保留 source 的同盘硬链接，可回收 26,346,516,480 B（26.35 GB）。** 所有候选目前都是不同 inode，export 的 nlink=1；大小、inode、mtime、权限、属主、块占用在各自哈希前后没有变化。同目录非 PNG 元数据、数值档案、失败、评分和回执全部保留；R2 不在候选中。

| 组 | source/video 对应 export/video | PNG 对数 | 可回收分配字节 |
|---|---|---:|---:|
| R1 Push | `n5_confirmation_20260930/push` → `n5_confirmation_20260930/exports/push` | 4,696 | 9,059,438,592 |
| R1 Stair | `n5_confirmation_20260930/stair` → `n5_confirmation_20260930/exports/stair` | 4,900 | 7,580,155,904 |
| Dev Push body | `push_body_grounding_dev_r2` → `push_body_grounding_dev_r2_export` | 1,870 | 3,462,094,848 |
| Dev Push dual | `push_dual_view_dev_r3` → `push_dual_view_dev_r3_export` | 1,870 | 3,462,103,040 |
| Dev Stair dual | `stair_dual_view_dev_r3` → `stair_dual_view_dev_r3_export` | 1,902 | 2,782,724,096 |

以上路径都相对于任务根；两侧用同一相对 PNG 文件名匹配，不按相似图像、文件名猜测或大小判等。扫描没有 source-only/export-only PNG，没有 SHA 差异，也没有哈希期间变化。

逐文件清单为 [png_duplicate_candidates.jsonl](./png_duplicate_candidates.jsonl)，18,301,500 B，SHA256 `811d36b1f192c502e384c2c912f6d238ef87498bd90af1d1068c8619b77febb6`。每条 `file_pair` 含两侧绝对路径、两个 SHA、两侧 device/inode/nlink/size/allocated_bytes/mtime/mode/uid/gid、同字节与可回收判定；每条仍标 `execution_status=NOT_EXECUTED`。完整清单由 [readonly_png_duplicate_inventory.py](./readonly_png_duplicate_inventory.py) 生成，远端仅流式输出，本地保存，4 个只读哈希线程运行 185.6 s，退出码 0；[manifest_verification.json](./manifest_verification.json) 已重新解析全部条目与最终 summary。

**容量与活跃状态**

最后只读快照 [readonly_process_and_capacity_snapshot.json](./readonly_process_and_capacity_snapshot.json) 中 `/data` 可用 19,848,663,040 B，`/tmp` 所在根分区可用 23,405,662,208 B。合并全部候选后理论 `/data` 可用约 46,195,179,520 B；实际值仍需操作后 df 确认。只合并 R1 可释放 16,639,594,496 B，约剩 36.49 GB，首个 Stair 后再占用 8–10 GB 可能不够后续 Push 登记门；全部五组更能避免再次阻塞，且不破坏旧路径。

该快照在哈希进程结束后检查当前 UID 的命令行和打开文件描述符，未发现引用上述候选根的进程或写文件描述符；存在 4 个进程读取竞争/拒绝，故这是时点证据，不是未来不可写的锁。五个原采集 `result.json` 均为已有 PASS 结束回执；其科学/语义资格状态不由这一字段推出。执行前仍应重验候选未变及无写者。未查阅他人私有目录，未查询 GPU 或调用模型。

**保留引用与历史含义**

source 原路径、export 原路径、PNG 字节/大小/SHA 全部保留，原 `recordings.json`、各 `*.snapshots.json`、`extraction.json`、`source_summary.json`、NPZ、源清单、采集/失败/评分/调用回执不动。因此现有路径/内容引用仍可复核；不用先删除导出目录或重生成证据。清单记录原来是独立 inode；合并后应另存执行前后 inode/nlink 和 SHA 回执，不能再称 export 为独立物理备份。以后不得通过任一链接原地改写 PNG。

已有 [R2 执行说明](../n5_confirmation_execution_r2/README_ZH.md) 第 34 行明确采用同盘 PNG hardlink、路径/内容核验不变、元数据独立写出，且说明共享 inode 不是额外备份。[R2 合同](../../N5_RUN_CARD_R2_ZH.md) 第 37 行要求保全路径、大小、内容 hash 与原件身份，并保留第一轮证据。本次拟操作保留原始 source 字节和所有证据路径，只改变重复导出 PNG 的存储关系；应追加这次存储变更回执，不能改写旧文档中“当时旧 capture/export 未改”的历史记录。在已查看的合同和回执中未发现必须永久保留两份独立物理 PNG 副本的要求。

初始整目录调查：约 70,439 个普通文件路径、60,957 个 inode，文件实际分配约 74.74 GB。R2 已有 9,482 个双路径共享 inode，约 16.20 GB 物理块已经只计一次，其 source/export 的表观大小不可再加总估算释放量。未见单个文件 ≥128 MiB；少量传输包不足以解决主要缺口，且未完成逐包解压/哈希来源闭合，均未列入删除候选。4 个 R2 `sol_transfer_media.zip` 按 root 已确认的无完整解包事实保留；本报告不扩展清理范围。
