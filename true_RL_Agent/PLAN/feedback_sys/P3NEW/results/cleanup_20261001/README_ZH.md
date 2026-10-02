# 两组已结束开发 PNG 的本机归档

2026-10-01：**归档与后续远端清理均已完成。** 3,740 个原图成员逐个通过路径、大小和 SHA 校验，root 另行重读完整 tar 核对 SHA。随后经当前成员、文件状态及可见进程引用检查，仅移除清单内 7,480 个 source/export PNG 名字，目录和全部非 PNG 保留。此次对应 6,924,193,792 B 已分配块；实测空闲空间净增 6,919,716,864 B，差额包含并发 R4 写入，不把两者混称精确释放量。

证据：[归档验收](./archive_verification.json)、[root 清理复核](./root_unlink_review.json)、[实际清理回执](./unlink_execution.stdout.json)、[逐项恢复定位](./archive_restore_locator.json)。远端也先落盘同一恢复定位，再执行 unlink；位置为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/cleanup_20261001/unlink_ended_push_png_once/`，其中保留逐名删除 journal 和回执。

这些旧 PNG 的历史路径现在需要恢复后才能直接读取。恢复时按定位中的 archive_member 提取至原 source_path，再将 export_path 建为同一原图的硬链接；历史元数据与原答案不改写。以下保留归档阶段的执行说明，其中“未来”“尚未”只描述归档当时，不替代上述终态。

范围仅为远端 `feedback_p3new_20260929` 下 `push_body_grounding_dev_r2/video` 与 `push_dual_view_dev_r3/video` 的各 1,870 张原 PNG。metadata、NPZ、响应和其他目录留在原地。本操作不删除远程文件，也不在服务器生成大 tar。

`archive_two_finished_png_groups.py` 是本次固定两组的一次性操作；不要在已有结果目录直接重跑。启动登记见 `archive_registration.json`。`remote_verification.jsonl` 保存当前 source/export 的 SHA、inode、链接数及有权限可见的进程引用快照，引用旧清单而不修改它。

传输期间只有 `two_finished_push_png_sources.tar.partial`。完成传输并逐 member 验证相对路径、数量、大小、SHA 后，脚本才将其改为 `two_finished_push_png_sources.tar` 并生成 `archive_verification.json`。只有该回执状态为 `ARCHIVE_VERIFIED_REMOTE_UNLINK_NOT_PERFORMED` 才表示归档已验收；文件存在或传输进度本身不是验收。

远端预检 3,740 对均为 source/export 两名字共享 inode、`nlink=2`。未来两名字均移除时预计释放 6,924,193,792 allocated bytes；仅移除一个名字不会释放对应块。该预计值不是本次已释放量。后续 unlink 必须由 root 先审查完整归档回执，再复核实时依赖；本次脚本没有 unlink 功能。进程快照无法证明未来或无权限进程不依赖这些路径。

`r4_readonly_preparation_facts.json` 仅封存此前完成的窄范围候选 seed/cohort 查重、派生和资源事实，不是新 R4 preflight，不是采集放行；未记录的精确命令时间没有补造。
