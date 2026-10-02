# N5 r2 本地源码快照与静态预检草案

2026-09-30。**LOCAL_SNAPSHOT_VERIFIED / NOT_RELEASED / preflight-only。** 本目录尚未部署服务器，未运行静态预检、采集、API、GPU 或训练；不构成 N.5 正式放行。

`freeze_preflight.py` 复用既有一次性复制/hash方法，只冻结当前 `src/`、`scripts/`、`configs/`（排除 `__pycache__` 和 `.pyc`）及以下四个既有采集助手：`extract_dual_view_evidence.py`、`prepare_native_observer.py`、`prepare_n5_media_selection.py`、`verify_multi_env_capture.py`。评阅协议、新算术脚本、最终模型角色/材料/容量/费用卡仍待分别封存，未放入本次源包。

- 源快照：400 文件；manifest SHA `3564dbfaebd363aaa28245068838bb5c63ab040741d28d57daf6cc50d1dfd365`。
- 源包：`source.tar.gz`，1,302,240 bytes，SHA `c4ba2a62fe5baedec79551b72875eb5549160f7cdc8566e9171a701d8b2bddff`。
- 新草案：`preflight_card.json`，SHA `b7ec61c62afdd37156eb6b713042acc8ccf2e68f7720676e607d8dceefc999e6`。
- 预定服务器源根：`/data/yuyangliu/AGENT/feedback_p3new_20260929/source_n5_preflight_r2`。
- 预定预检输出：`/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_preflight_r2/{stair,push}`。

完整 argv 位于新卡 `commands.*.preflight_argv_not_executed`，均明确 `--mode preflight`。卡内没有 run argv 或 capture release。checkpoint、训练 summary/source manifest、训练 binding 与旧 evaluation binding 仅作为继承的待核引用；新结果必须由服务器实际预检和身份核查产生，不能抄旧 PASS。旧 GPU/磁盘/process 与 seed 新鲜性证据均标记待刷新。

登记范围保持：Stair `public_dev_diagnostic_v1`、env 2/63；Push base seed 1500000000、env 3/59；各三 seed、两视角、1920×1080、interval=1、各回合自身终点裁尾。相机 eye/lookat、checkpoint 和 cohort 参数与旧登记逐项核对保持。Stair 协议路径改为本次源根，当前本地协议 SHA 另列，尚无远端确认。

相对旧 r1 manifest（392 文件、`6fe815234de2f577fdd5236805ed5319673b17a7e9b6f1869e6c447fc39423dc`），385 文件不变、8 新增、7 修改、0 删除，精确逐文件变化见 `source_changes.json`。新增项是四个上述助手、G1 身体锚点配置、通用投影/语义review模块及 Push 身体几何模块；修改项为媒体采集、观察者、共用/两任务过程报告与两任务视觉接线。

`local_freeze_receipt.json` 记录复制前后整个工作树纳入范围的 SHA、快照逐文件 SHA、tar 逐条内容 SHA 均相等；旧 `N5_CAPTURE_CURRENT_DRAFT.json` 与 r1 manifest 原哈希未变。命令和助手清单另经本地只读核对。当前结果只证明本地封存完整，下一步由 root 审阅后决定是否执行静态预检。


后续状态：本地封存之后，Stair behavior/process_report 已发生 v1.4.15 进度过程修正。本源包仍未部署或执行；其400文件哈希及草案保留作历史，不是最新运行冻结。最终preflight需以当前代码重新封存，不能复用本卡放行。
