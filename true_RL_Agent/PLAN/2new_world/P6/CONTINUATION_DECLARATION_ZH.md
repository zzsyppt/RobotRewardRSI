# P6 完整训练后的评测续接声明

此声明在首次 Push 正式 dev / accept / sealed 评测前签发，机器可核对版本见 [continuation_declaration_01.json](continuation_declaration_01.json)。原 [formal_declaration_01.json](formal_declaration_01.json) 继续拥有科学协议、奖励、样本数、资格门及资源预算；原失败记录保留。

两格已各完成一次 random-init、4096×24×500（49,152,000 env-steps）训练，最终 iteration=499。原资格进程因共享 Genesis 要求的 `formal_training_executed` 字段仅由 Stair writer 提供，在任何评测前退出。修复由共享 Runner 根据实际预算、最终 iteration、初始策略身份、摘要身份及 checkpoint 字节规范化完成状态；原 summary 与 checkpoint 不改写。明确的 False 不提升为 True。

续接通过共享 Genesis 的已完成训练入口保留旧 TrainJobSpec 与来源，在新目录 `outputs/p6_reference_continuation_20260912/{go2,g1}` 运行原定三相，各一次；禁止重训、增加样本、调整门槛或按成绩选择 checkpoint。原 session、科学身份、设备分别保持 Go2/cuda:0 与 G1/cuda:1。旧训练和新评测拥有各自的执行环境身份，不把旧训练重标为新源码执行。

原完整环境未在训练前落盘。主任务在更换源码前重建完整环境，其 manifest hash 与两份原始训练摘要完全一致；此重建事实必须保留。旧 371 个执行源码文件已逐字节匹配原冻结声明并归档，见 [original_training_source_archive_01.json](original_training_source_archive_01.json)。后续普通资格在启动训练前即保存 TrainJobSpec 与完整环境，关闭这一证据缺口。

原 Go2 耗时 666.8159349877387 s，G1 耗时 931.7349299285561 s。外层 wrapper 和内层 CLI 必须传同一份原 process receipt，扣减后本次上限分别为 **13733 s / 13468 s**；原每格 14400 s 总上限不清零。此为已定位宿主缺陷修复后的显式续接，自动基础设施重试仍为 0。续接再失败不得另换目录重跑；保存实际结论。

验收沿用原资格门。通过需要真实三相、可独立 load-only 的 Genesis、同 checkpoint 原生录像及成本/进程闭合；未通过则给出真实 NO-GO，不能生成伪 Genesis 或根据 holdout 回调参数。首次资格成绩前还需实际目标环境导入预检与当前 source snapshot 对齐。
