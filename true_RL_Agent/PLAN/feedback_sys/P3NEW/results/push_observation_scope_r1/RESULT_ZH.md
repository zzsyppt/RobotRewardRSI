# Push 缺测说明的模态范围修正

2026-09-30。修正在任务自己的 `src/true_rl_agent/tasks/push_recovery/process_report.py`，公用层未加入任务语义。

旧句笼统说 foot/contact/yaw observations unavailable，容易把数值通道未采集误读为图像也无法观察相关部位。现在明确是 **these numeric channels** 缺少这些量；图像可见姿态是另一证据来源，不能直接替代接触测量或证明控制策略。没有凭画面补造数值字段，也没有扩大可修改的奖励输入。

用 r12 原完整档案重新构造任务语句，只改变 `push.gaps.coverage` 的文字。其余数值、过程、视觉和成功反例均保持。默认投影由11308增至11344字节，小于12288字节上限；见 [verification.json](./verification.json)。`python -m unittest tests.unit.test_push_process_report tests.unit.test_task_visual_observation -q` 实际38项通过（10.833秒）。本机未安装pytest，初次pytest入口未运行任何测试，随后使用仓库已有unittest入口；没有安装依赖。

没有新模型调用。最终两次 N.4 作者烟测使用已冻结的远端 r3 代码，不冒称其看到后续这36字节修正；新确认前要重新冻结最终源码及预检引用。该措辞检查不等于 N.5 语义验收通过。
