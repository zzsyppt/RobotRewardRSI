# P1d replay CLI GPU binding 窄修复

用户在 P2a 交付期间追加授权，独立于 P2a `e79fa7ef` 提交；没有改变轨迹测量、阈值、模板、
卡片或三源结果。本地只有主 writer 修改；不启动 GPU、不写服务器、不改设备 settings。

## 已读取的实证

服务器只读原文：
`/data/yuyangliu/isolated/p1d_gpu_aaf67aa_20261003_qualification01/QUALIFICATION_REPORT_ZH.md`。
本地精确副本 `outputs/gpu_binding/QUALIFICATION_REPORT_ZH.md`，8900B，SHA256
`3497b31e70a526ec8cdc09668c2a5f3ddfd978c9c0c47695cc1cac8d416b253d`。

固定 `aaf67aa23bd7e4aa8aae7a9ec9539f3aa45c6c3e` 原 CLI 首次单卡隔离失败，同 PID 使用两块
GPU，资格线程发现后主动停止。唯一重试使用原样生产 API 显式传入既有 binding，完成正式 v3
public-dev 864 回合并观察到单物理 UUID。**该 API 重试成功不覆盖原 CLI 失败，也不是本修复
commit 的 GPU 实测通过**。本次开发线程没有新增 GPU 尝试；修复后的 CLI 交原资格线程后续验证。

## 改动与依据

正常 composition 的 `agent_core/generation_preparation.py` 使用
`GpuProcessBinding(host_execution_device, process_execution_device, resource_binding_hash)`。
replay CLI 原来仅创建 GPU lease/scope，没有向 `materialize_reference_jobs` 传此 binding，
故 runner 的 UUID 可见性及 Kit 隔离链未启用；仅 lease 不限制其他 GPU。

现在 run 从已发现 manager 的真实目标 identity 构造同一 binding：host仍取已有 `--device`，
child为 `cuda:0`，resource hash与 attempt_scope相同。通过已有 backend参数进入 eval plan，
formal/diagnostic phase继承；正常 runner按host租赁并设置子进程UUID环境。
preflight保持None、无GPU discovery/probe/launch。没有新CLI字段、手拼GPU argv、新管理框架、
全局环境设置或核心runner变更。

## CPU 验证

`tests.unit.test_stair_numeric_dev_replay` 10项PASS。扩展原有正式/命名诊断测试使用host `cuda:1`，
避免cuda:0掩盖映射缺口，核对：

- child phase/device为cuda:0，binding host为cuda:1，scope/resource identity hash相等；
- `--device`及三个`--gpu-*`参数各一次，值来自同一binding；
- 诊断路径保留真实`run_eval_phase`与真实lease manager，只模拟rollout/finalization；
  执行context `CUDA_VISIBLE_DEVICES=GPU-test`、`NVIDIA_VISIBLE_DEVICES=None`，host1租赁被占用；
- CPU finalization前租赁已释放，probe失败亦释放；preflight无binding或GPU参数；
  来源训练身份、checkpoint及正式864回合计划等原验证保持。

相关五模块完整执行143项＝**139 PASS＋4环境SKIP**，32.484s；原始命令/日志/hash见
[GPU_BINDING_VERIFICATION.json](./GPU_BINDING_VERIFICATION.json)。模块为replay CLI、
GPU binding projection、Isaac backend、runner P1 reliability、subprocess runner。
将新的两条binding断言应用于从git读取的旧CLI原文，2/2按预期失败，原因明确为binding=None；
未修改工作树源码来制造对照。修复代码下两条均PASS。

只读Astra ultra复核代码、报告SHA、原日志及测试覆盖PASS。它未独立运行GPU；本地CPU单测
不是单物理GPU运行资格。后续以本修复commit在服务器运行的明确回执为准。
