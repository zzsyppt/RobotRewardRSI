# 旧训练 v1 档案的冻结 checker 兼容回读

2026-09-24 在服务器使用启动锚点修订前的冻结源码，对既有四组训练 on/off 档案只做 CPU 检查；该次确切源码 SHA256 见 `summary.json.source_hashes`。本记录不是最终锚点版 checker 的真实档案回读。四组均 PASS：Push G1/Go2 每组 24 控制步、384 环境转移；Stair G1/Go2 每组 8 控制步、128 环境转移。全部仍按 `training_numeric_capture.v1` 读取，未将旧连续窗口重新解释成 v2 分散采样。

当前 checker 的算术、覆盖、身份、checkpoint 模型/优化器及检查结果，与原报告对应字段逐值相等。每组原报告、两份 result/summary/argv、数值 manifest 和两份 time 共 10 个文件的 SHA256 前后不变；checkpoint 与数值块由复用的正式读取器校验，无额外前后全文件重哈希。每份 compact JSON 保存这些来源 hash 及逐项结果，`summary.json` 保存执行源码指纹，`recheck.log` 保存四组输出。

实际命令（CPU 单线程，输出全部位于新目录）：

```sh
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /data/yuyangliu/envs/isaaclab6_validation/bin/python \
  /data/yuyangliu/AGENT/feedback_p2_audit_20260924/tools/recheck_training_v1.py \
  --project-root /data/yuyangliu/AGENT/feedback_p2_audit_20260924/source \
  --pair-verifier /data/yuyangliu/AGENT/feedback_p2_audit_20260924/tools/verify_training_pair.py \
  --legacy-reports /data/yuyangliu/AGENT/feedback_p2_20260923 \
  --output-dir /data/yuyangliu/AGENT/feedback_p2_audit_20260924/legacy_training_recheck
```

脚本内部还设置 `torch.set_num_threads(1)`。这是旧证据的派生回读：新模拟器运行数、训练数、模型调用数均为零，不重复登记原 GPU 成本，也不代替 v2 的真实模拟器验收。
