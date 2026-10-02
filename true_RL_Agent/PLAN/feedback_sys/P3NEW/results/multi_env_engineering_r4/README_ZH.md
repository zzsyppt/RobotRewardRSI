# r4 有限双 env 采集工程核验

结论：两任务工程核验通过。比较的是已完成的旧开发 r3/r4，未使用新 N5 数据；没有启动 GPU、API 或新采集。原始结果和冻结源码未修改。

| 项目 | Stair | Push |
|---|---:|---:|
| 原始有效首回合 | 864 | 192 |
| 有效 episode/control-step 总数 | 300105 | 67331 |
| 全部动态字段数 | 15 | 12 |
| 纳入逐字节比较的动态标量数 | 32711445 | 1885268 |
| 静态字段 | terrain_env_origins_w、terrain_types | 无 |
| r3/r4 有效数值 dtype、shape、原字节 | 全部相同 | 全部相同 |
| r3/r4 原始终局记录 | 全部相同 | 全部相同 |
| r4 同轨媒体回合 | 3 seed × env 282/0 | 3 seed × env 19/55 |
| 裁尾后 PNG | 62 | 60 |

比较遍历 manifest 中实际存在的全部 NPZ 字段，没有经报告规则筛选。每个 env 先核 valid 首回合行及 control_step 为 1..T，再逐字段比较；无效 padding 不作为物理观测比较。身份检查包含 checkpoint、程序、训练绑定、协议、cohort、run/seed；job_id 和 evaluation_plan_hash 的正常变化被单列，不当作轨迹差异。

| 任务/run | 第一个 env 的 T | 第二个 env 的 T | 实际拍摄步骤 |
|---|---:|---:|---|
| Stair/1 | 282:343 | 0:310 | 282 到 320；0 到 160 |
| Stair/2 | 282:317 | 0:310 | 两者到 160 |
| Stair/3 | 282:291 | 0:300 | 两者到 160 |
| Push/1 | 19:412 | 55:111 | 19 到 320；55 到 80 |
| Push/2 | 19:412 | 55:412 | 两者到 320 |
| Push/3 | 19:111 | 55:113 | 两者到 80 |

每条记录实际步骤严格等于 `[1,20,40,80,160,320]` 中不大于自身 T 的项，每步都有 v0/v1。12 个 T 均不在固定采样点中，因此 `includes_terminal_frame` **全部为 false，且与实际文件一致**。这次稀疏 r4 不提供终止帧像素证据；r3 的逐步采集是另一份证据。记录和 PNG 文件无冲突、无裁尾后遗留图。

122 张 PNG 的 SHA、1920×1080/RGB 格式与文件完整性通过；逐步 state_before/state_after（含 clock）相同，相机 world eye/lookat 与本 env root+offset 相符。这里核的是记录与文件，像素中的全身可见性、实际隐藏/切换效果由主代理另作图像审看。

同轨核对复用冻结 r4 的任务函数，并明确不是独立物理 oracle：Stair root xyz、xyzw→wxyz quaternion、按实际 joint_names 对齐的 12 joints 与 NPZ dtype 精确一致，最大误差 0。Push world vx/vy 精确一致；roll/pitch、速度范数的最大误差为 1.1920928955078125e-7（angular_speed），小于声明的 atol=1e-5、rtol=0。Push height/resetxy 缺同回合基准，不核；没有将 Push joint/full-body 或 PNG 像素声称已核。Push 数值 manifest 不直接存明文 seed，绑定采用 run_index、cohort hash 与该 run 的媒体 seed。

两个冻结目录各 391 个 src/scripts/configs 文件，差异恰为 capture_card 登记的 4 个采集文件且逐 SHA 匹配；核验前后源码及登记的 numeric/manifest 输入 SHA 未变。结果保留原始 chunk SHA、每字段 valid-byte SHA、每个图像 SHA 和逐 sample 对齐误差。

证据：

- `stair_verification.json` SHA256 `2c585c8d4a8bca5fba687d2fc2e4e8b64248ef3b4e87e327168dec6bcdf56ede`
- `push_verification.json` SHA256 `e49e2fb0b003689ee44e7cb7e63874195e506b0fd3d14b1c2b8261b70ec0615a`
- 脚本 `../../verify_multi_env_capture.py` SHA256 `3809f24850528e4c21891bcd9be96196d4c223b9d723fb0dfedd934ccc853e27`
- 本地 `.venv/Scripts/python.exe PLAN/feedback_sys/P3NEW/verify_multi_env_capture.py --self-test`：3 tests PASS，覆盖 dtype/一位浮点变化、排除 padding 但保留终局、缺字段不能 PASS。

实际远端运行目录为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/multi_env_engineering_audit_r4`。复核命令如下；`--output` 必须使用新的结果文件名，脚本拒绝覆盖历史回执。

```bash
base=/data/yuyangliu/AGENT/feedback_p3new_20260929
task=stair  # 另一项为 push
/data/yuyangliu/envs/isaaclab6_validation/bin/python -B \
  "$base/multi_env_engineering_audit_r4/verify_multi_env_capture.py" \
  --task "$task" --baseline "$base/${task}_dual_view_dev_r3" \
  --candidate "$base/${task}_multi_env_dev_r4" \
  --baseline-source "$base/source_dual_view_r3" --source "$base/source_multi_env_r4" \
  --capture-card "$base/multi_env_engineering_audit_r4/capture_card.json" \
  --output "$base/multi_env_engineering_audit_r4/${task}_verification_recheck.json"
```

这些结果支持“本次相同 checkpoint/cohort 的有限双 env 采集未改变已记录有效数值及终局”。不据此推广到任意采集配置，也不替代 N5 或正式视觉语义验收。
