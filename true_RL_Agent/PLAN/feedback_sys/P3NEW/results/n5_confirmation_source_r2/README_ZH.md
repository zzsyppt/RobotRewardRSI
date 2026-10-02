# N5 R2 统一源码与 CPU 静态预检

状态：`SOURCE_FROZEN_CPU_STATIC_PREFLIGHT_PASS_NOT_CAPTURE_RELEASE`。本记录不授权采集或模型调用；原 N5 保持 `INCONCLUSIVE`。

统一源码为本目录 `source/`，远端为 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_source_r2`。共 477 文件；`source_manifest.json` SHA256 为 `497d14b6a133d1ba3afa97e15c39b00515479692e528aac648970851245f3b76`。已有 463 文件 priority 清单作为继承基线，其内含原 445 文件完整后处理依赖；新增差分见 `source_delta.json`。此前 priority 的 114 项测试和 numeric binding 集成回执沿用，未重扫 588 项历史资产。

本轮新改动边界为六处 Stair cohort 贯通、可选 PNG hardlink exporter，以及三个相关测试文件。统一树同时纳入既有 selector、numeric export/pack、三个 visual priority 修正、R2 合同和可复用装配入口；不含新执行状态 README、driver 或最终 capture card。旧来源树及旧卡未改。

服务器环境 `/data/yuyangliu/envs/isaaclab6_validation/bin/python`，`CUDA_VISIBLE_DEVICES=''`、`PYTHONDONTWRITEBYTECODE=1`、`PYTHONPATH=<source>/src:<source>`。三个模块 `test_stair_diagnostic_evaluation`、`test_stair_numeric_dev_replay`、`test_native_snapshot_media` 共 18 项通过。两条实际 `--mode preflight` 均 exit 0；源码执行前后 477 文件完全一致。实际模块 `__file__`、SHA、命令、退出码与 stdout/stderr 见 `downloaded/control/`。

- Stair：`--cohort diagnostic --diagnostic-cohort n5_r2_20260930`，namespace 为 `public_dev_diagnostic_n5_r2_20260930`。runner seeds 为 `1945657272, 1544553419, 1187410`；旧 v1 的 `1988007583, 1057377687, 1542212762` 及 binding 保持。仅允许 public_dev。它改变同一地形域的 reset 随机流，不声称新地形泛化。
- Push：`--eval-base-seed 1600000000`，runner seeds 为 `1600010000, 1600010001, 1600010002`。
- 两个实际 preflight cohort hash 均不在已核对的 28 份历史 job commitment 内；范围见 `history_cohorts.complete.json`，不声称穷尽服务器所有潜在执行。

远端 preflight 原件位于 `/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_confirmation_preflight_r2/{stair,push}`，CPU 回执位于同 B 根的 `n5_confirmation_source_r2_control/`。已取回 27 文件并核对传输包 SHA。最终验证回执 SHA256：`248fc5b31eaa2ce626d4c0bfa6376231cc4c2f498fabe6edaa344e115d40b274`。

完整预检 argv 与未执行的采集 argv 见 `commands.candidate.json`；两者仅 mode、output、video-dir 三处 argv 不同，采集时按原 runner 要求解除 CUDA_VISIBLE_DEVICES 掩码。候选采集输出为 `B/n5_confirmation_20260930_r2`，实际控制目录为 `B/n5_execution_20260930_r2`；最终卡、资源门和 driver 由 root 独立绑定。

Push 独立 NPZ 原量来源链接仍未完整核验，源码/task-definition 绑定不替代该独立责任。旧完整长包模型执行仍未完成；这里的测试、静态容量及来源通过不构成语义资格。观察者准备器的 DeepSeek 请求 envelope 仅为既有离线模板，实际 fresh SOL 载体由 R2 卡另行绑定。此次模型、GPU 仿真和采集调用均为 0。
