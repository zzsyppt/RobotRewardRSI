# 正式准备路径开发复审

状态：**机械开发验收通过；不是 N5，不是语义评阅，不放行任何调用。**

本次只检查并修改 `export_confirmation_raw.py` 和 `prepare_confirmation_media.py` 的新正式入口。没有改原始材料、旧答案、评分、生产诊断逻辑或采集规则。

## 真实缺口与最小修复

1. 原导出入口仅核对 FROZEN 卡和协议哈希，未把传入原始 manifest 接到卡里的任务、检查点、评估队列和样本规模。现在正式导出在选择样本前核对：任务、checkpoint、完整 `current_evaluation_binding`、三次 public_dev 运行及环境数；Stair 再核 namespace/每次 outcome 数，Push 核 run 顺序、env 行顺序及三份 manifest 的共同来源身份。完成时重核 manifest 和卡/协议未变。
2. 媒体正式入口复用同一原始来源检查，前置检查 full/extraction 的选择绑定，拒绝重复 packet id、重复选择别名和成功类别错配，并在结束前重核读入材料。保留原有 dense native build 门；不允许重用历史回答或补重复容量案例。
3. 选择层的 `e1/e2` 表示成功/失败槽，单 episode 盲包仍统一使用 `e1`。回执新增 `packet_alias`，明确两者映射，避免把选择别名误认成包内别名。两家收到的原始证据内容必须逐字一致。

Push raw manifest **没有 runner_seed 字段**，因此这里没有声称直接核对这个不存在的字段。正式卡必须保存预检产生的完整 `tasks[task].current_evaluation_binding`；其中 `seed_cohort_hash` 绑定预检的 cohort。媒体选择工具原有的 recording.runner_seed 核对仍承担直接种子核对。没有新建签名体系或重新解释仿真测量。

## 实际检查

- 原 raw exporter 的 7 项 self-test 全部通过。
- `verify_paths.py` 的 13 项机械检查通过，结果见 `verification.json`。
- 从已有原始 NPZ 完整跑正式导出分支：Stair 864 + Push 192 的分母保留；24 案例、12 focus 不变；540,445 个选中原始值 JSON/NPZ 回读一致。
- 拒绝错误 checkpoint/cohort、Push 顺序或共同来源漂移、Stair 缺环境、Push outcome 行错位、未冻结卡和变化的协议。
- 媒体编排使用旧真实序列化包的数值、12 张原始 PNG 与新的冻结协议，实际构造两家 HTTP 请求，证据 parts 相同，facts 无 report inventory；历史回答和错误结果别名被拒绝。

媒体检查中 **`build_unit` 被明确替换为已有旧包读取器**，目的仅是验证新正式编排，不冒称又跑过当前完整 dense 采集/构建。冻结卡是带 `synthetic_only/not_N5` 标记的开发 fixture。临时请求和大导出已经由受限临时目录清理，小回执、manifest、命令脚本、原值计数和 source hash 保留，可复跑；临时路径不是新的正式材料地址。

首次媒体验收因本地 `.venv` 没有 Pillow 而停止，随后只把已有 Codex Python 同版本的 site-packages 加入本次 Python 进程搜索路径。没有安装依赖、联网、SSH、API、GPU、训练或模型调用。

复跑命令（仓库根目录）：

```powershell
.venv/Scripts/python.exe -B PLAN/feedback_sys/P3NEW/export_confirmation_raw.py --self-test
.venv/Scripts/python.exe -B -c "import sys, runpy; sys.path.append('C:/Users/lover/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/Lib/site-packages'); runpy.run_path('PLAN/feedback_sys/P3NEW/results/confirmation_formal_path_development_r1/verify_paths.py', run_name='__main__')"
```

这些检查没有证明正式新采集已完成、模型完整阅读、语义支持率或容量门通过；相关正式状态维持未放行。
