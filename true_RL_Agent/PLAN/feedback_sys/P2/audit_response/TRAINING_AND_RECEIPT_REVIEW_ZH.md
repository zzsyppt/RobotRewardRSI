# 训练采样与回执审计处置（2026-09-24）

本记录覆盖审计 5.3、5.4、6.3、6.4；只列本轮实际修改及证据边界。没有启动 GPU、训练或模型调用。

## 5.3：调整取点，保持数据预算

原审计相关性脚本将 `X[:-1]` 与 `X[1:]` 展平后计算相关系数；数据来自计划 8 或 24 控制步的短训练。该数值混合环境差异与相邻时刻关系，不能推出正式三阶段取样没有分辨力，更不能据此估计独立样本量。旧 v1 的 48000 步计划在早、中、晚各连续取八步，三个阶段跨越完整计划，并非都来自同一 PPO rollout。源码还累计非零计数及最大绝对加权值，“所有证据只保留 last”不准确。

已将新采样 schema 升为 `true_rl_agent.training_numeric_capture.v2`：把计划控制步划为三个互不重叠的区段，每段最多八个等宽子区间取中心点，再将 early 首个点固定为第 1 步启动锚点；最多 24 步 × 16 环境，未加 aggregator。最终 strategy 为 `step_one_anchor_then_equal_bin_centers_within_disjoint_training_thirds`。48000 步计划对应 1、3000、…、15000；17000、…、31000；33000、…、47000。三个区段样本跨度分别为 14999、14000、14000 步，24 步一轮时分别落在八个不同 PPO rollout。8/24 步短验收仍采集全部控制步。

私有档案明确保存计划/实际索引、区段范围、分母与未到达/部分采样状态。公开投影仅识别 v2 schema，不增加字段。checker 分版本核验：v1 连续窗口不被重新解释，v2 独立重建抽点并核对实际索引。它仍不能给出全程均值、罕见事件完整频率或精确首次非零时刻；固定前 16 个环境也不是随机代表性样本。

启动锚点修订时，本轮新快照尚未执行训练。锚点使第 100 步提前失败仍保存第 1 步 raw/weighted/total，避免只剩覆盖信息；该样本受 reset 分布影响，也不能代表第 100 步失败瞬间。最终两目标模块回归覆盖 48000 步预算与索引、实际第 100 步截断后的单点档案与算术复算、partial 不冒充全窗口通过。没有增加失败触发采样。新训练使用锚点版另行冻结的快照，先前评估快照及旧 v1 回读来源指纹保留历史身份。

## 6.3：只复用本次验证结果

`validate_generation_ledger_record` 增加仅输出参数 `validated_execution_records_out`，内部始终重新加载并验证绑定回执，所有校验成功后才写出结果。公开投影使用自己新建的局部字典，不接受外部缓存；失败 lane 复用已验证执行记录。formal 的公共入口仍先完整验证，已验证的公开 history 路径只调用私有 `_project_validated_generation_formal_round_record` 派生，因此 formal 有无均不再重复调用整个 validator。未修改持久化格式或哈希准入条件。

真实绑定的双失败 lane 与既有 formal CPU 集成证明：一次直接公开投影中，每个持有 execution hash 的 lane 只调用一次通用 `_load_bound_trial_execution`，结果与完整历史投影相等；后段校验失败不输出半成品；伪 execution hash 仍拒绝。formal 派生仍会读取并核对其正式回执，这是原有 formal 身份校验。本优化不宣称整个 acceptance replay 或所有回执类型只读一次，也未扩展成通用缓存。

## 5.4、6.4：保留的边界

- 普通 width-one 的 8 KiB recent history 确实会重复携带 `training_sampling` 详情，可能挤掉旧轮次；最新 task runtime 也独立提供详细覆盖。本轮不扩张旧 width-one 改造或预算。正式 generation 的 14 KiB 长期记忆已使用短执行状态；不能把普通轮次问题推广成 generation 容量回退。原审计的具体字节数没有保存独立重建驱动，本轮不当作可复现实测。
- 新 `scripts/stair_numeric_dev_replay.py` 的训练完成、预算、迭代、程序/绑定/来源验证，不能被 ancestry hook 的祖先准入校验完全替代。旧 PLAN replay 的 reference 模板替换方式及 metadata 不等价于当前入口；应标为历史入口，保留历史行为。无需为了删行再提取只有一个消费者的通用框架。

## 本轮 CPU 验证

| 命令 | 结果 | 日志 |
|---|---|---|
| `python -m unittest tests.unit.test_training_numeric_capture tests.unit.test_training_numeric_smoke tests.unit.test_training_sampling_projection -v` | 26 项通过，8.114 s | [日志](verification_training_sampling_v2.log) |
| `python -m unittest tests.unit.test_training_numeric_capture tests.unit.test_training_numeric_smoke -v` | 最终锚点版 21 项通过，3.340 s | [日志](verification_training_sampling_v2_anchor.log) |
| `python -m unittest discover -s PLAN/feedback_sys/P2 -p test_training_pair_checks.py -v` | 5 项通过，1.986 s | [日志](verification_training_pair_v2.log) |
| `python -m unittest tests.integration.test_fixed_width_generation -v` | 2 项通过，51.407 s | [日志](verification_generation_receipt_reuse.log) |

26 项日志对应启动锚点修订前的 v2；21 项日志是最终 sampler/checker 的相关回归，未重跑无关 generation。解释器为仓库 `.venv/Scripts/python.exe`。针对修改文件的 `git diff --check` 通过。真实新 profile 的模拟器验收由本目录其他验收记录报告，本记录不代替该证据。
