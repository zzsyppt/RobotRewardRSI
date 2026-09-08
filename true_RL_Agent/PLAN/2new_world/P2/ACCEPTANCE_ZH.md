# P2 验收记录：工程 GO

2026-09-08。**P2 剩余验收已完成，修复已部署正式目录，工程 GO。** `audit3_go2_full_train` 从随机初始化完成完整预算，新 checkpoint 经正式准入完成评测、原生录像和部署后的关录像对照；执行身份、历史实物与资源收尾均通过。总回执见 [本次闭环验收](results/audit_completion_acceptance.json)，逐项判断与源头修复见 [审计修复记录](AUDIT_FOLLOWUP_ZH.md)。旧/新后端数值等价仍未成立；本结论不签发 P3 新 reference 资格或 P4 新搜索成本合同。

此前 `go2_full_train_03` 的 669 条 PhysX 容量错误及首份同类失败训练继续作为失效证据；`audit2_go2_full_train` 是用户主动停止的前缀。新的完整回执没有拼接该前缀，也没有把原失败或停止回执改成通过。

## 当前修复与实测

| 项目 | 当前证据与结论 |
|---|---|
| 接触语义 | 源头恢复有状态 sensor 的按周期更新，保留上游 reset 零时刻陈旧力修复。Go2 同质量/惯量/材料、同初态同动作的 32 步对照中，接触历史、计时、首次接触及 feet_air_time 分项差均为 0；总 reward 残差最大 1.863e-8。G1 计时和历史接触判定对齐，实际受力/运动学仍有分叉。见 `results/audit_contact_sampling_comparison.json` |
| 配置与归属 | 四份 train/eval 配置的 318 处差异全部列明归因；两个接触 None 的有效值为 1 N / 4。Go2 物理默认值由其 factory 负责，共享实例化不按机器人 ID 分支。后端容量修正进入源码执行身份。见 `results/audit_environment_config_disposition.json` |
| 真实 learner / checkpoint | apply_to 只应用配置，verify_runner 验证实际构造的网络、PPO、Adam 和学习率；人为改变实际 learning_rate 的反例被拒绝。保留 policy/critic 协议词汇，原生 actor/critic 映射显式版本化。旧 checkpoint 转换信息进入评测和录像旁车；旧 optimizer 不迁移，新格式保存/加载可实际续跑 |
| 本地全量 | **1611 项、0 failure/error、19 skip**，363.688 s。真实 469 帧编码与 checkpoint tensor 回归已执行；本地不具备 Isaac 的专项仍明确 skip。见 `local_audit_regression_final.log` 与 `results/audit_final_tests.json` |
| 目标运行时 | **56 项、0 failure/error、0 skip**，564.558 s，包含三代生产形态 pause / crash / resume。准确 argv、cwd、PYTHONPATH、日志 hash 见 `results/audit_final_target_tests.json` |
| 短训练与续跑 | 最终源码 G1 与 G1-29 Push 各完成 384 control steps、checkpoint 保存/加载和实际续跑；G1 optimizer / actor / critic / normalizer 精确 round trip，Push 100 步评测覆盖施扰。此前接触修复后的 Go2 短续跑也通过。见 `results/audit_final_short_resumes.json` 及 `results/audit_go2_resume_01/result.json` |
| 原生录像 | 历史 anchor 的 Go2 C0/C1/C2 与 G1 共 **12 部影片**完整解码，36 个首/中/末帧实际查看；本次新 checkpoint 另有 **3 部影片、9 张抽帧**通过。全部帧差 0、50 FPS。G1 选中样本的两次失败保留。见 `results/audit_final_video_inspection.json` 与 `results/audit_completion_checkpoint_visual_review.json` |
| 录像开/关 | 原四组历史 anchor 对照通过；本次从正式目录关闭录像评测，完整 216-episode 任务摘要、执行身份、checkpoint 加载记录和后端核验均与新 checkpoint 开录像运行完全相同。见 `results/audit_final_video_on_off_comparison.json` 与 `results/audit_completion_deployed_video_on_off_comparison.json` |
| PhysX 失败验收 | 4096-env 作业所需 aggregate-pair 容量设为至少 2**23。四个生产 train/eval 入口在发布成功摘要前拒绝核心 PhysX error；真实溢出日志反例已通过拒绝验证。见 `results/audit_physx_error_rejection.json` |
| 正式历史 | 正式 15 rounds 的 138 份物理日志没有本次核心 PhysX 错误。部署后对 **全部 71 份 checkpoint、801 份正式证据、6 份 reference** 做 SHA256 核对，零变化；固定 Lab 的 2333 个源码文件也不变。见 `results/audit_after_deployment_verification.json` |
| 完整训练 | **通过。** `audit3_go2_full_train` 完成 2000 iterations / 196608000 control steps，最终 checkpoint iteration 1999，核心 PhysX error 为 0，实际 GPU lease 8472.390263615 s。见 `results/audit_completion_full_training_launch.json`。原 `audit2` 前缀、`model_600.pt`、停止和取消回执继续保留 |
| 新 checkpoint 评测 | 3 个公开评测 seed、216 episodes，**165/216 成功（76.39%），J_primary=0.883938290781893**。两个评测均为 `production_runner` 准入，加载格式为原生 `rsl_rl_5`，不是历史诊断豁免。见 `results/audit_completion_acceptance.json` |
| 部署 | 已向 `/data/yuyangliu/AGENT/true_RL_Agent` 交付逐文件核对的 **33 项变更**，其中删除一份无消费者的 backend YAML；部署前后均无非预期差异。新进程重算 manifest 与已验证副本一致。部署后的入口顺序测试 **1 项、0 failure/error/skip**。见 `results/audit_deployment_result.json` 与 `results/audit_completion_deployed_entrypoint_test.json` |
| 资源收尾 | 训练、两份评测及控制器均退出；三个作业进程组无残留，两块 GPU 无计算进程，既有 lease 文件均未持锁且未删除。见 `results/audit_completion_resource_closure.json` |

既有策略 ABI 的同输入 CPU/GPU 对照、新格式 optimizer round trip、timeout/bootstrap 源码检查及受控超时退出证据保留。本轮修复不改变其测试对象的科学定义；新加的实际 learner、log_std 转换回归和真实续跑补足受影响部分。不能把旧 2.4075 h 完整训练当成有效物理的性能证据。

## 冻结源码与执行位置

- 修复后执行副本：`/data/yuyangliu/isolated/new_world_p2_audit_final/true_RL_Agent`。
- 326 个项目执行文件 snapshot：`2e9958ef288164b0da8285876f057bc40ed809a1a576acb3c917d6540e895525`，见 `results/audit_final_source_identity.json`。
- 后端 execution manifest：`2d42240c8a2d34d8b67816ca23e2226ac2a2f146800f4f35f22de1b3d58bac31`，见 `results/audit_final_runtime_manifest.json`。
- 本次完整训练启动参数、预算、成功回执及成本：`results/audit_completion_full_training_launch.json`。此前 `results/audit_final_full_training_launch.json` 仍为 `STOPPED_BY_USER`，原始 runner / attempt 的失败回执保留。两次执行之间没有改产品源码，测试 fixture / Phase 报告更新不改变执行 snapshot。
- 新 checkpoint SHA256：`7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`。实际文件和最终迭代副本均保留于本次训练目录；本地记录远程实物路径，不复制为新 reference。
- 旧正式 15 rounds 仍在 `/data/yuyangliu/AGENT/true_RL_Agent/outputs/formal_go2_stair_15round_20260901_c6ab8ba`。所有旧 checkpoint 继续保护。

目标后端为 Python 3.12.14、Isaac Sim 6.0.1.0、Isaac Lab distribution 6.1.14 / source 3.0.0 beta2 patch1、RSL-RL 5.0.1、Torch 2.11.0+cu128。旧环境仅供历史迁移对照，不能作为新作业 fallback。声明身份时使用实际 manifest，不用文档简称替代源码 hash。

本次恢复验收的实际 GPU lease：完整训练 **8472.390 s**，原生录像评测 **261.606 s**，正式目录关录像评测 **129.145 s**，合计 **2.461984 GPU-h**。这是本次三份作业的占用计时，不是全部 P2 开发诊断成本，也不把编码 CPU 时间或旧失败运行混入有效训练吞吐。

## 数值边界与后续阶段

在原 public-dev 题面和三个 seed 上，旧 anchor 的对照如下。均为 216 episodes，不是新 reference 资格：

| 机器人 | 旧后端成功 | 修复后成功 | 旧 J_primary | 修复后 J_primary |
|---|---:|---:|---:|---:|
| G1 | 116/216 | 118/216 | 0.686291412047325 | 0.7124098944855968 |
| Go2 | 187/216 | 192/216 | 0.9344384820576132 | 0.9490025176954733 |

两者均存在 strata 超出预先固定的成功率差 0.10 阈值，详见 `results/audit_final_public_backend_comparison.json`。**旧/新数值等价未成立。** Lab 3 的质量采样位置和原生接触数值域变化已申报；没有改变随机化分布、任务公式或 PPO 追平旧结果，也没有实现旧 RNG 模拟层。

P3 必须在目标后端重训并资格化新协议 reference；P4 根据有效物理下的真实成本重新签发会话合同。旧 P6 generation cost v4/v5 保持历史解释，不能因单作业 timeout 提高便称其可直接启动新搜索。Go2 Push 的完整预算与任务闭环仍属于后续资格化，不由 G1-29 的 384 步诊断代替。

本次新训练只覆盖一个训练 seed，公开评测为 3 个 seed；165/216 与 J_primary 分别报告。其 `height_ood` 仅 6/27 成功、`joint_ood` 为 0/27，不能称为新 reference 已全面合格或已经证明泛化。P2 证明的是这份真实训练产物可经统一后端闭环执行。

## 证据的保留口径

原失败训练、失败录像和修复前比较保留为诊断，目录名称不代表当前资格。具体失效运行见 `results/audit_invalid_training_runs.json`；不重写原始成功回执来掩盖后来发现的物理错误。本文件只列当前可用结论。

最终历史 anchor 影片位于 `results/{audit2_go2_native_frames_c0_02,audit2_go2_native_frames_c1,audit2_go2_native_frames_c2,audit2_g1_native_frames}/video/`，准入为 `historical_diagnostic_cli_only`。每部影片从 control step 1 开始，止于 terminal 前一帧；E−1 帧、50 FPS 和时长逐部验证。录像只处理原生呈现和编码，任务 executor 决定终止与 outcome。

NPZ、MP4、抽帧 PNG、checkpoint 和 TensorBoard events 保留在磁盘，`results/.gitignore` 排除运行实物。旧、修复前、修复后源码副本和运行输出分开，避免用后来改动的源码替换原证据身份。
