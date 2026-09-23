# P2 训练数值采样实现与验收边界

2026-09-23。本记录只说明本次实现及 CPU 检查；真实 Isaac Lab 短训练结果由 P2 当次运行 receipt 补证。这里没有启动 GPU、训练或模型 API，也没有修改 P1 历史验收记录。

## 实现

`runner/training_numeric_capture.py` 只负责固定采样窗口、时钟、环境子集和共享 NPZ/manifest I/O。任务语义仍由 Stair `reward_program.py` 与 Push `runtime_adapter.py` 给出。一个训练进程只调用一次原 `runner.learn`；采样 hook 读取这次真实 reward dispatch 已有的张量，不重复执行候选 reward，不修改返回奖励、不消耗 RNG。

- 使用运行前已知的 `planned_control_steps = iterations × steps_per_env`。2026-09-24 审计修订后的 v2 在三个互不重叠的三等分内各分散取最多 8 个控制步，并将 early 的首个点固定为第 1 步启动锚点；每步只保存前 16 个环境。48000 步计划的 early 为 `[1, 3000, 5000, 7000, 9000, 11000, 13000, 15000]`。完整长训练最多 24 × 16 = 384 个 env transitions，不能代表全训练分布。锚点可保留提前失败时的启动样本，不代表失败瞬间；旧 v1 的首端/中心/尾端连续窗口按原 schema 解读，不重标历史档案。
- `learn` 前、activation preflight 和 runner 构造后冻结 `source_step_offset`；训练步从偏移后的 1 开始。保留源控制步、训练控制步、rollout 内步数、由控制步推导的 learning iteration 和 runner 当时报告的 iteration。后者明确为 reported 值，不替代推导时钟。
- 完成或失败时，每窗分别报告 `sampled / partial / not_reached / not_sampled`；极短区间还可能 `not_scheduled`。归档 `complete` 仅指 manifest 完整，`training_status` 单独记录训练成败。请求了采样但未开始为 `not_started`，关闭为 `not_enabled`。
- 保存原始 dtype 的 guarded/clipped raw term、真实 weighted term、真实 total，以及 term 的名称、索引、声明权重、clip 范围。Push 每步另外保存实际 schedule scale、input reward、candidate total 和 returned reward。
- Stair total 是 RewardManager 乘 `dt` 之前的 rate；physical Push returned reward 同样是 rate，外层 RewardManager 乘一次 `dt`；legacy additive Push 已是 per-step reward。保存 `control_dt_s` 和这层关系，**没有声称另行观测到 PPO 收到的 post-manager reward**。
- 归档身份包含真实程序 checksum/qualification、job/attempt、初始 policy hash、训练 seed、run dir、resolved cell manifest、当前 backend environment 验证；Stair 同时包含实际 source preflight。Stair 接口用 `numeric_evidence_identity` 接收已有入口事实，内部实际身份覆盖同名外部值，不另造绑定解析器。

backend 配置与底层执行接口的 capture 默认值仍为 False；当前正常 Stair unified composition 和 Push physical composition 已同时开启 `capture_public_dev_numeric / capture_training_numeric`，backend/CLI 已接通。smoke driver 用显式 on/off 覆盖训练采样 flag。产物目录固定为每个 fresh `run_dir/training_numeric_evidence`。

## CPU 检查

`tests/unit/test_training_numeric_capture.py` 覆盖真实 Stair qualified dispatcher、Push physical/additive adapter 的 on/off 数值与 runtime summary 相同、RNG 相同、raw/weighted 关系、预检偏移、固定窗口、失败/未采样状态、环境子集、dtype、NPZ 回读和重复结束。`tests/unit/test_training_numeric_smoke.py` 覆盖调用保真、错误预算/入口拒绝、CPU preflight 零启动、Stair smoke 路由不冒充正式训练，以及独立离线算术校验对坏 total 的拒绝。

r2 阶段的关联检查见 `training_numeric_cpu.log`：项目 `.venv/Scripts/python.exe` 下 64 项通过，11.772 s；其中采样/驱动直接回归 12 项。该阶段 `git diff --check` 通过，训练入口、sampler、两个 task hook、Stair executor、smoke driver 共 7 份源码 SHA 与 `source_snapshot_r2.json` 一致。它是历史快照，后续 driver/checker 修复另见文末，不代表当前源码仍等于 r2。生成器另在临时目录以**明确 mocked environment hash** 做了两个 task 的 composition/materialize CPU 检查；该检查只证明生成路径可用，不证明本地或远端 Isaac 环境合格。

## 真实验收准备与调用

先将当次冻结源码同步至独立隔离目录；下面 `/absolute/frozen/source` 必须替换为与该次 receipt 配套的真实快照，不能混用历史 r2 和已修复 driver。在该目录的真实 Isaac Python 下执行准备器；它只读取已安装源码/包、生成当前 composition 与合同，不启动 simulator 或训练：

```bash
/data/yuyangliu/envs/isaaclab6_validation/bin/python PLAN/feedback_sys/P2/prepare_training_smoke.py \
  --project-root /absolute/frozen/source \
  --task stair --output /absolute/fresh/stair_input --device cuda:0
```

Push 把 `--task` 改为 `push`，输出另一个 fresh 目录。默认均为 Go2 当前 v3 官方 reference。可指定 `--candidate-dir /absolute/intact/saved/candidate`：按当前 task admission 重新 seal/verify 完整原包，成功后生成 candidate job；没有放宽源身份/资格检查，没有变更原候选源码。缺少完整包或当前 ABI 不兼容就不得宣称真实 candidate 验证通过。

准备器现在接受 `--cell`，只允许与 `--task` 对应的当前 G1/Go2 v3：`stair.g1_37dof.unified_v3`、`stair.go2_12dof.unified_v3`、`push.g1_37dof.recovery_v3`、`push.go2_12dof.recovery_v3`。四种组合必须各自准备和运行 on/off，不能用 Go2 的通过代替 G1。四组合的 mocked-environment CPU materialization 检查见 `training_pair_preparation_cpu.log`，它们不是四次真实训练通过。

然后分别执行相同输入的 on/off。准备器生成 `train_argv.json` 原始列表；`TrainJobSpec.to_public_dict()` 自身不含 argv，不能直接代替这个文件。

```bash
/data/yuyangliu/envs/isaaclab6_validation/bin/python scripts/training_numeric_smoke.py \
  --train-argv /absolute/fresh/stair_input/train_argv.json \
  --output /absolute/fresh/stair_on --capture on --device cuda:0 \
  --isaac-python /data/yuyangliu/envs/isaaclab6_validation/bin/python --mode run --timeout-s 900
```

`--mode preflight` 仅生成参数，不访问 GPU；实际运行之前仍须检查当前 GPU 占用。运行模式使用现有 `GpuLeaseManager` 和 `run_owned_process`、单物理 UUID 映射与 owned-process 清理，真实 `on_started` 才增加启动数。不得预先设置 `CUDA_VISIBLE_DEVICES`。off 使用同一 input，另一个 fresh output，仅切换 `--capture off`。`result.json` 保留执行命令、输入 SHA、lease、启动数、墙钟、初始 policy hash 和归档引用。

- Stair 输入由正常 formal composition 生成并通过入口身份/候选资格检查，执行时明确调用现有 `execute_stair_training_smoke`：16 env × 8 steps × 1 iteration = 128 transitions。报告 `formal_training_executed=false`，三窗为 2/3/3 控制步；它们都位于同一 rollout iteration，**不是跨多次 PPO 更新的早中晚效果证据**。
- Push 用当前 composition **新建**诊断合同 16 × 8 × 3 = 384 transitions，再调用原训练入口；不是手改 formal argv 的预算。三窗各 8 个控制步，覆盖 3 次 rollout iteration。子 receipt 是该短合同的真实训练完成，外层明确 `formal_training_claim=false / policy_qualification=false`，不能归为原 49,152,000 步正式训练。
- on 归档需三个窗口均 sampled、identity 与输入一致、所有 chunk hash 可读、独立 raw × weight × scale → weighted、weighted sum → 实际 total 以及 Push input + candidate → returned 关系通过。独立算术容差 `rtol=atol=1e-6`。off 需 `not_enabled`。比较两次真实运行的初始 policy hash、reward/runtime 与成本，差异应原样报告；跨进程 GPU 数值不自动承诺逐位确定性。

每个子进程最多 900 s；Stair/Push × G1/Go2 四个 reference case 各跑 on/off，单轮计划为 8 次训练启动、120 GPU-minute 的超时上限，**是规划上界，不是已测成本，也不包含失败和重新验收的额外启动**。实际 r4/r5/r6 等每次启动与成本必须由当次 receipt 分别登记。若增加一对真实 candidate on/off，则再单独登记两次启动及成本。这个 smoke 能验证采样接线、时钟与数值保存；不能证明完整训练收益、全分布覆盖或正式策略资格。

## 独立 on/off 比对

`verify_training_pair.py` 只读两个现成的 driver 输出目录。在同一冻结源码/真实 CPU Torch 环境下运行，例如：

```bash
python PLAN/feedback_sys/P2/verify_training_pair.py \
  --project-root /absolute/frozen/source \
  --on /absolute/stair_g1_on --off /absolute/stair_g1_off \
  --cell stair.g1_37dof.unified_v3 --output /absolute/stair_g1_pair_check.json
```

它独立检查输入 source SHA、除 capture/output 外的完整 executed argv、实际 cell、程序与训练绑定、summary/checkpoint 文件绑定、初始 policy hash、实际 dispatcher 控制步（Stair 扣除 activation preflight）、三窗索引与分母，以及归档 checker 的 identity/coverage/数值重算。随后 `torch.load(weights_only=True, map_location="cpu")` 比对全部非 optimizer `state_dict` 与全部 optimizer 字段，逐张量核 shape/dtype/`torch.equal`，并比较其余状态标量。非状态 metadata 单独列出，不以训练文件容器 SHA 差异替代模型/优化器比较。

自动读取 `<on-dir>.time`、`<off-dir>.time`，格式为 `peak_rss_kib=... elapsed_s=... exit=...`；也接受 `--on-time / --off-time` 显式路径和 GNU time -v 原输出。缺失 RSS 会返回 `INCOMPLETE_RESOURCE_MEASUREMENT`，不会记作零。记录 on/off driver wall、外层 time elapsed、RSS、numeric archive 实际存储/原始字节及整个输出目录字节差。wall 包含启动、训练、退出和 driver 的数字校验；RSS 为 GNU time 的进程高水位观测，非同时进程树 RSS 总和；单次 pair 差值不是稳健的性能估计。

资源验收同时要求 numeric manifest+chunks ≤ 2 GiB、按 shape×dtype 重算的单 chunk 原始数组 ≤ 64 MiB、paired 额外 RSS ≤ 512 MiB；wall 只作观测，不设短训练耗时增幅门槛。`test_training_pair_checks.py` 用临时 synthetic fixtures 检查正常对照、metadata/文件 SHA 差异、优化器张量变更、缺失 RSS、运行步数不足和资源超限。这些 fixture 只验证验收工具能拒绝错误，不计入真实 simulator 证据。早期 3 项结果在 `training_pair_checks_cpu.log`；协议身份修复后，Stair fixture 改用真实 v3 manifest 与祖先绑定协议，训练张量仍为 synthetic，5 项通过（1.897 s，`training_pair_protocol_identity_fix_cpu.log`）。真实 GPU pair 如有状态差异则 invariance 不通过，但单个 pair 不能据此认定由 capture 导致。

## 首次真实训练启动后的解释器修复

`r4_train_push_go2_off/process/stderr.log` 暴露 `ModuleNotFoundError: isaaclab`：driver 将 `/data/yuyangliu/envs/isaaclab6_validation/bin/python` 经 `Path.resolve()` 展开成基础 Python，丢失 venv 的启动路径。修复只改 driver：解释器保留 `expanduser().absolute()` 路径并检查文件存在。其余 `resolve()` 只用于源码、入口脚本匹配与输出目录，不决定 Python 环境。

新增的便携 symlink 语义回归在修复前失败、修复后通过；sampler 与 driver 16 项 CPU 回归通过（2.721 s，`training_smoke_venv_fix_cpu.log`），diffcheck 通过。该修复随后纳入训练 source_r5；上文 r2 源码 SHA 一致性是当时检查的历史结果，不能代替 r5 指纹。没有更改 reward/runtime，也没有在本修复步骤启动 GPU 或重写失败 receipt。

## r5 Stair 归档验收的协议身份修复

`r5_train_stair_go2_on` 的子训练正常结束；driver 的归档校验将两种不同身份误设为相等。`--task-protocol-hash` 属于训练 binding 的合法祖先（`1dc2cca08e123f21d5fb6203ba8ab20f84a2daa051e732eae62e5d3e77667c00`）；归档 `task_protocol_hash` 与实际 resolved cell/source preflight 一致（`d85bebb001f72dd13c8107215ec5d81e391d795fd2541c8dc589ce4fbe9e35e9`）。runtime 记录正确，没有改写历史 manifest。

checker 现复用 `load_resolved_cell_manifest` 校验 argv 指定的 manifest hash/task/base，并据此核实际 protocol、完整 cell identity 和 source preflight；training binding 仍独立逐值核对。`verify_training_pair.py` 复用该 helper，没有第二处错误的 protocol 相等要求。真实当前 Go2 v3 配置回归验证两种身份可不同，错误 manifest hash 或错误 preflight 仍拒绝。

17 项 sampler/driver CPU 回归通过（2.248 s，`training_smoke_protocol_identity_fix_cpu.log`）。当前 checker 只在远端内存加载，对未修改的 r5 档案复算也通过：8 控制步、128 env-transitions、9 chunks，identity/coverage/raw-weighted-total 均一致，见 `training_smoke_protocol_identity_r5_readonly_recheck.log`。该只读补证不是重新训练，也不将原 driver FAIL receipt 改成 PASS；后续 r6 正常驱动验收仍由新快照与新 receipt 单独记录。
