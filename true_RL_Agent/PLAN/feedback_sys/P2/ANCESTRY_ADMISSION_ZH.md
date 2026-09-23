# P2：Stair evaluation-only 祖先 checkpoint 准入

日期：2026-09-23。范围仅限协议/manifest 祖先校验与 runner 启动前准入；不改变训练、评分、接受门或历史回执。

## 本次解决的缺口

Go2 v3 的直接 predecessor 是 v2；真实 parent `fca6a59c…` 来自 v1。此前 `SubprocessRunner._validate_checkpoint_before_launch` 只接纳当前 manifest 或 metadata 中的单个直接前驱，因而无法让 v1 沿已经声明的 `v3 → v2 → v1` 链进入 v3 评估。

本次保留现有协议、manifest、TrainingBinding 的身份，新增 Stair 自有的准入函数。composition 把该函数交给 runner；runner 仍先核真实 checkpoint 文件、SHA、sealed/role、argv 与当前 manifest，然后调用任务准入。共享 runner 不解释 Stair 的训练继承语义，也没有增加一个可由工件填写的祖先 hash 列表。

## 每条 manifest 从哪里来

`StairUnifiedProtocol.load` 原本已递归读取 row catalog 指向的 source protocol，检查环、source base protocol hash、训练 profile、实际机器人绑定以及训练/分布/运行控制/reward/authoring 投影不变。它只保留训练 origin，runner 原来没有消费完整链。

新 `build_stair_checkpoint_ancestry_admission` 沿同一 `training_lineage`，使用现有 `StairClimbingTaskConfig.defaults`、`StairClimbingTaskPlugin`、`resolve_stair_cell_manifest`，从仓库配置和 registry **重新编译**每个节点。每条边声明的 source manifest hash、source resolved task protocol hash 必须与编译结果一致；不是把声明字符串直接加入可接受集合。每个节点绑定它自己的 resolved protocol 后，完整 `formal_training_binding().to_private_dict()` 必须一致，机器人 profile、runtime adapter、角色/能力及动作、观测、传感器、policy I/O、reward input 合同也逐项核对。

本机当前源码实际编译结果：

| 节点 | manifest hash | resolved task protocol hash |
|---|---|---|
| Go2 v3 | `7e8fa0c45109bbfc0162a90cf68d4186a0a3f96a0df9cce1e9b1b09f50aa6de9` | `d85bebb001f72dd13c8107215ec5d81e391d795fd2541c8dc589ce4fbe9e35e9` |
| Go2 v2 | `193fa9038a6f46539fb71a7e73edc44bdc024be2b899785394b8f82113c91f37` | `fb0a30071b27af5b600a7dba65dd03d8a1184ab195c9fd987f770e5eeb667429` |
| Go2 v1 | `54cd180c9f39f65bffd71482ec8a87cda912d0e378e8b434d0c44c839a3ab1c2` | `1dc2cca08e123f21d5fb6203ba8ab20f84a2daa051e732eae62e5d3e77667c00` |

三者完整训练绑定均为 `c0112a87a7fb5750ef11d55a6bb50cfdeb0be662380c1387b2d8cab8c977de98`。这里必须先绑定 resolved protocol：直接用未绑定的 v1 defaults 计算训练 hash 会得到另一个值，不能拿它代替历史生产绑定。

身份来源为 [v3 protocol](../../../configs/tasks/stair_go2_12dof_unified_protocol_v3.json)、[v2 protocol](../../../configs/tasks/stair_go2_12dof_unified_protocol_v2.json)、[v1 protocol](../../../configs/tasks/stair_go2_12dof_unified_protocol_v1.json)、[row catalog](../../../src/true_rl_agent/tasks/stair_climbing/row_catalog.py)，以及由 row 指向的冻结 binding、profile、runtime 和环境描述。v1 的冻结协议/binding 身份沿已有检查生效。上述 v1/v2 编译值也与 P0 取回的 [parent manifest](../P0/inventory_remote_raw/outputs/p5_go2_derived_reference_ef6826e/resolved_cell_manifest.json)、[formal manifest](../P0/inventory_remote_raw/outputs/formal_go2_stair_15round_20260901_c6ab8ba/resolved_cell_manifest.json) 一致。

## checkpoint 和原训练回执的绑定

准入不改 checkpoint 的历史 cell/manifest。跨版本评估要求：

1. checkpoint 的 `resolved_cell_manifest_hash` 是上述已验证链上的一个实际节点；如果带 `resolved_cell_identity`，它须精确对应该源节点。
2. checkpoint 的 `training_binding_hash` 存在，并与整条链验证出的训练绑定一致；评估 argv 的 `--training-binding-hash` 也须一致。
3. checkpoint 文件的 bytes/SHA/路径/role/sealed 继续由现有 runner preflight 校验；任务准入只允许 eval，不为 v1 跨到 v3 的训练初始化授权。

来源分工保持既有边界：runner 正常训练产物已把原 train spec 的 `training_binding_hash` 和训练 summary 的 manifest/identity 放入 sealed checkpoint metadata。对 P2 的历史 replay，入口须从**已验证的原训练回执**保留这些字段，并先核回执 status、完成预算、checkpoint checksum 与真实权重；不得从当前 v3 config 补一个训练 hash。任务准入并不独立读取或认证任意 train-summary 文件，也不把调用方提供的 metadata 当作新的训练完成证明。

本次重读的 [fca 原训练 summary](../P0/inventory_remote_raw/outputs/p5_go2_derived_reference_ef6826e/runner/p5-go2-derived-reference-ef6826e/train/summary.json) 实际含 `status=pass`、完整预算 `196608000`、上述 `c011…`、`54cd…`、v1 source identity，以及 checkpoint checksum `sha256:fca6a59c8142d0120c5b534addab03885995dd64931fc3b974b9f94b997fe729`。其原件路径与文件 SHA、历史远端 checkpoint 检查见 [P0 parent 核验](../P0/inventory_parent_remote_verification.json)。这是已有历史证据；本次没有重新执行远端权重检查或 GPU 评估。

旧 P6 replay 入口已检查训练预算、真实权重 SHA 和训练绑定，但构造 CheckpointRef 时只带 source manifest。复用于新准入时，需从 `trained["training_binding_hash"]` 保留到 metadata，并可保留 `trained["resolved_cell_identity"]`；否则跨版本评估会明确拒绝。历史回执和旧运行结果不回填。当前同一 manifest 的普通 job 保持原有训练 metadata 可选规则；若提供该字段，仍检查一致性。

## 实现与实际验证

改动：

- [checkpoint_ancestry.py](../../../src/true_rl_agent/tasks/stair_climbing/checkpoint_ancestry.py)：Stair 自有链编译与 checkpoint 准入。
- [composition.py](../../../src/true_rl_agent/tasks/stair_climbing/composition.py)：对 unified Stair 安装准入函数。
- [subprocess_runner.py](../../../src/true_rl_agent/runner/subprocess_runner.py)：中立 `eval_checkpoint_admission_hook`，在现有文件/argv/manifest 核验后调用；没有此 hook 的任务保留既有行为。
- [test_stair_checkpoint_ancestry.py](../../../tests/unit/test_stair_checkpoint_ancestry.py)：10 项新增 CPU 测试。

2026-09-23 使用仓库 `.venv\Scripts\python.exe` 运行下列 6 个 unittest 模块：`test_stair_checkpoint_ancestry`、`test_subprocess_runner`、`test_robot_matrix_contract`、`test_stair_phase5_contract`、`test_stair_study_holdout`、`test_stair_phase_b_composition`。原始命令和完整输出：[verification_ancestry.log](./verification_ancestry.log)。

结果：**129 项，34.822 秒，128 通过、1 跳过、0 失败**。跳过的是 Windows 无法创建文件 symlink 时的既有 symlink 输出测试。新测试覆盖真实配置链重编译、current/v2/v1 准入、训练绑定缺失/错配、source identity 与文件 bytes 错配、metadata 不能扩充祖先范围、跨机器人拒绝、伪造 resolved source 身份拒绝、仅 eval 范围，以及 composition 安装。既有回归也覆盖 rehash 后偷偷改变训练配置时拒绝 successor。

测试中的 checkpoint bytes 是明确标注的合成小文件，只验证启动前合同，不是实际策略或新实验。当前完成的是合法祖先准入代码及 CPU 回归；fca/Gen8-A 两个真实 checkpoint 的当前 v3 source/environment/完整 public-dev capture 仍由 P2 执行阶段产生自己的新回执。
