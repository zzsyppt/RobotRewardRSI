# R2 正式 SOL 材料跨端准备

状态：**36 个数值 facts 已实际迁移并通过离线请求一致性验收；4 个 media facts 尚待准备。模型调用 0，GPU 操作 0。** 实际总回执见 `numeric_sol_relocation_batch_receipt.json`，状态为 `PREPARED_36_IDENTICAL_NO_MODEL_CALL`；最初的静态准备回执 `relocate_sol_facts.preparation.json` 原样保留。

36 项按真实 `packing_receipt.json` 原顺序执行（24 episode + 12 focus），服务器导出 ZIP 共 11,883,540 字节，一次 SCP 传输全部 74 个文件。传输后逐项核归档 SHA、源 manifest SHA、原 prepared 请求 SHA；Windows 冻结 prepare 的 36 个实际请求均与服务器对应 SOL 请求 SHA 一致，三个准备模块的 SHA 也一致。这仅证明这 36 项的离线输入等价，未执行评委，不构成上下文执行或语义资格证明。

总迁移回执 SHA：`e9caef5316f897122dafcac4546163c35ea52a34698292fc0d2866fdefc89ea1`。服务器 packing receipt SHA：`2377e6b249cf59f32b80f4f7070e2674c5903208c7994e503cfb7fde8498384b`；导出总回执 SHA：`f28262db8ed58baaea36137452941e65b210e5680952cd9f3be4e3b61b6d8f67`。

仅新增一个控制脚本 `relocate_sol_facts.py`，不修改生产代码、477 项冻结源、采集产物或旧结果。脚本处理一个真实 facts 单元：服务器打包 manifest 明确列出的原文件；Windows 只改变该 manifest 的 `materials[].file`，再调用冻结 `run_confirmation_review.py prepare`。不复制或手改 `prepare.json`。

`materials[].file` 不进入冻结 runner 生成的模型请求正文；材料的 ID、角色、标签、SHA、JSON 正文、原 PNG 和顺序均进入正文。迁移后同时要求真实请求 SHA 与服务器对应 **gpt6-sol** prepared 请求相同，runner、transport、coverage helper 的 SHA 相同。不能与 DeepSeek 请求直接比较 SHA，因为两者的评委身份文字不同。ZIP 只用于传输这个明确材料集合，不对整棵采集目录再做副本。

## 实际命令

以下列出实际使用接口。36 个数值单元已经执行，禁止对同一输出重跑；后续 media 材料齐备后使用其真实 packet ID。`<packet>` 必须取本轮 packing receipt 或 media pack 的真实 ID，不手选或重排单元。所有输出都要求新路径；失败输出保留，不能删除后自动重试。

控制脚本已由主线上传并核对 SHA；下列是其上传接口，当前不需要再次上传：

```powershell
$localControl = 'D:\humanoid_agent\true_RL_Agent\PLAN\feedback_sys\P3NEW\results\n5_confirmation_execution_r2'
scp "$localControl/relocate_sol_facts.py" humanoid-server:/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r2/relocate_sol_facts.py
```

服务器端，以实际 numeric packet 为例。4 个 media 单元的 manifest/prepared 根应改成实际 media pack 根，处理逻辑完全相同：

```bash
B=/data/yuyangliu/AGENT/feedback_p3new_20260929
C=$B/n5_execution_20260930_r2
PACK=$B/n5_confirmation_20260930_r2/numeric_materials
PY=/data/yuyangliu/envs/isaaclab6_validation/bin/python
PACKET='<packet>'
"$PY" -B "$C/relocate_sol_facts.py" export \
  --manifest "$PACK/packets/$PACKET/facts.manifest.json" \
  --prepared "$PACK/prepared/$PACKET/gpt6-sol" \
  --archive "$C/sol_transfer/$PACKET.zip"
```

`export` 当场核对 prepared 与真实 manifest、原请求正文及逐项材料 SHA，写出 ZIP 和 `<packet>.zip.receipt.json`。它不生成评阅结果。每条命令检查退出码后才执行下游。

Windows 本地接收、准备，使用本轮已经冻结的本地源：

```powershell
$repo = 'D:\humanoid_agent\true_RL_Agent'
$localControl = "$repo/PLAN/feedback_sys/P3NEW/results/n5_confirmation_execution_r2"
$frozenSource = "$repo/PLAN/feedback_sys/P3NEW/results/n5_confirmation_source_r2/source"
$packet = '<packet>'
New-Item -ItemType Directory -Path "$localControl/sol_transfer" -Force | Out-Null
scp "humanoid-server:/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r2/sol_transfer/$packet.zip" "$localControl/sol_transfer/$packet.zip"
scp "humanoid-server:/data/yuyangliu/AGENT/feedback_p3new_20260929/n5_execution_20260930_r2/sol_transfer/$packet.zip.receipt.json" "$localControl/sol_transfer/$packet.zip.receipt.json"
& "$repo/.venv/Scripts/python.exe" -B "$localControl/relocate_sol_facts.py" prepare `
  --archive "$localControl/sol_transfer/$packet.zip" `
  --source-root "$frozenSource" `
  --output "$localControl/sol_facts/$packet"
```

实际验收凭 `sol_facts/<packet>/relocation_receipt.json` 的 `PREPARED_IDENTICAL_NO_MODEL_CALL`、`request_sha256_equal=true`、`frozen_prepare_modules_equal=true`，并核下载 ZIP 的 SHA 等于服务器传输回执。原 manifest/card 及迁移表保存在同一目录，材料正文均逐字节保留；本地 manifest SHA 有意不同，不能拿服务器 SHA 冒充本地 SHA。此状态只证明离线输入等价，没有证明上下文承载或语义判断正确。

## 正式 SOL facts 调用与封存

必须先由主线确认：两任务全部原观察者输出的 root 模块审查已封存，40 个单元材料及上下文容量检查齐备。本文件不解除这些前置条件。

实际调用继续使用唯一冻结入口，不另写模型调用器。示例为**待执行命令**，不是当前批准运行记录：

```powershell
& "$repo/.venv/Scripts/python.exe" -B "$frozenSource/PLAN/feedback_sys/P3NEW/run_sol_review.py" `
  --prepared "$localControl/sol_facts/$packet/prepared" `
  --output "$localControl/sol_facts_execution/$packet" --execute-one-call
```

此入口每次为 fresh ephemeral SOL/high，不继承 root 或其他评委历史，不允许工具；真实事件、usage、非消息工具项、答案和覆盖检查仍逐项保存。不能预先生成虚构 facts seal。只有真实答案身份/阶段/complete 和覆盖验收满足既定口径后，seal 才绑定 **本地 `facts.manifest.json` 实际 SHA** 与该 SOL 原始答案 SHA。DeepSeek seal 仍绑定服务器自身事实 manifest。迁移回执负责证明两者材料一致，而不是强行让控制 manifest SHA 一致。

## Report 使用现成函数，不修补服务器 report

必须等两位评委的 **全部 40 个 facts** 封存屏障成立，才能准备或发送正式 report 阶段。另将该单元允许的 report 材料原件下载并核 SHA；清单中的文件定位可以在本地控制清单里改变，正文、SHA、ID、角色、顺序和评分口径不变。

直接复用冻结模块中的确切函数：

```python
import json
from pathlib import Path
from prepare_confirmation_materials import prepare_report_manifest
from run_confirmation_review import prepare

# 四个变量均由本轮真实本地材料和本人 seal 取得；不得指向其他评委答案。
facts_manifest = Path(actual_local_facts_manifest)
additional_materials = json.loads(Path(actual_local_report_material_list).read_bytes())
report_manifest = Path(new_local_report_manifest)
own_facts_seal = Path(actual_gpt6_sol_facts_seal)
prepare_report_manifest(facts_manifest, additional_materials, report_manifest)
prepare(report_manifest, 'gpt6-sol', Path(new_local_report_prepared_directory), own_facts_seal)
```

调用时 `PYTHONPATH` 仅设置为本轮 frozen source 的 `src`、根和 `PLAN/feedback_sys/P3NEW`。`prepare_report_manifest` 会保持 facts 材料顺序、追加明确的 report 材料，并自动把 `facts_manifest_sha256` 设为真正的本地 facts SHA。因此不需要迁移器修改服务器 report 的绑定字段，也不需要扩大冻结源码。

Report 的离线输入也应与服务器为 **同一 SOL、同一原始本人 facts 答案** 构造的参考请求核 SHA；其中双方控制 seal 必须如实登记各自 manifest 路径/哈希的迁移关系。不得与 DeepSeek 的本人 facts 请求比较，也不得为了追求 SHA 一致复制另一位评委答案。当前 report 参考请求、seals 和原答均不存在，本准备不声称已验证 report 跨端等价。
