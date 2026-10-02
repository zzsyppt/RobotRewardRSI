# 2026-10-03 审计用的探针脚本

配套文档：[../TRANSLATOR_PLAN_AUDIT_2026-10-03_ZH.md](../TRANSLATOR_PLAN_AUDIT_2026-10-03_ZH.md)。

全部只读：服务器上只用 CPU、不写任何文件；本地只读两份导出的 `.npz`。没有调用模型。

## 文件

| 文件 | 作用 |
|---|---|
| `server_stair_contacts.py` | 在服务器上用冻结源码解码封存的 R4 Stair 档案，把每一步的生产口径接触标记（足端合力 > 任务阈值）和足端三维力写到标准输出（NPZ） |
| `stair_behavior_probe.py` | 本地分析：A 成功回合的步态；B 超时与单足不承重；C 足端合力对体重；D 按状态变化分段的时间线；E 腾空时间与关节限位；F 超时的位置 × 形态表 |
| `stair_behavior_probe_stdout.txt`、`stair_behavior_probe_out.json` | 上面脚本的输出 |
| `familywise.py`、`familywise_stdout.txt` | 重算上一轮“成功参考范围”试验的整回合误报率 |

## 复现

需要两份导出文件放在同一个工作目录里（不入库，约 47 MB 和 4 MB）：

~~~bash
ssh humanoid-server 'cd /tmp && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /data/yuyangliu/envs/isaaclab6_validation/bin/python -B -' < ../../P3NEW/audit_20261002/generic_poc/server_stair_dump.py > stair_steps.npz
~~~

~~~bash
ssh humanoid-server 'cd /tmp && PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" /data/yuyangliu/envs/isaaclab6_validation/bin/python -B -' < server_stair_contacts.py > stair_contacts.npz
~~~

每次导出约 3 分钟。然后：

~~~bash
python stair_behavior_probe.py
~~~

`familywise.py` 还需要上一轮的 `run_generic_stair.py`、`generic_describe.py`（在 `P3NEW/audit_20261002/generic_poc/`）和 `stair_modes_out.json`（在 `P3NEW/audit_20261002/`）放在同一目录。

## 口径

- 接触：解码器自己的 `support_contacts`，即足端接触合力的模大于 1 N。足的顺序是 FL、FR、RL、RR。
- 体重基准：常规台阶（`c0_id`）成功回合里足端竖直合力的时间平均，150.9 N。档案没有记录每个环境的实际质量，这个基准约有一成不确定。
- 小腿关节伸直限位：取 Go2 URDF 的 −0.8378 rad。机器人清单里没有关节限位字段。
- 分段规则（D 部分）里的阈值是临时取的，只用来说明“全程覆盖可以由数值做到”，没有标定。
- 档案：封存的 R4 Stair 数值档案，清单哈希 `8c5f2d68db3df151c850beb6e2147ced671fb00acc802e23427f188c0d7e9d83`；策略是 `audit3_go2_full_train` 的 `best.pt`，用锚点奖励 `go2-stair-official-flat-derived-v1` 训练。
