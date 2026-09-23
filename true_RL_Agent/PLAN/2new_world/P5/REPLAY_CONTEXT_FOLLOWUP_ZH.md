# P5 历史回放与上下文复审

状态：**GO，2026-09-11 本轮复审已闭合**。结论来自本轮真实历史回放、上下文核对及两端各一次完整 1611 项回归；此前 1610 项记录不代替本轮证据。总目标、任务 × 机器人矩阵、冻结的科学协议和原实验 checkpoint 保护范围不变。

## 本轮实施顺序与边界

1. 复现服务器 P4 冻结 fixture 的正式只读 CLI 回放失败。当前已复现 `generation session contract already exists with different bytes`。共享层提供冻结契约读取，仅允许只读回放沿用历史 authoring policy；其余身份仍须一致，后续 Genesis、账本、成本、终态和报告完整重算。实际续跑仍要求完整契约一致。删除旧 reader 中已无执行消费者的 P6-B 特判，避免维护两套兼容规则。
2. 当前最优和最新已测反馈保留详细诊断；较早反馈保留成绩、family/guard 结果、主要失败。表头和评测身份在当前帧声明一次。保留真实内存字节计量和原容量上限，不靠改大/改小常量掩盖内容取舍。
3. 未进入 selector 的已完成 public-dev 同样生成相对当代 parent 的显示分差。简短公开失败原因随 trial 保留，区分 provider、训练与评测失败。所有 Stair 分数和原因投影留在任务层；共享记忆只承接任务提供的结果。
4. 定向回归覆盖读取与执行边界、无 selection 的已测 dev、私有错误排除、共享表头与非零列对齐。实际重建 c6/c8 历史提示，核对 G3-A、晚代 trial 留存和压缩量；实际回放服务器冻结 P4 fixture，并验证所有文件的字节、mtime、SHA256 不变。最后运行两端完整回归、检查部署一致性并更新本文件。

不新增 RL 训练、provider 请求或新一轮题池标定。P6/P7 不在本轮范围内。

## 已完成的修复与复验

- P4 原始 15 代 fixture 已经由生产 `audit_stair_generation_history.py` CLI 原样回放；251 文件的 bytes、mtime_ns、SHA256 均不变，报告 SHA256 为 `cec022d738bae577a715a1bfa41b839f50514f5f8b5255e65f3a6dd04437941a`。环境/checkout 替身与 P4 原验收一致，不是物理训练或新策略证明。
- 原来的两个 P6-B 私有函数测试随已无执行消费者的分支退役。其有效约束迁入共享契约测试：只读沿用 frozen authoring、拒绝 provider/selector/session 身份漂移、实际续跑严格、原文件不变。共享 Stair/Push 集成测试重新进入 production preparation 再做终态 replay，避免复用旧 foundation 对象掩盖差异。
- 真实 G3-A 暴露了第二个上游缺口：无 selection 时 parent dev 同样未被投影。现在共享 replay 校验 manifest 中绑定的 parent phase 回执和 checkpoint，调用任务提供的公开摘要投影；共享层不假设 `task_evidence` 的内部结构。Stair 用现有 canonical phase reader，结果只进入显示分差，不修改 selector 或 acceptance。当前真实分差为目标 `+0.000336`、guard `+0.000504`；B lane 仍为未测并保留 `candidate_runtime_failed`。
- 当前摘要 schema 为 v13；较早反馈保留 objective、family、named guard、主要失败。详细表只给 current-best 和 latest-evaluated，非零失败/诊断列合并后对齐，反馈内三个身份 hash 统一声明。最小容量路径同时保留表头与身份，不产生无列名数组。旧封存 prompt 不改写。
- 第一轮完整回归片段因新增父代投影取错字段、旧 8 列断言需要迁移而停止；`replay_*_partial_tests.*` 明确不作为验收。修复后固定宽度/CLI/任务投影定向验证通过，随后固定代码，在两端各完成一次完整 discovery。

本轮沿用 Codex 的有限借鉴：持久化证据与模型可见窗口分开，采用确定性压缩。没有引入摘要模型、额外记忆数据库或新的执行生命周期。

## 真实历史的内容取舍

`replay_context_results.json` 逐帧核对成功，不调用 provider 或训练。下表为紧凑 UTF-8 字节，不用 bytes/4 冒充 token 测量；“修复前”为上一版 P5 同一重建脚本的产物。

| 历史帧 | 修复前整帧 B | 当前整帧 B | 当前反馈 B | 当前记忆 B | 保留 trial |
|---|---:|---:|---:|---:|---:|
| c6 G1 | 58,959 | 59,330 | 7,153 | 996 | 0 |
| c6 G4 | — | 93,277 | 17,441 | 6,639 | 6 |
| c6 G7 | 112,455 | 98,667 | 18,598 | 8,843 | 12 |
| c6 G9 | 105,707 | 99,721 | 16,367 | 13,782 | 16 |
| c6 G13 | 114,943 | 102,380 | 18,483 | 14,197 | 24 |
| c6 G15 | 114,583 | 102,136 | 18,483 | 13,920 | 28 |
| c8 G9 | 112,363 | 99,764 | 18,642 | 12,922 | 16 |

G13 反馈从 36,389 B 降至 18,483 B，约减少 49.2%；整帧约减少 10.9%。两者降幅不同，是因为容量充足后原本被裁剪的 ABI 字段说明等静态内容约 5.1 KB 得以保留。初代新增共享定义有少量固定开销，不为追求每帧都更短而删掉必要描述。

两场历史的跨代索引没有遗漏。较早反馈仍有 family、guard 与主要失败，完整诊断表只给当前最优和最新已测项。G13 current-best 为 G8-A，latest 为 G12-B；源码与 runtime 的选择身份未改变。任务段 56 KiB、记忆 14 KiB 的上限保持原值。没有新增候选质量实验，不据此宣称压缩能提高策略收益。

原始 G13 repair 分支为 54,384 B，只包含当前失败候选、源码与修复反馈；不带历史诊断表是既有的修复帧边界。受控 source-followup 分支为 89,447 B，保留历史反馈及共享表头；它不是原实验中发生过的 source request。强制最小容量路径另有单元断言，保证表头和数组同时保留。

## 同类问题自审

- 沿只读 CLI 和共享宿主两条入口检查：二者共用冻结契约 reader；执行与续跑仍走严格 publisher。已退役的 P6-B 特判及测试没有残留调用。读取返回的原始 ref 继续参与 Genesis、账本、成本与终态重算。
- 沿 generation → public history → task adapter → memory 检查：parent 回执只在已有 dev、没有选择参考时补投影，核对 manifest 与 parent checkpoint；分数算术和失败原因的公开范围仍由 Stair 决定。没有把显示分差接入科学接受门。
- 沿详细/精简/最小容量与 source-followup 检查：共享表头和行一起保留，非零列取并集后重新排齐；旧八列记忆在读取时补空 reason，新写入九列。原始 prompt 和科学回执不改写。

`replay_protected_after.json` 在最终回归后再次核对受保护完整 c6 实验：932 文件、517,212,889 B、71 个 checkpoint 与 P0 清单相同。c8 Genesis/ledger 校验值也未变；本轮测试进程无遗留，四个已有 GPU lease 均未上锁，元数据未重写，见 `replay_resource_closeout.json`。

## 最终完整回归与交付

| 环境 | 一次完整 discovery | 结果 |
|---|---:|---|
| 本地 Python | 1611 项，约 1291 s | 0 failure/error，19 项平台/Isaac 安装条件跳过；`replay_local_final_tests.json`。 |
| 目标 Python | 1611 项，约 2737 s | 0 failure/error，仅两项显式 opt-in 跳过；`replay_target_final_tests.json`。 |

本地缺少目标 Isaac 安装而跳过的四组 Stair 会话用例，在目标完整运行中全部通过，覆盖接受后恢复、拒绝/未选择、reference 全程及终态回放、运行/基础设施失败。两项显式 opt-in 物理接口检查本轮未重跑；本轮未改动该物理路径，前一批专项通过记录保留其原范围。所有结论来自各自完整运行，不拼接中止片段或定向重试。

命令分别为 `.venv/Scripts/python.exe -X utf8 PLAN/2new_world/P5/run_full_tests.py replay_local` 与目标 Python 执行同一脚本的 `replay_target` 标签，底层均为标准 `unittest discover -v -s tests -t .`。

本轮冻结源码/测试清单为 `replay_source_files.json`：515 文件，SHA256 `cb571c766e5257561a429395b5a0748482d33d3a28599a55ee876d0f0671b9ce`。测试开始后仅整理验收文档，产品代码和测试未再改动。数量从前一批 1610 变为 1611，是新增三个回归并退役两个旧私有兼容分支测试；有效约束已迁入共享测试。

两端逐文件校验无差异，活动 Python 文件没有清单漏项，见 `replay_source_sync.json`。最终文件清单和两端回归回执由 `replay_deployment_receipt.json` 绑定。本轮恢复 P5 工程 GO；不宣称候选质量提升，P6/P7 保持未完成。
