# Push reference 准入修订

2026-09-22，外力统一实施验收之后。用户要求reference具备基础能力和改进空间，不再以20%等固定成功率区间判断。本轮已修改当前Push force新reference的默认准入；没有改动训练分布、施力、episode成功条件、原始结果或既有Stair评分。

## 当前代码规则

新默认策略为 `observed_outcomes_v1`，由[任务配置](../../../src/true_rl_agent/tasks/push_recovery/config.py)、[reference合同与判定](../../../src/true_rl_agent/tasks/push_recovery/reference_program.py)及[phased adapter](../../../src/true_rl_agent/tasks/push_recovery/phased_adapter.py)接入。

| 相位 | reference能力要求 |
|---|---|
| public_dev | 完整有效的恢复证据中，至少一次成功和一次失败 |
| public_accept | 观察到成功恢复，仍通过完整证据和预算等要求 |
| sealed_validation | 观察到成功恢复，保留accept→sealed最大0.1下降门 |

相对改进与统计判据保持。自动dev门仅排除全失败或全成功，不给“接近饱和”换一个百分比：1/192或191/192满足正反例条件，本身不证明余量充分。基础能力、困难层与饱和情况须结合分层计数、有效恢复和失败事实解释，不增加一套人工审批流程。

## 已有测量的复核

[复核记录](reference_admission_revision_01.json)使用原public-dev测量，没有新跑仿真或消费holdout。这是收到新要求后的规则修订与已有证据复核，不是预注册重测，也不是补发完整资格：

| 机器人 | 原始成功数 | 实施当时旧区间门 | 新dev准入 |
|---|---:|---|---|
| G1 | 33/192 | FAIL | PASS |
| Go2 | 141/192 | PASS | PASS |

G1有真实恢复正例，但high和repeated均为0/48；Go2的high为19/48、repeated为27/48。应保留这些能力差异，不用总分或自动PASS代替分层说明。两台新评价域的完整三相资格仍待验证，复核dev门不等于签发reference资格或科学GO。

新准入拥有新的qualification contract与manifest身份；训练、评价动力学保持相同。旧manifest仍沿用其冻结准入，原JSON、原FAIL/PASS及旧hash不改写，也不将旧测量重标为新的完整资格。

## Stair与验证边界

Stair的20–80%属于既有P3题池learnable-cell、family mask及评分合同，本轮不修改其题池、mask、分数或资格结果。基础能力与改进空间是后续新题池设计目标，不据此取消既有冻结评分条件。

## 验证

本次准入修订在两端执行相同9个模块，均为100项、0失败、0错误、0跳过、exit0：

- [本地测试](reference_admission_local_tests.json)：42.832秒，附[原日志](reference_admission_local_tests.log)；Ruff F/E9与差异空白检查通过。
- [目标测试](reference_admission_target_tests.json)：50.572秒，进程墙时51.601秒，附[原日志](reference_admission_target_tests.log)。
- 6个产品文件、4个测试文件及复核JSON共11个文件两端SHA核对通过，见[同步清单](reference_admission_source_sync.json)。

这些验证覆盖本次准入修订，不替代完整三相物理资格；外力统一实施前完成的1696项回归仍只解释其原源码范围。
