# P3-R 验收记录

日期：2026-09-29。状态：**按用户要求完成必要收尾后暂停，尚未通过原 P3-R 整体门**。用户决定不再追加“过程反馈是否有用”的作者价值实验，转向明确语义、未知情况和多模态能力的建设；主计划待用户修改。旧数值开发门存在评阅分歧，其通过声明撤回。详见[审计收尾](audit_response_20260929/AUDIT_RESPONSE_ZH.md)及[当前停止状态](audit_response_20260929/closeout_status.json)。

## 已完成且有证据的部分

| 项目 | 当前结论 | 证据与限制 |
|---|---|---|
| R.0-a 运行前提 | 第二轮 high / 65536 独立8/8完整响应，8/8候选静态通过 | [两轮索引](results/runtime_two_round_index.json)。首轮high / 32768仅6/8，历史失败保留。第二轮输出均低于32768，不能证明扩大上限导致改善；也不证明总体可靠率或语义正确 |
| 修复帧证据保留 | 原P3中8个丢失反馈的末次请求已重建，反馈、焦点与错误/源码保留 | [验收](PROMPT_EVIDENCE_RETENTION_ZH.md)。通用层管理必留及容量；任务提供已有证据内容，不引入任务专用语义 |
| CPU / GPU 生命周期 | 子进程退出、释放实际GPU lease后才执行Stair CPU派生，派生封存后才发布phase | [验收](EVAL_CPU_BOUNDARY_VALIDATION_ZH.md)。真实轻量子进程/物理文件锁接线及数值档案到作者输入验证通过；不把GPU计时减少冒充总时长加速 |
| Stair默认视图 | 真实864回合保留完整分母、核心匹配对照及已确认根部过程，8118字节 | [验收](STAIR_PROJECTION_V2_ZH.md)、[真实回归](results/stair_projection_v2_real_validation.json)。选例/裁剪修正、词义收窄，不改变原检测阈值及成功规则；不是独立语义门 |
| R.0-b 整理证据 | 两Stair案例完成45项交叉核对，Push192回合数值/稳定/终局与档案一致；中强度录像身份/端点27项通过 | [核对](CURATED_EVIDENCE_REVIEW_ZH.md)、[录像检查](results/medium_video_checks.json)。回放未证明逐步同轨迹，视觉文字与测量分开；不向作者发送图片 |
| R.0-b 输入准入 | 16个实际首包全部通过，最大task sections 51672/57344字节，最小余量5672字节 | [封存卡](results/ceiling_release_r1/ceiling_preflight_r1/stage.json)。完整feedback未被裁剪；共同聚合、父本、权限、提示保持相同，仅变化过程证据 |
| R.0-b 实际投递与运行 | 32/32候选终态，34/34实际请求核对通过，未知用量0 | [执行结算](results/ceiling_execution_closure.json)、[请求审计](results/ceiling_final_evidence_r1/actual_request_audit.json)。32首答中31份满足响应合同、30份静态通过；一次源码验证拒绝、一次JSON格式错误经修复通过。两次修复均保留完整原反馈及焦点。静态通过不是语义或价值通过 |
| R.1/R.2 Stair 当前工程 | v2 过程派生 / public v3 默认视图，864 开发回合独立重算、最终相关 42 测试通过 | [工程记录](STAIR_MOTION_V2_ZH.md)。默认 8073/8192 字节；开发回归不是独立语义通过 |
| R.1/R.2 Push 当前工程 | v2 事件过程、驻留与再失稳、相容对照，192 开发回合核验 | [工程记录](PUSH_R1_R2_IMPLEMENTATION_ZH.md)。默认 11713 字节、带父本差分 12274/12288 字节；22 项任务测试通过，未完成新资料确认 |

R.0-b 执行源码快照为 `source_r0b_r2_snapshot`，execution hash 为 `98d42b8b406dce1147930dead81e4ecede521de1abb4ccd6250a71a4db0400d2`。R.1/R.2 后续源码另封存于 `source_confirmation_r1_snapshot`，execution hash 为 `8fa8877b99c335ffcc76e884e30e26fb88a4d930b02f932d3824e78e1d9c9080`，上传准备不等于已采集。旧快照保持原件；实际系统代码不依赖本目录。测试详情采用各项原记录，不将平台跳过计成执行通过。

## 已封存与尚未执行

R.0-b按[冻结评阅规则](R0B_REVIEW_RULES_ZH.md)执行4案例×4变体×2次，共32个机会，生成、四组模型语义评阅及解盲均已完成。远端目录为 `/data/yuyangliu/AGENT/feedback_p3r_20260929/ceiling_run_r1`，封存卡SHA-256为 `61df1e5e1efe5b225e328213f31c296dbbc0ae74ea41846fa0e6c9e14afd23fd`。输入封存后不因输出调整。每次最多5个调用、2次源码请求；未启用更强作者或额外8机会。采用用户明确指定的模型专家评阅，不能称人类评阅。

R.0-b 原评阅曾判 Tc-num 两个案例、Tc-vis 三个案例有效；原分保留。审计确认 001 的“驻留姿态带保持完整阻尼”与实现矛盾，足以撤回原数值通过声明；022 及部分过程预测判定存在分歧，不能把局部复核当全体新验收。R.1/R.2 已有上述工程交付，R.3 新资料语义确认及 R.4 作者价值实验未执行，候选训练及 P4 未启动。用户已要求停止追加价值证明，本轮不再使用旧卡启动任何采集/调用。

## 费用与资源

R.0-a两轮峰价计量合计 **US$0.437974032**；R.0-b完整34次调用为 **US$0.993046416**，R.0累计50次实际调用、**US$1.431020448**，全部用量已知。这里是回执按登记峰价计算的金额，不冒充服务商发票。新媒体两次启动总GPU lease约 **0.039624小时**。原始回执及按臂输入/输出/修复费用见[执行结算](results/ceiling_execution_closure.json)。

[资源修订卡](stage_budget_r2.json)记录 R.0-b 执行时的分配；后续 `stage_budget_r3.json` 曾为条件性 R.4 预留 US$24，但没有产生运行放行。阶段上限仍为 US$30 / 4 GPU小时。审计收尾新增实验 provider 调用 0、仿真 0，剩余额度不自动转为新实验。桌面助手/模型评阅服务成本没有本实验 provider 回执，不混称上述费用包含全部工作成本。
