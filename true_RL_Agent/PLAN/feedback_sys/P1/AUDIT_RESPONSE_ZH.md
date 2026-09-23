# P1审计处置

2026-09-23。对应 [独立审计](./Audit_P1_2026-09-23_ZH.md)。以下按发现逐条处置；源码核对、已有审计的运行结果和本轮新增验证分开。本文不代替验收报告。

## 3.1 记忆保留退化

采纳并已在记忆表示源头收敛。每lane第10列改为奖励执行状态，generation删除顶层重复项；普通round同样只保留短状态。完整计数的生产者、原训练回执不变，当前被选择展示的runtime反馈仍按原选择规则提供详细计数；不能从长期短状态反推调用次数或因果效果。

状态为 `not_available`、`not_executed`、`no_reward_terms`、`terms_not_called`、`executed_zero`、`executed_nonzero`、`executed_value_unknown`、`unknown`。这不是新任务词表，而是已有通用计量的简短表示；零/非零沿用producer的奖励贡献计量口径，不表示每个term分别为零或有因果效果。没有照抄审计monkeypatch的全部分类：单独false、仅诊断或加载记录都不能推出奖励已执行，保留unknown；有正调用数但没有贡献观测时才是executed_value_unknown。旧9列和旧10列字典重压缩后前9列不变、输出幂等。没有增加持久schema、缓存或迁移服务。

在新真实回放前于[实施卡](./IMPLEMENTATION_PLAN_ZH.md)登记门槛：Stair第13代前至少11代、第16代前至少13代；Push第6代前保留5代；仍为16轮/14 KiB。新增正常测试fixture只存3个冻结公开记忆及必要runtime profile，41个源hash与原件匹配、保留字段与原公开payload一致，测试不读取PLAN。

服务器沿原历史回执路径，在两个隔离源码副本上使用原审计驱动、`--memory-variant full`，没有应用试验monkeypatch。下表的新基线与修订后结果均来自本轮完整回放，修订前P1列引用原审计；各格为“保留代数 / 省略代数 / 字节”。

| 检查点 | 原基线fe96455d | 修订前P1 | 修订后P1 |
|---|---|---|---|
| Stair第13代前 | 12 / 0 / 14,080 | 6 / 6 / 13,927 | **12 / 0 / 13,814** |
| Stair第16代前 | 14 / 1 / 13,912 | 7 / 8 / 13,732 | **13 / 2 / 14,111** |
| Push第6代前 | 5 / 0 / 6,087 | 5 / 0 / 8,563 | **5 / 0 / 6,525** |

全部达到事前门槛，与独立的正常fixture压缩结果相同。Stair仍比原基线少保留1代，属于已登记的取舍；没有宣称完全无容量成本。33/33与15/15个phase摘要重建相等，三个完整作者payload各自重复构建相等；两个任务的摘要篡改都拒绝，current-best与基线不变。再对已生成payload核对60行：前9列不变、末代及最近接受代保留、保留加省略等于实际历史代数、第10列均为短状态且无顶层副本。[真实报告与比较](./retention_regression/run_summary.json)记录完整命令、payload hash和检查范围；没有为这一步重跑训练、模拟器或模型。

## 3.2 摘要演化与旧证据

采纳设计约束，纠正两处表述。

1. [generation_record.py 的比较](../../../src/true_rl_agent/agent_core/generation_record.py:618) 是 Python 字典键值语义比较，不是逐字节。JSON空白、键顺序不触发；只重构实现、输出仍相等也不触发。新增旧回执重建结果的字段、改变其值或类型而导致不等才会拒绝。不能笼统写成“任何数值格式变化都会使全部历史失效”。
2. 独立工件不是回填旧回执的通道。[EvaluationPhaseResult](../../../src/true_rl_agent/evaluation/phases.py:149) 对 `task_evidence` 求hash，序列化完整结果；加载时再次校验。journal同时绑定完整phase result hash，[generation加载](../../../src/true_rl_agent/agent_core/generation_record.py:1158) 再核对这些hash及身份。后来采集的字段或工件引用不能塞入旧 `task_evidence` 而保持原身份。

已将边界加入 [P2范围卡](../P0/RESOURCE_SCOPE_AND_MIN_LOOP_CARD_ZH.md)：旧源的派生视图只能使用已保存事实并标注来源和生成时点；新capture是新运行、产生自己的回执，发布前封存并绑定新过程工件。两者都不能冒充历史原回合过程或获得新接受权限。先维持历史摘要语义，不提前实现版本管理、迁移服务或回执回填；确需演化持久合同后再按真实需求设计。

## 3.3 小项

| 条目 | 判断与处置 |
|---|---|
| Stair同一事实多个字段名 | 部分采纳可读性问题，当前不是两份独立真值：别名由同一dispatcher状态导出，服务两套已有wire合同。旧名仍由Stair gate、同transition覆盖、Isaac执行/smoke及已hash的精确键集消费；历史回执也保留旧名。按本轮Stair专项复核结论，本轮保留既有出口，不删字段、不建新兼容分支；省少量键的收益不足以重做历史协议。不写成已完成命名收敛。 |
| `available`易被误读 | 采纳语义澄清。[measurement_summary](../../../src/true_rl_agent/candidates/mechanism.py:76) 在任何一个计数/标志已知，或有stats/diagnostic/activation记录时均可返回available；不证明执行、全字段已测或非零。null计数仍未知。本轮在简短记忆类别中保留未知，P2范围卡已登记叙述约束；不因有声明/激活列表就推出执行次数。 |
| 历史绝对路径绑定 | 采纳并收窄“导出即可本地重放”的暗示。两份本地原件副本的首条ledger均含 `/data/yuyangliu/AGENT/true_RL_Agent/...` phase引用；[TrialReceiptRef.to_dict](../../../src/true_rl_agent/agent_core/trial_receipts.py:81) 含path，[guard比较](../../../src/true_rl_agent/agent_core/generation.py:512) 对整个引用求hash，[加载](../../../src/true_rl_agent/agent_core/trial_receipts.py:739) 比较完整ref。取回原件不自动等于生产原路径重放；需要该保证时在原路径只读重放。本轮不修改路径合同或实现迁移框架。 |
| 测试环境 | 采纳运行说明。`test_runtime_evidence_projection.py` 直接导入src包，没有自行插入路径；未安装项目且未设路径的独立执行需 `PYTHONPATH=src`，不特属于Linux。已安装项目的venv或其他测试先改过sys.path可能不暴露此条件。复跑显式设置环境，不为这一点批量改测试bootstrap。 |
| 第三任务测试分工 | 结论成立，名称需更正。generation的任务中立phase投影由 [_PhasedAdapter.project_public_phase_summary](../../../tests/integration/test_fixed_width_generation.py:607) 承担，只读fixture的score；搭配FakeTaskPlugin（历史测试ID为 `fake_push_recovery`），不是生产Push/Stair投影。`_Assembly`只是存selected trial和phase refs的dataclass，不是adapter。[fixture_orbit_tracking](../../../tests/unit/test_feedback_evidence_boundary.py:33) 则用独立任务ID、无schema通用round fixture，验证自有指标→任务记忆→最终请求JSON及投影隔离。两处职责互补，不声称orbit测试证明generation绑定或round-v7任意字段反篡改。 |

## 本轮边界

本轮修订版本267项相关CPU测试通过，无跳过，命令见[开发记录](./DEVLOG_ZH.md)，分批日志及范围见[验收](./ACCEPTANCE_ZH.md)，汇总与源码指纹见[检查记录](./audit_fix_checks.json)。服务器回放只读历史输入，所有新输出写入scratch；两会话共8个身份文件前后hash一致，10个同步生产源文件与本地最终内容一致。身份文件hash检查不冒充全目录文件审计。未调用provider、运行训练或模拟器，未改写历史receipt。原审计、原验收日志及首次检查记录保留。

同类复查覆盖普通round、generation、旧9列/旧10列的重复压缩、只有部分运行记录时的语义以及当前详细反馈；没有再发现需要增设公共抽象或兼容分支的问题。P2范围卡原有P0补修信息、manifest差异和合法祖先准入前置均保留。
