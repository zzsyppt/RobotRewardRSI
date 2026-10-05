# R1c／R2 CPU准备：同session复用R4训练、当前Q与两代

本阶段使用同一个session：导入R4已完成训练，执行当前Q三阶段，再进行两代。执行源码冻结在 `429e6027155d79bf391394b275f98c4fd500a19f`（同session实现 `cedcdb2d`，再修复原argv的采样标志恢复）。这里只执行CPU验证与prepare-only，没有真实训练、GPU评测、功能作者或模型API。没有生成真实reference Genesis；G1仍用正常闭合后暂停，G2须另行放行。

## R4真实身份恢复

正式入口新增 `--completed-reference-training-source`，使用本目录的 `r4_training_source.json`。它固定原launch、summary、checkpoint路径与校验值。loader按原argv恢复路径、reference artifact及GPU绑定，用当前请求的训练科学身份恢复原placement后，与原launch完整公开spec逐字段比较，再调用真实 `SubprocessRunner.recover_completed_train`；不删除身份字段，不重训。

原tuple保留 `audit3_go2_full_train-train`、原GPU绑定、完整summary／checkpoint元数据和执行环境来源。私有import receipt绑定这份tuple与六个原文件，并明确为 `imported_historical_training`。Genesis沿用既有 `execution_placement.reference_training_source`；账务据此排除旧训练，当前Q仍走本session的三个正常PreGenesis scopes、OperationAdmission与attempt journal。

prepare-only发布 `completed_reference_training_import.json`、`reference_genesis_preparation.json`及既有session／cost／continuation／final-plan合同。准备材料使用正式运行的同一scope构造函数，明确没有当前reference训练attempt；不把准备材料冒充已完成Genesis。resume重新恢复原证据并核对create-once材料。CLI保留只读nvidia-smi清单核对；不会触发Isaac环境probe或训练／评测。

此前 `07581294` 的恢复验证和 `918da162` 的独立验收保留为历史记录；“独立Q后再full-import”的执行组合已弃用。

真实R4已在上述冻结源码上通过正式CLI prepare-only及resume重载。服务器根目录 `/data/yuyangliu/isolated/r1c_r2_cpu_20261004`：`run02/` 是当前准备材料，`checks/current_session_prepare_result.json` 是验证结果，`checks/current_session_argv.json` 是实际参数。session为 `r1c-r4-current-q-two-generations`，session contract为 `9c97bfb537d45afacc0b08f6d944e8de898db99776f07b8b15a26634d533be79`。原六文件前后hash／mtime／大小均相同；原summary完整相等，`formal_training_executed=true`，checkpoint为 `sha256:7b582643510cc9ddd9ae6755fc551cfc89f6ea1a06fc7b6837dd3b203da11c4c`。准备材料重载后bytes及mtime不变；当前attempt为0，没有已完成Genesis。验证显式拦截训练／评测和backend环境probe，并以空CUDA可见设备运行。

本目录 `prepared_session_argv.json` 保存实际重载准备材料的CLI参数，包含 `--prepare-only --resume-existing`，入口为冻结checkout的 `scripts/run_generation_session.py`。它不启动Q或作者。最终文档提交不会改变隔离checkout的执行源码commit。

## 作者与训练数值

`external_file`进入现有generation provider选择、作者循环、source_request／repair、journal和成本门。共享JSON解析及candidate materializer从原HTTP实现移出，没有第二套解析、runner或agent平台；HTTP原错误分类、候选续号和backend标签保留。Stair旧的正式15代固定门改为核对当前final plan的代数；历史计划和live路径仍保留原解释。

每次作者请求在本lane的 `provider_candidates/<generation>/<lane>/requests/<编号>/` 发布实际 `system.txt`、序列化 `user.txt`，最后原子发布 `request.json`。外部执行者仅交付原始 `response.txt`，通过临时文件原子改名。源请求和repair自然产生下一请求。将来每次实际作者使用新 `gpt-6.1-sol`、`fork_turns="none"`，只接收这两个实际文本及最小输出交付说明；本轮由CPU fixture供给答复，没有启动作者。file provider本身不调用网络、模型或子进程。

`external_provider.json`没有API key、URL或伪造的provider token计量。receipt区分请求模型与已知backend，不宣称返回模型已被验证；usage为空，明确 `unknown_full_reserve`。每次已完成请求按完整input／output reserve扣账，只有冻结external政策和匹配receipt允许这种续行，原HTTP缺失usage仍拒绝。截取超限响应时receipt明确为 `byte_limit_prefix`，不把前缀冒充完整答复。300000／32768是保守记账预留及请求输出额度，不是已测token数或已验证模型context capacity。

训练仍只采24个分散控制步、每步前16个env。本次在原native VNext奖励输入快照加 `last_action`、`prev_action`、`joint_vel`、`joint_torque`；不扩展eval连续采集。shape／unit来自同一input contract，action／joint顺序来自实际runtime projection。动作是reward使用的当前／上一action-manager缓冲（policy_action），不是关节目标；速度rad/s，力矩N*m，为控制步末、reset前的缓冲快照，不是物理子步均值。Go2增加的四个12维float32向量最多73728字节，未改变采样schema v2。

两env独立用例：动作差平方和2／9，clip-max=5后归档raw为2／5；力矩平方和25／5，clip-max=10后raw为10／5；权重−0.1／−0.2得到total rate −2.2／−1.5。raw明确是guard／clip后、weight前；RewardManager的dt乘法在后。fixture另核qvel列、四路键集合、unit、实体顺序和源张量修改后的归档值。

## 同session成本与G1停点

`two_generation_cost.json`由真实composition的workload冻结：task protocol `d85bebb001f72dd13c8107215ec5d81e391d795fd2541c8dc589ce4fbe9e35e9`，每eval seed为288×645＝185760控制步。原600步估算不能用于此合同。真实workload在上述服务器checks目录的 `workload.json`／`workload_identity.json`。

| 范围 | 预留工作 | GPU时 | env控制步上限 |
|---|---|---:|---:|
| 三卷reference资格化Q | 3／5／7 seeds | 0.5201004328 | 2786400 |
| 当前Q＋完整两代 | Q＋4 train、4 dev、2 accept、5 sealed（含2次final与1次retry） | 14.5142598551 | 799806720 |
| Q＋G1，计一次允许的sealed retry | 2 train、Q及G1评测 | 7.3958233763 | 400646400 |

新成本v2只有一个 `reference_mode=imports_training_only`：0次reference train／retry，3个当前Q phase，4次候选训练。旧v1 reader保持原表示和hash；不改历史Stair cost。Q先冻结一个combined slot，再按3／5／7拆分，共1872361557988 ns；不重复加余量。

每train按10800秒timeout＋16秒清理余量预留，来源R4实测lease为8472.390263615秒。eval来源299577849278 ns是R1b评测receipt的299361920146 ns，加环境probe约0.215929131955秒后取整，**不是一条原始lease receipt整数，也不是216秒probe**。按seed比例外推并加25%；accept／sealed仍是预测。现有admission给train／dev／accept／sealed约10800／358／608／857秒timeout。

统一session的operator ceiling为 **15 GPU时／799806720控制步／6 GiB**。冻结GPU预留52251335478350 ns，输出5064166362字节；成本hash为 `90efcc12ec82b1c0550c3ba70073cdbc24a4370d2b44c8a4f49404cb9557ae03`。输出来源为R4目录18860360字节、R1b目录85131515字节，eval按seed比例并加25%，另含2 GiB session与256 MiB final余量。

G1使用 `--max-generations 2 --pause-after-generation 1`。固定前缀Q＋整个G1＋唯一sealed retry为26624964154638 ns、400646400步；即使纳入全部session／final输出余量也是3881928591字节，低于 **7.5 GPU时／4 GiB**。复用现OperationAdmission和正常pause，没有新逐job预算hook。7.5／4 GiB是本阶段前缀预留边界，不是额外运行中磁盘硬配额；CLI仍绑定完整session合同。GPU执行与G2都尚未放行。

## 验证

上一提交的external作者与采样已由独立验收通过，本轮没有重构。其149项历史回归记录保留；本轮只运行受影响的成本、入口、训练复用、Stair session共43项单元回归，以及训练复用的CPU端到端场景。

端到端使用真实composition／OperationAdmission／attempt journal／Genesis／pause-resume／cost replay，仅训练和评测工作体是假runner。恢复fixture从原保存的typed result读取；不能用它声称已覆盖真实checkpoint解析，真实R4验证另行使用生产recover。三个指定Q phase各一次；原训练无当前reservation；两代四个指定lane均有完成train receipt；G1后正常暂停；resume保持Genesis和已有attempt；最终terminal audit与报告相等。Stair第二代从prompt中的实际当前父代源码追加修改，不用reference源码冒充新父代。

实际结果：Windows Push闭环通过，Stair因缺Isaac资产被跳过；随后在服务器以空CUDA设备运行，Push通过，Stair Go2复跑通过（195.073秒），没有skip。Stair首次运行暴露原argv采样标志未恢复，已在 `429e6027` 修复。服务器日志为 `checks/current_session_cpu_tests.log`（首次两项，保留修复前失败）及 `checks/current_session_stair_cpu_test.log`（修复后通过）。没有重复扩大回归矩阵。

只读复核亲读了run02来源receipt、scope准备材料及CPU结果，并重新核对原六文件的当前hash／mtime／大小，未发现新问题。其结论仅到CPU恢复与prepare-only／resume，不代表真实Q、qualified anchor、promotion或真实generation已执行；本阶段仍交父级独立验收。

两份原始审计及其untracked目录保持原样；未改权限／凭证，未读K5隐藏数据，未干扰只读视觉线程。阶段完成后停止，交独立验收。
