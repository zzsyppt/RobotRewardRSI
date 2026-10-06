# 轨迹翻译阶段放行记录

当前修复、审计处置、验证范围和后续停点统一见 [CURRENT_REPAIR_ZH.md](./CURRENT_REPAIR_ZH.md)。下文保留为 2026-10-03 的历史快照；其中版本、预算、默认配置与放行状态不代表当前实现，也不作为启动下一阶段的授权。

更新于2026-10-03。阶段必须由父线程独立验收后继续；开发自测不替代放行。

| 阶段 | 版本 | 当前状态 |
|---|---|---|
| 升级前基线 | `e85d37c` | 可回溯；与标签 `baseline/pre-trajectory-translator-20261003` 对照 |
| P0 冻结目录、源与18例 | `2d888a846636ffb0e3a732d9d3423288ff3b2cac` | 已放行；冻结账/抽样不改 |
| P1a 数值、T1/T2修复 | `fe1f5c3775d402a5c245dd754c3041f3be92b53c` | 已放行；必要247/完整309 |
| P1b 卡片、B1语义绑定 | `00e63feb76123292984071cd4a6bc5e9d362bd1a` | 用户本轮明确：父线程已亲读源码/原数组/反例并独立验收放行 |
| P1c 最终作者输入 | `41f5a44bf6dc63e1fd8aa8efed291d13f44f7511`及[P1c证据](./P1c/DEVELOPMENT_ZH.md) | 用户本轮明确：独立及父线程亲核受限离线生产HTTP capture PASS，已放行 |
| P1d 正常任务自动接线 | [P1d证据](./P1d/DEVELOPMENT_ZH.md) | 正常composition双真实archive注入CPU链路PASS；634实际测试通过/8环境skip（P2a勘误）；用户本轮确认独立及父线程亲核已放行 |
| P2a Stair 三源描述性迁移 | [P2a证据](./P2a/DEVELOPMENT_ZH.md) | 用户本轮明确：三来源独立及root验收已放行；仅描述性案例，完整unseen E2资格未满足 |
| P2b1 Push 根部离线迁移 | [验收前语义修复](./P2b1/SEMANTIC_BINDING_FIX_ZH.md) | 历史首次语义绑定FAIL已修复；本次用户明确P2b2独立及root CPU亲核PASS，其所依赖的冻结P2b1离线接线基线已可用；22文件冻结保持 |
| P2b2 Push 正常自动接线 | `ca140cfd1e6084118e0c36a4c28f8fe97a2dc294`；[证据](./P2b2/DEVELOPMENT_ZH.md) | 本次用户明确：独立CPU接线及root亲核PASS，CPU阶段已放行；不等于新GPU rollout或模型理解通过 |
| P2c1 局部周期资格 | [预冻结](./P2c1/FREEZE_ZH.md)、[首轮交付](./P2c1/DEVELOPMENT_ZH.md)、[语义修复](./P2c1/SEMANTIC_BINDING_REPAIR_ZH.md) | 用户确认71c9cac独立及父级限定离线PASS；周期数值/语义正确，新增句仍全部被8192B预算省略，生产投递待后续专门解决 |
| P2c2 | [预冻结](./P2c2/FREEZE_ZH.md)、[交付证据](./P2c2/DEVELOPMENT_ZH.md) | 用户确认父级原代码/原数组/独立结果审阅，限定离线PASS；新句仍0/36进入8192B preview，不称生产投递；5%为浮点构造端点闭区间 |
| P2c3 | [预冻结](./P2c3/FREEZE_ZH.md)、[交付证据](./P2c3/DEVELOPMENT_ZH.md) | 18新测试+171兼容、三源4257915点名义地形独立面复算开发及内部只读审阅PASS，待父级独立验收；历史importer字节和runtime碰撞面未验证，新句0/45投递；生产资格/预算另阶段 |
| K5 / P3 | — | K5仍独立审查，开发不得读取行为调规则；另一个只读P3 pilot可固定ca140cfd既有payload进行，本线程不交换结果、不调用模型 |

P1c交付：full.v4/public.v5独立分发、Stair8192B整句选择、来源/phase/trial绑定；生产HTTP本地捕获四帧，18/801 claim、8094B；74+65+107+201次专项执行全部通过，冻结247/309、原数组与15个真实产物负例通过。原始输出不入Git，hash和命令见[P1c verification](./P1c/verification.json)。

仍未证明：真实模型理解（后续按用户要求clean 6.1 sol，真实DeepSeek另阶段）、新runner/GPU rollout、跨策略泛化/视觉/闭环收益。Push根部卡及正常CPU接线已通过本次用户确认的阶段门，不再列作未实现。P2a只完成三源描述性迁移，不是完整E2。P1d提供显式配置接线，默认仍为legacy。父代比较待接同当前环境/同协议/明确父子档案；版本变化按相关性解释，不永久要求所有source hash相同。周期/限位/地形/历史link-COM映射保持后续义务。父线程放行前不进入下一阶段。
