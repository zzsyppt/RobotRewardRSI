# R1a 预声明：统一生产选择与 CPU 投递

起点：`82828fd66d0cd687a63559f2fc90c834a4d35b43`，分支 `codex/trajectory-translator-upgrade`。
用户已确认 P2c3 限定离线 PASS；其旧文档待验收状态不再适用。真实 collision 仍未知。
本阶段只有一个 writer；Astra ultra 只读开发审查。禁止功能模型调用、K5/隐藏模型材料、GPU、训练、新采集及服务器修改。

## 先冻结必要事实，再测容量

原 18 槽保留可核对映射：队列/结果/有效行；净进度；低进度且有足活动、低进度且全足低速两比例；全队列长低速 off 模式；成功同模式及失败对应分母；最大成功回撤返回；翻转先于低速 off；短 history 事件或独立 manager 终点；activity/contact 定义；缺输入；延期能力；父代不可用；clock/history；coverage/截尾；comparison 范围。无可用事件时保留明确 unavailable，不能拿别源补主队列。定义可重写合并，但每槽到新句的映射必须留在内部 receipt。

新增必选规则在看预算结果前固定：

- 周期：按 qualified unique cycle 数降序、稳定实体名破同分，选择一足；保留 qualified/candidate windows、unique/complete cycles、qualified episodes/全部 episodes、episode-equal period、有效读数 edge 覆盖和 unknown rhythm。若全无合格周期仍报告 unknown，不能制造步态。
- 名义关节：按所有回合 minimum signed margin 的最小值升序、joint name 破同分，选择一关节；保留 finite/unknown 分母、5% 内侧 edge band 与越界计数、回合等权 margin/range occupancy。仅 USD nominal，effective limits unknown；不可用时明确资格缺口。
- 名义地形：每个实际存在的 C0/C1/C2 regime 一条，优先最低 pooled known fraction、再实体名破同分；保留 known/total/unknown 和回合等权竖直差，参考点/world/m 单位、nominal-only/runtime unknown。没有 regime 不造观察；边界/patch 外未知不变零。
- 成功/翻转反例、短事件、各量独立分母和截尾限定优先于额外同类实体。其它足/关节/地形实体有确定省略理由与 IDs；不要求 12 关节/15 地形句全投递。

## 边界小集与验收

同一冻结事实集比较 8192/10240/12288 紧凑 UTF-8 bytes，记录必要覆盖、独立事实密度、重复定义、选中/省略、system/user/整 provider request bytes，选覆盖合格的最小档。真实 tokenizer 不可用则 tokens=unknown，bytes 不换称 tokens。

使用 K1/K2/K3 已公开历史档案；原档案及重绑定 CPU fixture 分开标记。只回归受影响算子与投递。边界集：不同回合长度/padding、缺 history、无合格周期、缺名义资格、错误单位/实体顺序/源版本/几何、terrain 边界 unknown；成功返回/翻转/短终点反例；current/parent/sibling 与缓存 archive/profile/code/budget 切换；最终真实 provider serializer 的 system/user 字段逐句包含预定事实并能回溯 operand 与原 chunk。

版本方案：新 `trajectory_numeric_v2`、独立合并生产 catalog 和报告/selection 身份；旧 legacy/v1 与独立 v3/v4/v5 supplement 保持回放，Push root 原路径保持。新档案按 capture contract/资产/坐标/名称/生成证据准入，不扩大历史 manifest 白名单。历史 qualification 只用于原先已审来源；新 capture contract 是 CPU 结构接口，不能声称已 runtime 实录。

R1a 停点是 CPU 实现、预算与最终序列化投递证据经开发审查后 commit/push，等待父级独立验收。R1b 的实际每 env limits、运行地形证据及真实 GPU 档案未实施；动作/reward、父代数值差分、G1 肢体、完整 E2/E3 和训练收益仍未完成。
