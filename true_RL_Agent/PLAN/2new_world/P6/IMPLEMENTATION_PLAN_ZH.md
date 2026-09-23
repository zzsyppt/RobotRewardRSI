# P6 实际实施计划

2026-09-11；进场基线 `5cef484dfa5319130faa02d7e9bbcdd9e9b4a84e`。工作树进场干净。本计划承接 NEW_WORLD_PHASED_PLAN_ZH.md 的 P6，以及 P5 证据投影复审；未完成的项目不得以文档或模拟通过代替实测。

2026-09-12 00:32 更新：B 阶段协议、预算与资格门已 [正式签发冻结](FORMAL_DECLARATION_ZH.md)，默认 active；reference 物理资格仍待验证。以下保留阶段目标和进场状态，当前接线结果见 [开发记录](DEVELOPMENT_LOG_ZH.md)。

## 目标与边界

完成 Stair / Push × G1-37 / Go2 的四格装配、共享执行与真实物理证据。新 Push 复用 Stair 的两个既有 RobotProfile；G1-29 Push 单独保留历史读取及当前后端回归，不将不同 DoF 假称为同一矩阵行。P6 完成 reference 与任务资格化；P7 的 LLM 15 代搜索、多训练 seed 改进研究及最终 holdout 不提前宣称完成。

旧完整 15 代 checkpoint、ledger、原始 prompt 与科学身份不改写。任务成功、扰动、评分及分差算法归 TaskPlugin；资产、关节、动作接口及物理尺度事实归 RobotProfile / RuntimeAdapter；绑定只冻结二者组合。共享宿主持有生命周期和证据读取，禁止加入任务名或机器人名条件。

本地 Codex 的 `codex-rs/core/src/state/service.rs` 将运行服务的唯一所有者与固定绑定分开；本轮只借鉴“宿主提供已绑定能力，消费者共用权威视图”，复用现有 phased adapter、composition 和 runner，不复制其服务注册系统或增加插件框架。

## A. 先闭合 P5 公开证据链

1. 复用 `PhasedEvaluationAdapter.project_public_phase_summary`，由宿主注入历史、调度、提示及源码消费者；删除 Stair prompt 的重复接口。
2. Stair selector 与探索复用任务内 guard-floor 算术，保留历史统计版本；探索支持经身份验证的 parent public-dev，prompt 只消费探索分差。已测但未 selection 的证据不得伪装成 selector 判决。
3. 宿主公开原因由共享记忆保存；移除 Stair 私有白名单与无消费者的 8 列兼容补齐。最小核心结构变化更新版本。
4. 对剩余小项按实际消费者取舍，删除开发期冗余；有冻结历史消费者的 reader 保留并记明原因。

验收：真实 c6 G3-A 的调度/展示身份与分差一致、未测 sibling 无分数；Push 同一链可读；历史 P4 正式回放不变；定向测试及最后完整回归通过。不新增 LLM 请求。

## B. 冻结 Push 新主线

现状：G1 Push reference_v1 已能装配共享 Genesis，但仍有 legacy binding/grant；Go2 Push 未注册，训练/评测脚本仍直连 G1 构造器；旧成功分类没有以实际扰动与恢复驻留作为必要条件。不能把 P4 fake runner 证据称为真实任务通过。

1. 新建非 legacy Push 协议与 G1/Go2 薄绑定，复用各自现有 robot runtime / PPO。旧 G1 协议维持历史读取，不改其冻结科学定义。
2. 先确定单一实际施扰机制、单位、前后测量及任务状态机，再接入训练和评测。Go2 默认禁用 locomotion push 事件，故事件由 Push 所有并安装，不修改机器人默认或 Stair。
3. 冻结站立目标、施扰时间/方向/尺度、稳定驻留、恢复窗口、跌倒/越界与重复事件规则。公开信号只来源于已执行事件，不用自主运动的速度差冒充扰动，也不公开未来计划。
4. 训练、评测与原生录像共用相同环境和事件实现。记录真实 observation/action 内容 hash；录像附 checkpoint、场景、事件和评测身份。

正式测量前，在本目录补齐不可变的实验声明，写入具体参数、seed、样本数、资格门、资源预算及停止规则。参数不得依据 accept/sealed 成绩回调；有必要的机制修订开启新身份并记录作废证据。开发探针只解决接线或校准，不能冒充留出验收。

## C. 有界真实执行

1. 先运行无扰动、已知扰动、重复扰动与明确失败的短物理检查，测量注入前后状态、事件次数、恢复驻留与分类整数闭合；未被推不能记恢复成功。
2. 完成新 Push reference 的 random-init 固定预算训练。原 Push 预算为 4096 × 24 × 500 = 49,152,000 env-steps；是否沿用由 B 阶段冻结合同决定，不在结果出现后延长刷过门。
3. 事前冻结 dev / accept / sealed；只在 dev 做允许的开发诊断。完整评测后按预声明基本可学性和余量门给出 GO 或真实 NO-GO。低表现按接线、任务分布、策略能力区分。
4. 同一 reference checkpoint 获取原生成功与失败视频，并与评测事件/分类逐项对应。G1 Push 取得当前后端执行回归；Stair 两格复用已验收 reference，但执行当前代码的 ABI / 共享宿主回归。

GPU 启动前重新检查占用；最多两张 GPU 并行，使用项目现有 lease 和作业归属。不开新 provider 请求，不动其他任务。每个完整训练按冻结 iteration 上限停止；基础设施失败只使用已声明重试，不无限重训。长作业保存日志及可恢复身份。

## D. 终验与交付

四格同一入口的 list/resolve、reference/阶段评测/恢复/终态审计均有明确证据；真实训练和物理视频与模拟状态机测试分开报告。核对旧历史身份不变、任务语义未进入公用层、源码与服务器一致、证据引用闭合、进程/lease/成本收尾。

只新增保护真实合同的少量测试，复用现有完整 discovery 和 opt-in 动态检查。完成后审计相邻重复实现、遗漏消费者及过时文档；删除无消费者逻辑，不为小概率情形增加第二套防御框架。最终 `ACCEPTANCE_ZH.md` 逐项列测量、限制与必要保留的旧 reader；尚无证据的项目明确未完成。
