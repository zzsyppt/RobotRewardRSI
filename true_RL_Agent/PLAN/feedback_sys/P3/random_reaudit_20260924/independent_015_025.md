# 015 / 025：对照原评分前的独立判断

本记录先于读取这两项原 review_sheet 条目保存。主评此前接触过整个评测的方法审计，也知道 015 存在假说判定争议，因此不是全新盲评；本次仍从候选、实际输入、原始 rubric 和证据重新判断。未调用作者 API 或仿真。

## proposal-015

- 代码 24–28 行保持 projected_gravity 的第二分量平方，第一分量改成 max(abs(gx)-0.5, 0)。相对参考，roll 对应分量保持不变；不是 proposal-043 的 Euler roll² 改动。
- 68 行的 “unbounded quadratic pitch penalty” 不准确：被平方的是单位投影重力分量，原值本来有界。新项也不是对真实 pitch 角的线性惩罚。
- 70 行明确写 abs(sin(pitch))<=0.5，这部分准确。在正常姿态分支，相应角度约 0.524 rad；周期角度/翻转不能靠此单项约束。
- 实际输入中的 forbidden_contact 比例 0.324/0.333/0.176 可核，确为这些分层的主导终止类别；不支持 74 行 “because the previous flat penalty fought ...” 的因果归因。它应是尚未验证的假说，不能写成已观测根因。
- 代码确实在所有非零投影 pitch 分量处降低惩罚，包括 0.65 rad 以上；它没有保证不超过稳定阈值。这是需要报告的干预风险，不等于训练结果已知。
- 对 c1_frontier 成功/失败的姿态比较可以提供不利观测，不能反事实证明放宽适当 pitch 的学习效果必然无效。
- 暂定 action_relevance=1（可执行但弱证据假说）；testability=2（指定 public-dev 分层、相对参考、方向及保护条件）；restraint=1；authorization=pass。保留代码描述错误和过强因果表述；假说本身为 untested，不记 contradicted。

## proposal-025

- 当前输入中 0.5、medium 30/48、repeated 13/48、low 33/48 都有依据。
- 候选仅修改 stationary_v3 的函数体及元信息：加入 exp(-6*tilt)*(1-0.8*push_active)，速度尺度从 0.08→0.25、角速度尺度从 0.5→0.8。rationale 明确披露了这些变化，不能说它偷偷只改 gate 却未披露尺度。
- 在直立且无推力时 gate=1，但尺度变化仍使残余速度惩罚大幅减弱。scaled_speed=0.08 时平面惩罚由 0.125 降到 0.0128。其 “still damping” 字面成立，不应解释成承诺与旧项同强度；然而诊断需要说明这个风险。
- 开头 docstring 将 “suppresses stepping” 写成事实，依据不足；rationale 随后明确标为 Hypothesis。应指出表述不一致，而不是将迈步幅度/时机受到限制的假说直接证伪。
- 现有 G1 env55/r1 录像帧显示足部交替/迈步、步距增大，后段屈膝并明显失稳；支持“这个回合有迈步”，不能证明放松惩罚不会改善或会改善。
- env12/r1 末段帧显示身体保持直立、脚的位置改变；回执 final_planar_speed 0.343343 m/s 且 not_recovered。这个回合不支持“没有迈步所以未恢复”；但它属于 low，不能当作 medium/repeated 全部机制的直接证伪。
- 暂定 action_relevance=1；testability=2（public-dev 分层、方向与明确失败条件）；restraint=1（docstring 过强且将结果未改善叫机制被反驳）；authorization=pass。没有发现新的数值/代数/权限硬错；不把未测的因果假说算作已证伪。

## 视觉观察的范围

- g64_r1_zoom：所选帧清楚显示前身立起、随后向后并伴随侧向翻转，四足离开支撑面。对应 r1.e64，426 步 forbidden_contact；root low_progress 为 369–426，tilt 为 385–426（394 确认）。适合支持“停滞后倾斜并翻倒”的该回合过程，不能证明奖励修改的反事实效果。
- g156_r1_zoom：所选帧显示低身体姿态以及腿部反复伸屈；最后身体很接近台阶表面。图像不能精确识别首次受力点或接触面，故不单凭画面声称腹部碰撞位置已经查明。
- 这里只看保存的关键帧拼图，未完整播放视频。录像与数值是回放匹配，检查 checkpoint、环境/格子、步数和终局；不能将其说成原始采集时同步视频。
