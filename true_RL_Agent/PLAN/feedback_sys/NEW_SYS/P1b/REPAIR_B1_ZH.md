# P1b B1：聚合claim语义绑定修复

起点 `c0cafa0d4b1684784eddab5efe639ec7bad647b8`，同一专用分支。父线程暂不放行的唯一阻塞是四种generic聚合模板可在同cohort内换metric而保留旧label/entity/unit，重渲染及重算claim ID仍通过。固定产物的真实数值、801句与必要事实表达已通过父线程独立检查；本修复不改它们。

## 最小改动

仅修改生产 `report.py` 及对应 `test_trajectory_card.py`，不改测量/聚合算术、P0/P1a、来源、选择或阈值。新增一个有限函数 `cohort_metric_literals(template, key, metrics)`，从现有指标键派生标签、足实体、current/份额/组合角色和单位；单位还须与descriptor固有单位一致。`group_claim`不再接收独立自由label/unit/entity。

validator对scalar_distribution、ratio、presence、off_runs限定：scope为cohort、唯一operand为metric、path精确为`["metrics", key]`，完整literals须与该模板和指标键派生值一致。保留既有source/episode/cohort/condition绑定；没有新DSL或配置框架。history目录仍保留；只拒绝用history指标生成current-marker聚合句的非法组合。

关键实际diff：

```diff
-def group_claim(template, group, key, label, section="总体行为", **literals):
+def group_claim(template, group, key, section="总体行为"):
     g = groups[group]
-    return claim(template, section, group, {"metric": (g["record_id"], ["metrics", key])}, label=label, **literals)
+    return claim(template, section, group, {"metric": (g["record_id"], ["metrics", key])},
+                 **cohort_metric_literals(template, key, g["metrics"]))
```

```python
if claim["template_id"] in ("scalar_distribution", "ratio", "presence", "off_runs"):
    if scope["record_type"] != "cohort_summary" or set(claim["operands"]) != {"metric"}:
        raise ValueError("cohort metric operand role mismatch")
    path = claim["operands"]["metric"]["path"]
    if len(path) != 2 or path[0] != "metrics":
        raise ValueError("cohort metric path mismatch")
    expected = cohort_metric_literals(claim["template_id"], path[1], scope["metrics"])
    if claim["literals"] != expected:
        raise ValueError("cohort metric label/entity/unit/role mismatch")
```

## 真实变异前后

全部使用原final_card实际cohort记录，未改数值底账，变异后重新渲染并重新计算claim ID。由本线程在修复前后分别运行并保存完整原句、变异句、operand及结果。

| 变异 | 错误句中的数值/语义 | 修复前 | 修复后 |
|---|---|---|---|
| 净进度→终点时间 | 净进度4.254518m变成6.44m | ACCEPTED | REJECTED |
| 成功RL longlowoff→RR | 仍标RL，3/682变1/682 | ACCEPTED | REJECTED |
| FL current marker→history | 仍标current，池化.703179变.715006 | ACCEPTED | REJECTED |
| FL off_runs→RR | 仍标FL，11593段变15219段 | ACCEPTED | REJECTED |
| 仅label改成终点时刻 | 进度数值被错误命名 | ACCEPTED | REJECTED |
| 仅unit m→s | 进度被错误标秒 | ACCEPTED | REJECTED |
| 仅off_runs entity FL→RR | FL区间被错误标足 | ACCEPTED | REJECTED |
| 仅current标签→history | current数值被错误标角色 | ACCEPTED | REJECTED |

新增5项测试方法覆盖四类换metric、label/unit/entity/组合文案单改、附加operand覆盖unit、底账单位与descriptor不符，以及合法metric切换。所有变异均在重渲染/重hash后验证；合法切换同时派生新语义可以通过。原95条generic claim继续通过，不将“全部拒绝”当修复。

## 验证与交付

最终完整命令、原始日志及SHA、产物代码hash、前后内容比较、独立审查结果汇总在同目录 `repair_b1_verification.json`。新四产物目录为 `outputs/trajectory_p1b/repair_b1/final_card/`；原`final_card/`保留作B1修复前证据，不覆盖。

规定的相关测试、107+201、必要247/完整309、真实全队列原数组与全句追踪全部重新执行；不把上一轮通过冒称本轮执行。初版和修复版均保留15句主视图、801句完整审计、27例及原18冻结选择，legacy不入主队列。

实际最终结果：

```text
Ran 65 tests in 21.796s, OK
Ran 107 tests in 5.660s, OK
Ran 201 tests in 52.314s, OK
necessary pass247/247, full pass309/309, unsupported=0
raw audit passed:866 episodes,69280 metrics,8400 aggregate metrics,
  32042 references,801 claims,247 necessary facts;18.448963s
```

独立只读复验：95条generic claim × 其余79个实际指标共7505次替换，5947个可成句攻击全部拒绝；1558个不能成句未计作校验通过。另380个标签/单位/实体/附加operand攻击及95个底账单位错误全部拒绝；1687个合法metric+派生语义切换通过。最终801句、999引用、8生产依赖、两份P1a manifest及冻结selection hash均匹配。

真实重建21.919889s，峰值242003968 bytes；三内容产物与修复前逐字节相同，所有数值、claim ID、15句主视图均不变。manifest仅变化report代码SHA及两项运行成本：

```text
report.py 5125a6e7eb54fd5c3dd48bc9969b2ac2fe61bef4e74804a461cd57b1b32ce335
manifest  3bfd19515481faa6d9f273426e6b14d141fe616642536d15a6a55c2b2f9990f0
evidence  58f1014f557847c5ee32a3c03e4bebf4becdc2bc35552c6ac948a5fcc576af6e
card      d83ddf0b8ee28e8d00705de81029930474b47ba230532596d50aabd2e9b0d9a2
zh view   c5589b8b79efd605c5d5d305a84ba58f246de3fbba55a39d48cff0d7f90c90ac
```

本轮仍只有开发线程写入。只读独立review不调用付费模型API，不安装、不训练；P1c作者协议/字节预算/实际请求、P2/P3均未开始。修复以新commit推送，不amend初版；完成后停止写入，等待父线程复验。
