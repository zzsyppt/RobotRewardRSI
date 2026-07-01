# Phase 0 项目边界与独立骨架

## 目标

按照 `PLAN/true_rl_agent_refactor_plan.md` 的开端要求，先建立 `true_RL_Agent` 独立 Python 项目边界，避免后续实现时退化为父项目 `humanoid_agent` 的局部脚本改造或 LLM 调参入口。

## 本次实现

- 新增 `pyproject.toml`，声明独立包名 `true-rl-agent`，源码根目录为 `src`。
- 新增 `src/true_rl_agent` 包及主要架构子包骨架。
- 新增 `configs/agent`、`configs/tasks`、`configs/backends` 的 scaffold 配置，显式关闭第一版 LLM curriculum 控制和 tuning-only acceptance。
- 新增 `scripts/run_loop.py`、`scripts/run_round.py`、`scripts/materialize_job.py`，当前全部 fail closed，避免把未实现入口伪装成可运行系统。
- 新增 `README.md` 与 `docs/reference_inventory.md`，说明旧工程和本地 Codex 源码只作为参考，不作为运行依赖。
- 新增边界测试，检查可安装项目元数据、本地包导入、运行树不引用父项目包或旧工程路径。
- 更新 `.gitignore`，忽略本地 `codex/` 参考源码和 Python `*.egg-info/` 构建产物。

## 验证

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

结果：3 个测试通过。

```powershell
C:\Users\lover\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pip install --dry-run -e .
```

结果：`Would install true-rl-agent-0.1.0`。

服务器同步后使用默认 `python3` 运行边界测试时发现服务器是 Python 3.10，没有 stdlib `tomllib`。已将边界测试改为无额外依赖的文本断言，并将 `requires-python` 调整为 `>=3.10`，使 Phase 0 骨架与服务器默认 Python 兼容。

```powershell
git diff --check
```

结果：无空白错误；仅提示 `.gitignore` 未来会按 Windows 工作区换行处理。

```powershell
rg -n "from humanoid_agent|import humanoid_agent|D:\\humanoid_agent\\src\\humanoid_agent|/data/yuyangliu/AGENT/src/humanoid_agent" true_RL_Agent\src true_RL_Agent\scripts true_RL_Agent\configs
```

结果：无匹配。

## Review 结论

- 没有引入父项目运行时依赖。
- 没有提前开放 LLM 控制课程或 tuning-only acceptance 的入口。
- 未实现脚本入口全部 fail closed，后续阶段必须显式实现 AgentCore、BackendAdapter 和 Runner 后才能执行。
- 当前提交只完成 Phase 0 项目边界，不声称训练、评估或 ledger 已可用。

## 下一步

进入核心 schema 与 fake backend/fake runner 阶段前，继续保持测试先行：先固化 task invariant、policy initialization、candidate surface、result object 的行为测试，再补实现。
