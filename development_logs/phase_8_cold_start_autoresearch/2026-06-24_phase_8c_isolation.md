# Phase 8C：候选代码隔离最小实现

日期：2026-06-24

## 设计判断

Phase 8C 的目标是让候选代码不再在主 runner 进程中执行，并尽量用服务器实际支持的 OS 机制限制候选代码。

服务器探测结果：

- Docker 命令存在，但当前用户不能访问 `/var/run/docker.sock`，因此不能把 Docker 当作 8C 依赖；
- `bwrap` / `firejail` 不存在；
- `unshare -Urn` 可用；
- `unshare -Urnfp --mount-proc` 可用；
- `unshare -Urn` 下 CUDA smoke 通过，说明 network namespace 不会直接破坏 GPU 可见性；
- unprivileged bind mount / remount 不可用，因此不能用 mount namespace 实现 repo 只读；
- 采用 Landlock 作为文件系统限制主机制，若 Landlock 不可用，服务器强验收必须 fail。

因此 8C 不强行加入服务器不支持的 Docker/chroot/bind mount，而采用：

1. 候选代码独立子进程；
2. `unshare -Urnfp --mount-proc` 创建 user/network/PID namespace；
3. Landlock 限制文件系统：
   - 不授予整个 repo 读权限，避免候选读取 `configs/` 中的 hidden/iid/eval seed/profile 定义；
   - 只授予候选加载需要的最小 API 面：`src/humanoid_agent/__init__.py` 和 `src/humanoid_agent/code_sandbox/`；
   - candidate code 目录只读；
   - `candidate_output/` 可写；
   - `runner_artifacts/` 不授予候选写权限；
   - hidden/audit 路径不加入 allowlist；
4. 环境变量 allowlist，密钥类变量不进入候选进程；
5. timeout + process-group/PID namespace 清理；
6. 父进程只读取 child summary，不 import 候选 entrypoint。

## 实现文件

```text
src/humanoid_agent/code_sandbox/isolation.py
src/humanoid_agent/code_sandbox/isolated_loader.py
scripts/phase8c_candidate_child.py
scripts/phase8c_isolation_selftest.py
configs/cold_start/phase8c_isolation.yaml
```

## 明确边界

当前 8C 解决的是“候选代码加载/组件构建”的隔离原语，不直接开放 LLM-authored 训练代码。

仍不宣称完成：

- 任意 LLM-authored code-level candidate 的正式训练；
- reward/loss callable 在 Isaac/RSL-RL 长训循环内的全链路隔离；
- grad/optimizer/forward-backward runtime hook；
- eval 期 observation/action wrapper 开放。

这些属于 8D/8E 之后的工作。

## 验收标准

服务器强验收必须运行：

```bash
python scripts/phase8c_isolation_selftest.py --require-server-enforcement
```

覆盖：

1. no-op candidate 在隔离子进程中通过；
2. 父进程不 import 候选 entrypoint；
3. `DEEPSEEK_API_KEY` 不进入候选环境；
4. hidden/audit 文件读取失败；
5. 主 repo 写入失败；
6. 仓库内 protected config 读取失败，例如 `configs/phase2_baseline.yaml`；
7. runner artifact 目录写入失败；
8. candidate output 目录可写；
9. 网络连接失败；
10. 候选退出后无残留子进程。

若服务器不支持 `unshare` / PID namespace / Landlock，强验收必须 fail，不允许降级成“伪安全”。

## D1/D2 追加收口

- D1：真实存在，已修。旧实现把整个 `repo_root` 加入 Landlock read allowlist，会让候选读取仓库内 `configs/phase2_baseline.yaml` / `configs/phase6_loop.yaml` 等 protected eval profile/seed 定义。现在不再授予 whole-repo read，只授最小 candidate API 和 Python/system dependency read。
- D2：真实存在，已修。旧实现把 child `run_dir` 作为候选可写目录，未来 8E 若把 train summary / budget accounting / checkpoint selection artifact 放在同一路径，会被候选污染。现在隔离为：
  - `candidate_output/`：候选可写；
  - `runner_artifacts/`：runner 权威产物目录，候选不可写。
- Scope：8C 仍然只证明“隔离加载组件”可行，不等于 GPU 长训全链路已隔离。训练阶段 reward/obs/loss callable 的长程隔离仍是 8D/8E 接入问题。
