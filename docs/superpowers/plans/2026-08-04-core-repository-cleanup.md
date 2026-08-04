# Core Repository Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 收敛仓库为可提交的 MCTS 主线，同时保留完整数据和后续蒸馏所需的最小核心资产。

**Architecture:** 源码与数据保持原位；当前提交物和核心权重保留稳定路径；历史说明移入 archive；可再生产物直接删除。`NEXT_STEPS.md` 成为唯一操作入口。

**Tech Stack:** Git、PowerShell、Python、pytest、tar/zip、SHA-256

## Global Constraints

- 完整保留 `data/`，不得移动、裁剪或删除。
- 不修改 `.git/`。
- 保留当前 Kaggle MCTS 包、Flat Safe V0 回滚包、MCTS 完整结果和 MCTS v2 交接包。
- 保持当前构建器使用的 `artifacts/sl0_shared_full` 与 `artifacts/adapters_top10` 路径。
- 删除前先核对 Git 跟踪状态和源码引用。

---

### Task 1: 固化目录规则和下一步入口

**Files:**
- Create: `NEXT_STEPS.md`
- Modify: `README.md`
- Modify: `项目进度.md`
- Create: `docs/README.md`
- Create: `artifacts/README.md`
- Create: `final_submissions/README.md`
- Create: `server_uploads/README.md`

- [ ] 写清当前唯一 Kaggle 包、校验值、服务器包用途和下一轮蒸馏路线。
- [ ] 将 README 的历史包清单替换为当前目录结构。
- [ ] 更新项目进度中的最新 Kaggle 修复与清理状态。

### Task 2: 归档历史文档和报告

**Files:**
- Move: 根目录旧 handoff 文档到 `docs/archive/handoffs/`
- Move: 非当前操作文档到 `docs/archive/legacy/`
- Move: 非当前评估报告到 `reports/archive/`

- [ ] 保留 `TOP2_MCTS_SERVER_HANDOFF.md` 的可复现副本，并同步构建脚本引用。
- [ ] 当前报告只保留 MCTS 提交报告与必要的机器可读证据。
- [ ] 用 `rg` 检查移动后的失效链接和路径引用。

### Task 3: 删除可再生与旧版产物

**Files:**
- Delete: `.pytest_*`, `.pytest_tmp`, `.worktrees/top2-mcts-pilot`
- Delete: `experiments/`, `logs/`
- Prune: `artifacts/`, `final_submissions/`, `server_uploads/`

- [ ] 解析并验证所有目标绝对路径均位于仓库内，且不包含 `data/` 或 `.git/`。
- [ ] 删除测试缓存和已合并 worktree。
- [ ] 删除旧模型、旧提交包、旧服务器包，只保留设计规定的核心文件。

### Task 4: 收敛测试并验证

**Files:**
- Modify/Delete: 仅针对已删除旧提交物的陈旧测试
- Create: `docs/cleanup-reports/REPO_CLEANUP_2026-08-04_CORE.md`

- [ ] 校验当前 Kaggle 包和服务器包 SHA-256。
- [ ] 在仓库外临时目录运行全量 pytest，避免重新制造根目录缓存。
- [ ] 运行 `git diff --check` 和 `git status --short`。
- [ ] 在清理报告记录删除范围、保留资产、验证结果和后续行动。
