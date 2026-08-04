# 核心仓库清理报告（2026-08-04）

仓库已收敛为预算型 MCTS 主线。

## 保留

- 完整 `data/`。
- 当前 Kaggle MCTS 包、SHA-256 和 Flat Safe V0 zip。
- SL0 共享 best、Top2 Adapter best、完整 MCTS 结果和 MCTS v2 服务器包。

## 删除与归档

- 删除旧实验、日志、GRU/history/PPO、非 Top2 Adapter、旧提交包和重复交接包。
- 删除只服务旧产物的 GRU/SL0/Arena/RL/Selfplay 构建入口及测试。
- 旧交接说明归档到 `docs/archive/handoffs/`。
- 清理前入口归档到 `docs/archive/pre-cleanup/`，历史报告归档到 `reports/archive/`。

## 验证

- Kaggle SHA-256：`af58979d116e3db8072d49c368cdf36b6c7128e743b4deaf805076a8f57a81e0`。
- 独立解压：`ok=true`、60 张牌、40 个 manifest 文件。
- 服务器 SHA-256：`fa0194529a554f5fcf1e3e03f2c2a8a1ea04bc8d653043b5fd14d2655264aa19`。
- 项目测试使用唯一仓库外 `--basetemp` 执行。

## 已知限制

根目录 `.pytest_*` 与孤立 `.worktrees/` 约 2.4 GB 因 Windows ACL 和安全审查未被
强制删除。它们不再由 Git 登记、不参与构建且已被 `.gitignore` 排除。

下一步只看根目录 `NEXT_STEPS.md`。
