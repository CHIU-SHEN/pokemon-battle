# Top2 PPO 结果归档与仓库整理（2026-07-31）

## 结果处理

- 校验服务器回传包 SHA-256，与旁车文件一致。
- 生成 `reports/top2_server_ppo_20260730.md/.json` 作为紧凑、可追踪的评估入口。
- 原始回传包和旁车哈希归档到 `artifacts/archive/server-results/2026-07-30-top2-ppo/`，保留完整证据但不纳入 Git。

## 清理

- 将 `codex/top2-data-handoff` 的 8 个已完成提交快进合入本地 `main`。
- 移除已合入的临时工作树 `artifacts/worktrees/top2-data-handoff/` 和对应本地分支，释放约 381 MB。
- 删除已经完成上传用途且可由构建脚本重建的 Arena v1/v2、RL v1/v2 服务器交接压缩包及旁车哈希，共约 28.2 MB。
- 保留冻结模型、正式牌表、历史评估、最终提交包、训练数据和当前服务器原始回传包。

## 安全边界

- 未修改 `submission/deck.csv`。
- 未删除冻结 checkpoint 或训练数据。
- 服务器 PPO 候选未晋级，初始 Top2 Adapter 继续作为冻结版本。
