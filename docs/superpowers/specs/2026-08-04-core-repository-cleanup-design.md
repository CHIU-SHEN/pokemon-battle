# 核心仓库清理设计

## 目标

将仓库收敛为一条可提交、可复现、可继续研究的 MCTS 主线，同时保留完整原始数据，删除可再生缓存、旧实验副本和过时交接包。

## 保留边界

- 完整保留 `data/`，包括约 39.5 GB 的原始回放与训练数据。
- 保留当前 Kaggle 提交包 `final_submissions/primary_budgeted_mcts_v1.tar.gz` 及校验文件。
- 保留 Flat Safe V0 压缩包作为无模型回滚方案。
- 保留 `artifacts/sl0_shared_full/best.pt` 以及 primary/reserve 两个 Adapter 的核心权重；路径不改名，以兼容现有构建脚本。
- 保留完整 MCTS 结果归档及最新 MCTS v2 服务器交接包，供复现和后续蒸馏使用。
- 保留当前源码、构建脚本和与主线相关的测试。

## 清理边界

- 删除仓库内 `.pytest_*` / `.pytest_tmp` 测试缓存。
- 删除已经合并的 `.worktrees/top2-mcts-pilot`。
- 删除 `experiments/`、`logs/` 中可再生运行输出。
- 删除 GRU、history、PPO、旧 Adapter、旧服务器结果和旧提交包。
- 将仍有解释价值但不再是当前操作入口的文档、报告归档到 `docs/archive/` 和 `reports/archive/`。
- 移除只验证已删除旧提交物的陈旧测试，保留当前 MCTS 包、核心模型与 Flat Safe 回滚测试。

## 文档权威顺序

1. `NEXT_STEPS.md`：唯一的立即行动入口。
2. `README.md`：仓库结构、当前提交包和快速验证。
3. `项目进度.md`：阶段结果和历史进度。
4. `reports/primary_budgeted_mcts_v1_submission_report.md`：当前候选的详细证据。
5. `docs/archive/`、`reports/archive/`：历史材料，只用于追溯。

## 验收条件

- `data/` 未被改动。
- 当前 Kaggle 包和 SHA-256 校验通过。
- 最新服务器交接包和 SHA-256 校验通过。
- 当前测试套件通过，Git diff 无空白错误。
- 根目录不再堆放多个相互竞争的交接说明。
- `NEXT_STEPS.md` 明确写出上传、观察、失败回退及下一轮服务器实验顺序。
