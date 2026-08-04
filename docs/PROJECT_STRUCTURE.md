# 项目目录结构

- `submission/`：稳定 Kaggle 代理运行时。
- `candidates/`：MCTS 与学习型候选。
- `src/`：训练、数据与模型库代码。
- `eval/`、`scripts/`、`jobs/`：评估、本地命令与服务器任务。
- `tests/`：本项目测试；pytest 只从这里收集。
- `config/`、`baselines/`：配置与固定基线。
- `data/`：完整数据，禁止清理。
- `artifacts/`：核心权重与 MCTS 结果。
- `final_submissions/`：Kaggle 包和回滚包。
- `server_uploads/`：服务器交接包，不能上传 Kaggle。
- `reports/`：当前报告；历史内容在 `reports/archive/`。
- `docs/operations/`：当前操作说明；`docs/archive/`：历史资料。
- `.local_cache/`：Git 忽略的旧 pytest/worktree 本地缓存，不参与构建。

不要移动代码目录；现有 Python 导入、测试和服务器交接清单依赖这些稳定路径。
