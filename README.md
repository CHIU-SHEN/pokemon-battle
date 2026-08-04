# Pokémon TCG AI Battle

当前主线是 primary 卡组的预算型 belief-PUCT MCTS。立即行动只看
[`NEXT_STEPS.md`](NEXT_STEPS.md)，历史文档不再作为操作依据。

## 当前可提交文件

上传 `final_submissions/primary_budgeted_mcts_v1.tar.gz`。

SHA-256：`af58979d116e3db8072d49c368cdf36b6c7128e743b4deaf805076a8f57a81e0`

不要把 `server_uploads/` 中的交接包上传到 Kaggle。Flat Safe V0 zip 仅作为无模型
回滚资产保留。

## 已验证结果

- primary MCTS：400 局，261 胜、139 负、0 平，胜率 65.25%。
- Wilson 95% 下界 60.46%；平均决策 20.96 ms，p95 30.09 ms。
- 0 异常、0 非法动作。
- Kaggle raw-exec（没有 `__file__`）和顶层 `deck.csv` 布局均已修复。

详细证据见 [`reports/primary_budgeted_mcts_v1_submission_report.md`](reports/primary_budgeted_mcts_v1_submission_report.md)。

## 核心目录

完整维护边界见 [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)。

```text
submission/          当前规则代理和运行时源码
candidates/          MCTS/学习型候选实现
src/                 训练与数据处理源码
eval/                对局和统计工具
scripts/             构建、验证、训练与评估入口
tests/               当前回归测试
data/                完整原始数据与派生数据（保留，不清理）
artifacts/            当前核心权重与 MCTS 结果
final_submissions/    Kaggle 包和回滚包
server_uploads/       最新服务器交接包
reports/              当前证据；历史结果在 archive
docs/                 操作、研究和历史文档
```

## 本地验证

```powershell
conda activate pokemon-tcg
python scripts/verify_primary_budgeted_mcts_kaggle.py `
  final_submissions/primary_budgeted_mcts_v1.tar.gz
$testTemp = Join-Path $env:TEMP ("ptcg-tests-" + [guid]::NewGuid())
pytest -q --basetemp $testTemp
```

服务器说明见 [`docs/operations/TOP2_MCTS_SERVER_HANDOFF.md`](docs/operations/TOP2_MCTS_SERVER_HANDOFF.md)。

## 文档规则

- `NEXT_STEPS.md`：唯一近期行动列表。
- `项目进度.md`：已完成阶段和当前状态。
- `docs/operations/`：仍可执行的操作说明。
- `docs/archive/`、`reports/archive/`：历史追溯材料。
- `CLEANUP_REPORT.md`：本次保留、删除和验证记录。
