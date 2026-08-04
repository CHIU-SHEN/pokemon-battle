# Primary Budgeted MCTS V1 提交报告

## 可上传文件

Kaggle 候选：

```text
final_submissions/primary_budgeted_mcts_v1.tar.gz
```

SHA-256：

```text
6e118e346a405229eef114b142d21f429fcbdd0836eb1a6e66ccb340e375b5f6
```

旁车文件：`final_submissions/primary_budgeted_mcts_v1.tar.gz.sha256`。

该文件与 `server_uploads/pokemon-tcg-primary-budgeted-mcts-v1.tar.gz` 不同。后者是研发和
服务器交接包，含外层目录；前者是 Kaggle 上传包，`main.py` 与 `deck.csv` 位于归档顶层。

## 候选配置

- candidate：`crustle_kangaskhan_cage`
- deck：`top2-primary-crustle-kangaskhan-cage-v1`
- shared checkpoint：`artifacts/sl0_shared_full/best.pt`
- adapter checkpoint：`artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt`
- simulations：8
- belief particles：1
- max depth：4
- 单决策搜索预算：30ms
- 整局累计搜索预算：2s
- 默认设备：CPU

## 正式门结果

400 局 swapped-seat 对局结果：

| 指标 | 结果 | 门槛 |
| --- | ---: | ---: |
| 胜 / 负 / 平 | 261 / 139 / 0 | 400 局 |
| 胜率 | 65.25% | ≥55% |
| Wilson 95% 下界 | 60.46% | 参考证据 |
| 平均决策延迟 | 20.96ms | 参考证据 |
| p95 决策延迟 | 30.09ms | ≤35ms |
| 异常 | 0 | 0 |
| 非法动作 | 0 | 0 |

动作来源累计为：MCTS 20,265 次、策略回退 2,033 次、整局预算回退 28 次、单决策截止
回退 3 次。提交门禁结论为 `all_submission_gates_passed`。

## 构建与独立验证

- 构建脚本：`scripts/build_primary_budgeted_mcts_kaggle.py`
- 验证脚本：`scripts/verify_primary_budgeted_mcts_kaggle.py`
- 归档内 manifest：`KAGGLE_MANIFEST.json`
- 独立解压验证：39 个 manifest 文件哈希全部通过。
- 顶层 `deck.csv`：60 张。
- `agent(None)`：返回 60 张。
- Kaggle raw-exec（globals 中不提供 `__file__`）：返回 60 张。
- 运行时懒加载：成功创建 `Top2BeliefPUCTAgent`。
- reserve/Petrel、原始 MCTS 对局、训练数据和 JSONL：未包含。

## 风险与上线步骤

本地验证证明结构、权重、入口和 CPU 运行时完整，但不能替代 Kaggle 线上 Validation
Episode。该候选依赖 PyTorch；公开竞赛说明确认 `.tar.gz` 顶层结构要求，但未明确列出线上
环境的全部 Python 包版本。因此首次上传后必须检查 self-play 验证状态和日志。

若 Validation Episode 通过，可继续观察 ladder；若出现缺少 PyTorch、导入失败或超时，停止
晋级并保留历史 Flat Safe V0 回滚路径。构建过程没有修改 `submission/`，也没有自动上传。
