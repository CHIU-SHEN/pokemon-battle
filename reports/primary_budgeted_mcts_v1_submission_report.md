# Primary Budgeted MCTS V1 提交报告

## 当前候选

- 文件：`final_submissions/primary_budgeted_mcts_v1.tar.gz`
- SHA-256：`af58979d116e3db8072d49c368cdf36b6c7128e743b4deaf805076a8f57a81e0`
- primary：`crustle_kangaskhan_cage`
- shared：`artifacts/sl0_shared_full/best.pt`
- adapter：`artifacts/adapters_top10/crustle_kangaskhan_cage/best.pt`
- 预算：8 simulations、8 particles、depth 4、单决策 30 ms、整局 2 s。

## 400 局正式门

| 指标 | 结果 |
| --- | ---: |
| 胜 / 负 / 平 | 261 / 139 / 0 |
| 胜率 | 65.25% |
| Wilson 95% 下界 | 60.46% |
| 平均决策 | 20.96 ms |
| p95 决策 | 30.09 ms |
| 异常 / 非法动作 | 0 / 0 |

结论：本地门控通过，但仍需 Kaggle Validation Episode 验证线上依赖和时限。

## Kaggle 兼容修复

- `main.py` 在 raw-exec 不提供 `__file__` 时可定位运行资产。
- 顶层直接包含 `deck.csv`；初始牌组请求不依赖嵌套模型目录可见。
- 顶层直接包含 `main.py` 与 `cg/`，没有外层包装目录。
- manifest 40 个文件校验通过，完整 runtime 可创建 `Top2BeliefPUCTAgent`。

下一步以根目录 `NEXT_STEPS.md` 为唯一入口。
