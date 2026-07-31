# Top2 首轮服务器 PPO 结果评估

> Run ID：`20260730T081415Z`  
> 评估日期：2026-07-31  
> 结论：训练流程通过，候选 checkpoint 不晋级，不授权替换正式提交。

## 结果

| 分支 | 训练决策 | Holdout | 动作一致率 | Arena（候选 vs 初始） | 判断 |
| --- | ---: | ---: | ---: | ---: | --- |
| primary / `crustle_kangaskhan_cage` | 120,251 | 28,389 | 97.26% | 193-205-2，48.25% | 不晋级 |
| reserve / `crustle_kangaskhan_petrel` | 118,115 | 28,468 | 97.97% | 199-199-2，49.75% | 不晋级 |

两条分支各完成 3 epoch。末轮近似 KL 分别为 0.000879 和 0.000741，clip fraction 分别为 3.17% 和 2.21%，远低于安全上限；holdout 均为 0 非法 argmax，800 局 Arena 为 0 异常、0 非法动作。说明实现、数据隔离和保守更新机制是可靠的。

候选 action accuracy 相对初始 Adapter 分别下降 0.078 和 0.039 个百分点，基本持平；value MSE 明显下降，但没有转化为 Arena 胜率。primary 略负于初始 Adapter，reserve 完全持平，且比赛引擎不支持严格 RNG seed 控制，因此没有证据支持晋级。

## 决策与下一步

- 保持冻结的初始 Top2 Adapter 和 `submission/deck.csv` 不变。
- 不继续用同一批数据、同一配置堆叠 epoch，避免在无实战收益的方向上扩大更新。
- 下一轮先补异构对手、独立重复和 Top2 专属 V1 低置信/失利局面重分析。
- 预先定义晋级门槛，并至少加入历史 best、交叉 Top2、Random/FirstMin 及异构牌组矩阵；只有回归集与 Arena 同时通过才发布。

## 原始证据

- 归档包：`artifacts/archive/server-results/2026-07-30-top2-ppo/top2-training-results-20260730T081415Z.tar.gz`
- SHA-256：`c39c4b841d7b873a5e20e002c3c245e4f5e5e9839ed5c98c0d717e042f3cad0c`
- 包内保留训练 summary、holdout、两组 400 局 Arena、checkpoint、日志与交接 manifest。
