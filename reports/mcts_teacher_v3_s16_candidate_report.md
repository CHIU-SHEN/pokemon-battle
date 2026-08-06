# MCTS Teacher V3 S16 候选报告

日期：2026-08-06

## 训练与评估

- 强教师：302/98/0，胜率 75.50%，Wilson 下界 71.06%，安全指标全零。
- V3 数据：5,000 局、199,199 个训练样本、23,421 个 holdout 样本。
- 训练在 epoch 46 因连续两轮 holdout policy 恶化停止；冻结使用 epoch 44 `best_safe`。
- epoch 44 holdout policy/value/reference-KL：1.255514 / 0.699567 / 0.025036。

| 候选 | 局数 | 胜/负/平 | 有效胜率 | Wilson 下界 | P95 决策 | 安全指标 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| S0 | 100 | 44/56/0 | 44.00% | 34.67% | 2.83 ms | 全零 |
| S8 | 400 | 271/128/1 | 67.92% | 63.19% | 33.41 ms | 全零 |
| S16 | 400 | 295/105/0 | 73.75% | 69.23% | 50.16 ms | 全零 |
| S128 教师 | 400 | 302/98/0 | 75.50% | 71.06% | 251.63 ms | 全零 |

S16 是权威主候选；S8 保留为低延迟备用。S0 淘汰，S128 保留为教师和能力上界。

## 交付包

- 权威 S16：`final_submissions/pokemon-tcg-v3-s16-authority.tar.gz`
  - SHA-256：`855e16d9a4a22ac04b23dfd4d8c1e1dc5ae541f46285c9f55bd5609c2ddd226e`
  - 预算：16 simulations、3 particles、depth 10、250 ms/决策、120 s/局。
- Kaggle S16-60ms：`final_submissions/pokemon-tcg-v3-s16-kaggle-60ms.tar.gz`
  - SHA-256：`533bee35187a3ff9d6cbf1d6608cc12950fe030b206a6fd9e213b38d8b19234a`
  - 预算：16 simulations、3 particles、depth 10、60 ms/决策、5 s/局。
  - 状态：`kaggle_upload_ready=false`，尚未授权正式替换。

两个包的 39 个 manifest 文件哈希、60 张顶层牌组、epoch 44 checkpoint 身份均已验证。
Kaggle 包在不提供 `__file__` 的 raw-exec 环境中成功定位资产并创建完整 S16 runtime。

## 60ms 本地 smoke

最初的 3 秒整局预算在 10 局中产生 15 次 `mcts_game_budget_fallback`。评估器此前只统计
`mcts_fallback`，现已修正为统计所有 `mcts_*fallback` 来源。

调整为 60 ms/决策、5 s/局后重新运行 10 局：7/3/0，P95 53.65 ms，exceptions、
illegal actions 和完整 fallback rate 均为零，action sources 中无任何 MCTS fallback。

## 下一门控

不得直接上传 Kaggle。先在服务器使用低预算包的精确配置跑 100 局；只有安全指标全零、
有效胜率至少 53%、P95 不高于 60 ms，才扩到 400 局。400 局再次通过后，才可显式修改
readiness 并进入 Kaggle Validation Episode。
