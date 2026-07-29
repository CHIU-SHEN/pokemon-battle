# Top2 本地 PPO Pilot 报告

- 状态：`completed_preliminary`
- 运行 ID：`20260729T064313Z`
- 预算档：`full`
- 实际总时长：311.39 秒
- 推荐组：`conservative`
- 结论性质：preliminary，仅作为服务器正式训练起点

## Rollout

| 分支 | 局数 | 决策 | train | valid | test | 异常 | 非法动作 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| primary | 100 | 4430 | 79 | 12 | 9 | 0 | 0 |
| reserve | 100 | 4705 | 84 | 12 | 4 | 0 | 0 |

## Primary trials

| 组别 | eligible | Arena | KL | value MSE | PPO 秒 |
| --- | --- | ---: | ---: | ---: | ---: |
| conservative | True | 96/200 | 0.000579 | 0.217113 | 8.85 |
| baseline | True | 89/200 | 0.001715 | 0.206420 | 8.03 |
| exploratory | True | 100/200 | 0.003058 | 0.195159 | 8.50 |

## 服务器下一步

使用 `config/top2_rl_selected.json` 作为起点重新采集 on-policy rollout，扩大 V1 与 PPO；不得把本地 100 局结果当成正式晋级证据。
