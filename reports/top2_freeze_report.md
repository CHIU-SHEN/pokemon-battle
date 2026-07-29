# Top2 Arena 筛选结果

- 决策：已选出 Top2；仅冻结筛选结果，不启动训练，不覆盖 `submission/deck.csv`。
- 总对局数：10,100（首轮 8,500 + 前四复赛 1,200 + 前二决赛 400）。
- 全部阶段：0 失败、0 非法动作，Top2 顺序三阶段一致。

## 最终角色

1. Primary：`crustle_kangaskhan_cage`
2. Reserve：`crustle_kangaskhan_petrel`

决赛中 Primary 对 Reserve 的双座次合计胜分率为 59.25%，战绩为 237:163；95% Wilson 区间为 54.37%–63.96%。

## 重要边界

- 本地缺少 Sample baseline 源码；外部矩阵实际使用 Random、Exploiter-FirstMin、V0-current 和 V0-best。
- 对战引擎不暴露 RNG seed，已通过交换先后手和扩大样本降低偏差，但不属于严格成对 seed 实验。
- 后续训练地点与流程由项目负责人另行决定。
