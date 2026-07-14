# 数据目录

本目录按数据生命周期组织：

- `external/`：外部原始数据、下载审计和来源许可。大型 raw replay 不提交 Git。
- `processed/`：从原始日志转换出的统一决策数据、数据审计和转换摘要。
- `reanalysis/`：V1 关键局面候选队列、搜索老师标签和摘要。
- `training/`：V1/V0/实际动作合并后的正式训练集及其可追溯 manifest。
- `deck_elites/`：历史卡组优化产生的候选牌表。
- 根目录 JSON：卡牌库、卡牌标签、目标/候选卡组和卡组评估元数据。

大体积 `processed/*.jsonl`、`reanalysis/*.jsonl` 和 `training/*.jsonl` 是可重建派生数据，默认被 Git 忽略；对应摘要 JSON、转换脚本和原始来源必须保留。

旧 `data/distill/`、`data/reanalysis_queue.json` 和 50 条 fixture smoke 训练产物已经移除。开发冒烟测试统一写到被忽略的 `artifacts/dev_smoke/`。
