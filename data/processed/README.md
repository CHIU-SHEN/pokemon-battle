# 统一决策数据

主要产物：

- `bad_case_decisions.jsonl`：600 局本地失败日志转换出的 20,232 条决策，其中目标卡组行动方带 V0 标签。
- `kaggle_decisions.jsonl`：第一批 726 局 Kaggle replay 转换出的 100,141 条跨卡组决策。
- `*_summary.json`：转换数量、错误、分组和老师状态摘要。
- `local_data_audit.json`、`target_deck_profile.json`、`bad_case_index.json`：可追溯审计产物。

所有训练/验证/测试划分均按完整 `game_id` 固定，禁止按单条决策重新随机划分。
