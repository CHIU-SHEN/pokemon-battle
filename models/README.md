# 模型目录

本目录不再存放脱离训练运行上下文的单独 checkpoint。旧 `models/best.pt` 和 50 条 smoke 数据训练出的 `v2_policy_linear.json` 均已移除。

`SL-0-shared` 正式冻结产物以 `artifacts/sl0_shared_full/` 为准，其中同时保存 `best.pt`、`last.pt`、`run_config.json` 和 `metrics.jsonl`；冻结 test 报告为 `reports/sl0_shared_test.json`。开发冒烟产物仍统一写入 `artifacts/dev_smoke/`，不用于模型晋级。
