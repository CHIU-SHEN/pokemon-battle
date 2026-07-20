# 模型目录

本目录不再存放脱离训练运行上下文的单独 checkpoint。旧 `models/best.pt` 和 50 条 smoke 数据训练出的 `v2_policy_linear.json` 均已移除。

`SL-0-shared` 正式冻结产物以 `artifacts/sl0_shared_full/` 为准，其中同时保存 `best.pt`、`last.pt`、`run_config.json` 和 `metrics.jsonl`；冻结 test 报告为 `reports/sl0_shared_test.json`。

`SL-1-gru` 首次服务器全量产物位于 `artifacts/sl1_gru_full/`：`best.pt` 为 epoch 1 的最低 valid loss checkpoint，`last.pt` 为 epoch 5 checkpoint，原始服务器回收包保留为 `server_export.zip`。首次冻结 test 报告为 `reports/sl1_gru_test.json`；该旧版报告只有总体指标，正式晋级前须用新版 `src/train/eval_sequence.py` 对 `best.pt` 和 `last.pt` 重新生成包含非强制单选、来源分组、非法 top-1 和推理吞吐的报告。

开发冒烟产物仍统一写入 `artifacts/dev_smoke/`，不用于模型晋级。
