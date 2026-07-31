# 报告目录

- `final_freeze_report.json`：当前唯一基线提交包的冻结报告。
- `sl0_shared_test.json`：冻结 `SL-0-shared` 的详细 test 基线。
- `sl1_gru_test.json`：GRU best checkpoint 的首次总体 test 报告；正式晋级前须生成详细复评报告。
- `sl1_gru_analysis.md`：GRU 与 SL-0 的指标对比、限制和下一步晋级计划。
- `top10_adapter_v3_base_seed20260722_run1_review.md`：2026-07-28 回收包的 SHA、日志、离线指标和冻结版本差异复核；结论是不整体覆盖当前冻结 Adapter。
- `top10_adapter_online_smoke.md` / `.json`：10 套在线 Adapter 的 Random + mirror 共 200 局工程验收；10/10 通过、0 异常、0 非法动作。
- `top2_arena_report.json`：Top10 初赛和可用外部基线矩阵，共 8,500 局。
- `top4_playoff_report.json`：前四复赛，共 1,200 局。
- `top2_final_report.json`：前二决赛，共 400 局。
- `top2_freeze_report.md` / `.json`：10,100 局后的 Top2 选择、三阶段指标、牌表/Adapter/报告哈希与已知限制；当前 primary 为 `crustle_kangaskhan_cage`，reserve 为 `crustle_kangaskhan_petrel`。
- 2026-07-30 服务器 conservative PPO 是固定对手的一次性 pilot，不是循环自博弈；真正的双分支门控式自博弈状态与服务器执行见 `TOP2_SELFPLAY_SERVER_HANDOFF.md`。
- 后续正式训练和固定评估报告应记录数据版本、模型版本、随机种子、对手矩阵和晋级结论。

历史 M0～M6 说明已归档到 `docs/archive/mvp/`。
