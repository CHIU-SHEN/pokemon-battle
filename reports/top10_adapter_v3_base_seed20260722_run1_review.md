# Top10 Adapter v3 base 结果包复核

> 复核日期：2026-07-29  
> 来源包：`artifacts/top10-adapters-v3-base-seed20260722-run1.tar.gz`

## 结论

该结果包完整且离线评估合格，但**不应整体覆盖当前冻结的
`artifacts/adapters_top10/`**，也不能视为一次新的独立 seed 复现。

- 外层压缩包 SHA-256 为
  `8C31168018365B47CC994AD435D120E7E9B4E91D7F8A60E7DEA28843656BAC53`，
  与旁车文件一致。
- 包内 SHA 清单共 31 项，复核结果为 31/31 一致。
- 正式 Slurm 任务 `top10-adapters.27953` 完成 10 套训练与统一 test 评估；
  10/10 为 `advance`，非法预测总数为 0。
- batch=1 在线前向 p95 为 1.651～1.664ms，远低于 50ms 门槛。
- 三次较早任务因服务器 `/etc/bashrc` 中未定义 `BASHRCSOURCED` 而退出；
  最终任务无 stderr 且产物完整，这些早期失败不影响正式结果。

## 与当前冻结版本的关系

逐文件比较 `best.pt` 后：

- 9/10 个 checkpoint 与当前 `artifacts/adapters_top10/` 的 SHA-256 完全相同；
- 只有 `alakazam_battle_cage_split` 不同；
- 新包中的该候选是基础数据版本，exact test 仅 117 条，Top-1 变化为
  0.00pp；当前冻结版本已纳入按整局隔离的补充 test，共 452 条，Top-1
  相对主干提升 17.37pp。

因此，本包证明了基础版本结果和服务器执行链路可回收，但没有为其余 9 套
提供新的独立随机种子证据。`alakazam_battle_cage_split` 继续保留当前已完成
补充数据复评的冻结 checkpoint；其余 9 套无需替换，因为文件完全相同。

## 阶段判断

本包复核完成时项目仍位于 Top2 Arena 的入口门槛。随后已实现
`adapter:<candidate_id>` 在线代理入口，并于 2026-07-29 完成 10 套 Random +
mirror 共 200 局 smoke：10/10 通过、0 异常、0 非法动作。详见
`reports/top10_adapter_online_smoke.md`。当前剩余阻塞是正式 45 组循环赛编排与
外部矩阵，而不是在线推理接入。

后续顺序保持不变：

1. ~~实现并验证 10 套候选在线代理~~（已完成）；
2. ~~每套完成 Random 10 局和镜像 10 局 smoke~~（已完成）；
3. 运行 45 组内部循环赛、固定外部矩阵、前四复赛和前二决赛；
4. 按 `data/high_score_decks/top2_selection_policy.json` 冻结 primary/reserve。

筛选阶段不得覆盖 `submission/deck.csv`。
