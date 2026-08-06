# V3 S16 权威候选与 Kaggle 低预算候选设计

日期：2026-08-06

## 决策

以 V3 epoch 44 `best_safe_arena.pt` 为唯一学生权重，同时产出两个互不覆盖的候选：

1. 权威 S16：16 simulations、3 particles、depth 10、单决策 0.25 秒、整局 120 秒。
2. Kaggle S16-60ms：16 simulations、3 particles、depth 10、单决策 0.06 秒、整局 3 秒。

权威版本严格复现 400 局 295/105、73.75% 胜率的配置。低预算版本只收紧 deadline，
不减少 simulations、particles 或 depth；其依据是权威评估平均决策 36.25 ms、P95 50.16 ms。

## 构建边界

- 两个包必须使用不同名称、manifest schema 和输出文件，禁止相互覆盖。
- 两个包都必须携带并显式加载 V3 `best_safe_arena.pt`；不得继续隐式使用旧 Top10 Adapter。
- 继续携带共享主干、牌组、卡牌数据、搜索后端及合法动作 fallback。
- `formal_submission_replacement_authorized` 保持 `false`。
- 低预算包在通过新门控前不得标记为 Kaggle-ready。

## 归档布局

下载到根目录的两个服务器结果包及校验文件归档到：

```text
artifacts/mcts_teacher_v3/primary-5k/archives/
```

解压后的训练资产保留在：

```text
artifacts/mcts_teacher_v3/primary-5k/train/
```

正式评估 JSON 复制到 `reports/`，使用包含 `v3`、`s8`/`s16` 和局数的稳定名称。
运行 progress 和临时 smoke 输出不进入正式报告。

## 验证

权威 S16 包必须通过：

- archive SHA-256 和 manifest 文件哈希；
- checkpoint schema、candidate ID、deck ID 和 source epoch 44 校验；
- raw-exec 无 `__file__` 加载；
- 顶层 `deck.csv`、60 张牌和完整 runtime 创建；
- 小规模对局安全 smoke。

Kaggle S16-60ms 除上述验证外，还必须重新运行 100 局门控。只有满足以下条件才能进入
400 局正式评估：

- exceptions=0；
- illegal_actions=0；
- fallback_rate=0；
- 有效胜率至少 53%；
- P95 决策耗时不高于 60 ms。

400 局仍满足相同安全门、胜率目标和延迟预算后，才可显式决定是否标记为 Kaggle-ready。

## 失败样本迭代（下一阶段）

本轮不重新训练。后续允许采用 DAgger 式失败样本迭代：

1. 从 S16 实战中收集失败或低置信度状态，并按超时、belief 偏差、价值误判、策略误判和
   fallback 类型分类。
2. 使用冻结的 S128 教师对这些状态重新生成 visit distribution 和 value 标签。
3. 将重新标注样本与原始 5,000 局数据按受控比例混合，避免只记忆少数失败局。
4. 使用独立、未参与训练的种子和对局做 holdout 与 400 局门控。
5. 只有同时改善强度且不破坏延迟、安全门时，才替换当前 epoch 44 checkpoint。

禁止针对单局对手或评估 seed 编写硬编码动作规则，也禁止使用同一批失败局同时训练和证明提升。
