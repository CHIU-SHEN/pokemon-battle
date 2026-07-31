# Top2 隐信息 PUCT-MCTS 小规模验证设计

## 背景

双分支门控自博弈已完成三轮服务器试验。所有候选均未晋级，Arena 胜率持续在 50% 左右。PPO 训练后的参考策略 KL 约为 0.002，动作一致率约为 96%。

当前 PPO 只在最后一个决策写入终局 `+1/-1`，再以 `gamma=0.99`、`lambda=0.95` 的 GAE 向前传播。平均约 100 个决策时，早期终局信号衰减到约 `0.9405^100 ≈ 0.0022`。下一阶段停止未经验证的 PPO 第四、第五轮，改做接近 AlphaZero 的隐信息 PUCT-MCTS 小规模验证。

## 目标与范围

第一版目标：

- 当前 policy head 提供合法动作先验。
- 当前 value head 或终局胜负评估叶节点。
- 官方 `SearchBegin/SearchStep/SearchRelease/SearchEnd` 管理分支状态。
- belief particles 表示不可见牌库、奖赏卡、手牌和盖放宝可梦。
- PUCT 根访问次数生成策略训练目标。
- 整局胜负直接作为该局全部搜索样本的价值目标，不使用 GAE 衰减。
- primary/reserve 各约 200 局，每个关键决策 32 次模拟。

第一版不直接运行 3,000 局，不替换正式 submission，不自动晋级 best，也不同时修改牌组、特征编码器或监督主干。

## 数据流

```text
observation
  → 合法联合动作候选
  → 3 个 belief particles
  → policy/value 先验
  → PUCT 选择与 SearchStep 扩展
  → 叶节点价值反向传播
  → 根访问次数分布 π
  → 选择实际动作并保存公开训练样本
  → 终局结果 z 回填
  → policy CE(π) + value MSE(z) + reference KL
  → candidate
  → swapped-seat Arena 与安全回归
```

## 隐信息与搜索状态

官方 Search API 是唯一分支模拟入口。不得复制 ctypes 指针，也不得通过重放真实对局动作伪造搜索树。

每个根决策从 observation 和 `GameLedger` 构造 3 个合法 belief particles，每个 particle 独立 `SearchBegin`。采样到的隐藏卡牌不得写入模型特征、训练样本或报告。报告只保存粒子数量、有效率、节点统计和回退原因。

搜索正常或异常结束都必须释放子状态并调用 `SearchEnd`。没有 `search_begin_input`、粒子全部无效或 API 出错时，安全回退到当前策略动作，不生成伪造的 MCTS policy target。

## 动作空间

搜索复用现有 `ActionGenerator` 处理联合动作：

- 最多保留 8 个合法联合动作。
- 当前策略动作必须包含在候选集中。
- 少于 2 个候选时不搜索。
- 每次 `SearchStep` 前再次验证合法性。
- 单选动作先验直接来自 policy logits。
- 多选动作先验由所选 option 的对数概率和归一化，并使用固定长度校正。

## PUCT

每条边维护 prior `P`、访问次数 `N`、价值和 `W`、均值 `Q=W/N`：

```text
score(a) = Q(a) + c_puct * P(a) * sqrt(sum_b N(b)) / (1 + N(a))
```

初始参数：

- 32 simulations/关键决策。
- 最大深度 8 个选择节点。
- `c_puct=1.5`。
- 3 个 particles，round-robin 分配模拟预算。
- 自博弈根节点 Dirichlet `alpha=0.3`、`epsilon=0.25`。
- Arena 关闭噪声。
- 终局叶节点值为 `+1/0/-1`。
- 非终局叶节点值来自 value head 并裁剪到 `[-1,1]`。
- 行棋方切换时价值变号，同一玩家连续选择时不变号。

不同粒子的根边访问数和价值和相加，再生成聚合访问分布。

## 搜索范围与动作温度

只在有 `search_begin_input`、合法联合动作至少 2 个、且属于经过 smoke 验证的策略上下文时搜索。纯排序、强制确认和无策略含义选择直接使用当前策略。

- 每局前 20 个可搜索决策按访问次数 `tau=1.0` 采样。
- 后续使用 `tau=0.25`。
- Arena 使用访问次数最大动作。

## 训练样本与目标

每个有效样本保存 branch、deck_id、iteration_id、game_id、step、公开模型输入、合法 mask、联合动作候选、visit counts、策略目标 `π`、根价值、节点统计、best checkpoint SHA 与终局 `z`。

隐藏状态补全不得落盘。按整局稳定划分 train/valid/test，禁止一局跨 split。

候选从各分支当前 best 初始化，只训练现有 Adapter、policy delta 和 value delta：

```text
L = policy_cross_entropy(π, p)
  + 1.0 * mse(v, z)
  + 0.02 * KL(p || p_reference)
  - 0.005 * entropy(p)
```

学习率 `1e-4`，最多 6 epoch，按 valid policy CE 与 value MSE 早停。policy target 是访问次数分布而非执行动作 one-hot，value target 对该局所有有效搜索样本均为终局 `z`。

## 小规模试验

每个分支：

- 先运行 10 局 smoke。
- smoke 通过后采集约 200 局。
- 32 simulations、3 particles、深度 8。
- 训练一个 MCTS candidate。
- 先做 50 局 smoke Arena。
- 安全后做至少 400 局 swapped-seat candidate-vs-best Arena。

同时比较：

1. 纯当前 best。
2. 当前 best + MCTS，但不训练。
3. 由 MCTS 访问分布训练的 candidate。

这样可以区分搜索本身、搜索蒸馏和价值网络三个环节。

## 验收与停止条件

安全门：

- Search API 异常为 0。
- 非法动作为 0。
- 搜索回退率低于 5%。
- primary/reserve checkpoint、牌组和数据严格隔离。
- 搜索状态无泄漏。
- 记录单决策 p50/p95、节点吞吐和预计正式成本。

扩大到 3,000 局前必须满足：

- `best + MCTS` 对纯 best 的 400 局非平局胜率至少 55%。
- candidate 对纯 best 的 400 局至少 53%；若进入灰区，追加到 1,000 局后至少 55%。
- candidate 的策略 CE 与 value MSE 均不劣于初始 best。
- 0 异常、0 非法动作。

若搜索代理本身不能超过纯 best，则停止训练，修正叶节点价值或 belief particles。若搜索能提升但 candidate 不能复现，则调整蒸馏目标或模型容量，不增加自博弈局数。

## 测试与可恢复性

单元测试覆盖 PUCT 选择、零访问边、行棋方价值符号、联合动作先验、visit 分布、噪声开关、深度/节点/时间预算、粒子聚合、隐藏信息不落盘、状态释放和 fallback。

集成测试覆盖真实 Search API 连续搜索、单分支 2 局采集/1 epoch 训练/2 局 Arena、分支交叉 checkpoint 拒绝、candidate 回滚和 `--resume`。

服务器每 10 局写入完成数、节点数、回退率、吞吐和预计剩余时间；阶段性原子落盘。第一分支出现安全异常时不得自动扩大另一分支。

## 已回收 PPO 证据

`artifacts/top2-ppo-3round-evidence.tar.gz` 已校验 SHA256：

```text
85728312efdd8e0b529df650231dad6b0494be3470843a3cb7d9857cb2ab44e8
```

包内含三轮双分支日志、状态、候选 checkpoint、holdout、回归、iteration report、配置和服务器门控修复，共 93 个条目；不含原始 rollout。正式 best 已存在本地冻结交接包，服务器可以释放。
