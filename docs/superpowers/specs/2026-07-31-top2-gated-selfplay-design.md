# Top2 双分支门控式自博弈设计

## 目标

为 `crustle_kangaskhan_cage`（primary）和
`crustle_kangaskhan_petrel`（reserve）建立两条完全独立、可恢复、可审计的
masked PPO 自博弈循环。系统采用 AlphaGo 式的迭代结构，但第一版不引入
MCTS：当前 best 生成新轨迹，训练 candidate，经门控 Arena 验证后才允许替换
best。MCTS 作为后续增强，不属于本次范围。

当前已有的服务器 PPO 结果只是一轮固定初始 Adapter 对固定对手的数据采集与
训练，不构成自博弈闭环，也不能计为本设计的完成进度。

## 分支隔离

primary 和 reserve 分别维护独立的：

- `deck_id`、牌表和冻结初始 Adapter；
- `state.json` 和迭代编号；
- best、candidate、history checkpoint；
- rollout、train/valid/test 和永久回归集；
- Arena 报告、哈希与晋级记录。

两条分支不得混用训练轨迹、checkpoint 或永久回归集。跨 Top2 对战默认只用于
外部评估，不回流训练。只有配置显式开启时，才允许将有限比例的跨牌组轨迹加入
训练，并且必须在 manifest 中单独标记。

## 目录与状态

```text
selfplay/
  primary/
    state.json
    best/
    candidate/
    history/
    iterations/<iteration_id>/
      rollout/
      train/
      holdout/
      arena/
      reports/
  reserve/
    ...
```

`state.json` 至少记录：

- schema 版本、branch、deck_id 和当前迭代号；
- 当前 best 的路径、SHA-256、来源迭代和晋级指标；
- history 清单及其只读哈希；
- 当前迭代阶段和最后一个成功检查点；
- candidate 路径、训练数据 manifest 与状态；
- 所有阶段的开始/结束时间、退出状态和恢复信息。

状态更新使用临时文件加原子替换。重复执行已成功阶段必须是幂等的；中断恢复时
不得重复晋级、覆盖 best 或把同一批轨迹重复训练。

## 单轮数据流

每条分支的一轮流程为：

1. 读取并校验当前 best、history 和冻结配置。
2. 从对手池采样并交换先后手，生成全新的 on-policy 轨迹。
3. 按完整 `game_id` 稳定拆分 80% train、10% valid、10% test。
4. test/valid 作为本轮 holdout，不参与 PPO；永久回归集继续保持只读。
5. 仅从本轮 train 的低置信、失利和胜负转折状态生成 V1 重分析队列。
6. 从当前 best 初始化 candidate，执行保守 masked PPO。
7. 依次执行安全门、candidate-vs-best 动态 Arena、history/基线回归和延迟检查。
8. 全部门槛通过则把旧 best 原子移入 history，并将 candidate 晋级为新 best；
   否则淘汰 candidate，best 保持不变。
9. 写入不可变迭代报告并进入下一轮。

禁止重复使用旧 rollout 冒充新一轮 on-policy 数据。所有轨迹必须记录生成策略
哈希、对手策略哈希、迭代号、branch、deck_id、座次、结果和数据 split。

## 对手池

默认训练对手分布：

- 50%：当前 best 镜像；
- 30%：history，近期版本权重更高，同时保留少量早期版本；
- 20%：规则基线（Random、FirstMin）和其他显式配置的固定基线。

当 history 为空时，其 30% 权重回退到当前 best。跨 Top2 冻结 best 默认进入
评估矩阵，不进入训练池。对手抽样必须可复现并写入 manifest；比赛引擎本身不
支持严格 RNG seed 的限制要继续在报告中披露。

## 训练与安全门

- 每轮每分支默认生成 3,000 局训练候选轨迹。
- PPO 从当前 best 初始化，默认使用已验证的 conservative 参数。
- 训练前校验 checkpoint candidate_id、deck_id 和哈希。
- loss、KL、clip fraction、entropy 和梯度必须为有限值。
- 非法动作、异常、跨分支数据或非 train split 进入训练时立即终止本轮。
- candidate 不得直接覆盖 best，也不得修改 `submission/deck.csv`。

## 动态晋级门控

candidate 对当前 best 交换先后手：

1. 先运行 1,000 局。
2. 非平局胜率不低于 58%时，可进入其余回归门检查。
3. 非平局胜率不高于 52%时，直接淘汰。
4. 处于 52%～58%灰区时，分批追加比赛，最多累计 3,000 局。
5. 灰区最终晋级要求：
   - 点估计不低于 55%；
   - Wilson 95% 下界严格高于 52%。

无论 candidate-vs-best 结果如何，下列硬门都必须满足：

- smoke、Arena 和回归均为 0 异常、0 非法动作；
- 永久回归集没有显著退化；
- 对近期 history 的综合胜率不低于 55%，且无明确单点退化；
- 对早期 history 和规则弱基线的目标胜率不低于 70%；
- 跨 Top2 冻结 best 的表现相对晋级前下降不超过 2 个百分点；
- 推理 p95 不超过当前 best 的 1.25 倍。

发布到正式 submission 仍需独立发布评审；自博弈 best 晋级不授予发布权限。

## 服务器执行

正式训练在服务器执行，本机只承担开发、10～100 局 smoke 和恢复测试。

- primary、reserve rollout 使用两个并行 CPU 作业。
- 两条 PPO 训练使用 GPU 数组任务或可用 GPU 并行。
- Arena 可以按分支和批次并行，但同一批次报告必须原子聚合。
- 每轮墙钟预算上限为 2.5 小时。
- 默认连续运行 5 轮后停止并回收结果，由人工集中复盘后决定下一批轮次。

基于 2026-07-30 实测吞吐，6,000 局 rollout 约 56 分钟，两分支 PPO 并行约
2～4 分钟，首批 2,000 局门控约 19 分钟。明确胜负的一轮预计约 1.5 小时；
灰区追加到 3,000 局时预计约 2～2.5 小时，不含服务器排队。

## 失败处理

- 任一阶段失败都保留日志、manifest 和部分产物，状态标记为 failed。
- 从失败状态恢复时，只重跑未完成或校验失败的阶段。
- checkpoint、manifest 或状态哈希不一致时禁止自动恢复，要求人工审查。
- 晋级采用两阶段操作：先验证 candidate 与目标目录，再原子更新 best 指针；
  history 写入成功前不得移除旧 best。
- 超过轮次预算或墙钟上限时安全停止，不把未完整评估的 candidate 留作 best。

## 测试与验收

自动测试覆盖：

- primary/reserve 状态与数据隔离；
- 对手池权重、history 为空时的回退和确定性抽样；
- 整局 split、holdout 禁止训练和旧 rollout 禁止跨轮复用；
- checkpoint 身份与哈希校验；
- 1,000/3,000 局动态门控的晋级、淘汰和灰区边界；
- Wilson 区间计算；
- 中断恢复、重复执行幂等性和原子晋级；
- history 只读与 best 回滚；
- 服务器作业参数、分支输出和最终聚合报告。

端到端验收先在本机执行每分支 10 局、1 batch PPO、20 局门控的缩小版循环，
再构建服务器交接包。正式服务器第一批只运行 1 轮；验证回收状态和报告后，再
授权连续运行剩余 4 轮。

## 非目标

- 本轮不实现 MCTS、策略价值搜索教师或 AlphaZero 式搜索概率标签。
- 不自动替换正式 submission。
- 不将 primary 与 reserve 合并成同一个训练池。
- 不在第一版实现 AlphaStar 式多 exploiter 联赛。
