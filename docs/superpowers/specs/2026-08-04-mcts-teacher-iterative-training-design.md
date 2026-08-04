# MCTS 教师迭代训练与收敛验证设计

> 本文取代同日的 `mcts-teacher-strengthening-design.md`。修订原因是：教师模型必须继续进行参数优化，Arena 胜率本身不能代表梯度收敛。

## 目标与范围

在 MCTS 蒸馏 v2 前，先用高预算 MCTS 数据迭代训练 primary 的 policy/value 教师，直到参数更新与冻结 holdout 改善共同进入平台期，或达到墙钟上限。约 62%～65% 的 Arena 表现不是停止训练的依据。

本阶段只处理 primary；reserve 后续独立执行。本阶段不继续 PPO、不修改 Kaggle submission、不重跑旧 400 局门控、不删除或重建 `data/`，也不启动学生蒸馏。

## 冻结输入

- 权威归档：`artifacts/top2-mcts-complete-results-20260804.tar.gz`。
- SHA-256：`f926fbe822d18321d3e083bd30fd60a73da6f35517327f69d0e7bd44262cb531`。
- primary deck、初始 adapter、数据 split 和安全定义保持冻结。
- 新迭代使用独立目录，不覆盖旧归档。

## 教师训练闭环

每轮按以下顺序执行：

1. 从上一轮冻结 best 启动高预算 belief-PUCT MCTS。
2. 生成完整 visit counts、联合动作、终局 value 和公开状态特征。
3. 按 `game_id` 保持 train/valid/test 分流，holdout 永不参与优化。
4. 使用 train split 更新教师 policy/value。
5. 在冻结 holdout 上计算 policy、value、KL 和校准指标。
6. 运行 swapped-seat Arena 小门；通过后才晋级为下一轮 best。
7. 根据联合收敛条件继续或停止。

隐藏手牌、奖赏卡、完整牌库和 belief particle 内容不得写入训练样本。搜索可以在内存中使用 particles，教师网络只能接收部署时合法可观察的输入。

## 可训练参数与损失

教师不能采用学生侧“只训练 `policy_delta`”的限制。首选配置为：

- 冻结共享 base 主体；
- 解冻 deck adapter、`policy_delta` 和 `value_delta`；
- 保留相对初始教师的 reference KL；
- 使用低学习率和梯度裁剪。

只有 adapter 全部可训练仍连续欠拟合、且 holdout 不退化时，才单独实验解冻 base 最后一层；不得一次解冻完整共享 base。

总损失包含 MCTS visit distribution 的联合动作 soft-policy loss、终局 value loss、reference KL 和少量 entropy regularization。policy/value loss 必须分别记录。value 参与训练，因为 MCTS 叶节点依赖它；但 value 改善不能掩盖 policy holdout 或 Arena 退化。

## 参数收敛与停止条件

每个优化 step 记录裁剪前/后的梯度范数，以及：

`relative_update = ||theta_after - theta_before||_2 / max(||theta_before||_2, 1e-12)`

对 step 指标使用 EMA；一个评估窗口默认为一个完整 epoch。只有连续 3 个窗口同时满足以下条件，才判定平台期：

- `relative_update` EMA `< 1e-5`；
- holdout policy loss 相对改善 `< 0.2%`；
- holdout value loss 没有恶化超过 `1.0%`；
- reference KL 有限且未超过配置硬门；
- 最近 swapped-seat Arena 未触发安全或性能退化门。

不得只凭梯度范数小就停止，因为低学习率、强裁剪或饱和也会制造假收敛。

出现任一情况时提前停止并回滚到最近安全 best：非有限 loss/梯度/参数；`exceptions>0`；`illegal_actions>0`；`fallback_rate>0`；holdout policy loss 连续 2 个窗口恶化；KL 越过硬门；或 Arena Wilson 区间证明相对 best 明确退化。

## 时间阈值与恢复

- 本地 CPU/GPU smoke：最多 30 分钟。
- 服务器单轮教师训练：最多 6 小时。
- 完整“数据生成—训练—holdout—Arena”循环：最多 24 小时。
- 每个 epoch 结束且至少每 30 分钟，原子保存 checkpoint、优化器、随机状态、累计时间和阶段 JSON。

达到时间阈值后完成当前原子 batch 再安全停止，写入 `time_limit_reached=true`；超时不等于收敛。下次从最近完整 checkpoint 恢复。网络断开不影响 tmux，实例断电后不重复已完成原子阶段。

## 搜索预算与对手诊断

部署配置 `8 simulations / 1 particle / depth 4` 只作现实基线。教师数据默认从已验证的 `32 simulations / 3 particles / depth 8` 开始。是否继续增加预算由吞吐和训练收益决定，不要求机械超过固定 simulation 数，且必须服从 24 小时循环上限。

每轮保存不泄漏隐藏信息的聚合诊断：有效/无效 particles、粒子间根动作一致率、visit distribution 熵、前两名间隔、裸 policy 与 MCTS 分歧率、分歧状态终局结果、对手节点数量和深度分布。若对手节点错误沿用根玩家最大化方向，必须先以回归测试复现并修复。

## Checkpoint 门控

每轮 candidate 先运行 100 局 swapped-seat 筛选，要求：

- `exceptions=0`、`illegal_actions=0`、`fallback_rate=0`；
- holdout policy 不退化；
- Arena 点估计不低于上一轮 best，或 Wilson 区间无法证明退化。

联合收敛后运行 400 局正式门。最终教师不要求机械超过历史 65.25%；要求安全、表现稳定，并具有参数更新和 holdout 收敛证据。若明显弱于冻结初始 best，则不得用于蒸馏。

## 蒸馏准入

教师通过 400 局正式门后才冻结并进入学生蒸馏 v2。届时优先选择 MCTS 与裸 policy 明显分歧、visit target 高置信、多 particles 排序一致且有终局结果支持的状态。原始 visit counts 必须完整保留，筛选规则进入 manifest。学生仍采用保守的 policy-only 更新。

## 组件边界

- 数据生成器：分支隔离的 MCTS records 和恢复状态。
- 教师训练器：policy/value 优化、梯度与参数更新统计、时间判停。
- Holdout evaluator：只读冻结 split，计算泛化和校准指标。
- Arena gate：swapped-seat 对局、安全门和 Wilson 统计。
- Iteration controller：原子推进、best 晋级、恢复和总时间控制。
- 学生蒸馏器：不属于本阶段。

## TDD 验证范围

- 参数冻结范围及允许模块确实发生更新；
- 梯度范数与相对参数更新量的手算样例；
- 连续 3 窗口联合平台期判定；
- 小梯度但 holdout 仍改善时不得停止；
- 30 分钟、6 小时和 24 小时时间门及恢复；
- 非有限值、KL、holdout 退化和安全硬门回滚；
- primary/reserve 数据与 checkpoint 身份隔离；
- 聚合诊断不含隐藏字段；
- 100 局筛选与 400 局正式门的判定差异；
- CPU smoke 产生可恢复 checkpoint 和阶段 JSON；
- 交接包 manifest 与逐文件 SHA-256 校验。

## 成功标准

本阶段成功是以下任一有证据的结论：

1. 教师达到参数更新与 holdout 联合平台期，通过安全门和 400 局正式门，可进入蒸馏；
2. 时间上限内尚未收敛，但留下可恢复状态和明确继续训练证据；
3. 定位 belief、对手节点、value 或数据质量瓶颈，停止生成低质量 target。
