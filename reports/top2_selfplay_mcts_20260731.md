# Top2 三轮 PPO 自博弈与 MCTS 试验记录

> 日期：2026-07-31  
> 当前结论：三轮 PPO candidate 均未晋级；冻结 best 保持不变。PUCT-MCTS 搜索方向在 primary 的 100 局复评中取得 65% 胜率，值得继续扩大验证，但尚未授权替换 submission 或 best。

## 三轮门控式 PPO 自博弈

服务器使用 primary/reserve 独立的 `best/candidate/history` 状态，每轮每分支生成 3,000 局 rollout，再训练 candidate 并对当前 best 做交换先后手 Arena。

| 轮次 | 分支 | 胜-负-平 | 非平局胜率 | Wilson 95% 下界 | 门控结果 |
| --- | --- | ---: | ---: | ---: | --- |
| iter-0001 | primary | 1,523-1,461-16 | 51.04% | 49.25% | reject |
| iter-0001 | reserve | 1,486-1,490-24 | 49.93% | 48.14% | reject |
| iter-0002 | primary | 491-505-4 | 49.30% | 46.20% | 首批 1,000 局 reject |
| iter-0002 | reserve | 1,529-1,445-26 | 51.41% | 49.62% | 3,000 局 reject |
| iter-0003 | primary | 512-481-7 | 51.56% | 48.45% | 首批 1,000 局 reject |
| iter-0003 | reserve | 501-485-14 | 50.81% | 47.69% | 首批 1,000 局 reject |

三轮均为 0 异常、0 非法动作。所有 candidate 均被安全拒绝，两个分支的 `history` 仍为空，best 仍是冻结初始 Adapter。

iter-0002 使用 3 epoch；最终参考 KL 约 0.0020，动作一致率约 96%。iter-0003 将学习率提高到 `1.5e-4`、epoch 提高到 5，但最终 KL 仍只有约 0.00247，动作一致率约 95.6%，Arena 未形成稳定增益。

### 门控边界修复

iter-0001 暴露了总局数与非平局局数混用的问题：循环按 3,000 总局停止，门控却按 3,000 非平局局数判断，导致跑满后内部仍显示 `continue / gray_zone_sample_incomplete`。修复后门控采样上下限统一按总局数判断，胜率仍按非平局计算。真实 iter-0001 数据复测均得到 `final_rate_or_wilson_gate_failed`；本地全套测试为 43 passed。

该错误没有造成错误晋级，只造成额外 Arena 消耗和错误的内部原因文本。

### PPO 停止原因

当前 PPO 只在最后一个决策写入整局 `+1/-1`，再用 `gamma=0.99`、`lambda=0.95` 的 GAE 向前传播。平均约 100 个决策时，终局信号传到早期决策约为 `0.9405^100 ≈ 0.0022`。连续增加学习率和 epoch 没有解决长期信用分配，因此停止未经验证的 PPO 第四、第五轮。

## PUCT-MCTS 小规模试验

试验包使用官方 Search API、隐藏信息 belief particles、神经网络 policy/value、合法动作 PUCT 和访问次数策略目标。该包不自动晋级 best，也不替换 submission。

### 2 局链路 smoke

- 108 个搜索训练样本，864 个节点；
- 0 fallback、0 异常、0 非法动作；
- 13.45 秒，约 535 局/小时（缩小预算）；
- Search agent 2-0，candidate 1-1；只证明链路。

### 10 局正式预算 benchmark

参数为 32 simulations、3 particles、最大深度 8：

- 10 局产生 631 个样本、20,192 个节点；
- 250.92 秒，约 143.47 局/小时；
- 0 fallback、0 异常、0 非法动作；
- 470 个 train 样本，1 epoch；
- candidate 相对 reference KL 为 `2.12e-5`，几乎未改变；
- Search agent 7-3；candidate 1-9。

candidate 的 1-9 后续通过独立 100 局复评判定为小样本波动：复评为 49-51，0 异常、0 非法动作。

### primary Search 100 局复评

| 指标 | 结果 |
| --- | ---: |
| MCTS vs pure best | 65-35 |
| 胜率 | 65.00% |
| Wilson 95% | 55.25%～73.64% |
| fallback | 0 |
| 异常 / 非法动作 | 0 / 0 |
| mean 决策耗时 | 94.3 ms |
| p95 决策耗时 | 115.5 ms |

该结果首次显示搜索代理对冻结 best 的明显增益。报告仍写 `search_uplift_not_proven`，因为正式规则要求至少 400 局，而当前只有 100 局。

## 已停止和保留事项

- primary/reserve 的 400 局 MCTS 搜索任务曾准备启动，随后按用户要求停止；任何中途输出均不得作为正式结果。
- 服务器上的本项目训练、评估、MCTS 和相关 screen 进程已要求全部关闭；不得误杀系统、SSH、Jupyter 或其他用户进程。
- PPO 三轮精简证据已回收到 `artifacts/top2-ppo-3round-evidence.tar.gz`，SHA 文件同目录；压缩包约 771 KiB。
- 旧 PPO raw rollout 不再作为下一任 best 的训练依据。MCTS 将生成新的搜索访问分布数据。

## 后续执行顺序

1. 先在 reserve 分支运行同配置的 100 局 `best+MCTS vs pure best` 复评。
2. primary 和 reserve 只有在 100 局胜率仍有正向信号、且 0 异常/0 非法动作时，才分别扩到 400 局正式交换先后手验证。
3. 搜索正式验证通过后，每分支采集约 200 局 MCTS 访问分布数据；按当前实测，单分支采集约 1.4 小时，双分支并发时间取决于 CPU 竞争。
4. 用访问次数分布做 policy target、整局胜负做 value target；不再使用终局奖励经 GAE 长距离衰减的 PPO 目标。
5. 分别评估 pure best、best+MCTS、MCTS candidate，区分搜索增益和蒸馏增益。
6. candidate 只有通过 400～1,000 局 Arena、回归、时延、0 异常和 0 非法动作门槛后，才允许进入正式 best/history 池。
7. 正式提交和 `submission/deck.csv` 在全部门槛通过前保持不变。
