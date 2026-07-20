# SL-1 GRU 首次服务器结果解读

> 日期：2026-07-20  
> 状态：离线观察门槛通过，尚未正式晋级

## 结果

| 指标 | SL-0 frozen test | SL-1 GRU best test | 变化 |
| --- | ---: | ---: | ---: |
| 总 loss | 2.1134 | 2.0433 | -3.32% |
| Policy loss | 1.2281 | 1.2160 | -0.98% |
| Policy top-1 | 60.27% | 60.95% | +0.68 个百分点 |
| Value MSE | 0.8916 | 0.8275 | -7.19% |

GRU 在总 loss、policy 和 value 上同时改善，说明公开状态差分、上一己方动作和循环状态包含有效信息。主要收益来自 value，policy 增益较小，尚不能由此推断整局胜率一定提高。

## Checkpoint 选择问题

`best.pt` 是 epoch 1 的最低 valid loss checkpoint；`last.pt` 是 epoch 5。后者 valid policy top-1 更高（63.09% 对 60.61%），但 value MSE 更差（0.9419 对 0.8645），呈现 policy 继续改善、value 开始退化的多任务分叉。因此固定 Arena 必须同时测试 best 和 last，后续训练应分别保存 best-total、best-policy 和 best-value checkpoint。

## 尚缺的晋级证据

1. 用新版评估器补齐非强制单选、policy source、非法 top-1 和推理吞吐。
2. 用预先固定的第二 seed 完整复跑，确认提升方向可以复现。
3. 补齐 `SL-0-history` 全量对照，区分“历史信息收益”和“GRU 容量收益”。
4. 在共同 seed、交换先后手的固定 Arena 中比较 SL-0、SL-0-history、GRU best 和 GRU last，同时记录 Combo 完成率、异常、回退率和时延。
5. 通过上述门槛后，才实现 NumPy 在线运行时、提交包和 Adapter。
