# Top2 本地两小时调参试验与服务器交接设计

## 目标与边界

本地 RTX 5060 阶段只完成双分支数据链路冒烟和 primary 三组短调参实验，墙钟预算上限为两小时。它的输出是服务器正式训练的推荐起点，不宣称已经完成正式强化学习，也不授权替换 `submission/deck.csv`。

服务器阶段接收本地冻结的分支身份、数据 manifest、吞吐报告、三组试验结果和推荐超参数，再扩大 rollout、V1 重分析、masked PPO 与最终 Arena。

## 本地阶段

### 双分支 rollout

- primary：`crustle_kangaskhan_cage`
- reserve：`crustle_kangaskhan_petrel`
- 每个分支先对交叉 Top2 对手采集 100 局，并严格交换先后手。
- 每局按稳定 `game_id` 哈希进入 80% train、10% valid、10% test；valid/test 合计 20%，永不进入 PPO optimizer。
- 任一分支出现异常、非法动作、哈希漂移或 `deck_id` 串流时立即停止后续调参。

### primary 三组短实验

三组实验必须从完全相同的冻结 SL-0 主干、primary Adapter 和同一批 train rollout 开始：

| 组别 | learning rate | clip | KL coefficient | entropy | epoch 上限 |
| --- | ---: | ---: | ---: | ---: | ---: |
| conservative | `5e-5` | `0.10` | `0.10` | `0.005` | 3 |
| baseline | `1e-4` | `0.15` | `0.05` | `0.010` | 4 |
| exploratory | `2e-4` | `0.20` | `0.02` | `0.020` | 4 |

`gamma=0.99`、`GAE lambda=0.95`、value coefficient `0.5`、最大梯度范数 `0.5` 在本地阶段固定，不和学习率/clip/KL/entropy 同时搜索，避免 100 局样本上产生虚假最优。

### 自动停止与验收

每个训练组逐 epoch 写 checkpoint 和指标，满足任一条件即停止该组：

- loss、policy loss、value loss、entropy 或 KL 出现非有限值；
- 相对冻结初始 Adapter 的 KL 超过 `0.03`；
- clip fraction 超过 `0.30`；
- entropy 相对首个 batch 下降超过 50%；
- 两小时总墙钟预算即将耗尽。

每个未失败的 candidate 都运行：

- valid/test 上的 reference KL、value MSE、动作一致率和非法 argmax 检查；
- 200 局与初始 primary Adapter 的交换先后手 Arena；
- 推理 p95、异常和非法动作统计。

## 选择规则

先应用硬门槛：0 异常、0 非法动作、训练指标有限、KL 不超过 `0.03`、holdout 无明显退化。通过硬门槛后按以下顺序选择服务器推荐起点：

1. Arena 胜率及 Wilson 95% 下界；
2. holdout value MSE 与 reference KL；
3. 推理 p95 和回退率；
4. 若 200 局无法区分，则选择更保守的参数组。

100 局 rollout 和 200 局 Arena 只足以筛掉明显不稳定配置。如果三组置信区间重叠，报告必须写成“未区分，推荐 conservative/baseline 作为服务器起点”，不能标记为已找到最优超参数。

## 两小时预算控制

本地 orchestrator 记录以下阶段的独立墙钟：模型加载、每个分支 rollout、每组 PPO、holdout、每组 Arena、打包。默认先运行 10 局预基准，再用实际吞吐预测剩余时间：

- 预测总时长不超过 90 分钟：执行完整本地方案；
- 预测为 90～120 分钟：保持双分支各 100 局，将每组 Arena 降至 100 局；
- 预测超过 120 分钟：完成双分支冒烟，只跑 conservative 与 baseline 各 2 epoch，并在报告中标记预算降级。

任何降级都不能减少 20% holdout 或放宽异常/非法动作门槛。

## 产物与服务器交接

本地阶段生成：

- `reports/top2_local_pilot_report.json` 与 `.md`；
- 每个分支的 rollout manifest、split 数量、吞吐、磁盘增长；
- 三组 primary checkpoint、逐 epoch metrics、holdout 与 Arena 报告；
- `config/top2_rl_selected.json`，记录推荐组、参数、证据和“preliminary”状态；
- 更新后的 `pokemon-tcg-top2-rl-handoff-v2.tar.gz` 与 SHA-256 sidecar。

v2 服务器包包含正式阶段命令和本地报告，但不携带不必要的原始开发临时文件。服务器仍需从新 rollout 开始 on-policy 迭代，不能在同一批本地轨迹上无限重复训练。

## 失败处理

- 任何安全门槛失败：保留日志和 candidate，不发布推荐配置。
- 5060 CUDA 不可用或显存不足：自动降低 batch size；仍失败则改用 CPU 完成链路验证并停止调参。
- 两小时预算耗尽：完成当前原子步骤后停止，不写“成功完成全套”的状态。
- 三组均未超过初始 Adapter：服务器默认使用 conservative 配置继续扩大数据，初始 Adapter 保持发布基线。
