# Top2 强化学习交接包设计

## 目标

交付一个可在本机或 Linux GPU 服务器运行的 Top2 强化学习包。它必须把 primary 与 reserve 作为两个互不混用的 `deck_id` 分支，能够完成 rollout、20% 永不训练回归集冻结、V1 候选生成、masked PPO 更新和 Arena 验收，但本次构建不启动长训练，也不修改正式提交牌表。

## 冻结输入

- primary：`crustle_kangaskhan_cage`
- reserve：`crustle_kangaskhan_petrel`
- 公共主干：`artifacts/sl0_shared_full/best.pt`
- 初始策略：各自的 Top10 Adapter `best.pt`
- 身份来源：`reports/top2_freeze_report.json` 中的牌表及 checkpoint SHA-256

## 运行边界

每个分支使用自己的目录、`deck_id`、rollout 清单、回归集、训练日志和 checkpoint。默认 reserve rollout/PPO 预算为 primary 的 40%，允许在 30%～50% 内调整。引擎不支持严格 seed 控制，因此所有评估必须交换先后手并在报告中保留 `engine_seed_controlled=false`。

策略只接管当前在线 Adapter 已覆盖的“非强制、单选、合法 option”决策。多选、可空选和异常状态继续走规则或安全回退，且不进入 PPO 样本。动作分布在合法 mask 后归一化；rollout 保存行为策略 log-prob、value、可见特征及终局回报。训练采用 clipped PPO、GAE、value loss、entropy bonus、相对初始 Adapter 的 KL 约束和梯度裁剪。

## 数据协议

原始 rollout 使用 `top2_rl_rollout_v1`。每局必须记录分支身份、牌表哈希、初始 Adapter 哈希、对手、先后手、结果、异常和逐决策样本。`game_id` 经过稳定哈希后分为 80% train、10% valid、10% test；valid 与 test 合计 20%，任何训练命令都必须拒绝读取这两个 split。

V1 只处理失败、低置信或策略分歧的训练候选。回归集、V1 标签和 PPO rollout 都保留分支 `deck_id`，禁止跨分支合并成无身份的数据流。

## 交付结构

压缩包名为 `pokemon-tcg-top2-rl-handoff-v1.tar.gz`，包含：

- 两副冻结牌表、两套 Adapter 和共享主干；
- rollout、PPO、Arena 和 V1 入口；
- `config/top2_rl_policy.json`；
- 快速开始与停止条件；
- 包内清单、逐文件 SHA-256 和压缩包 sidecar 哈希；
- 自包含测试和验证脚本。

## 安全门槛

100 局/分支冒烟必须为 0 异常、0 非法动作后才能扩大 rollout。训练产物只有在冻结 20% 回归集、V0/初始 Adapter 回归、交换先后手 Arena、时延和包自检全部通过后才允许进入发布评审。本包不授权覆盖 `submission/deck.csv`。
