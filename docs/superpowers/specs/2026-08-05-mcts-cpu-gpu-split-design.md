# MCTS CPU 采集与 GPU 训练拆分设计（已被取代）

> 本方案已被 `2026-08-05-mcts-all-in-one-pipeline-design.md` 取代。保留本文仅用于历史追踪；当前目标是在一台 48 vCPU + Tesla V100 服务器上串行完成采集与训练。

## 目标

把现有混合交接包拆成两条物理和职责均独立的链路：无 GPU 的多核 CPU 服务器只生成 MCTS 搜索样本；GPU 服务器只校验数据、训练教师并输出安全 checkpoint。两条链路只通过带 manifest 和 SHA-256 的不可变数据归档通信。

primary 第一轮先完成 10 局 smoke，再完成 2,400 局吞吐门；安全和吞吐达标后扩到累计 10,000 局。reserve 不进入本轮。

## CPU Collector 包

包名为 `mcts-cpu-collector-v1.tar.gz`，不要求 CUDA。它包含冻结 primary adapter、共享 base、游戏运行时、采集器、并行调度器、验证器和恢复脚本。

调度器为每个 worker 分配独立的：

- shard 目录；
- seed；
- iteration ID；
- stdout/stderr 日志；
- `progress.json`。

所有 worker 强制 `--device cpu`，并设置：

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
```

默认并发根据逻辑 CPU 数保守选择，但允许显式覆盖：16 vCPU 默认 4 worker，32 核默认 12 worker，48 核默认 16 worker，64 核默认 24 worker。每个 worker 有独立输出，因此不得让多个进程写同一 `games/` 目录。

## 分阶段采集

1. 10 局单 worker smoke：验证 Search API、公开特征 schema、原子文件和安全指标。
2. 2,400 局吞吐门：分成多个等量 shard，交换先后手，各 shard 游戏数为偶数。
3. 10,000 局正式集：只在吞吐门通过后补足累计目标，不重跑已完成 shard。

吞吐门要求：

- `exceptions=0`；
- `illegal_actions=0`；
- `fallback_rate=0`；
- 所有 game ID 全局唯一；
- 每个 shard 的完成数与目标一致；
- 至少存在 train、valid、test 三种 split；
- 实际总样本数大于 0；
- 无隐藏 belief 字段进入样本。

现有采集器没有正确累计 fallback，因此实施时必须补齐 action-source/fallback 统计，并在恢复后重算全量 shard 汇总，不能只报告本次进程增量。

## CPU 数据归档

采集完成后生成 `mcts-primary-dataset-v2.tar.gz`。manifest 至少包含：

- schema version；
- branch、deck ID 和冻结 checkpoint SHA-256；
- simulations、particles、depth 和全部 seeds；
- worker/shard 数量；
- game、sample、node、fallback、exception、illegal 总计；
- split 计数；
- game ID 去重结果；
- 每个成员文件 SHA-256；
- 原始旧 600 局归档 SHA-256，仅作来源记录，不自动混入新归档。

归档构建器拒绝不完整 shard、安全指标非零、隐藏字段、重复 game ID 或身份不匹配。

## GPU Trainer 包

包名为 `mcts-gpu-trainer-v2.tar.gz`。它不包含 MCTS 数据生成入口和 resilient collector job，只包含：

- 数据归档验证与合并工具；
- 教师训练器；
- GPU train/resume smoke；
- holdout evaluator；
- checkpoint 选择器；
- GPU resilient job。

GPU 包接收一个或多个已验证的数据归档。默认训练集由旧 600 局权威归档与新 CPU 归档合并，按 game ID 去重；split 继续由稳定 game ID 哈希决定，任何来源不得重写 split。

## 教师参数和损失

- 冻结共享 base 主体；
- 解冻 adapter、`policy_delta`、`value_delta`；
- 使用 visit-count joint-action soft-policy loss、terminal value loss、reference KL 和 entropy regularization；
- policy/value/holdout/KL/梯度/参数更新分别记录；
- primary/reserve checkpoint 身份严格校验。

## Best-safe checkpoint

每个 epoch 在 holdout 后先判断安全性。只有同时满足下列条件才允许覆盖 `best_safe.pt`：

- 所有 loss、梯度和参数有限；
- holdout reference KL `<= 0.03`；
- holdout policy loss 优于当前 best-safe；
- holdout value loss未比当前 best-safe 恶化超过 1%；
- branch、deck ID 和 reference hash 一致。

`last.pt` 始终保存最新可恢复状态，包括越界状态；`best_safe.pt` 永远指向最近的安全最优 epoch。达到 KL 硬门时停止并保留两者，禁止用 unsafe `last.pt` 进行 Arena。

## 自适应 KL

训练仍保持 KL 硬门 0.03，但不再只用固定系数。默认初始 `kl_coef=0.05`：

- holdout KL `< 0.015`：下一 epoch 系数乘 0.8，最低 0.01；
- holdout KL 在 `[0.015, 0.025]`：系数不变；
- holdout KL 在 `(0.025, 0.03]`：下一 epoch 系数乘 2，最高 1.0；
- holdout KL `> 0.03`：停止并回滚候选选择到 `best_safe.pt`。

optimizer 恢复后必须显式应用当前自适应系数和 CLI 学习率，避免 optimizer state 静默覆盖新参数。

## 时间与恢复

CPU：smoke 30 分钟上限；2,400 局门 6 小时上限；10,000 局总任务 24 小时上限。GPU：smoke 30 分钟；单轮训练 6 小时；完整训练与 holdout 24 小时。

所有 shell 文件以 LF 打包。CPU/GPU job 使用 tmux，阶段 JSON 和 checkpoint 原子写入。超时只标记 `time_limit_reached=true`，不等于收敛。

## 测试

实施使用 TDD，覆盖：

- worker 数量选择和显式覆盖；
- shard seed/ID/路径互斥；
- 恢复后的累计统计重建；
- fallback/action-source 真实统计；
- 隐藏字段和重复 game ID 拒绝；
- 2,400/10,000 阶段门；
- CPU 数据归档逐文件哈希；
- 多归档身份校验、去重和稳定 split；
- `last.pt` 与 `best_safe.pt` 行为；
- 自适应 KL 四个区间；
- resume 后 CLI 学习率与 KL 生效；
- CPU-only smoke 和 GPU smoke；
- 所有 `.sh` 成员不含 CRLF。

## 成功标准

CPU 阶段成功意味着产生经过安全、身份、去重和哈希校验的 10,000 局 primary 数据归档。GPU 阶段成功意味着教师训练保留可恢复 `last.pt` 和可用于后续 Arena 的 `best_safe.pt`，且不会因为 KL 越界丢失最后一个安全 epoch。
