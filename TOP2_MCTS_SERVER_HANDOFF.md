# Top2 belief-PUCT MCTS 服务器小规模验证

## 2026-07-31 服务器实测状态

已在 Tesla T4、PyTorch 2.7.1+cu128 环境完成交接包校验，87 个文件全部通过。

- primary 极小链路 smoke：2 局、108 个样本、864 个节点，0 fallback、0 exception、0 illegal action。
- primary 正式搜索预算 benchmark：10 局、32 simulations、3 particles、max depth 8；产生 631 个样本和 20,192 个节点，采集耗时 250.92 秒，约 143.47 局/小时。
- 1 epoch 蒸馏使用 470 个训练样本：policy loss 1.1805、value loss 1.7792、reference KL 0.0000212；candidate 没有发生明显策略漂移。
- candidate 独立 100 局复评为 49 胜 51 负，0 exception、0 illegal action、0 fallback，确认先前 10 局的 1:9 是小样本波动，不是 checkpoint 崩坏。
- best+MCTS 对 pure best 的独立 100 局复评为 65 胜 35 负，胜率 65%，Wilson 95% 区间为 55.25%～73.64%；0 exception、0 illegal action、0 fallback，平均决策 94.3ms、p95 115.5ms。

上述 100 局结果是明确的正向初步证据，但正式门仍要求 400 局。primary/reserve 的 400 局任务曾启动，随后按用户要求主动停止；任何部分输出均不得作为正式评估结果。服务器当前应保持无项目进程状态。

## 校验

```bash
sha256sum -c pokemon-tcg-top2-mcts-pilot-v1.tar.gz.sha256
tar -xzf pokemon-tcg-top2-mcts-pilot-v1.tar.gz
cd pokemon-tcg-top2-mcts-pilot-v1
python scripts/verify_top2_mcts_handoff.py
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 先跑 10 局 smoke

```bash
export PILOT_GAMES=10
export ARENA_GAMES=50
screen -S top2-mcts-smoke
bash jobs/top2_mcts_pilot_single_node.sh
```

## Power-loss-resumable run (recommended)

For a single RTX 4060 Ti and a CPU with many cores, run the two branches
sequentially. Every arena game is saved atomically. After a reboot, enter the
same extracted directory and run the same command again; completed work is
loaded with `--resume`.

```bash
export PILOT_ROOT="$PWD/mcts-pilot-200"
export PILOT_GAMES=200
export ARENA_GAMES=400
screen -S top2-mcts-resilient
bash jobs/top2_mcts_pilot_resilient.sh
```

Detach with `Ctrl-A`, then `D`. After a host reboot:

```bash
cd pokemon-tcg-top2-mcts-pilot-v2
screen -S top2-mcts-resilient
bash jobs/top2_mcts_pilot_resilient.sh
```

Per-game arena checkpoints are written as `search-eval.progress.json` and
`candidate-eval.progress.json` under each branch directory. Do not delete them
until the final reports have been downloaded.

每 10 局查看：

```bash
cat mcts-pilot/primary/samples/progress.json
cat mcts-pilot/reserve/samples/progress.json
```

必须确认两个分支均为 0 exceptions、0 illegal actions、fallback rate <5%，再运行 200 局。primary 已完成该链路验证；reserve 仍需补齐。

## 200 局试验

```bash
export PILOT_ROOT="$PWD/mcts-pilot-200"
export PILOT_GAMES=200
export ARENA_GAMES=400
screen -S top2-mcts-200
bash jobs/top2_mcts_pilot_single_node.sh
```

该包不授权替换 submission，也不自动晋级 best。先比较纯 best、best+MCTS 和 MCTS candidate，再决定是否扩大到 3,000 局。

## 下次恢复时的执行顺序

1. 校验服务器上没有遗留的 `run_top2_mcts_pilot.py`、`evaluate_top2_mcts.py` 或 `collect_top2_mcts.py` 进程。
2. 从零重新运行 primary 的 400 局 `best+MCTS vs pure best` 正式门；中止任务的部分结果不续用。
3. primary 正式门通过后，以同一搜索预算运行 reserve 的 100 局预检和 400 局正式门。
4. 两分支搜索增益均确认后，再运行每分支 200 局 MCTS 数据采集与蒸馏；不要把“搜索提升”与“无搜索 candidate 提升”混为同一指标。
5. candidate 先做 100 局 sanity check，再按正式门扩到 400 局；未通过前不自动晋级 best、不替换 submission。
