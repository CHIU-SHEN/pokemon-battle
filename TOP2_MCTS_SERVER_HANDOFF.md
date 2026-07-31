# Top2 belief-PUCT MCTS 服务器小规模验证

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

每 10 局查看：

```bash
cat mcts-pilot/primary/samples/progress.json
cat mcts-pilot/reserve/samples/progress.json
```

必须确认两个分支均为 0 exceptions、0 illegal actions、fallback rate <5%，再运行 200 局。

## 200 局试验

```bash
export PILOT_ROOT="$PWD/mcts-pilot-200"
export PILOT_GAMES=200
export ARENA_GAMES=400
screen -S top2-mcts-200
bash jobs/top2_mcts_pilot_single_node.sh
```

该包不授权替换 submission，也不自动晋级 best。先比较纯 best、best+MCTS 和 MCTS candidate，再决定是否扩大到 3,000 局。
