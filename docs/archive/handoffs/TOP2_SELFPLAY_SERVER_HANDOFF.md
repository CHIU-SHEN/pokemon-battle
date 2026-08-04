# Top2 门控式自博弈服务器执行

## 目标

primary 与 reserve 分别维护独立的 `best/candidate/history`、数据和状态。每轮从
当前 best 生成全新轨迹，训练 candidate，并通过 1,000～3,000 局动态 Arena
门控决定晋级或淘汰。自博弈晋级不授权替换正式 submission。

## 校验

```bash
sha256sum -c pokemon-tcg-top2-gated-selfplay-v1.tar.gz.sha256
tar -xzf pokemon-tcg-top2-gated-selfplay-v1.tar.gz
cd pokemon-tcg-top2-gated-selfplay-v1
python scripts/verify_top2_selfplay_handoff.py
mkdir -p logs selfplay
```

## 第一轮

```bash
export ITERATION_ID=iter-0001
export SELFPLAY_ROOT="$PWD/selfplay"

ROLLOUT_JOB=$(sbatch --parsable jobs/top2_selfplay_rollout.slurm)
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$ROLLOUT_JOB jobs/top2_selfplay_train.slurm)
GATE_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB jobs/top2_selfplay_gate.slurm)
echo "$ROLLOUT_JOB $TRAIN_JOB $GATE_JOB"
```

两个数组下标固定对应 `0=primary`、`1=reserve`。rollout 阶段各生成 3,000 局，
训练阶段从各自当前 best 初始化 candidate，gate 阶段先跑 1,000 局；52%～58%
灰区自动追加，最多 3,000 局。

## 回收检查

第一轮结束后回收：

- `selfplay/primary/state.json`
- `selfplay/reserve/state.json`
- 两个 `iterations/iter-0001/iteration-report.json`
- 本轮 candidate、Arena、holdout、rollout summary 和全部日志

确认两个分支无交叉哈希、0 异常、0 非法动作，且晋级记录与 Arena 门控一致后，
才允许以 `iter-0002`～`iter-0005` 重复相同三阶段提交。不要一次性预提交五轮，
以免第一轮暴露的问题扩散到后续数据。
