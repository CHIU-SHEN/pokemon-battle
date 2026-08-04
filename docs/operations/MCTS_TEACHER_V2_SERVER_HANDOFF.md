# MCTS Teacher v2 服务器交接

本包只训练 primary 教师，不发布 submission，也不自动启动 100/400 局 Arena。

## 验证与 smoke

```bash
python3 scripts/verify_mcts_teacher_v2_handoff.py ../mcts-distill-v2-teacher.tar.gz
bash jobs/mcts_teacher_v2_resilient.sh
```

默认命令只校验权威归档并运行 CPU train/resume smoke。必须看到 `passed=true`、`resume_verified=true`，且异常、非法动作和 fallback 均为 0。

## 启动有界 primary 训练

```bash
tmux new -s mcts-teacher-v2
RUN_TEACHER_TRAIN=1 bash jobs/mcts_teacher_v2_resilient.sh 2>&1 | tee experiments/mcts_teacher_v2/primary/run.log
```

单轮训练最多 6 小时，每 30 分钟或 epoch 保存一次原子 checkpoint。再次执行同一命令会从 `train/last.pt` 恢复。完整数据生成—训练—holdout—Arena 循环的外层调度不得超过 24 小时。

`time_limit_reached=true` 只表示安全超时，不表示收敛。`converged=true` 必须来自连续三个窗口的参数更新与 holdout 联合平台期。

## 后续人工门

训练 summary 安全且 eligible 后，先人工启动 100 局 swapped-seat 筛选。只有异常、非法动作、fallback 全为 0 且 holdout 不退化，才运行 400 局正式门。不得自动修改 Kaggle submission；reserve 必须另建隔离目录独立处理。
