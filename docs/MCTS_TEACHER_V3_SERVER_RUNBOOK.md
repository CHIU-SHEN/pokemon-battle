# MCTS Teacher v3 服务器运行说明

## 1. 上传与校验

将下面两个文件上传到服务器同一目录：

- `mcts-teacher-v3-quality-gated.tar.gz`
- `mcts-teacher-v3-quality-gated.tar.gz.sha256`

执行：

```bash
sha256sum -c mcts-teacher-v3-quality-gated.tar.gz.sha256
tar -xzf mcts-teacher-v3-quality-gated.tar.gz
cd mcts-teacher-v3-quality-gated
python3 scripts/verify_mcts_teacher_v3_handoff.py \
  ../mcts-teacher-v3-quality-gated.tar.gz
```

校验必须输出 `verified: true`。

## 2. 环境检查

```bash
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

训练阶段需要 CUDA；教师评估和样本生成使用 CPU。

## 3. 冒烟测试

```bash
sed -i 's/\r$//' jobs/mcts_teacher_v3_quality_gated.sh
bash jobs/mcts_teacher_v3_quality_gated.sh
```

这一步只规划/运行 10 局教师搜索，不开始完整采样。

## 4. 完整运行

建议在 Vast.ai 自带 tmux 中执行：

```bash
RUN_FULL_PIPELINE=1 MCTS_WORKERS=32 \
  bash jobs/mcts_teacher_v3_quality_gated.sh 2>&1 | tee mcts-v3.log
```

48 个物理核心的机器从 24 workers 开始；CPU 利用率稳定且内存充足时可提高到
32。每个 worker 固定单线程，避免 BLAS 线程过度竞争。

流水线首先用 128 simulations、depth 10 对教师进行 400 局评估。教师有效胜率
低于 58%，或者出现异常、非法动作、MCTS 安全回退时，任务会保留报告并停止，
不会继续花费 5,000 局采样与 GPU 训练费用。

## 5. 查看进度

教师评估：

```bash
watch -n 10 'python3 -c '\''import json,pathlib; p=pathlib.Path("experiments/mcts_teacher_v3/primary/teacher-eval.progress.json"); print(json.loads(p.read_text())["completed_games"] if p.exists() else 0)'\'''
```

采样阶段：

```bash
watch -n 10 'python3 -c '\''import json,pathlib; ps=pathlib.Path("experiments/mcts_teacher_v3/primary/collection/shards").glob("*/progress.json"); ds=[json.loads(p.read_text()) for p in ps]; print("workers",len(ds),"games",sum(d.get("completed_games",0) for d in ds),"target",sum(d.get("target_games",0) for d in ds))'\'''
```

训练阶段：

```bash
watch -n 5 'nvidia-smi; tail -n 2 mcts-v3.log'
```

网络断开后重新执行同一条完整运行命令即可续跑。不要修改 `RUN_ROOT`、worker
数量或搜索参数后复用旧进度。

## 6. 大致耗时

在 V100 32GB、48 物理核心、32 workers 的机器上，预计：

- 400 局教师门控：约 20–60 分钟
- 5,000 局高质量采样：约 4–10 小时
- 数据冻结与审计：约 10–30 分钟
- GPU 训练：约 1–4 小时，或由收敛/时间阈值提前停止

实际搜索速度由 CPU 单核性能决定，GPU 型号主要影响最后训练阶段。

## 7. 下载结果

成功完成后下载：

```text
experiments/mcts_teacher_v3/primary/mcts-teacher-v3-results.tar.gz
experiments/mcts_teacher_v3/primary/mcts-teacher-v3-results.tar.gz.sha256
```

若教师门控失败，则下载：

```text
experiments/mcts_teacher_v3/primary/teacher-eval.json
experiments/mcts_teacher_v3/primary/teacher-gate.json
```

流水线不会自动替换或晋级任何现有模型。
