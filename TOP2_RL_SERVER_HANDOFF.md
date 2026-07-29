# Top2 强化学习服务器交接说明

## 这是什么

这是 `crustle_kangaskhan_cage`（primary）和 `crustle_kangaskhan_petrel`（reserve）的强化学习交接包。它包含真实可运行的 rollout、Top2 V1 重分析、masked PPO 和交换先后手 Arena 入口。两条分支有不同的 `deck_id`、数据目录和 checkpoint 身份校验，不能混训。

本包不会自动开始长训练，也不会修改 `submission/deck.csv`。第一次只运行 100 局/分支的采集冒烟和 1 batch PPO 冒烟；根据实际吞吐再决定留在 RTX 5060 8GB 本机还是迁移服务器。

## 资源要求

- rollout：CPU 即可，建议 8 个以上逻辑核和 4 GB 以上可用内存；当前比赛引擎不支持严格 seed 控制。
- PPO：PyTorch 2.7；模型只更新小型 Adapter，8 GB 显存足够做保守 batch 冒烟。若使用 CPU 也能运行，但正式迭代较慢。
- 磁盘：包本身很小；轨迹体积随保存的完整 observation 增长，正式运行前应至少预留 10～30 GB。
- 本项目在 Windows 使用 CUDA 12.8 wheel；Linux 服务器按实际 CUDA 版本安装匹配的 PyTorch。

## 1. 校验包

Linux：

```bash
sha256sum -c pokemon-tcg-top2-rl-handoff-v1.tar.gz.sha256
tar -xzf pokemon-tcg-top2-rl-handoff-v1.tar.gz
cd pokemon-tcg-top2-rl-handoff-v1
python scripts/verify_top2_rl_handoff.py
python tests/test_top2_rl_handoff.py
```

PowerShell：

```powershell
Get-FileHash -Algorithm SHA256 .\pokemon-tcg-top2-rl-handoff-v1.tar.gz
tar -xzf .\pokemon-tcg-top2-rl-handoff-v1.tar.gz
Set-Location .\pokemon-tcg-top2-rl-handoff-v1
python .\scripts\verify_top2_rl_handoff.py
$env:PYTHONPATH='.'
python .\tests\test_top2_rl_handoff.py
```

校验器必须报告 2 个不同 `deck_id`、5 个冻结输入哈希、20% holdout 和 `submission_replacement_authorized=false`。

## 2. 安装环境

Windows RTX 50 系：

```powershell
python -m pip install -r requirements-train.txt
```

Linux 服务器应先根据 CUDA 驱动安装匹配的 PyTorch，再安装其余依赖。不要盲目沿用 Windows 的 CUDA 12.8 wheel 地址。

## 3. 每个分支先采 100 局冒烟

primary：

```bash
python scripts/collect_top2_rollouts.py \
  --branch primary \
  --opponents cross-top2 \
  --games-per-opponent 100 \
  --device cpu \
  --output-root experiments/adapter_top2_rl_rollouts
```

reserve：

```bash
python scripts/collect_top2_rollouts.py \
  --branch reserve \
  --opponents cross-top2 \
  --games-per-opponent 100 \
  --device cpu \
  --output-root experiments/adapter_top2_rl_rollouts
```

每个 summary 必须满足：`games=100`、`exceptions=0`、`illegal_actions=0`。脚本按局稳定分成 80% train、10% valid、10% test；PPO 加载器遇到 valid/test 会直接报错，因此训练时必须把 `--rollouts` 指向该分支的 `train/` 子目录或包含仅 train 文件的目录。

## 4. 生成 Top2 专属 V1 队列

以 primary 为例：

```bash
python scripts/select_top2_v1_candidates.py \
  --rollouts experiments/adapter_top2_rl_rollouts/<RUN_ID>/primary \
  --deck-id top2-primary-crustle-kangaskhan-cage-v1 \
  --output experiments/adapter_top2_rl_rollouts/<RUN_ID>/primary/v1_candidates.jsonl \
  --max-items 5000

python scripts/run_top2_v1_reanalysis.py \
  --queue experiments/adapter_top2_rl_rollouts/<RUN_ID>/primary/v1_candidates.jsonl \
  --deck data/high_score_decks/crustle_kangaskhan_cage/deck.csv \
  --deck-id top2-primary-crustle-kangaskhan-cage-v1 \
  --output experiments/adapter_top2_rl_rollouts/<RUN_ID>/primary/v1_labels.jsonl \
  --max-items 100
```

reserve 使用自己的 `deck_id` 和牌表路径。队列只接受 train split；valid/test 的低置信状态只用于回归分析，不能回流训练。

## 5. 先跑 1 batch PPO 冒烟

```bash
python scripts/train_top2_ppo.py \
  --branch primary \
  --rollouts experiments/adapter_top2_rl_rollouts/<RUN_ID>/primary/train \
  --output experiments/adapter_top2_rl_ppo/primary-smoke \
  --device auto \
  --epochs 1 \
  --batch-size 16 \
  --max-batches 1
```

确认生成 `last.pt` 和 `metrics.json`，loss 全部为有限值后，再去掉 `--max-batches`。正式默认是 4 epoch、clip 0.15、KL 0.05、最大梯度范数 0.5。primary 使用主要预算，reserve 的 rollout 和训练预算默认为 primary 的 40%。

## 6. 交换先后手 Arena

```bash
python scripts/evaluate_top2_ppo.py \
  --branch primary \
  --checkpoint experiments/adapter_top2_rl_ppo/primary-smoke/last.pt \
  --games 400 \
  --device cpu \
  --output reports/top2_ppo_primary_arena.json
```

正式 candidate 至少要完成：

- 0 异常、0 非法动作；
- 与初始 Adapter 交换先后手 Arena；
- valid+test 20% 冻结回归；
- 对 Random、FirstMin、交叉 Top2 和历史 best 的对手矩阵；
- 推理 p95、回退率和模型动作覆盖率；
- primary 与 reserve 分别保存 checkpoint、配置、数据 manifest 和 SHA-256。

## 停止条件

以下任一情况出现就停止扩大训练：冒烟存在异常或非法动作、训练加载到非 train split、`deck_id` 不匹配、KL 或 loss 非有限、回归集退化、Arena 收益不稳定。即使 Arena 通过，本包也不授权替换正式提交；仍需单独完成最终发布评审。
