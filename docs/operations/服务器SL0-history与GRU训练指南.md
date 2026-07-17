# 服务器 SL-0-history 与 GRU 训练指南

本阶段先验证 24 维显式历史特征是否相对冻结的 `SL-0-shared` 基线有稳定增益。只有离线 A/B 和固定 Arena 都通过后，才进入 `SL-1-gru`。`SL-0-history` 已在本机完成小规模冒烟；正式全量训练建议放到服务器，因为需要与基线使用全量数据和可复现配置做公平比较。

## 1. 已完成的本机验证

24 维特征只读取当前决策之前、同一玩家视角可见的动作，包括最近 8 步动作计数、当前回合资源使用计数和距关键动作的步数，不读取未来动作。

```bash
python -m src.train.train_history \
  --device cuda --epochs 3 --batch-size 64 --num-workers 0 \
  --max-train-samples 10000 --max-valid-samples 2000 \
  --shuffle-buffer 2048 --output artifacts/dev_smoke/sl0_history_10k
```

3 个 epoch 均正常完成，每轮约 6–7 秒；最佳 valid loss 为 `2.1112`，并生成全部 checkpoint 和运行记录。该结果只证明数据、反向传播、CUDA 和 checkpoint 流程可用，不能证明稳定增益。

## 2. 服务器文件与校验

优先上传自包含交接包：

```text
release_assets/pokemon-tcg-sl0-history-handoff-v1.tar.gz
SHA-256: 0D29DE10854F2354C589E4988C7D09BB706009DC8FE50070FD4597E589AC76A2
```

在服务器解压后，先阅读 `START_HERE.md`。该压缩包已经包含下面列出的数据、SL-0 最优 checkpoint、训练与评估代码、测试和本指南，不需要另外克隆仓库。

需要以下文件：

- `data/training/training_decisions_history_v1.jsonl`（约 5.20 GiB）；
- `data/training/training_history_manifest_v1.json`；
- `artifacts/sl0_shared_full/best.pt`；
- 当前仓库代码。

历史数据 SHA-256：

```text
35AF23BEC88280A879DD2A641A4EB315C3A5445EBA8FF5F3509A63EE13C80CE4
```

```bash
sha256sum data/training/training_decisions_history_v1.jsonl
python tests/test_shared_training.py
python tests/test_history_training.py
```

## 3. 正式训练 SL-0-history

先做服务器冒烟：

```bash
python -m src.train.train_history \
  --device cuda --epochs 1 --batch-size 64 --num-workers 0 \
  --max-train-samples 10000 --max-valid-samples 2000 \
  --output artifacts/sl0_history_server_smoke
```

再做全量单卡训练：

```bash
python -m src.train.train_history \
  --device cuda --epochs 6 --batch-size 256 --num-workers 4 \
  --shuffle-buffer 8192 --learning-rate 3e-4 \
  --output artifacts/sl0_history_full
```

训练默认从 `artifacts/sl0_shared_full/best.pt` 热启动。新增 24 维输入对应的权重初始化为零，因此开始时输出与原 SL-0 一致。若显存不足，先减小 `--batch-size`，再用 `--grad-accum` 保持有效 batch。

断点续训不需要再次传初始化 checkpoint：

```bash
python -m src.train.train_history \
  --device cuda --epochs 8 \
  --resume artifacts/sl0_history_full/last.pt \
  --output artifacts/sl0_history_full
```

## 4. 冻结测试集评估与 A/B 门槛

```bash
python -m src.train.eval_shared \
  --checkpoint artifacts/sl0_history_full/best.pt \
  --data data/training/training_decisions_history_v1.jsonl \
  --manifest data/training/training_history_manifest_v1.json \
  --split test --device cuda --batch-size 256 \
  --output reports/sl0_history_test.json
```

基线为 `reports/sl0_shared_test.json`：test loss `2.1134`、policy top-1 `60.27%`、非强制单选 top-1 `57.72%`、value MSE `0.8916`、非法 top-1 为 `0`。

进入 GRU 前至少满足：

1. 非强制单选 top-1 提升至少 `0.3` 个百分点，且非法 top-1 仍为 `0`；
2. test loss 不高于基线，value MSE 不出现明显退化；
3. 换一个 seed 复跑时，核心指标提升方向一致；
4. 固定 Arena 的胜率或 Combo 完成率提升，且推理延迟可接受。

若只有单次离线评估的小幅波动，不算稳定增益，应保留 SL-0 基线并停止扩大模型。

## 5. GRU 的执行边界

GRU 只在 SL-0-history 通过上述门槛后实现。本机仅做少量 batch 的 shape、mask、reset、padding、反向传播和 checkpoint 冒烟；全量短序列训练放服务器。

```text
SL-0-shared 基线
  -> SL-0-history 全量 A/B
  -> 固定 Arena 验证
  -> SL-1-gru 本机冒烟
  -> SL-1-gru 服务器全量训练与三模型 A/B
```

不要在 SL-0-history 结果出来前直接训练 GRU，否则无法判断收益来自显式历史特征还是循环状态，也会浪费服务器训练预算。
