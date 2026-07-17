# 服务器 SL-0-history 与 GRU 训练指南

本阶段先验证 24 维显式历史特征是否相对冻结的 `SL-0-shared` 基线有稳定增益，再训练 `SL-1-gru`。GRU 不读取对手隐藏选择，而是编码“当前公开局面 + 上一步己方动作 + 相邻两次己方观察之间的公开局面差分”。`SL-0-history` 和新版 GRU 均已完成本机小规模冒烟；正式全量训练建议放到服务器。

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

优先上传新版自包含交接包：

```text
release_assets/pokemon-tcg-sl0-sl1-handoff-v2.tar.gz
SHA-256 见同目录 pokemon-tcg-sl0-sl1-handoff-v2.tar.gz.sha256
```

在服务器解压后，先阅读 `START_HERE.md` 并运行 `sha256sum -c SHA256SUMS`。该压缩包已包含下面列出的两份数据视图、序列索引、Combo 标签、SL-0 最优 checkpoint、训练与评估代码、测试和本指南，不需要另外克隆仓库。

需要以下文件：

- `data/training/training_decisions_history_v1.jsonl`（约 5.20 GiB）；
- `data/training/training_history_manifest_v1.json`；
- `data/training/training_decisions_v1.jsonl`；
- `data/training/sequence_trajectories_v1.jsonl`；
- `data/training/sequence_manifest_v1.json`；
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
python tests/test_sequence_index.py
python tests/test_sequence_model.py
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

## 5. SL-1-gru 输入契约

每条轨迹仍按 `game_id + current_player` 构建，保证训练和比赛推理使用相同视角。每个时间步包含：

- 当前 `SL-0` 状态与动态 options；
- 上一步己方实际动作对应的 option 特征均值；
- 当前与上一次己方观察之间的 24 维变化：前 22 个公开动态状态的差值、回合切换标志、可见日志存在标志。

该差分会显式呈现对手造成的公开结果，例如对手手牌/牌库/弃牌、场上宝可梦、主动位 HP/能量、双方奖赏卡和我方受伤的变化。`public_history` 可能为空，因此只作为附加标志，不能作为唯一对手信息源。禁止把录像中对手不可见的 option 或隐藏手牌写入输入。

实现位置：

- `src/train/transition_features.py`：差分和上一动作编码；
- `src/train/sequence_data.py`：窗口、前置样本、padding 和终点监督；
- `src/train/sequence_model.py`：SL-0 编码器、单层 GRU 与时间残差；
- `src/train/train_sequence.py`：训练、热启动、验证和 checkpoint。

## 6. GRU 服务器训练

先做冒烟：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 1 --batch-size 8 --window-length 8 \
  --num-workers 0 --max-train-windows 1000 --max-valid-windows 200 \
  --output artifacts/sl1_gru_server_smoke
```

再做长度 16 的正式训练：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 6 --batch-size 32 --window-length 16 \
  --num-workers 4 --learning-rate 3e-4 \
  --output artifacts/sl1_gru_full
```

默认从 `artifacts/sl0_shared_full/best.pt` 热启动。时间残差投影初始化为零，因此初始 policy/value 与 SL-0 一致；随后梯度才逐步启用差分、上一动作和循环状态。每个滑窗只在终点计算监督，避免同一历史样本在一个窗口内重复计权。

断点续训：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 8 --resume artifacts/sl1_gru_full/last.pt \
  --output artifacts/sl1_gru_full
```

冻结 test 评估：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_test.json
```

checkpoint 会同时校验原始训练集和序列索引哈希。对比时必须固定相同的 test endpoint 与窗口长度。

## 7. 执行顺序

```text
SL-0-shared 基线
  -> SL-0-history 全量 A/B
  -> 固定 Arena 验证
  -> SL-1-gru（公开局面差分 + 上一步己方动作）本机冒烟
  -> SL-1-gru 服务器全量训练与三模型 A/B
```

不要把双方训练记录简单交错后直接送入 GRU：比赛时看不到对手的隐藏选择。正式晋级仍比较 `SL-0-shared`、`SL-0-history` 和 `SL-1-gru`；GRU checkpoint 通过离线与 Arena 门槛后，才实现并验证 NumPy 在线运行时和提交包。
