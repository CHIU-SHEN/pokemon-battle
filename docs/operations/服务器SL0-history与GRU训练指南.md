# 服务器 SL-0-history 与 GRU 训练指南

本阶段比较 24 维显式历史特征、冻结的 `SL-0-shared` 基线与 `SL-1-gru`。GRU 不读取对手隐藏选择，而是编码“当前公开局面 + 上一步己方动作 + 相邻两次己方观察之间的公开局面差分”。截至 2026-07-20，GRU 已完成服务器 6 epoch 全量训练和首次冻结 test；`SL-0-history` 全量 A/B、GRU 详细复评、第二 seed 与固定 Arena 仍待完成。

## 0. 当前起点与最短执行路线

本指南保留了从零训练所需的完整命令，但当前服务器不应重跑已经完成的首轮 GRU。假定服务器已有：

- `artifacts/sl1_gru_full/best.pt`；
- `artifacts/sl1_gru_full/last.pt`；
- 第 2 节列出的训练数据和 manifest；
- CUDA 可用的 Python/PyTorch 环境。

从当前状态严格按以下编号执行：

1. 执行第 2 节的代码、文件和哈希检查；
2. 执行第 6.2 节，详细复评首轮 GRU 的 best/last；
3. 按第 6.3 节判断是否继续；未通过就停止，不跑第二 seed；
4. 通过后执行第 3～4 节，补齐 `SL-0-history` 全量对照；
5. 执行第 6.4 节，在独立目录训练第二 seed 并详细评估；
6. 下载第 8 节列出的结果，由本地完成三模型离线汇总；
7. 离线门槛通过后再开发在线运行时；当前指南不会直接产出 Arena 胜率。

所有命令都从交接包或仓库根目录执行。运行前先创建报告目录：

```bash
mkdir -p reports
```

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
release_assets/pokemon-tcg-sl0-sl1-handoff-v3.tar.gz
SHA-256 见同目录 pokemon-tcg-sl0-sl1-handoff-v3.tar.gz.sha256
```

在服务器解压后，先阅读 `START_HERE.md` 并运行 `sha256sum -c SHA256SUMS`。V3 包已包含下面列出的两份数据视图、序列索引、Combo 标签、SL-0 最优 checkpoint、首轮 GRU best/last checkpoint、首次 test 报告、训练与详细评估代码、测试和本指南，不需要另外克隆仓库，也不需要另外上传首轮 GRU 产物。

如果沿用第一次 GRU 训练目录而不是重新解压交接包，必须至少同步以下新版文件：

```text
src/train/eval_sequence.py
src/train/shared_data.py
```

其中 `shared_data.py` 包含序列嵌套 batch 的 CUDA 搬运修复，不能只上传评估脚本。

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
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
test -f data/training/training_decisions_v1.jsonl
test -f data/training/sequence_trajectories_v1.jsonl
test -f data/training/sequence_manifest_v1.json
test -f artifacts/sl0_shared_full/best.pt
test -f artifacts/sl1_gru_full/best.pt
test -f artifacts/sl1_gru_full/last.pt
sha256sum data/training/training_decisions_history_v1.jsonl
PYTHONPATH=. python tests/test_shared_training.py
PYTHONPATH=. python tests/test_history_training.py
PYTHONPATH=. python tests/test_sequence_index.py
PYTHONPATH=. python tests/test_sequence_model.py
```

预期历史数据 SHA-256 必须严格等于上面的值，所有 `test -f` 和测试命令退出码必须为 `0`。任一失败都先修复文件或环境，不继续训练/评估。

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

断点续训只用于同一次训练被中断的情况。首轮训练已正常完成，不要执行下面命令；否则会继续写入并覆盖 `artifacts/sl1_gru_full/last.pt`：

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

### 6.1 首次服务器结果

正式训练参数为长度 16、batch 32、学习率 `3e-4`、seed `20260717`、6 epoch。产物已归档到 `artifacts/sl1_gru_full/`：

- `best.pt`：epoch 1，valid loss `2.0841`、policy top-1 `60.61%`、value MSE `0.8645`；
- `last.pt`：epoch 5，valid loss `2.0905`、policy top-1 `63.09%`、value MSE `0.9419`；
- `reports/sl1_gru_test.json`：使用 best checkpoint 的首次 test，loss `2.0433`、policy top-1 `60.95%`、value MSE `0.8275`。

相对 `SL-0-shared`，首次 test 的总 loss 降低 3.32%，policy top-1 提升 0.68 个百分点，value MSE 降低 7.19%。这构成继续验证的正向证据，但旧报告没有非强制单选、来源分组、非法 top-1 和性能明细，不能单独作为正式晋级结论。

### 6.2 立即执行的详细复评

新版 `src/train/eval_sequence.py` 已补齐与 SL-0 同口径的总体、非强制单选、policy source、非法 top-1 和吞吐指标。分别评估两个 checkpoint，避免“总 loss 最优”和“policy 最优”选择目标不一致。以下命令不会修改 checkpoint：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_best_detailed_test.json

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_full/last.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_last_detailed_test.json
```

两个命令都应正常结束，并分别生成 JSON。先检查文件存在且 JSON 可解析：

```bash
python -c "import json; [json.load(open(p)) for p in ['reports/sl1_gru_best_detailed_test.json','reports/sl1_gru_last_detailed_test.json']]; print('OK: detailed GRU reports')"
```

### 6.3 首轮详细复评停止条件

以 `reports/sl0_shared_test.json` 的非强制 top-1 `57.72%` 为主基线。best/last 至少有一个同时满足：

1. `non_forced.policy_top1 >= 0.5802`，即相对 SL-0 至少提高约 0.3 个百分点；
2. `overall.loss <= 2.1134`；
3. `overall.value_mse` 不明显差于 `0.8916`；
4. `legality.illegal_top1_predictions == 0`；
5. 各主要 `by_policy_source` 没有无法解释的大幅退化。

若两个 checkpoint 都未通过，立即停止第二 seed 和 `SL-0-history` 之后的模型扩张，先分析 loss 权重、checkpoint 选择与历史输入。若至少一个通过，记录首轮候选 checkpoint，并继续下面步骤。离线通过只代表取得继续验证资格，不代表已经晋级。

### 6.4 第二 seed 独立复跑

第二 seed 固定为 `20260721`，输出到全新目录，禁止覆盖首轮 `artifacts/sl1_gru_full/`：

```bash
python -m src.train.train_sequence \
  --device cuda --epochs 6 --batch-size 32 --window-length 16 \
  --num-workers 4 --learning-rate 3e-4 --seed 20260721 \
  --output artifacts/sl1_gru_seed20260721
```

训练结束后同样详细评估 best/last：

```bash
python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_seed20260721/best.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_seed20260721_best_detailed_test.json

python -m src.train.eval_sequence \
  --checkpoint artifacts/sl1_gru_seed20260721/last.pt \
  --split test --device cuda --batch-size 64 --window-length 16 \
  --output reports/sl1_gru_seed20260721_last_detailed_test.json
```

不要根据结果再更换 seed。第二 seed 至少一个 checkpoint 也应满足第 6.3 节硬条件，且核心指标相对 SL-0 的方向与首轮一致；否则 GRU 暂不冻结。

## 7. 执行顺序

```text
SL-0-shared 基线
  -> SL-1-gru 服务器全量训练与首次 test ✅
  -> best/last 详细同口径复评
  -> SL-0-history 全量对照
  -> 第二 seed 独立复跑与详细复评
  -> 下载并汇总三模型离线结果
  -> 三模型固定 Arena + Combo + 时延验收
  -> 通过后实现 NumPy 在线运行时
```

不要把双方训练记录简单交错后直接送入 GRU：比赛时看不到对手的隐藏选择。正式晋级仍比较 `SL-0-shared`、`SL-0-history` 和 `SL-1-gru`；GRU checkpoint 通过离线与 Arena 门槛后，才实现并验证 NumPy 在线运行时和提交包。

## 8. 本轮必须下载的结果

完成上述服务器步骤后，下载以下文件，不需要下载 5 GiB JSONL：

```text
reports/sl1_gru_best_detailed_test.json
reports/sl1_gru_last_detailed_test.json
artifacts/sl0_history_full/best.pt
artifacts/sl0_history_full/last.pt
artifacts/sl0_history_full/metrics.jsonl
artifacts/sl0_history_full/run_config.json
reports/sl0_history_test.json
artifacts/sl1_gru_seed20260721/best.pt
artifacts/sl1_gru_seed20260721/last.pt
reports/sl1_gru_seed20260721_best_detailed_test.json
reports/sl1_gru_seed20260721_last_detailed_test.json
```

若训练脚本实际没有生成某个 history 元数据文件，以目录中的真实产物为准，但 checkpoint 与 test JSON 必须保留。下载完成后计算 SHA-256，并与服务器端 `sha256sum` 输出一起保存。

## 9. 本指南能够与不能够直接得到的结果

严格照做可以得到：

- 首轮 GRU best/last 的同口径详细 test；
- `SL-0-history` 全量 checkpoint 与详细 test；
- 第二 seed GRU 的完整 checkpoint 与详细 test；
- 是否值得开发在线运行时的离线结论。

严格照做仍不能直接得到：

- 固定 Arena 胜率；
- 在线回退率、整局 p95 延迟；
- NumPy 提交包表现。

这三项依赖尚未实现的 GRU 在线运行时。离线结果通过后，下一份操作指南应先完成 PyTorch/NumPy 一致性测试和代理接入，再运行 Arena；不能把离线 top-1 当作最终实战结论。
