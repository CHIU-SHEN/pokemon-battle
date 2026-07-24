# Top10 Adapter 服务器交接

> 状态更新（2026-07-24）：首轮 10 套训练结果已回收、校验和离线复评。
> 补充数据转换和 `alakazam_battle_cage_split` 重训复核已完成，10/10 通过。
> 完整结果见 `reports/top10_adapter_offline_eval.md`。

## 目标

在冻结的 `SL-0-shared` 主干上顺序训练 10 个轻量 Deck Adapter。采样视图已经 10/10 通过审计，869,433 条记录中没有跨 split 对局。

## 依赖与硬件

- Python 3.11
- 与服务器 CUDA 匹配的 PyTorch 2.2+
- `pip install -r requirements-train.txt`
- 建议单卡显存至少 8GB；默认 batch size 256，如显存不足改为 128 或 64
- 训练数据约 5.4GB，建议预留至少 20GB 工作空间

## 文件来源

本增量包包含 Adapter 代码、10 份 sampling view、32.6MB 定向补充数据、冻结主干和校验清单。基础数据 `data/training/training_decisions_v1.jsonl` 可从既有 `pokemon-tcg-sl0-sl1-handoff-v3.tar.gz` 复用，避免重复上传。

两个压缩包解压到同一父目录后，在本包根目录创建只读软链接：

```bash
cd pokemon-tcg-adapter-handoff-v1
ln -s "$(realpath ../pokemon-tcg-sl0-sl1-handoff-v3/data/training/training_decisions_v1.jsonl)" \
  data/training/training_decisions_v1.jsonl
```

如果服务器上的 v3 目录名称不同，把 `realpath` 后面的路径替换为实际位置即可；不要复制这份 5.4GB 文件。

## 校验

在项目根目录执行：

```bash
sha256sum -c HANDOFF_SHA256SUMS
python tests/test_adapter_sampling_views.py
```

关键哈希：

- 基础训练集：`E8DC4DC2784A3505EAA159255A735A2C50B907DB66A5F9AB7759BEC326062370`
- 冻结主干：`73F7702197365EC29E0F1AB480581B024152DC8B25850968DB02972769449A8B`
- 定向补充集：`9002A50F6052D303199D86216D34F1500DC5C8FA7DC4369FFF55A903EA694217`

## 先做 smoke test

```bash
python src/train/train_adapter.py \
  --view data/adapter_views/alakazam_battle_cage_split/view.json \
  --data data/training/training_decisions_v1.jsonl \
  --base-checkpoint artifacts/sl0_shared_full/best.pt \
  --output artifacts/adapter_smoke \
  --epochs 1 --batch-size 8 --num-workers 0 --max-batches 1
```

## 正式训练

```bash
chmod +x scripts/train_top10_adapters.sh
EPOCHS=4 BATCH_SIZE=256 NUM_WORKERS=1 bash scripts/train_top10_adapters.sh
```

默认顺序训练，避免十个任务争抢同一张 GPU 和磁盘。产物位于 `artifacts/adapters_top10/<candidate>/`，每套包含 `best.pt`、`last.pt` 和 `metrics.json`。

### 补充数据修正

检查发现 `exact_supplement_v1.jsonl` 是 `observed_decision_v1`，没有
`supervision.soft_policy` 和 `supervision.head_weights`，不能直接传给
`train_adapter.py`。`scripts/train_top10_adapters.sh` 已改为默认不加载补充
文件；只有先把补充轨迹转换、校验为 `training_decision_v1` 后，才通过
`SUPPLEMENT=<converted.jsonl>` 显式启用。

正式转换已生成 7,494 条有效训练增量，0 重复、0 非法监督动作、0 跨
split。RTX 5060 重训 4 epoch 用时 24 分 23 秒，新旧 checkpoint 参数最大
差异仅 `3.5e-6`。纳入 335 条按整局隔离的补充 test 后，完整 452 条 exact
记录上 policy top-1 相对主干提升 17.37pp，最终 10/10 通过离线门槛。

## 回传内容

训练结束后打包并回传：

```bash
tar -czf adapters_top10_results.tar.gz artifacts/adapters_top10
sha256sum adapters_top10_results.tar.gz > adapters_top10_results.tar.gz.sha256
```

不要回传 5.4GB 基础训练 JSONL。收到结果后本地进行离线复评、非法动作检查、推理延迟检查和 Top10 循环赛。

首轮结果包 SHA-256：
`66702A8C85B1FFDDBB29293A33E750F023F671CC46B0B62800B7A9A4AF70FCB7`。
