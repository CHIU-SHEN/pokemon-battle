# Top2 Arena 服务器交接

## 目标

使用冻结的 `SL-0-shared` 主干和 10 个已通过离线门槛的 Deck Adapter，
完成在线代理接入检查、45 组内部循环赛、固定外部对手矩阵和 Top2 主备冻结。

最终保留：

- 第 1 名：`primary` 主卡组；
- 第 2 名：`reserve` 备选卡组。

两套都必须保存独立牌表、Adapter、提交包、评估报告和 SHA-256。只有主卡组
通过最终发布门槛后才允许更新 `submission/deck.csv`。

## 包名

```text
server_uploads/pokemon-tcg-top2-arena-handoff-v1.tar.gz
server_uploads/pokemon-tcg-top2-arena-handoff-v1.tar.gz.sha256
```

该包不包含 5.4GB 监督训练 JSONL；Arena 不需要重新训练共享主干。

## 环境

- Python 3.11
- 与服务器 CUDA 匹配的 PyTorch 2.2+
- NumPy 1.26+
- 对战引擎依赖已包含在 `submission/cg/`

安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-train.txt
```

## 从上传文件开始逐步执行

### 第 1 步：检查服务器

```bash
nvidia-smi
python3.11 --version
df -h .
```

要求：

- Python 3.11；
- PyTorch 能识别 CUDA；
- 至少预留 5GB 空间；
- Arena 主要消耗 CPU，对局并行前先记录 CPU 核数。

### 第 2 步：解压与校验

```bash
sha256sum -c pokemon-tcg-top2-arena-handoff-v1.tar.gz.sha256
tar -xzf pokemon-tcg-top2-arena-handoff-v1.tar.gz
cd pokemon-tcg-top2-arena-handoff-v1
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-train.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python scripts/verify_top2_handoff.py
```

校验脚本必须报告：

- 10/10 牌表存在且每副 60 张；
- 10/10 Adapter schema、candidate ID 和基础数据哈希一致；
- `primary` / `reserve` 两个最终角色存在；
- 综合分权重合计为 1；
- Alakazam 重训复评为 `advance`、0 非法预测。

如果 `verify_top2_handoff.py` 失败，立即停止，不要启动比赛或修改 checkpoint。

### 第 3 步：理解当前阶段——不要重新训练

10 个 Adapter 已经训练完成，本包内已经包含：

```text
artifacts/sl0_shared_full/best.pt
artifacts/adapters_top10/<candidate>/best.pt
data/high_score_decks/<candidate>/deck.csv
```

因此本阶段没有“启动 10 个 Adapter 训练”的命令。服务器要做的是把这些
checkpoint 接入在线代理并运行 Arena。不要执行 `train_adapter.py`，因为本包
刻意不包含 5.4GB `training_decisions_v1.jsonl`。

只有项目负责人明确要求重新训练时，才另外上传冻结训练 JSONL，并使用：

```bash
python src/train/train_adapter.py \
  --view data/adapter_views/<candidate>/view.json \
  --data data/training/training_decisions_v1.jsonl \
  --base-checkpoint artifacts/sl0_shared_full/best.pt \
  --output artifacts/adapters_retrain/<candidate> \
  --epochs 4 \
  --batch-size 256 \
  --num-workers 1 \
  --device cuda
```

这条命令不属于当前 Top2 Arena 任务。

### 第 4 步：先完成 Adapter 在线推理接入

当前仓库已有 PyTorch Adapter 模型定义和 checkpoint 加载逻辑：

```text
src/train/shared_model.py
src/train/adapter_model.py
src/train/eval_adapters.py
```

需要新增一个候选代理构建入口，使每个候选能够同时加载：

1. 对应的 `deck.csv`；
2. 共享主干 `best.pt`；
3. 对应 Adapter `best.pt`；
4. observation 特征与动态合法 option；
5. 合法动作安全回退。

在线接入的最低验收条件：

- 10/10 候选均能启动；
- 每个候选返回自己的 60 张牌表；
- Adapter candidate ID 与牌表目录一致；
- 模型动作始终在合法 option 内；
- 模型异常时回退到安全动作；
- 不读取隐藏信息；
- 保存每次模型选择耗时和回退次数。

在这一步完成前，禁止启动正式 45 组循环赛。离线评估器不能替代在线代理。

### 第 5 步：运行在线 smoke test

为每个完成接入的候选运行：

```bash
python eval/run_match.py \
  --agent0 <candidate_agent_path> \
  --agent1 random \
  --games 10 \
  --out-dir experiments/top2_adapter_smoke/<candidate>/vs_random

python eval/run_match.py \
  --agent0 <candidate_agent_path> \
  --mirror \
  --games 10 \
  --out-dir experiments/top2_adapter_smoke/<candidate>/mirror
```

`<candidate_agent_path>` 必须替换为在线接入阶段实际生成的候选代理目录或
`main.py`。10/10 候选都必须满足：

- 0 exceptions；
- 0 illegal actions；
- 能完成 20 局；
- P95 决策耗时低于项目门槛；
- deck hash 与 candidate ID 一致。

### 第 6 步：启动正式 Arena

smoke test 10/10 通过后再启动：

1. 10 套完整单循环，共 45 个组合；
2. 每个组合使用同一比赛配置并交换先后手；
3. 同时运行 Random、Sample、Exploiter-FirstMin、V0-best 和稳定历史模型；
4. 所有结果写入新的 `experiments/top2_adapter_arena/`，禁止混入历史目录；
5. 每轮保存 `games.json`、`summary.json` 和总榜 checkpoint，支持中断续跑。

正式矩阵尚需在在线代理接入后实现统一编排脚本。实现脚本时必须读取：

```text
data/high_score_decks/top2_selection_policy.json
```

不得在代码里重新发明另一套权重。

### 第 7 步：Top2 冻结

综合分：

```text
0.50 * internal_round_robin
+ 0.30 * external_baseline_score
+ 0.10 * worst_matchup_score
+ 0.10 * stability_and_latency
```

执行：

- 前四名追加复赛；
- 前两名追加决赛和消融；
- 第 1 名标记为 `primary`；
- 第 2 名标记为 `reserve`；
- 第 2、3 名置信区间无法区分时追加对局；
- 两套都保存独立牌表、Adapter、提交包、评估报告和 SHA-256；
- 筛选期间不覆盖正式 `submission/deck.csv`。

### 第 8 步：服务器回传

回传内容至少包括：

```text
experiments/top2_adapter_smoke/
experiments/top2_adapter_arena/
reports/top2_arena_report.json
reports/top2_freeze_report.json
final_submissions/<primary-package>
final_submissions/<reserve-package>
```

打包：

```bash
tar -czf pokemon-tcg-top2-arena-results-v1.tar.gz \
  experiments/top2_adapter_smoke \
  experiments/top2_adapter_arena \
  reports/top2_arena_report.json \
  reports/top2_freeze_report.json \
  final_submissions

sha256sum pokemon-tcg-top2-arena-results-v1.tar.gz \
  > pokemon-tcg-top2-arena-results-v1.tar.gz.sha256
```

如果在线接入尚未完成，只回传代码、smoke 结果和阻塞说明，不得伪造 Top2
排名或空白决赛报告。

## 执行边界

当前包用于下一阶段开发和比赛。必须先完成 Adapter 在线推理接入与代理 smoke
test，再启动正式矩阵；不得把离线 Top-1 直接当作 Arena 胜率。

正式比赛口径：

1. 10 套完整单循环，共 45 个组合；
2. 共同对局设置并交换先后手；
3. 固定外部对手矩阵；
4. 前四追加复赛，前二追加决赛；
5. 按 `data/high_score_decks/top2_selection_policy.json` 冻结 Top2；
6. 第 2 名与第 3 名统计上无法区分时追加对局；
7. 不在筛选阶段覆盖正式 `submission/deck.csv`。

## 关键报告

- `reports/top10_adapter_offline_eval.md`
- `reports/top10_adapter_offline_eval.json`
- `reports/alakazam_battle_cage_split_retrain_eval.json`
- `data/adapter_views/alakazam_battle_cage_split/exact_supplement_training_v1.manifest.json`
