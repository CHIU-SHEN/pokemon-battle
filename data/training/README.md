# 正式训练数据

- `training_decisions_v1.jsonl`：合并 Kaggle 实际动作、本地失败对局、目标卡组 V0 和 V1 搜索 soft policy 的第一版正式训练集。
- `training_manifest_v1.json`：样本数、数据哈希、划分、监督来源、权重和泄漏审计。

监督优先级为 `V1 > V0 > 原始实际动作`。强制单选样本降权，异构 Kaggle 动作不会冒充目标卡组老师。当前 risk head 在没有可靠目标前保持关闭。

JSONL 是可重建的大型派生数据，默认不提交 Git；manifest、schema、构建脚本和来源摘要应提交。

本版 V1/V0 目标方来自当时的 Abomasnow 旧基线。它们仍是合法的跨卡组监督样本，但不能表述为 Top10 最终冠军的专属老师数据。Top10 冠军确定后，需要补充冠军卡组轨迹并生成新的数据版本，不能静默覆盖本版语义。

## SL-0-shared 训练入口

- 流式数据与动态 batch：`src/train/shared_data.py`
- State/Option/Deck 模型：`src/train/shared_model.py`
- 单 GPU/DDP 训练入口：`python -m src.train.train_shared`
- CPU/GPU 冒烟测试：`python tests/test_shared_training.py`
- 服务器操作指南：`docs/operations/服务器共享模型训练指南.md`

训练入口支持动态合法 option mask、加权 soft policy/value、AMP、梯度累积、checkpoint 和数据哈希约束的断点续训。当前没有可靠 risk target，因此仍不训练 risk head。

## 短序列派生索引

- `sequence_trajectories_v1.jsonl`：按 `game_id + current_player` 分组、按显式 `step` 排序的玩家视角轨迹索引。它只保存原 JSONL 的字节偏移和排序元数据，不复制特征与 options。
- `sequence_manifest_v1.json`：轨迹数、窗口数、长度分布、split/排序/可见性审计和索引哈希。
- 重建命令：`python scripts/build_sequence_index.py`。

`public_history` 是每个 observation 附带的可见事件块，不是稳定累积前缀。序列构造时必须将它保留在原时间步，不能用后一条样本的 history 回填早期步。

## 历史特征与 Combo 弱标签

- `combo_labels_v1.jsonl`：每个决策的 24 维过去/当前回合历史特征、未来 8/16 步里程碑和五类 Combo 弱标签。
- `combo_manifest_v1.json`：特征名、输入/目标防泄漏契约、标签覆盖率和文件哈希。
- 重建命令：`python scripts/build_combo_labels.py`。

`history_features` 只由该步之前的 8 个同视角决策、当前回合已执行动作和距最近 attach/evolve/attack 的步数构成。`future_8`、`future_16` 和 `combo_targets` 只是辅助监督，严禁作为模型输入。
