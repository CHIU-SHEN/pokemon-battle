# 正式训练数据

- `training_decisions_v1.jsonl`：合并 Kaggle 实际动作、本地失败对局、目标卡组 V0 和 V1 搜索 soft policy 的第一版正式训练集。
- `training_manifest_v1.json`：样本数、数据哈希、划分、监督来源、权重和泄漏审计。

监督优先级为 `V1 > V0 > 原始实际动作`。强制单选样本降权，异构 Kaggle 动作不会冒充目标卡组老师。当前 risk head 在没有可靠目标前保持关闭。

JSONL 是可重建的大型派生数据，默认不提交 Git；manifest、schema、构建脚本和来源摘要应提交。

本版 V1/V0 目标方来自当时的 Abomasnow 旧基线。它们仍是合法的跨卡组监督样本，但不能表述为 Top10 最终冠军的专属老师数据。Top10 冠军确定后，需要补充冠军卡组轨迹并生成新的数据版本，不能静默覆盖本版语义。
