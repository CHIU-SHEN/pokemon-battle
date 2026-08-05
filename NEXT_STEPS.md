# 下一步（唯一操作入口）

更新日期：2026-08-04

## 当前结论

- primary/reserve 共 400 局 MCTS 搜索验证与完整结果回收已经完成，不再重跑旧 pilot。
- 搜索策略有增益，但第一版无搜索 candidate 蒸馏没有稳定继承该增益。
- 下一研究目标是 **MCTS 蒸馏 v2**，不是 Kaggle 上传，也不是继续 PPO。

## 新对话的执行顺序

1. 先读本文件、`项目进度.md`、`docs/PROJECT_STRUCTURE.md`。
2. 检查 `artifacts/top2-mcts-complete-results-20260804.tar.gz` 及 `.sha256`；这是已有训练输入的权威归档。
3. 检查当前训练入口 `scripts/train_top2_mcts.py` 和测试 `tests/test_mcts_dataset.py`、`tests/test_mcts_train.py`，确认旧数据 schema 能否表达 visit-count soft targets、联合动作和可变长 option。
4. 先在本地设计并实现蒸馏 v2：policy-only；冻结共享主干或只解冻最后层；低学习率；保留 reference KL；primary/reserve 严格隔离。
5. 补齐测试并用小数据做 CPU smoke，禁止直接在服务器盲跑长任务。
6. 构建新的 `mcts-distill-v2` 服务器交接包；旧 `pokemon-tcg-top2-mcts-pilot-v2.tar.gz` 只用于参考和复现。
7. 服务器先跑 primary 小训练和 100 局 gate；只有安全指标为 0 且胜率改善才扩到 400 局，再独立处理 reserve。

## 蒸馏 v2 的验收重点

- 训练 target 使用 MCTS visit counts，而不是只复制最终选中动作。
- 优先验证 policy 改善；value loss 单独记录，不允许拖坏 policy。
- candidate 必须在无 MCTS 搜索时对冻结 best 做 swapped-seat 对局。
- `exceptions=0`、`illegal_actions=0`、`fallback_rate=0` 是硬门。
- 100 局只作筛选；晋级结论必须使用 400 局和 Wilson 区间。

## 服务器与断电

网络断开不会中止 tmux 内进程；实例断电依赖 resilient 脚本、阶段性 JSON 和
checkpoint 恢复。旧操作命令见 `docs/operations/TOP2_MCTS_SERVER_HANDOFF.md`，但其中
“重新跑旧 400 局”的段落已经过时，不得照做。

## 暂时不要做

- 不继续第四、第五轮 PPO；已有三轮没有稳定增益。
- 不恢复 GRU/history 旧路线，除非出现新的胜率证据。
- 不删除或重建 `data/`；完整数据是后续蒸馏与复现基础。

## 2026-08-05 混合 MCTS 路线更新

- v3 先执行 400 局强教师门控；教师达到 58% 后才采集 5,000 局并训练。
- 纯学生不再是唯一部署目标。v3 结果返回后固定比较 0、8、16、128 simulations。
- 0 表示纯学生；8/16 表示学生 policy/value 引导的小型在线 MCTS；128 是教师上界。
- 四组使用相同 swapped-seat 种子，先跑 100 局筛选，再跑 400 局正式评估。
- 晋级仍要求安全指标全零和至少 53% 有效胜率；在通过者中选择 P95 延迟最低的方案。
- 后续多对手/DAgger 数据迭代必须在教师门控通过之后进行。
- 权威说明：`docs/MCTS_HYBRID_POLICY_DECISION.md`。
