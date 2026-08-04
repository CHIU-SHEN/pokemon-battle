# 下一步（唯一操作入口）

更新日期：2026-08-04

## 1. 先提交当前候选

上传 `final_submissions/primary_budgeted_mcts_v1.tar.gz`，提交前核对：

```text
af58979d116e3db8072d49c368cdf36b6c7128e743b4deaf805076a8f57a81e0
```

上传后先看 Kaggle self-play Validation Episode，不要只根据本地 400 局判断成功。

## 2. 根据线上结果处理

- Validation 通过：保留该版本，观察 ladder，不要立即覆盖。
- 缺少 PyTorch、导入失败或超时：停止晋级，优先改为 NumPy runtime 或缩小搜索预算。
- 路径错误：归档顶层必须直接含 `main.py`、`deck.csv`、`cg/`；不要上传服务器交接包。
- 紧急回滚：以 `final_submissions/submission_flat_safe_v0.zip` 为代码参考，按当前格式重建。

## 3. 下一轮服务器工作

不要原样重复已完成的 MCTS v2。`server_uploads/pokemon-tcg-top2-mcts-pilot-v2.tar.gz`
只用于复现、断点恢复和提取流程。

从 `artifacts/top2-mcts-complete-results-20260804.tar.gz` 的已有访问分布开始做
MCTS 蒸馏 v2：

1. 先做 policy-only 蒸馏，避免不稳定的 value loss 主导更新。
2. 冻结共享主干或只解冻最后层，降低学习率，保留 reference KL。
3. 用 MCTS visit counts 作为 soft policy target，补齐联合动作和可变长 option 表示。
4. 每个分支先跑 100 局 smoke gate；有改善才扩到 400 局。
5. primary 优先；reserve 独立对照，不混合 checkpoint 和 `deck_id`。

网络断开不会中止 tmux 内进程；实例断电依赖 resilient 脚本和阶段性结果恢复。完整命令见
`docs/operations/TOP2_MCTS_SERVER_HANDOFF.md`。

## 4. 暂时不要做

- 不继续第四、第五轮 PPO；已有三轮没有稳定增益。
- 不恢复 GRU/history 旧路线，除非出现新的胜率证据。
- 不删除或重建 `data/`；完整数据是后续蒸馏与复现基础。
