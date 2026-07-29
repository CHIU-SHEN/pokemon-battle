# Top2 Arena 收尾与仓库整理报告（2026-07-29）

## 范围与安全模式

本次使用安全整理模式，只处理 Top2 Arena 本轮新增的临时 smoke、断点测试报告、生成产物忽略规则和状态文档。未删除或移动正式 10,100 局 Arena 证据、冻结 checkpoint、训练数据、历史实验、服务器交接包或现有正式提交。

## 结果评价

- 初赛 8,500 局、Top4 复赛 1,200 局、Top2 决赛 400 局，共 10,100 局。
- 170 个首轮任务、12 个复赛任务和 2 个决赛任务全部完成，0 失败、0 非法动作。
- 三阶段顺序一致：primary=`crustle_kangaskhan_cage`，reserve=`crustle_kangaskhan_petrel`。
- 最终决赛 primary 为 237:163，胜率 59.25%，Wilson 95% 为 54.37%～63.96%。
- 第 2 名与第 3 名在首轮内部胜分率区间已明显分离，不需要追加候选筛选局。

这是一份可信的 Top2 筛选结论，但还不是发布结论。两套 Top2 共享 Crustle/Mega Kangaskhan 核心，说明该体系在当前矩阵中稳定领先，也意味着主备对未知克制环境可能存在相关风险。后续应使用更多异构对手和独立分支数据验证，而不是把 reserve 当作完全不同的风险对冲。

## 已整理

- 保留正式聚合报告：`reports/top2_arena_report.json`、`top4_playoff_report.json`、`top2_final_report.json`、`top2_freeze_report.md/.json`。
- 保留可断点续跑编排与冻结脚本：`scripts/run_top10_adapter_arena.py`、`scripts/finalize_top2_arena.py`。
- `.gitignore` 增加正式 Arena、Top4 复赛、Top2 决赛和开发 smoke 原始目录；Git 只收录紧凑报告，不收录数百个可再生成的逐场日志。
- 删除 2 局开发 smoke 和断点参数测试报告；正式结果未受影响。
- 删除本轮验证重新生成的 6 个 `__pycache__/` 目录。
- 同步更新 `README.md`、`项目进度.md`、`docs/plan/建模方案.md`、`docs/plan/数据进度与待办.md` 和 `reports/README.md`。

## 整理指标

| 项目 | 整理前 | 整理后 |
| --- | ---: | ---: |
| 开发 smoke / 错误参数测试文件 | 9 个，23,075 字节 | 0 |
| 本轮 Python 缓存目录 | 6 个 | 0 |
| 正式逐场 Arena 文件 | 557 个，本地保留 | 557 个，本地保留并由 Git 忽略 |
| 正式紧凑报告 | 5 个 | 5 个，保留为项目入口 |

## 验证结果

- Adapter 在线 fixture、Top10 牌表合法性和 Adapter 采样视图测试通过。
- Top2 交接校验通过：10 个 Adapter、10 副牌表、角色定义、在线 smoke 和基础数据哈希一致。
- Top2 冻结报告可从三阶段报告重新生成；4 个正式 JSON 均通过解析。
- 修改文档中的本地 Markdown 链接检查通过；`git diff --check` 通过，仅有预期的 Windows CRLF 提示。
- `submission/deck.csv` 无改动。

## 已知限制

- 本地缺少官方 Sample baseline 源码；外部矩阵实际使用 Random、Exploiter-FirstMin、V0-current 和 V0-best。
- 比赛引擎不暴露内部 RNG seed；通过交换先后手和扩大样本降低偏差，但不是严格配对 seed 实验。
- 当前只冻结 Top2 选择；没有启动后续训练，没有构建学习模型发布包，也没有覆盖 `submission/deck.csv`。

## 下一步唯一动作

先决定 Top2 强化使用本机 RTX 5060 8GB 还是服务器，并确定 primary/reserve 的预算比例。随后为两套卡组分别建立严格隔离的 `deck_id` 数据流，先生成专属自比赛轨迹、低置信/失败局面和 V1 标签，冻结回归集后再决定是否启动监督微调或 masked PPO。

建议提交说明：

```text
完成 Top2 Arena 筛选并同步项目进度
```
