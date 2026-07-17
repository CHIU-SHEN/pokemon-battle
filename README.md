# Pokémon TCG AI Battle Agent

## 本地运行环境

Windows 本地开发统一使用现有 Conda 环境 `pokemon-tcg`，不创建或依赖项目内 `.venv`。该环境当前使用 Python 3.11。

```powershell
conda activate pokemon-tcg
python --version
uv --version
```

Python 依赖优先通过 `uv` 安装到这个 Conda 环境：

```powershell
uv pip install --python "$env:CONDA_PREFIX\python.exe" <package>
```

非交互执行统一使用 `conda run -n pokemon-tcg <command>`。Kaggle CLI 同样安装在该环境中。新版凭据文件为 `%USERPROFILE%\.kaggle\access_token`，其内容必须只有 Kaggle 设置页生成的原始 API token，不能包含标签、说明文字或命令。

这是一个面向 **Pokémon TCG AI Battle Challenge** 的卡牌对战智能体项目。根目录 [`项目进度.md`](项目进度.md) 提供总进度条和未完成事项；当前唯一研发主线见 `docs/plan/建模方案.md`，详细数据状态见 `docs/plan/数据进度与待办.md`。项目从稳定的卡组特化规则策略出发，逐步构建单步监督预训练、短决策序列与 Combo 学习、Top10 轻量适配择优、冠军卡组强化和固定评估晋级闭环。

当前推荐的正式提交物是：

```text
final_submissions/submission_flat_safe_v0.zip
```

该版本是当前历史基线提交包。新的学习模型尚未晋级，必须在固定评估中稳定超过基线后才能替换。

## 项目状态

| 版本 | 定位 | 当前状态 |
| --- | --- | --- |
| Flat Safe V0 | 卡组特化规则 + 安全兜底 | 推荐主提交 |
| Multi-module V0 | 模块化规则策略 | 本地开发与备份 |
| Search V1 | 信念采样 + 有限预算搜索 | 实验版，默认关闭 |
| 学习模型 | 单步共享主干 → 短序列/Combo 编码 → Top10 轻量 Adapter | `SL-0-shared` 已完成 6 epoch 全量训练、冻结 test 评估并冻结为单步基线；序列阶段尚未实现 |
| 排行榜 Top10 | 10 套候选牌表 | 10/10 静态合法且通过阶段 B，等待 Adapter 训练与循环赛 |
| M5 Elite Decks | MAP-Elites 历史候选 | 辅助候选池，未替换主卡组 |

旧 M0～M6 最小验证文档已移至 `docs/archive/mvp/`，只用于解释历史代码来源，不再作为当前路线或数据状态依据。

`SL-0-shared` 是必要的单步能力基线，不是学习模型的终点。它先学习“在当前局面选哪个合法动作”；通过验收后，还要把同局决策按时间顺序组成短窗口，训练模型理解检索、铺场、进化、贴能、换位和攻击等需要多步完成的 Combo。序列模型必须与单步基线做消融和 Arena 对比，确认整局胜率或 Combo 指标稳定提升后才能进入正式 Adapter 与冠军筛选链路。

## 核心工作流

```mermaid
flowchart LR
    O[Observation] --> P[局面解析与 GameLedger]
    P --> R[V0 卡组特化规则]
    P --> B[隐藏信息信念采样]
    B --> S[V1 有限预算搜索]
    R --> G[动作合法性检查]
    S --> G
    G -->|合法| A[返回 option 索引]
    G -->|异常或非法| F[安全兜底]
    F --> A
```

Kaggle 入口为 `submission/main.py` 中的：

```python
def agent(obs_dict):
    ...
```

- `agent(None)` 或初始无选择请求时返回 `deck.csv` 中的 60 张卡牌 ID。
- 常规回合先解析 observation，再尝试搜索或规则策略。
- 所有策略输出都会经过合法性检查；解析、搜索或规则发生异常时回退到安全动作。
- V1 搜索默认关闭，因此日常运行仍采用 V0 规则策略。

## 目录结构

```text
.
├── submission/                 # 当前模块化智能体与官方 cg 运行库
│   ├── main.py                 # agent(obs_dict) 入口
│   ├── deck.csv                # 当前 60 张主卡组
│   ├── agent/                  # 解析、规则、belief、搜索、估值与兜底
│   └── cg/                     # 对战引擎 Python 接口及跨平台动态库
├── eval/                       # 单场批量评估、联赛、统计与坏例回放
├── src/
│   ├── cards/                  # 卡牌数据库、标签、卡组规则与优化器
│   └── train/                  # 蒸馏数据、特征、训练与 reanalysis
├── tests/                      # 回归、战术、搜索、卡组和蒸馏测试
├── scripts/                    # 数据审计、转换、V1 重分析和提交构建
├── jobs/                       # Slurm 批处理脚本
├── data/
│   ├── external/              # 外部原始数据及来源清单
│   ├── processed/             # 统一决策样本、审计和转换摘要
│   └── reanalysis/            # V1 候选队列与搜索标签
├── docs/
│   ├── plan/                  # 唯一主方案与数据进度
│   ├── research/              # 数据源和卡组研究
│   ├── operations/            # 发布与维护说明
│   └── archive/mvp/           # 历史最小验证文档
├── artifacts/                  # 完整训练运行产物与开发冒烟产物
├── models/                     # 历史模型目录说明；SL-0 以完整运行目录为准
├── experiments/               # 历史对局结果与联赛报告
├── logs/bad_cases/             # 可回放的失败对局
├── final_submissions/          # 冻结目录及 zip 提交包
└── reports/                    # 最终冻结报告
```

`experiments/`、`logs/` 和 `final_submissions/` 中包含较多历史产物及提交副本；研发主线代码集中在 `submission/`、`eval/`、`src/`、`scripts/` 和 `tests/`。

## 环境要求

- Python 3.11 或更高版本；项目当前产物也在 Python 3.13 环境中使用过。
- NumPy：仅蒸馏训练和 reanalysis 需要。
- pytest：推荐用于一次性执行全部测试。
- CPU 即可；当前正式提交、回归评估和线性蒸馏均不要求 GPU。

建议创建独立环境：

```bash
conda create -n pokemon-tcg python=3.11 -y
conda activate pokemon-tcg
pip install numpy pytest
```

请始终从项目根目录执行下文命令。`submission/cg/` 已包含 Windows、Linux、macOS 和 Linux ARM64 对应的动态库，加载逻辑会根据当前平台选择文件。

## 快速开始

### 1. 验证对战引擎

```bash
python submission/test_sim.py
```

### 2. 运行规则智能体对随机基线

```bash
python eval/run_match.py --agent0 submission --agent1 random --games 100
```

结果默认写入：

```text
experiments/YYYYMMDD_HHMMSS_submission_vs_random/
├── games.json
└── summary.json
```

常用评估方式：

```bash
# 镜像对局
python eval/run_match.py --agent0 submission --mirror --games 100

# 与 first-min 弱规则基线对战
python eval/run_match.py --agent0 submission --agent1 first-min --games 100

# 评估另一个 main.py
python eval/run_match.py --agent0 path/to/main.py --agent1 submission --games 100

# 保存回归 observation 和失败对局
python eval/run_match.py --games 100 --save-fixtures 20 --bad-case-dir logs/bad_cases/manual
```

`summary.json` 包含胜负、平均步数、异常、非法动作、决策耗时及 p95 延迟等指标。

### 3. 执行测试

```bash
pytest -q
```

也可以按交付顺序直接执行各测试脚本：

```bash
python tests/test_regression.py
python tests/test_fallback.py
python tests/test_m3_search.py
python tests/test_card_db.py
python tests/test_tactics.py
python tests/test_deck_optimizer.py
python tests/test_distill_pipeline.py
python tests/test_flat_submission.py
```

## 评估与分析

### 固定基线联赛

```bash
python eval/league.py --candidate submission --games 500
```

默认对手为 `Random`、`Sample` 和 `Exploiter-FirstMin`。如果本地没有项目所引用的官方 Sample 路径，可显式限制为仓库内可用基线：

```bash
python eval/league.py --candidate submission --games 500 --baselines Random,Exploiter-FirstMin
```

### 统计晋级判断

```bash
python eval/stats.py experiments/<run>/summary.json --baseline-win-rate 0.5
```

脚本使用 Wilson 95% 区间，并结合异常、非法动作、样本量和相对基线提升给出“晋级 / 观察 / 淘汰”结论。

### 回放失败用例

```bash
python eval/replay_case.py logs/bad_cases/<case>.json --agent submission
```

`--agent` 可以重复指定，用于比较多个策略在同一局面上的选择。

## 启用 V1 搜索

模块化入口的搜索由环境变量控制，默认关闭：

```bash
PTCG_ENABLE_SEARCH=1 python eval/run_match.py --agent0 submission --agent1 random --games 100
```

PowerShell 写法：

```powershell
$env:PTCG_ENABLE_SEARCH = "1"
python eval/run_match.py --agent0 submission --agent1 random --games 100
```

可调参数：

| 环境变量 | 默认值 | 含义 |
| --- | ---: | --- |
| `PTCG_SEARCH_CANDIDATES` | 6 | 最大候选动作数 |
| `PTCG_SEARCH_PARTICLES` | 3 | belief 粒子数 |
| `PTCG_SEARCH_NODE_BUDGET` | 64 | 节点预算 |
| `PTCG_SEARCH_TIME_BUDGET` | 0.035 | 单次搜索时间预算（秒） |
| `PTCG_SEARCH_SWITCH_MARGIN` | 175 | 搜索替换规则动作的分数阈值 |

搜索版需要先通过稳定性、非法动作和耗时检查，再与 V0 做足量对局比较；当前不建议直接替换正式提交。

## 卡牌知识库与卡组优化

生成或刷新卡牌数据库、标签，并检查当前卡组：

```bash
python src/cards/card_db.py
python src/cards/tags.py
python src/cards/deck_rules.py submission/deck.csv
```

运行候选生成与 MAP-Elites 筛选：

```bash
python src/cards/deck_optimizer.py --per-archetype 150
```

主要输出包括：

- `data/cards.json`：统一卡牌数据库；
- `data/card_tags.json`：规则标签；
- `data/deck_candidates.json`：候选与代理评分；
- `data/deck_elites/*.csv`：不同特征格的精英卡组；
- `data/deck_coevolution_plan.json`：后续实战联赛计划。

代理评分只用于缩小候选空间，不能替代真实对局评估。替换 `submission/deck.csv` 前应先检查合法性，再运行固定基线联赛和完整回归测试。

## 当前数据与老师链路

正式数据链路不再使用旧的 50 条 fixture 训练产物：

```bash
python scripts/audit_existing_data.py
python scripts/convert_bad_cases.py --with-v0
python scripts/convert_kaggle_replays.py
python scripts/select_v1_candidates.py --max-items 5000
python scripts/run_v1_reanalysis.py --max-items 5000
```

主要摘要位于 `data/processed/` 和 `data/reanalysis/`。大体积 JSONL 可由原始日志重建，默认不提交 Git。`src/train/collect_distill.py`、`train_distill.py` 和 `reanalysis.py` 仅保留为开发回归工具，默认写入被忽略的 `artifacts/dev_smoke/`。

## 构建与冻结提交包

构建 raw-exec 兼容的 Flat V0 包：

```bash
python scripts/build_flat_submission.py --name safe_v0
```

默认生成：

```text
final_submissions/submission_flat_safe_v0/
final_submissions/submission_flat_safe_v0.zip
```

完整冻结全部候选并生成报告：

```bash
python scripts/freeze_final.py
```

冻结操作只重新生成当前基线 `submission_flat_safe_v0` 及 `reports/final_freeze_report.json`，建议仅在完整测试通过、准备交付时执行。

Flat 包必须满足：

- `main.py` 不依赖 `__file__`；
- `main.py` 不导入 `agent/` 子包；
- zip 根目录直接包含 `main.py`、`deck.csv` 和 `cg/`；
- `agent(None)` 返回恰好 60 个卡牌 ID；
- raw `exec(main_py_code, env)` 和本地回归测试均通过。

## Slurm 作业

`jobs/` 提供各阶段 CPU 评估、联赛、卡牌测试、卡组搜索、蒸馏和最终回归脚本。例如：

```bash
PROJECT_DIR=/path/to/project sbatch jobs/final_regression.slurm
```

其他脚本包括 `m2_league.slurm`、`m3_search_eval.slurm` 和 `m5_deck_search.slurm`。旧 `m6_distill.slurm` 已随最小验证产物移除；正式训练作业将在训练集 schema 冻结后重新建立。

## 开发注意事项

- 优先保证动作合法、无异常、无超时，再比较胜率。
- 新策略应保留 V0 规则和 `safe_action` 作为兜底。
- 不要用少量 smoke 对局直接晋级策略或卡组。
- 修改 observation 解析、规则分支或卡组后，至少运行回归、战术和兜底测试。
- 修改搜索后额外运行 `test_m3_search.py`；修改卡组后运行卡组合法性与优化器测试；修改提交构建逻辑后运行 `test_flat_submission.py`。
- 正式上传优先使用已经冻结和复验的 `submission_flat_safe_v0.zip`。

## 文档入口

- `docs/plan/建模方案.md`：唯一研发主线、训练路线和晋级原则。
- `docs/plan/数据进度与待办.md`：当前数据数量、已完成事项和下一步。
- `docs/research/`：回放数据源、真实比赛和高分卡组研究。
- `docs/operations/`：数据发布与仓库维护说明。
- `docs/archive/mvp/`：旧 M0～M6 最小验证文档，仅供历史参考。
