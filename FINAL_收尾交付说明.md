# 最终收尾交付说明

## 最终选择

主提交建议使用：

```text
final_submissions/submission_flat_safe_v0.zip
```

原因：Kaggle validation runner 会以 raw `exec(main_py_code, env)` 风格执行 `main.py`，不保证存在 `__file__`，也不适合依赖 `agent/` 子包导入。当前最可靠的提交工程是 flat single-file V0：`main.py + deck.csv + cg/`，策略仍是卡组特化强规则 + 安全兜底。

## 冻结产物

- `final_submissions/submission_flat_safe_v0/` 与 `final_submissions/submission_flat_safe_v0.zip`：推荐上传的 Kaggle raw-exec 兼容 flat 规则版。
- `final_submissions/submission_safe_v0/` 与 `final_submissions/submission_safe_v0.zip`：本地多模块安全规则备份，不作为优先上传包。
- `final_submissions/submission_search_v1_experimental/` 与 `final_submissions/submission_search_v1_experimental.zip`：Search 默认开启的实验备份。
- `final_submissions/v2_distill_experimental/` 与 `final_submissions/v2_distill_experimental.zip`：V2 蒸馏复现实验档案，不是 Kaggle 提交包。
- `reports/final_freeze_report.json`：最终冻结 JSON 报告。

## 当前晋级判断

| 模块 | 状态 | 判断 |
|---|---|---|
| flat V0 规则策略 | 主提交 | raw-exec 兼容、稳定、轻量、无训练依赖 |
| 多模块 V0 | 本地备份 | 本地可用，但 Kaggle raw exec 下有包导入/`__file__` 风险 |
| V1 搜索 | 实验备份 | 5000 局稳定性通过，但没有证明稳定强于 V0 |
| M5 elite deck | 不晋级 | 有候选价值，但联赛未证明应替换当前 deck |
| V2 蒸馏 | 不晋级 | 只有 smoke 数据，value 指标不足，未接入主线 |

## 最终 smoke

- `submission_flat_safe_v0.zip` vs random：200 局 185-15-0，胜率 0.925，异常 0，非法动作 0，p95/decision 约 0.00020s。
- `submission_flat_safe_v0.zip` self-play：50 局无异常，非法动作 0。
- `submission_safe_v0.zip` vs random：200 局 184-16-0，胜率 0.920，异常 0，非法动作 0，p95/decision 约 0.00017s。
- `submission_search_v1_experimental.zip` vs random：100 局 92-8-0，胜率 0.920，异常 0，非法动作 0，p95/decision 约 0.00421s。
- 精简包复验：safe V0 200 局 186-14-0，胜率 0.930，异常 0，非法动作 0；Search V1 100 局 94-6-0，胜率 0.940，异常 0，非法动作 0。

Search 实验包在短测中没有形成稳定、显著、可推广的胜率优势，且耗时明显更高，所以不替代安全包。

## 提交前本地检查

```bash
cd /Users/hank/Desktop/pokemon

python3 tests/test_regression.py
python3 tests/test_fallback.py
python3 tests/test_m3_search.py
python3 tests/test_card_db.py
python3 tests/test_tactics.py
python3 tests/test_deck_optimizer.py
python3 tests/test_distill_pipeline.py
python3 tests/test_flat_submission.py

python3 scripts/freeze_final.py
```

flat 包必须额外满足：

- `main.py` 中不出现 `__file__`；
- `main.py` 不导入 `agent/` 子包；
- zip 内容只有 `main.py`、`deck.csv`、`cg/`；
- raw exec 测试通过：

```python
env = {"__builtins__": __builtins__}
exec(open("final_submissions/submission_flat_safe_v0/main.py").read(), env)
assert len(env["agent"](None)) == 60
```

## 服务器最终回归

CPU 即可，不需要 GPU。

```bash
ssh 172.23.160.47
cd /share/home/shenhanxi/pokemon
PROJECT_DIR=/share/home/shenhanxi/pokemon sbatch jobs/final_regression.slurm
```

当前 Slurm 输出在 `/share/home/shenhanxi/job.<JOBID>.out` 和 `/share/home/shenhanxi/job.<JOBID>.err`。

环境：

```bash
conda create -n pokemon-tcg python=3.11 -y
conda activate pokemon-tcg
pip install numpy pytest
```

当前最终回归和提交包不需要 `torch` 或 `scikit-learn`。只有未来把 V2 升级成神经网络训练时，才需要额外安装深度学习依赖。

## 止损规则

如果最后提交前时间紧，优先上传 `submission_flat_safe_v0.zip`。Search V1 只有在服务器和本地都重新证明无异常、无非法动作、且胜率显著提升后才考虑替换主提交。
