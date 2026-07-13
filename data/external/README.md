# 外部对局数据

获取/复核日期：2026-07-13。

## 运行环境

本项目后续统一使用 Conda 环境 `pokemon-tcg`，不创建项目内 `.venv`。Python 包使用 `uv pip install --python "$CONDA_PREFIX/python.exe" <package>` 安装到该环境。

## 当前已获取

- `kaggle_replays/raw`：通过 Kaggle 官方 Simulation Competition Replay CLI 下载的 Pokémon TCG AI Agent 完整 episode JSON。每个文件保留双方逐 step 的 `observation`、`action`、`reward`、`status`，以及 observation 内的 `logs`、`select` 等原始轨迹字段。
- `kagd_pokemon_tcg_battle_replay`：1 局完整 PTCG Live Battle Log、同一局的 12 回合 JSON，以及一个截断样例。原始日志和两个 JSON 只能计为同一局。
- `ptcg_bench`：MIT 许可的完整轨迹生成环境和 replay 记录脚本；仓库没有附带预生成对局，因此当前计为 0 局。
- `kaggle_pokemon_tcg_ai_battle`：6 个官方 Competition Data 文件，包括英/日卡牌 CSV 和本地 battle/visualize API。Competition Data 本身不含历史 replay；历史回放由独立的官方 replay 端点提供。

运行下面的命令复核所有回放并重建去重索引：

```powershell
conda activate pokemon-tcg
python scripts/audit_kaggle_replays.py
```

官方获取链路为：

```powershell
kaggle competitions leaderboard pokemon-tcg-ai-battle -s -v
kaggle competitions team-submissions <TEAM_ID> -v
kaggle competitions episodes <SUBMISSION_ID> -v
kaggle competitions replay <EPISODE_ID> -p data/external/kaggle_replays/raw
```

下载时以 episode ID 为唯一键；一局会关联双方提交，不能因出现在两个 submission 的 episode 列表中而重复计算。CLI 可直接重复运行，存在 `episode-<ID>-replay.json` 时应跳过。

批量任务先 dry-run 查看规模，再显式下载。例如榜首高分 submission `54603674`：

```powershell
python scripts/download_kaggle_replays.py --submission-id 54603674 --limit 20
python scripts/download_kaggle_replays.py --submission-id 54603674 --limit 20 --download
python scripts/audit_kaggle_replays.py
```

去掉 `--limit` 会拉取该 submission 返回的全部公开完整 episode；可重复提供 `--submission-id`，脚本会跨提交按 episode ID 去重并断点续传。

大规模下载使用 `scripts/download_kaggle_replays_fast.py`。它在单一进程中复用官方 Kaggle API 认证、对网络错误指数退避重试，并将状态写入 `kaggle_replays/download_progress.json`。当前 5000 局任务日志为 `kaggle_replays/download.log`，错误日志为 `kaggle_replays/download.err.log`。任务完成后运行审计脚本，届时再更新最终 manifest 统计。

与项目高分目标卡组相近的核心学习样本使用 `scripts/select_core_combo_replays.py` 生成。它从已知高分 seed replay 恢复完整 60 张牌表，并对排行榜回放双方牌表计算多重集合加权 Jaccard；结果写入 `kaggle_replays/core_combo_candidates.json`。默认阈值为 0.70，优先排序目标方获胜、相似度高且轨迹较完整的对局。

## 训练准入

- Kaggle replay 是 AI Agent 模拟对局，不是现实职业选手比赛。
- 仅将索引中 `valid_complete_trajectory=true` 的 JSON 送入后续规范化流程，原始文件保持只读。
- `battle-short.txt` 不完整，不进入训练。
- PTCGL 的 `battle.json` 只是摘要，不作为单独对局；`battle.battle-by-turn.json` 在人工对照原始日志前标为 `needs_verification`。
- 未下载 YouTube、Twitch 或 Bilibili 视频；平台可观看不代表允许第三方下载或训练。
- Kaggle 数据的使用与再分发必须遵守比赛 Rules、Kaggle Terms 和对应 API 限制；不要提交 token，也不要绕过登录或访问控制。

机器可读来源、commit、SHA-256、许可与质量说明见 `acquisition_manifest.json` 和 `kaggle_replays/replay_index.json`。
