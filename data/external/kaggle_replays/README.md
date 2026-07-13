# Kaggle AI Battle Replay 数据目录

本目录保存 Pokémon TCG AI Battle 的官方 Simulation Competition Replay 数据和派生索引。

## 目录约定

| 路径 | 用途 | GitHub |
| --- | --- | --- |
| `raw/episode-*-replay.json` | 官方完整双 Agent 逐步轨迹 | 不提交 |
| `replay_index.json` | 完整性、步数、终局、SHA-256 索引 | 提交 |
| `core_combo_candidates.json` | 与目标高分牌表相似的对局清单 | 提交 |
| `download_progress.json` | 本地下载断点状态 | 不提交 |
| `download*.log` | 本地运行日志 | 不提交 |

原始 Replay 可由项目脚本和合法的个人 Kaggle 认证重新获取。不要提交 Kaggle token，也不要将本目录中的原始数据公开再分发，除非已经确认 Kaggle Competition Rules 与 Terms 允许。

## 本地重建

```powershell
conda activate pokemon-tcg
python scripts/audit_kaggle_replays.py
python scripts/select_core_combo_replays.py
```

审计完成后，应满足：

- 每个文件可以解析为 JSON；
- 双方终局状态为 `DONE`；
- `valid_complete_trajectory=true`；
- 索引中的 SHA-256 与本地原始文件一致。

