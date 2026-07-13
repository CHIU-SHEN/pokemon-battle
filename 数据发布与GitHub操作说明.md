# Replay 数据发布与 GitHub 操作说明

## 1. 当前采用的结构

本项目不把大体积原始 Replay 放进普通 Git 历史，也不要求先建立数据库。

- GitHub 仓库保存代码、下载脚本、审计脚本、说明、manifest 和小型派生索引。
- `data/external/kaggle_replays/raw/` 保留在本机，已由 `.gitignore` 排除。
- 下载进度和日志仅用于本地排错，已由 `.gitignore` 排除。
- 后续训练可以直接读取 JSON；需要高效聚合时再转换成分片 JSONL/Parquet，并可用 DuckDB 查询。

## 2. 推送前检查

先确认大文件和认证信息都被忽略：

```powershell
git check-ignore -v data/external/kaggle_replays/raw/episode-85213633-replay.json
git check-ignore -v data/external/kaggle_replays/download_slow.log
git status --short
```

检查暂存区内是否意外包含大文件：

```powershell
git diff --cached --stat
git diff --cached --name-only
```

不要使用未经检查的 `git add .`。建议明确添加本次数据工程文件：

```powershell
git add .gitignore
git add scripts/audit_kaggle_replays.py
git add scripts/download_kaggle_replays.py
git add scripts/download_kaggle_replays_fast.py
git add scripts/select_core_combo_replays.py
git add data/external/README.md
git add data/external/acquisition_manifest.json
git add data/external/kaggle_replays/README.md
git add data/external/kaggle_replays/replay_index.json
git add data/external/kaggle_replays/core_combo_candidates.json
git add 数据发布与GitHub操作说明.md
```

再次检查后再提交：

```powershell
git diff --cached --stat
git commit -m "Add reproducible Kaggle replay data pipeline"
git push
```

## 3. 原始数据如何保存

当前优先保存在本地，并至少另做一份私人备份。可选择私人对象存储、移动硬盘或私人数据集；备份时保留 `replay_index.json`，以便用 SHA-256 检查损坏。

在确认 Kaggle Competition Rules 和 Terms 允许公开再分发后，才考虑公开数据。若使用 GitHub Release，应把数据压缩成多个小于 2 GiB 的分片，并发布分片 SHA-256 清单；不要把压缩包提交到 Git 历史。

## 4. 恢复下载

下载脚本按 episode ID 检查已存在文件，因此可以断点续传。恢复前应保留请求间隔，并在 `429 Too Many Requests` 时进行长时间退避。

恢复下载后重新执行：

```powershell
python scripts/audit_kaggle_replays.py
python scripts/select_core_combo_replays.py
```

