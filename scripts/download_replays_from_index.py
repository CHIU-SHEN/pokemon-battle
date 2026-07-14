"""Restore the exact raw Kaggle replay set recorded in replay_index.json."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


EPISODE_FILE_RE = re.compile(r"episode-(\d+)-replay\.json$")


def save_progress(path: Path, **values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values["updated_at"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_episode_ids(index_path: Path) -> list[int]:
    data = json.loads(index_path.read_text(encoding="utf-8"))
    result: list[int] = []
    seen: set[int] = set()
    for item in data.get("replays", []):
        match = EPISODE_FILE_RE.search(str(item.get("path", "")).replace("\\", "/"))
        if not match:
            raise ValueError(f"索引中的回放路径无法解析: {item.get('path')!r}")
        episode_id = int(match.group(1))
        if episode_id not in seen:
            seen.add(episode_id)
            result.append(episode_id)
    if not result:
        raise ValueError(f"索引中没有回放记录: {index_path}")
    return result


def existing_episode_ids(output: Path) -> set[int]:
    result: set[int] = set()
    for path in output.glob("episode-*-replay.json"):
        match = EPISODE_FILE_RE.fullmatch(path.name)
        if match and path.stat().st_size > 0:
            result.add(int(match.group(1)))
    return result


def download(kaggle: str, episode_id: int, output: Path, retries: int) -> None:
    last_error = ""
    for attempt in range(1, retries + 1):
        result = subprocess.run(
            [kaggle, "competitions", "replay", str(episode_id), "-p", str(output)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout)[-2000:]
        if attempt < retries:
            wait = min(2 ** (attempt - 1), 30)
            print(f"  第 {attempt}/{retries} 次失败，{wait} 秒后重试")
            time.sleep(wait)
    raise RuntimeError(last_error or f"Kaggle CLI 下载 episode {episode_id} 失败")


def main() -> None:
    parser = argparse.ArgumentParser(description="按 replay_index.json 恢复精确的原始 Kaggle 回放集")
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--delay", type=float, default=2.0, help="每次成功下载后的等待秒数")
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="本次最多下载多少条；0 表示不限")
    parser.add_argument("--kaggle", default=shutil.which("kaggle") or "kaggle")
    args = parser.parse_args()

    episode_ids = load_episode_ids(args.index)
    args.output.mkdir(parents=True, exist_ok=True)
    existing = existing_episode_ids(args.output)
    all_missing = [episode_id for episode_id in episode_ids if episode_id not in existing]
    selected = all_missing[: args.limit] if args.limit > 0 else all_missing

    print(
        f"索引总数={len(episode_ids)}，已存在={len(episode_ids) - len(all_missing)}，"
        f"待下载={len(all_missing)}，本次计划={len(selected)}"
    )
    save_progress(
        args.progress,
        status="complete" if not all_missing else "downloading",
        indexed=len(episode_ids),
        existing=len(episode_ids) - len(all_missing),
        remaining=len(all_missing),
        selected_this_run=len(selected),
        downloaded_this_run=0,
        failed_episode_id=None,
    )
    if not selected:
        print("原始回放已经齐全，无需联网下载。")
        return

    if shutil.which(args.kaggle) is None and not Path(args.kaggle).exists():
        raise RuntimeError("找不到 kaggle 命令。请先安装 kaggle，并配置 API 凭据。")

    downloaded = 0
    for number, episode_id in enumerate(selected, 1):
        print(f"[{number}/{len(selected)}] 下载 episode {episode_id}")
        try:
            download(args.kaggle, episode_id, args.output, args.retries)
        except Exception:
            current_existing = existing_episode_ids(args.output)
            save_progress(
                args.progress,
                status="failed",
                indexed=len(episode_ids),
                existing=len(current_existing & set(episode_ids)),
                remaining=len(set(episode_ids) - current_existing),
                selected_this_run=len(selected),
                downloaded_this_run=downloaded,
                failed_episode_id=episode_id,
            )
            raise
        downloaded += 1
        remaining = len(all_missing) - downloaded
        save_progress(
            args.progress,
            status="complete" if remaining == 0 else "downloading",
            indexed=len(episode_ids),
            existing=len(episode_ids) - remaining,
            remaining=remaining,
            selected_this_run=len(selected),
            downloaded_this_run=downloaded,
            failed_episode_id=None,
        )
        if args.delay > 0 and number < len(selected):
            time.sleep(args.delay)

    print(f"本次下载完成：{downloaded} 条。可重复运行，已有文件会自动跳过。")


if __name__ == "__main__":
    main()
