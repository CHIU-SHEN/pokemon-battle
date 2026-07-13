"""Fast, resumable bulk replay downloads using one authenticated Kaggle API process."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi


def retry(label: str, operation, attempts: int = 6):
    error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except Exception as exc:  # Kaggle SDK wraps several network exception types.
            error = exc
            if attempt == attempts:
                break
            if "429" in repr(exc) or "Too Many Requests" in repr(exc):
                delay = min(60 * attempt, 300)
            else:
                delay = min(2 ** (attempt - 1), 30)
            print(f"{label}: attempt {attempt}/{attempts} failed; retry in {delay}s", flush=True)
            time.sleep(delay)
    assert error is not None
    raise error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-id", type=int, action="append", required=True)
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Minimum seconds between successful replay requests.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/external/kaggle_replays/raw"))
    parser.add_argument(
        "--progress",
        type=Path,
        default=Path("data/external/kaggle_replays/download_progress.json"),
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    args.progress.parent.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    retry("authenticate", api.authenticate)

    ordered: list[int] = []
    seen: set[int] = set()
    for submission_id in args.submission_id:
        episodes = retry(
            f"submission {submission_id}",
            lambda sid=submission_id: api.competition_list_episodes(sid),
        )
        for episode in episodes:
            episode_id = int(episode.id)
            if episode_id not in seen and "COMPLETED" in str(episode.state):
                seen.add(episode_id)
                ordered.append(episode_id)
                if len(ordered) >= args.limit:
                    break
        if len(ordered) >= args.limit:
            break

    existing = {
        int(path.name.split("-")[1])
        for path in args.output.glob("episode-*-replay.json")
        if path.name.split("-")[1].isdigit()
    }
    missing = [episode_id for episode_id in ordered if episode_id not in existing]
    failures: list[dict] = []
    downloaded = 0

    def save_progress(current: int | None = None) -> None:
        document = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "target_unique_episodes": len(ordered),
            "already_present_at_start": len(ordered) - len(missing),
            "downloaded_this_run": downloaded,
            "failed": failures,
            "current_episode": current,
            "complete": downloaded + len(ordered) - len(missing) == len(ordered),
        }
        args.progress.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    print(f"target={len(ordered)} existing={len(ordered) - len(missing)} missing={len(missing)}", flush=True)
    save_progress()
    for number, episode_id in enumerate(missing, 1):
        try:
            retry(
                f"episode {episode_id}",
                lambda eid=episode_id: api.competition_episode_replay(
                    eid, str(args.output), quiet=True
                ),
            )
            downloaded += 1
            if args.delay > 0:
                time.sleep(args.delay)
        except Exception as exc:
            failures.append({"episode_id": episode_id, "error": repr(exc)})
        if number % 10 == 0 or failures or number == len(missing):
            print(
                f"processed={number}/{len(missing)} downloaded={downloaded} failed={len(failures)}",
                flush=True,
            )
            save_progress(episode_id)
    save_progress()


if __name__ == "__main__":
    main()
