"""Enumerate and resume Kaggle simulation replay downloads via the official CLI."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import subprocess
import time
from pathlib import Path


def run_cli(kaggle: str, *args: str, retries: int = 5) -> str:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, retries + 1):
        try:
            result = subprocess.run(
                [kaggle, *args],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return result.stdout
        except subprocess.CalledProcessError as exc:
            last_error = exc
            if attempt == retries:
                break
            delay = min(2 ** (attempt - 1), 30)
            print(f"CLI attempt {attempt}/{retries} failed; retrying in {delay}s")
            time.sleep(delay)
    assert last_error is not None
    raise RuntimeError(last_error.stderr[-2000:]) from last_error


def episode_ids(kaggle: str, submission_id: int) -> list[int]:
    output = run_cli(kaggle, "competitions", "episodes", str(submission_id), "-v")
    lines = output.splitlines()
    try:
        header = next(i for i, line in enumerate(lines) if line.startswith("id,"))
    except StopIteration as exc:
        raise RuntimeError(f"No episode CSV returned for submission {submission_id}") from exc

    ids: list[int] = []
    for row in csv.DictReader(io.StringIO("\n".join(lines[header:]))):
        value = (row.get("id") or "").strip()
        if value.isdigit() and "COMPLETED" in (row.get("state") or ""):
            ids.append(int(value))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="List or download complete replay JSON for public Kaggle submissions."
    )
    parser.add_argument("--submission-id", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/external/kaggle_replays/raw"))
    parser.add_argument("--limit", type=int, default=0, help="Newest N unique episodes; 0 means all.")
    parser.add_argument("--download", action="store_true", help="Download; otherwise dry-run.")
    parser.add_argument("--kaggle", default=shutil.which("kaggle") or "kaggle")
    args = parser.parse_args()

    ordered: list[int] = []
    seen: set[int] = set()
    for submission_id in args.submission_id:
        for episode_id in episode_ids(args.kaggle, submission_id):
            if episode_id not in seen:
                seen.add(episode_id)
                ordered.append(episode_id)
    if args.limit > 0:
        ordered = ordered[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    existing = {
        int(path.name.split("-")[1])
        for path in args.output.glob("episode-*-replay.json")
        if path.name.split("-")[1].isdigit()
    }
    missing = [episode_id for episode_id in ordered if episode_id not in existing]
    print(f"unique={len(ordered)} existing={len(ordered) - len(missing)} missing={len(missing)}")

    if not args.download:
        print("dry-run: add --download to fetch the missing episode JSON files")
        return

    for number, episode_id in enumerate(missing, 1):
        print(f"[{number}/{len(missing)}] episode {episode_id}")
        run_cli(
            args.kaggle,
            "competitions",
            "replay",
            str(episode_id),
            "-p",
            str(args.output),
        )


if __name__ == "__main__":
    main()
