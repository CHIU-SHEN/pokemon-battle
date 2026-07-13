"""Validate downloaded Kaggle simulation replay JSON files and build an index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect(path: Path) -> dict:
    replay = json.loads(path.read_text(encoding="utf-8"))
    steps = replay.get("steps", [])
    agents_per_step = [len(step) for step in steps if isinstance(step, list)]
    records = [agent for step in steps if isinstance(step, list) for agent in step]
    last = steps[-1] if steps else []
    observations = [r.get("observation") for r in records if isinstance(r, dict)]
    observations = [o for o in observations if isinstance(o, dict)]

    required_agent_fields = {"action", "observation", "reward", "status"}
    valid = bool(steps) and all(
        isinstance(record, dict) and required_agent_fields <= record.keys()
        for record in records
    )
    return {
        "episode_id": replay.get("id") or replay.get("info", {}).get("EpisodeId"),
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "schema_version": replay.get("schema_version"),
        "module_version": replay.get("module_version"),
        "team_names": replay.get("info", {}).get("TeamNames"),
        "steps": len(steps),
        "agents_per_step": sorted(set(agents_per_step)),
        "terminal_statuses": [r.get("status") for r in last if isinstance(r, dict)],
        "terminal_rewards": [r.get("reward") for r in last if isinstance(r, dict)],
        "records_with_action": sum(r.get("action") is not None for r in records),
        "records_with_observation": len(observations),
        "records_with_logs": sum("logs" in o for o in observations),
        "records_with_select": sum("select" in o for o in observations),
        "records_with_visualize": sum("visualize" in r for r in records),
        "valid_complete_trajectory": valid
        and bool(last)
        and all(r.get("status") == "DONE" for r in last),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/external/kaggle_replays/raw"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/kaggle_replays/replay_index.json"),
    )
    args = parser.parse_args()

    files = sorted(args.input.glob("episode-*-replay.json"))
    entries = [inspect(path) for path in files]
    document = {
        "schema_version": "1.0.0",
        "source": "Kaggle official simulation competitions replay endpoint",
        "complete_replays": sum(e["valid_complete_trajectory"] for e in entries),
        "total_steps": sum(e["steps"] for e in entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "replays": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"indexed={len(entries)} complete={document['complete_replays']} "
        f"steps={document['total_steps']} bytes={document['total_bytes']}"
    )


if __name__ == "__main__":
    main()
