"""Rank complete Kaggle replays by similarity to selected high-score decks."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SEED_EPISODES = {
    85216002,
    85222378,
    85221419,
    85221901,
    85221910,
    85220459,
    85220950,
    85218535,
    85220501,
    85219491,
    85213633,
    85302664,
}


def load_replay(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def initial_decks(replay: dict) -> list[Counter[int]]:
    """Recover both full 60-card lists from the first omniscient visualize frame."""
    for step in replay.get("steps", []):
        for record in step:
            for frame in record.get("visualize", []) if isinstance(record, dict) else []:
                players = frame.get("current", {}).get("players", [])
                if len(players) != 2:
                    continue
                decks = []
                for player in players:
                    cards = player.get("deck", [])
                    ids = [int(card["id"]) for card in cards if isinstance(card, dict) and "id" in card]
                    decks.append(Counter(ids))
                if all(sum(deck.values()) == 60 for deck in decks):
                    return decks
    raise ValueError("No two complete 60-card decks in visualize frames")


def weighted_jaccard(left: Counter[int], right: Counter[int]) -> float:
    keys = left.keys() | right.keys()
    denominator = sum(max(left[key], right[key]) for key in keys)
    return sum(min(left[key], right[key]) for key in keys) / denominator if denominator else 0.0


def deck_hash(deck: Counter[int]) -> str:
    return ";".join(f"{card_id}:{count}" for card_id, count in sorted(deck.items()))


def seed_profiles(raw: Path, current_deck: Path | None) -> list[dict]:
    profiles: dict[str, dict] = {}
    for episode_id in sorted(SEED_EPISODES):
        path = raw / f"episode-{episode_id}-replay.json"
        if not path.exists():
            continue
        replay = load_replay(path)
        teams = replay.get("info", {}).get("TeamNames") or [None, None]
        for player, deck in enumerate(initial_decks(replay)):
            key = deck_hash(deck)
            profile = profiles.setdefault(
                key,
                {
                    "profile_id": f"high_score_{len(profiles) + 1:02d}",
                    "deck": deck,
                    "seed_episodes": [],
                    "seed_teams": [],
                },
            )
            profile["seed_episodes"].append(episode_id)
            if player < len(teams) and teams[player] not in profile["seed_teams"]:
                profile["seed_teams"].append(teams[player])

    if current_deck and current_deck.exists():
        ids = [int(line.strip()) for line in current_deck.read_text().splitlines() if line.strip()]
        if len(ids) == 60:
            deck = Counter(ids)
            profiles.setdefault(
                deck_hash(deck),
                {
                    "profile_id": "current_submission",
                    "deck": deck,
                    "seed_episodes": [],
                    "seed_teams": ["local submission/deck.csv"],
                },
            )
    return list(profiles.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/external/kaggle_replays/raw"))
    parser.add_argument("--current-deck", type=Path, default=Path("submission/deck.csv"))
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/kaggle_replays/core_combo_candidates.json"),
    )
    args = parser.parse_args()
    profiles = seed_profiles(args.raw, args.current_deck)
    matches = []
    invalid = []

    for path in sorted(args.raw.glob("episode-*-replay.json")):
        try:
            replay = load_replay(path)
            decks = initial_decks(replay)
        except (json.JSONDecodeError, OSError, ValueError) as exc:
            invalid.append({"path": path.as_posix(), "error": str(exc)})
            continue
        terminal = replay.get("steps", [])[-1]
        rewards = [record.get("reward") for record in terminal]
        teams = replay.get("info", {}).get("TeamNames") or [None, None]
        for player, deck in enumerate(decks):
            ranked = sorted(
                ((weighted_jaccard(deck, p["deck"]), p) for p in profiles),
                key=lambda item: item[0],
                reverse=True,
            )
            similarity, profile = ranked[0]
            if similarity < args.threshold:
                continue
            matches.append(
                {
                    "episode_id": int(
                        replay.get("info", {}).get("EpisodeId")
                        or path.name.split("-")[1]
                    ),
                    "path": path.as_posix(),
                    "player_index": player,
                    "team_name": teams[player] if player < len(teams) else None,
                    "target_profile": profile["profile_id"],
                    "deck_similarity": round(similarity, 6),
                    "won": player < len(rewards) and rewards[player] == 1,
                    "terminal_rewards": rewards,
                    "steps": len(replay.get("steps", [])),
                    "deck": dict(sorted(deck.items())),
                }
            )

    matches.sort(key=lambda row: (row["won"], row["deck_similarity"], row["steps"]), reverse=True)
    output = {
        "schema_version": "1.0.0",
        "similarity": "multiset weighted Jaccard over complete 60-card ID lists",
        "threshold": args.threshold,
        "profiles": [
            {
                **{key: value for key, value in profile.items() if key != "deck"},
                "deck": dict(sorted(profile["deck"].items())),
            }
            for profile in profiles
        ],
        "candidate_player_trajectories": len(matches),
        "candidate_replays": len({row["episode_id"] for row in matches}),
        "invalid_or_incomplete_files": invalid,
        "matches": matches,
    }
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"profiles={len(profiles)} replay_candidates={output['candidate_replays']} "
        f"player_trajectories={len(matches)} invalid={len(invalid)}"
    )


if __name__ == "__main__":
    main()
