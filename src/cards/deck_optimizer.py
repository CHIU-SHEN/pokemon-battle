#!/usr/bin/env python3
"""Archetype deck generation, proxy scoring, and MAP-Elites selection."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import random
from typing import Any

from deck_rules import check_deck, is_basic_energy, load_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CARDS_JSON = PROJECT_ROOT / "data" / "cards.json"
TAGS_JSON = PROJECT_ROOT / "data" / "card_tags.json"
CANDIDATES_JSON = PROJECT_ROOT / "data" / "deck_candidates.json"
COEVOLUTION_JSON = PROJECT_ROOT / "data" / "deck_coevolution_plan.json"
ELITE_DECK_DIR = PROJECT_ROOT / "data" / "deck_elites"

WATER = 3
KYOGRE = 721
SNOVER = 722
ABOMASNOW = 723
MEGA_SIGNAL = 1145
MAX_BELT = 1158
CYRANO = 1205
LILLIE = 1227
WAITRESS = 1235


@dataclass(frozen=True)
class Archetype:
    name: str
    energy_range: tuple[int, int]
    kyogre_range: tuple[int, int]
    mega_signal_range: tuple[int, int]
    cyrano_range: tuple[int, int]
    lillie_range: tuple[int, int]
    waitress_range: tuple[int, int]
    tech_slots_range: tuple[int, int]
    description: str


ARCHETYPES = [
    Archetype("abomasnow_high_energy", (34, 39), (1, 2), (3, 4), (1, 2), (3, 4), (3, 4), (0, 2), "高水能密度，最大化 Hammer-lanche 和贴能稳定性"),
    Archetype("stable_setup", (28, 34), (1, 2), (4, 4), (2, 2), (4, 4), (3, 4), (2, 5), "更厚抽牌/检索，降低卡手"),
    Archetype("aggressive_prize", (26, 32), (2, 3), (3, 4), (1, 2), (3, 4), (2, 4), (4, 7), "更多进攻/伤害补强和次攻击手"),
    Archetype("defensive_recovery", (29, 35), (1, 2), (3, 4), (1, 2), (3, 4), (3, 4), (3, 6), "加入回复/换人/防守 tech"),
    Archetype("anti_ex_tech", (28, 34), (1, 2), (3, 4), (1, 2), (3, 4), (2, 4), (4, 8), "针对 ex/Mega 的工具和干扰卡"),
]


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def write_deck_csv(path: Path, deck: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for card_id in deck:
            f.write(f"{card_id}\n")


class DeckOptimizer:
    def __init__(self, cards_db: dict[str, Any], tags_db: dict[str, Any], seed: int = 20260706) -> None:
        self.cards_db = cards_db
        self.tags_db = tags_db
        self.rng = random.Random(seed)
        self.pools = self._build_pools()

    def tags(self, card_id: int) -> set[str]:
        return set(self.tags_db["cards"].get(str(card_id), {}).get("tags", []))

    def _ids_with(self, *required: str) -> list[int]:
        ids = []
        for cid, item in self.tags_db["cards"].items():
            tagset = set(item.get("tags", []))
            if all(tag in tagset for tag in required):
                ids.append(int(cid))
        return ids

    def _trainer_only(self, ids: list[int]) -> list[int]:
        allowed_types = {1, 2, 3, 4}  # Item, Tool, Supporter, Stadium
        return [
            cid
            for cid in ids
            if self.cards_db["cards"].get(str(cid), {}).get("engine", {}).get("card_type") in allowed_types
        ]

    def _build_pools(self) -> dict[str, list[int]]:
        banned = {MAX_BELT, WATER, KYOGRE, SNOVER, ABOMASNOW, MEGA_SIGNAL, CYRANO, LILLIE, WAITRESS}
        pools = {
            "draw": self._trainer_only([cid for cid in self._ids_with("draw") if cid not in banned]),
            "search_pokemon": self._trainer_only([cid for cid in self._ids_with("search_pokemon") if cid not in banned]),
            "search_energy": self._trainer_only([cid for cid in self._ids_with("search_energy") if cid not in banned]),
            "attach_energy": self._trainer_only([cid for cid in self._ids_with("attach_energy") if cid not in banned]),
            "damage_boost": self._trainer_only([cid for cid in self._ids_with("damage_boost") if cid not in banned]),
            "heal": self._trainer_only([cid for cid in self._ids_with("heal") if cid not in banned]),
            "switch": self._trainer_only([cid for cid in self._ids_with("switch") if cid not in banned]),
            "gust": self._trainer_only([cid for cid in self._ids_with("gust") if cid not in banned]),
            "ace_spec": self._trainer_only([cid for cid in self._ids_with("ace_spec") if cid not in {MAX_BELT}]),
        }
        for key, values in pools.items():
            values.sort()
        return pools

    def _add_with_limit(self, deck: list[int], card_id: int, count: int) -> None:
        if card_id == WATER:
            deck.extend([card_id] * count)
            return
        current = deck.count(card_id)
        deck.extend([card_id] * max(0, min(count, 4 - current)))

    def _sample_tech(self, archetype: Archetype) -> list[int]:
        tech_slots = self.rng.randint(*archetype.tech_slots_range)
        if tech_slots <= 0:
            return []
        if archetype.name == "defensive_recovery":
            weighted_groups = ["heal", "switch", "search_energy", "draw"]
        elif archetype.name == "anti_ex_tech":
            weighted_groups = ["damage_boost", "gust", "switch", "draw"]
        elif archetype.name == "aggressive_prize":
            weighted_groups = ["damage_boost", "gust", "draw", "search_pokemon"]
        elif archetype.name == "stable_setup":
            weighted_groups = ["draw", "search_pokemon", "search_energy", "attach_energy"]
        else:
            weighted_groups = ["draw", "search_energy", "attach_energy"]
        tech: list[int] = []
        attempts = 0
        while len(tech) < tech_slots and attempts < tech_slots * 20:
            attempts += 1
            group = self.rng.choice(weighted_groups)
            pool = self.pools.get(group, [])
            if not pool:
                continue
            cid = self.rng.choice(pool[:80])
            if cid in tech and tech.count(cid) >= 2:
                continue
            if "ace_spec" in self.tags(cid):
                continue
            tech.append(cid)
        return tech[:tech_slots]

    def generate_variant(self, archetype: Archetype) -> dict[str, Any]:
        deck: list[int] = []
        self._add_with_limit(deck, SNOVER, 4)
        self._add_with_limit(deck, ABOMASNOW, 4)
        self._add_with_limit(deck, KYOGRE, self.rng.randint(*archetype.kyogre_range))
        self._add_with_limit(deck, MEGA_SIGNAL, self.rng.randint(*archetype.mega_signal_range))
        self._add_with_limit(deck, CYRANO, self.rng.randint(*archetype.cyrano_range))
        self._add_with_limit(deck, LILLIE, self.rng.randint(*archetype.lillie_range))
        self._add_with_limit(deck, WAITRESS, self.rng.randint(*archetype.waitress_range))
        self._add_with_limit(deck, MAX_BELT, 1)
        for cid in self._sample_tech(archetype):
            self._add_with_limit(deck, cid, 1)

        energy_target = self.rng.randint(*archetype.energy_range)
        self._add_with_limit(deck, WATER, max(0, min(energy_target, 60 - len(deck))))
        fill_groups = ["draw", "search_pokemon", "search_energy", "attach_energy"]
        attempts = 0
        while len(deck) < 60 and attempts < 300:
            attempts += 1
            cid = self.rng.choice(self.pools[self.rng.choice(fill_groups)][:100])
            if "ace_spec" in self.tags(cid):
                continue
            before = len(deck)
            self._add_with_limit(deck, cid, 1)
            if len(deck) == before and WATER not in deck:
                self._add_with_limit(deck, WATER, 1)
        while len(deck) < 60:
            self._add_with_limit(deck, WATER, 1)
        if len(deck) > 60:
            deck = deck[:60]
        self.rng.shuffle(deck)
        score, explanation, features = self.score_deck(deck, archetype.name)
        return {
            "archetype": archetype.name,
            "description": archetype.description,
            "deck": deck,
            "score": score,
            "features": features,
            "explanation": explanation,
            "legal": asdict(check_deck(deck, self.cards_db, self.tags_db)),
        }

    def score_deck(self, deck: list[int], archetype_name: str = "") -> tuple[float, dict[str, float], dict[str, Any]]:
        counts = Counter(deck)
        tag_counts = defaultdict(int)
        for cid, n in counts.items():
            for tag in self.tags(cid):
                tag_counts[tag] += n
        energy = counts[WATER]
        snovers = counts[SNOVER]
        abomas = counts[ABOMASNOW]
        search = counts[MEGA_SIGNAL] + counts[CYRANO] + tag_counts["search_pokemon"] * 0.4
        draw = counts[LILLIE] + tag_counts["draw"] * 0.45
        attach = counts[WAITRESS] + tag_counts["attach_energy"] * 0.35 + tag_counts["search_energy"] * 0.25
        tech = sum(counts[cid] for cid in counts if cid not in {WATER, KYOGRE, SNOVER, ABOMASNOW, MEGA_SIGNAL, CYRANO, LILLIE, WAITRESS, MAX_BELT})
        deckout_risk = max(0, 30 - energy) * 1.5 + max(0, counts[ABOMASNOW] - 2) * 3
        setup = 75 * min(snovers, 4) + 65 * min(abomas, 4) + 34 * min(search, 8)
        energy_consistency = 18 * min(energy, 36) - 10 * max(0, energy - 37) - 28 * max(0, 26 - energy)
        trainer_quality = 24 * min(draw, 8) + 20 * min(attach, 7)
        finish = 55 * counts[MAX_BELT] + 18 * counts[KYOGRE] + 8 * tag_counts["damage_boost"]
        flexibility = 7 * tag_counts["switch"] + 7 * tag_counts["gust"] + 5 * tag_counts["heal"]
        card_hand_risk = 16 * max(0, tech - 8) + 35 * max(0, 4 - search)
        score = setup + energy_consistency + trainer_quality + finish + flexibility - deckout_risk - card_hand_risk
        explanation = {
            "setup_stability": setup,
            "energy_consistency": energy_consistency,
            "draw_search_density": trainer_quality,
            "finish_power": finish,
            "tech_flexibility": flexibility,
            "card_hand_risk_penalty": -card_hand_risk,
            "deckout_risk_penalty": -deckout_risk,
        }
        features = {
            "archetype": archetype_name,
            "energy_count": energy,
            "draw_count": int(tag_counts["draw"] + counts[LILLIE]),
            "search_count": int(tag_counts["search_pokemon"] + tag_counts["search_energy"] + counts[MEGA_SIGNAL] + counts[CYRANO]),
            "tech_count": tech,
            "ace_spec_count": counts[MAX_BELT] + sum(counts[cid] for cid in counts if "ace_spec" in self.tags(cid) and cid != MAX_BELT),
            "deckout_risk": round(deckout_risk, 2),
        }
        return round(score, 3), explanation, features

    def map_elites(self, variants: list[dict[str, Any]], min_elites: int = 20, max_elites: int = 50) -> list[dict[str, Any]]:
        cells: dict[tuple[Any, ...], dict[str, Any]] = {}
        for variant in variants:
            f = variant["features"]
            cell = (
                f["archetype"],
                min(4, f["energy_count"] // 4),
                min(4, f["tech_count"] // 2),
                min(4, int(f["deckout_risk"] // 4)),
                min(4, f["search_count"] // 4),
            )
            if cell not in cells or variant["score"] > cells[cell]["score"]:
                cells[cell] = variant
        elites = sorted(cells.values(), key=lambda v: v["score"], reverse=True)
        if len(elites) < min_elites:
            seen = {tuple(v["deck"]) for v in elites}
            for variant in sorted(variants, key=lambda v: v["score"], reverse=True):
                key = tuple(variant["deck"])
                if key not in seen:
                    elites.append(variant)
                    seen.add(key)
                if len(elites) >= min_elites:
                    break
        return elites[:max_elites]

    def random_legal_deck(self) -> list[int]:
        valid_ids = [int(cid) for cid in self.cards_db["cards"]]
        basic_pokemon = [cid for cid in valid_ids if self.cards_db["cards"][str(cid)]["engine"].get("basic")]
        deck = [self.rng.choice(basic_pokemon)]
        attempts = 0
        while len(deck) < 60 and attempts < 1000:
            attempts += 1
            cid = self.rng.choice(valid_ids)
            if is_basic_energy(self.cards_db, self.tags_db, cid) or deck.count(cid) < 4:
                deck.append(cid)
            if check_deck(deck + [WATER] * (60 - len(deck)), self.cards_db, self.tags_db).valid:
                pass
        while len(deck) < 60:
            deck.append(WATER)
        if check_deck(deck, self.cards_db, self.tags_db).valid:
            return deck
        return [WATER] * 35 + [SNOVER] * 4 + [ABOMASNOW] * 4 + [MEGA_SIGNAL] * 4 + [LILLIE] * 4 + [WAITRESS] * 4 + [CYRANO] * 2 + [KYOGRE] * 2 + [MAX_BELT]


def build_candidate_set(per_archetype: int = 150, seed: int = 20260706) -> dict[str, Any]:
    cards_db = load_json(CARDS_JSON)
    tags_db = load_json(TAGS_JSON)
    optimizer = DeckOptimizer(cards_db, tags_db, seed=seed)
    variants: list[dict[str, Any]] = []
    counts_by_archetype: dict[str, int] = {}
    for archetype in ARCHETYPES:
        generated = []
        attempts = 0
        while len(generated) < per_archetype and attempts < per_archetype * 8:
            attempts += 1
            variant = optimizer.generate_variant(archetype)
            if variant["legal"]["valid"]:
                generated.append(variant)
        counts_by_archetype[archetype.name] = len(generated)
        variants.extend(generated)
    elites = optimizer.map_elites(variants)
    random_scores = [optimizer.score_deck(optimizer.random_legal_deck(), "random")[0] for _ in range(100)]
    top_scores = [v["score"] for v in sorted(variants, key=lambda v: v["score"], reverse=True)[: max(1, len(variants) // 10)]]
    return {
        "metadata": {
            "per_archetype_requested": per_archetype,
            "generated_legal_variants": len(variants),
            "counts_by_archetype": counts_by_archetype,
            "elite_count": len(elites),
            "random_proxy_mean": round(sum(random_scores) / len(random_scores), 3),
            "top10pct_proxy_mean": round(sum(top_scores) / len(top_scores), 3),
            "score_note": "Proxy score only; candidates must pass M2 battle matrix before replacing deck.csv.",
        },
        "archetypes": [asdict(a) for a in ARCHETYPES],
        "elites": elites,
    }


def build_coevolution_plan(candidate_set: dict[str, Any]) -> dict[str, Any]:
    top = candidate_set["elites"][:10]
    return {
        "version": "m5_initial",
        "status": "proxy_screened_not_battle_promoted",
        "next_steps": [
            "Copy one elite deck into a temporary submission/deck.csv clone.",
            "Run M2 league against Random, Sample, Exploiter-FirstMin, V0-best.",
            "Record deck-only, strategy-only, and deck+strategy results separately.",
            "Promote only if external baselines improve and exploiter does not regress.",
        ],
        "strategy_weight_suggestions": [
            {
                "condition": "energy_count < 30",
                "suggestion": "Increase avoid_deckout importance and reduce Hammer-lanche prior.",
            },
            {
                "condition": "tech_count > 6",
                "suggestion": "Increase search/draw priority to offset card-hand risk.",
            },
            {
                "condition": "aggressive_prize archetype promoted",
                "suggestion": "Raise damage_boost and take_prize weights; add tactical tests for Maximum Belt lines.",
            },
        ],
        "top_elite_summaries": [
            {
                "rank": i + 1,
                "archetype": item["archetype"],
                "score": item["score"],
                "features": item["features"],
            }
            for i, item in enumerate(top)
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-archetype", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260706)
    parser.add_argument("--out", default=str(CANDIDATES_JSON))
    parser.add_argument("--coevolution-out", default=str(COEVOLUTION_JSON))
    parser.add_argument("--elite-deck-dir", default=str(ELITE_DECK_DIR))
    args = parser.parse_args()
    candidate_set = build_candidate_set(args.per_archetype, args.seed)
    write_json(Path(args.out), candidate_set)
    write_json(Path(args.coevolution_out), build_coevolution_plan(candidate_set))
    elite_dir = Path(args.elite_deck_dir)
    for i, elite in enumerate(candidate_set["elites"], start=1):
        write_deck_csv(elite_dir / f"elite_{i:03d}_{elite['archetype']}.csv", elite["deck"])
    print(json.dumps(candidate_set["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
