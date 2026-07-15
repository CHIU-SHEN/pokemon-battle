#!/usr/bin/env python3
"""Build and query a CardDB from official CSV data and engine metadata."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "external" / "kaggle_pokemon_tcg_ai_battle"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from cg.api import all_attack, all_card_data  # noqa: E402


RULESET_ID = "ptcg_abc_2026_simulation_designated_pool_v1"
COMPETITION_URL = "https://www.kaggle.com/competitions/pokemon-tcg-ai-battle"
RULES_OVERVIEW_URL = "https://ptcg-abc.pokemon.co.jp/"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


EN_COLUMNS = {
    "id": "Card ID",
    "name": "Card Name",
    "expansion": "Expansion",
    "collection_no": "Collection No.",
    "stage_type": "Stage (Pokémon)/Type (Energy and Trainer)",
    "rule": "Rule",
    "category": "Category",
    "previous_stage": "Previous stage",
    "hp": "HP",
    "type": "Type",
    "weakness": "Weakness",
    "resistance": "Resistance (Type)",
    "retreat": "Retreat",
    "move_name": "Move Name",
    "cost": "Cost",
    "damage": "Damage",
    "effect": "Effect Explanation",
}

JP_COLUMNS = {
    "id": "カード ID",
    "name": "カード名",
    "expansion": "エキスパンションマーク",
    "collection_no": "コレクション番号",
    "stage_type": "ポケモンの進化の段階/エネルギー・トレーナーズの種類",
    "rule": "ルール",
    "category": "カテゴリ",
    "previous_stage": "進化前",
    "hp": "HP",
    "type": "タイプ",
    "weakness": "弱点",
    "resistance": "抵抗力",
    "retreat": "にげる",
    "move_name": "ワザ名",
    "cost": "コスト",
    "damage": "ダメージ",
    "effect": "効果の説明",
}


@dataclass
class AttackRow:
    name_en: str
    name_jp: str | None
    cost: str
    damage: str
    text_en: str
    text_jp: str | None = None
    engine_attack_id: int | None = None
    engine_damage: int | None = None
    engine_energies: list[int] = field(default_factory=list)


@dataclass
class CardRecord:
    card_id: int
    name_en: str
    name_jp: str | None
    expansion: str
    collection_no: str
    stage_type: str
    rule_box: str
    category: str
    previous_stage: str | None
    hp: int
    type_text: str
    weakness: str | None
    resistance: str | None
    retreat_cost: int
    attacks: list[AttackRow] = field(default_factory=list)
    engine: dict[str, Any] = field(default_factory=dict)


def clean(value: str | None) -> str:
    value = (value or "").strip()
    return "" if value in {"n/a", "N/A", "-"} else value


def parse_int(value: str | None, default: int = 0) -> int:
    value = clean(value)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        digits = "".join(ch for ch in value if ch.isdigit())
        return int(digits) if digits else default


def read_csv(path: Path, columns: dict[str, str]) -> dict[int, list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        card_id = int(row[columns["id"]])
        grouped.setdefault(card_id, []).append(row)
    return grouped


def build_card_db(data_dir: Path = DATA_DIR) -> dict[str, Any]:
    en = read_csv(data_dir / "EN_Card_Data.csv", EN_COLUMNS)
    jp = read_csv(data_dir / "JP_Card_Data.csv", JP_COLUMNS)
    engine_cards = {card.cardId: card for card in all_card_data()}
    engine_attacks = {attack.attackId: attack for attack in all_attack()}
    records: dict[str, Any] = {}

    for card_id, en_rows in sorted(en.items()):
        first = en_rows[0]
        jp_rows = jp.get(card_id, [])
        jp_first = jp_rows[0] if jp_rows else {}
        engine = engine_cards.get(card_id)
        attack_rows: list[AttackRow] = []
        engine_attack_ids = list(getattr(engine, "attacks", []) or [])
        for i, row in enumerate(en_rows):
            move_name = clean(row[EN_COLUMNS["move_name"]])
            effect = clean(row[EN_COLUMNS["effect"]])
            if not move_name and not effect:
                continue
            jp_row = jp_rows[i] if i < len(jp_rows) else {}
            attack_id = engine_attack_ids[i] if i < len(engine_attack_ids) else None
            engine_attack = engine_attacks.get(attack_id) if attack_id is not None else None
            attack_rows.append(
                AttackRow(
                    name_en=move_name,
                    name_jp=clean(jp_row.get(JP_COLUMNS["move_name"])) or None,
                    cost=clean(row[EN_COLUMNS["cost"]]),
                    damage=clean(row[EN_COLUMNS["damage"]]),
                    text_en=effect,
                    text_jp=clean(jp_row.get(JP_COLUMNS["effect"])) or None,
                    engine_attack_id=attack_id,
                    engine_damage=getattr(engine_attack, "damage", None) if engine_attack else None,
                    engine_energies=list(getattr(engine_attack, "energies", []) or []) if engine_attack else [],
                )
            )
        record = CardRecord(
            card_id=card_id,
            name_en=clean(first[EN_COLUMNS["name"]]),
            name_jp=clean(jp_first.get(JP_COLUMNS["name"])) or None,
            expansion=clean(first[EN_COLUMNS["expansion"]]),
            collection_no=clean(first[EN_COLUMNS["collection_no"]]),
            stage_type=clean(first[EN_COLUMNS["stage_type"]]),
            rule_box=clean(first[EN_COLUMNS["rule"]]),
            category=clean(first[EN_COLUMNS["category"]]),
            previous_stage=clean(first[EN_COLUMNS["previous_stage"]]) or None,
            hp=parse_int(first[EN_COLUMNS["hp"]]),
            type_text=clean(first[EN_COLUMNS["type"]]),
            weakness=clean(first[EN_COLUMNS["weakness"]]) or None,
            resistance=clean(first[EN_COLUMNS["resistance"]]) or None,
            retreat_cost=parse_int(first[EN_COLUMNS["retreat"]]),
            attacks=attack_rows,
            engine={
                "card_type": int(getattr(engine, "cardType", -1)) if engine else None,
                "energy_type": int(getattr(engine, "energyType", -1)) if engine else None,
                "basic": bool(getattr(engine, "basic", False)) if engine else False,
                "stage1": bool(getattr(engine, "stage1", False)) if engine else False,
                "stage2": bool(getattr(engine, "stage2", False)) if engine else False,
                "ex": bool(getattr(engine, "ex", False)) if engine else False,
                "mega_ex": bool(getattr(engine, "megaEx", False)) if engine else False,
                "tera": bool(getattr(engine, "tera", False)) if engine else False,
                "ace_spec": bool(getattr(engine, "aceSpec", False)) if engine else False,
                "evolves_from": getattr(engine, "evolvesFrom", None) if engine else None,
                "skills": [asdict(skill) for skill in (getattr(engine, "skills", []) or [])] if engine else [],
                "attack_ids": engine_attack_ids,
            },
        )
        records[str(card_id)] = asdict(record)

    metadata = {
        "ruleset": {
            "id": RULESET_ID,
            "name": "Pokémon TCG AI Battle Challenge 2026 Simulation designated-card-pool rules",
            "basis": "Standard format with tournament-specific adjustments",
            "designated_card_pool_only": True,
            "designated_card_count": len(records),
            "deck_size": 60,
            "player_time_limit_seconds": 600,
            "competition_url": COMPETITION_URL,
            "competition_rules_url": COMPETITION_URL + "/rules",
            "rules_overview_url": RULES_OVERVIEW_URL,
            "retrieved_at": "2026-07-15",
            "usage_terms": "Governed by the Kaggle Competition Rules; no separate open-data redistribution license is asserted.",
            "note": "The organizer does not identify this environment as an unmodified real-world regulation mark; the supplied engine and designated card list are authoritative for simulation behavior.",
        },
        "source_files": {
            "en_csv": {
                "path": (data_dir / "EN_Card_Data.csv").relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(data_dir / "EN_Card_Data.csv"),
                "source_url": COMPETITION_URL + "/data",
                "retrieved_at": "2026-07-13",
            },
            "jp_csv": {
                "path": (data_dir / "JP_Card_Data.csv").relative_to(PROJECT_ROOT).as_posix(),
                "sha256": sha256_file(data_dir / "JP_Card_Data.csv"),
                "source_url": COMPETITION_URL + "/data",
                "retrieved_at": "2026-07-13",
            },
            "engine_api": "submission/cg/api.py::all_card_data/all_attack",
        },
        "row_counts": {"en_csv": sum(len(v) for v in en.values()), "jp_csv": sum(len(v) for v in jp.values())},
        "unique_card_ids": len(records),
        "engine_card_count": len(engine_cards),
        "engine_attack_count": len(engine_attacks),
    }
    return {"metadata": metadata, "cards": records}


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(PROJECT_ROOT / "data" / "cards.json"))
    args = parser.parse_args()
    db = build_card_db()
    write_json(Path(args.out), db)
    print(json.dumps(db["metadata"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

