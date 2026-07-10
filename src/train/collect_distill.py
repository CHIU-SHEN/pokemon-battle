#!/usr/bin/env python3
"""Collect M6 distillation samples from visible observations and a V1 teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from agent.action_gen import ActionGenerator  # noqa: E402
from agent.belief import read_deck  # noqa: E402
from agent.fallback import is_legal_action  # noqa: E402
from agent.parser import GameLedger, parse_observation  # noqa: E402
from agent.rules import choose_action  # noqa: E402
from agent.search import SearchConfig, SearchManager  # noqa: E402
from src.train.distill_schema import SCHEMA_VERSION, schema_document, validate_dataset, write_jsonl  # noqa: E402
from src.train.features import FEATURE_NAMES, TAG_FEATURES, load_card_tags, sample_features  # noqa: E402


def _action_key(action: list[int]) -> tuple[int, ...]:
    return tuple(int(x) for x in action)


def _score_arrays(option_count: int, report: dict[str, Any]) -> tuple[list[float | None], list[int]]:
    q_values: list[float | None] = [None] * option_count
    visits = [0] * option_count
    for item in report.get("scores", []) or []:
        action = item.get("action", [])
        score = item.get("score")
        if not isinstance(action, list) or score is None:
            continue
        for idx in action:
            if isinstance(idx, int) and 0 <= idx < option_count:
                visits[idx] += 1
                q_values[idx] = float(score) if q_values[idx] is None else max(float(score), q_values[idx])
    return q_values, visits


def _chosen_score(action: list[int], q_values: list[float | None], rule_scores: list[int]) -> float:
    if action:
        values = [q_values[i] for i in action if 0 <= i < len(q_values) and q_values[i] is not None]
        if values:
            return sum(float(v) for v in values) / len(values)
        rules = [rule_scores[i] for i in action if 0 <= i < len(rule_scores)]
        if rules:
            return sum(float(v) for v in rules) / len(rules)
    return max([float(v) for v in rule_scores], default=0.0)


def _risk_target(parsed, final_action: list[int]) -> float:
    active_hp = parsed.me.active.hp if parsed.me.active else 0
    thin_deck = max(0.0, (6 - parsed.me.deck_count) / 6.0)
    fragile_active = max(0.0, (90 - active_hp) / 90.0) if active_hp else 0.4
    optional_empty = 0.2 if not final_action and parsed.select and parsed.select.min_count == 0 else 0.0
    return max(0.0, min(1.0, 0.45 * thin_deck + 0.45 * fragile_active + optional_empty))


def collect_samples(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations = json.loads(args.fixtures.read_text(encoding="utf-8"))
    if args.max_samples:
        observations = observations[: args.max_samples]

    card_tags = load_card_tags(args.card_tags)
    ledger = GameLedger()
    manager = SearchManager(
        deck=read_deck(),
        config=SearchConfig(
            enabled=args.search,
            max_candidates=args.max_candidates,
            particles=args.particles,
            node_budget=args.node_budget,
            time_budget_sec=args.time_budget,
            switch_margin=args.switch_margin,
        ),
    )
    generator = ActionGenerator()
    samples: list[dict[str, Any]] = []
    skipped = 0

    for idx, obs_dict in enumerate(observations):
        parsed = parse_observation(obs_dict)
        if parsed.select is None:
            skipped += 1
            continue
        ledger.update(parsed)
        v0_action = choose_action(parsed)
        manager.stats.last_report = {}
        search_action = manager.choose(obs_dict, parsed, ledger) if args.search else v0_action
        if not is_legal_action(parsed.select, search_action):
            search_action = v0_action
        final_action = search_action if is_legal_action(parsed.select, search_action) else v0_action

        global_vec, option_vecs, rule_scores, option_summaries = sample_features(parsed, card_tags)
        option_count = len(parsed.select.options)
        report = dict(manager.stats.last_report or {})
        q_values, visits = _score_arrays(option_count, report)
        candidates = [
            {"action": c.as_list(), "source": c.source, "prior": c.prior}
            for c in generator.generate(parsed, k=args.max_candidates)
        ]
        chosen_score = _chosen_score(final_action, q_values, rule_scores)
        value_target = max(-1.0, min(1.0, chosen_score / 1200.0))
        is_key = (
            _action_key(v0_action) != _action_key(final_action)
            or option_count >= 5
            or (parsed.select.max_count > parsed.select.min_count)
            or parsed.me.deck_count <= 8
            or bool(report.get("scores"))
        )
        game_id = f"fixture_{idx // max(1, args.group_size):04d}"
        sample = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": f"{args.sample_prefix}_{idx:06d}",
            "game_id": game_id,
            "turn": parsed.turn,
            "current_player": parsed.current_player,
            "select": {
                "type": parsed.select.type,
                "context": parsed.select.context,
                "min_count": parsed.select.min_count,
                "max_count": parsed.select.max_count,
                "option_count": option_count,
            },
            "legal_mask": [True] * option_count,
            "v0_action": v0_action,
            "search_action": search_action,
            "final_action": final_action,
            "candidate_actions": candidates,
            "search": {
                "enabled": args.search,
                "chosen_source": report.get("chosen_source", "v0_no_search"),
                "elapsed_sec": report.get("elapsed_sec", 0.0),
                "candidate_count": report.get("candidate_count", len(candidates)),
                "scored_count": report.get("scored_count", 0),
                "visits": visits,
                "q_values": q_values,
                "raw_scores": report.get("scores", []),
            },
            "game_result": None if parsed.result == 0 else parsed.result,
            "budget": {
                "teacher": "v1_search" if args.search else "v0_rules",
                "max_candidates": args.max_candidates,
                "particles": args.particles,
                "node_budget": args.node_budget,
                "time_budget_sec": args.time_budget,
            },
            "is_key": is_key,
            "features": global_vec,
            "options": option_summaries,
            "option_features": option_vecs,
            "rule_scores": rule_scores,
            "value_target": value_target,
            "risk_target": _risk_target(parsed, final_action),
        }
        samples.append(sample)

    summary = validate_dataset(samples)
    summary.update(
        {
            "skipped": skipped,
            "search_calls": manager.stats.calls,
            "search_used": manager.stats.used_search,
            "search_fallbacks": manager.stats.fallbacks,
            "search_errors": manager.stats.errors,
            "nodes": manager.stats.nodes,
        }
    )
    return samples, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, default=PROJECT_ROOT / "tests" / "fixtures" / "observations.json")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "distill" / "smoke_samples.jsonl")
    parser.add_argument("--schema-out", type=Path, default=PROJECT_ROOT / "data" / "distill" / "schema.json")
    parser.add_argument("--summary-out", type=Path, default=PROJECT_ROOT / "data" / "distill" / "collect_summary.json")
    parser.add_argument("--card-tags", type=Path, default=PROJECT_ROOT / "data" / "card_tags.json")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--group-size", type=int, default=10)
    parser.add_argument("--sample-prefix", default="m6_smoke")
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--particles", type=int, default=1)
    parser.add_argument("--node-budget", type=int, default=16)
    parser.add_argument("--time-budget", type=float, default=0.02)
    parser.add_argument("--switch-margin", type=float, default=175.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    samples, summary = collect_samples(args)
    write_jsonl(args.out, samples)
    args.schema_out.parent.mkdir(parents=True, exist_ok=True)
    args.schema_out.write_text(
        json.dumps(schema_document(FEATURE_NAMES, TAG_FEATURES), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

