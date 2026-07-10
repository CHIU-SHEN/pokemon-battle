"""Legal-action fallback for unknown or low-confidence selections."""

from __future__ import annotations

from typing import Any

from .deck_profile_abomasnow import CARD_BASE_VALUE
from .parser import ParsedState, card_id_from_area, enum_value, safe_get


def _option_card_value(parsed: ParsedState | None, option: Any) -> int:
    direct = safe_get(option, "cardId")
    if direct is not None:
        return CARD_BASE_VALUE.get(int(direct), 0)
    if parsed is None:
        return 0
    card_id = card_id_from_area(
        parsed,
        enum_value(safe_get(option, "area")),
        safe_get(option, "index"),
        safe_get(option, "playerIndex"),
    )
    if card_id is None:
        card_id = card_id_from_area(
            parsed,
            enum_value(safe_get(option, "inPlayArea")),
            safe_get(option, "inPlayIndex"),
            parsed.current_player,
        )
    return CARD_BASE_VALUE.get(int(card_id), 0) if card_id is not None else 0


def _select_attr(select: Any, camel: str, snake: str, default: Any = None) -> Any:
    return safe_get(select, camel, safe_get(select, snake, default))


def safe_action(select: Any, parsed: ParsedState | None = None, prefer_empty: bool = True) -> list[int]:
    """Return a legal action for any official SelectData-like object.

    The fallback is deliberately conservative: optional unknown selections return
    an empty list, while mandatory selections choose the highest known card-value
    options and otherwise the first legal indices.
    """

    min_count = int(_select_attr(select, "minCount", "min_count", 0) or 0)
    max_count = int(_select_attr(select, "maxCount", "max_count", min_count) or min_count)
    options = list(_select_attr(select, "option", "options", []) or [])

    if min_count == 0 and prefer_empty:
        return []
    if max_count <= 0 or not options:
        return []

    ranked = sorted(
        range(len(options)),
        key=lambda i: (_option_card_value(parsed, options[i]), -i),
        reverse=True,
    )
    count = min(max(min_count, 0), max_count, len(options))
    return ranked[:count]


def is_legal_action(select: Any, action: list[int]) -> bool:
    min_count = int(_select_attr(select, "minCount", "min_count", 0) or 0)
    max_count = int(_select_attr(select, "maxCount", "max_count", min_count) or min_count)
    options = list(_select_attr(select, "option", "options", []) or [])
    return (
        isinstance(action, list)
        and all(isinstance(x, int) for x in action)
        and min_count <= len(action) <= max_count
        and len(set(action)) == len(action)
        and all(0 <= x < len(options) for x in action)
    )

