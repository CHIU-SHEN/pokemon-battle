"""Unit checks for Adapter deck similarity and tier rules."""

from scripts.build_adapter_sampling_views import classify, similarities, sorted_deck_hash


def main() -> int:
    exact = [1] * 20 + [2] * 20 + [3] * 20
    close = [1] * 18 + [2] * 18 + [3] * 18 + [4] * 6
    far = [10] * 20 + [11] * 20 + [12] * 20
    assert sorted_deck_hash(exact) == sorted_deck_hash(list(reversed(exact)))
    assert classify(similarities(exact, exact), exact=True) == "exact"
    assert classify(similarities(exact, close), exact=False) == "similar"
    assert classify(similarities(exact, far), exact=False) == "general"
    print("OK: Adapter deck similarity and tier rules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
