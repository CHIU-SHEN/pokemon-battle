"""PyTorch Deck-Adapter agent for local and server Arena evaluation.

The supervised policy is deliberately gated to non-trivial mandatory
single-choice decisions, matching the existing SL-0 online runtime. Other
selection shapes use the reviewed rules agent and finally the safe fallback.
"""

from __future__ import annotations

from contextlib import contextmanager
import os
import pathlib
from pathlib import Path
import sys
from typing import Iterator

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_DIR = PROJECT_ROOT / "submission"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from agent.fallback import is_legal_action, safe_action  # noqa: E402
from agent.parser import parse_observation  # noqa: E402
from agent.rules import choose_action  # noqa: E402
from cg.api import Observation, to_observation_class  # noqa: E402
from src.train.adapter_model import DeckAdapterPolicyValueNet  # noqa: E402
from src.train.features import load_card_tags, sample_features  # noqa: E402
from src.train.shared_model import SharedModelConfig, SharedPolicyValueNet  # noqa: E402


@contextmanager
def portable_checkpoint_paths() -> Iterator[None]:
    """Load Linux-created pathlib objects from checkpoints on Windows."""

    original = pathlib.PosixPath
    if sys.platform == "win32":
        pathlib.PosixPath = pathlib.WindowsPath  # type: ignore[misc,assignment]
    try:
        yield
    finally:
        pathlib.PosixPath = original  # type: ignore[misc,assignment]


def _read_deck(path: Path) -> list[int]:
    deck = [int(line.strip()) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(deck) != 60:
        raise ValueError(f"candidate deck must contain 60 cards: {path}={len(deck)}")
    return deck


def _device_from_environment() -> torch.device:
    requested = os.environ.get("PTCG_ADAPTER_DEVICE", "cpu").strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"PTCG_ADAPTER_DEVICE={requested!r}, but CUDA is unavailable")
    return torch.device(requested)


class AdapterArenaAgent:
    """Callable online agent bound to one candidate deck and Adapter."""

    def __init__(
        self,
        candidate_id: str,
        *,
        project_root: Path = PROJECT_ROOT,
        device: torch.device | str | None = None,
    ) -> None:
        if not candidate_id or Path(candidate_id).name != candidate_id:
            raise ValueError(f"invalid candidate id: {candidate_id!r}")
        self.candidate_id = candidate_id
        self.project_root = Path(project_root)
        self.device = torch.device(device) if device is not None else _device_from_environment()
        self.deck_path = self.project_root / "data" / "high_score_decks" / candidate_id / "deck.csv"
        self.adapter_path = self.project_root / "artifacts" / "adapters_top10" / candidate_id / "best.pt"
        self.base_path = self.project_root / "artifacts" / "sl0_shared_full" / "best.pt"
        for path in (self.deck_path, self.adapter_path, self.base_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        self.deck = _read_deck(self.deck_path)
        self.tags = load_card_tags(self.project_root / "data" / "card_tags.json")
        self.model = self._load_model()
        self._last_source = "initialized"
        self.model_calls = 0
        self.fallback_calls = 0
        self.exceptions = 0

    def _load_model(self) -> DeckAdapterPolicyValueNet:
        with portable_checkpoint_paths():
            base_checkpoint = torch.load(self.base_path, map_location="cpu", weights_only=False)
            adapter_checkpoint = torch.load(self.adapter_path, map_location="cpu", weights_only=False)
        if base_checkpoint.get("schema_version") != "sl0_shared_checkpoint_v1":
            raise ValueError("unsupported shared checkpoint schema")
        if adapter_checkpoint.get("schema_version") != "deck_adapter_checkpoint_v1":
            raise ValueError("unsupported Adapter checkpoint schema")
        if adapter_checkpoint.get("candidate_id") != self.candidate_id:
            raise ValueError(
                f"Adapter candidate mismatch: expected={self.candidate_id!r} "
                f"actual={adapter_checkpoint.get('candidate_id')!r}"
            )
        base_hash = str(base_checkpoint.get("dataset_sha256", "")).upper()
        adapter_hash = str(adapter_checkpoint.get("base_dataset_sha256", "")).upper()
        if not base_hash or adapter_hash != base_hash:
            raise ValueError("Adapter is not bound to the frozen shared-model dataset")

        base = SharedPolicyValueNet(SharedModelConfig(**base_checkpoint["model_config"]))
        base.load_state_dict(base_checkpoint["model_state"], strict=True)
        model = DeckAdapterPolicyValueNet(base, int(adapter_checkpoint["bottleneck_dim"]))
        state = adapter_checkpoint["adapter_state"]
        model.adapter.load_state_dict(state["adapter"], strict=True)
        model.policy_delta.load_state_dict(state["policy_delta"], strict=True)
        model.value_delta.load_state_dict(state["value_delta"], strict=True)
        return model.to(self.device).eval()

    def action_source(self) -> str:
        return self._last_source

    def diagnostics(self) -> dict[str, int | str]:
        return {
            "candidate_id": self.candidate_id,
            "model_calls": self.model_calls,
            "fallback_calls": self.fallback_calls,
            "exceptions": self.exceptions,
        }

    def _model_action(self, obs: Observation, obs_dict: dict) -> list[int]:
        parsed = parse_observation(obs_dict)
        select = parsed.select
        if select is None or select.min_count != 1 or select.max_count != 1 or len(select.options) <= 1:
            raise ValueError("Adapter is gated to non-trivial mandatory single-choice decisions")
        global_vec, option_vecs, _, _ = sample_features(parsed, self.tags)
        option_count = len(option_vecs)
        batch = {
            "global_features": torch.tensor([global_vec], dtype=torch.float32, device=self.device),
            "option_features": torch.tensor([option_vecs], dtype=torch.float32, device=self.device),
            "legal_mask": torch.ones((1, option_count), dtype=torch.bool, device=self.device),
            "player_deck": torch.tensor([self.deck], dtype=torch.long, device=self.device),
            "player_deck_mask": torch.ones((1, len(self.deck)), dtype=torch.bool, device=self.device),
            "opponent_deck": torch.zeros((1, 1), dtype=torch.long, device=self.device),
            "opponent_deck_mask": torch.zeros((1, 1), dtype=torch.bool, device=self.device),
        }
        with torch.inference_mode():
            logits = self.model(batch)["policy_logits"][0]
        action = [int(logits.argmax().item())]
        if not is_legal_action(obs.select, action):
            raise ValueError(f"Adapter produced an illegal action: {action}")
        return action

    def __call__(self, obs_dict: dict | None) -> list[int]:
        if obs_dict is None or obs_dict.get("select") is None:
            self._last_source = "deck"
            return list(self.deck)
        obs: Observation = to_observation_class(obs_dict)
        if obs.select is None:
            self._last_source = "deck"
            return list(self.deck)
        if obs.select.minCount == obs.select.maxCount == 1 and len(obs.select.option) == 1:
            self._last_source = "forced"
            return [0]
        if obs.select.minCount == obs.select.maxCount == 1:
            try:
                action = self._model_action(obs, obs_dict)
                self.model_calls += 1
                self._last_source = "adapter"
                return action
            except Exception:
                self.exceptions += 1

        self.fallback_calls += 1
        try:
            parsed = parse_observation(obs_dict)
            action = choose_action(parsed)
            if is_legal_action(obs.select, action):
                self._last_source = "rules_fallback"
                return action
        except Exception:
            self.exceptions += 1
        self._last_source = "safe_fallback"
        return safe_action(obs.select, prefer_empty=False)

