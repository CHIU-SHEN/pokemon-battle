"""State and safety primitives for the sequential all-in-one pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Callable, Iterable


STAGES = (
    "initialized",
    "verified",
    "smoke_complete",
    "benchmark_complete",
    "gate_complete",
    "collection_complete",
    "dataset_frozen",
    "gpu_smoke_complete",
    "training_complete",
)


def select_worker_count(results: Iterable[dict]) -> int:
    eligible = [
        result
        for result in results
        if bool(result.get("safe")) and float(result.get("memory_fraction", 1.0)) < 0.8
    ]
    if not eligible:
        raise ValueError("no safe benchmark candidate below the memory limit")
    return int(max(eligible, key=lambda result: float(result["games_per_second"]))["workers"])


def collector_environment() -> dict[str, str]:
    return {
        "MCTS_DEVICE": "cpu",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


@dataclass
class PipelineState:
    path: Path
    stage: str = "initialized"
    managed_pids: list[int] = field(default_factory=list)
    values: dict = field(default_factory=dict)

    @classmethod
    def new(cls, path: Path) -> "PipelineState":
        state = cls(path=path)
        state.save()
        return state

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        value = json.loads(path.read_text(encoding="utf-8"))
        return cls(path, value["stage"], list(value.get("managed_pids", [])), dict(value.get("values", {})))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"schema_version": "mcts_all_in_one_state_v1", "stage": self.stage, "managed_pids": self.managed_pids, "values": self.values}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def advance(self, stage: str) -> None:
        current = STAGES.index(self.stage)
        expected = STAGES[current + 1] if current + 1 < len(STAGES) else None
        if stage != expected:
            raise ValueError(f"expected stage {expected!r}, got {stage!r}")
        self.stage = stage
        self.save()

    def record_managed_pids(self, pids: Iterable[int]) -> None:
        self.managed_pids = [int(pid) for pid in pids]
        self.save()


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def assert_collectors_stopped(
    state: PipelineState,
    *,
    is_alive: Callable[[int], bool] = _pid_alive,
) -> None:
    alive = [pid for pid in state.managed_pids if is_alive(pid)]
    if alive:
        raise RuntimeError(f"managed collector processes are still running: {alive}")
