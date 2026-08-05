from __future__ import annotations

from pathlib import Path

import pytest

from src.rl.mcts_pipeline import (
    PipelineState,
    assert_collectors_stopped,
    collector_environment,
    select_worker_count,
)


def test_selects_fastest_safe_candidate_below_memory_limit() -> None:
    results = [
        {"workers": 12, "games_per_second": 1.0, "memory_fraction": 0.4, "safe": True},
        {"workers": 16, "games_per_second": 1.8, "memory_fraction": 0.6, "safe": True},
        {"workers": 20, "games_per_second": 2.1, "memory_fraction": 0.9, "safe": True},
    ]
    assert select_worker_count(results) == 16


def test_rejects_benchmark_when_no_candidate_is_safe() -> None:
    with pytest.raises(ValueError, match="safe benchmark"):
        select_worker_count([{"workers": 12, "games_per_second": 2, "memory_fraction": 0.5, "safe": False}])


def test_stage_machine_cannot_skip_or_train_before_collectors_stop(tmp_path: Path) -> None:
    state = PipelineState.new(tmp_path / "state.json")
    with pytest.raises(ValueError, match="expected stage"):
        state.advance("dataset_frozen")
    for stage in ("verified", "smoke_complete", "benchmark_complete", "gate_complete", "collection_complete", "dataset_frozen"):
        state.advance(stage)
    state.record_managed_pids([123])
    with pytest.raises(RuntimeError, match="collector"):
        assert_collectors_stopped(state, is_alive=lambda pid: pid == 123)
    assert_collectors_stopped(state, is_alive=lambda pid: False)
    state.advance("gpu_smoke_complete")


def test_state_round_trip_is_atomic_and_environment_limits_threads(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    state = PipelineState.new(path)
    state.advance("verified")
    loaded = PipelineState.load(path)
    assert loaded.stage == "verified"
    assert not path.with_suffix(".json.tmp").exists()
    env = collector_environment()
    assert env["MCTS_DEVICE"] == "cpu"
    assert {env[name] for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")} == {"1"}
