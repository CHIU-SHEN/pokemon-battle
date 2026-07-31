"""Resumable stage orchestration for one branch-bound self-play iteration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.rl.selfplay_state import SelfPlayState


Stage = Callable[[dict[str, Any]], dict[str, Any]]
STAGE_ORDER = ("rollout", "train", "holdout", "gate", "regression")


@dataclass
class SelfPlayRunner:
    state: SelfPlayState
    iteration_id: str
    stages: dict[str, Stage]

    def run(self, *, stop_after: str | None = None) -> dict[str, Any]:
        iteration = self.state.begin_iteration(self.iteration_id)
        if iteration["status"] != "running":
            return self._report()
        for name in STAGE_ORDER:
            if name not in self.stages:
                continue
            if name in iteration["stages"]:
                artifacts = iteration["stages"][name]["artifacts"]
            else:
                context = self._context()
                artifacts = self.stages[name](context)
                if not isinstance(artifacts, dict):
                    raise TypeError(f"stage {name} must return a dict")
                self.state.complete_stage(name, artifacts)
                iteration = self.state.iteration
            if name == "gate" and artifacts.get("status") == "reject":
                self.state.reject(artifacts.get("reason", "arena_gate"), artifacts)
                return self._report()
            if stop_after == name:
                return self._report()

        train = self._stage_artifacts("train")
        gate = self._stage_artifacts("gate")
        regression = self._stage_artifacts("regression")
        if gate and gate.get("status") == "promote_ready":
            if not regression or not regression.get("passed", False):
                self.state.reject("regression_gate", regression or {})
            elif int(regression.get("illegal_actions", 0)) != 0:
                self.state.reject("illegal_actions", regression)
            else:
                checkpoint = Path(train["checkpoint"])
                self.state.promote(checkpoint, {**gate, "regression": regression})
        elif gate:
            self.state.reject(gate.get("reason", "arena_gate_incomplete"), gate)
        return self._report()

    def _stage_artifacts(self, name: str) -> dict[str, Any] | None:
        stage = self.state.iteration["stages"].get(name)
        return stage["artifacts"] if stage else None

    def _context(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "iteration_id": self.iteration_id,
            "iteration_root": self.state.root / "iterations" / self.iteration_id,
            "best": dict(self.state.best),
            "stages": {
                key: value["artifacts"]
                for key, value in self.state.iteration["stages"].items()
            },
        }

    def _report(self) -> dict[str, Any]:
        return {
            "schema_version": "top2_selfplay_iteration_report_v1",
            "branch": self.state.data["branch"],
            "deck_id": self.state.data["deck_id"],
            "iteration_id": self.iteration_id,
            "status": self.state.iteration["status"],
            "best": dict(self.state.best),
            "stages": dict(self.state.iteration["stages"]),
        }
