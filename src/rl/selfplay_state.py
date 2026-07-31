"""Atomic state and immutable checkpoint pools for gated Top2 self-play."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


SCHEMA_VERSION = "top2_selfplay_state_v1"
BRANCHES = {"primary", "reserve"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class SelfPlayState:
    root: Path
    data: dict[str, Any]

    @property
    def path(self) -> Path:
        return self.root / "state.json"

    @property
    def best(self) -> dict[str, Any]:
        return self.data["best"]

    @property
    def history(self) -> list[dict[str, Any]]:
        return self.data["history"]

    @property
    def iteration(self) -> dict[str, Any] | None:
        return self.data.get("iteration")

    @classmethod
    def load(
        cls,
        root: Path,
        *,
        expected_branch: str | None = None,
        expected_deck_id: str | None = None,
    ) -> "SelfPlayState":
        root = Path(root).resolve()
        data = json.loads((root / "state.json").read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported self-play state schema")
        if expected_branch is not None and data.get("branch") != expected_branch:
            raise ValueError(f"self-play branch mismatch: {data.get('branch')} != {expected_branch}")
        if expected_deck_id is not None and data.get("deck_id") != expected_deck_id:
            raise ValueError(f"self-play deck_id mismatch: {data.get('deck_id')} != {expected_deck_id}")
        state = cls(root=root, data=data)
        state._verify_checkpoint(state.best)
        for item in state.history:
            state._verify_checkpoint(item)
        return state

    @classmethod
    def load_or_initialize(
        cls,
        root: Path,
        branch: str,
        deck_id: str,
        initial_checkpoint: Path,
    ) -> "SelfPlayState":
        if branch not in BRANCHES:
            raise ValueError(f"unsupported branch: {branch}")
        root = Path(root).resolve()
        if (root / "state.json").is_file():
            return cls.load(root, expected_branch=branch, expected_deck_id=deck_id)
        source = Path(initial_checkpoint).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        state = cls(
            root=root,
            data={
                "schema_version": SCHEMA_VERSION,
                "branch": branch,
                "deck_id": deck_id,
                "created_at": _utc_now(),
                "best": {},
                "history": [],
                "iteration": None,
            },
        )
        state.root.mkdir(parents=True, exist_ok=True)
        best = state._copy_checkpoint(source, state.root / "best", "initial")
        best.update(
            {
                "branch": branch,
                "deck_id": deck_id,
                "checkpoint_kind": "adapter",
                "source_iteration": None,
                "promoted_at": state.data["created_at"],
                "metrics": {},
            }
        )
        state.data["best"] = best
        state._save()
        return state

    def begin_iteration(self, iteration_id: str) -> dict[str, Any]:
        if not iteration_id or "/" in iteration_id or "\\" in iteration_id:
            raise ValueError("invalid iteration_id")
        current = self.iteration
        if current is not None:
            if current["id"] == iteration_id:
                return current
            if current["status"] == "running":
                raise ValueError(f"iteration already running: {current['id']}")
            previous_ids = set(self.data.get("completed_iterations", []))
            previous_ids.add(current["id"])
            self.data["completed_iterations"] = sorted(previous_ids)
        if iteration_id in set(self.data.get("completed_iterations", [])):
            raise ValueError(f"iteration already completed: {iteration_id}")
        self.data["iteration"] = {
            "id": iteration_id,
            "status": "running",
            "started_at": _utc_now(),
            "best_sha256_at_start": self.best["sha256"],
            "stages": {},
        }
        (self.root / "iterations" / iteration_id).mkdir(parents=True, exist_ok=True)
        self._save()
        return self.iteration

    def complete_stage(self, stage: str, artifacts: dict[str, Any]) -> None:
        if self.iteration is None or self.iteration["status"] != "running":
            raise ValueError("no running iteration")
        stages = self.iteration["stages"]
        if stage in stages:
            if stages[stage]["artifacts"] == artifacts:
                return
            raise ValueError(f"stage already completed with different artifacts: {stage}")
        stages[stage] = {"completed_at": _utc_now(), "artifacts": artifacts}
        self._save()

    def promote(self, candidate: Path, metrics: dict[str, Any]) -> None:
        self._require_running()
        source = Path(candidate).resolve()
        candidate_sha = _sha256(source)
        if candidate_sha == self.best["sha256"]:
            raise ValueError("candidate checkpoint is identical to current best")
        iteration_id = self.iteration["id"]
        old_best_path = Path(self.best["path"])
        archived = self._copy_checkpoint(
            old_best_path,
            self.root / "history",
            f"{iteration_id}-{self.best['sha256'][:12]}",
        )
        archived.update(
            {
                "branch": self.data["branch"],
                "deck_id": self.data["deck_id"],
                "checkpoint_kind": self.best.get("checkpoint_kind", "ppo"),
                "source_iteration": self.best.get("source_iteration"),
                "archived_at": _utc_now(),
                "metrics": self.best.get("metrics", {}),
            }
        )
        self.history.append(archived)
        promoted = self._copy_checkpoint(source, self.root / "best", f"{iteration_id}-{candidate_sha[:12]}")
        promoted.update(
            {
                "branch": self.data["branch"],
                "deck_id": self.data["deck_id"],
                "checkpoint_kind": "ppo",
                "source_iteration": iteration_id,
                "promoted_at": _utc_now(),
                "metrics": metrics,
            }
        )
        self.data["best"] = promoted
        self.iteration.update(
            {"status": "promoted", "completed_at": _utc_now(), "metrics": metrics}
        )
        self._save()

    def reject(self, reason: str, metrics: dict[str, Any]) -> None:
        self._require_running()
        self.iteration.update(
            {
                "status": "rejected",
                "reason": reason,
                "metrics": metrics,
                "completed_at": _utc_now(),
            }
        )
        self._save()

    def _require_running(self) -> None:
        if self.iteration is None or self.iteration["status"] != "running":
            raise ValueError("no running iteration")

    def _copy_checkpoint(self, source: Path, directory: Path, label: str) -> dict[str, Any]:
        if not source.is_file():
            raise FileNotFoundError(source)
        digest = _sha256(source)
        directory.mkdir(parents=True, exist_ok=True)
        destination = (directory / f"{label}-{digest[:12]}.pt").resolve()
        if destination.is_file():
            if _sha256(destination) != digest:
                raise ValueError(f"checkpoint hash collision: {destination}")
        else:
            shutil.copy2(source, destination)
        return {"path": str(destination), "sha256": digest}

    @staticmethod
    def _verify_checkpoint(item: dict[str, Any]) -> None:
        path = Path(item["path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != item["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {path}")

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self.data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
