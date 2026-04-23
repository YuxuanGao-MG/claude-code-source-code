from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .config import Config

_session_ctx: ContextVar[str | None] = ContextVar("openmesh_session", default=None)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class SpendTotals:
    run: float = 0.0
    by_model: dict[str, float] = field(default_factory=dict)
    by_task: dict[str, float] = field(default_factory=dict)


class Tracer:
    """JSONL trace writer + spend guardrails + task resume."""

    def __init__(self, run_dir: Path, config: Config):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.calls_path = self.run_dir / "calls.jsonl"
        self.completed_path = self.run_dir / "completed_tasks.jsonl"
        self.config = config
        self._lock = threading.Lock()
        self.totals = SpendTotals()

    def log_call(self, row: dict) -> None:
        line = json.dumps(row, default=_json_default)
        with self._lock, self.calls_path.open("a") as f:
            f.write(line + "\n")

    def mark_task_done(self, task_id: str) -> None:
        with self._lock, self.completed_path.open("a") as f:
            f.write(json.dumps({"task_id": task_id, "ts": time.time()}) + "\n")

    def completed_tasks(self) -> set[str]:
        if not self.completed_path.exists():
            return set()
        out: set[str] = set()
        for line in self.completed_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            out.add(json.loads(line)["task_id"])
        return out

    def record_spend(self, model: str, usd: float, task_id: str | None) -> None:
        with self._lock:
            self.totals.run += usd
            self.totals.by_model[model] = self.totals.by_model.get(model, 0.0) + usd
            if task_id:
                self.totals.by_task[task_id] = self.totals.by_task.get(task_id, 0.0) + usd

            c = self.config
            if c.max_usd_per_run is not None and self.totals.run > c.max_usd_per_run:
                raise BudgetExceeded(
                    f"run spend ${self.totals.run:.2f} exceeded cap ${c.max_usd_per_run:.2f}"
                )
            if c.max_usd_per_model is not None and self.totals.by_model[model] > c.max_usd_per_model:
                raise BudgetExceeded(
                    f"model {model} spend ${self.totals.by_model[model]:.2f} "
                    f"exceeded cap ${c.max_usd_per_model:.2f}"
                )
            if (
                task_id
                and c.max_usd_per_task is not None
                and self.totals.by_task[task_id] > c.max_usd_per_task
            ):
                raise BudgetExceeded(
                    f"task {task_id} spend ${self.totals.by_task[task_id]:.2f} "
                    f"exceeded cap ${c.max_usd_per_task:.2f}"
                )


@contextmanager
def trace_session(task_id: str) -> Iterator[str]:
    token = _session_ctx.set(task_id)
    try:
        yield task_id
    finally:
        _session_ctx.reset(token)


def current_session() -> str | None:
    return _session_ctx.get()


def _json_default(o):
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if isinstance(o, (set, frozenset)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)
