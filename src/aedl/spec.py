"""Task specification loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Requirement:
    """One pass/fail criterion. Exactly one of max/min is set."""

    id: str
    metric: str
    max: float | None = None
    min: float | None = None

    def __post_init__(self) -> None:
        if (self.max is None) == (self.min is None):
            raise ValueError(
                f"requirement {self.id!r}: exactly one of max/min must be set"
            )

    def check(self, value: float) -> bool:
        if self.max is not None:
            return value <= self.max
        assert self.min is not None
        return value >= self.min

    @property
    def limit(self) -> str:
        if self.max is not None:
            return f"<= {self.max}"
        return f">= {self.min}"


@dataclass(frozen=True)
class TaskSpec:
    id: str
    tier: int
    title: str
    summary: str
    evaluator: str
    evaluator_params: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    deliverable: dict = field(default_factory=dict)
    requirements: tuple[Requirement, ...] = ()
    path: Path | None = None


def load_task(task_yaml: Path) -> TaskSpec:
    raw = yaml.safe_load(task_yaml.read_text())
    for key in ("id", "tier", "title", "evaluator", "requirements"):
        if key not in raw:
            raise ValueError(f"{task_yaml}: missing required key {key!r}")
    reqs = tuple(
        Requirement(
            id=r["id"],
            metric=r["metric"],
            max=r.get("max"),
            min=r.get("min"),
        )
        for r in raw["requirements"]
    )
    return TaskSpec(
        id=raw["id"],
        tier=int(raw["tier"]),
        title=raw["title"],
        summary=raw.get("summary", "").strip(),
        evaluator=raw["evaluator"]["name"] if isinstance(raw["evaluator"], dict) else raw["evaluator"],
        evaluator_params=raw["evaluator"].get("params", {}) if isinstance(raw["evaluator"], dict) else {},
        context=raw.get("context", {}),
        deliverable=raw.get("deliverable", {}),
        requirements=reqs,
        path=task_yaml,
    )


def discover_tasks(tasks_dir: Path) -> list[TaskSpec]:
    """Find all task.yaml files under tasks_dir, sorted by task id."""
    specs = [load_task(p) for p in sorted(tasks_dir.glob("*/task.yaml"))]
    ids = [s.id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate task ids in {tasks_dir}")
    return specs


def find_task(tasks_dir: Path, task_id: str) -> TaskSpec:
    for spec in discover_tasks(tasks_dir):
        if spec.id == task_id:
            return spec
    raise KeyError(f"task {task_id!r} not found in {tasks_dir}")
