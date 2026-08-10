"""Evaluation result types and JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RequirementResult:
    requirement_id: str
    metric: str
    value: float
    limit: str
    passed: bool


@dataclass(frozen=True)
class EvaluationResult:
    task_id: str
    passed: bool
    requirements: tuple[RequirementResult, ...]
    info: dict[str, Any] = field(default_factory=dict)
    cost: dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(asdict(self), indent=indent, default=float)

    def summary_table(self) -> str:
        lines = [f"task: {self.task_id}  ->  {'PASS' if self.passed else 'FAIL'}"]
        if not self.requirements:
            return lines[0] + "\n  (no requirements)"
        width = max(len(r.requirement_id) for r in self.requirements)
        for r in self.requirements:
            status = "pass" if r.passed else "FAIL"
            lines.append(
                f"  {r.requirement_id:<{width}}  {r.metric} = {r.value:.4g}  "
                f"(required {r.limit})  [{status}]"
            )
        return "\n".join(lines)
