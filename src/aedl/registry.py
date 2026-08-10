"""Evaluator registry.

An evaluator is a callable (spec: TaskSpec, submission: Path) -> EvaluationResult.
Evaluators must be deterministic: same spec + same submission -> same result.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from aedl.result import EvaluationResult
from aedl.spec import TaskSpec


class Evaluator(Protocol):
    def __call__(self, spec: TaskSpec, submission: Path) -> EvaluationResult: ...


_EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(name: str) -> Callable[[Evaluator], Evaluator]:
    def deco(fn: Evaluator) -> Evaluator:
        if name in _EVALUATORS:
            raise ValueError(f"evaluator {name!r} already registered")
        _EVALUATORS[name] = fn
        return fn

    return deco


def get_evaluator(name: str) -> Evaluator:
    try:
        return _EVALUATORS[name]
    except KeyError:
        raise KeyError(f"unknown evaluator {name!r}; registered: {sorted(_EVALUATORS)}") from None
