"""AEDL: benchmark tasks and deterministic evaluators for RF/microwave design agents."""

__version__ = "0.0.1"

from aedl.spec import TaskSpec, load_task, discover_tasks
from aedl.result import EvaluationResult, RequirementResult
from aedl.registry import get_evaluator, register_evaluator

# Importing the package registers the built-in evaluators.
from aedl import evaluators as _evaluators  # noqa: F401

__all__ = [
    "TaskSpec",
    "load_task",
    "discover_tasks",
    "EvaluationResult",
    "RequirementResult",
    "get_evaluator",
    "register_evaluator",
]
