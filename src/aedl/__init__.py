"""AEDL: benchmark tasks and deterministic evaluators for RF/microwave design agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aedl")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0.0.0.dev0"

# Importing the package registers the built-in evaluators.
from aedl import evaluators as _evaluators  # noqa: F401
from aedl.registry import get_evaluator, register_evaluator
from aedl.result import EvaluationResult, RequirementResult
from aedl.spec import TaskSpec, discover_tasks, load_task

__all__ = [
    "EvaluationResult",
    "RequirementResult",
    "TaskSpec",
    "discover_tasks",
    "get_evaluator",
    "load_task",
    "register_evaluator",
]
