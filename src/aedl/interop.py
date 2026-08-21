"""Bridge from sysml2kit requirement specs to aedl requirements.

sysml2kit extracts machine-checkable requirements from a SysML v2 model as
``RequirementSpec`` objects carrying dual-form thresholds: operator form
(``op`` + ``value``) and bound form (``minimum``/``maximum``). aedl's
:class:`~aedl.spec.Requirement` is bound-form with exactly one bound, so
one-sided specs map directly and an equality spec (both bounds set) splits
into a ``-lo``/``-hi`` pair.

Duck-typed input: pass sysml2kit ``RequirementSpec`` objects or their
``model_dump()`` dicts (what the sysml2kit MCP tool ``requirements_extract``
returns). sysml2kit is not a dependency of aedl.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from aedl.spec import Requirement


def _get(spec: Any, key: str) -> Any:
    if isinstance(spec, Mapping):
        return spec.get(key)
    return getattr(spec, key, None)


def requirements_from_spec(spec: Any) -> tuple[Requirement, ...]:
    """Convert one sysml2kit spec into one or two aedl requirements.

    Returns an empty tuple for specs without a threshold (prose-only
    requirements are not checkable here). An equality threshold returns a
    ``-lo``/``-hi`` pair, since aedl requirements carry exactly one bound.
    """
    minimum = _get(spec, "minimum")
    maximum = _get(spec, "maximum")
    spec_id = str(_get(spec, "id"))
    metric = str(_get(spec, "metric_key"))
    if minimum is None and maximum is None:
        return ()
    if minimum is not None and maximum is not None:
        return (
            Requirement(id=f"{spec_id}-lo", metric=metric, min=float(minimum)),
            Requirement(id=f"{spec_id}-hi", metric=metric, max=float(maximum)),
        )
    if minimum is not None:
        return (Requirement(id=spec_id, metric=metric, min=float(minimum)),)
    return (Requirement(id=spec_id, metric=metric, max=float(maximum)),)


def requirements_from_specs(specs: Iterable[Any]) -> tuple[Requirement, ...]:
    """Convert a list of sysml2kit specs, dropping prose-only entries."""
    out: list[Requirement] = []
    for spec in specs:
        out.extend(requirements_from_spec(spec))
    return tuple(out)
