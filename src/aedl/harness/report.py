"""Aggregate run bundles into a leaderboard."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_STATUS_GLYPH = {
    "pass": "pass",
    "fail": "fail",
    "timeout": "timeout",
    "no_submission": "no submission",
    "agent_error": "agent error",
    "evaluator_error": "evaluator error",
}


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Read every manifest under runs_dir, skipping ones that cannot be parsed.

    A run bundle is written by a `finally` block and may be truncated if the
    machine died mid-write, so reading must never be the thing that fails.
    """
    records = []
    for manifest in sorted(runs_dir.glob("*/manifest.json")):
        try:
            record = json.loads(manifest.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _failed_requirements(rec: dict[str, Any]) -> str:
    failed = [
        r.get("requirement_id", "?")
        for r in rec.get("requirements", [])
        if isinstance(r, dict) and not r.get("passed", False)
    ]
    return ", ".join(failed) if failed else "—"


def model_of(rec: dict[str, Any]) -> str:
    """Attribute a run to the model that produced most of its output.

    Preferred over the recorded `model` field: bundles written before the
    adapter learned to disambiguate `modelUsage` name the wrong model, and
    manifests are never rewritten after the fact.
    """
    from aedl.harness.adapters.claude_cli import primary_model

    usage = rec.get("usage") or {}
    extra = usage.get("extra") or {}
    derived = primary_model(extra.get("model_usage") or {})
    return derived or rec.get("model") or "—"


def _fmt(value: object, spec: str = "", dash: str = "—") -> str:
    """Format a number for a table cell, falling back to a dash.

    Guards the format code as well as the type: a manifest that stored a turn
    count as 2.0 would otherwise blow up on an integer format spec.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return dash
    try:
        return format(value, spec)
    except (ValueError, TypeError):
        return format(value)


def runs_table(records: list[dict[str, Any]]) -> str:
    header = (
        "| run | task | agent | model | outcome | failed requirements | "
        "turns | tokens out | cost USD | wall s | model calls |"
    )
    rows = [header, "|" + "---|" * 11]
    for r in records:
        usage = r.get("usage") or {}
        calls = r.get("calls") or {}
        status = r.get("status", "unknown")
        rows.append(
            f"| `{str(r.get('run_id', '?'))[:24]}` | {r.get('task_id', '?')} | "
            f"{r.get('agent', '?')} | "
            f"{model_of(r)} | {_STATUS_GLYPH.get(status, status)} | "
            f"{_failed_requirements(r)} | {_fmt(usage.get('num_turns'), 'd')} | "
            f"{_fmt(usage.get('output_tokens'), ',d')} | {_fmt(usage.get('cost_usd'), '.3f')} | "
            f"{_fmt(r.get('agent_wall_time_s'), '.0f')} | {_fmt(calls.get('total_calls'), 'd')} |"
        )
    return "\n".join(rows)


def summary_table(records: list[dict[str, Any]]) -> str:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        grouped[(r.get("task_id", "?"), model_of(r))].append(r)

    rows = [
        "| task | model | attempts | passed | pass rate | median cost USD |",
        "|" + "---|" * 6,
    ]
    for (task, agent), group in sorted(grouped.items()):
        passed = sum(1 for r in group if r.get("status") == "pass")
        costs = sorted(
            cost
            for r in group
            if isinstance(cost := (r.get("usage") or {}).get("cost_usd"), (int, float))
        )
        median = costs[len(costs) // 2] if costs else None
        rows.append(
            f"| {task} | {agent} | {len(group)} | {passed} | "
            f"{passed / len(group):.0%} | {_fmt(median, '.3f')} |"
        )
    return "\n".join(rows)


def render(records: list[dict[str, Any]]) -> str:
    if not records:
        return "No runs found."
    return (
        "## Pass rate\n\n" + summary_table(records) + "\n\n## Runs\n\n" + runs_table(records) + "\n"
    )
