"""Aggregate run bundles into a leaderboard."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

_STATUS_GLYPH = {
    "pass": "pass",
    "fail": "fail",
    "no_submission": "no submission",
    "agent_error": "agent error",
    "evaluator_error": "evaluator error",
}


def load_runs(runs_dir: Path) -> list[dict]:
    records = []
    for manifest in sorted(runs_dir.glob("*/manifest.json")):
        try:
            records.append(json.loads(manifest.read_text()))
        except json.JSONDecodeError:
            continue
    return records


def _failed_requirements(rec: dict) -> str:
    failed = [r["requirement_id"] for r in rec.get("requirements", []) if not r["passed"]]
    return ", ".join(failed) if failed else "—"


def model_of(rec: dict) -> str:
    """Attribute a run to the model that produced most of its output.

    Preferred over the recorded `model` field: bundles written before the
    adapter learned to disambiguate `modelUsage` name the wrong model, and
    manifests are never rewritten after the fact.
    """
    from aedl.harness.adapters.claude_cli import primary_model

    usage = rec.get("usage", {})
    derived = primary_model(usage.get("extra", {}).get("model_usage") or {})
    return derived or rec.get("model") or "—"


def _fmt(value, spec: str = "", dash: str = "—") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else dash


def runs_table(records: list[dict]) -> str:
    header = (
        "| run | task | agent | model | outcome | failed requirements | "
        "turns | tokens out | cost USD | wall s | model calls |"
    )
    rows = [header, "|" + "---|" * 11]
    for r in records:
        usage = r.get("usage", {})
        calls = r.get("calls", {})
        rows.append(
            f"| `{r['run_id'][:24]}` | {r['task_id']} | {r['agent']} | "
            f"{model_of(r)} | {_STATUS_GLYPH.get(r['status'], r['status'])} | "
            f"{_failed_requirements(r)} | {_fmt(usage.get('num_turns'), 'd')} | "
            f"{_fmt(usage.get('output_tokens'), ',d')} | {_fmt(usage.get('cost_usd'), '.3f')} | "
            f"{_fmt(r.get('agent_wall_time_s'), '.0f')} | {_fmt(calls.get('total_calls'), 'd')} |"
        )
    return "\n".join(rows)


def summary_table(records: list[dict]) -> str:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["task_id"], model_of(r))].append(r)

    rows = [
        "| task | model | attempts | passed | pass rate | median cost USD |",
        "|" + "---|" * 6,
    ]
    for (task, agent), group in sorted(grouped.items()):
        passed = sum(1 for r in group if r["status"] == "pass")
        costs = sorted(
            r["usage"]["cost_usd"] for r in group if isinstance(r.get("usage", {}).get("cost_usd"), (int, float))
        )
        median = costs[len(costs) // 2] if costs else None
        rows.append(
            f"| {task} | {agent} | {len(group)} | {passed} | "
            f"{passed / len(group):.0%} | {_fmt(median, '.3f')} |"
        )
    return "\n".join(rows)


def render(records: list[dict]) -> str:
    if not records:
        return "No runs found."
    return (
        "## Pass rate\n\n"
        + summary_table(records)
        + "\n\n## Runs\n\n"
        + runs_table(records)
        + "\n"
    )
