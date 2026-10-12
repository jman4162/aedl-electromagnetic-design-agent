"""Run one task across the EdgeArray baseline ladder.

Each baseline (b0 rules only through b4 full agent) runs ``--repeats`` times
against one model endpoint through the ``edgearray`` adapter. Every attempt
writes an ordinary run bundle under ``--runs-dir``. The script prints a
markdown table (task, baseline, model, runs, pass rate, mean tokens, mean
cost, plus a tier column for screening-tier verdicts and mean recomputed
passes for tasks that report them) under a heading stating the approval
policy and EdgeArray budget, and appends both to the leaderboard file only
with ``--write-leaderboard``. b0 runs without a model and shows "-".

Usage:
  python scripts/run_ladder.py --task t1-001 --model ollama/qwen3.5:9b
  python scripts/run_ladder.py --task t4-001 --model anthropic/claude-sonnet \
      --baselines b2,b4 --repeats 5 --write-leaderboard

The EdgeArray executable is ``$AEDL_EDGEARRAY_BIN`` or ``edgearray`` on PATH.

Attempts are isolated: the adapter gives each one a data directory and project
id of its own, so baselines and repeats never share design state. The
deliverable is built from that data directory (t1-001 and t4-001 only; a task
without a mapping ends as ``no_submission`` with the reason in the record).
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aedl.harness import get_adapter, run_task
from aedl.harness.adapters.edgearray import BASELINES
from aedl.harness.record import utc_stamp
from aedl.spec import find_task

#: Shown in the model column for b0, which runs EdgeArray's rules without an LLM.
NO_MODEL = "-"


def run_ladder(
    task_id: str,
    model: str,
    *,
    baselines: tuple[str, ...] = BASELINES,
    repeats: int = 3,
    tasks_dir: Path = Path("tasks"),
    runs_dir: Path = Path("runs"),
    timeout_s: int = 900,
    instrumented: bool = True,
    approve_all: bool = False,
    budget_s: int | None = None,
) -> list[dict[str, Any]]:
    """Run every (baseline, repeat) and return one row per attempt.

    Every baseline gets the same approval policy and the same EdgeArray
    budget, so the rows differ only in what the baseline enables.
    """
    spec = find_task(tasks_dir, task_id)
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        uses_model = baseline != "b0"
        adapter = get_adapter(
            "edgearray",
            model=model if uses_model else None,
            baseline=baseline,
            approve_all=approve_all,
            budget_s=budget_s,
        )
        for attempt in range(1, repeats + 1):
            record, bundle = run_task(
                spec, adapter, runs_dir=runs_dir, timeout_s=timeout_s, instrumented=instrumented
            )
            usage = record.usage
            tokens = [usage.get("input_tokens"), usage.get("output_tokens")]
            rows.append(
                {
                    "task": spec.id,
                    "baseline": baseline,
                    "model": model if uses_model else NO_MODEL,
                    "attempt": attempt,
                    "status": record.status,
                    "tier": record.tier,
                    "recomputed_passes": record.score.get("recomputed_passes"),
                    "tokens": sum(t for t in tokens if isinstance(t, int))
                    if any(isinstance(t, int) for t in tokens)
                    else None,
                    "cost_usd": usage.get("cost_usd"),
                    "bundle": str(bundle),
                }
            )
            print(f"{baseline} attempt {attempt}/{repeats}: {record.status} ({bundle})")
    return rows


def _mean(values: list[Any]) -> float | None:
    numbers = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return statistics.fmean(numbers) if numbers else None


def ladder_table(rows: list[dict[str, Any]]) -> str:
    """One row per (task, baseline, model).

    A tier column appears when any run was scored at a screening tier, and a
    mean-recomputed-passes column when any task reports that score (t4-001).
    """
    tiers = any(r.get("tier") for r in rows)
    passes = any(r.get("recomputed_passes") is not None for r in rows)
    columns = ["task", "baseline", "model", "runs", "pass rate"]
    columns += ["tier"] if tiers else []
    columns += ["mean recomputed passes"] if passes else []
    columns += ["mean tokens", "mean cost USD"]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]

    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["baseline"], row["model"]), []).append(row)
    for (task, baseline, model), group in groups.items():
        passed = sum(r["status"] == "pass" for r in group)
        cells = [task, baseline, model, str(len(group)), f"{passed / len(group):.0%}"]
        if tiers:
            labels = sorted({str(r["tier"]) for r in group if r.get("tier")})
            cells.append(", ".join(labels) or "—")
        if passes:
            mean_passes = _mean([r.get("recomputed_passes") for r in group])
            cells.append("—" if mean_passes is None else f"{mean_passes:.1f}")
        tokens = _mean([r["tokens"] for r in group])
        cost = _mean([r["cost_usd"] for r in group])
        cells.append("—" if tokens is None else f"{tokens:,.0f}")
        cells.append("—" if cost is None else f"{cost:.4f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ladder_heading(task: str, model: str, approve_all: bool, budget_s: int | None) -> str:
    """The leaderboard heading, stating the run conditions shared by every row."""
    approvals = (
        "all approved (--approve-all)" if approve_all else "all rejected (EdgeArray default)"
    )
    budget = f"{budget_s} s" if budget_s is not None else "EdgeArray default"
    return (
        f"## EdgeArray baseline ladder: {task}, {model} ({utc_stamp()})\n\n"
        f"Approvals: {approvals}. EdgeArray budget per attempt: {budget}. "
        f"b0 uses no model.\n"
    )


def _baselines(text: str) -> tuple[str, ...]:
    chosen = tuple(b.strip() for b in text.split(",") if b.strip())
    unknown = [b for b in chosen if b not in BASELINES]
    if unknown or not chosen:
        raise argparse.ArgumentTypeError(f"baselines must be drawn from {BASELINES}, got {text!r}")
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, help="task id, e.g. t1-001")
    parser.add_argument("--model", required=True, help="EdgeArray model endpoint id")
    parser.add_argument("--baselines", type=_baselines, default=BASELINES, help="e.g. b0,b2,b4")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tasks-dir", type=Path, default=Path("tasks"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--timeout", type=int, default=900, help="per attempt, seconds")
    parser.add_argument("--no-instrument", action="store_true")
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="pass --approve-all to EdgeArray (default: its approvals are rejected)",
    )
    parser.add_argument(
        "--budget-s",
        type=int,
        default=None,
        help="EdgeArray --budget-s, the same for every baseline",
    )
    parser.add_argument("--leaderboard", type=Path, default=Path("leaderboard.md"))
    parser.add_argument(
        "--write-leaderboard",
        action="store_true",
        help="append the table to --leaderboard (default: print only)",
    )
    args = parser.parse_args(argv)

    rows = run_ladder(
        args.task,
        args.model,
        baselines=args.baselines,
        repeats=args.repeats,
        tasks_dir=args.tasks_dir,
        runs_dir=args.runs_dir,
        timeout_s=args.timeout,
        instrumented=not args.no_instrument,
        approve_all=args.approve_all,
        budget_s=args.budget_s,
    )
    heading = ladder_heading(args.task, args.model, args.approve_all, args.budget_s)
    table = ladder_table(rows)
    print(heading)
    print(table)
    if args.write_leaderboard:
        with args.leaderboard.open("a") as fh:
            fh.write("\n" + heading + "\n" + table + "\n")
        print(f"appended to {args.leaderboard}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
