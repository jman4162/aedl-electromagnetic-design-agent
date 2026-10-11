"""Run one task across the EdgeArray baseline ladder.

Each baseline (b0 rules only through b4 full agent) runs ``--repeats`` times
against one model endpoint through the ``edgearray`` adapter. Every attempt
writes an ordinary run bundle under ``--runs-dir``. The script prints a
markdown table (task, baseline, model, runs, pass rate, mean tokens, mean
cost) and appends it to the leaderboard file only with
``--write-leaderboard``.

Usage:
  python scripts/run_ladder.py --task t1-001 --model ollama/qwen3.5:9b
  python scripts/run_ladder.py --task t4-001 --model anthropic/claude-sonnet \
      --baselines b2,b4 --repeats 5 --write-leaderboard

The EdgeArray executable is ``$AEDL_EDGEARRAY_BIN`` or ``edgearray`` on PATH.
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

HEADER = (
    "| task | baseline | model | runs | pass rate | mean tokens | mean cost USD |\n"
    "|---|---|---|---|---|---|---|"
)


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
) -> list[dict[str, Any]]:
    """Run every (baseline, repeat) and return one row per attempt."""
    spec = find_task(tasks_dir, task_id)
    rows: list[dict[str, Any]] = []
    for baseline in baselines:
        adapter = get_adapter("edgearray", model=model, baseline=baseline)
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
                    "model": model,
                    "attempt": attempt,
                    "status": record.status,
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
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["task"], row["baseline"], row["model"]), []).append(row)
    lines = [HEADER]
    for (task, baseline, model), group in groups.items():
        passed = sum(r["status"] == "pass" for r in group)
        tokens = _mean([r["tokens"] for r in group])
        cost = _mean([r["cost_usd"] for r in group])
        lines.append(
            f"| {task} | {baseline} | {model} | {len(group)} | {passed / len(group):.0%} | "
            f"{'—' if tokens is None else f'{tokens:,.0f}'} | "
            f"{'—' if cost is None else f'{cost:.4f}'} |"
        )
    return "\n".join(lines)


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
    )
    table = ladder_table(rows)
    print(table)
    if args.write_leaderboard:
        heading = f"\n## EdgeArray baseline ladder: {args.task}, {args.model} ({utc_stamp()})\n\n"
        with args.leaderboard.open("a") as fh:
            fh.write(heading + table + "\n")
        print(f"appended to {args.leaderboard}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
