"""Command-line interface: list tasks, evaluate submissions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from aedl.registry import get_evaluator
from aedl.spec import discover_tasks, find_task


def _add_tasks_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        default=Path("tasks"),
        help="directory containing <task>/task.yaml files (default: ./tasks)",
    )


def cmd_list(args: argparse.Namespace) -> int:
    specs = discover_tasks(args.tasks_dir)
    if not specs:
        print(f"no tasks found in {args.tasks_dir}", file=sys.stderr)
        return 2
    for s in specs:
        print(f"{s.id}  tier {s.tier}  {s.title}")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    spec = find_task(args.tasks_dir, args.task)
    evaluator = get_evaluator(spec.evaluator)
    result = evaluator(spec, args.submission)
    print(result.summary_table())
    if args.json:
        args.json.write_text(result.to_json())
        print(f"wrote {args.json}")
    return 0 if result.passed else 1


def cmd_run(args: argparse.Namespace) -> int:
    from aedl.harness import get_adapter, run_task

    spec = find_task(args.tasks_dir, args.task)
    adapter = get_adapter(
        args.agent,
        model=args.model,
        template=args.agent_command,
        max_budget_usd=args.max_budget_usd,
        mcp_config=args.mcp_config,
        mcp_tools=args.mcp_tools,
    )

    statuses = []
    for attempt in range(1, args.attempts + 1):
        if args.attempts > 1:
            print(f"--- attempt {attempt}/{args.attempts} ---")
        record, bundle = run_task(
            spec,
            adapter,
            runs_dir=args.runs_dir,
            isolation=args.isolation,
            timeout_s=args.timeout,
            instrumented=not args.no_instrument,
        )
        print(f"{record.status}: {bundle}")
        if record.error:
            print(f"  {record.error}")
        for req in record.requirements:
            mark = "pass" if req["passed"] else "FAIL"
            print(
                f"  {req['requirement_id']}: {req['metric']} = {req['value']:.4g} "
                f"(required {req['limit']}) [{mark}]"
            )
        if record.calls.get("total_calls"):
            print(f"  model calls: {record.calls['calls_by_tier']}")
        statuses.append(record.status)

    # Exit 0 only when every attempt produced a scored pass, matching the
    # contract `aedl evaluate` already follows, so CI can gate on this.
    if all(s == "pass" for s in statuses):
        return 0
    return 1 if all(s in ("pass", "fail") for s in statuses) else 2


def cmd_report(args: argparse.Namespace) -> int:
    from aedl.harness.report import load_runs, render

    records = load_runs(args.runs_dir)
    text = render(records)
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="aedl",
        description="AEDL benchmark: RF/microwave design tasks with deterministic evaluators",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="list available tasks")
    _add_tasks_dir(p_list)
    p_list.set_defaults(func=cmd_list)

    p_eval = sub.add_parser("evaluate", help="score a submission against a task")
    _add_tasks_dir(p_eval)
    p_eval.add_argument("--task", required=True, help="task id, e.g. t2-001")
    p_eval.add_argument("--submission", required=True, type=Path, help="path to submission file")
    p_eval.add_argument("--json", type=Path, help="also write the full result as JSON")
    p_eval.set_defaults(func=cmd_evaluate)

    p_run = sub.add_parser("run", help="run an agent against a task and score it")
    _add_tasks_dir(p_run)
    p_run.add_argument("--task", required=True, help="task id, e.g. t2-001")
    p_run.add_argument("--agent", default="claude", help="adapter name (claude, command, mock)")
    p_run.add_argument("--model", help="model identifier passed to the adapter")
    p_run.add_argument(
        "--agent-command",
        default="",
        help="command template for the 'command' adapter; may use {brief} {task} {submission}",
    )
    p_run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p_run.add_argument("--isolation", choices=("tmpdir", "inplace"), default="tmpdir")
    p_run.add_argument("--timeout", type=int, default=900, help="agent wall-clock limit, seconds")
    p_run.add_argument("--attempts", type=int, default=1, help="number of independent attempts")
    p_run.add_argument("--max-budget-usd", type=float, default=5.0)
    p_run.add_argument(
        "--mcp-config",
        type=Path,
        help="MCP server config to attach (composes with --strict-mcp-config: "
        "exactly these servers, nothing inherited)",
    )
    p_run.add_argument(
        "--mcp-tools",
        default="",
        help="comma-separated mcp__<server>__<tool> names to allow",
    )
    p_run.add_argument(
        "--no-instrument",
        action="store_true",
        help="skip model-call instrumentation (recorded in the manifest)",
    )
    p_run.set_defaults(func=cmd_run)

    p_report = sub.add_parser("report", help="aggregate run bundles into a leaderboard")
    p_report.add_argument("--runs-dir", type=Path, default=Path("runs"))
    p_report.add_argument("--out", type=Path, help="also write the markdown to this path")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
