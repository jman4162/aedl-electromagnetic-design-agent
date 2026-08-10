# Contributing to AEDL

The most useful contribution is a new benchmark task. This document covers how
to write one so that its result means something.

## Development setup

```bash
git clone https://github.com/jman4162/aedl-electromagnetic-design-agent
cd aedl-electromagnetic-design-agent
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

.venv/bin/pytest            # tests
.venv/bin/ruff check .      # lint
.venv/bin/ruff format .     # format
.venv/bin/mypy              # types
scripts/slopcheck.sh        # prose linter, advisory
```

CI runs all of the above.

## Authoring a task

A task lives in `tasks/<id>-<slug>/`:

```
tasks/t2-001-steered-beam-2bit/
  task.yaml            the specification an agent receives
  reference/solve.py   a known-good solution, never shown to agents
```

`task.yaml` needs `id`, `tier`, `title`, `evaluator`, and `requirements`.
`context` holds whatever the evaluator needs; each requirement names one metric
the evaluator produces and one bound (`max` or `min`, exactly one).

Tier 1 is element-level (unit cells, single radiators), tier 2 is aperture-level
(array synthesis), tier 3 is system-level (requirement-driven trades).

### The three checks a task must pass

A benchmark task is only worth adding if you can show all three numerically.
Put the numbers in the pull request.

1. **The textbook approach fails.** If the obvious first thing an engineer would
   try already satisfies the requirements, the task measures nothing. For
   `t2-001`, rounding steering phases onto the 2-bit grid reaches −10.7 dB
   against a −14 dB requirement.
2. **A real technique passes**, with margin, and it is implemented in
   `reference/solve.py`. If your own best effort barely clears the bar, the
   threshold is probably infeasible rather than hard.
3. **No cheap invariance collapses the problem.** This is the one that bites.
   Before trusting a threshold, ask what transformations are free in the
   physics — a global phase offset, a uniform scaling, a relabelling — and
   check that none of them turns the naive approach into a passing one. The
   first version of `t2-001` died exactly here: at (30°, 0°) a constant 45°
   phase shift made the quantization error vanish, and an agent found it on the
   first run. `tests/test_t2_001.py` now guards that property.

### Tests

Each task gets a `tests/test_<id>.py` asserting that the reference passes, that
each requirement catches a specific wrong solution, that scoring is
deterministic, and that the invariance from check 3 does not rescue the naive
approach.

### Briefs

The brief handed to the agent is generated from `task.yaml`, so anything in
`summary` reaches the agent. Stating how far the naive approach falls short is
allowed and consistent across tasks; naming the technique the reference uses is
not.

## Adding an evaluator

An evaluator is a callable `(TaskSpec, Path) -> EvaluationResult`, registered
with `@register_evaluator("name")` in `src/aedl/evaluators/`, and selected by
`evaluator.name` in `task.yaml`. It must:

- be deterministic, with no dependence on model output, wall-clock time, or
  unseeded randomness;
- produce every metric that any requirement references, and raise a clear error
  naming the metric if a task asks for one it does not compute;
- enforce the task's hardware constraints itself rather than trusting the
  submission to have respected them;
- put its raw metric values in `EvaluationResult.info["metrics"]`.

## Adding an agent adapter

Implement the `AgentAdapter` protocol in `src/aedl/harness/adapters/` and
register it with `@register_adapter("name")`. Report whatever usage your agent
exposes through `AgentUsage`; leave fields `None` when it does not.

## Pull requests

Branch from `main`, keep the change focused, and make sure CI is green. For a
new task, include the three numbers from the checks above in the description.
