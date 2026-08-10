# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Pre-release. The strategic decision (Aug 2026): AEDL is a **benchmark-first,
RF/microwave-band** project, scoped to benchmark tasks and deterministic
evaluators for AI design agents. General agentic-science frameworks are out of
scope.

Phase 1 goal: 6 calibrated tasks across three tiers (element / aperture / system), a
harness to score any coding agent, and a single-strong-agent baseline. The harness and
one task exist. `docs/roadmap.md` holds the rest, the rationale, and the calibration
log of what agent runs have already invalidated.

## Commands

```bash
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                                     # all tests
.venv/bin/pytest tests/test_t2_001.py -k sidelobes   # one test
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy
scripts/slopcheck.sh                                 # prose linter, advisory

.venv/bin/aedl list
.venv/bin/aedl evaluate --task t2-001 --submission w.npz
.venv/bin/aedl run --task t2-001 --agent claude --model sonnet
.venv/bin/aedl report --runs-dir runs --out leaderboard.md
```

`evaluate` exits 0 when all requirements pass and 1 otherwise; `run` adds 2 for
a run that produced no score (timeout, crash, no submission).

If `import aedl` fails outside pytest, the editable install's `.pth` is not
being honored; `PYTHONPATH=src` is the workaround. The pytest config sets
`pythonpath = ["src"]` so tests never depend on it.

## Architecture

The core contract: **agents propose designs; deterministic physics code scores them.**
Nothing in an evaluator may depend on model output or randomness (fixed seeds only in
reference solutions, never in scoring).

- `src/aedl/spec.py`: `TaskSpec`/`Requirement` loaded from `tasks/*/task.yaml`. Each
  requirement is one metric with a max or min bound; evaluators must produce every
  metric a task references.
- `src/aedl/registry.py`: evaluator registry. An evaluator is
  `(TaskSpec, submission_path) -> EvaluationResult`, registered by name with
  `@register_evaluator`; task YAML selects it via `evaluator.name`.
- `src/aedl/evaluators/array_pattern.py`: Tier-2 evaluator built on
  `phased-array-modeling` (John's own library). Enforces hardware constraints
  (phase-only amplitude, phase-shifter bit grid), applies dead elements itself
  (never trusts the agent to), then computes pattern metrics.
- `tasks/<id>-<slug>/task.yaml`: the spec an agent receives; `reference/solve.py`
  beside it is the known-good solution used only for evaluator verification.
- `tests/`: evaluator sanity. Every task's reference must pass and each requirement
  must be shown to catch a specific wrong solution (naive quantization, unquantized
  phases, wrong steering, amplitude taper).
- `src/aedl/harness/`: the agent harness. `workspace.py` materializes what the
  agent sees (task.yaml + generated BRIEF.md, never `reference/`, by default in a
  tmpdir outside the repo). `adapter.py` is the protocol + registry, mirroring
  `registry.py`; adapters are `claude_cli`, `command`, `mock`. `instrument.py`
  counts physics-model calls by fidelity tier by injecting a `sitecustomize.py` on
  the child's PYTHONPATH. `run.py` orchestrates and always writes a bundle in a
  `finally`. `report.py` renders the leaderboard.

Task design rules, both learned the hard way:

1. Thresholds are set so the textbook approach fails. For t2-001, direct 2-bit
   rounding reaches −10.7 dB sidelobes, the requirement is −14 dB, and joint
   optimization of the phase states reaches −16.7 dB. Verify all three numerically
   before fixing a threshold.
2. Hunt for degenerate shortcuts before trusting a task. The first t2-001 steered
   to (30°, 0°), where the ideal phases are uniformly half a quantization step off
   the grid, so a free global phase rotation made them exact and the task
   collapsed, and an agent found this on the first run. See the calibration log in
   `docs/roadmap.md`; the regression test is in `tests/test_t2_001.py`.

Upstream repos this builds on (all pip deps, all John's): `phased-array-modeling`,
`phased-array-systems`, `edgefem` (Tier 1, not yet wired in), `apab` (MCP tool layer,
planned for the agent harness). Fix library bugs upstream rather than working around
them here. One known case: `phased_array.compute_directivity` uses `np.trapz`,
removed in NumPy 2 (aedl's evaluator carries a local `_directivity_dbi` until fixed).

## Longer-term direction (from planning docs)

The benchmark is Phase 1. Phase 2 experiments, each measured as a delta against a
single-strong-agent baseline on the same tasks: agent-managed fidelity allocation
(analytic → reduced-order → surrogate → EdgeFEM → HFSS/CST), agents mapped to physical
abstraction levels (unit-cell → aperture → system) with an evaluator holding veto
power, and an experiment store recording (requirement, hypothesis, design, simulation,
result, decision) tuples with provenance and cost. Agents get constrained MCP tool
interfaces (APAB), never raw shell access to solvers.
