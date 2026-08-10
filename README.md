# AEDL: Agentic Electromagnetic Design Lab

[![CI](https://github.com/jman4162/aedl-electromagnetic-design-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jman4162/aedl-electromagnetic-design-agent/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![mypy](https://img.shields.io/badge/type--checked-mypy-blue.svg)](https://mypy-lang.org/)

Benchmark tasks and deterministic evaluators for testing whether AI agents can do
RF/microwave design: metasurfaces, phased arrays, and the systems built from them.

The premise: an agent proposes a design; deterministic physics code decides whether it
meets the spec. Every task is a machine-readable requirement set with a pass/fail
evaluator. Agents never grade themselves.

Scope is deliberately narrow: RF/microwave-band electromagnetic design. General
scientific-agent frameworks are a non-goal.

| | |
| --- | --- |
| ![Two radiation patterns at 30 degrees steering: rounding phases onto the 2-bit grid leaves sidelobes above the requirement, while adding a constant 45 degrees to every element first leaves them well below it](https://raw.githubusercontent.com/jman4162/aedl-electromagnetic-design-agent/main/docs/_static/free-lunch.svg) | The first task was broken, and an agent found it in one run. Steering a centred array to 30° puts every ideal phase exactly half a quantization step off the 2-bit grid, so adding a constant 45°, which costs nothing physically, made every phase exactly representable. The quantization error the task was built around disappeared. |

## Status

Pre-release. What works today: the task format, one Tier-2 evaluator, one
calibrated task (`t2-001`), and the agent harness (`aedl run` / `aedl report`)
with cost accounting. What does not exist yet: Tiers 1 and 3, and the remaining
five tasks. `docs/roadmap.md` tracks the rest and records why each design
decision was made.

Planned tiers:

| tier | scope | physics |
|---|---|---|
| 1, element | unit cells, patch antennas | full-wave, [EdgeFEM](https://github.com/jman4162/EdgeFEM) |
| 2, aperture | array synthesis under hardware constraints | [phased-array-modeling](https://github.com/jman4162/Phased-Array-Antenna-Model) |
| 3, system | requirement-driven trades, link budgets | [phased-array-systems](https://github.com/jman4162/phased-array-systems) |

## Install

Clone the repository rather than installing from an index: the tasks live in
`tasks/` and are the point of the project.

```bash
git clone https://github.com/jman4162/aedl-electromagnetic-design-agent
cd aedl-electromagnetic-design-agent
pip install -e ".[dev]"
```

## Use

```bash
aedl list                                              # available tasks
aedl evaluate --task t2-001 --submission weights.npz   # score a submission
aedl evaluate --task t2-001 --submission weights.npz --json result.json
```

Exit code 0 means every requirement passed, 1 means at least one failed.
`aedl run` follows the same convention and adds 2 for a run that produced no
score at all (timeout, crash, or no submission), so CI can gate on it.

### Running an agent against a task

```bash
aedl run --task t2-001 --agent claude --model sonnet --attempts 3
aedl report --runs-dir runs --out leaderboard.md
```

`aedl run` builds an isolated workspace containing only `task.yaml` and a
generated `BRIEF.md`, runs the agent there, scores whatever it leaves in
`submission.npz`, and writes a run bundle:

```
runs/<UTC>_<task>_<agent>_<id>/
  manifest.json   provenance, status, token/cost usage, model-call counts
  result.json     per-requirement scoring
  workspace/      what the agent produced
  calls.jsonl     every physics-model call, tagged by fidelity tier
  agent.stdout.log / agent.stderr.log
```

A bundle is written even when the agent crashes, times out, or submits nothing,
so every attempt stays in the record as scorable evidence.

Other adapters: `--agent command --agent-command "my-agent {brief}"` runs any
CLI (placeholders `{brief}`, `{task}`, `{submission}`), and `mock` is used by the
tests.

**Run benchmarks in a sandbox.** The `claude` adapter passes
`--permission-mode bypassPermissions` because a headless run has no TTY to answer
prompts. It is paired with `--isolation tmpdir` (the default), which puts the
agent's working directory outside this repository. It also pins the tool list and
setting sources so a run does not inherit your personal config, which is what
lets the benchmark reproduce on someone else's machine.

### Result integrity

An agent may legitimately re-derive the scoring metric from the public libraries;
the question is whether it can design, not whether it can guess the metric. It
must not read the worked solution.

Workspace isolation alone does not guarantee that, because the agent's shell can
reach the whole filesystem. One was observed running `find / -iname "*array_pattern*"`
hunting for the evaluator. So `aedl run` strips read permission from every
`tasks/*/reference/` directory while a run is in progress, then restores it.

That is a deterrent, not a sandbox: the agent runs as the same user and could
undo it. **For published numbers, run the harness in a container or VM that has
no copy of this repository.** If a run is killed hard enough to skip the restore,
`chmod 755 tasks/*/reference` puts it back.

### Cost accounting

Two layers, recorded separately because they measure different things:

- **Agent cost**: tokens, dollars, and turns, read from the adapter.
- **Physics cost**: every call into `phased-array-modeling` /
  `phased-array-systems`, counted by fidelity tier. The harness injects a
  `sitecustomize.py` on `PYTHONPATH`, so any Python the agent shells out to is
  instrumented. Nested internal calls are flagged and excluded from the headline
  count.

The second layer exists so that "did the agent allocate simulation fidelity
sensibly?" is answerable later. It cannot be reconstructed after a run.

A task directory contains `task.yaml` (the spec an agent receives) and `reference/`
(a known-good solution, used to verify the evaluator and withheld from agents):

```
tasks/t2-001-steered-beam-2bit/
  task.yaml          # array geometry, hardware constraints, requirements
  reference/solve.py # coordinate descent over the phase states; passes
```

## Calibration run (not a result)

Claude Code (Sonnet 5) against `t2-001`, three attempts, 2026-08-10.
Requirement: sidelobes ≤ −14 dB. Reported for calibration, not as a benchmark
result: one task with no failing cell measures nothing about relative ability,
and the caveats below limit what these numbers support.

For scale, direct 2-bit quantization — the approach the task is designed to
defeat — reaches −10.66 dB and fails.

| | |
| --- | --- |
| ![Peak sidelobe for three approaches against the minus 14 dB requirement: direct rounding and the best global phase rotation both reach minus 10.66 dB and fail, while greedy coordinate descent reaches minus 16.72 dB and passes](https://raw.githubusercontent.com/jman4162/aedl-electromagnetic-design-agent/main/docs/_static/recalibration.svg) | After retargeting to (27°, 10°) no global rotation helps: the best one still leaves 43° of residual against a 90° grid, and scores identically to plain rounding. The requirement now separates approaches instead of being satisfied by a change of phase reference. |

| attempt | outcome | sidelobes | directivity | turns | est. cost | wall |
|---|---|---|---|---|---|---|
| 1 | pass | −15.96 dB | 26.55 dBi | 42 | $1.83 | 855 s |
| 2 | pass | −16.52 dB | 26.40 dBi | 32 | $1.06 | 406 s |
| 3 | pass | −17.02 dB | 26.25 dBi | 27 | $0.93 | 291 s |

Three of three passed, by 1.96, 2.52 and 3.02 dB. Only attempt 3 beat the
reference solution's −16.72 dB; attempts 1 and 2 were 0.76 and 0.20 dB worse
than it.

Two measurement caveats that matter more than the pass rate:

- **The scored metric is optimistically biased for optimized designs.** Read off
  the task's 361×721 grid, the reference measures −16.72 dB; under continuous
  local refinement it is −16.50 dB. Naive quantization shows only 0.003 dB of
  such bias, but any design that sculpts nulls picks up 0.22–0.26 dB, because a
  sculpted null is narrow enough to fall between samples. The bias is
  non-monotonic in grid density, so simply refining the grid is not a fix. The
  benchmark therefore over-credits exactly the designs it exists to
  discriminate. Fixing this means scoring on a verification grid the agent is
  not given.

  | | |
  | --- | --- |
  | ![Error between the grid reading and continuous refinement, against grid density: direct rounding stays at minus 0.003 dB while coordinate descent reads 0.189 dB better than it actually achieves, peaking at 0.227 dB before falling](https://raw.githubusercontent.com/jman4162/aedl-electromagnetic-design-agent/main/docs/_static/grid-bias.svg) | Direct rounding has broad sidelobes that no grid misses. An optimized design's residual peaks are narrow enough to fall between samples, so the grid reports a better number than the design achieves, and refining the grid does not correct it monotonically. |
- **These attempts were not isolated.** They predate the reference-hiding fix,
  and attempt 1's own transcript records that it re-ran the evaluator source
  (`src/aedl/evaluators/array_pattern.py`) to check its work. Re-deriving the
  metric is permitted, but it confirms the agent had this repository in reach
  during a scored attempt, and `--output-format json` keeps no tool-call log
  that could show whether it also opened `reference/`.

None of the three recorded a call to an instrumented `phased-array-modeling`
entry point. Attempt 1's claim to have re-run the evaluator is in tension with
that, since the evaluator calls `compute_full_pattern`, so either the
instrumentation missed the agent's interpreter or the claim is inaccurate. This
is unresolved, and until it is, run-level physics-call counts should be read as
best-effort only.

Treat this as a floor rather than a headline. The task is non-degenerate:
direct quantization fails at −10.7 dB, and no global phase rotation rescues it.
But a frontier model clears it reliably, so discrimination between agents has to
come from the harder tasks in `docs/roadmap.md`. These attempts also predate the
reference-hiding fix described above, so they cannot be certified uncontaminated.

Figures are generated from the task and the reference solution by
`scripts/generate_readme_figures.py`, which first asserts that its physics
agrees with the evaluator. CI fails if they drift from the code that produced
them.

## Task design rules

1. Every metric is computed by deterministic code; same submission, same score.
2. Every task ships with a reference solution that passes and tests proving that
   naive/wrong solutions fail (`tests/`).
3. Thresholds are set so the obvious textbook approach is insufficient. In `t2-001`,
   rounding steering phases onto the 2-bit grid reaches −10.7 dB sidelobes; the
   requirement is −14 dB, and joint optimization of the four phase states reaches
   −16.7 dB.
4. Hardware constraints (phase-only control, shifter bit depth, dead elements) are
   enforced by the evaluator, not trusted to the agent.
5. **Check for degenerate shortcuts before trusting a threshold.** The first
   version of `t2-001` steered to (30°, 0°), where the ideal phases sit exactly
   half a quantization step off the grid. Adding a constant 45° to every element
   costs nothing physically, made those phases exactly representable, and
   collapsed the task. An agent found this on the first run. `tests/` now carries
   a regression test that sweeps global phase offsets and asserts none of them
   clears the sidelobe bar.

## Development

```bash
pytest             # evaluator sanity: references pass, wrong solutions fail
```

## Citation

```bibtex
@software{aedl,
  title  = {AEDL: Agentic Electromagnetic Design Lab},
  author = {John Hodge},
  year   = {2026},
  url    = {https://github.com/jman4162/aedl-electromagnetic-design-agent}
}
```

## Contributing

New benchmark tasks are the most useful contribution. `CONTRIBUTING.md`
describes the three checks a task must pass before its result means anything.

## License

MIT, see [LICENSE](LICENSE).
