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

## Baseline

Claude Code (Sonnet 5) against `t2-001`, three independent attempts,
2026-08-09. Requirement: sidelobes ≤ −14 dB.

| attempt | outcome | sidelobes | directivity | turns | cost | wall |
|---|---|---|---|---|---|---|
| 1 | pass | −15.96 dB | 26.55 dBi | 42 | $1.83 | 855 s |
| 2 | pass | −16.52 dB | 26.40 dBi | 32 | $1.06 | 406 s |
| 3 | pass | −17.02 dB | 26.25 dBi | 27 | $0.93 | 291 s |

3/3, with 2–3 dB of margin, matching or beating the reference solution's
−16.72 dB. None of the three called into `phased-array-modeling`; each wrote its
own array factor.

Treat this as a floor rather than a headline. The task is now non-degenerate:
direct quantization fails at −10.7 dB, and no global phase rotation rescues it.
But a frontier model clears it reliably, so discrimination between agents has to
come from the harder tasks in `docs/roadmap.md`. These attempts also predate the
reference-hiding fix described above, so they cannot be certified uncontaminated.

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
