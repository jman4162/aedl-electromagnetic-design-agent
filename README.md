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

Pre-release. What works today: the task format, evaluators for Tiers 2 and 3,
two calibrated tasks (`t2-001`, `t3-001`), and the agent harness (`aedl run` /
`aedl report`) with cost accounting and optional MCP server attachment. What
does not exist yet: Tier 1 and the remaining four Tier-2 tasks.
`docs/roadmap.md` tracks the rest and records why each design decision was
made.

Tiers:

| tier | scope | physics | status |
|---|---|---|---|
| 1, element | unit cells, patch antennas | full-wave, [EdgeFEM](https://github.com/jman4162/EdgeFEM) | planned |
| 2, aperture | array synthesis under hardware constraints | [phased-array-modeling](https://github.com/jman4162/Phased-Array-Antenna-Model) | `t2-001` |
| 3, system | requirement-driven architecture trades | [phased-array-systems](https://github.com/jman4162/phased-array-systems), cross-checked by [phased-array-modeling](https://github.com/jman4162/Phased-Array-Antenna-Model) and [opensatcom](https://github.com/jman4162/opensatcom) | `t3-001` |

`t3-001` is the cross-layer task: design a 28 GHz LEO terminal architecture
(aperture, taper, quantization, PA class, digitization) that closes the link
worst-case over a held-out envelope of scan/rain/sky/failure conditions,
under prime-power and unit-cost ceilings priced from a parts table. Pattern
claims are recomputed by full pattern integration, and the link margin is
independently recomputed by opensatcom, a codebase the design model shares
nothing with; clear-sky agreement between the two is itself a scored
requirement (their rain models diverge by an order of magnitude, which is why
rain agreement deliberately is not). Every threshold was frozen from
measurement; `scripts/calibrate_t3_001.py` reproduces each one.

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

To attach MCP servers (the composition experiment t3-001 exists to measure),
pass an explicit config; it composes with the pinned `--strict-mcp-config`,
so a run uses exactly these servers and inherits nothing from your personal
configuration:

```bash
aedl run --task t3-001 --agent claude --model sonnet \
  --mcp-config mcp-servers.json \
  --mcp-tools mcp__opensatcom__link_snapshot,mcp__opensatcom__link_validate_config
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

`calls.jsonl` appears only when an instrumented call is actually seen. None of the
three committed bundles has one, and all three record `total_calls: 0`, which is the
open instrumentation question described under the calibration run below.

The three bundles in `runs/` are committed rather than regenerated, because a
nondeterministic agent cannot reproduce them. `runs/README.md` explains how to read
them, in particular that each `result.json` records the score as of 2026-08-10, before
the sidelobe metric was fixed.

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
| ![Peak sidelobe for three approaches against the minus 14 dB requirement: direct rounding and the best global phase rotation both reach minus 10.66 dB and fail, while greedy coordinate descent reaches minus 16.98 dB and passes](https://raw.githubusercontent.com/jman4162/aedl-electromagnetic-design-agent/main/docs/_static/recalibration.svg) | After retargeting to (27°, 10°) no global rotation helps: the best one still leaves 43° of residual against a 90° grid, and scores identically to plain rounding. The requirement now separates approaches instead of being satisfied by a change of phase reference. |

| attempt | outcome | sidelobes | directivity | turns | est. cost | wall |
|---|---|---|---|---|---|---|
| 1 | pass | −15.96 dB | 26.55 dBi | 42 | $1.83 | 855 s |
| 2 | pass | −16.52 dB | 26.40 dBi | 32 | $1.06 | 406 s |
| 3 | pass | −17.02 dB | 26.25 dBi | 27 | $0.93 | 291 s |

Three of three passed. The sidelobe column above is what the metric said at the
time; the metric has since been fixed, and the caveat below gives the corrected
values.

Two measurement caveats that matter more than the pass rate:

- **The scored metric was reading the main-lobe skirt.** It reported the highest
  sample outside a fixed 8° radius of the target. For a 16×16 array steered to
  27° that radius sits inside the main lobe, so a design whose true sidelobes
  fall below its own skirt level at 8° was scored on the skirt instead. The
  reference and attempt 3 both hit that, each reporting a value at exactly 8.00°
  from the target. The metric is now the second-highest local maximum of the
  pattern. Corrected: the reference achieves −16.98 dB rather than −16.72, and
  the three attempts −15.96, −16.52 and −17.59 dB, so only attempt 3 still comes
  within 0.6 dB of the reference and none beats it.

  The offset is the small part. The metric saturated: once a design pushed its
  sidelobes below the skirt, further suppression stopped registering, which is
  the one thing a benchmark has to be able to see.

  This also retires an earlier entry here, which attributed the gap to grid
  sampling flattering optimized designs by 0.22–0.26 dB. That diagnosis was
  wrong twice over: the direction was backwards, and the continuous measurement
  used to establish it was itself clamped to the same 8° radius, so it climbed
  the skirt and stopped on the boundary. The check had inherited the defect it
  was checking. With the metric fixed, the 361×721 reading agrees with
  unconstrained continuous refinement to within 0.01 dB.
  `scripts/verify_sidelobe_metric.py` reproduces all of it.

  | | |
  | --- | --- |
  | ![Error against continuous refinement versus grid density for the coordinate-descent design: the fixed 8 degree radius reads 0.26 dB high at the scored grid and wanders to 0.46 dB as the grid is refined, while the local-maximum metric sits on zero at every density](https://raw.githubusercontent.com/jman4162/aedl-electromagnetic-design-agent/main/docs/_static/grid-bias.svg) | The fixed radius reports the hottest sample on the 8° circle, so which sample that is depends on the grid: it wanders over 0.23 dB and never converges. Local-maximum detection returns a real lobe, and reads the same value at every density. Direct rounding is unaffected either way, because its true sidelobes sit well above the skirt. |
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
   −17.0 dB.
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
