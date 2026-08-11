# Changelog

All notable changes to AEDL will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Task specification format (`tasks/<id>/task.yaml`) with requirements expressed
  as a metric and a bound, scored by a named evaluator from a registry.
- `array_pattern` evaluator for Tier-2 aperture tasks, built on
  `phased-array-modeling`. It enforces hardware constraints and applies element
  failures itself rather than trusting the submission.
- `t2-001`: low-sidelobe steered beam with 2-bit phase control and 13 dead
  elements.
- Agent harness: `aedl run` executes an agent in an isolated workspace, scores
  the artifact it leaves, and always writes a run bundle. Adapters for the
  Claude CLI, an arbitrary command, and a scripted mock.
- Two-layer cost accounting: agent tokens/cost/turns, and physics-model calls
  counted by fidelity tier through an injected `sitecustomize` shim.
- `aedl report` renders a leaderboard from run bundles.
- Reference solutions are made unreadable during a run so an agent cannot find
  the answer key by searching the filesystem.
- Packaging and tooling: MIT `LICENSE`, ruff, mypy strict, CI, `py.typed`.

### Changed

- `t2-001` retargeted from (30°, 0°) to (27°, 10°) and its sidelobe requirement
  tightened from −9 dB to −14 dB. At the original geometry the ideal phases sat
  uniformly half a quantization step off the 2-bit grid, so a global phase
  rotation — which costs nothing physically — made them exactly representable
  and erased the quantization error the task was meant to pose. An agent found
  this on the first run.
- The reference solution is now greedy coordinate descent over the four phase
  states rather than dithered quantization.
- `peak_sidelobe_level_db` is now the second-highest local maximum of the
  pattern, replacing "highest sample outside a fixed angular radius of the
  target". `exclusion_radius_deg` is still honoured when a task sets it, and
  `t2-001` no longer does.

### Fixed

- The peak-sidelobe metric could report a point on the main-lobe skirt. A fixed
  exclusion radius has to be wider than the main lobe, and the main lobe widens
  with scan angle and with any taper. `t2-001` used 8°, which for a 16×16 array
  steered to 27° sits inside the main lobe, so once a design pushed its true
  sidelobes below its own skirt level at 8° the metric reported the skirt and
  stopped responding to the sidelobes. Two of the five designs measured hit
  exactly that, both at exactly 8.00° from the target. Reported values move:

  | design | before | after | independent refinement |
  |---|---|---|---|
  | direct 2-bit rounding | −10.66 | −10.66 | −10.67 |
  | reference solution | −16.72 | −16.98 | −16.98 |
  | agent `d68c3564` | −15.96 | −15.96 | −15.96 |
  | agent `4eb80fae` | −16.52 | −16.52 | −16.52 |
  | agent `85d86723` | −17.02 | −17.59 | −17.60 |

  The metric was pessimistic for the two affected designs, so it understated
  them and, worse, saturated: further sidelobe suppression stopped registering,
  which is what a benchmark needs most to be able to see. `t2-001` still
  discriminates, with naive rounding failing at −10.66 dB against the −14 dB
  bar. `scripts/verify_sidelobe_metric.py` reproduces the table above, and
  `tests/test_t2_001.py` pins both the local-maximum property and agreement
  with continuous refinement.

- The new peak-sidelobe metric returned 0 dB for any beam steered to phi = 0.
  `compute_full_pattern` spans a full turn with the endpoint duplicated, so the
  first and last columns hold identical samples. A peak at phi = 0 appears in
  both, and the descending walk counted the second copy as a fresh lobe.
  Already-visited samples are now skipped. `t2-001` steers to phi = 10 degrees
  and was never affected; the figure generator, which plots the original
  (30 degrees, 0) geometry, is what surfaced it. Under the fixed metric naive
  rounding at that geometry reads −7.18 dB rather than the −5.92 dB the
  fixed-radius metric gave, and still fails the −9 dB bar it faced.

  This also retires an earlier diagnosis. The gap between the grid reading and
  a continuous measurement was attributed to grid sampling flattering optimized
  designs by 0.22–0.26 dB. It was the exclusion radius the whole time: with the
  metric fixed, the 361×721 grid reading matches continuous refinement to
  within 0.01 dB on every design measured.


- A timed-out agent could be scored as `pass`. Timeouts are now a distinct
  terminal status and the submission is not read.
- `aedl run` always exited 0. It now exits non-zero when any attempt fails to
  produce a scored pass.
- `numpy>=1.24` was declared while the evaluators require `np.trapezoid` from
  NumPy 2.0.
- Concurrent runs could permanently strip permissions from reference
  directories; a sentinel now records the real mode and only the run that hid a
  directory restores it.
- `aedl report` raised on truncated or partial run manifests.
