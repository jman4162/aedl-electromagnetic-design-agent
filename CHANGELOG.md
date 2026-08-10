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
  states (−16.7 dB) rather than dithered quantization.

### Fixed

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
