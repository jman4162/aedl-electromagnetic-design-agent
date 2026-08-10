# Roadmap: Tiers 2 and 3 to first baseline result

Target: **AEDL-Bench v0.1**, comprising 6 calibrated tasks, an agent harness, and a
single-strong-agent baseline, in 2–3 months at ~5 hrs/week (~50 hours).

## Status

The harness is built (`aedl run`, `aedl report`, cost accounting, three
adapters). The sequencing argument below was vindicated on the first real agent
run; see **Calibration log**.

## Sequencing decision: harness before tasks

*Recorded as written before the harness existed; the reasoning is why it was
built first, and the calibration log below is what it caught.*

Right now every threshold in `t2-001` is a hand-derived guess. It is calibrated
against *analytic* solutions only. (Those were the original (30°, 0°) figures:
naive rounding −5.9 dB, dithered quantization −12 dB. Both were superseded by
the retarget to (27°, 10°) recorded in the calibration log.) No agent had
attempted it, so where a frontier model landed relative to that line was
unmeasured. Writing five more tasks before
finding out risks a suite that is uniformly trivial or uniformly impossible,
discovered at the end when there is no budget left to re-calibrate.

The harness interface is already fully determined by `t2-001` (task.yaml in, `.npz`
out, evaluator scores), so there is nothing to learn by waiting. Cost accounting in
particular must be designed in now: fidelity allocation is the Phase-2 research claim,
and per-call cost cannot be reconstructed after the fact.

Budget: ~8 hours. Every task after it costs less because calibration becomes a
measurement instead of an argument.

### Harness scope

```
aedl run --task t2-001 --agent claude --model sonnet --attempts 3
```

- Materializes a workspace: `task.yaml` plus a generated `BRIEF.md` stating the
  submission contract. Excludes `reference/`.
- Runs the agent with a turn/wall-clock budget, captures the transcript.
- Scores the produced submission, writes `result.json` + the run record.
- **Cost record per run**: evaluator calls, model calls broken out by fidelity tier
  (`aperture_model` / `reduced_order` / `full_wave`), wall time, tokens.
- `aedl report --runs-dir runs` renders the leaderboard for the README and the paper.

Agents may reimplement the metrics, which is fine and expected. The question is
whether an agent can *design*, not whether it can guess the scoring function. What
must stay hidden is the held-out evaluation conditions used in Tier 3 (below).

## Tier 2 — aperture (4 more tasks)

Chosen so each defeats a *different* textbook approach. Topic diversity is worthless
if every task is solved by the same move.

| id | task | textbook approach that fails | technique needed |
|----|------|------------------------------|------------------|
| t2-001 ✅ | low-sidelobe steered beam, 2-bit phase, 13 dead elements | rounding to the phase grid (−10.7 dB SLL vs −14 required) | joint optimization over the four phase states |
| t2-002 | two simultaneous beams, phase-only control | phase-only projection of the beam sum (floor −9.2 dB) | iterative phase retrieval under a constant-modulus constraint |
| t2-003 | restore sidelobes at 45° scan after a failure cluster | re-steering the surviving elements | amplitude/phase re-optimization around the hole |
| t2-004 | hold beam pointing across a 20% band with per-subarray TTD + 3-bit element phase | phase-only steering (squints out of spec) | delay/phase split across two hierarchy levels |
| t2-005 | minimum element count meeting SLL + directivity in a fixed aperture | uniform thinning | density taper / combinatorial search |

**t2-004 is the one to prioritize** if scope has to shrink. It is the only Tier-2 task
that spans two abstraction levels (subarray delay vs element phase), which is the
multi-scale claim the whole project rests on, in miniature and at Tier-2 cost.

Build it from `compute_subarray_weights_hybrid` (`phased_array/wideband.py:621`),
which returns `subarray_delays` and `element_phases` separately. Call it once at
band centre, freeze the element phases, and advance only the delay term with
frequency. Four traps found in a source audit, each of which would silently
produce a meaningless task:

- `steering_vector_hybrid` recomputes the element phase at the *call* frequency,
  so it reproduces full-element TTD exactly (agreement to 5e-16) and shows **zero**
  squint. It does not model frozen phase shifters.
- Consequently `compute_beam_squint(steering_mode='hybrid')` returns identically
  0.0 squint at every frequency, and `analyze_instantaneous_bandwidth` falls into
  its "Exceeds 100%" branch with a hardcoded `ibw_ratio = 1.5`.
- `compute_beam_squint` reads the peak off a 361-point grid spanning θ0±30°, so
  squint is quantized to 0.167°. Use `n_points >= 3001` for a sub-0.02° criterion.
  Its `relative_gain` is measured at the squinted peak, not at the intended angle,
  so it does not capture gain loss on target; compute that separately.
- `create_rectangular_subarrays` integer-truncates a non-dividing partition and
  silently lumps the orphaned elements into subarray 0. Use divisor-exact layouts.

Also note `compute_subarray_weights` applies one phase per *subarray* with no
intra-subarray steering (verified: only 2 distinct phases across 4 subarrays at
θ0=30°). That is the classic quantization-lobe generator, useful as the deliberately
bad arm of the comparison, not as a stand-in for hybrid steering.

Measured difficulty (10 GHz, 16×16, cos element):

- **t2-002**: naive phase-only projection of the two-beam sum gives 0.31 dB beam
  imbalance and a −9.19 dB sidelobe floor. Setting the floor requirement at −12 dB
  makes the naive move fail. **Verify a Gerchberg-Saxton-style iteration actually
  reaches −12 dB before fixing the threshold**, or the task is infeasible.

## Calibration log

### 2026-08-09 — first agent run invalidated `t2-001` (and justified the harness)

The original `t2-001` steered to (30°, 0°). The first agent attempt passed
immediately with a −12.32 dB sidelobe level and 28.39 dBi, *exactly* the ideal
unquantized figures, which should be unreachable with 2-bit phase shifters.

Cause: with a centred 16×16 array at 0.5λ, element x-positions are half-integer
multiples of the spacing, so steering to 30° puts every ideal phase on
{45°, 135°, 225°, 315°}, uniformly half a quantization step off the 2-bit grid.
Adding a constant 45°, which is a free change of phase reference, makes every
phase exactly representable and eliminates quantization error entirely. The
agent's submission differed from the ideal phases by exactly 45° on all 256
elements. The intended difficulty never existed, and the dithered-quantization
reference solution (26.42 dBi) was strictly worse than this trivial exact one.

The general lesson is physical, not incidental: **when the phase ramp is
commensurate with the quantization grid, the coherent worst case is always
removable by a global phase offset.** Coarse-quantization tasks must use a
steering direction where the ramp is incommensurate.

Fix applied: retarget to (27°, 10°), where no global rotation helps (43°
residual to the grid), and raise the sidelobe requirement from −9 dB to −14 dB.
Measured at the new geometry:

| approach | sidelobes | directivity | verdict |
|---|---|---|---|
| direct rounding onto the 2-bit grid | −10.66 dB | 27.87 dBi | fails |
| best global phase rotation | −10.66 dB | 27.87 dBi | fails (shortcut closed) |
| greedy coordinate descent over the four states | −16.72 dB | 26.40 dBi | passes |

`tests/test_t2_001.py` now asserts that the best global rotation still leaves
more than 20° of quantization error, and separately that it still fails the
sidelobe requirement end-to-end. This class of degeneracy cannot silently return.

**Post-fix baseline** (Sonnet 5, three attempts): 3/3 pass at −15.96, −16.52 and
−17.02 dB against the −14 dB bar, so margins of 1.96, 2.52 and 3.02 dB. Only the
third beat the reference's −16.72 dB; the first two came in 0.76 and 0.20 dB
worse than it. Turns and estimated cost fell across attempts (42→27 turns,
$1.83→$0.93 API-equivalent; these runs were subscription-covered, not billed).

Note both figures above are read off the scoring grid, which flatters optimized
designs by 0.22–0.26 dB (see the grid-bias entry below). The reference's true
continuous value is −16.50 dB.

Verdict: the task is now *correct* but not *discriminating*, since a frontier model
clears it reliably. Do not tighten the threshold to manufacture difficulty; the
reference itself only reaches −16.72 dB, so a −18 dB bar risks being infeasible
rather than hard. Keep `t2-001` as the suite's floor and let discrimination come
from t2-002…t2-005 and Tier 3. This also means the current suite cannot yet rank
agents, which is an argument for building t2-004 (multi-scale) next.

Harness lessons from the same run:

- **Pin `--model`.** The first run silently defaulted to Haiku 4.5 (22 turns,
  $0.73, 200 s). Worse, the result JSON has no top-level `model` field and
  `modelUsage` lists two models, because Claude Code delegates cheap internal work
  to a small model, so naive parsing attributed a Sonnet run to Haiku. Runs are now
  attributed to whichever model produced the most output tokens.
- **Record the interpreter the agent actually uses.** The harness venv is not
  on the agent's PATH, so it reaches a different Python whose library versions
  could differ from the ones scoring it. `aedl run` now probes and records it.
- **Zero physics-library calls.** The agent made no calls to any instrumented
  pattern-evaluation entry point; it wrote its own array factor in numpy. The
  libraries were importable (verified), so this was a choice. Whether agents use
  the provided physics tooling at all is now a measurable property.

### Result integrity: workspace isolation is not enough

During attempt 2 the agent ran `find / -iname "*array_pattern*"`, hunting the
filesystem for the scoring code. Re-deriving the metric is legitimate; reading
`tasks/*/reference/solve.py` is not, and `--isolation tmpdir` does not prevent it
because the agent's shell reaches the whole filesystem.

`aedl run` now strips read permission from every `tasks/*/reference/` directory
while a run is in progress, then restores it. That is a deterrent rather than a
sandbox, since the agent runs as the same user and could reverse it.

**The three baseline attempts below predate that fix and cannot be certified
uncontaminated.** Treat the first pass rate as provisional and re-run under the
hardened harness, ideally in a container with no copy of this repository, before
quoting a number anywhere.

### Rules carried forward from t2-001

1. Always include an element pattern. A bare array factor over the full sphere has a
   back-hemisphere mirror lobe at peak level, which silently pins any sidelobe metric
   to 0 dB. (Hit this while probing; `t2-001` avoids it via the cos element.)
2. The evaluator enforces hardware constraints and applies element failures itself,
   never trust the submission to have done it.
3. Numerically verify feasibility *and* infeasibility-of-the-naive-approach before
   setting a threshold.

## Tier 3 — system (2 tasks)

`phased-array-systems` v0.9.0 supplies 68 metrics (EIRP, link margin, scan loss, prime
power, cost, MTBF, availability, ADC/beamformer load) plus `Requirement` /
`RequirementSet` / `VerificationReport` with pass/fail semantics. The evaluator is
mostly wiring, and cheap. The submission is an **architecture** (element count, spacing,
taper, phase bits, TX power, PA efficiency, noise figure, digitization level, subarray
size), not a weight vector.

### The central design problem

The library ships `optimize_design` (NSGA-II) and `generate_doe`. If the agent
optimizes against the same model that scores it, the task measures tool-calling
rather than engineering, since a competent agent just calls the optimizer. Two
mechanisms fix this:

**1. Held-out evaluation conditions.** The agent designs against a nominal scenario;
the evaluator scores over an envelope withheld from it: scan angles across the field
of regard, rain rate, temperature, element-failure seeds, and mission hour points.
The requirement is worst-case over the envelope. This turns the task from
optimization into design that must tolerate variation, and mirrors the
generalization gap the 2026 metasurface-agent work reported (in-distribution
success 74%, held-out families far worse).

**2. Cross-model verification.** Pattern-related claims get recomputed with
`phased-array-modeling`'s full pattern integration rather than the reduced-order
metric the agent optimized against. This is the "evaluator has veto power" principle
made concrete, and it is not hypothetical here:

> For a 32×32 array at 28 GHz with a Taylor −35 dB taper at broadside,
> `phased-array-systems` reports `sll_db = −14.03`. Direct full-pattern computation
> gives **−27.88 dB**. The reported value is also non-monotonic in taper depth
> (−25 dB taper → −17.5; −35 dB → −14.0; −45 dB → −14.7) and completely insensitive
> to `phase_bits`, which should raise sidelobes under coarse quantization.

**Root cause found** (read-only source audit, 2026-08-09) —
`phased_array_systems/models/antenna/metrics.py:62-98`, `compute_sidelobe_level`.
The main-lobe exclusion mask is wrong:

```python
bw = compute_beamwidth(pattern_db, angles_deg, -3.0)  # full -3 dB width
main_lobe_width_deg = bw * 2  # line 86
half_width = main_lobe_width_deg / 2  # == bw  — the /2 cancels the *2
mask = np.abs(angles_deg - peak_angle) > half_width
```

The mask therefore excludes only ±1×HPBW, but the first null of a 32-element
aperture sits at roughly 1.3–1.8×HPBW (further with heavier taper). The reported
"peak sidelobe" is a sample on the **main-lobe skirt**, not a sidelobe. There is
no null detection and no local-maximum detection. That also explains the
non-monotonicity: deeper taper widens the beam and steepens the skirt, so the
first grid sample clearing the mask lands at a different point on the skirt each
time. The underlying pattern is correct: for Taylor −35 dB the true local maxima
are at −35.24 dB (±6.75°), −35.34 dB (±9.5°) — only the extraction is broken.

Phase-bit insensitivity has two independent causes: scenarios default to
`scan_angle_deg = 0`, where all steering phases are zero and `quantize_phase` is a
no-op; and off-broadside, 3-bit quantization lobes near −25 dB sit far below the
≈−12.6 dB skirt reading, so the broken extraction hides them. Quantization *is*
accounted for in gain via `phase_quantization_loss_db`
(`models/antenna/errors.py:62-75`, Ruze), just not in SLL. Note the analytical
fallback path used when `phased_array` is absent (`adapter.py:337-345`) *does*
fold quantization into SLL, so the two code paths disagree by construction.

Fix direction: exclude out to the first null, or detect local maxima and drop the
main peak. `nbar` is a side issue — `taylor_taper_2d` hardcodes `nbar=4`, which is
adequate at −25 and −35 dB and costs 3.5 dB only at −45 dB (−41.5 achieved).

**Do not build a Tier-3 task that reads `sll_db` until this is fixed upstream.**

### The two tasks

- **t3-001 — 28 GHz SATCOM terminal.** Meet link margin at the field-of-regard edge in
  rain, under a prime-power ceiling and a unit-cost ceiling. Tests whether the agent
  finds the aperture-size / power-per-element / taper trade instead of brute-forcing
  element count.
- **t3-002 — X-band search radar.** Meet detection probability against a Swerling-1
  target with clutter and CFAR loss, within a frame-time constraint and an
  availability floor at end of mission. Reliability and search timing pull against
  aperture size in a way that pure link-budget reasoning misses.

## What I recommend against, and why

- **Tier 1 (EdgeFEM) in v0.1.** It is the layer that draws most directly on the PhD, which
  makes it tempting, but it is single-threaded with no true PML, prebuilt wheels for
  macOS arm64 only, and minutes per solve. That makes agent runs slow, CI impossible,
  and the time budget unpredictable. Ship the baseline on Tiers 2–3 first, then add
  Tier 1 with the harness and calibration method already proven.
- **Mutual coupling / scan blindness tasks.** `apply_mutual_coupling` and
  `scan_blindness_model` are theoretical models of unvalidated fidelity. A benchmark
  whose ground truth is an unvalidated model is indefensible — that is the one
  criticism that would sink the paper.
- **10–15 tasks.** The original plan's number does not survive contact with the
  calibration cost. Six well-calibrated tasks with a defensible baseline beats fifteen
  guesses, and the suite is designed to grow after publication.
- **Phase-2 architecture work (hierarchy agents, fidelity allocation, evaluator veto
  as a system).** Every one of these needs the baseline to measure against. Building
  them first produces exactly the unfalsifiable multi-agent claim the project exists
  to avoid.

## Harness follow-ups

Ordered by how much they affect whether a published number is trustworthy.

1. **Container isolation.** The only real fix for the integrity hole above. A
   thin Docker image with the libraries and no repo checkout, plus
   `--isolation container`, would make results defensible without the chmod
   deterrent.
2. **Transcript capture.** `--output-format json` returns only the final result
   object, so there is no record of which files the agent read. Switching the
   adapter to `stream-json` would give a full tool-call log, enabling an
   automatic `integrity: suspect` flag when a run touches `reference/`.
3. **Per-attempt seeds and variance.** Three attempts is enough to notice a
   degenerate task, not enough for a pass rate with error bars. Decide on a
   standard attempt count before publishing.
4. **Cost model for non-Claude adapters.** The `command` adapter reports no
   usage. If a second agent family is benchmarked, token accounting needs a
   provider-neutral path.

## Budget

| work | hours |
|------|------:|
| harness (`run`, cost accounting, `report`) | 8 |
| t2-002 … t2-005 (4 tasks × ~4 h) | 16 |
| upstream `sll_db` investigation + fix | 4 |
| t3-001, t3-002 (incl. held-out envelope + cross-check) | 12 |
| baseline runs + leaderboard | 6 |
| arXiv preprint + essay | 10 |
| **total** | **56** |

Roughly 11 weeks at 5 hrs/week. Cut t2-005 and t3-002 first if it slips; keep t2-004
(multi-scale) and t3-001 (held-out robustness) — those two carry the research claim.
