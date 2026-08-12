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
naive rounding −5.9 dB, dithered quantization −12 dB, both read under the
pre-fix fixed-radius metric. Under local-maximum detection naive rounding at
that geometry is −7.18 dB, still failing the −9 dB bar it faced. All of it was
superseded by the retarget to (27°, 10°) recorded in the calibration log.) No agent had
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
| greedy coordinate descent over the four states | −16.98 dB | 26.40 dBi | passes |

`tests/test_t2_001.py` now asserts that the best global rotation still leaves
more than 20° of quantization error, and separately that it still fails the
sidelobe requirement end-to-end. This class of degeneracy cannot silently return.

**Post-fix baseline** (Sonnet 5, three attempts): 3/3 pass against the −14 dB
bar. Under the metric as it now stands the three attempts reach −15.96, −16.52
and −17.59 dB, so margins of 1.96, 2.52 and 3.59 dB. None beats the reference's
−16.98 dB, though attempt 3 comes within 0.6 dB. Turns and estimated cost fell
across attempts (42→27 turns, $1.83→$0.93 API-equivalent; these runs were
subscription-covered, not billed).

At the time these ran, the metric reported −15.96, −16.52 and −17.02 against a
reference of −16.72, because it read the highest sample outside a fixed 8°
radius and that radius sits inside the main lobe. See the sidelobe-extraction
entry below.

Verdict: the task is now *correct* but not *discriminating*, since a frontier model
clears it reliably. Do not tighten the threshold to manufacture difficulty; the
reference itself only reaches −16.98 dB, so a −18 dB bar risks being infeasible
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

### 2026-08-10 — the sidelobe metric was reading the main-lobe skirt

`t2-001` scored the highest pattern sample outside a fixed 8° radius of the
target. For a 16×16 array steered to 27° that radius sits inside the main lobe.
A design whose true sidelobes fall below its own skirt level at 8° therefore got
scored on the skirt: the reported point sat at exactly 8.00° from the target and
climbed uphill to the main peak under refinement. Two of five designs measured
did exactly that.

| design | 8° radius | local maxima | continuous refinement |
|---|---|---|---|
| direct 2-bit rounding | −10.66 | −10.66 | −10.67 |
| reference solution | −16.72 | −16.98 | −16.98 |
| agent `d68c3564` | −15.96 | −15.96 | −15.96 |
| agent `4eb80fae` | −16.52 | −16.52 | −16.52 |
| agent `85d86723` | −17.02 | −17.59 | −17.60 |

The offset is the small part. The metric *saturated*: once a design pushed its
sidelobes below the skirt, further suppression stopped registering, which is
precisely the discrimination the suite exists to provide.

Fix applied: `peak_sidelobe_level_db` is the second-highest local maximum of the
pattern. `exclusion_radius_deg` is still honoured when a task sets one, and
`t2-001` no longer does.

Two things worth carrying forward. First, this is the same defect that a source
audit found in `phased-array-systems` on 2026-08-09, in a different codebase,
written at a different time, for a different purpose. A main-lobe exclusion that
does not reach the first null is apparently the default mistake in this
operation, so any new metric that separates a main beam from its sidelobes gets
checked against local-maximum detection before it is trusted.

Second, the earlier grid-sampling diagnosis was wrong, and wrong in an
instructive way. The continuous measurement used to establish it was itself
clamped to the same 8° radius, so it hill-climbed the skirt and stopped on the
boundary, roughly agreeing with the metric and appearing to confirm a small
sampling bias. A check that shares an assumption with the thing it checks will
agree with it. With the metric fixed and the refinement unconstrained, the
361×721 reading matches continuous refinement to within 0.01 dB on every design,
so grid sampling was never the problem. `scripts/verify_sidelobe_metric.py`
reproduces the whole comparison.

### Rules carried forward from t2-001

1. Always include an element pattern. A bare array factor over the full sphere has a
   back-hemisphere mirror lobe at peak level, which silently pins any sidelobe metric
   to 0 dB. (Hit this while probing; `t2-001` avoids it via the cos element.)
2. The evaluator enforces hardware constraints and applies element failures itself,
   never trust the submission to have done it.
3. Numerically verify feasibility *and* infeasibility-of-the-naive-approach before
   setting a threshold.
4. Separate a main beam from its sidelobes by detecting local maxima, never by a
   hand-set angular radius. The main lobe widens with scan angle and with taper,
   so a radius that is safe for one design sits inside the main lobe for another.
5. A verification path must not inherit the assumption it is verifying.

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
generalization gap the 2026 metasurface-agent work reported. (Corrected against
the abstract of arXiv:2604.01480: skill evolution raised same-type success from
38% to 74%; of the two held-out families, one held near ceiling at 0.90 and the
other started at 0.20 before skill evolution recovered it to 0.90. The earlier
wording here, "held-out families far worse", overstated it. The lesson stands
in a sharper form: initial generalization can be weak, and an evaluator only
measures the cases it has.)

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
The main-lobe exclusion mask was too narrow:

```python
bw = compute_beamwidth(pattern_db, angles_deg, -3.0)  # full -3 dB width
main_lobe_width_deg = bw * 2  # total exclusion window
half_width = main_lobe_width_deg / 2  # == bw, so the mask spans ±1×HPBW
mask = np.abs(angles_deg - peak_angle) > half_width
```

An earlier version of this entry called the `/2` a cancellation of the `*2`. That
was wrong: the arithmetic is deliberate and matches its comment, a total exclusion
window of two beamwidths. The defect is that ±1×HPBW does not reach the first
null, which for a 32-element aperture sits at roughly 1.3–1.8×HPBW and further as
the taper deepens. The reported "peak sidelobe" was a sample on the **main-lobe
skirt**. There was no null detection and no local-maximum detection. That also
explains the non-monotonicity: deeper taper widens the beam and steepens the
skirt, so the first grid sample clearing the mask lands at a different point on
the skirt each time. The underlying pattern was correct: for Taylor −35 dB the
true local maxima are at −35.24 dB (±6.75°), −35.34 dB (±9.5°) — only the
extraction was broken.

Phase-bit insensitivity had two independent causes: scenarios default to
`scan_angle_deg = 0`, where all steering phases are zero and `quantize_phase` is a
no-op, which is correct behavior; and off-broadside, 3-bit quantization lobes near
−25 dB sat far below the ≈−12.6 dB skirt reading, so the broken extraction hid
them. Quantization *is* accounted for in gain via `phase_quantization_loss_db`
(`models/antenna/errors.py:62-75`, Ruze). The analytical fallback path used when
`phased_array` is absent (`adapter.py:337-345`) folds quantization into SLL, so
the two code paths disagreed by construction.

**Fixed upstream in phased-array-systems 0.10.0** (2026-08-10). The main lobe is
now excluded out to its first null on each side; `main_lobe_width_deg` still
forces a fixed window. Measured after the fix:

| case | before | after |
|---|---|---|
| 32×32 Taylor −35 dB, broadside | −14.0 dB | −35.24 dB |
| golden DBF case (`tests/data/golden_dbf_case.json`) | −16.56 dB | −30.39 dB |
| taper depth −25 / −35 / −45 dB | −17.5 / −14.0 / −14.7 | −25.3 / −35.1 / −41.5 |
| 32×32 Taylor −35 dB at 45°, 2 / 3 / 6 bits / ideal | flat | −4.96 / −16.14 / −28.64 / −32.26 dB |

The two code paths now agree to 0.24 dB on the broadside case. `nbar` remains a
side issue: `taylor_taper_2d` hardcodes `nbar=4`, adequate at −25 and −35 dB and
costing 3.5 dB only at −45 dB (−41.5 achieved).

Tier-3 tasks may now read `sll_db`, against `phased-array-systems>=0.10`. Pin that
floor before building one: 0.9.x reports a different number for the same design.

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
2. **Transcript capture.** Done (2026-08-11): the adapter runs `stream-json`,
   `transcript.jsonl` lands in every bundle, and the manifest carries an
   `integrity` field exactly as described below.
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

## Ecosystem slices (recorded 2026-08-11)

Slice 1 shipped as phased-array-systems 0.11.0 (T/R modules, cascaded
P1dB/compression/SNDR, DAC path, load-pull against EdgeFEM active-impedance
scans, cited technology catalog). Slices 2 and 3 are specified here so they
can be executed when prioritized, and so their absence is not mistaken for
completeness.

### Slice 2 — measurement-in-the-loop v1

No repo in the stack has a concept of measured data today: APAB's importers
parse then discard arrays, three incompatible pattern-CSV formats exist, and
EdgeFEM's docs state no comparison against commercial solvers or measurement
exists. The slice, in dependency order:

1. **Measurement artifact contract.** A provenance block (instrument, date,
   calibration state, uncertainty, operator) required on any measured
   dataset. Touchstone via scikit-rf's reader wherever *measured* data is
   consumed (antenna-cad already depends on scikit-rf and uses it nowhere;
   opensatcom's hand-rolled weak Touchstone reader gets retired or fixed).
   One blessed complex pattern-CSV format: EdgeFEM's NTF full-grid columns,
   with a documented mapping from the two-cut format. phased-array-systems'
   `LoadPullTable.from_csv` schema (Gamma_real, Gamma_imag, pout_drop_db,
   pae_drop_pct, ampm_deg) is the load-pull entry point for bench data.
2. **APAB**: `emtool_import_results` / `io_import_touchstone` stop
   discarding arrays — parsed data persists into the run bundle; new
   `compare_sim_measured` tool producing an RMSE / max-deviation report
   artifact.
3. **antenna-cad**: a `MeasuredSolver` implementing the existing `EMSolver`
   protocol (reads the artifact into `SimulationResult`'s xarray shape) so
   measured boards flow through the existing verify/report path; the
   hardcoded acceptance numbers at `report.py:161` become config.
4. **opensatcom**: wire `load_array_package` into config/CLI (today only
   the never-produced `.npz` path is wired).
5. **Fixture**: one real bench `.s2p` committed as the first measured
   fixture; until a bench exists, contract tests use a synthetic file whose
   provenance block marks it synthetic.

Out of scope until hardware exists: pyvisa/SCPI instrument control.
Unlocked by this slice: finite-array full-wave validation (EdgeFEM vs
measurement), and fabricating an antenna-cad board as the first real
fixture — the loop the ecosystem post named as its most honest gap.

### Slice 3 — observability closure (done 2026-08-11)

The honest remainder of the "agent operating system" question (answered no
on the t3-001 gate: library-only 0/3 vs MCP-attached 2/3; no coordination
layer earned). The pieces exist and are disconnected:

1. **APAB**: `apab mcp serve` calls `init_observability` (env-gated, as
   elsewhere) so server-side tool spans exist at all — today the OTel stack
   is real but never initialized on the MCP path, so agent-driven runs emit
   no spans and no bundles. W3C `traceparent` accepted via env for
   cross-process correlation.
2. **AEDL claude adapter**: an `--output-format stream-json` variant
   capturing the tool-call transcript into the run bundle. This closes both
   open items from the t3-001 measurement at once: no per-call transcript
   under `json`, and `calls.jsonl` recording zero instrumented calls even
   in the MCP arm (the server processes did not inherit the shim; with a
   transcript, tool usage stops being inferred from workspace shape).
3. **Strands**: one real, un-mocked integration test behind a marker (the
   `strands` extra grows `[otel]`); example 07's telemetry verified against
   the Jaeger lab in a documented manual check.

Explicitly not building: any new orchestration package. The LangGraph
pipeline remains the deterministic path; agents remain replaceable shells.

### Registered, not scheduled

The empty HFSS/CST adapter registry in APAB (seam exists, no implementors);
metasurface-py island + stale 0.3.0 release; t2-002…005 above.
