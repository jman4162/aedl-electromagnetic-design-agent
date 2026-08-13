# Run bundles

Committed agent attempts, kept so the numbers quoted in the top-level README can be
checked by someone else. Four groups:

- **t2-001, 2026-08-10** (three bundles): the original calibration attempts.
- **t3-001, 2026-08-11** (six bundles): the library-vs-MCP composition measurement,
  described below.
- **t3-001, 2026-08-12** (two bundles): harness-instrumentation smoke runs after
  the Slice-3 observability work, described below.
- **t3-002, 2026-08-12** (six bundles): the composition measurement repeated on
  the X-band search radar task, described below.

## t3-002: the composition measurement, second task (2026-08-12)

Six attempts on the X-band search radar task, Claude Code (Sonnet 5), 30-minute
timeout: three with the physics libraries importable and no tools attached, three
with the opensatcom and APAB MCP servers attached (`--mcp-config`, config sha256
in each manifest). Same protocol as the t3-001 measurement above.

| series | attempt | outcome | failed on | design | est. cost | wall |
|---|---|---|---|---|---|---|
| MCP | 1 | **pass** | - | 64x32, class B, 6.0 W, 32 pulses | $1.03 | 468 s |
| MCP | 2 | **pass** | - | 40x40, class C, 7.0 W, 64 pulses | $2.14 | 1172 s |
| MCP | 3 | **pass** | - | 64x32, class B, 8.0 W, 32 pulses | $2.46 | 648 s |
| library | 1 | fail | frame-time (occupancy 1.07 > 1.0) | 48x48, class B, 9.0 W, 36 pulses | $4.89 | 1295 s |
| library | 2 | **pass** | - | 48x48, class B, 4.0 W, 56 pulses | $4.66 | 1612 s |
| library | 3 | timeout | still sweeping at 1800 s | no submission | $4.43 | 1800 s |

Three things the bundles show that the pass rate does not.

**The MCP arm barely used the MCP tools.** Tool-call transcripts record 1, 0, and
1 MCP calls across the three attempts. Whatever separates the arms here, it is not
tool-call volume. One untested hypothesis: the tool schemas are themselves a menu
of the variables the system model prices (Swerling case, CFAR type, clutter,
search extents, PRF), visible in the system prompt whether or not the agent calls
anything. Three attempts per arm cannot distinguish that from variance.

**The cost gap is real and shows up in physics calls.** Instrumented model calls
per attempt: MCP arm 238, 1714, 1009; library arm 3062, 8187, 30602. The library
attempts wrote and ran their own parameter sweeps, which is where the 2.5x cost
and 2x wall time went. The timed-out attempt was still sweeping at 1800 s with
30,602 logged pattern computations.

**Aperture shape split the arms.** Both 64x32 designs came from the MCP arm and
are the two cheapest passing designs on record ($482,560 and $486,400), below the
reference's $512,928. Both library attempts that submitted anything chose 48x48
squares. The reference solver enumerated square apertures only, so the rectangular
option was absent from the search space it defines, and the agents that found it
found it on their own.

Caveats. Three attempts per arm is directional and far too few for significance;
one model family; the first MCP attempt ran hours before the other five against
an identical task and harness. Costs are API-equivalent estimates
(subscription-covered). Unlike the t3-001 bundles, every attempt here carries a
tool-call transcript, an integrity flag (all six clean), and per-process physics
call counts.

## t3-001: the composition measurement (2026-08-11)

Claude Code (Sonnet 5), 30-minute timeout, hardened harness (reference hiding
active, post-fix sidelobe metric). Two series of three attempts: library-only
(the agent has the pip packages) and MCP-attached (`--mcp-config` with the
opensatcom and APAB servers, config sha256 in each manifest).

| series | attempt | outcome | failed on | est. cost | wall |
|---|---|---|---|---|---|
| library | 1 | fail | prime-power (521 W / 450) with 3.55 dB unneeded margin | $3.20 | 955 s |
| library | 2 | no_submission | ran out of time mid-verification | $2.53 | 882 s |
| library | 3 | timeout | still sweeping designs at 1800 s | $4.24 | 1800 s |
| MCP | 1 | **pass** | — | $4.93 | 1062 s |
| MCP | 2 | **pass** | — | $5.00 | 1374 s |
| MCP | 3 | fail | link-margin (−1.13 dB worst-case; tuned too close to the edge) | $3.10 | 998 s |

What the workspaces show: the library-only agents wrote and debugged their own
physics scripts (`verify.py`, `design.py`, sweep outputs are still in their
workspaces) and two of three ran out of road doing it; every MCP-attached
attempt left only `architecture.yaml`.

Caveats that bound what this supports. Three attempts per arm is enough to see
a direction and far too few for significance. `--output-format json` keeps no
tool-call transcript, so MCP tool usage is inferred from the result text and
workspace shape, not logged per call. And `calls.jsonl` recorded zero
instrumented physics calls in every bundle, including the MCP arm, where the
server processes were expected to inherit the shim; that instrumentation gap
is still open **for these six bundles**. Costs are API-equivalent estimates
(subscription-covered). Both gaps were closed on 2026-08-12 (see the
instrumentation smoke runs above); runs from that date onward carry a
transcript, an integrity flag, and per-process call logs.

## t2-001 attempts (2026-08-10)

Three attempts by Claude Code (Sonnet 5) against `t2-001`. They are committed rather than
regenerated because they cannot be regenerated: the agent is nondeterministic, so `aedl run`
produces different designs, and the per-attempt figures refer to these specific submissions.

Each bundle holds the provenance manifest, the scoring result, the agent's stdout, the
instrumentation shim, and the workspace the agent worked in, including whatever scripts it
wrote and the `submission.npz` it left behind.

## Read `result.json` with the date in mind

`result.json` records what the evaluator said **at run time**, on 2026-08-10. That was before
the sidelobe metric was fixed, so its `peak_sidelobe_level_db` is the pre-fix value: the
highest pattern sample outside a fixed 8 degree radius of the target, which for this geometry
sits inside the main lobe. Attempt 3 is the one where that matters, recorded as −17.02 dB where
the design actually achieves −17.59 dB.

To score these submissions under the current evaluator:

```bash
python scripts/verify_sidelobe_metric.py
```

That prints, for every design, what the evaluator reports now, what an independent continuous
refinement finds, and what the old fixed radius would have said.

## What these attempts do not establish

Three attempts on one task, all passing, from one model family. They also predate the fix that
strips read permission from `tasks/*/reference/` during a run, so they cannot be certified
uncontaminated. One attempt ran `find / -iname "*array_pattern*"` across the filesystem, and
another's transcript records re-running the evaluator source. Re-deriving the metric is
permitted; reading the worked solution is not, and the run format used here keeps no tool-call
log that could settle which happened.

All three manifests record `calls.total_calls: 0`, meaning no call to an instrumented physics
entry point was seen. That is in tension with the claim to have re-run the evaluator, since the
evaluator calls into `phased-array-modeling`, and it is unresolved. Read run-level physics-call
counts as best-effort.
