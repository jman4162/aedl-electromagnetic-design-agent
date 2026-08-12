# Run bundles

Committed agent attempts, kept so the numbers quoted in the top-level README can be
checked by someone else. Three groups:

- **t2-001, 2026-08-10** (three bundles): the original calibration attempts.
- **t3-001, 2026-08-11** (six bundles): the library-vs-MCP composition measurement,
  described below.
- **t3-001, 2026-08-12** (two bundles): harness-instrumentation smoke runs after
  the Slice-3 observability work, described below.
- **t3-002, 2026-08-12** (one bundle): first agent attempt at the new X-band
  search radar task, described below.

## t3-002: first attempt (2026-08-12)

`20260812T173726Z…73aa112d` — Claude Code (Sonnet 5), MCP-attached, passed all
nine requirements on the first attempt ($1.03, 28 turns, 468 s). The design is
not the reference's: the agent chose a 64x32 *rectangular* aperture — a
narrower azimuth beam shrinks the clutter cell directly — and beat the
reference on unit cost ($486k vs $513k). The full instrumentation shows the
method: one `mcp__apab__system_evaluate` call plus its own pattern-sweep
scripts (234 aperture-model calls across 4 processes in `calls.jsonl`),
`integrity: clean`, APAB server spans in `server-trace.jsonl`. One attempt is
an existence proof that the task is solvable by an agent, not a pass rate.

## t3-001: instrumentation smoke runs (2026-08-12)

Both open integrity items from the composition measurement are closed, and these
two bundles are the evidence. `20260812T034040Z…371bb314` (transcript capture):
the stream-json adapter landed `transcript.jsonl` (714 events, 69 tool calls, 14
of them MCP) and the manifest's `integrity: clean` — but `calls.jsonl` was still
empty because the shim env paths were relative and resolved nowhere in processes
with a different cwd. `20260812T070655Z…8a38b7ac` (after the absolute-path fix):
`calls.jsonl` records 221 calls across 10 processes, including
`DefaultLinkEngine.evaluate_snapshot` from inside the opensatcom MCP server —
the call class that was invisible in every earlier bundle — and
`server-trace.jsonl` carries the APAB server's own tool spans. Both runs passed
all eight requirements.

That second bundle also shows something newly measurable: the agent used the MCP
tools for link physics (1 opensatcom + 8 APAB calls) *and* wrote its own pattern
scripts (107 aperture-model calls from Bash-spawned Pythons). Whether agents use
the provided physics tooling stopped being an inference from workspace shape.

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
