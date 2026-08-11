# Run bundles

Three attempts by Claude Code (Sonnet 5) against `t2-001` on 2026-08-10, kept so the numbers
quoted in the top-level README can be checked by someone else. They are committed rather than
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
