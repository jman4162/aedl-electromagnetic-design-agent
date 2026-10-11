# t4-001 reference bundle (synthetic)

`submission/` is a synthetic bundle. No design tool or agent produced it.
`build_bundle.py` assembles it from AEDL's own pieces:

- `architecture`: the t3-001 reference design, copied from the output of
  `tasks/t3-001-satcom-terminal-28ghz/reference/solve.py`;
- `element/design.yaml`: a copy of `tasks/t1-001-patch-28ghz-ro4350b/reference/design.yaml`;
- `board/patch.kicad_pcb`: antenna-cad 0.2's KiCad emitter run on that
  element spec (its UUIDs are name-derived, so the file is reproducible);
- `verdicts`: the t4-001 evaluator's own recomputation, with each number
  tagged by the SHA-256 of the architecture JSON or the element spec it came
  from, origin `derived`, fidelity `analytic`.

Because the verdicts come from the evaluator, this bundle shows that the
evaluator accepts a correct bundle and that every requirement is closable.
It does not show that any tool can produce such a bundle. A bundle generated
by the EdgeArray hero workflow (`edgearray workflow hero`, not yet available)
is planned to replace it; until then this one is the evaluator's known-good
input.

Regenerate (about 20 s, runs the t3-001 and t1-001 evaluators once each):

```bash
.venv/bin/python tasks/t4-001-requirements-to-board/reference/build_bundle.py
```
