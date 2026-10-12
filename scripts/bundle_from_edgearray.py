"""Convert an EdgeArray hero-workflow run into a t4-001 ``submission/`` bundle.

This script reads the hero's on-disk outputs and imports nothing from
EdgeArray, so AEDL stays decoupled from it. The inputs are the files under
``DATA_DIR/projects/<id>/``: the DesignState head
(``design/head.json`` -> ``design/revisions/<rev>.json``) and the artifact
directories ``artifacts/<aa>/<hash>/`` (``manifest.json``, ``payload.json``
and the files the manifest lists).

Usage::

    python scripts/bundle_from_edgearray.py --data-dir DATA_DIR --project ID --out DIR

The bundle is built in a staging directory and moved to ``DIR`` only when the
conversion succeeds; any failure prints ``error: ...``, exits 1 and leaves
``DIR`` as it was. Symlinked artifact directories and files are refused.

Field mappings
--------------

``architecture`` (t3-001 shape) from the revision's ``architecture`` node,
an ``architecture.case`` payload:

==================================  =====================================
bundle                              hero payload
==================================  =====================================
array.nx, ny                        ``nx``, ``ny``
array.dx_lambda, dy_lambda          ``dx_lambda``, ``dy_lambda``
array.taper_type                    ``taper``
array.taper_sll_db                  ``taper_sll_db``
rf.tx_power_w_per_elem              ``study.rf.tx_power_w_per_elem``
rf.pa_class                         the t4-001 parts-table class whose
                                    ``pa_efficiency`` equals
                                    ``study.rf.pa_efficiency``
digital.subarray_nx, subarray_ny    rule R6-tile ``inputs.subarray_n``
==================================  =====================================

The hero does not choose the phase-shifter resolution or the digitization,
which t3-001's architecture requires. Its PAS study models ideal phase
shifters and uses PAS's default DigitalConfig. The converter fills these
in and says so in ``conversion.json`` (``filled_in``) and on stderr:

- ``array.phase_bits`` = 6, the finest priced resolution (the hero's
  pattern numbers assume no phase quantization);
- ``digital.digitization_level`` = ``element`` and ``digital.adc_enob`` = 12.0,
  PAS's DigitalConfig defaults, which the hero's study ran with.

``element_spec`` (``element/design.yaml``) from the ``element`` node, an
``element.design`` payload: ``design_spec.name`` -> ``design.name``,
``design_spec.problem`` -> ``requirements``, ``design_spec.element_type`` ->
``element.type``, ``design_spec.acceptance`` -> ``acceptance``. The hero
puts a unicode ohm sign in the impedance string; it is written as ``ohm``.

``board`` (``board/<file>.kicad_pcb``) is the file named by the ``board``
node's ``board.layout`` payload (``kicad_pcb_file``), copied byte for byte.
``board_provenance`` has the SHA-256 of that file, the capability
``<producer> via <engine_version>`` from the artifact manifest, origin
``derived`` and fidelity ``analytic``.

``verdicts`` from the ``verification`` node, a ``verification.report``
payload, one entry per ``verdicts[]`` item:

==================================  =====================================
bundle                              hero verdict
==================================  =====================================
status                              ``pass``/``fail`` as is;
                                    ``unverified`` -> ``unknown``
actual                              ``actual`` (null when unverified)
unit                                ``unit``; an empty string -> null
provenance.artifact_hash            the verification report artifact id
                                    (the artifact that records the number)
provenance.capability               ``engine`` (for example
                                    ``edgearray-array-envelope@1.0.0``)
provenance.origin                   ``assumed`` when a hero warning names
                                    the verdict as checked "on assumed
                                    inputs", else ``derived``
provenance.fidelity                 ``fidelity`` (the hero's value)
==================================  =====================================

Origin rule. The hero's report carries warnings of the form
``REQ-A, REQ-B: checked by ENGINE on assumed inputs (key, key)``. A verdict
named in one of them was computed from inputs the hero assumed, not from
inputs the requirements or the design supply, so its origin is ``assumed``.
The rest are ``derived``. For the committed hero run that is all four
verified verdicts: REQ-GAIN-XCHECK, REQ-GRATING and REQ-SLL (assumed
element pattern exponent) and REQ-POWER (assumed transmit power, PA
efficiency, noise figure and others). ``conversion.json`` lists them under
``assumed_verdicts``. Unverified verdicts carry no provenance.

A warning that mentions "assumed" but does not match the form above is not
ignored. It is listed under ``unmatched_assumption_warnings`` in
``conversion.json`` and printed on stderr, and every verdict whose id appears
in its text is labelled ``assumed`` as well.

A verdict that is unverified or has no ``actual``/``engine`` gets
``provenance: null``. The hero's model has no element requirements, so
ELEM-S11-IN-BAND and ELEM-FOOTPRINT are reported as ``unknown`` with a null
actual: the hero did not judge them. (A report that does carry a verdict for
one of them keeps it; only absent ones are filled in.) The report's warnings, and which
numbers rest on assumed inputs, are copied into ``conversion.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

BUNDLE_SCHEMA = "aedl.t4-001.bundle/1"
TASK_YAML = (
    Path(__file__).resolve().parents[1] / "tasks" / "t4-001-requirements-to-board" / "task.yaml"
)

#: Finest priced phase-shifter resolution (t4-001 parts table).
FILLED_PHASE_BITS = 6
#: PAS DigitalConfig defaults, which the hero's study ran with.
FILLED_DIGITIZATION_LEVEL = "element"
FILLED_ADC_ENOB = 12.0

ELEMENT_REQUIREMENT_IDS = ("ELEM-FOOTPRINT", "ELEM-S11-IN-BAND")
#: Hero report warning: "REQ-A, REQ-B: checked by ENGINE on assumed inputs (...)".
_ASSUMED_WARNING = re.compile(
    r"^(?P<ids>[A-Z][A-Z0-9-]*(?:, [A-Z][A-Z0-9-]*)*): checked by .+ on assumed inputs"
)
_ARTIFACT_ID = re.compile(r"[0-9a-f]{64}")


class ConversionError(RuntimeError):
    """The hero output is missing something the bundle needs."""


def _load_json(path: Path) -> Any:
    if path.is_symlink():
        raise ConversionError(f"refusing symlink {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"cannot read {path}: {exc}") from exc


def _object(value: Any, what: str) -> dict[str, Any]:
    """``value`` if it is a JSON object, else a ConversionError naming ``what``."""
    if not isinstance(value, dict):
        raise ConversionError(f"{what} is {type(value).__name__}, expected an object")
    return value


def _load_object(path: Path) -> dict[str, Any]:
    return _object(_load_json(path), str(path))


def project_dir(data_dir: Path, project: str) -> Path:
    path = data_dir / "projects" / project
    if not path.is_dir():
        raise ConversionError(f"no project directory {path}")
    return path


def head_revision(proj: Path) -> dict[str, Any]:
    rev = _load_object(proj / "design" / "head.json").get("rev")
    if not isinstance(rev, str) or not _ARTIFACT_ID.fullmatch(rev):
        raise ConversionError(f"head.json rev {rev!r} is not a 64-hex id")
    return _load_object(proj / "design" / "revisions" / f"{rev}.json")


def artifact_dir(proj: Path, artifact_id: str) -> Path:
    if not _ARTIFACT_ID.fullmatch(artifact_id):
        raise ConversionError(f"artifact id {artifact_id!r} is not a 64-hex id")
    path = proj / "artifacts" / artifact_id[:2] / artifact_id
    if (
        (proj / "artifacts").is_symlink()
        or path.parent.is_symlink()
        or path.is_symlink()
        or not path.is_dir()
    ):
        raise ConversionError(f"artifact {artifact_id} not found under {proj / 'artifacts'}")
    return path


def node_artifact(
    proj: Path, revision: dict[str, Any], node: str, kind: str
) -> tuple[str, Path, dict[str, Any], dict[str, Any]]:
    """(artifact id, directory, manifest, payload) of a revision node of the given kind."""
    nodes = _object(revision.get("nodes", {}), "revision nodes")
    if node not in nodes:
        raise ConversionError(f"revision has no {node!r} node (nodes: {sorted(nodes)})")
    artifact_id = str(_object(nodes[node], f"revision node {node!r}").get("artifact"))
    directory = artifact_dir(proj, artifact_id)
    manifest = _load_object(directory / "manifest.json")
    payload = _load_object(directory / "payload.json")
    if manifest.get("kind") != kind or payload.get("kind") != kind:
        raise ConversionError(
            f"node {node!r} is {manifest.get('kind')!r}/{payload.get('kind')!r}, expected {kind!r}"
        )
    return artifact_id, directory, manifest, payload


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def pa_classes(task_yaml: Path) -> dict[str, float]:
    """PA class -> efficiency from the t4-001 parts table."""
    doc = yaml.safe_load(task_yaml.read_text())
    classes = doc["context"]["system"]["parts"]["pa_classes"]
    return {str(name): float(entry["pa_efficiency"]) for name, entry in classes.items()}


def convert_architecture(
    payload: dict[str, Any], classes: dict[str, float]
) -> tuple[dict[str, Any], list[str]]:
    """t3-001 architecture from an architecture.case payload, and what was filled in."""
    study_rf = payload["study"]["rf"]
    efficiency = float(study_rf["pa_efficiency"])
    matches = [name for name, eff in sorted(classes.items()) if abs(eff - efficiency) < 1e-9]
    if len(matches) != 1:
        raise ConversionError(
            f"hero pa_efficiency {efficiency} matches PA classes {matches} in {classes}"
        )
    tile = [r for r in payload["rules"] if r["rule"] == "R6-tile"]
    if len(tile) != 1:
        raise ConversionError("architecture.case has no single R6-tile rule")
    subarray = int(tile[0]["inputs"]["subarray_n"])
    architecture = {
        "array": {
            "nx": int(payload["nx"]),
            "ny": int(payload["ny"]),
            "dx_lambda": float(payload["dx_lambda"]),
            "dy_lambda": float(payload["dy_lambda"]),
            "taper_type": str(payload["taper"]),
            "taper_sll_db": float(payload["taper_sll_db"]),
            "phase_bits": FILLED_PHASE_BITS,
        },
        "rf": {
            "tx_power_w_per_elem": float(study_rf["tx_power_w_per_elem"]),
            "pa_class": matches[0],
        },
        "digital": {
            "digitization_level": FILLED_DIGITIZATION_LEVEL,
            "adc_enob": FILLED_ADC_ENOB,
            "subarray_nx": subarray,
            "subarray_ny": subarray,
        },
    }
    filled = [
        f"array.phase_bits={FILLED_PHASE_BITS}: the hero models ideal phase shifters",
        f"digital.digitization_level={FILLED_DIGITIZATION_LEVEL} and "
        f"digital.adc_enob={FILLED_ADC_ENOB}: PAS DigitalConfig defaults, used by the hero's study",
    ]
    return architecture, filled


def element_yaml(payload: dict[str, Any]) -> str:
    """DesignSpec YAML text from an element.design payload."""
    spec = _object(payload.get("design_spec"), "element payload design_spec")
    if spec.get("array"):
        raise ConversionError("element design has an array section; t4-001 takes one element")
    raw_problem = _object(spec.get("problem"), "element design_spec problem")
    problem = {k: str(v).replace("Ω", "ohm") for k, v in raw_problem.items()}
    doc: dict[str, Any] = {
        "design": {"name": spec["name"]},
        "requirements": problem,
        "element": {"type": spec["element_type"]},
    }
    if spec.get("acceptance"):
        doc["acceptance"] = spec["acceptance"]
    return yaml.safe_dump(doc, sort_keys=False)


def _names(text: str, verdict_id: str) -> bool:
    """Whether ``text`` contains ``verdict_id`` as a whole id, not as part of a longer one."""
    return re.search(rf"(?<![\w-]){re.escape(verdict_id)}(?![\w-])", text) is not None


def assumed_verdict_ids(warnings: list[Any], verdict_ids: list[str]) -> tuple[set[str], list[str]]:
    """Verdict ids to label ``assumed``, and the warnings the rule did not parse.

    A warning matching ``_ASSUMED_WARNING`` names its ids exactly. A warning
    that mentions "assumed" but does not match (the hero reworded it) is
    returned in the second list, and every verdict whose id appears in its text
    is labelled ``assumed`` too, so a rewording cannot turn an assumption into
    ``derived`` silently. If no verdict id appears in it, the warning is only
    recorded.
    """
    ids: set[str] = set()
    unmatched: list[str] = []
    for warning in warnings:
        text = str(warning)
        match = _ASSUMED_WARNING.match(text)
        if match:
            ids.update(match.group("ids").split(", "))
        elif "assum" in text.lower():
            unmatched.append(text)
            ids.update(vid for vid in verdict_ids if _names(text, vid))
    return ids, unmatched


def _verdict(
    raw: dict[str, Any],
    report_id: str,
    vocab_fidelities: tuple[str, ...],
    assumed: set[str],
) -> dict[str, Any]:
    status = {"pass": "pass", "fail": "fail", "unverified": "unknown"}.get(str(raw["status"]))
    if status is None:
        raise ConversionError(f"verdict {raw['id']}: unmapped status {raw['status']!r}")
    actual = raw.get("actual")
    unit = raw.get("unit") or None
    provenance: dict[str, str] | None = None
    if status != "unknown" or actual is not None:
        fidelity = raw.get("fidelity")
        engine = raw.get("engine")
        if actual is None or not engine or fidelity not in vocab_fidelities:
            raise ConversionError(
                f"verdict {raw['id']}: {status} without actual/engine/fidelity "
                f"({actual!r}, {engine!r}, {fidelity!r})"
            )
        provenance = {
            "artifact_hash": "sha256:" + report_id,
            "capability": str(engine),
            "origin": "assumed" if str(raw["id"]) in assumed else "derived",
            "fidelity": str(fidelity),
        }
    return {"status": status, "actual": actual, "unit": unit, "provenance": provenance}


def convert(
    data_dir: Path, project: str, out_dir: Path, *, task_yaml: Path = TASK_YAML
) -> dict[str, Any]:
    """Write the bundle into ``out_dir`` and return it.

    The files are built in a staging directory next to ``out_dir`` and moved
    into place only when the whole conversion has succeeded, so a failure
    leaves no partial output. If ``out_dir`` already exists, the staged files
    are copied over it; if the conversion fails it is not touched.
    """
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}.staging-", dir=out_dir.parent))
    try:
        bundle = _convert_into(data_dir, project, staging, task_yaml)
        if out_dir.exists():
            shutil.copytree(staging, out_dir, dirs_exist_ok=True)
        else:
            staging.rename(out_dir)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return bundle


def _convert_into(data_dir: Path, project: str, out_dir: Path, task_yaml: Path) -> dict[str, Any]:
    proj = project_dir(data_dir, project)
    revision = head_revision(proj)
    classes = pa_classes(task_yaml)
    vocab_fidelities = tuple(
        yaml.safe_load(task_yaml.read_text())["context"]["provenance"]["fidelities"]
    )

    _, _, _, arch_payload = node_artifact(proj, revision, "architecture", "architecture.case")
    architecture, filled = convert_architecture(arch_payload, classes)

    _, _, _, element_payload = node_artifact(proj, revision, "element", "element.design")
    element_text = element_yaml(element_payload)

    board_id, board_dir, board_manifest, board_payload = node_artifact(
        proj, revision, "board", "board.layout"
    )
    board_name = str(board_payload["kicad_pcb_file"])
    board_src = board_dir / board_name
    if (
        Path(board_name).name != board_name
        or board_name not in board_manifest.get("files", [])
        or board_src.is_symlink()
        or not board_src.is_file()
    ):
        raise ConversionError(f"board file {board_name!r} is not in artifact {board_id}")
    if board_payload["design_hash"] != element_payload["ir_hash"]:
        raise ConversionError("board.design_hash differs from the element ir_hash")

    report_id, _, _, report = node_artifact(proj, revision, "verification", "verification.report")

    (out_dir / "element").mkdir(parents=True, exist_ok=True)
    (out_dir / "board").mkdir(parents=True, exist_ok=True)
    element_rel = "element/design.yaml"
    board_rel = f"board/{board_name}"
    (out_dir / element_rel).write_text(element_text)
    shutil.copyfile(board_src, out_dir / board_rel)
    board_bytes = (out_dir / board_rel).read_bytes()

    raw_verdicts = report.get("verdicts")
    if not isinstance(raw_verdicts, list) or not all(isinstance(v, dict) for v in raw_verdicts):
        raise ConversionError("verification report verdicts is not a list of objects")
    raw_warnings = report.get("warnings", [])
    if not isinstance(raw_warnings, list):
        raise ConversionError("verification report warnings is not a list")
    assumed, unmatched = assumed_verdict_ids(raw_warnings, [str(v["id"]) for v in raw_verdicts])
    verdicts: dict[str, Any] = {}
    for raw in raw_verdicts:
        verdicts[str(raw["id"])] = _verdict(raw, report_id, vocab_fidelities, assumed)
    # The hero's model has no element requirements today. If a report does
    # judge one, its verdict stands; only the omitted ones are filled in.
    filled_elements = [rid for rid in ELEMENT_REQUIREMENT_IDS if rid not in verdicts]
    for rid in filled_elements:
        verdicts[rid] = {"status": "unknown", "actual": None, "unit": None, "provenance": None}
    verdicts = dict(sorted(verdicts.items()))

    bundle = {
        "schema": BUNDLE_SCHEMA,
        "architecture": architecture,
        "element_spec": element_rel,
        "board": board_rel,
        "board_provenance": {
            "artifact_hash": _sha256(board_bytes),
            "capability": f"{board_manifest['producer']} via {board_manifest['engine_version']}",
            "origin": "derived",
            "fidelity": "analytic",
        },
        "verdicts": verdicts,
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n")

    record = {
        "source_revision": revision["rev"],
        "nodes": {name: node["artifact"] for name, node in sorted(revision["nodes"].items())},
        "verification_report": report_id,
        "board_artifact": board_id,
        "producer": board_manifest["producer"],
        "engine_versions": {
            "architecture": _load_json(
                artifact_dir(proj, str(revision["nodes"]["architecture"]["artifact"]))
                / "manifest.json"
            )["engine_version"],
            "board": board_manifest["engine_version"],
            "verification": _load_json(artifact_dir(proj, report_id) / "manifest.json")[
                "engine_version"
            ],
        },
        "filled_in": filled,
        "assumed_verdicts": sorted(
            rid for rid, v in verdicts.items() if v["provenance"] and rid in assumed
        ),
        "unmatched_assumption_warnings": unmatched,
        "element_requirements": (
            f"{', '.join(filled_elements)}: "
            if len(filled_elements) < len(ELEMENT_REQUIREMENT_IDS)
            else ""
        )
        + "not in the hero's model; reported as unknown with a null actual",
        "hero_warnings": list(raw_warnings),
    }
    (out_dir / "conversion.json").write_text(json.dumps(record, indent=2) + "\n")
    return bundle


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out", type=Path, required=True, help="submission directory to write")
    args = parser.parse_args(argv)
    try:
        bundle = convert(args.data_dir, args.project, args.out)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # a malformed hero output is an error message, not a traceback
        print(f"error: conversion failed ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1
    record = json.loads((args.out / "conversion.json").read_text())
    for note in record["filled_in"]:
        print(f"filled in: {note}", file=sys.stderr)
    for text in record["unmatched_assumption_warnings"]:
        print(
            f"warning: unparsed assumption warning (see conversion.json): {text}", file=sys.stderr
        )
    counts: dict[str, int] = {}
    for verdict in bundle["verdicts"].values():
        counts[verdict["status"]] = counts.get(verdict["status"], 0) + 1
    print(f"wrote {args.out / 'bundle.json'}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
