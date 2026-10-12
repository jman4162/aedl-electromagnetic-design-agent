"""Deterministic evaluator for tier-4 cross-layer tasks (t4-001).

The submission is a design bundle: ``bundle.json`` plus the files it names,
in one directory. The bundle states a system architecture, an element
DesignSpec, a board file with its provenance, and a verdict for every
requirement, each reported number carrying a provenance record.

Nothing the bundle says about itself is trusted. The evaluator

1. reads the requirement set from the SysML v2 model named in the task
   context (sysml2kit + sysml2kit-rf-library, the ``sysml`` extra) and adds
   the element task's requirements;
2. re-runs verification from the submitted architecture with the system
   task's evaluator (``satcom_terminal``: phased-array-systems, the PAM
   pattern recompute and the opensatcom cross-check over that task's held-out
   envelope) and from the element spec with the element task's evaluator
   (``patch_element``, analytic tier), and regenerates the board from the
   element spec with antenna-cad's KiCad emitter;
3. scores how well the bundle's verdicts, numbers and bookkeeping hold up.

Honest "unknown": a requirement the recomputing tier cannot certify (the
element evaluator lists it in ``screening_metrics``; at t1-001's analytic
tier that is the in-band S11) is recomputed as ``unknown`` with the
analytic answer kept as its screening status. A submitted ``unknown``
agrees with it. A submitted pass/fail agrees only when it equals the
screening status and its provenance says ``fidelity: analytic``, so a
bundle cannot present an analytic screen as a certified verdict.

Metrics produced:

- ``verdict_agreement``: fraction of recomputed requirements whose submitted
  status agrees under the rule above (a missing verdict disagrees)
- ``actual_agreement``: fraction of recomputed requirements whose submitted
  ``actual`` is within the task's tolerance of the recomputed value (a null
  actual agrees only on a requirement recomputed as unknown)
- ``provenance_completeness``: over every requirement and every submitted
  verdict, the fraction that is accounted for: an ``unknown`` verdict with a
  null actual, or a numeric actual with a complete provenance record (a
  SHA-256 ``artifact_hash``, a non-empty ``capability``, and ``origin`` and
  ``fidelity`` from the task's vocabularies). A missing verdict, or a
  pass/fail verdict without a number, is incomplete.
- ``board_valid``: 1 when the board path stays inside the bundle, ends in
  ``.kicad_pcb``, parses as a single ``(kicad_pcb ...)`` S-expression with
  ``version`` and ``generator`` children, carries a complete
  ``board_provenance`` whose hash is the shipped file's SHA-256, and is
  byte-identical to a board antenna-cad emits for the submitted element
  spec (straight from the synthesized design, or from that design after a
  round trip through its stored IR file); else 0
- ``recomputed_pass_count``: requirements the design closes at a certifying
  tier. A score, not a gate: the task asks for as many as possible.

Verdict hashes are checked for form only; the board hash is checked against
the file.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from aedl.registry import get_evaluator, register_evaluator
from aedl.result import EvaluationResult, RequirementResult
from aedl.spec import TaskSpec, find_task

BUNDLE_NAME = "bundle.json"
STATUSES = ("pass", "fail", "unknown")
DEFAULT_ORIGINS = ("derived", "vendor_claimed", "imported", "measured", "assumed")
DEFAULT_FIDELITIES = ("analytic", "fullwave_open", "fullwave_commercial", "measured")
#: Used when the task names no tolerance for a metric's unit suffix.
DEFAULT_TOLERANCE = {"rel": 1e-3}

_BUNDLE_KEYS = {"schema", "architecture", "element_spec", "board", "board_provenance", "verdicts"}
_REQUIRED_KEYS = {"architecture", "element_spec", "board", "verdicts"}
_VERDICT_KEYS = {"status", "actual", "unit", "provenance"}
_HASH = re.compile(r"(sha256:)?([0-9a-f]{64})")
_UNIT_SUFFIX = {"_db": "dB", "_mm": "mm", "_ghz": "GHz", "_w": "W", "_usd": "USD"}


@dataclass(frozen=True)
class Recomputed:
    """One requirement, verified from the submitted design by AEDL's code.

    ``status`` is ``unknown`` when the recomputing tier cannot certify the
    requirement; ``screening_status`` then holds that tier's pass/fail.
    """

    requirement_id: str
    metric: str
    op: str
    threshold: float
    actual: float
    status: str
    unit: str | None
    source: str
    screening_status: str | None = None
    screening_fidelity: str | None = None


@dataclass(frozen=True)
class Recomputation:
    """Everything the evaluator derives from the submitted design itself."""

    requirements: dict[str, Recomputed]
    #: SHA-256 of the board antenna-cad emits for the submitted element spec.
    board_sha256: str
    #: SHA-256 of the board emitted from that design after an IR round trip
    #: (``to_dict`` -> JSON -> ``from_dict``), which is how a flow that stores
    #: the IR as a file emits it. Coordinates differ in the sixth decimal.
    board_sha256_via_ir: str = ""


def _check(op: str, threshold: float, actual: float) -> bool:
    checks = {
        ">=": actual >= threshold,
        "<=": actual <= threshold,
        ">": actual > threshold,
        "<": actual < threshold,
        "==": actual == threshold,
    }
    if op not in checks:
        raise ValueError(f"unsupported requirement operator {op!r}")
    return checks[op]


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def load_bundle(submission: Path) -> tuple[Path, dict[str, Any]]:
    """Locate and validate a bundle; ``submission`` is the directory or its bundle.json."""
    path = submission / BUNDLE_NAME if submission.is_dir() else submission
    bundle = json.loads(path.read_text())
    if not isinstance(bundle, dict):
        raise ValueError(f"{BUNDLE_NAME} must be a JSON object")
    unknown = set(bundle) - _BUNDLE_KEYS
    if unknown:
        raise ValueError(f"{BUNDLE_NAME} has unknown keys {sorted(unknown)}")
    missing = _REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"{BUNDLE_NAME} is missing keys {sorted(missing)}")
    if not isinstance(bundle["architecture"], dict):
        raise ValueError("architecture must be an object")
    verdicts = bundle["verdicts"]
    if not isinstance(verdicts, dict):
        raise ValueError("verdicts must map requirement id -> verdict")
    for rid, verdict in verdicts.items():
        if not isinstance(verdict, dict):
            raise ValueError(f"verdict {rid!r} must be an object")
        extra = set(verdict) - _VERDICT_KEYS
        if extra:
            raise ValueError(f"verdict {rid!r} has unknown keys {sorted(extra)}")
        if verdict.get("status") not in STATUSES:
            raise ValueError(f"verdict {rid!r}: status must be one of {STATUSES}")
    return path.parent, bundle


def _inside(bundle_dir: Path, rel: Any, what: str) -> Path:
    """Resolve a bundle-relative path, refusing anything outside the bundle."""
    if not isinstance(rel, str) or not rel:
        raise ValueError(f"{what} must be a relative path string")
    root = bundle_dir.resolve()
    target = (root / rel).resolve()
    if Path(rel).is_absolute() or not target.is_relative_to(root):
        raise ValueError(f"{what} {rel!r} must stay inside the bundle directory")
    return target


def system_requirements(spec: TaskSpec) -> list[Any]:
    """Machine-checkable requirements from the task's SysML v2 model."""
    try:
        import sysml2kit_rf_library
        from sysml2kit.interop.requirements import extract_requirements
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the requirements_bundle evaluator reads requirements from a SysML v2 "
            "model: pip install 'aedl[sysml]'"
        ) from exc
    model = sysml2kit_rf_library.load_model(str(spec.context["sysml_model"]["model"]))
    specs = [s for s in extract_requirements(model) if s.op is not None and s.value is not None]
    if not specs:
        raise RuntimeError(
            f"model {spec.context['sysml_model']['model']!r} has no checkable requirements"
        )
    return sorted(specs, key=lambda s: str(s.id))


def _unit(metric: str) -> str | None:
    for suffix, unit in _UNIT_SUFFIX.items():
        if metric.endswith(suffix):
            return unit
    return None


def regenerate_board_sha256s(element_path: Path) -> tuple[str, str]:
    """SHA-256 of the boards antenna-cad 0.2 emits for an element DesignSpec.

    Returns (direct, via IR). The emitter is deterministic: UUIDs are derived
    from the design name, and the output depends only on the spec and the
    antenna-cad version (which it writes into the file as generator_version).

    A flow that stores the physical design as an IR file and emits the board
    from that file gets polygons rounded to six decimals in the IR, so its
    board differs from the direct one by up to 1e-6 mm in a few coordinates
    (the IR content hash is the same). Both are accepted.
    """
    from antenna_cad.backends.kicad import write_kicad_project
    from antenna_cad.designspec import DesignSpec

    design = DesignSpec.load(element_path).synthesize()
    stored = type(design).from_dict(json.loads(json.dumps(design.to_dict(), sort_keys=True)))
    hashes = []
    for variant in (design, stored):
        with tempfile.TemporaryDirectory(prefix="aedl-t4-board-") as td:
            board = write_kicad_project(variant, td)
            hashes.append(hashlib.sha256(board.read_bytes()).hexdigest())
    return hashes[0], hashes[1]


def regenerate_board_sha256(element_path: Path) -> str:
    """SHA-256 of the board emitted straight from the synthesized design."""
    return regenerate_board_sha256s(element_path)[0]


def recompute(spec: TaskSpec, bundle_dir: Path, bundle: dict[str, Any]) -> Recomputation:
    """Verify every requirement from the submitted architecture and element spec."""
    assert spec.path is not None, "TaskSpec must come from a file to find its sibling tasks"
    params = spec.evaluator_params
    element_path = _inside(bundle_dir, bundle["element_spec"], "element_spec")
    if not element_path.is_file():
        raise ValueError(f"element_spec {bundle['element_spec']!r} not found in the bundle")

    tasks_dir = spec.path.parent.parent
    system_task = find_task(tasks_dir, str(params["system_task"]))
    element_task = find_task(tasks_dir, str(params["element_task"]))
    sysml_reqs = system_requirements(spec)

    with tempfile.TemporaryDirectory(prefix="aedl-t4-") as td:
        arch_path = Path(td) / "architecture.yaml"
        arch_path.write_text(yaml.safe_dump(bundle["architecture"], sort_keys=False))
        system_result = get_evaluator(system_task.evaluator)(system_task, arch_path)
    system_metrics: dict[str, float] = system_result.info["metrics"]

    out: dict[str, Recomputed] = {}
    for req in sysml_reqs:
        if req.metric_key not in system_metrics:
            raise RuntimeError(
                f"SysML requirement {req.id} needs metric {req.metric_key!r}, which the "
                f"{system_task.id} evaluator does not produce ({sorted(system_metrics)})"
            )
        actual = float(system_metrics[req.metric_key])
        out[str(req.id)] = Recomputed(
            requirement_id=str(req.id),
            metric=str(req.metric_key),
            op=str(req.op),
            threshold=float(req.value),
            actual=actual,
            status="pass" if _check(str(req.op), float(req.value), actual) else "fail",
            unit=req.units or _unit(str(req.metric_key)),
            source=f"sysml:{spec.context['sysml_model']['package']}",
        )

    element_result = get_evaluator(element_task.evaluator)(element_task, element_path)
    screening = set(element_result.info.get("screening_metrics", []))
    element_fidelity = element_result.info.get("fidelity")
    limits = {r.id: r for r in element_task.requirements}
    prefix = str(params.get("element_prefix", "ELEM-"))
    for r in element_result.requirements:
        rid = prefix + r.requirement_id.upper()
        req = limits[r.requirement_id]
        op, threshold = (">=", req.min) if req.min is not None else ("<=", req.max)
        assert threshold is not None
        verdict = "pass" if r.passed else "fail"
        uncertified = r.metric in screening
        out[rid] = Recomputed(
            requirement_id=rid,
            metric=r.metric,
            op=op,
            threshold=float(threshold),
            actual=float(r.value),
            status="unknown" if uncertified else verdict,
            unit=_unit(r.metric),
            source=f"task:{element_task.id}",
            screening_status=verdict if uncertified else None,
            screening_fidelity=str(element_fidelity) if uncertified else None,
        )
    direct, via_ir = regenerate_board_sha256s(element_path)
    return Recomputation(requirements=out, board_sha256=direct, board_sha256_via_ir=via_ir)


def provenance_complete(
    record: Any,
    origins: tuple[str, ...] = DEFAULT_ORIGINS,
    fidelities: tuple[str, ...] = DEFAULT_FIDELITIES,
) -> bool:
    if not isinstance(record, dict):
        return False
    digest = record.get("artifact_hash")
    capability = record.get("capability")
    return (
        isinstance(digest, str)
        and bool(_HASH.fullmatch(digest))
        and isinstance(capability, str)
        and bool(capability.strip())
        and record.get("origin") in origins
        and record.get("fidelity") in fidelities
    )


def _bare_hash(digest: str) -> str:
    m = _HASH.fullmatch(digest)
    return m.group(2) if m else ""


_TOKEN = re.compile(r'\s*(?:(\()|(\))|"((?:[^"\\]|\\.)*)"|([^\s()"]+))', re.DOTALL)


def _parse_header(text: str) -> tuple[str | None, dict[str, list[str]], str]:
    """(top head, depth-1 child heads -> their atom args, error) of an S-expression.

    A minimal tokenizer, not a KiCad parser: it checks that the text is one
    balanced list, and records the head and atoms of each direct child.
    """
    pos, depth, top_closed = 0, 0, False
    head: str | None = None
    children: dict[str, list[str]] = {}
    current: list[str] | None = None
    expect_head = False
    while pos < len(text):
        if not text[pos:].strip():
            break
        m = _TOKEN.match(text, pos)
        if m is None:
            return head, children, f"parse error at offset {pos}"
        pos = m.end()
        if top_closed:
            return head, children, "content after the closing parenthesis"
        open_, close, string, atom = m.groups()
        if open_:
            depth += 1
            expect_head = True
            continue
        if close:
            depth -= 1
            if depth < 0:
                return head, children, "parse error: unbalanced ')'"
            if depth == 1:
                current = None
            if depth == 0:
                top_closed = True
            continue
        value = string if string is not None else atom
        if expect_head:
            expect_head = False
            if depth == 1 and head is None:
                head = value
            elif depth == 2 and value is not None:
                current = children.setdefault(value, [])
            continue
        if depth == 2 and current is not None and value is not None:
            current.append(value)
    if depth != 0:
        return head, children, f"parse error: {depth} unclosed '('"
    return head, children, ""


def check_board(bundle_dir: Path, rel: Any) -> tuple[bool, str]:
    """Whether the board file is present and parses as a KiCad PCB."""
    try:
        path = _inside(bundle_dir, rel, "board")
    except ValueError as exc:
        return False, str(exc)
    if path.suffix != ".kicad_pcb":
        return False, f"board {rel!r} is not a .kicad_pcb file"
    if not path.is_file():
        return False, f"board {rel!r} not found in the bundle"
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, f"board unreadable: {exc}"
    head, children, error = _parse_header(text)
    if head != "kicad_pcb":
        return False, f"top-level list is {head!r}, expected kicad_pcb"
    if error:
        return False, error
    version = children.get("version") or []
    if not version or not version[0].isdigit():
        return False, "missing (version <integer>)"
    if "generator" not in children:
        return False, "missing (generator ...)"
    return True, "ok"


def board_verdict(
    bundle_dir: Path,
    bundle: dict[str, Any],
    expected_sha256: str,
    origins: tuple[str, ...] = DEFAULT_ORIGINS,
    fidelities: tuple[str, ...] = DEFAULT_FIDELITIES,
    also_accept_sha256: str = "",
) -> tuple[bool, dict[str, Any]]:
    """Structure, declared provenance, and regeneration checks on the board."""
    ok, reason = check_board(bundle_dir, bundle["board"])
    details: dict[str, Any] = {"structure": reason, "expected_sha256": expected_sha256}
    if also_accept_sha256:
        details["expected_sha256_via_ir"] = also_accept_sha256
    if not ok:
        return False, details
    shipped = hashlib.sha256(_inside(bundle_dir, bundle["board"], "board").read_bytes()).hexdigest()
    details["shipped_sha256"] = shipped
    record = bundle.get("board_provenance")
    if not provenance_complete(record, origins, fidelities):
        details["provenance"] = "missing or incomplete board_provenance"
        return False, details
    assert isinstance(record, dict)
    if _bare_hash(str(record["artifact_hash"])) != shipped:
        details["provenance"] = "board_provenance.artifact_hash is not the shipped file's SHA-256"
        return False, details
    details["provenance"] = "ok"
    if shipped == expected_sha256:
        details["regeneration"] = "ok"
    elif also_accept_sha256 and shipped == also_accept_sha256:
        details["regeneration"] = "ok (emitted from the stored IR)"
    else:
        details["regeneration"] = (
            "board differs from what antenna-cad emits for the submitted element spec"
        )
        return False, details
    return True, details


def _tolerance(metric: str, table: dict[str, Any]) -> dict[str, float]:
    for suffix, tol in table.items():
        if metric.endswith(str(suffix)):
            return {k: float(v) for k, v in dict(tol).items()}
    return dict(DEFAULT_TOLERANCE)


def _within(submitted: float, recomputed: float, tol: dict[str, float]) -> bool:
    bound = max(tol.get("abs", 0.0), tol.get("rel", 0.0) * abs(recomputed))
    return abs(submitted - recomputed) <= bound


def _status_agrees(rec: Recomputed, verdict: dict[str, Any] | None) -> bool:
    if verdict is None:
        return False
    status = verdict.get("status")
    if rec.status != "unknown":
        return bool(status == rec.status)
    if status == "unknown":
        return True
    # A definite verdict on an uncertifiable requirement must be the
    # screening answer, and must say it is one.
    prov = verdict.get("provenance")
    labelled = isinstance(prov, dict) and prov.get("fidelity") == rec.screening_fidelity
    return status == rec.screening_status and labelled


def _accounted(
    verdict: dict[str, Any] | None,
    origins: tuple[str, ...],
    fidelities: tuple[str, ...],
) -> bool:
    if verdict is None:
        return False
    if _number(verdict.get("actual")) is None:
        return verdict.get("status") == "unknown" and verdict.get("actual") is None
    return provenance_complete(verdict.get("provenance"), origins, fidelities)


def score(
    bundle_dir: Path,
    bundle: dict[str, Any],
    recomputation: Recomputation,
    origins: tuple[str, ...] = DEFAULT_ORIGINS,
    fidelities: tuple[str, ...] = DEFAULT_FIDELITIES,
    tolerances: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Bundle-level metrics against a recomputation, plus a per-requirement table."""
    recomputed = recomputation.requirements
    verdicts: dict[str, Any] = bundle["verdicts"]
    table = tolerances or {}
    rows: dict[str, dict[str, Any]] = {}
    agree = actual_agree = 0
    for rid, rec in sorted(recomputed.items()):
        verdict = verdicts.get(rid)
        status_ok = _status_agrees(rec, verdict)
        submitted_actual = _number(verdict.get("actual")) if verdict else None
        tol = _tolerance(rec.metric, table)
        if submitted_actual is not None:
            actual_ok = _within(submitted_actual, rec.actual, tol)
        else:
            actual_ok = (
                rec.status == "unknown" and verdict is not None and verdict.get("actual") is None
            )
        agree += status_ok
        actual_agree += actual_ok
        rows[rid] = {
            **{k: v for k, v in asdict(rec).items() if k != "requirement_id"},
            "recomputed_status": rec.status,
            "submitted_status": verdict.get("status") if verdict else None,
            "submitted_actual": verdict.get("actual") if verdict else None,
            "tolerance": tol,
            "agrees": status_ok,
            "actual_agrees": actual_ok,
        }

    ids = sorted(set(recomputed) | set(verdicts))
    accounted = {rid for rid in ids if _accounted(verdicts.get(rid), origins, fidelities)}
    board_ok, board_details = board_verdict(
        bundle_dir,
        bundle,
        recomputation.board_sha256,
        origins,
        fidelities,
        also_accept_sha256=recomputation.board_sha256_via_ir,
    )

    n = len(recomputed)
    metrics = {
        "verdict_agreement": agree / n if n else 0.0,
        "actual_agreement": actual_agree / n if n else 0.0,
        "provenance_completeness": len(accounted) / len(ids) if ids else 0.0,
        "board_valid": 1.0 if board_ok else 0.0,
        "recomputed_pass_count": float(sum(r.status == "pass" for r in recomputed.values())),
    }
    details = {
        "requirements": rows,
        "provenance_incomplete": sorted(set(ids) - accounted),
        "unrecognized_verdicts": sorted(set(verdicts) - set(recomputed)),
        "board": {"valid": board_ok, **board_details},
    }
    return metrics, details


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for dist in ("sysml2kit", "sysml2kit-rf-library", "antenna-cad", "phased-array-systems"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


@register_evaluator("requirements_bundle")
def evaluate(spec: TaskSpec, submission: Path) -> EvaluationResult:
    t_start = time.perf_counter()
    bundle_dir, bundle = load_bundle(submission)
    recomputation = recompute(spec, bundle_dir, bundle)
    vocab = spec.context.get("provenance", {})
    metrics, details = score(
        bundle_dir,
        bundle,
        recomputation,
        origins=tuple(vocab.get("origins", DEFAULT_ORIGINS)),
        fidelities=tuple(vocab.get("fidelities", DEFAULT_FIDELITIES)),
        tolerances=spec.evaluator_params.get("actual_tolerance_by_suffix"),
    )

    req_results = []
    for req in spec.requirements:
        if req.metric not in metrics:
            raise KeyError(
                f"task {spec.id}: requirement {req.id!r} references unknown "
                f"metric {req.metric!r}; available: {sorted(metrics)}"
            )
        value = metrics[req.metric]
        req_results.append(
            RequirementResult(
                requirement_id=req.id,
                metric=req.metric,
                value=value,
                limit=req.limit,
                passed=req.check(value),
            )
        )

    recomputed = recomputation.requirements
    architecture_sha = hashlib.sha256(
        json.dumps(bundle["architecture"], sort_keys=True).encode()
    ).hexdigest()
    return EvaluationResult(
        task_id=spec.id,
        passed=all(r.passed for r in req_results),
        requirements=tuple(req_results),
        info={
            "metrics": metrics,
            "score": {
                "recomputed_passes": int(metrics["recomputed_pass_count"]),
                "unknown_at_this_tier": sum(r.status == "unknown" for r in recomputed.values()),
                "of": len(recomputed),
            },
            **details,
            "architecture_sha256": architecture_sha,
            "versions": _versions(),
        },
        cost={"evaluation_wall_time_s": round(time.perf_counter() - t_start, 3)},
    )
