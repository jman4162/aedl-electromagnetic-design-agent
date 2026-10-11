"""Deterministic evaluator for tier-4 cross-layer tasks (t4-001).

The submission is a design bundle: ``bundle.json`` plus the files it names,
in one directory. The bundle states a system architecture, an element
DesignSpec, a board file, and a verdict for every requirement, each reported
number carrying a provenance record.

Nothing the bundle says about itself is trusted. The evaluator

1. reads the requirement set from the SysML v2 model named in the task
   context (sysml2kit + sysml2kit-rf-library, the ``sysml`` extra) and adds
   the element task's requirements;
2. re-runs verification from the submitted architecture with the system
   task's evaluator (``satcom_terminal``: phased-array-systems, the PAM
   pattern recompute and the opensatcom cross-check over that task's held-out
   envelope) and from the element spec with the element task's evaluator
   (``patch_element``, analytic tier);
3. scores how well the bundle's verdicts and bookkeeping hold up.

Metrics produced:

- ``verdict_agreement``: fraction of recomputed requirements whose submitted
  status equals the recomputed status (a missing verdict disagrees)
- ``provenance_completeness``: fraction of reported numbers (verdicts whose
  ``actual`` is a number) carrying a complete provenance record: a SHA-256
  ``artifact_hash``, a non-empty ``capability``, and ``origin`` and
  ``fidelity`` from the task's vocabularies
- ``board_valid``: 1 when the board path stays inside the bundle, ends in
  ``.kicad_pcb``, and parses as a single ``(kicad_pcb ...)`` S-expression with
  ``version`` and ``generator`` children; else 0
- ``recomputed_pass_count``: requirements the design actually closes. A
  score, not a gate: the task asks for as many as possible.

Hashes are checked for form, not resolved against files; a bundle can cite a
hash of an artifact it does not ship.
"""

from __future__ import annotations

import hashlib
import json
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

_BUNDLE_KEYS = {"schema", "architecture", "element_spec", "board", "verdicts"}
_REQUIRED_KEYS = {"architecture", "element_spec", "board", "verdicts"}
_VERDICT_KEYS = {"status", "actual", "unit", "provenance"}
_HASH = re.compile(r"(sha256:)?[0-9a-f]{64}")
_UNIT_SUFFIX = {"_db": "dB", "_mm": "mm", "_ghz": "GHz", "_w": "W", "_usd": "USD"}


@dataclass(frozen=True)
class Recomputed:
    """One requirement, verified from the submitted design by AEDL's code."""

    requirement_id: str
    metric: str
    op: str
    threshold: float
    actual: float
    status: str
    unit: str | None
    source: str


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


def recompute(spec: TaskSpec, bundle_dir: Path, bundle: dict[str, Any]) -> dict[str, Recomputed]:
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
    prefix = str(params.get("element_prefix", "ELEM-"))
    for r in element_result.requirements:
        rid = prefix + r.requirement_id.upper()
        op, threshold = r.limit.split()
        out[rid] = Recomputed(
            requirement_id=rid,
            metric=r.metric,
            op=op,
            threshold=float(threshold),
            actual=float(r.value),
            status="pass" if r.passed else "fail",
            unit=_unit(r.metric),
            source=f"task:{element_task.id}",
        )
    return out


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


def score(
    bundle_dir: Path,
    bundle: dict[str, Any],
    recomputed: dict[str, Recomputed],
    origins: tuple[str, ...] = DEFAULT_ORIGINS,
    fidelities: tuple[str, ...] = DEFAULT_FIDELITIES,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Bundle-level metrics against a recomputation, plus a per-requirement table."""
    verdicts: dict[str, Any] = bundle["verdicts"]
    rows: dict[str, dict[str, Any]] = {}
    agree = 0
    for rid, rec in sorted(recomputed.items()):
        submitted = verdicts.get(rid)
        status = submitted.get("status") if isinstance(submitted, dict) else None
        same = status == rec.status
        agree += same
        rows[rid] = {
            **{k: v for k, v in asdict(rec).items() if k != "requirement_id"},
            "recomputed_status": rec.status,
            "submitted_status": status,
            "submitted_actual": submitted.get("actual") if isinstance(submitted, dict) else None,
            "agrees": same,
        }

    reported = {
        rid: v
        for rid, v in verdicts.items()
        if isinstance(v.get("actual"), (int, float)) and not isinstance(v.get("actual"), bool)
    }
    complete = {
        rid
        for rid, v in reported.items()
        if provenance_complete(v.get("provenance"), origins, fidelities)
    }
    board_ok, board_reason = check_board(bundle_dir, bundle["board"])

    metrics = {
        "verdict_agreement": agree / len(recomputed) if recomputed else 0.0,
        "provenance_completeness": len(complete) / len(reported) if reported else 0.0,
        "board_valid": 1.0 if board_ok else 0.0,
        "recomputed_pass_count": float(sum(r.status == "pass" for r in recomputed.values())),
    }
    details = {
        "requirements": rows,
        "provenance_incomplete": sorted(set(reported) - complete),
        "unrecognized_verdicts": sorted(set(verdicts) - set(recomputed)),
        "board": {"valid": board_ok, "reason": board_reason},
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
    recomputed = recompute(spec, bundle_dir, bundle)
    vocab = spec.context.get("provenance", {})
    metrics, details = score(
        bundle_dir,
        bundle,
        recomputed,
        origins=tuple(vocab.get("origins", DEFAULT_ORIGINS)),
        fidelities=tuple(vocab.get("fidelities", DEFAULT_FIDELITIES)),
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
                "of": len(recomputed),
            },
            **details,
            "architecture_sha256": architecture_sha,
            "versions": _versions(),
        },
        cost={"evaluation_wall_time_s": round(time.perf_counter() - t_start, 3)},
    )
