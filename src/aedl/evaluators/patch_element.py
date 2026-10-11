"""Deterministic evaluator for tier-1 patch element tasks (t1-001).

The submission is an antenna-cad ``DesignSpec`` YAML: design intent only
(center frequency, feed impedance, laminate, polarization). Submissions never
carry coordinates; the evaluator synthesizes the geometry itself with
antenna-cad (the ``element`` extra) and scores what that geometry does.

Two fidelity tiers, chosen by ``evaluator.params.fidelity``:

- ``analytic`` (default): the cavity model in
  :mod:`aedl.evaluators.patch_cavity`, implemented here and computed from the
  synthesized dimensions alone. A pass at this tier is a screening verdict;
  ``info["certifies_band_coverage"]`` is False and ``info["uncertainty"]``
  reports the in-band match with the resonance moved by the model's stated
  accuracy band.
- ``openems``: an FDTD run of the same synthesized board through antenna-cad's
  openEMS backend. Only when openEMS is installed (native bindings or the
  antenna-cad Docker image); a nightly tier, never part of the default tests.

Metrics produced (requirements reference these by name):

- ``s11_max_in_band_db``: worst |S11| over the task band, referenced to the
  task's system impedance
- ``footprint_mm``: the larger side of the top-copper bounding box (patch and
  feed line)
- ``f_res_ghz``: reported, not required by t1-001; TM010 resonance computed
  from geometry (analytic) or the solver's resonance (openems)
"""

from __future__ import annotations

import math
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from aedl.evaluators.patch_cavity import (
    CavityModel,
    PatchGeometry,
    cavity_model,
    port_s11,
)
from aedl.registry import register_evaluator
from aedl.result import EvaluationResult, RequirementResult
from aedl.spec import TaskSpec

FIDELITIES = ("analytic", "openems")

#: Resonance accuracy of the analytic tier, as a fraction (docs/tasks/t1-001.md).
F_RES_UNCERTAINTY = 0.03
_SHIFTS = {"-3%": 0.97, "-1%": 0.99, "+1%": 1.01, "+3%": 1.03}

_TOP_KEYS = {"design", "requirements", "element", "acceptance"}
_DESIGN_KEYS = {"name"}
_REQUIREMENT_KEYS = {
    "center_frequency",
    "impedance",
    "substrate",
    "substrate_height",
    "polarization",
}
_ELEMENT_KEYS = {"type"}


def _require_antenna_cad() -> None:
    try:
        import antenna_cad  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "the patch_element evaluator needs antenna-cad to synthesize geometry: "
            "pip install 'aedl[element]'"
        ) from exc


def _check_keys(section: Any, allowed: set[str], name: str) -> None:
    if not isinstance(section, dict):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(section) - allowed
    if unknown:
        raise ValueError(
            f"{name} has unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}. "
            "Geometry is synthesized by the evaluator and cannot be submitted."
        )


def _validate_document(doc: Any) -> None:
    if not isinstance(doc, dict):
        raise ValueError("design.yaml must be a mapping")
    if "array" in doc:
        raise ValueError("this is a single element task: remove the array section")
    _check_keys(doc, _TOP_KEYS, "design.yaml")
    for key, allowed in (
        ("design", _DESIGN_KEYS),
        ("requirements", _REQUIREMENT_KEYS),
        ("element", _ELEMENT_KEYS),
    ):
        if key not in doc:
            raise ValueError(f"design.yaml is missing the {key!r} section")
        _check_keys(doc[key], allowed, key)


def load_design(spec: TaskSpec, submission: Path) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Validate a DesignSpec file and synthesize it.

    Returns (DesignSpec, RectangularPatch, PhysicalDesign, raw document).
    """
    _require_antenna_cad()
    from antenna_cad.core.units import to_mm
    from antenna_cad.designspec import DesignSpec
    from antenna_cad.elements import RectangularPatch
    from pydantic import ValidationError

    doc = yaml.safe_load(submission.read_text())
    _validate_document(doc)
    try:
        design_spec = DesignSpec.load(submission)
    except (ValidationError, KeyError) as exc:
        raise ValueError(f"not a valid antenna-cad DesignSpec: {exc}") from exc

    laminate = spec.context["substrate"]
    problem = design_spec.problem
    if str(problem.substrate).lower() != str(laminate["name"]).lower():
        raise ValueError(f"substrate must be {laminate['name']}, got {problem.substrate!r}")
    height_mm = to_mm(problem.substrate_height)
    stock = [float(h) for h in laminate["stock_heights_mm"]]
    if not any(abs(height_mm - h) < 1e-6 for h in stock):
        raise ValueError(
            f"substrate_height {height_mm} mm is not a stock {laminate['name']} thickness: {stock}"
        )
    if str(problem.polarization).lower() != "linear":
        raise ValueError("only linear polarization is synthesized (inset-fed patch)")

    # A material table that disagrees with the task would score a different
    # laminate than the one the brief states. Fail loudly rather than pick one.
    material = problem.substrate_obj
    if not (
        math.isclose(material.eps_r, float(laminate["eps_r"]))
        and math.isclose(material.tan_delta, float(laminate["tan_delta"]))
    ):
        raise RuntimeError(
            f"antenna-cad's {material.name} (eps_r {material.eps_r}, tan_delta "
            f"{material.tan_delta}) disagrees with the task "
            f"(eps_r {laminate['eps_r']}, tan_delta {laminate['tan_delta']})"
        )

    patch = RectangularPatch.synthesize(problem)
    design = patch.to_design(name=design_spec.name)
    return design_spec, patch, design, doc


def footprint(design: Any) -> dict[str, float]:
    """Top-copper extents in mm; raises if copper leaves the board outline."""
    from shapely.ops import unary_union

    copper = unary_union([s.polygon for s in design.shapes if s.layer == "top"])
    if not design.board.outline.covers(copper):
        raise ValueError("synthesized copper extends past the board outline")
    minx, miny, maxx, maxy = copper.bounds
    return {
        "extent_x_mm": maxx - minx,
        "extent_y_mm": maxy - miny,
        "copper_min_y_mm": miny,
        "copper_max_y_mm": maxy,
    }


def patch_geometry(spec: TaskSpec, patch: Any, design: Any) -> PatchGeometry:
    """The cavity model's inputs, read from the synthesized design."""
    from antenna_cad.core.units import to_mm

    laminate = spec.context["substrate"]
    extents = footprint(design)
    length_mm = to_mm(patch.length)
    inset_mm = to_mm(patch.inset)
    port_y_mm = float(design.ports[0].position[1])
    # The patch's top edge is the highest copper; the feed enters from below.
    patch_bottom_mm = extents["copper_max_y_mm"] - length_mm
    return PatchGeometry(
        width_m=to_mm(patch.width) / 1e3,
        length_m=length_mm / 1e3,
        inset_m=inset_mm / 1e3,
        feed_width_m=to_mm(patch.feed_width) / 1e3,
        feed_length_m=(patch_bottom_mm + inset_mm - port_y_mm) / 1e3,
        height_m=to_mm(patch.problem.substrate_height) / 1e3,
        eps_r=float(laminate["eps_r"]),
        tan_delta=float(laminate["tan_delta"]),
        sigma_s_per_m=float(spec.context["conductor"]["sigma_s_per_m"]),
    )


def _band(spec: TaskSpec) -> tuple[float, float]:
    lo, hi = (float(x) for x in spec.context["band_ghz"])
    return lo * 1e9, hi * 1e9


def _s11_db(s11: Any) -> Any:
    return 20.0 * np.log10(np.abs(s11))


def _analytic(spec: TaskSpec, geom: PatchGeometry) -> tuple[dict[str, float], dict[str, Any]]:
    lo, hi = _band(spec)
    n = int(spec.evaluator_params.get("n_band_points", 101))
    z_ref = float(spec.context["system_impedance_ohm"])
    band = np.linspace(lo, hi, n)
    model: CavityModel = cavity_model(geom)

    worst = float(np.max(_s11_db(port_s11(model, geom, band, z_ref))))
    shifted = {
        label: round(float(np.max(_s11_db(port_s11(model, geom, band, z_ref, scale)))), 4)
        for label, scale in _SHIFTS.items()
    }

    # Wide sweep for the info block: where the match actually is.
    centre = 0.5 * (lo + hi)
    sweep = np.linspace(0.8 * centre, 1.2 * centre, 2001)
    sweep_db = _s11_db(port_s11(model, geom, sweep, z_ref))
    below = sweep[sweep_db <= -10.0]
    metrics = {"s11_max_in_band_db": worst, "f_res_ghz": model.f_res_hz / 1e9}
    info: dict[str, Any] = {
        "cavity": {
            "q_rad": model.q_rad,
            "q_cond": model.q_cond,
            "q_diel": model.q_diel,
            "q_total": model.q_total,
            "r_edge_ohm": model.r_edge_ohm,
            "r_in_ohm": model.r_in_ohm,
            "z_feed_ohm": model.z_feed_ohm,
            "eps_eff_res": model.eps_eff_res,
        },
        "s11_min_db": float(np.min(sweep_db)),
        "s11_min_freq_ghz": float(sweep[int(np.argmin(sweep_db))]) / 1e9,
        "match_10db_ghz": [float(below[0]) / 1e9, float(below[-1]) / 1e9] if below.size else None,
        "uncertainty": {
            "f_res_fraction": F_RES_UNCERTAINTY,
            "s11_max_in_band_db_with_f_res_shift": shifted,
            "note": (
                "analytic tier: resonance accurate to about +/-3 %, bandwidth low by up to "
                "about 25 %; the shifted values show what the resonance band does to the "
                "in-band match"
            ),
        },
    }
    return metrics, info


def _openems(spec: TaskSpec, design: Any) -> tuple[dict[str, float], dict[str, Any]]:
    from antenna_cad.solvers import SimulationConfig
    from antenna_cad.solvers.openems.solver import OpenEMS, OpenEMSNotAvailableError

    lo, hi = _band(spec)
    with tempfile.TemporaryDirectory(prefix="aedl-openems-") as workdir:
        try:
            solver = OpenEMS(workdir=workdir, mode="auto")
        except OpenEMSNotAvailableError as exc:
            raise RuntimeError(f"fidelity 'openems' cannot run here: {exc}") from exc
        result = solver.simulate(
            design,
            SimulationConfig(f_start=0.8 * lo, f_stop=1.2 * hi, n_freq=801),
        )
    freq = np.asarray(result.s_parameters["frequency"].values, dtype=float)
    s11_db = _s11_db(np.asarray(result.s_parameters["s11"].values))
    n = int(spec.evaluator_params.get("n_band_points", 101))
    band = np.linspace(lo, hi, n)
    worst = float(np.max(np.interp(band, freq, s11_db)))
    metrics = {"s11_max_in_band_db": worst, "f_res_ghz": float(result.metrics["f_res_hz"]) / 1e9}
    return metrics, {"solver": dict(result.solver), "solver_metrics": dict(result.metrics)}


def _versions() -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for dist in ("antenna-cad", "numpy", "scipy"):
        try:
            out[dist] = version(dist)
        except PackageNotFoundError:
            out[dist] = None
    return out


def element_metrics(spec: TaskSpec, submission: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Every metric and the info block for one DesignSpec file."""
    fidelity = str(spec.evaluator_params.get("fidelity", "analytic"))
    if fidelity not in FIDELITIES:
        raise ValueError(f"unknown fidelity {fidelity!r}; expected one of {FIDELITIES}")

    _, patch, design, doc = load_design(spec, submission)
    extents = footprint(design)
    geom = patch_geometry(spec, patch, design)

    if fidelity == "analytic":
        metrics, info = _analytic(spec, geom)
    else:
        metrics, info = _openems(spec, design)
    metrics["footprint_mm"] = max(extents["extent_x_mm"], extents["extent_y_mm"])

    info.update(
        {
            "fidelity": fidelity,
            "certifies_band_coverage": fidelity != "analytic",
            # Metrics this tier reports but cannot certify against a built
            # board; consumers (t4-001) treat verdicts on them as screening.
            "screening_metrics": ["s11_max_in_band_db"] if fidelity == "analytic" else [],
            "design": doc,
            "design_content_hash": design.content_hash(),
            "footprint": extents,
            "geometry_mm": {
                "width": geom.width_m * 1e3,
                "length": geom.length_m * 1e3,
                "inset": geom.inset_m * 1e3,
                "feed_width": geom.feed_width_m * 1e3,
                "feed_length": geom.feed_length_m * 1e3,
                "substrate_height": geom.height_m * 1e3,
            },
            "versions": _versions(),
        }
    )
    return metrics, info


@register_evaluator("patch_element")
def evaluate(spec: TaskSpec, submission: Path) -> EvaluationResult:
    t_start = time.perf_counter()
    metrics, info = element_metrics(spec, submission)

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
    info["metrics"] = metrics
    return EvaluationResult(
        task_id=spec.id,
        passed=all(r.passed for r in req_results),
        requirements=tuple(req_results),
        info=info,
        cost={"evaluation_wall_time_s": round(time.perf_counter() - t_start, 3)},
    )
