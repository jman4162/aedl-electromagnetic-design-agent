"""Evaluator sanity for t4-001: the (synthetic) reference bundle passes, and
each requirement catches a specific defect in a bundle.

The expensive part, recomputing every requirement from the submitted
architecture and element spec, runs a handful of times. Bundle-only defects
(verdicts, provenance, board) are scored against one cached recomputation.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from aedl import get_evaluator
from aedl.spec import find_task

pytest.importorskip("phased_array_systems", reason="requires the 'systems' extra")
pytest.importorskip("opensatcom", reason="requires the 'systems' extra")
pytest.importorskip("antenna_cad", reason="requires the 'element' extra (antenna-cad)")
pytest.importorskip("sysml2kit_rf_library", reason="requires the 'sysml' extra")

from aedl.evaluators import requirements_bundle as rb

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
REFERENCE = TASKS / "t4-001-requirements-to-board" / "reference" / "submission"

SYSTEM_IDS = {
    "REQ-LINK-MARGIN",
    "REQ-SLL",
    "REQ-INDEP-LINK",
    "REQ-CLEARSKY-AGREE",
    "REQ-GAIN-XCHECK",
    "REQ-POWER",
    "REQ-COST",
    "REQ-GRATING",
}
ELEMENT_IDS = {"ELEM-S11-IN-BAND", "ELEM-FOOTPRINT", "ELEM-F-RES-LO", "ELEM-F-RES-HI"}


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t4-001")


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory):
    # Copied before any harness run hides the reference directory, and so
    # that tests can damage their own copy.
    dest = tmp_path_factory.mktemp("ref") / "submission"
    shutil.copytree(REFERENCE, dest)
    return dest


@pytest.fixture(scope="module")
def reference_result(spec, reference_dir):
    return get_evaluator(spec.evaluator)(spec, reference_dir / "bundle.json")


@pytest.fixture(scope="module")
def recomputed(spec, reference_dir):
    bundle_dir, bundle = rb.load_bundle(reference_dir / "bundle.json")
    return rb.recompute(spec, bundle_dir, bundle)


@pytest.fixture()
def bundle_copy(reference_dir, tmp_path):
    dest = tmp_path / "submission"
    shutil.copytree(reference_dir, dest)
    return dest


def _bundle(path: Path) -> dict:
    return json.loads((path / "bundle.json").read_text())


def _score(path: Path, bundle: dict, recomputed) -> dict:
    metrics, _ = rb.score(path, bundle, recomputed)
    return metrics


def _failures(result):
    return {r.requirement_id for r in result.requirements if not r.passed}


# --- task wiring -------------------------------------------------------------


def test_task_loads_through_the_registry(spec):
    assert spec.tier == 4
    assert spec.evaluator == "requirements_bundle"
    assert {r.metric for r in spec.requirements} == {
        "verdict_agreement",
        "provenance_completeness",
        "board_valid",
    }


def test_context_mirrors_the_system_and_element_tasks(spec):
    """The agent sees t3-001's parts table and bounds and t1-001's laminate
    through this task's context; a drift between the copies would score a
    design against numbers the agent never saw."""
    t3 = find_task(TASKS, "t3-001")
    t1 = find_task(TASKS, "t1-001")
    assert spec.context["system"] == {k: v for k, v in t3.context.items() if k != "mission"}
    assert spec.context["element"] == t1.context


def test_requirement_ids_in_context_match_the_sources(spec, recomputed):
    assert set(spec.context["system_requirement_ids"]) == SYSTEM_IDS
    assert set(spec.context["element_requirement_ids"]) == ELEMENT_IDS
    assert set(recomputed) == SYSTEM_IDS | ELEMENT_IDS


# --- reference -----------------------------------------------------------------


def test_reference_passes(reference_result):
    assert reference_result.passed, _failures(reference_result)


def test_reference_closes_every_requirement_on_recompute(reference_result):
    metrics = reference_result.info["metrics"]
    assert metrics["recomputed_pass_count"] == 12
    assert metrics["verdict_agreement"] == 1.0
    assert metrics["provenance_completeness"] == 1.0
    assert metrics["board_valid"] == 1.0


def test_pass_count_is_reported_not_gated(spec, reference_result):
    assert "recomputed_pass_count" not in {r.metric for r in spec.requirements}
    assert reference_result.info["score"]["recomputed_passes"] == 12


def test_determinism(spec, reference_dir, reference_result):
    again = get_evaluator(spec.evaluator)(spec, reference_dir / "bundle.json")
    assert again.info["metrics"] == reference_result.info["metrics"]
    assert again.info["requirements"] == reference_result.info["requirements"]


def test_directory_and_file_submissions_resolve_alike(reference_dir):
    assert rb.load_bundle(reference_dir) == rb.load_bundle(reference_dir / "bundle.json")


# --- verdict agreement -----------------------------------------------------------


def _flip(bundle: dict, *ids: str) -> dict:
    out = copy.deepcopy(bundle)
    for rid in ids:
        status = out["verdicts"][rid]["status"]
        out["verdicts"][rid]["status"] = "fail" if status == "pass" else "pass"
    return out


def test_one_wrong_verdict_is_tolerated(bundle_copy, recomputed):
    metrics = _score(bundle_copy, _flip(_bundle(bundle_copy), "REQ-POWER"), recomputed)
    assert metrics["verdict_agreement"] == pytest.approx(11 / 12)
    assert metrics["verdict_agreement"] >= 0.9


def test_two_wrong_verdicts_fail_agreement(bundle_copy, recomputed):
    bundle = _flip(_bundle(bundle_copy), "REQ-POWER", "ELEM-FOOTPRINT")
    assert _score(bundle_copy, bundle, recomputed)["verdict_agreement"] == pytest.approx(10 / 12)


def test_missing_verdicts_count_as_disagreement(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    del bundle["verdicts"]["REQ-SLL"]
    del bundle["verdicts"]["ELEM-F-RES-HI"]
    assert _score(bundle_copy, bundle, recomputed)["verdict_agreement"] == pytest.approx(10 / 12)


def test_changed_architecture_with_stale_verdicts_fails(spec, bundle_copy):
    """More power per element busts the prime-power ceiling and a uniform
    taper busts the sidelobe spec. A bundle that still claims both pass is
    caught by recomputation, not by anything it says about itself."""
    bundle = _bundle(bundle_copy)
    bundle["architecture"]["rf"]["tx_power_w_per_elem"] = 0.5
    bundle["architecture"]["array"]["taper_type"] = "uniform"
    (bundle_copy / "bundle.json").write_text(json.dumps(bundle))
    result = get_evaluator(spec.evaluator)(spec, bundle_copy / "bundle.json")
    assert "verdict-agreement" in _failures(result)
    rows = result.info["requirements"]
    assert rows["REQ-POWER"]["recomputed_status"] == "fail"
    assert rows["REQ-SLL"]["recomputed_status"] == "fail"
    assert result.info["metrics"]["recomputed_pass_count"] <= 10


# --- provenance --------------------------------------------------------------------


def test_one_missing_provenance_record_fails_completeness(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    del bundle["verdicts"]["REQ-COST"]["provenance"]
    metrics = _score(bundle_copy, bundle, recomputed)
    assert metrics["provenance_completeness"] == pytest.approx(11 / 12)
    assert metrics["provenance_completeness"] < 0.95


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("origin", "guessed"),
        ("fidelity", "full_wave"),
        ("artifact_hash", "abc123"),
        ("artifact_hash", "ab" * 32 + "\n"),
        ("capability", ""),
    ],
)
def test_malformed_provenance_is_incomplete(bundle_copy, recomputed, field, value):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-LINK-MARGIN"]["provenance"][field] = value
    assert _score(bundle_copy, bundle, recomputed)["provenance_completeness"] < 1.0


def test_both_hash_spellings_are_accepted(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    prov = bundle["verdicts"]["REQ-LINK-MARGIN"]["provenance"]
    prov["artifact_hash"] = prov["artifact_hash"].removeprefix("sha256:")
    assert _score(bundle_copy, bundle, recomputed)["provenance_completeness"] == 1.0


def test_verdict_without_a_number_needs_no_provenance(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-COST"]["actual"] = None
    del bundle["verdicts"]["REQ-COST"]["provenance"]
    assert _score(bundle_copy, bundle, recomputed)["provenance_completeness"] == 1.0


# --- board -------------------------------------------------------------------------


def _board_path(bundle_dir: Path) -> Path:
    return bundle_dir / _bundle(bundle_dir)["board"]


def test_reference_board_parses(bundle_copy):
    ok, reason = rb.check_board(bundle_copy, _bundle(bundle_copy)["board"])
    assert ok, reason


def test_truncated_board_is_invalid(bundle_copy, recomputed):
    board = _board_path(bundle_copy)
    board.write_text(board.read_text()[:-200])
    assert _score(bundle_copy, _bundle(bundle_copy), recomputed)["board_valid"] == 0.0


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('(kicad_sch (version 20231120) (generator "eeschema"))', "kicad_pcb"),
        ("(kicad_pcb (generator pcbnew))", "version"),
        ("(kicad_pcb (version 20240108))", "generator"),
        ('(kicad_pcb (version 20240108) (generator "x")) trailing', "after"),
        ('(kicad_pcb (version 20240108) (generator "unterminated))', "parse"),
        ("not an s-expression", "kicad_pcb"),
    ],
)
def test_board_header_checks(bundle_copy, text, reason):
    board = _board_path(bundle_copy)
    board.write_text(text)
    ok, why = rb.check_board(bundle_copy, _bundle(bundle_copy)["board"])
    assert not ok
    assert reason in why


@pytest.mark.parametrize(
    ("rel", "reason"),
    [
        ("board/missing.kicad_pcb", "not found"),
        ("../outside.kicad_pcb", "inside"),
        ("/etc/hosts", "inside"),
        ("element/design.yaml", ".kicad_pcb"),
    ],
)
def test_board_path_checks(bundle_copy, rel, reason):
    ok, why = rb.check_board(bundle_copy, rel)
    assert not ok
    assert reason in why


# --- malformed bundles ---------------------------------------------------------------


def test_unknown_bundle_keys_rejected(bundle_copy):
    bundle = _bundle(bundle_copy)
    bundle["geometry"] = {}
    (bundle_copy / "bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="unknown keys"):
        rb.load_bundle(bundle_copy / "bundle.json")


def test_bad_status_rejected(bundle_copy):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-SLL"]["status"] = "probably"
    (bundle_copy / "bundle.json").write_text(json.dumps(bundle))
    with pytest.raises(ValueError, match="status"):
        rb.load_bundle(bundle_copy / "bundle.json")


def test_element_spec_outside_the_bundle_rejected(spec, bundle_copy):
    bundle = _bundle(bundle_copy)
    bundle["element_spec"] = "../../design.yaml"
    with pytest.raises(ValueError, match="inside"):
        rb.recompute(spec, bundle_copy, bundle)
