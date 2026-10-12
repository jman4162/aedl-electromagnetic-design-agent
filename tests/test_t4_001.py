"""Evaluator sanity for t4-001: the (synthetic) reference bundle passes, and
each requirement catches a specific defect in a bundle.

The expensive part, recomputing every requirement from the submitted
architecture and element spec, runs a handful of times. Bundle-only defects
(verdicts, provenance, board) are scored against one cached recomputation.
"""

from __future__ import annotations

import copy
import hashlib
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
HERO = TASKS / "t4-001-requirements-to-board" / "reference" / "edgearray_hero" / "submission"
TOLERANCES = find_task(TASKS, "t4-001").evaluator_params["actual_tolerance_by_suffix"]

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
ELEMENT_IDS = {"ELEM-S11-IN-BAND", "ELEM-FOOTPRINT"}
N = len(SYSTEM_IDS | ELEMENT_IDS)  # 10


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
    metrics, _ = rb.score(path, bundle, recomputed, tolerances=TOLERANCES)
    return metrics


def _details(path: Path, bundle: dict, recomputed) -> dict:
    _, details = rb.score(path, bundle, recomputed, tolerances=TOLERANCES)
    return details


def _save(path: Path, bundle: dict) -> None:
    (path / "bundle.json").write_text(json.dumps(bundle))


def _failures(result):
    return {r.requirement_id for r in result.requirements if not r.passed}


# --- task wiring -------------------------------------------------------------


def test_task_loads_through_the_registry(spec):
    assert spec.tier == 4
    assert spec.evaluator == "requirements_bundle"
    assert {r.metric for r in spec.requirements} == {
        "verdict_agreement",
        "actual_agreement",
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
    assert set(recomputed.requirements) == SYSTEM_IDS | ELEMENT_IDS


# --- reference -----------------------------------------------------------------


def test_reference_passes(reference_result):
    assert reference_result.passed, _failures(reference_result)


def test_reference_scores_on_recompute(reference_result):
    metrics = reference_result.info["metrics"]
    # Nine certified passes; the in-band S11 is unknown at the analytic tier.
    assert metrics["recomputed_pass_count"] == 9
    assert metrics["verdict_agreement"] == 1.0
    assert metrics["actual_agreement"] == 1.0
    assert metrics["provenance_completeness"] == 1.0
    assert metrics["board_valid"] == 1.0


def test_pass_count_is_reported_not_gated(spec, reference_result):
    assert "recomputed_pass_count" not in {r.metric for r in spec.requirements}
    assert reference_result.info["score"] == {
        "recomputed_passes": 9,
        "unknown_at_this_tier": 1,
        "of": N,
    }


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
    assert metrics["verdict_agreement"] == pytest.approx(9 / N)
    assert metrics["verdict_agreement"] >= 0.9


def test_two_wrong_verdicts_fail_agreement(bundle_copy, recomputed):
    bundle = _flip(_bundle(bundle_copy), "REQ-POWER", "ELEM-FOOTPRINT")
    assert _score(bundle_copy, bundle, recomputed)["verdict_agreement"] == pytest.approx(8 / N)


def test_missing_verdicts_count_as_disagreement(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    del bundle["verdicts"]["REQ-SLL"]
    del bundle["verdicts"]["ELEM-FOOTPRINT"]
    assert _score(bundle_copy, bundle, recomputed)["verdict_agreement"] == pytest.approx(8 / N)


def test_changed_architecture_with_stale_verdicts_fails(spec, bundle_copy):
    """More power per element busts the prime-power ceiling and a uniform
    taper busts the sidelobe spec. A bundle that still claims both pass is
    caught by recomputation, not by anything it says about itself."""
    bundle = _bundle(bundle_copy)
    bundle["architecture"]["rf"]["tx_power_w_per_elem"] = 0.5
    bundle["architecture"]["array"]["taper_type"] = "uniform"
    _save(bundle_copy, bundle)
    result = get_evaluator(spec.evaluator)(spec, bundle_copy / "bundle.json")
    assert {"verdict-agreement", "actual-agreement"} <= _failures(result)
    rows = result.info["requirements"]
    assert rows["REQ-POWER"]["recomputed_status"] == "fail"
    assert rows["REQ-SLL"]["recomputed_status"] == "fail"
    assert result.info["metrics"]["recomputed_pass_count"] <= 7


# --- honest unknown ------------------------------------------------------------------


def test_s11_band_coverage_is_unknown_at_the_analytic_tier(recomputed):
    rec = recomputed.requirements["ELEM-S11-IN-BAND"]
    assert rec.status == "unknown"
    assert rec.screening_status == "pass"
    assert rec.screening_fidelity == "analytic"
    assert recomputed.requirements["ELEM-FOOTPRINT"].status == "pass"


def _s11(bundle_dir: Path, recomputed, status: str, fidelity: str = "analytic") -> dict:
    bundle = _bundle(bundle_dir)
    verdict = bundle["verdicts"]["ELEM-S11-IN-BAND"]
    verdict["status"] = status
    verdict["provenance"]["fidelity"] = fidelity
    return _details(bundle_dir, bundle, recomputed)["requirements"]["ELEM-S11-IN-BAND"]


def test_unknown_agrees_with_unknown(bundle_copy, recomputed):
    assert _s11(bundle_copy, recomputed, "unknown")["agrees"]


def test_screening_verdict_labelled_analytic_agrees(bundle_copy, recomputed):
    assert _s11(bundle_copy, recomputed, "pass", "analytic")["agrees"]


def test_screening_verdict_claimed_as_full_wave_disagrees(bundle_copy, recomputed):
    assert not _s11(bundle_copy, recomputed, "pass", "fullwave_open")["agrees"]


def test_verdict_contradicting_the_screen_disagrees(bundle_copy, recomputed):
    assert not _s11(bundle_copy, recomputed, "fail", "analytic")["agrees"]


def test_unknown_on_a_certifiable_requirement_disagrees(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-POWER"]["status"] = "unknown"
    rows = _details(bundle_copy, bundle, recomputed)["requirements"]
    assert not rows["REQ-POWER"]["agrees"]


# --- actual agreement ----------------------------------------------------------------


def test_one_number_off_is_tolerated_two_are_not(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-LINK-MARGIN"]["actual"] += 1.0  # 1 dB, tolerance 0.1 dB
    assert _score(bundle_copy, bundle, recomputed)["actual_agreement"] == pytest.approx(9 / N)
    bundle["verdicts"]["REQ-POWER"]["actual"] *= 1.05  # 5 %, tolerance 1 %
    assert _score(bundle_copy, bundle, recomputed)["actual_agreement"] == pytest.approx(8 / N)


def test_numbers_within_tolerance_agree(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-LINK-MARGIN"]["actual"] += 0.05
    bundle["verdicts"]["REQ-COST"]["actual"] *= 1.005
    assert _score(bundle_copy, bundle, recomputed)["actual_agreement"] == 1.0


def test_null_actual_agrees_only_on_an_unknown_requirement(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["ELEM-S11-IN-BAND"]["actual"] = None
    assert _score(bundle_copy, bundle, recomputed)["actual_agreement"] == 1.0
    bundle["verdicts"]["REQ-SLL"]["actual"] = None
    assert _score(bundle_copy, bundle, recomputed)["actual_agreement"] == pytest.approx(9 / N)


# --- provenance --------------------------------------------------------------------


def test_one_missing_provenance_record_fails_completeness(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    del bundle["verdicts"]["REQ-COST"]["provenance"]
    metrics = _score(bundle_copy, bundle, recomputed)
    assert metrics["provenance_completeness"] == pytest.approx(9 / N)
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


def test_pass_verdict_without_a_number_is_incomplete(bundle_copy, recomputed):
    """Nulling the number cannot shrink the denominator."""
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-COST"]["actual"] = None
    del bundle["verdicts"]["REQ-COST"]["provenance"]
    assert _score(bundle_copy, bundle, recomputed)["provenance_completeness"] == pytest.approx(
        9 / N
    )


def test_unknown_verdict_without_a_number_is_accounted_for(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    verdict = bundle["verdicts"]["ELEM-S11-IN-BAND"]
    verdict.update(status="unknown", actual=None)
    del verdict["provenance"]
    assert _score(bundle_copy, bundle, recomputed)["provenance_completeness"] == 1.0


def test_missing_verdict_is_incomplete_provenance(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    del bundle["verdicts"]["REQ-GRATING"]
    assert _score(bundle_copy, bundle, recomputed)["provenance_completeness"] == pytest.approx(
        9 / N
    )


# --- board -------------------------------------------------------------------------


def _board_path(bundle_dir: Path) -> Path:
    return bundle_dir / _bundle(bundle_dir)["board"]


def test_reference_board_parses(bundle_copy):
    ok, reason = rb.check_board(bundle_copy, _bundle(bundle_copy)["board"])
    assert ok, reason


def test_reference_board_matches_regeneration(bundle_copy, recomputed):
    details = _details(bundle_copy, _bundle(bundle_copy), recomputed)["board"]
    assert details["valid"], details
    assert details["shipped_sha256"] == recomputed.board_sha256


def test_truncated_board_is_invalid(bundle_copy, recomputed):
    board = _board_path(bundle_copy)
    board.write_text(board.read_text()[:-200])
    assert _score(bundle_copy, _bundle(bundle_copy), recomputed)["board_valid"] == 0.0


def _rehash_board(bundle_dir: Path) -> dict:
    """The bundle with board_provenance updated to the board file now on disk."""
    import hashlib

    bundle = _bundle(bundle_dir)
    digest = hashlib.sha256(_board_path(bundle_dir).read_bytes()).hexdigest()
    bundle["board_provenance"]["artifact_hash"] = "sha256:" + digest
    return bundle


def test_edited_board_with_honest_hash_fails_regeneration(bundle_copy, recomputed):
    """A well-formed board that is not the one the element spec produces."""
    board = _board_path(bundle_copy)
    board.write_text(board.read_text().replace('(paper "A4")', '(paper "A3")'))
    bundle = _rehash_board(bundle_copy)
    details = _details(bundle_copy, bundle, recomputed)["board"]
    assert not details["valid"]
    assert details["provenance"] == "ok"
    assert "antenna-cad" in details["regeneration"]


def test_board_hash_must_match_the_shipped_file(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    bundle["board_provenance"]["artifact_hash"] = "0" * 64
    details = _details(bundle_copy, bundle, recomputed)["board"]
    assert not details["valid"]
    assert "SHA-256" in details["provenance"]


def test_board_without_provenance_is_invalid(bundle_copy, recomputed):
    bundle = _bundle(bundle_copy)
    del bundle["board_provenance"]
    assert _score(bundle_copy, bundle, recomputed)["board_valid"] == 0.0


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


# --- the degenerate bundle -------------------------------------------------------------


def test_degenerate_bundle_fails(spec, bundle_copy, recomputed):
    """The review's shortcut: a one-line board, one fabricated provenance
    record, and statuses copied from a recompute with every number nulled.
    It may agree on verdicts; it must fail everything else."""
    import hashlib

    board = _board_path(bundle_copy)
    board.write_text('(kicad_pcb (version 20240108) (generator "x"))\n')
    bundle = _bundle(bundle_copy)
    bundle["board_provenance"] = {
        "artifact_hash": hashlib.sha256(board.read_bytes()).hexdigest(),
        "capability": "made.up",
        "origin": "derived",
        "fidelity": "analytic",
    }
    verdicts = {}
    for i, (rid, rec) in enumerate(sorted(recomputed.requirements.items())):
        verdicts[rid] = {"status": rec.status, "actual": None, "unit": None}
        if i == 0:
            verdicts[rid]["provenance"] = {
                "artifact_hash": "f" * 64,
                "capability": "made.up",
                "origin": "derived",
                "fidelity": "analytic",
            }
    bundle["verdicts"] = verdicts
    _save(bundle_copy, bundle)

    result = get_evaluator(spec.evaluator)(spec, bundle_copy / "bundle.json")
    assert not result.passed
    assert {"actual-agreement", "provenance", "board"} <= _failures(result)
    metrics = result.info["metrics"]
    assert metrics["provenance_completeness"] == pytest.approx(1 / N)  # the unknown S11 only
    assert metrics["actual_agreement"] == pytest.approx(1 / N)


# --- malformed bundles ---------------------------------------------------------------


def test_unknown_bundle_keys_rejected(bundle_copy):
    bundle = _bundle(bundle_copy)
    bundle["geometry"] = {}
    _save(bundle_copy, bundle)
    with pytest.raises(ValueError, match="unknown keys"):
        rb.load_bundle(bundle_copy / "bundle.json")


def test_bad_status_rejected(bundle_copy):
    bundle = _bundle(bundle_copy)
    bundle["verdicts"]["REQ-SLL"]["status"] = "probably"
    _save(bundle_copy, bundle)
    with pytest.raises(ValueError, match="status"):
        rb.load_bundle(bundle_copy / "bundle.json")


def test_element_spec_outside_the_bundle_rejected(spec, bundle_copy):
    bundle = _bundle(bundle_copy)
    bundle["element_spec"] = "../../design.yaml"
    with pytest.raises(ValueError, match="inside"):
        rb.recompute(spec, bundle_copy, bundle)


# --- EdgeArray hero bundle -------------------------------------------------------
#
# reference/edgearray_hero/submission is the output of `edgearray workflow hero`
# converted by scripts/bundle_from_edgearray.py. It does not pass: the hero
# leaves half the requirements unverified and computes two of its numbers from
# inputs the task's parts table overrides. These tests pin that outcome and its
# causes (docs/tasks/t4-001.md) so a change in either the evaluator or the
# workflow shows up here.


@pytest.fixture(scope="module")
def hero_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("hero") / "submission"
    shutil.copytree(HERO, dest)
    return dest


@pytest.fixture(scope="module")
def hero_result(spec, hero_dir):
    return get_evaluator(spec.evaluator)(spec, hero_dir / "bundle.json")


def test_hero_board_is_accepted_from_the_stored_ir(hero_dir, hero_result):
    """The hero emits its board from the IR file, whose polygons are rounded to
    six decimals, so the file differs from a direct emission in a few
    coordinates and matches the IR round trip."""
    board = hero_result.info["board"]
    assert board["valid"], board
    assert board["regeneration"] == "ok (emitted from the stored IR)"
    direct, via_ir = rb.regenerate_board_sha256s(hero_dir / "element" / "design.yaml")
    assert direct != via_ir
    assert board["shipped_sha256"] == via_ir


def test_synthetic_reference_board_still_matches_direct_emission(reference_result):
    assert reference_result.info["board"]["regeneration"] == "ok"


def test_ir_board_with_an_edit_still_fails_regeneration(hero_dir):
    bundle = _bundle(hero_dir)
    direct, via_ir = rb.regenerate_board_sha256s(hero_dir / "element" / "design.yaml")
    board = hero_dir / bundle["board"]
    edited = board.read_text().replace('(paper "A4")', '(paper "A3")')
    assert edited != board.read_text()
    digest = hashlib.sha256(edited.encode()).hexdigest()
    assert digest not in (direct, via_ir)
    scratch = hero_dir.parent / "edited"
    shutil.copytree(hero_dir, scratch)
    (scratch / bundle["board"]).write_text(edited)
    bundle["board_provenance"]["artifact_hash"] = "sha256:" + digest
    ok, details = rb.board_verdict(scratch, bundle, direct, also_accept_sha256=via_ir)
    assert not ok
    assert "antenna-cad" in details["regeneration"]


def test_hero_bundle_is_scored_not_passed(hero_result):
    metrics = hero_result.info["metrics"]
    assert not hero_result.passed
    assert metrics["board_valid"] == 1.0
    assert metrics["provenance_completeness"] == 1.0
    assert metrics["verdict_agreement"] == pytest.approx(0.5)
    assert metrics["actual_agreement"] == pytest.approx(0.3)
    assert _failures(hero_result) == {"verdict-agreement", "actual-agreement"}


def test_hero_verdict_disagreements_are_its_unverified_requirements(hero_result):
    """Five verdicts disagree. Four are requirements the hero reported as
    unverified that AEDL recomputes; the fifth is the element footprint, which
    the hero's model does not contain."""
    rows = hero_result.info["requirements"]
    disagree = {rid for rid, row in rows.items() if not row["agrees"]}
    assert disagree == {
        "REQ-CLEARSKY-AGREE",
        "REQ-COST",
        "REQ-INDEP-LINK",
        "REQ-LINK-MARGIN",
        "ELEM-FOOTPRINT",
    }
    assert all(rows[rid]["submitted_status"] == "unknown" for rid in disagree)
    assert all(rows[rid]["recomputed_status"] != "unknown" for rid in disagree)


def test_hero_numbers_differ_from_the_recompute_by_more_than_the_tolerance(hero_result):
    """REQ-SLL and REQ-POWER carry numbers whose verdicts agree and whose
    values do not. The hero searches two scan planes without element failures
    or phase quantization; its power omits the receive chain and the digital
    section that the parts table adds."""
    rows = hero_result.info["requirements"]
    assert rows["REQ-SLL"]["agrees"] and not rows["REQ-SLL"]["actual_agrees"]
    assert rows["REQ-POWER"]["agrees"] and not rows["REQ-POWER"]["actual_agrees"]
    assert rows["REQ-SLL"]["actual"] - rows["REQ-SLL"]["submitted_actual"] > 2.0
    assert rows["REQ-POWER"]["actual"] / rows["REQ-POWER"]["submitted_actual"] > 1.1
    # The two numbers that do agree are the ones computed the same way.
    assert rows["REQ-GRATING"]["actual_agrees"]
    assert rows["REQ-GAIN-XCHECK"]["actual_agrees"]


def test_hero_element_fails_the_band_screen(hero_result):
    """ELEM-S11-IN-BAND is unknown at this tier, but the analytic screen it
    keeps says the hero's 28.0 GHz, 50 ohm patch misses -10 dB over the band."""
    row = hero_result.info["requirements"]["ELEM-S11-IN-BAND"]
    assert row["recomputed_status"] == "unknown"
    assert row["screening_status"] == "fail"
    assert row["actual"] > -6.0
