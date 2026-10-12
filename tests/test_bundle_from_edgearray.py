"""scripts/bundle_from_edgearray.py: hero-run outputs on disk -> t4-001 bundle.

The fixture under tests/fixtures/edgearray_hero is a trimmed copy of a real
`edgearray workflow hero` data directory (head, revision, and the architecture,
element, board and verification artifacts). The committed reference bundle in
tasks/t4-001-requirements-to-board/reference/edgearray_hero/ was produced from
it, so a test can regenerate that bundle and compare. These tests need only
PyYAML: they do not import EdgeArray or the scoring extras.
"""

from __future__ import annotations

import ast
import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "bundle_from_edgearray.py"
FIXTURE = REPO / "tests" / "fixtures" / "edgearray_hero"
COMMITTED = (
    REPO / "tasks" / "t4-001-requirements-to-board" / "reference" / "edgearray_hero" / "submission"
)
PROJECT = "aedl-hero"

SYSTEM_IDS = {
    "REQ-CLEARSKY-AGREE",
    "REQ-COST",
    "REQ-GAIN-XCHECK",
    "REQ-GRATING",
    "REQ-INDEP-LINK",
    "REQ-LINK-MARGIN",
    "REQ-POWER",
    "REQ-SLL",
}


@pytest.fixture(scope="module")
def mod():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture(scope="module")
def converted(mod, tmp_path_factory):
    out = tmp_path_factory.mktemp("hero") / "submission"
    mod["convert"](FIXTURE, PROJECT, out)
    return out


def _bundle(path: Path) -> dict:
    return json.loads((path / "bundle.json").read_text())


@pytest.fixture()
def fixture_copy(tmp_path):
    dest = tmp_path / "data"
    shutil.copytree(FIXTURE, dest)
    return dest


def _artifact(data: Path, kind_dir: str, name: str) -> Path:
    (path,) = (data / "projects" / PROJECT / "artifacts").glob(f"*/{kind_dir}*/{name}")
    return path


def test_script_does_not_import_edgearray():
    tree = ast.parse(SCRIPT.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not {m for m in imported if "edgearray" in m.lower()}
    assert "aedl" not in imported, "the converter should work on a bare checkout"


def test_regenerates_the_committed_reference_bundle(converted):
    assert COMMITTED.is_dir()
    committed = sorted(p.relative_to(COMMITTED) for p in COMMITTED.rglob("*") if p.is_file())
    produced = sorted(p.relative_to(converted) for p in converted.rglob("*") if p.is_file())
    assert produced == committed
    for rel in committed:
        assert (converted / rel).read_bytes() == (COMMITTED / rel).read_bytes(), rel


def test_architecture_follows_the_hero_payload(converted):
    arch = _bundle(converted)["architecture"]
    assert arch["array"] == {
        "nx": 16,
        "ny": 16,
        "dx_lambda": 0.5,
        "dy_lambda": 0.5,
        "taper_type": "taylor",
        "taper_sll_db": -25.0,
        "phase_bits": 6,
    }
    # 0.25 is class B in t4-001's parts table; 0.3 W/elem is the hero's assumption.
    assert arch["rf"] == {"tx_power_w_per_elem": 0.3, "pa_class": "B"}
    # The sub-array size comes from the hero's tiling rule R6.
    assert (arch["digital"]["subarray_nx"], arch["digital"]["subarray_ny"]) == (8, 8)


def test_fields_the_hero_does_not_choose_are_listed_as_filled_in(converted):
    record = json.loads((converted / "conversion.json").read_text())
    text = " ".join(record["filled_in"])
    assert "phase_bits" in text
    assert "digitization_level" in text
    assert "adc_enob" in text


def test_verdicts_cover_every_requirement(converted):
    verdicts = _bundle(converted)["verdicts"]
    assert set(verdicts) == SYSTEM_IDS | {"ELEM-FOOTPRINT", "ELEM-S11-IN-BAND"}
    for v in verdicts.values():
        assert set(v) == {"status", "actual", "unit", "provenance"}


def test_verified_verdicts_carry_numbers_and_engine_provenance(converted):
    bundle = _bundle(converted)
    report_id = json.loads((converted / "conversion.json").read_text())["verification_report"]
    sll = bundle["verdicts"]["REQ-SLL"]
    assert sll["status"] == "pass"
    assert sll["actual"] == pytest.approx(-22.3183296558838)
    assert sll["unit"] == "dB"
    assert sll["provenance"] == {
        "artifact_hash": "sha256:" + report_id,
        "capability": "edgearray-array-envelope@1.0.0",
        "origin": "assumed",
        "fidelity": "analytic",
    }
    assert bundle["verdicts"]["REQ-POWER"]["unit"] == "W"
    assert bundle["verdicts"]["REQ-POWER"]["actual"] == pytest.approx(368.64)


def test_verdicts_resting_on_assumed_inputs_are_labelled_assumed(converted):
    """The hero's report warns which verdicts were computed on assumed inputs
    (the element pattern exponent for the array verdicts; transmit power, PA
    efficiency and others for power). All four verified verdicts are named."""
    verdicts = _bundle(converted)["verdicts"]
    verified = {rid for rid, v in verdicts.items() if v["provenance"]}
    assert verified == {"REQ-GAIN-XCHECK", "REQ-GRATING", "REQ-SLL", "REQ-POWER"}
    for rid in verified:
        assert verdicts[rid]["provenance"]["origin"] == "assumed", rid
    record = json.loads((converted / "conversion.json").read_text())
    assert record["assumed_verdicts"] == sorted(verified)


def test_a_verdict_not_named_in_an_assumed_inputs_warning_stays_derived(
    mod, fixture_copy, tmp_path
):
    payload = _artifact(fixture_copy, "e9", "payload.json")
    doc = json.loads(payload.read_text())
    doc["warnings"] = [w for w in doc["warnings"] if not w.startswith("REQ-POWER: checked by")]
    payload.write_text(json.dumps(doc))
    out = tmp_path / "out"
    mod["convert"](fixture_copy, PROJECT, out)
    verdicts = _bundle(out)["verdicts"]
    assert verdicts["REQ-POWER"]["provenance"]["origin"] == "derived"
    assert verdicts["REQ-SLL"]["provenance"]["origin"] == "assumed"


def test_empty_unit_becomes_null(converted):
    grating = _bundle(converted)["verdicts"]["REQ-GRATING"]
    assert grating["unit"] is None
    assert grating["actual"] == pytest.approx(0.03589838486224539)


def test_unverified_verdicts_become_unknown_without_a_number(converted):
    verdicts = _bundle(converted)["verdicts"]
    for rid in ("REQ-COST", "REQ-INDEP-LINK", "REQ-LINK-MARGIN", "REQ-CLEARSKY-AGREE"):
        assert verdicts[rid]["status"] == "unknown", rid
        assert verdicts[rid]["actual"] is None, rid
        assert verdicts[rid]["provenance"] is None, rid
    assert "unit_cost_usd" not in json.dumps(verdicts)


def test_element_requirements_are_unknown_because_the_hero_did_not_judge_them(converted):
    verdicts = _bundle(converted)["verdicts"]
    for rid in ("ELEM-FOOTPRINT", "ELEM-S11-IN-BAND"):
        assert verdicts[rid] == {
            "status": "unknown",
            "actual": None,
            "unit": None,
            "provenance": None,
        }


def test_board_is_copied_byte_for_byte_with_its_hash(converted):
    bundle = _bundle(converted)
    shipped = (converted / bundle["board"]).read_bytes()
    (source,) = (FIXTURE / "projects" / PROJECT / "artifacts").glob("11/*/hero_patch.kicad_pcb")
    assert shipped == source.read_bytes()
    assert bundle["board_provenance"]["artifact_hash"] == (
        "sha256:" + hashlib.sha256(shipped).hexdigest()
    )
    assert bundle["board_provenance"]["capability"] == (
        "edgearray-workflows/hero@1.0.0 via antenna-cad 0.2.0"
    )


def test_element_spec_is_a_designspec_without_geometry(converted):
    import yaml

    doc = yaml.safe_load((converted / _bundle(converted)["element_spec"]).read_text())
    assert set(doc) == {"design", "requirements", "element", "acceptance"}
    assert doc["design"] == {"name": "hero_patch"}
    assert doc["requirements"]["substrate"] == "RO4350B"
    assert doc["requirements"]["impedance"] == "50.0 ohm"
    assert doc["element"] == {"type": "rectangular_patch"}


def test_bundle_has_only_the_schema_keys(converted):
    assert set(_bundle(converted)) == {
        "schema",
        "architecture",
        "element_spec",
        "board",
        "board_provenance",
        "verdicts",
    }


def test_main_prints_a_summary_and_notes_on_stderr(mod, tmp_path, capsys):
    code = mod["main"](["--data-dir", str(FIXTURE), "--project", PROJECT, "--out", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 0
    assert "'pass': 4" in captured.out
    assert "'unknown': 6" in captured.out
    assert "phase_bits" in captured.err


def test_missing_project_is_an_error(mod, tmp_path, capsys):
    code = mod["main"](["--data-dir", str(tmp_path), "--project", "nope", "--out", str(tmp_path)])
    assert code == 1
    assert "no project directory" in capsys.readouterr().err


def test_unmatched_pa_efficiency_is_an_error(mod, fixture_copy, tmp_path):
    payload = _artifact(fixture_copy, "6c", "payload.json")
    doc = json.loads(payload.read_text())
    doc["study"]["rf"]["pa_efficiency"] = 0.31
    payload.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="pa_efficiency"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_board_not_built_from_the_element_is_an_error(mod, fixture_copy, tmp_path):
    payload = _artifact(fixture_copy, "11", "payload.json")
    doc = json.loads(payload.read_text())
    doc["design_hash"] = "0" * 64
    payload.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="design_hash"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


@pytest.mark.parametrize("name", ["../payload.json", "sub/hero_patch.kicad_pcb", "/etc/hosts"])
def test_board_file_outside_the_artifact_is_an_error(mod, fixture_copy, tmp_path, name):
    for rel in ("payload.json", "manifest.json"):
        path = _artifact(fixture_copy, "11", rel)
        doc = json.loads(path.read_text())
        if rel == "payload.json":
            doc["kicad_pcb_file"] = name
        else:
            doc["files"].append(name)
        path.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="board file"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


@pytest.mark.parametrize("kind_dir,name", [("ff", "payload.json"), ("ff", "manifest.json")])
def test_wrong_type_json_is_a_conversion_error(mod, fixture_copy, tmp_path, kind_dir, name):
    _artifact(fixture_copy, kind_dir, name).write_text("[]")
    with pytest.raises(mod["ConversionError"], match="expected an object"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


@pytest.mark.parametrize("field,value", [("design_spec", "a string"), ("design_spec", None)])
def test_non_object_design_spec_is_a_conversion_error(mod, fixture_copy, tmp_path, field, value):
    payload = _artifact(fixture_copy, "ff", "payload.json")
    doc = json.loads(payload.read_text())
    doc[field] = value
    payload.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="design_spec"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_non_object_head_and_revision_nodes_are_conversion_errors(mod, fixture_copy, tmp_path):
    head = fixture_copy / "projects" / PROJECT / "design" / "head.json"
    head.write_text("[]")
    with pytest.raises(mod["ConversionError"], match="expected an object"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_verdicts_that_are_not_objects_are_a_conversion_error(mod, fixture_copy, tmp_path):
    payload = _artifact(fixture_copy, "e9", "payload.json")
    doc = json.loads(payload.read_text())
    doc["verdicts"] = ["REQ-SLL"]
    payload.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="list of objects"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_symlinked_artifact_directory_is_refused(mod, fixture_copy, tmp_path):
    folder = _artifact(fixture_copy, "ff", "payload.json").parent
    moved = tmp_path / "elsewhere"
    folder.rename(moved)
    folder.symlink_to(moved)
    with pytest.raises(mod["ConversionError"], match="not found"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def _reword(fixture_copy: Path, old_prefix: str, new_text: str) -> None:
    payload = _artifact(fixture_copy, "e9", "payload.json")
    doc = json.loads(payload.read_text())
    doc["warnings"] = [new_text if w.startswith(old_prefix) else w for w in doc["warnings"]]
    payload.write_text(json.dumps(doc))


def test_a_reworded_assumption_warning_is_reported_and_still_labels_the_verdict(
    mod, fixture_copy, tmp_path
):
    """The hero changes its wording. REQ-POWER's id is still in the text, so it
    stays labelled assumed, and the warning is surfaced for a human."""
    reworded = "REQ-POWER was computed from assumed values for transmit power"
    _reword(fixture_copy, "REQ-POWER: checked by", reworded)
    out = tmp_path / "out"
    mod["convert"](fixture_copy, PROJECT, out)
    assert _bundle(out)["verdicts"]["REQ-POWER"]["provenance"]["origin"] == "assumed"
    record = json.loads((out / "conversion.json").read_text())
    assert record["unmatched_assumption_warnings"] == [reworded]
    assert record["assumed_verdicts"] == sorted(
        ["REQ-GAIN-XCHECK", "REQ-GRATING", "REQ-POWER", "REQ-SLL"]
    )


def test_an_assumption_warning_without_ids_is_recorded_and_printed(
    mod, fixture_copy, tmp_path, capsys
):
    reworded = "all system numbers rest on assumed inputs"
    _reword(fixture_copy, "REQ-POWER: checked by", reworded)
    out = tmp_path / "out"
    code = mod["main"](["--data-dir", str(fixture_copy), "--project", PROJECT, "--out", str(out)])
    assert code == 0
    assert "unparsed assumption warning" in capsys.readouterr().err
    record = json.loads((out / "conversion.json").read_text())
    assert record["unmatched_assumption_warnings"] == [reworded]
    # The ids cannot be parsed from this one, so REQ-POWER falls back to derived.
    assert _bundle(out)["verdicts"]["REQ-POWER"]["provenance"]["origin"] == "derived"


def test_the_committed_hero_run_has_no_unmatched_assumption_warnings(converted):
    record = json.loads((converted / "conversion.json").read_text())
    assert record["unmatched_assumption_warnings"] == []


def test_unmapped_verdict_status_is_an_error(mod, fixture_copy, tmp_path):
    payload = _artifact(fixture_copy, "e9", "payload.json")
    doc = json.loads(payload.read_text())
    doc["verdicts"][0]["status"] = "waived"
    payload.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="unmapped status"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_verified_verdict_without_an_engine_is_an_error(mod, fixture_copy, tmp_path):
    payload = _artifact(fixture_copy, "e9", "payload.json")
    doc = json.loads(payload.read_text())
    for verdict in doc["verdicts"]:
        if verdict["id"] == "REQ-SLL":
            verdict["engine"] = None
    payload.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match="REQ-SLL"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_node_of_the_wrong_kind_is_an_error(mod, fixture_copy, tmp_path):
    manifest = _artifact(fixture_copy, "ff", "manifest.json")
    doc = json.loads(manifest.read_text())
    doc["kind"] = "board.layout"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(mod["ConversionError"], match=r"expected 'element\.design'"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


# --- final review fixes ------------------------------------------------------------


def _hero_verdicts(fixture_copy: Path, mutate) -> None:
    payload = _artifact(fixture_copy, "e9", "payload.json")
    doc = json.loads(payload.read_text())
    mutate(doc)
    payload.write_text(json.dumps(doc))


def test_element_verdicts_in_the_hero_report_are_not_overwritten(mod, fixture_copy, tmp_path):
    """A report that does judge ELEM-FOOTPRINT keeps its verdict; only the
    requirements the report omits are filled in as unknown."""

    def add(doc):
        doc["verdicts"].append(
            {
                "id": "ELEM-FOOTPRINT",
                "status": "pass",
                "actual": 5.6,
                "unit": "mm",
                "engine": "edgearray-element@1.0.0",
                "fidelity": "analytic",
            }
        )

    _hero_verdicts(fixture_copy, add)
    out = tmp_path / "out"
    mod["convert"](fixture_copy, PROJECT, out)
    verdicts = _bundle(out)["verdicts"]
    assert verdicts["ELEM-FOOTPRINT"]["status"] == "pass"
    assert verdicts["ELEM-FOOTPRINT"]["actual"] == 5.6
    assert verdicts["ELEM-FOOTPRINT"]["provenance"]["capability"] == "edgearray-element@1.0.0"
    assert verdicts["ELEM-S11-IN-BAND"]["status"] == "unknown"
    record = json.loads((out / "conversion.json").read_text())
    assert "ELEM-FOOTPRINT" not in record["element_requirements"]
    assert "ELEM-S11-IN-BAND" in record["element_requirements"]


@pytest.mark.parametrize("name", ["manifest.json", "payload.json"])
def test_symlinked_manifest_or_payload_is_refused(mod, fixture_copy, tmp_path, name):
    target = _artifact(fixture_copy, "ff", name)
    moved = tmp_path / f"elsewhere-{name}"
    target.rename(moved)
    target.symlink_to(moved)
    with pytest.raises(mod["ConversionError"], match="symlink"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_symlinked_artifact_shard_directory_is_refused(mod, fixture_copy, tmp_path):
    shard = _artifact(fixture_copy, "ff", "payload.json").parent.parent
    moved = tmp_path / "shard"
    shard.rename(moved)
    shard.symlink_to(moved)
    with pytest.raises(mod["ConversionError"], match=r"symlink|not found"):
        mod["convert"](fixture_copy, PROJECT, tmp_path / "out")


def test_assumption_warning_ids_match_whole_ids_only(mod):
    """The warning names REQ-GAIN-XCHECK; REQ-GAIN is only a prefix of it."""
    ids, unmatched = mod["assumed_verdict_ids"](
        ["REQ-GAIN-XCHECK was reworded to rest on assumed inputs"],
        ["REQ-GAIN", "REQ-GAIN-XCHECK", "REQ-SLL"],
    )
    assert ids == {"REQ-GAIN-XCHECK"}
    assert len(unmatched) == 1


def test_assumption_warning_id_at_the_end_or_before_punctuation_matches(mod):
    ids, _ = mod["assumed_verdict_ids"](
        ["rests on assumed inputs: REQ-GAIN-XCHECK, (REQ-SLL)."],
        ["REQ-GAIN", "REQ-GAIN-XCHECK", "REQ-SLL"],
    )
    assert ids == {"REQ-GAIN-XCHECK", "REQ-SLL"}


def _siblings(path: Path) -> list[str]:
    return sorted(p.name for p in path.parent.iterdir())


def test_failed_conversion_leaves_no_output(mod, fixture_copy, tmp_path):
    """The verdict error comes after the element and board are staged."""
    _hero_verdicts(fixture_copy, lambda doc: doc["verdicts"][0].update(status="waived"))
    out = tmp_path / "work" / "out"
    out.parent.mkdir()
    with pytest.raises(mod["ConversionError"], match="unmapped status"):
        mod["convert"](fixture_copy, PROJECT, out)
    assert not out.exists()
    assert _siblings(out) == []


def test_failed_conversion_leaves_an_existing_output_directory_untouched(
    mod, fixture_copy, tmp_path
):
    _hero_verdicts(fixture_copy, lambda doc: doc["verdicts"][0].update(status="waived"))
    out = tmp_path / "work" / "out"
    out.mkdir(parents=True)
    (out / "keep.txt").write_text("previous")
    with pytest.raises(mod["ConversionError"]):
        mod["convert"](fixture_copy, PROJECT, out)
    assert sorted(p.name for p in out.iterdir()) == ["keep.txt"]
    assert _siblings(out) == ["out"]


def test_main_turns_any_exception_into_a_clean_error(mod, fixture_copy, tmp_path, capsys):
    """A verdict with no id raises KeyError deep in the conversion."""
    _hero_verdicts(fixture_copy, lambda doc: doc["verdicts"][0].pop("id"))
    out = tmp_path / "work" / "out"
    out.parent.mkdir()
    code = mod["main"](["--data-dir", str(fixture_copy), "--project", PROJECT, "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.startswith("error:")
    assert "Traceback" not in captured.err
    assert _siblings(out) == []


def test_successful_conversion_into_an_existing_directory_still_works(mod, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    mod["convert"](FIXTURE, PROJECT, out)
    assert (out / "bundle.json").is_file()
    assert _siblings(out) == ["out"]
