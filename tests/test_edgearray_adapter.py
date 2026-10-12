"""The EdgeArray adapter, run against a fake ``edgearray`` executable.

The real CLI (``edgearray agent run``) lives in a separate repository. The
fake honours the same arguments, writes a canned ``submission.json`` and
``transcript.jsonl`` into ``--workspace``, and lays out a minimal data
directory under ``--data-dir`` (``projects/<project>/...``), which is all the
adapter relies on. EdgeArray cannot write into the AEDL workspace, so the
adapter builds the task deliverable from that data directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from aedl.harness import adapter_names, get_adapter, run_task
from aedl.harness import workspace as ws
from aedl.harness.adapters.edgearray import ENV_BIN, EdgeArrayAdapter
from aedl.spec import find_task

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
HERO_FIXTURE = REPO / "tests" / "fixtures" / "edgearray_hero" / "projects" / "aedl-hero"
HERO_COMMITTED = (
    TASKS / "t4-001-requirements-to-board" / "reference" / "edgearray_hero" / "submission"
)
T1_REFERENCE = TASKS / "t1-001-patch-28ghz-ro4350b" / "reference" / "design.yaml"

FAKE = """#!{python}
import json, pathlib, shutil, sys, time

argv = sys.argv[1:]
assert argv[:2] == ["agent", "run"], argv
opts = {{}}
i = 2
while i < len(argv):
    if argv[i] == "--approve-all":
        opts["approve_all"] = True
        i += 1
    else:
        opts[argv[i][2:]] = argv[i + 1]
        i += 2
ws = pathlib.Path(opts["workspace"])
ws.mkdir(parents=True, exist_ok=True)
assert pathlib.Path(opts["brief"]).is_file()
data_dir = pathlib.Path(opts["data-dir"])
# Every run starts from an empty data directory of its own.
assert not data_dir.exists(), data_dir
(ws / "argv.json").write_text(json.dumps(argv))
if {data_fixture!r}:
    shutil.copytree({data_fixture!r}, data_dir / "projects" / opts["project"], symlinks=True)
llm = opts.get("baseline", "b4") != "b0"
doc = {{
    "final_text": "done",
    "model": opts.get("model"),
    "usage": {usage!r} if llm else {{"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}},
    "tool_calls": [
        {{"name": "edgearray.pattern.compute", "ok": True, "ms": 12}},
        {{"name": "edgearray.verify.run", "ok": True, "ms": 40}},
    ],
    "verification_report_hash": "ab" * 32,
}}
if {write_submission!r}:
    (ws / "submission.json").write_text(json.dumps(doc))
(ws / "transcript.jsonl").write_text(json.dumps({{"type": "tool_call"}}) + "\\n")
{extra_code}
"""

USAGE = {
    "input_tokens": 1200,
    "output_tokens": 340,
    "cache_read_tokens": 700,
    "cache_write_tokens": 90,
    "cost_usd": 0.0125,
    "num_turns": 4,
}


def make_fake(
    tmp_path: Path,
    data_fixture: Path | None = None,
    write_submission: bool = True,
    usage: dict | None = None,
    extra_code: str = "",
) -> Path:
    fake = tmp_path / "bin" / "edgearray"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(
        FAKE.format(
            python=sys.executable,
            data_fixture=str(data_fixture) if data_fixture else "",
            write_submission=write_submission,
            usage=usage or USAGE,
            extra_code=extra_code,
        )
    )
    fake.chmod(0o755)
    return fake


def make_project(
    root: Path,
    elements: list[tuple[Path, str]],
    head: int | None,
    head_has_element: bool = True,
) -> Path:
    """A minimal project directory: element.design artifacts and a design head.

    ``elements`` is (DesignSpec YAML file, manifest created_at) per artifact.
    ``head`` indexes the artifact the head revision's element node points at.
    """
    ids = []
    for src, created_at in elements:
        spec = yaml.safe_load(src.read_text())
        artifact_id = hashlib.sha256(src.read_bytes() + created_at.encode()).hexdigest()
        folder = root / "artifacts" / artifact_id[:2] / artifact_id
        folder.mkdir(parents=True)
        (folder / "manifest.json").write_text(
            json.dumps({"kind": "element.design", "created_at": created_at, "files": []})
        )
        (folder / "payload.json").write_text(
            json.dumps(
                {
                    "kind": "element.design",
                    "ir_hash": "0" * 64,
                    "design_spec": {
                        "name": spec["design"]["name"],
                        "problem": spec["requirements"],
                        "element_type": spec["element"]["type"],
                    },
                }
            )
        )
        ids.append(artifact_id)
    if head is not None:
        rev = hashlib.sha256(b"rev" + ids[head].encode()).hexdigest()
        (root / "design" / "revisions").mkdir(parents=True)
        nodes = {"element": {"artifact": ids[head]}} if head_has_element else {}
        (root / "design" / "revisions" / f"{rev}.json").write_text(
            json.dumps({"rev": rev, "nodes": nodes})
        )
        (root / "design" / "head.json").write_text(json.dumps({"rev": rev}))
    return root


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t2-001")


@pytest.fixture(scope="module")
def t1():
    return find_task(TASKS, "t1-001")


@pytest.fixture(scope="module")
def t4():
    return find_task(TASKS, "t4-001")


@pytest.fixture()
def reference_project(tmp_path):
    return make_project(tmp_path / "ref-project", [(T1_REFERENCE, "2026-10-01T00:00:00Z")], head=0)


def _run(spec, adapter, tmp_path):
    return run_task(
        spec,
        adapter,
        runs_dir=tmp_path / "runs",
        isolation="tmpdir",
        timeout_s=60,
        instrumented=False,
    )


def _direct(spec, adapter, tmp_path, name="w", timeout_s=60):
    """Run the adapter in a fresh workspace without scoring anything."""
    parent = tmp_path / name
    parent.mkdir()
    work = ws.materialize(spec, isolation="inplace", parent=parent)
    info = adapter.run(work, dict(os.environ), timeout_s)
    return work, info


def test_adapter_is_registered():
    assert "edgearray" in adapter_names()


def test_command_line(spec, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_BIN, raising=False)
    work = ws.materialize(spec, isolation="inplace", parent=tmp_path)
    adapter = EdgeArrayAdapter(model="ollama/qwen3.5:9b", baseline="b2", budget_s=600)
    cmd = adapter.build_command(work)
    token = cmd[cmd.index("--project") + 1].removeprefix("aedl-t2-001-")
    assert len(token) == 8
    assert cmd == [
        "edgearray",
        "agent",
        "run",
        "--brief",
        str(work / "BRIEF.md"),
        "--workspace",
        str(work / "edgearray"),
        "--project",
        f"aedl-t2-001-{token}",
        "--data-dir",
        str(work / "edgearray" / f"data-{token}"),
        "--baseline",
        "b2",
        "--model",
        "ollama/qwen3.5:9b",
        "--budget-s",
        "600",
        "--approve-all",
    ]


def test_executable_comes_from_the_environment(spec, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, "/opt/edgearray/bin/edgearray")
    work = ws.materialize(spec, isolation="inplace", parent=tmp_path)
    cmd = get_adapter("edgearray", model=None, baseline="b0", approve_all=False).build_command(work)
    assert cmd[0] == "/opt/edgearray/bin/edgearray"
    assert "--model" not in cmd
    assert "--approve-all" not in cmd


def test_unknown_baseline_rejected():
    with pytest.raises(ValueError, match="baseline"):
        EdgeArrayAdapter(baseline="b9")


def test_an_explicit_project_is_kept(spec, tmp_path):
    work = ws.materialize(spec, isolation="inplace", parent=tmp_path)
    cmd = EdgeArrayAdapter(project="mine").build_command(work)
    assert cmd[cmd.index("--project") + 1] == "mine"


# --- per-run isolation -------------------------------------------------------------


def test_each_run_gets_its_own_data_dir_and_project(t1, reference_project, tmp_path, monkeypatch):
    """The fake fails if its data dir already exists, so a shared directory
    between two runs of one adapter instance would show up as an agent error."""
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_project)))
    adapter = get_adapter("edgearray", baseline="b2", model="m")
    seen = []
    for n in range(2):
        work, info = _direct(t1, adapter, tmp_path, name=f"run{n}")
        assert info.returncode == 0
        argv = json.loads((work / "edgearray" / "argv.json").read_text())
        data_dir = Path(argv[argv.index("--data-dir") + 1])
        assert data_dir.is_relative_to(work)
        assert (data_dir / "projects" / argv[argv.index("--project") + 1]).is_dir()
        assert info.extra["project"] == argv[argv.index("--project") + 1]
        assert info.extra["data_dir"] == str(data_dir.relative_to(work))
        seen.append((argv[argv.index("--project") + 1], data_dir.name))
    assert seen[0][0] != seen[1][0]
    assert seen[0][1] != seen[1][1]
    assert all(project.startswith("aedl-t1-001-") for project, _ in seen)


# --- usage -------------------------------------------------------------------------


def test_usage_maps_cache_write_tokens(t1, reference_project, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_project)))
    _, info = _direct(t1, get_adapter("edgearray", model="anthropic/x", baseline="b2"), tmp_path)
    usage = info.usage
    assert usage.input_tokens == 1200
    assert usage.output_tokens == 340
    assert usage.cache_read_tokens == 700
    assert usage.cache_creation_tokens == 90
    assert usage.cost_usd == pytest.approx(0.0125)
    assert usage.num_turns == 4


def test_missing_submission_json_is_recorded_not_raised(
    t1, reference_project, tmp_path, monkeypatch
):
    fake = make_fake(tmp_path, reference_project, write_submission=False)
    monkeypatch.setenv(ENV_BIN, str(fake))
    record, _ = _run(t1, get_adapter("edgearray", baseline="b0"), tmp_path)
    assert record.status == "pass"  # the deliverable comes from the data dir
    assert "submission.json" in record.usage["extra"]["parse_error"]


# --- t1-001: the element DesignSpec ------------------------------------------------


def test_t1_run_builds_design_yaml_and_scores_it(t1, reference_project, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_project)))
    adapter = get_adapter("edgearray", model="anthropic/claude-sonnet", baseline="b2")
    record, bundle = _run(t1, adapter, tmp_path)

    assert record.status == "pass", record.error
    assert record.agent == "edgearray"
    assert record.usage["cache_creation_tokens"] == 90
    extra = record.usage["extra"]
    assert extra["baseline"] == "b2"
    assert extra["verification_report_hash"] == "ab" * 32
    assert extra["deliverable"]["files"] == ["design.yaml"]
    assert extra["deliverable"]["source"] == "design head"
    assert record.transcript["tool_calls_total"] == 2
    assert (bundle / "transcript.jsonl").exists()
    built = yaml.safe_load((bundle / "workspace" / "design.yaml").read_text())
    assert built["design"] == {"name": "t1_001_reference"}
    assert built["requirements"]["substrate_height"] == "0.508 mm"
    assert built["element"] == {"type": "rectangular_patch"}
    # The EdgeArray workspace and its data dir are kept in the bundle as evidence.
    (argv_file,) = (bundle / "workspace" / "edgearray").glob("argv.json")
    assert argv_file.is_file()


def _two_elements(tmp_path):
    other = tmp_path / "other.yaml"
    doc = yaml.safe_load(T1_REFERENCE.read_text())
    doc["design"]["name"] = "newer_patch"
    other.write_text(yaml.safe_dump(doc))
    return [(T1_REFERENCE, "2026-10-01T00:00:00Z"), (other, "2026-10-02T00:00:00Z")]


def _design_name(work: Path) -> str:
    return yaml.safe_load((work / "design.yaml").read_text())["design"]["name"]


def test_t1_prefers_the_head_element_over_a_newer_artifact(t1, tmp_path, monkeypatch):
    project = make_project(tmp_path / "p", _two_elements(tmp_path), head=0)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))
    work, info = _direct(t1, get_adapter("edgearray", baseline="b4", model="m"), tmp_path)
    assert _design_name(work) == "t1_001_reference"
    assert info.extra["deliverable"]["source"] == "design head"


def test_t1_without_a_head_uses_the_most_recent_element(t1, tmp_path, monkeypatch):
    project = make_project(tmp_path / "p", _two_elements(tmp_path), head=None)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))
    work, info = _direct(t1, get_adapter("edgearray", baseline="b4", model="m"), tmp_path)
    assert _design_name(work) == "newer_patch"
    assert info.extra["deliverable"]["source"] == "most recent element.design artifact"
    assert any("no design head" in w for w in info.extra["warnings"])


def test_t1_head_without_an_element_node_falls_back_with_a_warning(t1, tmp_path, monkeypatch):
    project = make_project(tmp_path / "p", _two_elements(tmp_path), head=0, head_has_element=False)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))
    work, info = _direct(t1, get_adapter("edgearray", baseline="b4", model="m"), tmp_path)
    assert _design_name(work) == "newer_patch"
    assert any("no element node" in w for w in info.extra["warnings"])


def test_t1_with_no_element_artifact_is_a_recorded_error(t1, tmp_path, monkeypatch):
    project = tmp_path / "p"
    (project / "artifacts").mkdir(parents=True)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))
    record, _ = _run(t1, get_adapter("edgearray", baseline="b1"), tmp_path)
    assert record.status == "no_submission"
    assert "no element.design artifact" in record.error
    assert "no element.design artifact" in record.usage["extra"]["deliverable_error"]


def test_a_run_that_wrote_no_data_dir_is_a_recorded_error(t1, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, None)))
    record, _ = _run(t1, get_adapter("edgearray", baseline="b1"), tmp_path)
    assert record.status == "no_submission"
    assert "no project directory" in record.error


def _break_payloads(project: Path, replace) -> None:
    for payload in (project / "artifacts").glob("*/*/payload.json"):
        payload.write_text(json.dumps(replace(json.loads(payload.read_text()))))


@pytest.mark.parametrize(
    "replace,expected",
    [
        (lambda doc: [], "expected an object"),
        (lambda doc: {**doc, "design_spec": "a string"}, "design_spec"),
    ],
    ids=["payload-is-a-list", "design-spec-is-a-string"],
)
def test_wrong_type_json_keeps_usage_and_records_the_error(
    t1, tmp_path, monkeypatch, replace, expected
):
    project = make_project(tmp_path / "p", [(T1_REFERENCE, "2026-10-01T00:00:00Z")], head=0)
    _break_payloads(project, replace)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))
    record, _ = _run(t1, get_adapter("edgearray", baseline="b2", model="m"), tmp_path)
    assert record.status == "no_submission"
    assert expected in record.error
    assert record.usage["input_tokens"] == 1200


def test_wrong_type_json_in_a_t4_data_dir_keeps_usage(t4, tmp_path, monkeypatch):
    broken = tmp_path / "broken"
    shutil.copytree(HERO_FIXTURE, broken)
    (broken / "design" / "head.json").write_text("[]")
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, broken)))
    record, _ = _run(t4, get_adapter("edgearray", baseline="b3", model="m"), tmp_path)
    assert record.status == "no_submission"
    assert "expected an object" in record.error
    assert record.usage["output_tokens"] == 340


@pytest.mark.parametrize("exc", [AttributeError("'list' has no attribute 'get'"), TypeError("x")])
def test_unexpected_converter_exceptions_are_recorded_not_raised(
    t1, reference_project, tmp_path, monkeypatch, exc
):
    """Backstop: a shape the converter does not check still cannot crash the run."""
    from aedl.harness.adapters import edgearray as mod

    def boom(data_dir, project, workspace):
        raise exc

    monkeypatch.setitem(mod.DELIVERABLE_BUILDERS, "t1-001", boom)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_project)))
    record, _ = _run(t1, get_adapter("edgearray", baseline="b2", model="m"), tmp_path)
    assert record.status == "no_submission"
    assert f"malformed artifact ({type(exc).__name__}" in record.error
    assert record.usage["input_tokens"] == 1200


def test_a_symlinked_element_artifact_is_not_used_as_the_latest(t1, tmp_path, monkeypatch):
    project = make_project(tmp_path / "p", [(T1_REFERENCE, "2026-10-01T00:00:00Z")], head=None)
    (folder,) = [p.parent for p in (project / "artifacts").glob("*/*/manifest.json")]
    moved = tmp_path / "outside"
    folder.rename(moved)
    folder.symlink_to(moved)
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, project)))
    record, _ = _run(t1, get_adapter("edgearray", baseline="b2", model="m"), tmp_path)
    assert record.status == "no_submission"
    assert "no element.design artifact" in record.error


def test_an_existing_deliverable_is_never_overwritten(t1, reference_project, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_project)))
    parent = tmp_path / "w"
    parent.mkdir()
    work = ws.materialize(t1, isolation="inplace", parent=parent)
    (work / "design.yaml").write_text("keep: me\n")
    info = get_adapter("edgearray", baseline="b2", model="m").run(work, dict(os.environ), 60)
    assert (work / "design.yaml").read_text() == "keep: me\n"
    assert any("kept existing" in w for w in info.extra["warnings"])


def test_a_timed_out_run_builds_no_deliverable(t1, reference_project, tmp_path, monkeypatch):
    fake = make_fake(tmp_path, reference_project, extra_code="time.sleep(60)")
    monkeypatch.setenv(ENV_BIN, str(fake))
    work, info = _direct(
        t1, get_adapter("edgearray", baseline="b2", model="m"), tmp_path, timeout_s=2
    )
    assert info.timed_out
    assert not (work / "design.yaml").exists()
    assert "deliverable" not in info.extra


# --- t4-001: the requirements-to-board bundle --------------------------------------


def test_t4_builds_the_bundle_from_the_run_data_dir(t4, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, HERO_FIXTURE)))
    work, info = _direct(t4, get_adapter("edgearray", baseline="b4", model="m"), tmp_path)
    out = work / "submission"
    # The fixture is the hero's data directory, so the result is the committed hero bundle.
    assert (out / "bundle.json").read_bytes() == (HERO_COMMITTED / "bundle.json").read_bytes()
    assert (out / "element" / "design.yaml").is_file()
    assert (out / "board" / "hero_patch.kicad_pcb").is_file()
    deliverable = info.extra["deliverable"]
    assert deliverable["task"] == "t4-001"
    assert deliverable["source"] == "design head"
    assert "submission/bundle.json" in deliverable["files"]
    assert any("phase_bits" in note for note in deliverable["filled_in"])
    assert deliverable["assumed_verdicts"] == [
        "REQ-GAIN-XCHECK",
        "REQ-GRATING",
        "REQ-POWER",
        "REQ-SLL",
    ]
    assert any("filled in" in w for w in info.extra["warnings"])


def test_t4_conversion_failure_is_recorded_and_leaves_no_partial_bundle(t4, tmp_path, monkeypatch):
    broken = tmp_path / "broken"
    shutil.copytree(HERO_FIXTURE, broken)
    for manifest in (broken / "artifacts").glob("e9/*/payload.json"):
        manifest.unlink()
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, broken)))
    record, bundle = _run(t4, get_adapter("edgearray", baseline="b3", model="m"), tmp_path)
    assert record.status == "no_submission"
    assert "cannot read" in record.error
    assert "payload.json" in record.error
    assert not (bundle / "workspace" / "submission").exists()
    assert not any((bundle / "workspace").glob(".submission*"))


# --- tasks without a mapping -------------------------------------------------------


def test_a_task_without_a_deliverable_mapping_is_an_explicit_error(spec, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, HERO_FIXTURE)))
    record, _ = _run(spec, get_adapter("edgearray", baseline="b2", model="m"), tmp_path)
    assert record.status == "no_submission"
    assert "no deliverable mapping for task t2-001" in record.error
    extra = record.usage["extra"]
    assert "no deliverable mapping for task t2-001" in extra["deliverable_error"]
    assert "no deliverable mapping for task t2-001" in " ".join(extra["warnings"])
