"""The EdgeArray adapter, run against a fake ``edgearray`` executable.

The real CLI (``edgearray agent run``) lives in a separate repository. The
fake honours the same arguments, writes a canned ``submission.json`` and
``transcript.jsonl`` into ``--workspace``, and drops the task deliverable
there, which is all the adapter relies on.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

from aedl.harness import adapter_names, get_adapter, run_task
from aedl.harness import workspace as ws
from aedl.harness.adapters.edgearray import ENV_BIN, EdgeArrayAdapter
from aedl.spec import find_task

REPO = Path(__file__).resolve().parents[1]
TASKS = REPO / "tasks"
SOLVE = TASKS / "t2-001-steered-beam-2bit" / "reference" / "solve.py"

FAKE = """#!{python}
import json, pathlib, shutil, sys

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
(ws / "argv.json").write_text(json.dumps(argv))
if {deliverable_src!r}:
    shutil.copyfile({deliverable_src!r}, ws / {deliverable_name!r})
llm = opts.get("baseline", "b4") != "b0"
doc = {{
    "final_text": "done",
    "model": opts.get("model"),
    "usage": {{
        "input_tokens": 1200 if llm else 0,
        "output_tokens": 340 if llm else 0,
        "cost_usd": 0.0125 if llm else 0.0,
    }},
    "tool_calls": [
        {{"name": "edgearray.pattern.compute", "ok": True, "ms": 12}},
        {{"name": "edgearray.verify.run", "ok": True, "ms": 40}},
    ],
    "verification_report_hash": "ab" * 32,
    "deliverables": {deliverables!r},
}}
if {write_submission!r}:
    (ws / "submission.json").write_text(json.dumps(doc))
(ws / "transcript.jsonl").write_text(json.dumps({{"type": "tool_call"}}) + "\\n")
{extra_code}
"""


@pytest.fixture(scope="module")
def spec():
    return find_task(TASKS, "t2-001")


@pytest.fixture(scope="module")
def reference_npz(tmp_path_factory):
    path = tmp_path_factory.mktemp("ref") / "weights.npz"
    np.savez(path, weights=runpy.run_path(str(SOLVE))["solve"]())
    return path


def make_fake(
    tmp_path: Path,
    deliverable_src: Path | None,
    deliverable_name: str = ws.SUBMISSION_NAME,
    write_submission: bool = True,
    deliverables: list[str] | None = None,
    extra_code: str = "",
) -> Path:
    fake = tmp_path / "bin" / "edgearray"
    fake.parent.mkdir(parents=True, exist_ok=True)
    fake.write_text(
        FAKE.format(
            python=sys.executable,
            deliverable_src=str(deliverable_src) if deliverable_src else "",
            deliverable_name=deliverable_name,
            write_submission=write_submission,
            deliverables=deliverables or [],
            extra_code=extra_code,
        )
    )
    fake.chmod(0o755)
    return fake


def _run(spec, adapter, tmp_path):
    return run_task(
        spec,
        adapter,
        runs_dir=tmp_path / "runs",
        isolation="tmpdir",
        timeout_s=60,
        instrumented=False,
    )


def test_adapter_is_registered():
    assert "edgearray" in adapter_names()


def test_command_line(spec, tmp_path, monkeypatch):
    monkeypatch.delenv(ENV_BIN, raising=False)
    work = ws.materialize(spec, isolation="inplace", parent=tmp_path)
    adapter = EdgeArrayAdapter(model="ollama/qwen3.5:9b", baseline="b2", budget_s=600)
    assert adapter.build_command(work) == [
        "edgearray",
        "agent",
        "run",
        "--brief",
        str(work / "BRIEF.md"),
        "--workspace",
        str(work / "edgearray"),
        "--project",
        "aedl-t2-001",
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


def test_run_scores_the_copied_deliverable_and_maps_usage(
    spec, reference_npz, tmp_path, monkeypatch
):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_npz)))
    adapter = get_adapter("edgearray", model="anthropic/claude-sonnet", baseline="b2")
    record, bundle = _run(spec, adapter, tmp_path)

    assert record.status == "pass", record.error
    assert record.agent == "edgearray"
    assert record.model == "anthropic/claude-sonnet"
    assert record.usage["input_tokens"] == 1200
    assert record.usage["output_tokens"] == 340
    assert record.usage["cost_usd"] == pytest.approx(0.0125)
    extra = record.usage["extra"]
    assert extra["baseline"] == "b2"
    assert extra["deliverables_copied"] == [ws.SUBMISSION_NAME]
    assert extra["verification_report_hash"] == "ab" * 32
    assert record.transcript["tool_calls_total"] == 2
    assert record.transcript["tool_calls_by_name"]["edgearray.verify.run"] == 1
    assert (bundle / "transcript.jsonl").exists()
    # The EdgeArray workspace is kept in the bundle as evidence.
    argv = json.loads((bundle / "workspace" / "edgearray" / "argv.json").read_text())
    assert argv[argv.index("--baseline") + 1] == "b2"


def test_missing_submission_json_is_recorded_not_raised(spec, reference_npz, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, reference_npz, write_submission=False)))
    record, _ = _run(spec, get_adapter("edgearray", baseline="b0"), tmp_path)
    assert record.status == "pass"
    assert "submission.json" in record.usage["extra"]["parse_error"]


def test_no_deliverable_is_no_submission(spec, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_BIN, str(make_fake(tmp_path, None)))
    record, _ = _run(spec, get_adapter("edgearray", baseline="b1"), tmp_path)
    assert record.status == "no_submission"


def test_listed_deliverables_outside_the_workspace_are_refused(
    spec, reference_npz, tmp_path, monkeypatch
):
    fake = make_fake(tmp_path, reference_npz, deliverables=["../../escape.txt", "notes.md"])
    monkeypatch.setenv(ENV_BIN, str(fake))
    record, _ = _run(spec, get_adapter("edgearray", baseline="b3"), tmp_path)
    warnings = record.usage["extra"]["warnings"]
    assert any("escape.txt" in w for w in warnings)
    assert any("notes.md" in w for w in warnings)  # listed but never written
    assert not (tmp_path / "escape.txt").exists()


def test_deliverables_containing_symlinks_are_refused(spec, reference_npz, tmp_path, monkeypatch):
    """A listed directory holding a link to a file outside the workspace is
    not copied: copytree would follow the link and import the target."""
    secret = tmp_path / "outside-secret.txt"
    secret.write_text("not for the agent")
    code = (
        "extras = ws / 'extras'\n"
        "extras.mkdir()\n"
        f"(extras / 'link.txt').symlink_to({str(secret)!r})\n"
    )
    fake = make_fake(tmp_path, reference_npz, deliverables=["extras"], extra_code=code)
    monkeypatch.setenv(ENV_BIN, str(fake))
    record, bundle = _run(spec, get_adapter("edgearray", baseline="b2"), tmp_path)
    assert any("symlink" in w for w in record.usage["extra"]["warnings"])
    assert "extras" not in record.usage["extra"]["deliverables_copied"]
    assert not (bundle / "workspace" / "extras").exists()
