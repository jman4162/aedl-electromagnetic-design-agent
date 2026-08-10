"""Run an agent against a task and score whatever it produces."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path

from aedl.harness import instrument, workspace as ws
from aedl.harness.adapter import AgentAdapter
from aedl.harness.record import RunRecord, utc_stamp
from aedl.registry import get_evaluator
from aedl.spec import TaskSpec

#: Variables that would leak the operator's session into the agent under test.
_SCRUB_PREFIXES = ("CLAUDE", "CLAUDECODE", "AI_AGENT", "AEDL_")


class hidden_references:
    """Make reference solutions unreadable for the duration of a run.

    Workspace isolation keeps the agent's working directory outside the repo,
    but its shell can still reach the whole filesystem — an agent under test was
    observed running `find / -iname "*array_pattern*"` looking for the scoring
    code. Re-deriving the metric is legitimate; reading the worked solution is
    not. Stripping read permission blocks incidental discovery.

    This is a deterrent, not a sandbox: the agent runs as the same user and
    could restore the mode. Publish results from a container.
    """

    def __init__(self, tasks_dir: Path):
        self._dirs = sorted(tasks_dir.glob("*/reference"))
        self._modes: dict[Path, int] = {}

    def __enter__(self):
        for path in self._dirs:
            try:
                self._modes[path] = path.stat().st_mode & 0o777
                path.chmod(0o000)
            except OSError:
                self._modes.pop(path, None)
        return self

    def __exit__(self, *exc):
        for path, mode in self._modes.items():
            try:
                path.chmod(mode)
            except OSError:
                pass
        return False


_PROBE = (
    "import json,sys\n"
    "def v(m):\n"
    "    try:\n"
    "        return __import__(m).__version__\n"
    "    except Exception:\n"
    "        return None\n"
    "print(json.dumps({'executable': sys.executable,\n"
    "                  'python': sys.version.split()[0],\n"
    "                  'numpy': v('numpy'),\n"
    "                  'phased_array': v('phased_array'),\n"
    "                  'phased_array_systems': v('phased_array_systems')}))"
)


def probe_agent_interpreter(env: dict[str, str], cwd: Path) -> dict:
    """What `python3` resolves to for the agent.

    The harness venv is not on the agent's PATH, so the interpreter it reaches
    for — and the library versions it sees — can differ from the ones scoring
    its work. Record them rather than assume.
    """
    import subprocess

    try:
        proc = subprocess.run(
            ["python3", "-c", _PROBE], env=env, cwd=cwd,
            capture_output=True, text=True, timeout=60,
        )
        return json.loads(proc.stdout)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _child_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not any(k.startswith(p) for p in _SCRUB_PREFIXES)
    }
    env.update(extra or {})
    return env


def run_task(
    spec: TaskSpec,
    adapter: AgentAdapter,
    runs_dir: Path,
    isolation: str = "tmpdir",
    timeout_s: int = 900,
    instrumented: bool = True,
) -> tuple[RunRecord, Path]:
    """Execute one attempt and return its record plus the bundle directory."""
    run_id = f"{utc_stamp()}_{spec.id}_{adapter.name}_{uuid.uuid4().hex[:8]}"
    bundle = runs_dir / run_id
    bundle.mkdir(parents=True, exist_ok=True)

    record = RunRecord(
        run_id=run_id,
        task_id=spec.id,
        task_sha256=ws.task_digest(spec),
        agent=adapter.name,
        status="agent_error",
        started_utc=utc_stamp(),
        isolation=isolation,
        instrumented=instrumented,
    )

    work = ws.materialize(spec, isolation=isolation, parent=bundle)
    call_log = bundle / "calls.jsonl"
    env = _child_env()
    if instrumented:
        shim = instrument.write_payload(bundle / ".shim")
        env = instrument.build_env(env, shim, call_log)

    record.agent_interpreter = probe_agent_interpreter(env, work)

    try:
        assert spec.path is not None
        with hidden_references(spec.path.parent.parent):
            info = adapter.run(work, env, timeout_s)
        record.agent_returncode = info.returncode
        record.agent_wall_time_s = round(info.wall_time_s, 3)
        record.timed_out = info.timed_out
        record.agent_command = info.command
        record.model = info.usage.model
        record.usage = {k: v for k, v in asdict(info.usage).items() if v is not None}
        if info.extra:
            record.usage["extra"] = info.extra

        submission = work / ws.SUBMISSION_NAME
        if not submission.exists():
            record.status = "no_submission"
            record.error = f"agent did not create {ws.SUBMISSION_NAME}"
        else:
            try:
                result = get_evaluator(spec.evaluator)(spec, submission)
                record.status = "pass" if result.passed else "fail"
                record.requirements = [asdict(r) for r in result.requirements]
                (bundle / "result.json").write_text(result.to_json())
            except Exception as exc:  # a malformed submission is a result, not a crash
                record.status = "evaluator_error"
                record.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        record.status = "agent_error"
        record.error = f"{type(exc).__name__}: {exc}"
    finally:
        record.calls = instrument.summarize(call_log) if instrumented else {}
        _collect_logs(work, bundle)
        try:
            ws.collect(work, bundle / "workspace")
        except OSError as exc:
            record.error = (record.error or "") + f" [workspace collect failed: {exc}]"
        if isolation == "tmpdir":
            shutil.rmtree(work, ignore_errors=True)
        record.write(bundle / "manifest.json")

    return record, bundle


def _collect_logs(work: Path, bundle: Path) -> None:
    for src, dst in ((".aedl-agent.stdout", "agent.stdout.log"),
                     (".aedl-agent.stderr", "agent.stderr.log")):
        path = work / src
        if path.exists():
            shutil.copy2(path, bundle / dst)
            path.unlink()
