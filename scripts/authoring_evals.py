"""Measure whether coding agents use Cayu's features or rebuild them.

Each task is a product request phrased the way a user would ask, with no Cayu
vocabulary. ``run`` scaffolds a fresh project against a Cayu checkout, hands the
request to a coding agent CLI, then grades the result: did the agent use the
intended Cayu feature, and did it leave known reinvention signals behind?

    python scripts/authoring_evals.py list
    python scripts/authoring_evals.py run --agent copilot --task chat --out results/
    python scripts/authoring_evals.py grade --task chat ~/Desktop/my-project

Runs call a live coding agent and model, so they cost money and are not part of
CI. ``grade`` only reads files (plus ``cayu inspect`` for the approval task).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SKIPPED_DIRECTORIES = {".venv", ".cayu", "node_modules", "__pycache__", ".pytest_cache", "data"}
# Tests and evals may name a feature without the application using it.
NON_APPLICATION_DIRECTORIES = {"tests", "evals"}

# ``{prompt}`` is replaced with the task request; commands run in the project directory.
AGENT_COMMANDS: dict[str, list[str]] = {
    "copilot": ["copilot", "-p", "{prompt}", "--allow-all-tools", "--allow-all-paths", "-s"],
    "copilot-sol": [
        "copilot-sol",
        "-p",
        "{prompt}",
        "--allow-all-tools",
        "--allow-all-paths",
        "-s",
    ],
    "claude": ["claude", "-p", "{prompt}", "--dangerously-skip-permissions"],
    "codex": ["codex", "exec", "--sandbox", "workspace-write", "{prompt}"],
}


@dataclass(frozen=True)
class Project:
    root: Path
    # Relative paths the agent added or modified; None grades every file.
    changed: frozenset[str] | None = None

    def files(self, *suffixes: str) -> Iterator[Path]:
        """Application source files, excluding environments, data, tests and evals."""

        skipped = SKIPPED_DIRECTORIES | NON_APPLICATION_DIRECTORIES
        for path in sorted(self.root.rglob("*")):
            relative = path.relative_to(self.root)
            if (
                path.is_file()
                and path.suffix in suffixes
                and not (skipped & set(relative.parts))
                and (self.changed is None or relative.as_posix() in self.changed)
            ):
                yield path

    def matches(self, pattern: str, *suffixes: str) -> list[str]:
        """Return ``relative/path:line`` for every application line matching ``pattern``."""

        expression = re.compile(pattern)
        found = []
        for path in self.files(*suffixes):
            for number, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if expression.search(line):
                    found.append(f"{path.relative_to(self.root)}:{number}")
        return found


@dataclass
class Grade:
    used_feature: bool
    evidence: list[str] = field(default_factory=list)
    reinvention_signals: list[str] = field(default_factory=list)
    # Whether the agent followed the intended path even if the platform then failed it.
    attempted_feature: bool | None = None


@dataclass(frozen=True)
class Task:
    id: str
    feature: str
    prompt: str
    grade: Callable[[Project], Grade]


def grade_chat(project: Project) -> Grade:
    evidence = project.matches(r"\bResumeRequest\b|\bapp\.resume\(|\.resume\(", ".py")
    signals = []
    if not evidence:
        handlers = project.matches(r"\bRunRequest\(", ".py")
        if project.matches(r"(?i)\bchat\b|conversation|history", ".py") and handlers:
            signals.append(
                "starts a new RunRequest per message instead of resuming the session: "
                + ", ".join(handlers[:3])
            )
    return Grade(bool(evidence), evidence[:5], signals)


def grade_structured_output(project: Project) -> Grade:
    evidence = project.matches(r"\bStructuredOutputSpec\b|structured_output\.output", ".py")
    signals = [
        f"parses JSON out of model text: {location}"
        for location in project.matches(
            r"(json\.loads|re\.(search|match|findall))\([^)\n]*final_text"
            r"|re\.(search|match|findall)\(\s*r?[\"']\\\{",
            ".py",
        )
    ]
    return Grade(bool(evidence), evidence[:5], signals)


def grade_live_ui(project: Project) -> Grade:
    evidence = project.matches(r"client\.js|\bmount_cayu\(", ".py", ".js", ".html")
    fetching = {
        location.split(":", 1)[0] for location in project.matches(r"\bfetch\(", ".js", ".html")
    }
    signals = [
        f"polls with a timer and fetch: {location}"
        for location in project.matches(r"\b(setInterval|setTimeout)\(", ".js", ".html")
        if location.split(":", 1)[0] in fetching
    ]
    used = bool(project.matches(r"client\.js", ".py", ".js", ".html"))
    return Grade(used, evidence[:5], signals)


_STARTER_TOOLS = {"ask_user", "remember_knowledge", "list_artifacts", "list_knowledge"}
_STARTER_TOOLS |= {"search_knowledge", "read_knowledge"}
_APPROVAL_PLUMBING = re.compile(r"(?i)approv")


def grade_approval(project: Project, manifest: dict | None = None) -> Grade:
    if manifest is None:
        manifest = _inspect(project.root)
    gated = [
        f"{tool['name']}: {tool.get('policy_coverage')}"
        for agent in (manifest or {}).get("agents", [])
        for tool in agent.get("tools", [])
        if tool.get("name") not in _STARTER_TOOLS
        and tool.get("policy_coverage") == "approval_required"
    ]
    external = [
        tool["name"]
        for agent in (manifest or {}).get("agents", [])
        for tool in agent.get("tools", [])
        if tool.get("name") not in _STARTER_TOOLS and tool.get("effect") == "external"
    ]
    listing = "\n".join(
        line
        for relative in ("agents/registration.py", "tools/registration.py")
        if (project.root / relative).is_file()
        for line in (project.root / relative).read_text(errors="replace").splitlines()
        if "external" in line or line.strip().startswith(("return", "(", '"', "'"))
    )
    attempted = any(name in listing or f"{name.upper()}_TOOL_NAME" in listing for name in external)
    if manifest is None or "agents" not in manifest:
        # The app could not boot for inspection; fall back to the approval wiring in source.
        wiring = project.matches(r"AlwaysRequireApprovalToolPolicy\(|EveryCallRule\(", ".py")
        gated = [
            f"{location} (from source; the app could not boot for cayu inspect)"
            for location in wiring
        ]
        attempted = attempted or bool(wiring)
    owned = {"policies", "tests", "evals"}
    signals = [
        f"custom approval code outside Cayu's policy seam: {location}"
        for location in project.matches(_APPROVAL_PLUMBING.pattern, ".py", ".js", ".html")
        if location.split("/", 1)[0] not in owned
        and not location.startswith(("agents/registration.py", "tools/registration.py"))
    ]
    return Grade(bool(gated), gated, signals[:10], attempted_feature=bool(gated) or attempted)


TASKS: dict[str, Task] = {
    task.id: task
    for task in (
        Task(
            id="chat",
            feature="session continuation (ResumeRequest)",
            prompt=(
                "Add a chat to this app: a Python function chat(conversation_id, message) "
                "in workflows/chat.py that lets a user ask the agent follow-up questions. "
                "Later answers must take the earlier messages in the same conversation into "
                "account. Add a test."
            ),
            grade=grade_chat,
        ),
        Task(
            id="chat-casual",
            feature="session continuation (ResumeRequest)",
            # The phrasing a real user used when an agent rebuilt chat per message.
            prompt=(
                "Can we add a chat to control the agent? Like if I could just talk to the "
                "agent back and forth from the command line."
            ),
            grade=grade_chat,
        ),
        Task(
            id="approval",
            feature="external-effect tool behind the approval policy",
            prompt=(
                "Let the agent send a notification to the on-call channel (for now, append "
                "it to data/notifications.log). A person must confirm every notification "
                "before it is sent. Add a test."
            ),
            grade=grade_approval,
        ),
        Task(
            id="structured-output",
            feature="StructuredOutputSpec",
            prompt=(
                "Make the agent return its findings as data my Python code can use: a list "
                "of items with title and severity, not just text. Add a test."
            ),
            grade=grade_structured_output,
        ),
        Task(
            id="live-ui",
            feature="mount_cayu with client.js session following",
            prompt=(
                "Add a small web page, served by this project, that shows what the agent is "
                "doing while it works on a request, updating live. Add a test."
            ),
            grade=grade_live_ui,
        ),
    )
}


def _inspect(root: Path) -> dict | None:
    result = subprocess.run(
        ["uv", "run", "--no-sync", "cayu", "inspect", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return None


def _verify(root: Path) -> dict:
    check = subprocess.run(
        ["uv", "run", "--no-sync", "cayu", "check", "--json"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        diagnostics = json.loads(check.stdout).get("diagnostics", [])
    except json.JSONDecodeError:
        diagnostics = [{"code": "CHECK_OUTPUT_UNREADABLE", "severity": "error"}]
    tests = subprocess.run(
        ["uv", "run", "--no-sync", "pytest", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "check_errors": sorted(
            item["code"] for item in diagnostics if item.get("severity") == "error"
        ),
        "tests_passed": tests.returncode == 0,
        "tests_summary": (tests.stdout.strip().splitlines() or [""])[-1],
    }


def scaffold(directory: Path, cayu_source: Path) -> Path:
    """Create a fresh project that imports Cayu from ``cayu_source``."""

    env = dict(os.environ, PYTHONPATH=str(cayu_source / "src"))
    created = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from cayu.cli import main; raise SystemExit(main(sys.argv[1:]))",
            "new",
            "evalproj",
            "--dir",
            str(directory),
            "--json",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if created.returncode != 0:
        raise RuntimeError(f"cayu new failed: {created.stdout.strip()} {created.stderr.strip()}")
    project = directory / "evalproj"
    pyproject = project / "pyproject.toml"
    # `cayu new` run from a checkout already points the project at it; older
    # generators wrote a PyPI pin, so add the checkout source only if it's missing.
    if "[tool.uv.sources]" not in pyproject.read_text():
        pyproject.write_text(
            pyproject.read_text()
            + f'\n[tool.uv.sources]\ncayu = {{ path = "{cayu_source}", editable = true }}\n'
        )
    subprocess.run(["uv", "sync", "--extra", "dev", "--quiet"], cwd=project, check=True)
    subprocess.run(["git", "init", "-q"], cwd=project, check=True)
    subprocess.run(["git", "add", "-A"], cwd=project, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=eval",
            "-c",
            "user.email=eval@localhost",
            "commit",
            "-qm",
            "scaffold",
        ],
        cwd=project,
        check=True,
    )
    return project


def changed_files(project: Path, scaffold_commit: str) -> frozenset[str]:
    """Files the agent added, modified or deleted since the scaffold commit.

    Compares the final worktree against that fixed commit, so work the agent
    committed counts the same as uncommitted or untracked work.
    """

    def git(*args: str) -> list[str]:
        result = subprocess.run(
            ["git", *args], cwd=project, capture_output=True, text=True, check=True
        )
        return [line for line in result.stdout.splitlines() if line]

    tracked = git("diff", "--name-only", "--no-renames", scaffold_commit)
    untracked = git("ls-files", "--others", "--exclude-standard")
    return frozenset(tracked + untracked)


def scaffold_commit(project: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project, capture_output=True, text=True, check=True
    ).stdout.strip()


def run_task(task: Task, agent_command: list[str], cayu_source: Path, timeout: int) -> dict:
    # cayu new refuses paths through symlinks, and macOS temp dirs sit under /var -> /private/var.
    workdir = Path(tempfile.mkdtemp(prefix=f"cayu-authoring-{task.id}-")).resolve()
    project = scaffold(workdir, cayu_source)
    base = scaffold_commit(project)
    command = [part.replace("{prompt}", task.prompt) for part in agent_command]
    started = time.monotonic()
    try:
        agent = subprocess.run(
            command, cwd=project, capture_output=True, text=True, timeout=timeout, check=False
        )
        agent_status = f"exit {agent.returncode}"
        transcript = agent.stdout[-20_000:] + agent.stderr[-5_000:]
    except subprocess.TimeoutExpired:
        agent_status, transcript = "timeout", ""
    grade = task.grade(Project(project, changed_files(project, base)))
    return {
        "task": task.id,
        "feature": task.feature,
        "agent_status": agent_status,
        "seconds": round(time.monotonic() - started),
        "project": str(project),
        **asdict(grade),
        **_verify(project),
        "transcript_tail": transcript,
    }


# Multi-turn replays need an agent CLI that can resume one session by ID.
REPLAY_COMMANDS: dict[str, list[str]] = {
    name: [binary, "-p", "{prompt}", "--session-id", "{session_id}", "--allow-all", "-s"]
    for name, binary in (("copilot", "copilot"), ("copilot-sol", "copilot-sol"))
}


def _find_project(workdir: Path) -> Path | None:
    for pyproject in sorted(workdir.rglob("pyproject.toml")):
        relative = pyproject.relative_to(workdir)
        if SKIPPED_DIRECTORIES & set(relative.parts) or len(relative.parts) > 3:
            continue
        if "[tool.cayu" in pyproject.read_text(errors="replace"):
            return pyproject.parent
    return None


def _scaffold_options(project: Path) -> list[str]:
    """Recreate the project's preset and adapter choices from [tool.cayu.scaffold]."""

    text = (project / "pyproject.toml").read_text(errors="replace")
    section = text.split("[tool.cayu.scaffold]", 1)[-1].split("\n[", 1)[0]
    options = []
    for key, flag in (
        ("preset", "--preset"),
        ("execution", "--execution"),
        ("coding_toolchain", "--coding-toolchain"),
        ("coding_command_authority", "--coding-command-authority"),
    ):
        match = re.search(rf'^{key}\s*=\s*"([^"]+)"', section, re.MULTILINE)
        if match and match.group(1) not in {"none", "neutral"}:
            options += [flag, match.group(1)]
    return options


def _changed_from_scaffold(project: Path, cayu_source: Path) -> frozenset[str]:
    """Files that differ from a fresh scaffold with the same project name."""

    reference_root = Path(tempfile.mkdtemp(prefix="cayu-replay-reference-")).resolve()
    try:
        env = dict(os.environ, PYTHONPATH=str(cayu_source / "src"))
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; from cayu.cli import main; raise SystemExit(main(sys.argv[1:]))",
                "new",
                project.name,
                "--dir",
                str(reference_root),
                "--json",
                *_scaffold_options(project),
            ],
            env=env,
            capture_output=True,
            check=False,
        )
        reference = reference_root / project.name
        changed = set()
        for path in project.rglob("*"):
            relative = path.relative_to(project)
            if not path.is_file() or SKIPPED_DIRECTORIES & set(relative.parts):
                continue
            original = reference / relative
            if not original.is_file() or original.read_bytes() != path.read_bytes():
                changed.add(relative.as_posix())
        return frozenset(changed)
    finally:
        shutil.rmtree(reference_root, ignore_errors=True)


def _stop_processes_under(directory: Path) -> None:
    listed = subprocess.run(
        ["lsof", "-t", "+D", str(directory)], capture_output=True, text=True, check=False
    )
    for pid in {line for line in listed.stdout.split() if line.isdigit()}:
        subprocess.run(["kill", pid], capture_output=True, check=False)


def run_replay(
    script: dict, agent_command: list[str], cayu_source: Path, timeout: int, out: Path
) -> dict:
    """Send a recorded user session's turns, in order, to one resumed agent session."""

    workdir = Path(tempfile.mkdtemp(prefix=f"cayu-replay-{script['name']}-")).resolve()
    # The checkout's CLI lives outside the replay directory so the agent doesn't see it as code.
    cli = Path(tempfile.mkdtemp(prefix="cayu-replay-cli-")).resolve()
    subprocess.run(["uv", "venv", "--quiet", str(cli)], check=True)
    subprocess.run(
        [
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            str(cli / "bin" / "python"),
            "-e",
            str(cayu_source),
        ],
        check=True,
    )
    env = dict(os.environ, PATH=f"{cli / 'bin'}{os.pathsep}{os.environ['PATH']}")
    session_id = str(uuid.uuid4())
    turns = []
    for index, template in enumerate(script["turns"], 1):
        prompt = template.replace("{WORKDIR}", str(workdir)).replace(
            "{CAYU_SOURCE}", str(cayu_source)
        )
        command = [
            part.replace("{prompt}", prompt).replace("{session_id}", session_id)
            for part in agent_command
        ]
        started = time.monotonic()
        try:
            agent = subprocess.run(
                command,
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            status, output = (
                f"exit {agent.returncode}",
                agent.stdout[-4_000:] + agent.stderr[-1_000:],
            )
        except subprocess.TimeoutExpired:
            status, output = "timeout", ""
        turns.append(
            {
                "turn": index,
                "prompt": prompt,
                "status": status,
                "seconds": round(time.monotonic() - started),
                "output_tail": output,
            }
        )
        (out / "turns.json").write_text(json.dumps(turns, indent=2))
        print(
            f"turn {index}/{len(script['turns'])}: {status} in {turns[-1]['seconds']}s", flush=True
        )
    _stop_processes_under(workdir)
    events = Path.home() / ".copilot" / "session-state" / session_id / "events.jsonl"
    if events.is_file():
        shutil.copy(events, out / "session-events.jsonl")
    project = _find_project(workdir)
    result: dict = {
        "replay": script["name"],
        "cayu_source": str(cayu_source),
        "session_id": session_id,
        "workdir": str(workdir),
        "project": None if project is None else str(project),
        "turn_statuses": [turn["status"] for turn in turns],
    }
    if project is not None:
        scoped = Project(project, _changed_from_scaffold(project, cayu_source))
        result["grades"] = {
            task_id: asdict(TASKS[task_id].grade(scoped))
            for task_id in ("chat", "approval", "structured-output", "live-ui")
        }
        result.update(_verify(project))
    shutil.rmtree(cli, ignore_errors=True)
    return result


def _summary(results: list[dict]) -> str:
    lines = [
        "| task | used feature | reinvention signals | check errors | tests |",
        "|---|---|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {result['task']} | {'yes' if result['used_feature'] else 'no'} | "
            f"{len(result['reinvention_signals'])} | {len(result.get('check_errors', []))} | "
            f"{'pass' if result.get('tests_passed') else 'fail'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List tasks and the Cayu feature each one expects.")
    run = commands.add_parser("run", help="Scaffold, run a coding agent, and grade.")
    run.add_argument("--agent", choices=sorted(AGENT_COMMANDS), default="copilot")
    run.add_argument(
        "--agent-command",
        help='Custom command as a JSON list containing "{prompt}"; overrides --agent.',
    )
    run.add_argument("--task", action="append", choices=sorted(TASKS), dest="tasks")
    run.add_argument("--cayu-source", type=Path, default=REPOSITORY_ROOT)
    run.add_argument("--timeout", type=int, default=1800, help="Seconds per agent run.")
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--keep", action="store_true", help="Keep generated projects.")
    replay = commands.add_parser(
        "replay", help="Replay a recorded multi-turn user session, then grade the result."
    )
    replay.add_argument(
        "script", type=Path, help="Replay JSON, e.g. scripts/authoring_replays/csv_auditor.json"
    )
    replay.add_argument("--agent", choices=sorted(REPLAY_COMMANDS), default="copilot")
    replay.add_argument(
        "--agent-command",
        help='Custom command as a JSON list containing "{prompt}" and "{session_id}".',
    )
    replay.add_argument("--cayu-source", type=Path, default=REPOSITORY_ROOT)
    replay.add_argument("--timeout", type=int, default=3600, help="Seconds per turn.")
    replay.add_argument("--out", type=Path, required=True)
    grade = commands.add_parser("grade", help="Grade an existing project for one task.")
    grade.add_argument("--task", choices=sorted(TASKS), required=True)
    grade.add_argument("--verify", action="store_true", help="Also run cayu check and pytest.")
    grade.add_argument("project", type=Path)
    args = parser.parse_args(argv)

    if args.command == "list":
        for task in TASKS.values():
            print(f"{task.id:<18} expects {task.feature}\n{'':<18} {task.prompt}\n")
        return 0
    if args.command == "replay":
        command = (
            json.loads(args.agent_command) if args.agent_command else REPLAY_COMMANDS[args.agent]
        )
        if not any("{session_id}" in part for part in command):
            parser.error('the replay command must contain "{session_id}" to resume one session')
        args.out.mkdir(parents=True, exist_ok=True)
        script = json.loads(args.script.read_text())
        result = run_replay(script, command, args.cayu_source.resolve(), args.timeout, args.out)
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print(
            json.dumps({key: value for key, value in result.items() if key != "grades"}, indent=2)
        )
        for task_id, grade in result.get("grades", {}).items():
            print(
                f"{task_id}: used_feature={grade['used_feature']} signals={len(grade['reinvention_signals'])}"
            )
        return 0
    if args.command == "grade":
        project = Project(args.project.expanduser().resolve())
        result = {"task": args.task, **asdict(TASKS[args.task].grade(project))}
        if args.verify:
            result.update(_verify(project.root))
        print(json.dumps(result, indent=2))
        return 0

    agent_command = (
        json.loads(args.agent_command) if args.agent_command else AGENT_COMMANDS[args.agent]
    )
    if not any("{prompt}" in part for part in agent_command):
        parser.error('the agent command must contain "{prompt}"')
    if shutil.which(agent_command[0]) is None:
        parser.error(f"agent command not found on PATH: {agent_command[0]}")
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    for task_id in args.tasks or sorted(TASKS):
        result = run_task(TASKS[task_id], agent_command, args.cayu_source.resolve(), args.timeout)
        (args.out / f"{task_id}.json").write_text(json.dumps(result, indent=2))
        if not args.keep:
            shutil.rmtree(Path(result["project"]).parent, ignore_errors=True)
        results.append(result)
        print(
            f"{task_id}: used_feature={result['used_feature']} signals={len(result['reinvention_signals'])}"
        )
    summary = _summary(results)
    (args.out / "summary.md").write_text(summary + "\n")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
