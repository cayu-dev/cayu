from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "authoring_evals.py"
evals = runpy.run_path(str(SCRIPT))


def _project(root: Path, files: dict[str, str]):
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return evals["Project"](root)


def test_chat_grader_credits_session_resume(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "workflows/chat.py": (
                "from cayu import ResumeRequest, run_to_completion\n"
                "async def chat(app, session_id, message):\n"
                "    return await run_to_completion(app, ResumeRequest(session_id=session_id,"
                " messages=[message]))\n"
            )
        },
    )

    grade = evals["grade_chat"](project)

    assert grade.used_feature is True
    assert grade.reinvention_signals == []


def test_chat_grader_flags_a_new_run_per_message(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "workflows/chat.py": (
                "from cayu import RunRequest\n"
                "def chat(app, history, message):\n"
                "    return RunRequest(agent_name='a', messages=[*history, message])\n"
            ),
            # A test that mentions the feature must not count as using it.
            "tests/test_chat.py": "from cayu import ResumeRequest\n",
        },
    )

    grade = evals["grade_chat"](project)

    assert grade.used_feature is False
    assert grade.reinvention_signals and "workflows/chat.py:3" in grade.reinvention_signals[0]


def test_structured_output_grader_flags_json_extraction(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "workflows/findings.py": (
                "import json, re\n"
                "def parse(outcome):\n"
                "    match = re.search(r'\\{.*\\}', outcome.final_text)\n"
                "    return json.loads(outcome.final_text)\n"
            )
        },
    )

    grade = evals["grade_structured_output"](project)

    assert grade.used_feature is False
    assert len(grade.reinvention_signals) == 2


def test_live_ui_grader_flags_timer_polling_and_credits_client_js(tmp_path: Path) -> None:
    polling = _project(
        tmp_path / "polling",
        {
            "web/app.js": "async function tick() { await fetch('/api/jobs'); }\nsetTimeout(tick, 5000);\n"
        },
    )
    following = _project(
        tmp_path / "following",
        {"web/app.js": "import { connect } from '/cayu/client.js';\nconnect({ sessionId });\n"},
    )

    assert evals["grade_live_ui"](polling).reinvention_signals == [
        "polls with a timer and fetch: web/app.js:2"
    ]
    assert evals["grade_live_ui"](following).used_feature is True


@pytest.mark.parametrize(
    ("coverage", "expected"), (("approval_required", True), ("allowed", False))
)
def test_approval_grader_reads_policy_coverage(
    tmp_path: Path, coverage: str, expected: bool
) -> None:
    project = _project(tmp_path, {"web/approve.js": "fetch('/api/approve', {method: 'POST'})\n"})
    manifest = {
        "agents": [
            {
                "tools": [
                    {"name": "remember_knowledge", "policy_coverage": "approval_required"},
                    {"name": "send_notice", "policy_coverage": coverage},
                ]
            }
        ]
    }

    grade = evals["grade_approval"](project, manifest)

    assert grade.used_feature is expected
    assert grade.reinvention_signals == [
        "custom approval code outside Cayu's policy seam: web/approve.js:1"
    ]


def test_every_task_is_phrased_without_cayu_vocabulary() -> None:
    for task in evals["TASKS"].values():
        lowered = task.prompt.lower()
        for term in ("cayu", "resumerequest", "structuredoutput", "client.js", "toolpolicy"):
            assert term not in lowered, (task.id, term)


def test_approval_grader_separates_attempt_from_platform_gating(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "agents/registration.py": (
                "    starter_external_tool_names.append(SEND_NOTICE_TOOL_NAME)\n"
            )
        },
    )
    manifest = {
        "agents": [
            {"tools": [{"name": "send_notice", "effect": "external", "policy_coverage": "allowed"}]}
        ]
    }

    grade = evals["grade_approval"](project, manifest)

    assert grade.used_feature is False
    assert grade.attempted_feature is True


def test_graders_ignore_files_the_agent_did_not_touch(tmp_path: Path) -> None:
    files = {
        # Shipped by the scaffold: mentions approvals but is not agent work.
        "operations/approvals.py": "def approve(): ...\n",
        "web/approve.js": "fetch('/api/approve')\n",
    }
    project = _project(tmp_path, files)
    scoped = evals["Project"](project.root, frozenset({"web/approve.js"}))

    signals = evals["grade_approval"](scoped, {"agents": []}).reinvention_signals

    assert signals == ["custom approval code outside Cayu's policy seam: web/approve.js:1"]


def test_changed_files_counts_work_the_agent_committed(tmp_path: Path) -> None:
    def git(*args: str) -> None:
        subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
            cwd=tmp_path,
            check=True,
            capture_output=True,
        )

    _project(tmp_path, {"app.py": "scaffold\n", "README.md": "scaffold\n"})
    git("init", "-q")
    git("add", "-A")
    git("commit", "-qm", "scaffold")
    base = evals["scaffold_commit"](tmp_path)

    _project(tmp_path, {"workflows/chat.py": "committed\n", "app.py": "edited\n"})
    git("add", "-A")
    git("commit", "-qm", "agent work")
    _project(tmp_path, {"notes.txt": "untracked\n", "README.md": "uncommitted\n"})

    assert evals["changed_files"](tmp_path, base) == {
        "workflows/chat.py",
        "app.py",
        "notes.txt",
        "README.md",
    }


def test_live_ui_grader_credits_client_js_in_inline_html(tmp_path: Path) -> None:
    project = _project(
        tmp_path,
        {
            "web.py": "PAGE = '<script type=module>import {connect} from \"/cayu/client.js\"</script>'\n"
        },
    )

    assert evals["grade_live_ui"](project).used_feature is True
