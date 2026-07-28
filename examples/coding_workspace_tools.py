from __future__ import annotations

import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import cast

from cayu import (
    EditFileTool,
    GitChangesTool,
    LocalRunner,
    LocalWorkspace,
    ReadFileTool,
    RunnerHandle,
    ToolContext,
)


def git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cayu-coding-tools-") as directory:
        root = Path(directory)
        git(root, "init", "-q")
        source = root / "answer.py"
        source.write_text("def answer():\n    return 41\n")
        git(root, "add", "answer.py")
        git(
            root,
            "-c",
            "user.name=Cayu Example",
            "-c",
            "user.email=example@cayu.dev",
            "commit",
            "-qm",
            "initial",
        )

        context = ToolContext(
            session_id="coding-tools-example",
            workspace=LocalWorkspace(root),
            runner=cast("RunnerHandle", LocalRunner(root)),
        )
        read = await ReadFileTool().run(context, {"path": "answer.py"})
        assert read.structured is not None
        revision = read.structured["revision"]
        assert isinstance(revision, str)

        edited = await EditFileTool().run(
            context,
            {
                "path": "answer.py",
                "expected_revision": revision,
                "edits": [{"old_text": "return 41", "new_text": "return 42"}],
            },
        )
        summary = await GitChangesTool().run(context, {"mode": "summary"})
        diff = await GitChangesTool().run(
            context,
            {"mode": "diff", "paths": ["answer.py"]},
        )

        print(edited.content)
        print(summary.content)
        print(diff.content)


if __name__ == "__main__":
    asyncio.run(main())
