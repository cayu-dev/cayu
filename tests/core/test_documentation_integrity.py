from __future__ import annotations

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

_ROOT = Path(__file__).parents[2]
_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
_REFERENCE_LINK = re.compile(r"^\s*\[[^\]]+\]:\s*(\S+)", re.MULTILINE)


def _tracked_files(pattern: str | None = None) -> set[Path]:
    command = ["git", "ls-files"]
    if pattern is not None:
        command.extend(["--", pattern])
    output = subprocess.run(
        command,
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {Path(line) for line in output.splitlines() if line}


def _tracked_markdown() -> set[Path]:
    return _tracked_files("*.md")


def _local_destination(raw: str) -> str | None:
    destination = raw.strip()
    if destination.startswith("<") and ">" in destination:
        destination = destination[1 : destination.index(">")]
    else:
        destination = destination.split(maxsplit=1)[0]
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    return unquote(parsed.path)


def test_tracked_markdown_links_resolve_to_tracked_repository_content() -> None:
    tracked = _tracked_files()
    for document in sorted(_tracked_markdown()):
        source = (_ROOT / document).read_text(encoding="utf-8")
        destinations = [*_INLINE_LINK.findall(source), *_REFERENCE_LINK.findall(source)]
        for raw in destinations:
            destination = _local_destination(raw)
            if destination is None:
                continue
            assert not destination.startswith("/"), f"{document}: absolute local link {raw}"
            target = (document.parent / destination).resolve()
            assert target.is_relative_to(_ROOT), f"{document}: link escapes repository: {raw}"
            relative = target.relative_to(_ROOT)
            assert relative in tracked or any(path.is_relative_to(relative) for path in tracked), (
                f"{document}: untracked or missing local link target: {raw}"
            )


def test_documentation_index_classifies_every_markdown_document() -> None:
    tracked_docs = {path for path in _tracked_markdown() if path.parts[:1] == ("docs",)}
    index = (_ROOT / "docs" / "README.md").read_text(encoding="utf-8")
    for document in sorted(tracked_docs - {Path("docs/README.md")}):
        relative = document.relative_to("docs").as_posix()
        assert f"({relative})" in index, f"docs/README.md does not classify {document}"
