from __future__ import annotations

import argparse
import re
import sys
from importlib.resources import files
from typing import Any

_GUIDES = {
    "anatomy": ("application-anatomy.md", "Application lifecycle and process roles."),
    "authoring": ("authoring.md", "Concept map and the supported authoring loop."),
    "diagnostics": ("diagnostics.md", "Stable `cayu check` findings and fixes."),
    "providers": ("providers.md", "Explicit provider and compatible model selection."),
    "references": ("references.md", "Offline references for optional capabilities."),
    "structured-output": (
        "structured-output.md",
        "Credential-free structured-output runtime proof.",
    ),
    "tool-effects": ("tool-effects.md", "Replay and mutation effect decisions."),
}
_INCLUDES = {
    "<!-- cayu-guide-include:pytest-selector -->": (
        "command_selectors.py",
        "# cayu-guide-include:pytest-selector:start",
        "# cayu-guide-include:pytest-selector:end",
    ),
}


def _render_includes(content: str) -> str:
    for placeholder, (resource_name, start_marker, end_marker) in _INCLUDES.items():
        count = content.count(placeholder)
        if count == 0:
            continue
        if count != 1:
            raise RuntimeError(f"guide include must appear exactly once: {placeholder}")
        source = files("cayu.guides").joinpath(resource_name).read_text(encoding="utf-8")
        try:
            recipe = source.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
        except IndexError as error:
            raise RuntimeError(f"guide include markers are malformed: {resource_name}") from error
        content = content.replace(placeholder, f"```python\n{recipe}\n```")
    return content


def add_guide_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "guide",
        help="Print package-shipped Cayu application guidance.",
        description=(
            "Read version-matched guidance shipped in the installed Cayu package. "
            "Use TOPIC#SECTION for a specific emitted documentation anchor."
        ),
        epilog="Topics:\n"
        + "\n".join(f"  {name:<18} {description}" for name, (_, description) in _GUIDES.items()),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("name", nargs="?", metavar="TOPIC[#SECTION]")


def run_guide(args: argparse.Namespace) -> int:
    if args.name is None:
        print("Package-shipped Cayu guides:")
        for name, (_, description) in _GUIDES.items():
            print(f"  {name:<18} {description}")
        print("Run `cayu guide TOPIC` or `cayu guide TOPIC#SECTION`.")
        return 0
    name, separator, anchor = args.name.partition("#")
    guide_record = _GUIDES.get(name)
    if guide_record is None:
        print(
            f"error: unknown guide topic {name!r}; choose from: {', '.join(_GUIDES)}",
            file=sys.stderr,
        )
        return 2
    if separator and not anchor:
        print("error: guide section after `#` must not be empty", file=sys.stderr)
        return 2
    guide = files("cayu.guides").joinpath(guide_record[0])
    content = _render_includes(guide.read_text(encoding="utf-8"))
    if anchor:
        section = _guide_section(content, anchor)
        if section is None:
            print(f"error: section {anchor!r} was not found in guide {name!r}", file=sys.stderr)
            return 2
        content = section
    print(content, end="")
    return 0


def _guide_section(content: str, anchor: str) -> str | None:
    lines = content.splitlines(keepends=True)
    match_index: int | None = None
    match_level = 0
    for index, line in enumerate(lines):
        heading = re.fullmatch(r"(#{1,6})\s+(.+?)\s*\n?", line)
        if heading is None or _heading_anchor(heading.group(2)) != anchor:
            continue
        match_index = index
        match_level = len(heading.group(1))
        break
    if match_index is None:
        return None
    end = len(lines)
    for index in range(match_index + 1, len(lines)):
        heading = re.match(r"(#{1,6})\s+", lines[index])
        if heading is not None and len(heading.group(1)) <= match_level:
            end = index
            break
    return "".join(lines[match_index:end])


def _heading_anchor(heading: str) -> str:
    normalized = re.sub(r"[^a-z0-9 -]", "", heading.casefold())
    return re.sub(r"[ -]+", "-", normalized).strip("-")
