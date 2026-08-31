from __future__ import annotations

import argparse
import json
import re
import sys
from importlib.resources import files
from typing import Any

from cayu._version import package_version

_GUIDES = {
    "anatomy": ("application-anatomy.md", "Application lifecycle and process roles."),
    "applications": (
        "applications.md",
        "Generated application convention, placement, planning, and drift checks.",
    ),
    "authoring": ("authoring.md", "Concept map and the supported authoring loop."),
    "diagnostics": ("diagnostics.md", "Stable `cayu check` findings and fixes."),
    "durable-operations": (
        "durable-operations.md",
        "Observe, propose, authorize, act once, verify, and recover.",
    ),
    "evals-ai-quality": (
        "evals-ai-quality.md",
        "Rubrics, references, judge authority, calibration, and interpretation.",
    ),
    "evals-first": (
        "evals-first.md",
        "First Control Plane suite, run, baseline, and comparison.",
    ),
    "evals-production": (
        "evals-production.md",
        "Production sessions, scenarios, tools, process behavior, and memory.",
    ),
    "providers": (
        "providers.md",
        "Primary integrations and compatible Chat Completions endpoints.",
    ),
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
_RELATED = {
    "anatomy": ("applications", "authoring", "diagnostics"),
    "applications": ("anatomy", "authoring", "diagnostics", "references"),
    "authoring": (
        "anatomy",
        "tool-effects",
        "durable-operations",
        "evals-first",
        "references",
    ),
    "diagnostics": ("anatomy", "authoring"),
    "durable-operations": ("tool-effects", "references"),
    "evals-ai-quality": ("evals-first", "evals-production", "authoring"),
    "evals-first": ("evals-ai-quality", "evals-production", "authoring"),
    "evals-production": ("evals-first", "evals-ai-quality", "durable-operations"),
    "providers": ("authoring", "diagnostics"),
    "references": ("authoring", "durable-operations"),
    "structured-output": ("authoring", "diagnostics"),
    "tool-effects": ("authoring", "durable-operations"),
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit stable topic, section, package, relationship, and verification metadata.",
    )


def run_guide(args: argparse.Namespace) -> int:
    if args.name is None:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "installed_cayu_version": package_version(),
                        "topics": [
                            {
                                "topic": name,
                                "summary": description,
                                "package_source": f"cayu.guides/{resource}",
                                "related_topics": list(_RELATED.get(name, ())),
                            }
                            for name, (resource, description) in _GUIDES.items()
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        print("Package-shipped Cayu guides:")
        for name, (_, description) in _GUIDES.items():
            print(f"  {name:<18} {description}")
        print("Run `cayu guide TOPIC` or `cayu guide TOPIC#SECTION`.")
        return 0
    name, separator, anchor = args.name.partition("#")
    guide_record = _GUIDES.get(name)
    if guide_record is None:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {
                            "code": "UNKNOWN_GUIDE_TOPIC",
                            "message": (
                                f"unknown guide topic {name!r}; choose from: {', '.join(_GUIDES)}"
                            ),
                        },
                    },
                    sort_keys=True,
                )
            )
            return 2
        print(
            f"error: unknown guide topic {name!r}; choose from: {', '.join(_GUIDES)}",
            file=sys.stderr,
        )
        return 2
    if separator and not anchor:
        if args.json:
            print(
                json.dumps(
                    {
                        "schema_version": 1,
                        "error": {
                            "code": "EMPTY_GUIDE_SECTION",
                            "message": "guide section after `#` must not be empty",
                        },
                    },
                    sort_keys=True,
                )
            )
            return 2
        print("error: guide section after `#` must not be empty", file=sys.stderr)
        return 2
    guide = files("cayu.guides").joinpath(guide_record[0])
    content = _render_includes(guide.read_text(encoding="utf-8"))
    if anchor:
        section = _guide_section(content, anchor)
        if section is None:
            if args.json:
                print(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "error": {
                                "code": "GUIDE_SECTION_NOT_FOUND",
                                "message": (f"section {anchor!r} was not found in guide {name!r}"),
                            },
                        },
                        sort_keys=True,
                    )
                )
                return 2
            print(f"error: section {anchor!r} was not found in guide {name!r}", file=sys.stderr)
            return 2
        content = section
    if args.json:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "topic": name,
                    "section": anchor or None,
                    "installed_cayu_version": package_version(),
                    "package_source": f"cayu.guides/{guide_record[0]}",
                    "summary": guide_record[1],
                    "content": content,
                    "related_topics": list(_RELATED.get(name, ())),
                    "verification_commands": [
                        "uv run --no-sync cayu inspect --json",
                        "uv run --no-sync cayu check --fail-on warning --json",
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
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
