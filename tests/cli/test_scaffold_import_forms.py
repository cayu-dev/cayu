"""Reviewed declarative import forms pass; effects and opaque calls stay rejected."""

from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold_check import check_declared_scaffold_source

_ACCEPTED = {
    "toolspec_nested_identity": """from cayu import (
    ExecutionProfileBehaviorIdentity,
    Tool,
    ToolContext,
    ToolEffect,
    ToolResult,
    ToolSpec,
)


class AuditCsv(Tool):
    spec = ToolSpec(
        name="audit_csv",
        description="Audit a CSV file.",
        input_schema={"type": "object", "properties": {"path": {"type": "string"}}},
        effect=ToolEffect.NONE,
        execution_profile_identity=ExecutionProfileBehaviorIdentity(
            name="probe.audit_csv", behavior_version="1", implementation_version="1"
        ),
    )

    async def run(self, ctx: ToolContext, args: dict) -> ToolResult:
        return ToolResult(content="ok")
""",
    "toolspec_identity_constant": """from cayu import ExecutionProfileBehaviorIdentity, Tool, ToolEffect, ToolSpec

_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="probe.tool", behavior_version="1", implementation_version="1"
)


class Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="d",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
        execution_profile_identity=_IDENTITY,
    )
""",
    "toolspec_annotated_identity_constant": """from typing import Final

from cayu import ExecutionProfileBehaviorIdentity, Tool, ToolEffect, ToolSpec

_IDENTITY: Final[ExecutionProfileBehaviorIdentity]
_IDENTITY = ExecutionProfileBehaviorIdentity(
    name="probe.tool", behavior_version="1", implementation_version="1"
)


class Probe(Tool):
    spec = ToolSpec(
        name="probe",
        description="d",
        input_schema={"type": "object"},
        effect=ToolEffect.NONE,
        execution_profile_identity=_IDENTITY,
    )
""",
    "logger": "import logging\n\nlogger = logging.getLogger(__name__)\n",
    "dataclass_field": (
        "from dataclasses import dataclass, field\n\n\n@dataclass\nclass Row:\n"
        "    tags: list[str] = field(default_factory=list)\n    count: int = field(default=0)\n"
    ),
    "enum": 'from enum import StrEnum\n\n\nclass Mode(StrEnum):\n    FAST = "fast"\n',
    "typed_dict": (
        "from typing import TypedDict\n\n\nclass Row(TypedDict):\n    name: str\n    count: int\n"
    ),
    "memoized": (
        "import functools\nfrom functools import lru_cache\n\n\n@functools.cache\n"
        "def first(key: str) -> str:\n    return key\n\n\n@lru_cache(maxsize=32)\n"
        "def second(key: str) -> str:\n    return key\n"
    ),
}

_REJECTED = {
    "subprocess": (
        "import subprocess\n\nsubprocess.run(['ls'])\n",
        "import_time_effect",
        "subprocess.run(...)",
    ),
    "network": (
        'import urllib.request\n\nDATA = urllib.request.urlopen("https://example.com").read()\n',
        "import_time_effect",
        "urllib.request.urlopen(...).read(...)",
    ),
    "write": (
        'from pathlib import Path\n\nPath("x").write_text("y")\n',
        "import_time_effect",
        "Path(...).write_text(...)",
    ),
    "open": ('CONFIG = open("config.txt").read()\n', "import_time_effect", "open(...).read(...)"),
    "application_call": (
        "def load():\n    return 1\n\n\nVALUE = load()\n",
        "unsupported_expression",
        "load(...)",
    ),
    "toolspec_application_call": (
        "from cayu import Tool, ToolEffect, ToolSpec\n\n\ndef describe():\n    return 'd'\n\n\n"
        "class Probe(Tool):\n    spec = ToolSpec(\n        name='probe',\n"
        "        description=describe(),\n        input_schema={'type': 'object'},\n"
        "        effect=ToolEffect.NONE,\n    )\n",
        "unsupported_expression",
        "describe(...)",
    ),
    "computed_field_factory": (
        "from dataclasses import dataclass, field\n\n\ndef build():\n    return []\n\n\n"
        "@dataclass\nclass Row:\n    tags: list[str] = field(default_factory=build)\n",
        "unsupported_expression",
        None,
    ),
    "environment_get": (
        'import os\n\nREGION = os.environ.get("REGION", "us")\n',
        "import_time_configuration",
        "os.environ.get(...)",
    ),
    "environment_subscript": (
        'import os\n\nREGION = os.environ["REGION"]\n',
        "import_time_configuration",
        "os.environ[...]",
    ),
    "getenv": (
        'import os\n\nZONE = os.getenv("ZONE")\n',
        "import_time_configuration",
        "os.getenv(...)",
    ),
    # An identity constant counts only after its assignment...
    "identity_constant_before_assignment": (
        "from cayu import ExecutionProfileBehaviorIdentity, Tool, ToolEffect, ToolSpec\n\n\n"
        "class Probe(Tool):\n    spec = ToolSpec(\n        name='probe',\n"
        "        description='d',\n        input_schema={'type': 'object'},\n"
        "        effect=ToolEffect.NONE,\n        execution_profile_identity=_IDENTITY,\n    )\n"
        "\n\n_IDENTITY = ExecutionProfileBehaviorIdentity(\n    name='probe.tool', "
        "behavior_version='1', implementation_version='1'\n)\n",
        "unsupported_expression",
        None,
    ),
    # ...and only while nothing else can rebind the name.
    "identity_constant_rebound": (
        "from cayu import ExecutionProfileBehaviorIdentity, Tool, ToolEffect, ToolSpec\n\n"
        "_IDENTITY = ExecutionProfileBehaviorIdentity(\n    name='probe.tool', "
        "behavior_version='1', implementation_version='1'\n)\n"
        "from cayu import ToolResult as _IDENTITY  # noqa: E402\n\n\n"
        "class Probe(Tool):\n    spec = ToolSpec(\n        name='probe',\n"
        "        description='d',\n        input_schema={'type': 'object'},\n"
        "        effect=ToolEffect.NONE,\n        execution_profile_identity=_IDENTITY,\n    )\n",
        "unsupported_expression",
        None,
    ),
}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("import-forms")
    assert main(["new", "probe", "--dir", str(root)]) == 0
    return root / "probe"


def _findings(project: Path, source: str) -> list:
    module = project / "tools" / "candidate.py"
    module.write_text(source, encoding="utf-8")
    try:
        return [
            diagnostic
            for diagnostic in check_declared_scaffold_source(project)
            if diagnostic.path.startswith("tools/candidate.py")
        ]
    finally:
        module.unlink()


@pytest.mark.parametrize("name", sorted(_ACCEPTED))
def test_reviewed_declarative_forms_pass_the_import_check(project: Path, name: str) -> None:
    assert _findings(project, _ACCEPTED[name]) == []


@pytest.mark.parametrize("name", sorted(_REJECTED))
def test_effects_and_opaque_calls_are_named_and_classified(project: Path, name: str) -> None:
    source, reason, construct = _REJECTED[name]

    findings = _findings(project, source)

    assert [finding.code for finding in findings] == ["SCAFFOLD_IMPORT_SIDE_EFFECT"]
    assert findings[0].parameters["reason"] == reason
    if construct is not None:
        assert findings[0].parameters["construct"] == construct
        assert f"`{construct}`" in findings[0].message
