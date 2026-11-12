"""Reviewed declarative import forms pass; effects and opaque calls stay rejected."""

from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli.scaffold_check import check_declared_scaffold_source

_ACCEPTED = {
    "toolspec_nested_identity": """from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu import Tool, ToolContext, ToolEffect, ToolResult, ToolSpec


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
    "toolspec_identity_constant": """from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu import Tool, ToolEffect, ToolSpec

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

from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity
from cayu import Tool, ToolEffect, ToolSpec

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
    # A pattern built from another compiled constant's source.
    "pattern_from_compiled_constant": (
        'import re\n\n_REPOSITORY = re.compile(r"[a-z]+/[a-z]+")\n'
        'ISSUE = re.compile(_REPOSITORY.pattern + r"#\\d+", _REPOSITORY.flags)\n'
    ),
    "pattern_source_in_toolspec": (
        "import re\n\nfrom cayu import Tool, ToolEffect, ToolSpec\n\n"
        '_REPOSITORY = re.compile(r"^[a-z]+/[a-z]+$")\n\n\n'
        "class Probe(Tool):\n    spec = ToolSpec(\n        name='probe',\n"
        "        description='d',\n"
        "        input_schema={'type': 'object', 'properties': "
        "{'repository': {'type': 'string', 'pattern': _REPOSITORY.pattern}}},\n"
        "        effect=ToolEffect.NONE,\n    )\n"
    ),
    "httpx_transport_attribute": (
        "import httpx\n\n\nclass ReadOnly(httpx.AsyncBaseTransport):\n"
        "    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:\n"
        "        self._inner = inner\n\n"
        "    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:\n"
        "        return await self._inner.handle_async_request(request)\n"
    ),
    "httpx_transport_name": (
        "from httpx import BaseTransport, Request, Response\n\n\n"
        "class Refuse(BaseTransport):\n"
        "    def handle_request(self, request: Request) -> Response:\n"
        "        raise RuntimeError('read only')\n"
    ),
    "frozen_dataclass_data": (
        "from dataclasses import dataclass, field\n\n\n@dataclass(frozen=True)\n"
        'class Question:\n    """A demo question."""\n\n    text: str\n'
        "    answer: str = ''\n    tags: tuple[str, ...] = field(default_factory=tuple)\n\n"
        "    def short(self) -> str:\n        return self.text[:10]\n\n\n"
        "@dataclass(frozen=True, slots=True)\nclass Followup(Question):\n    after: int = 0\n\n\n"
        "QUESTIONS = (Question('a', answer='b'), Followup(text='c', after=1))\n"
    ),
    "zoneinfo_cached_function": (
        "import functools\nfrom zoneinfo import ZoneInfo\n\n\n@functools.cache\n"
        "def pacific() -> ZoneInfo:\n    return ZoneInfo('America/Los_Angeles')\n"
    ),
    "memoized": (
        "import functools\nfrom functools import lru_cache\n\n\n@functools.cache\n"
        "def first(key: str) -> str:\n    return key\n\n\n@lru_cache(maxsize=32)\n"
        "def second(key: str) -> str:\n    return key\n"
    ),
}

# Both public entry points must preserve the same declaration proof.
_ACCEPTED.update(
    {
        name + "_public_package": source.replace(
            "from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity",
            "from cayu.runtime import ExecutionProfileBehaviorIdentity",
        )
        for name, source in _ACCEPTED.items()
        if name.startswith("toolspec_")
    }
)

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
    # An application function call passes only when its body is proven inert;
    # tests/cli/test_scaffold_import_helpers.py covers the accepted bodies.
    "application_call": (
        "def load():\n    return object()\n\n\nVALUE = load()\n",
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
    "pattern_method": (
        'import re\n\n_WORD = re.compile(r"[a-z]+")\nFIRST = _WORD.match("a")\n',
        "unsupported_expression",
        "_WORD.match",
    ),
    "frozen_dataclass_post_init": (
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Question:\n"
        "    text: str\n\n    def __post_init__(self) -> None:\n        print(self.text)\n\n\n"
        "QUESTIONS = (Question('a'),)\n",
        "unsupported_expression",
        "Question(...)",
    ),
    "frozen_dataclass_lambda_factory": (
        "from dataclasses import dataclass, field\n\n\n@dataclass(frozen=True)\nclass Question:\n"
        "    tags: list = field(default_factory=lambda: print('x'))\n\n\n"
        "QUESTIONS = (Question(),)\n",
        "unsupported_expression",
        "Question(...)",
    ),
    "mutable_dataclass_instance": (
        "from dataclasses import dataclass\n\n\n@dataclass\nclass Question:\n    text: str\n\n\n"
        "QUESTIONS = (Question('a'),)\n",
        "unsupported_expression",
        "Question(...)",
    ),
    "frozen_dataclass_unpacked": (
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True)\nclass Question:\n"
        "    text: str\n\n\nQUESTIONS = (Question(*['a']),)\n",
        "unsupported_expression",
        "Question(...)",
    ),
    "frozen_dataclass_property": (
        "from dataclasses import dataclass\n\n\n@dataclass(frozen=True, init=False)\n"
        "class Question:\n    text: str\n\n    @property\n    def label(self) -> str:\n"
        "        return self.text\n\n\nQUESTIONS = {Question()}\n",
        "unsupported_expression",
        "Question(...)",
    ),
    "frozen_dataclass_dunder_field": (
        "from dataclasses import dataclass\nfrom typing import Any\n\n\n@dataclass(frozen=True)\n"
        "class Record:\n    __post_init__: Any = None\n\n\nVALUE = Record(lambda: None)\n",
        "unsupported_expression",
        "Record(...)",
    ),
    "zone_inside_an_effect": (
        "import subprocess\nfrom zoneinfo import ZoneInfo\n\n"
        "OUT = subprocess.run(['date'], env={'TZ': str(ZoneInfo('UTC'))})\n",
        "import_time_effect",
        "subprocess.run(...)",
    ),
    "zoneinfo": (
        'from zoneinfo import ZoneInfo\n\nPACIFIC = ZoneInfo("America/Los_Angeles")\n',
        "unsupported_expression",
        "ZoneInfo(...)",
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
        "from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity\nfrom cayu import Tool, ToolEffect, ToolSpec\n\n\n"
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
        "from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity\nfrom cayu import Tool, ToolEffect, ToolSpec\n\n"
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


_REJECTED.update(
    {
        name + "_public_package": (
            source.replace(
                "from cayu.runtime.execution_identity import ExecutionProfileBehaviorIdentity",
                "from cayu.runtime import ExecutionProfileBehaviorIdentity",
            ),
            reason,
            construct,
        )
        for name, (source, reason, construct) in _REJECTED.items()
        if name.startswith("identity_constant_")
    }
)


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


def test_zoneinfo_finding_says_why_and_names_the_accepted_form(project: Path) -> None:
    # Zone construction at import stays refused on purpose.
    (finding,) = _findings(project, _REJECTED["zoneinfo"][0])

    assert "tzdata" in finding.message
    assert "@functools.cache" in finding.hint
    assert finding.parameters["accepted_form"] == "cached_function"


@pytest.mark.parametrize(
    ("name", "accepted"),
    (
        ("frozen_dataclass_post_init", "@dataclass(frozen=True)"),
        ("mutable_dataclass_instance", "@dataclass(frozen=True)"),
        ("pattern_method", "`.pattern`"),
    ),
)
def test_refused_forms_name_the_accepted_way(project: Path, name: str, accepted: str) -> None:
    (finding,) = _findings(project, _REJECTED[name][0])

    assert accepted in finding.hint
