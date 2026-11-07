"""Module-level calls to application helpers pass the import check only when their bodies are inert."""

from __future__ import annotations

from pathlib import Path

import pytest

from cayu.cli import main
from cayu.cli._import_helper_calls import MAX_HELPER_DEPTH
from cayu.cli.scaffold_check import check_declared_scaffold_source

# cayu-scout's configuration/settings.py and agents/scout.py.
_SCOUT_SETTINGS = '''"""Settings."""

import os


def configured_reasoning_summaries() -> bool:
    """Whether OpenAI sessions ask for reasoning summaries."""

    selected = os.environ.get("CAYU_SCOUT_REASONING_SUMMARIES", "off").strip().lower()
    if selected not in ("on", "off"):
        raise ValueError(f"CAYU_SCOUT_REASONING_SUMMARIES must be on or off, got {selected!r}")
    return selected == "on"


def scout_provider_options() -> dict:
    """Provider options every scout agent sends."""

    return {} if configured_reasoning_summaries() else {"openai": {"reasoning": {"summary": None}}}
'''
_SCOUT_AGENT = """from cayu import AgentSpec

from configuration import configured_model, configured_provider_name
from tools.helper_settings import scout_provider_options

RANKER = AgentSpec(
    name="scout-ranker",
    model=configured_model(),
    provider_name=configured_provider_name(),
    provider_options=scout_provider_options(),
    system_prompt="Rank.",
)
HUNTER = AgentSpec(
    name="scout-hunter",
    model=configured_model(),
    provider_name=configured_provider_name(),
    provider_options=scout_provider_options(),
    system_prompt="Hunt.",
)
"""
# cayu-scout's operations/coding.py.
_SCOUT_EXCLUSIONS = """_FILE_PATTERNS = (".env", "*.pem", "credentials")
_PATH_PATTERNS = (
    "data",
    *(
        pattern
        for file_pattern in _FILE_PATTERNS
        for pattern in (file_pattern, f"**/{file_pattern}")
    ),
)


def _exclusion_key(path: str) -> str:
    return "/".join(
        part.rstrip(" .").casefold()
        for part in path.replace("\\\\", "/").split("/")
        if part not in {"", "."}
    )


_PATH_PATTERN_KEYS = tuple(_exclusion_key(pattern) for pattern in _PATH_PATTERNS)
"""

_ACCEPTED: dict[str, dict[str, str]] = {
    "scout_provider_options": {
        "tools/helper_settings.py": _SCOUT_SETTINGS,
        "tools/candidate.py": _SCOUT_AGENT,
    },
    "scout_exclusion_key": {"tools/candidate.py": _SCOUT_EXCLUSIONS},
    # A package re-export, as configuration/__init__.py does for configured_model.
    "reexported_helper": {
        "tools/helper_pkg/__init__.py": (
            "from tools.helper_pkg.impl import region_label as region_label\n"
        ),
        "tools/helper_pkg/impl.py": (
            "import os\n\n_DEFAULT = 'us'\n\n\n"
            "def region_label(prefix: str = 'region') -> str:\n"
            "    region = os.getenv('REGION') or _DEFAULT\n"
            "    return f'{prefix}-{region.lower()}'\n"
        ),
        "tools/candidate.py": (
            "from tools.helper_pkg import region_label\n\nLABEL = region_label(prefix='zone')\n"
        ),
    },
    # A helper that builds an AgentSpec from settings.
    "agent_builder": {
        "tools/candidate.py": (
            "from cayu import AgentSpec\n\n"
            "from configuration import configured_model, configured_provider_name\n\n\n"
            "def _agent(name: str, prompt: str) -> AgentSpec:\n"
            "    return AgentSpec(\n"
            "        name=name,\n"
            "        model=configured_model(),\n"
            "        provider_name=configured_provider_name(),\n"
            "        system_prompt=prompt,\n"
            "    )\n\n\n"
            "RANKER = _agent('ranker', 'Rank.')\n"
        ),
    },
    # With no augmented assignment at import anywhere, nothing can change a
    # module-level dict or set before the helper reads it.
    "dict_and_set_constants": {
        "tools/candidate.py": (
            "_SUPPORTED = {'openai', 'anthropic'}\n"
            "_MODELS = {'openai': 'gpt', 'anthropic': 'claude'}\n\n\n"
            "def model(provider: str) -> str:\n"
            "    if provider not in _SUPPORTED:\n"
            "        raise RuntimeError(f'unknown provider: {provider}')\n"
            "    return _MODELS[provider]\n\n\n"
            "MODEL = model('openai')\n"
        ),
    },
    # A tuple of strings can't be changed by other modules, so its items are data.
    "frozen_tuple_constant": {
        "tools/candidate.py": (
            "_SUFFIXES = ('.py', '.pyi')\n_ALL = (*_SUFFIXES, *(f'{s}c' for s in _SUFFIXES))\n\n\n"
            "def is_source(name: str) -> bool:\n"
            "    return name.endswith(_ALL) and name.lower() not in _SUFFIXES\n\n\n"
            "SOURCE = is_source('app.py')\n"
        ),
    },
    # Passing and returning an object is inert; only operations on it can run its code.
    "opaque_value_passed_through": {
        "tools/candidate.py": (
            "from pathlib import Path\n\n_ROOT = Path(__file__).resolve()\n\n\n"
            "def pick(value, fallback):\n"
            "    return fallback if value is None else value\n\n\n"
            "ROOT = pick(None, _ROOT)\n"
        ),
    },
}

# name: (modules, reason, obstacle construct or None)
_REJECTED: dict[str, tuple[dict[str, str], str, str | None]] = {
    "file_read": (
        {
            "tools/candidate.py": (
                "from pathlib import Path\n\n\n"
                "def load() -> str:\n    return Path('settings.txt').read_text()\n\n\n"
                "VALUE = load()\n"
            )
        },
        "import_time_effect",
        "Path(...).read_text(...)",
    ),
    "open": (
        {"tools/candidate.py": "def load():\n    return open('x.txt')\n\n\nVALUE = load()\n"},
        "import_time_effect",
        "open(...)",
    ),
    "network": (
        {
            "tools/candidate.py": (
                "import urllib.request\n\n\n"
                "def fetch():\n    return urllib.request.urlopen('https://example.com')\n\n\n"
                "DATA = fetch()\n"
            )
        },
        "import_time_effect",
        "urllib.request.urlopen(...)",
    ),
    "subprocess": (
        {
            "tools/candidate.py": (
                "import subprocess\n\n\n"
                "def revision():\n"
                "    return subprocess.run(['git', 'rev-parse', 'HEAD'], check=True)\n\n\n"
                "REVISION = revision()\n"
            )
        },
        "import_time_effect",
        "subprocess.run(...)",
    ),
    "global_collection_mutation": (
        {
            "tools/candidate.py": (
                "_SEEN = {}\n\n\n"
                "def remember(key: str) -> str:\n    _SEEN.update({key: True})\n    return key\n\n\n"
                "KEY = remember('a')\n"
            )
        },
        "unsupported_expression",
        "_SEEN.update(...)",
    ),
    "global_rebinding": (
        {
            "tools/candidate.py": (
                "_COUNT = 0\n\n\n"
                "def bump() -> int:\n    global _COUNT\n    _COUNT = _COUNT + 1\n"
                "    return _COUNT\n\n\nVALUE = bump()\n"
            )
        },
        "unsupported_expression",
        "global",
    ),
    "attribute_store": (
        {
            "tools/candidate.py": (
                "import logging\n\n\n"
                "def quiet(name: str) -> str:\n"
                "    logger = logging.getLogger(__name__)\n"
                "    logger.disabled = True\n"
                "    return name\n\n\n"
                "NAME = quiet('x')\n"
            )
        },
        "unsupported_expression",
        "logger.disabled",
    ),
    "recursion": (
        {
            "tools/candidate.py": (
                "def depth(path: str) -> int:\n"
                "    return 0 if not path else 1 + depth(path[1:])\n\n\n"
                "DEPTH = depth('abc')\n"
            )
        },
        "unsupported_expression",
        "depth(...)",
    ),
    "mutual_recursion": (
        {
            "tools/candidate.py": (
                "def even(text: str) -> bool:\n    return not text or odd(text[1:])\n\n\n"
                "def odd(text: str) -> bool:\n    return bool(text) and even(text[1:])\n\n\n"
                "EVEN = even('ab')\n"
            )
        },
        "unsupported_expression",
        "even(...)",
    ),
    # str.replace is pure; Path.replace renames a file. The proof follows the value.
    "method_on_an_object_argument": (
        {
            "tools/candidate.py": (
                "from pathlib import Path\n\n\n"
                "def normalized(path):\n    return path.replace('a', 'b')\n\n\n"
                "TARGET = normalized(Path('a'))\n"
            )
        },
        "import_time_effect",
        "path.replace(...)",
    ),
    "outside_function_in_body": (
        {
            "tools/candidate.py": (
                "import textwrap\n\n\n"
                "def banner(text: str) -> str:\n    return textwrap.dedent(text)\n\n\n"
                "BANNER = banner('  x')\n"
            )
        },
        "unsupported_expression",
        "textwrap.dedent(...)",
    ),
    "decorated_helper": (
        {
            "tools/candidate.py": (
                "import functools\n\n\n@functools.cache\n"
                "def label() -> str:\n    return 'x'\n\n\nLABEL = label()\n"
            )
        },
        "unsupported_expression",
        None,
    ),
    # The local name is the module's list, so += would extend it in place.
    "augmented_assignment": (
        {
            "tools/candidate.py": (
                "_NAMES = ['a']\n\n\n"
                "def add(name: str) -> list:\n    names = _NAMES\n    names += [name]\n"
                "    return names\n\n\nNAMES = add('b')\n"
            )
        },
        "unsupported_expression",
        "names += ...",
    ),
    # Another module extends the dict at import (here with a Path), so its items
    # are not assumed to be plain data.
    "shared_dict_constant": (
        {
            "tools/candidate.py": (
                "_TARGETS = {'docs': 'docs'}\n\n\n"
                "def target(name: str) -> str:\n"
                "    return _TARGETS.get(name).replace('a', 'b')\n\n\n"
                "DOCS = target('docs')\n"
            ),
            "tools/extend.py": (
                "from pathlib import Path\n\nfrom tools.candidate import _TARGETS\n\n"
                "_TARGETS |= {'docs': Path('docs')}\n"
            ),
        },
        "unsupported_expression",
        "_TARGETS.get(...)",
    ),
    # A generator runs when it is iterated, after `value` is rebound to the Path.
    "deferred_generator": (
        {
            "tools/candidate.py": (
                "from pathlib import Path\n\n\n"
                "def move(path):\n"
                "    value = 'a'\n"
                "    pending = (value.replace('a', 'b') for _ in (0,))\n"
                "    value = path\n"
                "    return tuple(pending)\n\n\n"
                "RESULT = move(Path('source.txt'))\n"
            )
        },
        "unsupported_expression",
        "generator expression",
    ),
    "loop_in_body": (
        {
            "tools/candidate.py": (
                "def total(values) -> int:\n    count = 0\n    for _ in values:\n"
                "        count += 1\n    return count\n\n\nTOTAL = total((1, 2))\n"
            )
        },
        "unsupported_expression",
        "for",
    ),
}


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("import-helpers")
    assert main(["new", "probe", "--dir", str(root)]) == 0
    return root / "probe"


def _findings(project: Path, modules: dict[str, str]) -> list:
    written: list[Path] = []
    try:
        for relative, source in modules.items():
            path = project / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source, encoding="utf-8")
            written.append(path)
        return [
            diagnostic
            for diagnostic in check_declared_scaffold_source(project)
            if diagnostic.path.startswith("tools/")
        ]
    finally:
        for path in written:
            path.unlink()
        helper_package = project / "tools" / "helper_pkg"
        if helper_package.exists():
            helper_package.rmdir()


@pytest.mark.parametrize("name", sorted(_ACCEPTED))
def test_helpers_with_inert_bodies_pass_the_import_check(project: Path, name: str) -> None:
    assert _findings(project, _ACCEPTED[name]) == []


@pytest.mark.parametrize("name", sorted(_REJECTED))
def test_helpers_with_effects_or_unproven_bodies_are_reported(project: Path, name: str) -> None:
    modules, reason, obstacle = _REJECTED[name]

    findings = _findings(project, modules)

    assert [finding.code for finding in findings] == ["SCAFFOLD_IMPORT_SIDE_EFFECT"]
    finding = findings[0]
    assert finding.parameters["reason"] == reason
    assert finding.parameters["helper"]
    assert finding.parameters["obstacle_location"].startswith("tools/candidate.py:")
    if obstacle is not None:
        assert finding.parameters["obstacle"] == obstacle
        assert obstacle in finding.message


def test_recursive_helpers_name_the_recursive_call(project: Path) -> None:
    modules, _reason, _obstacle = _REJECTED["recursion"]

    [finding] = _findings(project, modules)

    assert finding.path == "tools/candidate.py:5"
    assert finding.parameters["obstacle_kind"] == "recursion"
    assert finding.parameters["obstacle_location"] == "tools/candidate.py:2"
    assert "recursive helpers are not followed" in finding.message


def test_helper_chains_deeper_than_the_bound_are_reported(project: Path) -> None:
    depth = MAX_HELPER_DEPTH + 1
    source = "".join(
        f"def step{index}() -> int:\n    return {'1' if index == depth else f'step{index + 1}()'}\n\n\n"
        for index in range(1, depth + 1)
    )

    [finding] = _findings(project, {"tools/candidate.py": source + "VALUE = step1()\n"})

    assert finding.parameters["obstacle_kind"] == "depth_limit"
    assert finding.parameters["obstacle"] == f"step{depth}(...)"

    # One level less is followed to the end.
    shorter = source.replace(f"return step{depth}()", "return 1")
    assert _findings(project, {"tools/candidate.py": shorter + "VALUE = step1()\n"}) == []


@pytest.mark.parametrize(
    "source",
    [
        # Functions from installed packages are never followed, even pure ones.
        "from textwrap import dedent\n\nBANNER = dedent('  x')\n",
        "import posixpath\n\nPATH = posixpath.join('a', 'b')\n",
        # A name bound twice has no single body to read.
        "def label():\n    return 'a'\n\n\ndef label():\n    return 'b'\n\n\nLABEL = label()\n",
        # A star import can replace any name.
        "from json import *  # noqa: F403\n\n\ndef label():\n    return 'a'\n\n\nLABEL = label()\n",
    ],
)
def test_calls_that_are_not_to_one_application_function_are_not_followed(
    project: Path, source: str
) -> None:
    last_line = source.rstrip().count("\n") + 1
    findings = _findings(project, {"tools/candidate.py": source})

    # The star import is a finding of its own.
    [finding] = [item for item in findings if item.path == f"tools/candidate.py:{last_line}"]
    assert finding.code == "SCAFFOLD_IMPORT_SIDE_EFFECT"
    assert finding.parameters["reason"] == "unsupported_expression"
    assert "helper" not in finding.parameters


def test_module_level_settings_reads_stay_reported(project: Path) -> None:
    # A helper is the call-time form; reading the environment directly at
    # module level keeps its own finding.
    [finding] = _findings(
        project, {"tools/candidate.py": "import os\n\nREGION = os.environ.get('REGION', 'us')\n"}
    )

    assert finding.parameters["reason"] == "import_time_configuration"


def test_a_project_module_named_os_is_not_read_as_the_environment(project: Path) -> None:
    findings = _findings(
        project,
        {
            "os.py": "environ = {}\n",
            "tools/candidate.py": (
                "import os\n\n\ndef region():\n    return os.environ.get('REGION')\n\n\n"
                "REGION = region()\n"
            ),
        },
    )

    assert [finding.parameters.get("helper") for finding in findings] == ["region"]


def test_logger_handler_setup_names_where_to_move_it(project: Path) -> None:
    source = (
        "import logging\nimport sys\n\n"
        "logger = logging.getLogger('probe.timing')\n"
        "_handler = logging.StreamHandler(sys.stderr)\n"
        "_handler.setFormatter(logging.Formatter('%(message)s'))\n"
        "logger.addHandler(_handler)\n"
        "logger.setLevel(logging.INFO)\n"
        "logger.propagate = False\n"
    )

    findings = _findings(project, {"tools/candidate.py": source})

    assert [finding.path for finding in findings] == [
        f"tools/candidate.py:{line}" for line in range(5, 10)
    ]
    assert {finding.parameters["reason"] for finding in findings} == {"import_time_logging_setup"}
    assert all("`app:build_app`" in finding.hint for finding in findings)
    assert all("logging.getLogger(...)" in finding.hint for finding in findings)


def test_a_body_too_deep_to_read_is_reported_not_raised(project: Path) -> None:
    terms = " + ".join(f"'p{index}'" for index in range(2000))
    source = f"def prompt() -> str:\n    return {terms}\n\n\nPROMPT = prompt()\n"

    [finding] = _findings(project, {"tools/candidate.py": source})

    assert finding.parameters["obstacle_kind"] == "proof_limit"


def test_a_default_that_calls_back_into_its_caller_is_bounded(project: Path) -> None:
    source = (
        "def a(flag=False):\n    if flag:\n        return b()\n    return 0\n\n\n"
        "def b(value=a()):\n    return value\n\n\nX = a()\n"
    )

    assert _findings(project, {"tools/candidate.py": source}) == []


@pytest.mark.parametrize(
    ("other", "candidate", "location"),
    [
        # A call that doesn't match the signature is reported where it is.
        (
            "def label(x):\n    return x\n",
            "from tools.other import label\n\nVALUE = label('a', 'b')\n",
            "tools/candidate.py:3",
        ),
        # The recursive call is in the other module.
        (
            "def pong(n):\n    from_other = n\n    return from_other\n\n\n"
            "def bounce(n):\n    return ping(n)\n\n\n"
            "from tools.candidate import ping  # noqa: E402\n",
            "from tools.other import bounce\n\n\ndef ping(n):\n    return bounce(n)\n\n\n"
            "VALUE = ping(1)\n",
            "tools/other.py:7",
        ),
    ],
)
def test_obstacles_point_at_the_module_that_holds_them(
    project: Path, other: str, candidate: str, location: str
) -> None:
    findings = _findings(project, {"tools/other.py": other, "tools/candidate.py": candidate})

    assert [finding.parameters["obstacle_location"] for finding in findings] == [location]
